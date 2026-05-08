"""Guided rollout: think + per-slot JSON-bool scoring via KV-forked suffixes.

One core function `_rollout_kv_fork` does all the work. Public entry points:
- `guided_rollout` / `guided_rollout_batch`: 1-slot binary {"choice": true/false}.
- `guided_rollout_multibool`: 2 * N_foundations slots, aggregated per foundation.

Phase 1: batched think generation (greedy until </think> or budget).
Phase 2a: one batched forward over `chat + think + </think>`, store KV cache.
Phase 2b: for each slot, build a per-row suffix `<close-turn> + user(nudge) +
          assistant(prefill)` via apply_chat_template, deepcopy the cache, run
          a short suffix forward, read logits at the prefill's last position.

Cost: 1 prefix forward + N_slots small suffix forwards (each ≤14 tokens), all
batched over B prompts. Requires a full-attention model -- asserts at entry.

Why turn-boundary close+nudge: matches what a chat UI emits when a human
interrupts a partial assistant turn. On-policy in chat-tuned data, where the
prior `\\nI should answer now.</think>` mid-turn splice was OOD."""
from __future__ import annotations

import copy
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F
from loguru import logger

_CLOSE_MARKER: str = "</think>"
_NUDGE: str = "true or false?"
_ASSISTANT_SENTINEL: str = "ZZUNIQ_ASSISTANT_SENTINEL_ZZ"


@dataclass
class GuidedResult:
    user_prompt: str
    think_text: str
    pmass_format: float
    logratio_ab: float
    p_true: float
    think_tokens: int
    emitted_close: bool


@dataclass
class MultiBoolResult:
    user_prompt: str
    think_text: str
    logratios: dict[str, float]      # foundation -> 0.5*(lr_violation - lr_ok); positive = violates
    pmass_format: dict[str, float]   # foundation -> avg(pm_violation, pm_ok); ≈1 if formatting holds
    lr_violation: dict[str, float] = field(default_factory=dict)
    lr_ok: dict[str, float] = field(default_factory=dict)
    think_tokens: int = 0
    emitted_close: bool = False


_DEFAULT_SCHEMA_HINT: str = (
    "Think briefly, then answer immediately and only with: "
    '{"choice": true}  or  {"choice": false}.'
)

_DEFAULT_MULTIBOOL_HINT: str = (
    "Moral Foundations:\n"
    "care=harm/unkindness  fairness=cheating  loyalty=betrayal  "
    "authority=disobedience  sanctity=degrading  liberty=coercion\n\n"
    "Think about which foundations the scenario violates. "
    "You will then be asked about each one."
)

_DEFAULT_FOUNDATIONS: tuple[str, ...] = (
    "care", "fairness", "loyalty", "authority", "sanctity", "liberty",
)

_FOUNDATION_DESCS: dict[str, str] = {
    "care":      "harm/unkindness",
    "fairness":  "cheating/injustice",
    "loyalty":   "betrayal of group",
    "authority": "disobedience/subversion",
    "sanctity":  "degrading/disgusting",
    "liberty":   "coercion/domination",
}


def _assistant_close(tok) -> str:
    """Probe chat template for the assistant-turn close marker (e.g. `<|im_end|>\\n`
    on Qwen/ChatML, `<|eot_id|>` on Llama3). Tokenizer-agnostic: mimics what a chat
    UI emits when a human stops a partial assistant turn before sending a new user
    message. Sentinel-diff because Qwen3 auto-injects empty `<think></think>` in
    non-generation-prompt mode, breaking opened-vs-closed prefix-diff."""
    closed = tok.apply_chat_template(
        [{"role": "user", "content": "_"},
         {"role": "assistant", "content": _ASSISTANT_SENTINEL}],
        tokenize=False,
    )
    assert _ASSISTANT_SENTINEL in closed, f"sentinel not in template output: {closed!r}"
    return closed.split(_ASSISTANT_SENTINEL, 1)[1]


def _split_choice_ids(choice_token_ids: list) -> tuple[list[int], list[int]]:
    if len(choice_token_ids) == 2 and all(isinstance(x, (list, tuple)) for x in choice_token_ids):
        return list(choice_token_ids[0]), list(choice_token_ids[1])
    return list(choice_token_ids), []


def _assert_full_attention(model) -> None:
    """Hybrid models (Qwen3.5 linear attention, Mamba SSM, RWKV) carry conv/recurrent
    state that doesn't round-trip through past_key_values without fast-path libs.
    Result: KV-fork suffix logits diverge from a flat forward by 0.1-0.7 nats per
    slot. Fail loud rather than silently produce wrong logratios."""
    layer_types = getattr(model.config, "layer_types", None)
    if layer_types is None:
        return
    non_full = sorted({t for t in layer_types if t != "full_attention"})
    assert not non_full, (
        f"requires full-attention layers; model.config.layer_types contains {non_full}. "
        f"Hybrid models lose conv/SSM state through past_key_values, breaking KV-fork."
    )


def choice_token_ids_tf(tok) -> list[list[int]]:
    """[true_ids, false_ids] covering common variants ('true', ' true', 'True', '1', ...)."""
    def _variants(words):
        seen = []
        for s in words:
            tid = tok.encode(s, add_special_tokens=False)[-1]
            if tid not in seen:
                seen.append(tid)
        return seen
    return [_variants(["true", " true", "\ntrue", "True", " True", "\nTrue", "1"]),
            _variants(["false", " false", "\nfalse", "False", " False", "\nFalse", "0"])]


@torch.no_grad()
def _rollout_kv_fork(
    model, tok,
    user_prompts: list[str],
    schema_hint: str,
    max_think_tokens: int,
    scoring_slots: list[tuple[str, str]],   # (nudge_user_text, prefill) per slot
    choice_token_ids: list,                 # [a_ids, b_ids]
    verbose: bool = False,
    gather_token_ids: list[int] | None = None,
) -> tuple[list[tuple[str, int, bool]], list[list[dict]]]:
    """Returns (thinks, slots).
    thinks[i] = (think_text, n_think_tokens, emitted_close)
    slots[i][j] = {pmass_format, logratio, p_true, [lp_gather]}

    If `gather_token_ids` is provided, slot dict also has `lp_gather`: list[float]
    of log-probs at last suffix position for those token ids (in the same order).
    Used by forced-choice (K-way) probing where a 2-way logratio is insufficient.
    """
    _assert_full_attention(model)
    if tok.padding_side != "left":
        raise ValueError("tok.padding_side must be 'left'")
    device = next(model.parameters()).device
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    close = _assistant_close(tok)

    # === Phase 1: think generation (batched) ===
    chats = [
        tok.apply_chat_template(
            [{"role": "user", "content": f"{up}\n\n{schema_hint}" if schema_hint else up}],
            tokenize=False, add_generation_prompt=True,
        ) + "<think>\n"
        for up in user_prompts
    ]

    think_end_id = tok.convert_tokens_to_ids("</think>")
    if think_end_id in (None, getattr(tok, "unk_token_id", None)):
        think_end_id = tok.eos_token_id

    enc = tok(chats, return_tensors="pt", padding=True).to(device)
    prompt_len = enc.input_ids.shape[1]
    phase1 = model.generate(
        **enc, max_new_tokens=max_think_tokens, do_sample=False,
        eos_token_id=think_end_id, pad_token_id=pad_id,
    )

    # === Build per-row scoring prefix: chat + think + </think> ===
    sp_per_row: list[str] = []
    sp_ids_per_row: list[list[int]] = []
    thinks: list[tuple[str, int, bool]] = []
    for i, p in enumerate(chats):
        gen_ids = phase1[i, prompt_len:]
        keep = gen_ids != pad_id
        gen_ids = gen_ids[keep] if keep.any() else gen_ids[:0]
        gen_text = tok.decode(gen_ids, skip_special_tokens=True)
        n_think = int(gen_ids.shape[0])
        emitted_close = _CLOSE_MARKER in gen_text
        think_text = gen_text.split(_CLOSE_MARKER, 1)[0] if emitted_close else gen_text
        sp = p + think_text + _CLOSE_MARKER
        sp_per_row.append(sp)
        sp_ids_per_row.append(tok(sp, add_special_tokens=False)["input_ids"])
        thinks.append((think_text, n_think, emitted_close))

    B = len(sp_per_row)
    P_max = max(len(s) for s in sp_ids_per_row)

    # === Phase 2a: batched prefix forward (left-padded), cache ===
    pref_input = torch.full((B, P_max), pad_id, dtype=torch.long, device=device)
    pref_attn = torch.zeros((B, P_max), dtype=torch.long, device=device)
    pref_real = torch.zeros(B, dtype=torch.long, device=device)
    for i, sp_ids in enumerate(sp_ids_per_row):
        L = len(sp_ids)
        pref_input[i, P_max - L:] = torch.tensor(sp_ids, device=device)
        pref_attn[i, P_max - L:] = 1
        pref_real[i] = L
    pref_out = model(input_ids=pref_input, attention_mask=pref_attn, use_cache=True)
    cache = pref_out.past_key_values

    # === Phase 2b: per-slot KV-fork ===
    a_ids, b_ids = _split_choice_ids(choice_token_ids)
    a_t = torch.tensor(a_ids, device=device, dtype=torch.long) if a_ids else None
    b_t = torch.tensor(b_ids, device=device, dtype=torch.long) if b_ids else None
    all_ids = torch.tensor(a_ids + b_ids, device=device, dtype=torch.long)

    def suf_ids_for(nudge: str, prefill: str) -> list[list[int]]:
        # close-turn + apply_chat_template([user(nudge), assistant(prefill)], continue_final_message=True)
        # gives the on-policy interrupt-then-renudge structure.
        interrupt = tok.apply_chat_template(
            [{"role": "user", "content": nudge},
             {"role": "assistant", "content": prefill}],
            tokenize=False, continue_final_message=True,
        )
        suf_text = close + interrupt
        return [tok(sp + suf_text, add_special_tokens=False)["input_ids"][len(sp_ids):]
                for sp, sp_ids in zip(sp_per_row, sp_ids_per_row)]

    def fork(suffixes: list[list[int]]) -> torch.Tensor:
        """Forward suffix with deepcopied cache. Returns [B, V] logp at last real suffix token."""
        J_max = max(len(s) for s in suffixes)
        suf_input = torch.full((B, J_max), pad_id, dtype=torch.long, device=device)
        suf_mask = torch.zeros((B, J_max), dtype=torch.long, device=device)
        last_pos = torch.zeros(B, dtype=torch.long, device=device)
        for i, s in enumerate(suffixes):
            L = len(s)
            suf_input[i, :L] = torch.tensor(s, device=device)
            suf_mask[i, :L] = 1
            last_pos[i] = L - 1
        full_attn = torch.cat([pref_attn, suf_mask], dim=1)
        position_ids = pref_real[:, None] + torch.arange(J_max, device=device)[None, :]
        forked = copy.deepcopy(cache)
        out = model(
            input_ids=suf_input, attention_mask=full_attn,
            position_ids=position_ids, past_key_values=forked, use_cache=False,
        )
        logp = F.log_softmax(out.logits.float(), dim=-1)
        return logp[torch.arange(B, device=device), last_pos]

    slots: list[list[dict]] = [[] for _ in range(B)]
    for j, (nudge, prefill) in enumerate(scoring_slots):
        suf_ids = suf_ids_for(nudge, prefill)
        if verbose:
            full_text = sp_per_row[0] + tok.decode(suf_ids[0], skip_special_tokens=False)
            # Free-form generate to see what the model actually says after the prefill.
            full_ids = torch.tensor(
                [sp_ids_per_row[0] + suf_ids[0]], device=device, dtype=torch.long,
            )
            gen = model.generate(full_ids, max_new_tokens=64, do_sample=False, pad_token_id=pad_id)
            free = tok.decode(gen[0, full_ids.shape[1]:], skip_special_tokens=False)
            logger.info(
                f"--- slot {j} (nudge={nudge!r}, prefill={prefill!r}) ---\n"
                f"{full_text}<<<MODEL CONTINUES>>>{free}\n--- end slot {j} ---"
            )
        lp_last = fork(suf_ids)
        pmass = lp_last[:, all_ids].exp().sum(-1)
        if a_t is not None and b_t is not None:
            la = torch.logsumexp(lp_last[:, a_t], dim=-1)
            lb = torch.logsumexp(lp_last[:, b_t], dim=-1)
            logratio = la - lb
            p_true = torch.softmax(torch.stack([la, lb], dim=-1), dim=-1)[:, 0]
        else:
            logratio = torch.full((B,), float("nan"), device=device)
            p_true = torch.full((B,), float("nan"), device=device)
        for i in range(B):
            top5 = lp_last[i].topk(5)
            top5_str = " ".join(
                f"{tok.decode([int(idx)])!r}:{float(prob.exp()):.3f}"
                for idx, prob in zip(top5.indices, top5.values)
            )
            d = {
                "pmass_format": float(pmass[i].item()),
                "logratio": float(logratio[i].item()),
                "p_true": float(p_true[i].item()),
                "top5_str": top5_str,
            }
            if gather_token_ids is not None:
                gid_t = torch.tensor(gather_token_ids, device=device, dtype=torch.long)
                d["lp_gather"] = lp_last[i, gid_t].cpu().tolist()
            slots[i].append(d)

    return thinks, slots


def guided_rollout_batch(
    model, tok,
    user_prompts: list[str],
    choice_token_ids: list,
    max_think_tokens: int = 128,
    schema_hint: str = _DEFAULT_SCHEMA_HINT,
    prefill: str = '{"choice": ',
    verbose: bool = False,
) -> list[GuidedResult]:
    """Single-slot binary rollout. See `_rollout_kv_fork`."""
    thinks, slots = _rollout_kv_fork(
        model, tok, user_prompts, schema_hint, max_think_tokens,
        scoring_slots=[(_NUDGE, prefill)],
        choice_token_ids=choice_token_ids,
        verbose=verbose,
    )
    return [GuidedResult(
        user_prompt=up,
        think_text=t[0], think_tokens=t[1], emitted_close=t[2],
        pmass_format=s[0]["pmass_format"],
        logratio_ab=s[0]["logratio"],
        p_true=s[0]["p_true"],
    ) for up, t, s in zip(user_prompts, thinks, slots)]


def guided_rollout(
    model, tok,
    user_prompt: str,
    choice_token_ids: list,
    max_think_tokens: int = 128,
    schema_hint: str = _DEFAULT_SCHEMA_HINT,
    prefill: str = '{"choice": ',
    verbose: bool = False,
) -> GuidedResult:
    """Single-prompt convenience wrapper around `guided_rollout_batch`."""
    return guided_rollout_batch(
        model, tok, [user_prompt], choice_token_ids,
        max_think_tokens=max_think_tokens, schema_hint=schema_hint,
        prefill=prefill, verbose=verbose,
    )[0]


def guided_rollout_multibool(
    model, tok,
    user_prompts: list[str],
    foundations: list[str] | None = None,
    *,
    max_think_tokens: int = 256,
    schema_hint: str = _DEFAULT_MULTIBOOL_HINT,
    verbose: bool = False,
) -> list[MultiBoolResult]:
    """Per-foundation violation logratio via 2*|foundations| KV-forked slots.

    Each foundation has 2 slots (is_violation / is_ok framing).
        lr[f] = 0.5 * (lr_violation[f] - lr_ok[f])  -- cancels per-key prior
        pm[f] = 0.5 * (pm_violation[f] + pm_ok[f])"""
    if foundations is None:
        foundations = list(_DEFAULT_FOUNDATIONS)

    scoring_slots: list[tuple[str, str]] = []
    slot_keys: list[tuple[str, str]] = []  # (frame, foundation)
    for frame in ("is_violation", "is_ok"):
        for f in foundations:
            desc = _FOUNDATION_DESCS.get(f, f)
            if frame == "is_violation":
                q = f"Does this violate {f} norms ({desc})? Answer as a JSON bool."
            else:
                q = f"Is this acceptable under {f} norms ({desc})? Answer as a JSON bool."
            scoring_slots.append((q, '{"Answer": '))
            slot_keys.append((frame, f))

    thinks, slots = _rollout_kv_fork(
        model, tok, user_prompts, schema_hint, max_think_tokens,
        scoring_slots=scoring_slots,
        choice_token_ids=choice_token_ids_tf(tok),
        verbose=verbose,
    )

    results: list[MultiBoolResult] = []
    for i, (think_text, n_think, emitted_close) in enumerate(thinks):
        lr_v: dict[str, float] = {}
        lr_o: dict[str, float] = {}
        pm_v: dict[str, float] = {}
        pm_o: dict[str, float] = {}
        for j, (frame, f) in enumerate(slot_keys):
            slot = slots[i][j]
            if frame == "is_violation":
                lr_v[f] = slot["logratio"]
                pm_v[f] = slot["pmass_format"]
            else:
                lr_o[f] = slot["logratio"]
                pm_o[f] = slot["pmass_format"]
        results.append(MultiBoolResult(
            user_prompt=user_prompts[i],
            think_text=think_text,
            logratios={f: 0.5 * (lr_v[f] - lr_o[f]) for f in foundations},
            pmass_format={f: 0.5 * (pm_v[f] + pm_o[f]) for f in foundations},
            lr_violation=lr_v,
            lr_ok=lr_o,
            think_tokens=n_think,
            emitted_close=emitted_close,
        ))

    low = [(i, f, results[i].pmass_format[f])
           for i in range(len(results)) for f in foundations
           if results[i].pmass_format[f] < 0.5]
    if low:
        per_f: dict[str, int] = {}
        for _, f, _ in low:
            per_f[f] = per_f.get(f, 0) + 1
        f_summary = " ".join(f"{f}:{n}" for f, n in sorted(per_f.items(), key=lambda x: -x[1]))
        wi, wf, wpm = min(low, key=lambda x: x[2])
        # find worst slot (is_violation or is_ok) for that foundation to get its top5
        f_idx = foundations.index(wf)
        j_v, j_o = f_idx, len(foundations) + f_idx
        pm_v = slots[wi][j_v]["pmass_format"]
        pm_o = slots[wi][j_o]["pmass_format"]
        j_worst = j_v if pm_v <= pm_o else j_o
        top5 = slots[wi][j_worst].get("top5_str", "?")
        logger.warning(
            f"pmass<0.5: {len(low)}/{len(results) * len(foundations)} (row,f) pairs  per-f: {f_summary}\n"
            f"  worst: row={wi} {wf} pm={wpm:.3f}  top5: {top5}"
        )

    return results


# ===== Forced-choice (K-way primary foundation) =====

# Foundation set + descriptions adapted from the response options in
# Clifford et al. (2015) Study 1. Their "social" option = "not morally wrong"
# (the social-norms control set). Descriptions are parallel and short
# (no repetitive "violates norms of" prefix) so no option stands out
# structurally.
_DEFAULT_FORCED_FOUNDATIONS: tuple[str, ...] = (
    "care", "fairness", "loyalty", "authority", "sanctity", "liberty", "social",
)

_FORCED_FOUNDATION_DESCS: dict[str, str] = {
    "care":      "harm or unkindness, causing pain to another",
    "fairness":  "cheating or reducing equality",
    "loyalty":   "betrayal of a group",
    "authority": "subversion or lack of respect for tradition",
    "sanctity":  "purity, degrading or disgusting acts",
    "liberty":   "bullying or dominating",
    "social":    "weird or unusual behaviour, but not morally wrong",
}


def _format_options(foundations: list[str]) -> str:
    """JSON-schema-style enum lines with `# description` comments per option."""
    return "\n".join(
        f'          "{f}", # {_FORCED_FOUNDATION_DESCS.get(f, f)}'
        for f in foundations
    )


# Schema hint (shown ONCE in the user-side prompt that precedes the think).
# JSON-pseudo-schema with enum + per-option comments puts the description
# right at the option, not in separate prose. The prefill
# `This is wrong because {"violation": "` then forces a single-token answer.
def _make_forced_hint(foundations: list[str]) -> str:
    return (
        "This is wrong because:\n"
        "  {\n"
        '    "properties": {\n'
        '      "violation": {\n'
        '        "enum": [\n'
        f"{_format_options(foundations)}\n"
        "        ]\n"
        "      }\n"
        "    }\n"
        "  }"
    )


_DEFAULT_FORCED_HINT: str = _make_forced_hint(list(_DEFAULT_FORCED_FOUNDATIONS))


@dataclass
class ForcedChoiceResult:
    user_prompt: str
    # Two thinks: one per enum-ordering frame. think_fwd uses the forward enum
    # order, think_rev uses the reversed enum order. These cancel position bias
    # when the resulting logprobs are averaged.
    think_text: str       # forward-frame think (for backward compatibility)
    think_text_rev: str   # reversed-frame think
    # Per-frame raw logprobs (unnormalised) at the prefill position.
    lp_fwd: dict[str, float]   # enum listed [care, ..., social]
    lp_rev: dict[str, float]   # enum listed [social, ..., care]
    # Debiased score: average of lp_fwd and lp_rev. Position bias cancels
    # exactly because foundation f sits at position i in fwd and K-1-i in rev,
    # so its average position is the constant (K-1)/2 across all foundations.
    score: dict[str, float]
    p: dict[str, float]    # softmax over the K options of `score`
    top1: str
    margin: float          # score[top1] - score[top2], in nats
    think_tokens: int
    emitted_close: bool


def _resolve_first_token_ids(tok, words: list[str]) -> tuple[list[int], dict[str, int]]:
    """Return (ids_in_order, word->id). Each id is the first token of the word
    when it appears immediately after a `"` in JSON, i.e. with no leading space.
    Asserts the K first-tokens are distinct.

    The forced-choice prefill is `... "violates": "` so the model's
    next token is the first BPE piece of the foundation word with no space prefix."""
    ids: list[int] = []
    mapping: dict[str, int] = {}
    for w in words:
        toks = tok.encode(w, add_special_tokens=False)
        assert toks, f"tokenizer returned empty for {w!r}"
        ids.append(toks[0])
        mapping[w] = toks[0]
    assert len(set(ids)) == len(ids), (
        f"first-token collision among forced-choice words: "
        f"{[(w, i, tok.decode([i])) for w, i in mapping.items()]}"
    )
    return ids, mapping


def guided_rollout_forced_choice(
    model, tok,
    user_prompts: list[str],
    foundations: list[str] | None = None,
    *,
    max_think_tokens: int = 256,
    schema_hint: str | None = None,
    verbose: bool = False,
) -> list[ForcedChoiceResult]:
    """K-way forced-choice over MFT foundations with two-pass position-bias debias.

    Why K-way: pairwise yes/no probes (multibool) leak a global "badness" prior
    across foundations, collapsing per-foundation discrimination. A K-way
    softmax over foundation-name first-tokens forces exactly one and sidesteps
    that prior.

    Why two passes (forward + reversed enum):
    - Position bias is real: models often favour the first option listed.
    - If foundation `f` is at position `i` in the forward enum, it's at position
      `K-1-i` in the reversed enum. Averaging logprobs across the two frames
      makes the position contribution `(P[i] + P[K-1-i]) / 2` -- a constant
      independent of `f`, so it falls out of the softmax over foundations.
    - Two SEPARATE thinks (one per ordering) keep the framing internally
      consistent. We don't try to share KV across orderings since the schema
      text itself differs.

    Format (one user turn per frame):
        <user>
        This is wrong because of which moral foundation
        > {scenario}
        {JSON-pseudo-schema with enum + # comments}
        </user>
        <assistant>
        <think>...</think>
        </assistant>
        <user>Just answer</user>
        <assistant>This is wrong because {"violation": "<scored token>
    """
    if foundations is None:
        foundations = list(_DEFAULT_FORCED_FOUNDATIONS)
    K = len(foundations)
    assert K >= 2, "need >=2 foundations for forced choice"

    foundations_rev = list(reversed(foundations))
    schema_fwd = schema_hint if schema_hint is not None else _make_forced_hint(foundations)
    schema_rev = _make_forced_hint(foundations_rev)

    first_ids, _ = _resolve_first_token_ids(tok, foundations)

    nudge = "Just answer"
    prefill = 'This is wrong because {"violation": "'
    scoring_slot = [(nudge, prefill)]

    # Frame A: forward enum order.
    thinks_fwd, slots_fwd = _rollout_kv_fork(
        model, tok, user_prompts, schema_fwd, max_think_tokens,
        scoring_slots=scoring_slot,
        choice_token_ids=[[first_ids[0]]],  # unused; satisfies API
        verbose=verbose,
        gather_token_ids=first_ids,
    )
    # Frame B: reversed enum order. Same gather order (by foundation name) so
    # lp_rev[f] is comparable to lp_fwd[f].
    thinks_rev, slots_rev = _rollout_kv_fork(
        model, tok, user_prompts, schema_rev, max_think_tokens,
        scoring_slots=scoring_slot,
        choice_token_ids=[[first_ids[0]]],
        verbose=verbose,
        gather_token_ids=first_ids,
    )

    results: list[ForcedChoiceResult] = []
    import math
    for i in range(len(user_prompts)):
        think_fwd, n_fwd, close_fwd = thinks_fwd[i]
        think_rev, _, _ = thinks_rev[i]
        lp_f = slots_fwd[i][0]["lp_gather"]
        lp_r = slots_rev[i][0]["lp_gather"]
        score = [(lp_f[k] + lp_r[k]) / 2.0 for k in range(K)]

        m = max(score)
        exps = [math.exp(x - m) for x in score]
        Z = sum(exps)
        p = {foundations[k]: exps[k] / Z for k in range(K)}
        order_sorted = sorted(range(K), key=lambda k: -score[k])
        top1 = foundations[order_sorted[0]]
        margin = score[order_sorted[0]] - score[order_sorted[1]]
        results.append(ForcedChoiceResult(
            user_prompt=user_prompts[i],
            think_text=think_fwd,
            think_text_rev=think_rev,
            lp_fwd={foundations[k]: lp_f[k] for k in range(K)},
            lp_rev={foundations[k]: lp_r[k] for k in range(K)},
            score={foundations[k]: score[k] for k in range(K)},
            p=p,
            top1=top1,
            margin=float(margin),
            think_tokens=n_fwd,
            emitted_close=close_fwd,
        ))

    return results

