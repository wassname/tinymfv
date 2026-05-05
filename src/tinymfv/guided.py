from __future__ import annotations

import contextlib
import copy
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F
from loguru import logger

_CLOSE_MARKER: str = "</think>"

@dataclass
class GuidedResult:
    user_prompt: str
    think_text: str
    answer_text: str
    raw_full_text: str
    pmass_format: float
    logratio_ab: float
    rep_ratio_think: float
    think_tokens: int
    emitted_close: bool
    emitted_prefill: bool
    p_true: float
    # Mean negative-log-likelihood per token over the scoring text (prompt +
    # think + JSON-prefix). Free: we already compute full-sequence logits,
    # just gather instead of slicing [:, -1]. Higher = model less coherent
    # on this prompt under whatever steering is attached.
    prompt_nll: float = float("nan")

_REP_MIN_TOKENS: int = 32

def _ngram_rep_ratio(text: str, n: int = 4) -> float:
    tokens = text.split()
    if len(tokens) < _REP_MIN_TOKENS:
        return float("nan")
    ngrams = [tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]
    return len(set(ngrams)) / len(ngrams)

_DEFAULT_SCHEMA_HINT: str = (
    "Think briefly, then answer immediately and only with: "
    '{"choice": true}  or  {"choice": false}.'
)

@torch.no_grad()
def guided_rollout(
    model, tok,
    user_prompt: str,
    choice_token_ids: list,
    max_think_tokens: int = 128,
    answer_tokens: int = 4,
    schema_hint: str = _DEFAULT_SCHEMA_HINT,
    prefill: str = '\n{"choice": ',
    verbose: bool = False,
) -> GuidedResult:
    device = next(model.parameters()).device
    full_user = f"{user_prompt}\n\n{schema_hint}" if schema_hint else user_prompt
    messages = [{"role": "user", "content": full_user}]
    
    try:
        prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    except TypeError:
        prompt = tok.apply_chat_template(messages, tokenize=False)
        
    prompt = prompt + "<think>\n"

    enc = tok(prompt, return_tensors="pt").to(device)
    prompt_len = enc.input_ids.shape[1]

    think_end_id = tok.convert_tokens_to_ids("</think>")
    if think_end_id in (None, getattr(tok, "unk_token_id", None)):
        think_end_id = tok.eos_token_id
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id

    phase1 = model.generate(
        **enc,
        max_new_tokens=max_think_tokens,
        do_sample=False,
        eos_token_id=think_end_id,
        pad_token_id=pad_id,
    )
    gen_ids = phase1[0, prompt_len:]
    keep = gen_ids != pad_id
    gen_ids = gen_ids[keep] if keep.any() else gen_ids[:0]
    gen_text = tok.decode(gen_ids, skip_special_tokens=True)

    force_suffix = "\nI should answer now." + _CLOSE_MARKER + prefill
    emitted_close = _CLOSE_MARKER in gen_text
    
    if emitted_close:
        think_text, after = gen_text.split(_CLOSE_MARKER, 1)
        if prefill.lstrip() in after:
            emitted_prefill = True
            before_value = after.split(prefill.lstrip(), 1)[0]
            scoring_text = prompt + think_text + _CLOSE_MARKER + before_value + prefill.lstrip()
        else:
            emitted_prefill = False
            scoring_text = prompt + think_text + _CLOSE_MARKER + prefill
    else:
        think_text = gen_text
        emitted_prefill = False
        scoring_text = prompt + gen_text + force_suffix

    score_ids = tok(scoring_text, return_tensors="pt", add_special_tokens=False).input_ids.to(device)

    full_logits = model(score_ids).logits[0].float()  # [T, V]
    # Per-token NLL over the scoring text. Predicting position t from t-1.
    full_logp = F.log_softmax(full_logits, dim=-1)
    target_ids = score_ids[0, 1:]
    pred_logp = full_logp[:-1].gather(-1, target_ids.unsqueeze(-1)).squeeze(-1)
    prompt_nll = float(-pred_logp.mean().item()) if pred_logp.numel() else float("nan")
    logits = full_logits[-1]
    logp = full_logp[-1]

    if (len(choice_token_ids) == 2 and all(isinstance(x, (list, tuple)) for x in choice_token_ids)):
        a_ids, b_ids = list(choice_token_ids[0]), list(choice_token_ids[1])
    else:
        a_ids, b_ids = list(choice_token_ids), []
        
    all_ids = torch.tensor(a_ids + b_ids, device=device, dtype=torch.long)
    pmass_format = float(logp[all_ids].exp().sum().item())

    # SHOULD: pmass≈1 (model picks one of the JSON-bool tokens). pmass<0.9
    # means the model is leaking probability to other tokens -> the schema
    # is being ignored or the steering vector has pushed the model OOD.
    if pmass_format < 0.5:
        topk = torch.topk(logp.exp(), k=5)
        toks = [tok.decode([i]) for i in topk.indices.tolist()]
        probs = topk.values.tolist()
        top5 = ", ".join(f"{repr(t)}={p:.3f}" for t, p in zip(toks, probs))
        logger.warning(f"pmass={pmass_format:.3f}<0.5 — top-5 ans tokens: {top5}. This could be a format issue")

    if a_ids and b_ids:
        a_t = torch.tensor(a_ids, device=device, dtype=torch.long)
        b_t = torch.tensor(b_ids, device=device, dtype=torch.long)
        logratio = float(torch.logsumexp(logp[a_t], dim=0).item() - torch.logsumexp(logp[b_t], dim=0).item())
        p_true = float(torch.softmax(torch.stack([torch.logsumexp(logp[a_t], dim=0), torch.logsumexp(logp[b_t], dim=0)]), dim=0)[0].item())
    else:
        logratio = float("nan")
        p_true = float("nan")

    cont = model.generate(
        score_ids,
        max_new_tokens=answer_tokens,
        do_sample=False,
        pad_token_id=pad_id,
    )
    answer_ids = cont[0, score_ids.shape[1]:]
    answer_text = tok.decode(answer_ids, skip_special_tokens=True)

    raw_full_text = tok.decode(cont[0], skip_special_tokens=False)

    return GuidedResult(
        user_prompt=user_prompt,
        think_text=think_text,
        answer_text=answer_text,
        raw_full_text=raw_full_text,
        pmass_format=pmass_format,
        logratio_ab=logratio,
        rep_ratio_think=_ngram_rep_ratio(think_text, n=4),
        think_tokens=int(score_ids.shape[1] - prompt_len),
        emitted_close=emitted_close,
        emitted_prefill=emitted_prefill,
        p_true=p_true,
        prompt_nll=prompt_nll,
    )

@torch.no_grad()
def guided_rollout_batch(
    model, tok,
    user_prompts: list[str],
    choice_token_ids: list,
    max_think_tokens: int = 128,
    schema_hint: str = _DEFAULT_SCHEMA_HINT,
    prefill: str = '\n{"choice": ',
) -> list[GuidedResult]:
    """Batched guided rollout. Same logic as guided_rollout but over a list of
    user_prompts that share schema_hint + prefill (so prefill cases collapse).

    Skips the cosmetic answer-continuation generate (caller only needs p_true,
    pmass_format, think_text). Two model calls per batch instead of 3 per row:
    one phase1 generate (think) + one scoring forward.

    Tokenizer must already have padding_side='left' and pad_token set."""
    if tok.padding_side != "left":
        raise ValueError("tok.padding_side must be 'left' for batched rollout")
    device = next(model.parameters()).device

    prompts = []
    for up in user_prompts:
        full_user = f"{up}\n\n{schema_hint}" if schema_hint else up
        msgs = [{"role": "user", "content": full_user}]
        try:
            p = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        except TypeError:
            p = tok.apply_chat_template(msgs, tokenize=False)
        prompts.append(p + "<think>\n")

    think_end_id = tok.convert_tokens_to_ids("</think>")
    if think_end_id in (None, getattr(tok, "unk_token_id", None)):
        think_end_id = tok.eos_token_id
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id

    enc = tok(prompts, return_tensors="pt", padding=True).to(device)
    prompt_len = enc.input_ids.shape[1]

    phase1 = model.generate(
        **enc,
        max_new_tokens=max_think_tokens,
        do_sample=False,
        eos_token_id=think_end_id,
        pad_token_id=pad_id,
    )

    scoring_texts = []
    per_row = []  # (think_text, emitted_close, emitted_prefill, n_think_tokens)
    for i, p in enumerate(prompts):
        gen_ids = phase1[i, prompt_len:]
        keep = gen_ids != pad_id
        gen_ids = gen_ids[keep] if keep.any() else gen_ids[:0]
        gen_text = tok.decode(gen_ids, skip_special_tokens=True)
        n_think = int(gen_ids.shape[0])

        emitted_close = _CLOSE_MARKER in gen_text
        if emitted_close:
            think_text, after = gen_text.split(_CLOSE_MARKER, 1)
            if prefill.lstrip() in after:
                emitted_prefill = True
                before_value = after.split(prefill.lstrip(), 1)[0]
                scoring_text = p + think_text + _CLOSE_MARKER + before_value + prefill.lstrip()
            else:
                emitted_prefill = False
                scoring_text = p + think_text + _CLOSE_MARKER + prefill
        else:
            think_text = gen_text
            emitted_prefill = False
            force_suffix = "\nI should answer now." + _CLOSE_MARKER + prefill
            scoring_text = p + gen_text + force_suffix

        scoring_texts.append(scoring_text)
        per_row.append((think_text, emitted_close, emitted_prefill, n_think))

    score_enc = tok(scoring_texts, return_tensors="pt", padding=True,
                    add_special_tokens=False).to(device)
    full_logits = model(**score_enc).logits.float()  # [B, T, V]
    full_logp = F.log_softmax(full_logits, dim=-1)
    # Per-row mean NLL over non-pad positions of the scoring text. Free
    # coherence proxy under whatever steering is attached.
    target_ids = score_enc.input_ids[:, 1:]
    pred_logp = full_logp[:, :-1].gather(-1, target_ids.unsqueeze(-1)).squeeze(-1)
    mask = (target_ids != pad_id).float()
    nll_per_row = (-pred_logp * mask).sum(-1) / mask.sum(-1).clamp(min=1)
    score_logp = full_logp[:, -1]

    if (len(choice_token_ids) == 2 and all(isinstance(x, (list, tuple)) for x in choice_token_ids)):
        a_ids, b_ids = list(choice_token_ids[0]), list(choice_token_ids[1])
    else:
        a_ids, b_ids = list(choice_token_ids), []
    all_ids = torch.tensor(a_ids + b_ids, device=device, dtype=torch.long)
    a_t = torch.tensor(a_ids, device=device, dtype=torch.long) if a_ids else None
    b_t = torch.tensor(b_ids, device=device, dtype=torch.long) if b_ids else None

    results = []
    low_pmass = []  # (idx, pmass) for rows with pmass<0.5 (heavy schema break)
    for i, (up, (think_text, emitted_close, emitted_prefill, n_think)) in enumerate(zip(user_prompts, per_row)):
        logp = score_logp[i]
        pmass_format = float(logp[all_ids].exp().sum().item())
        if pmass_format < 0.5:
            low_pmass.append((i, pmass_format))
        if a_t is not None and b_t is not None:
            la = torch.logsumexp(logp[a_t], dim=0)
            lb = torch.logsumexp(logp[b_t], dim=0)
            logratio = float((la - lb).item())
            p_true = float(torch.softmax(torch.stack([la, lb]), dim=0)[0].item())
        else:
            logratio = float("nan")
            p_true = float("nan")
        results.append(GuidedResult(
            user_prompt=up,
            think_text=think_text,
            answer_text="",
            raw_full_text="",
            pmass_format=pmass_format,
            logratio_ab=logratio,
            rep_ratio_think=_ngram_rep_ratio(think_text, n=4),
            think_tokens=n_think,
            emitted_close=emitted_close,
            emitted_prefill=emitted_prefill,
            p_true=p_true,
            prompt_nll=float(nll_per_row[i].item()),
        ))

    # Aggregate-once warning: one line per batch with worst-case top-5 instead
    # of N spammy per-row lines (heavy steering pushes many rows OOD at once).
    if low_pmass:
        worst_i, worst_pm = min(low_pmass, key=lambda x: x[1])
        topk = torch.topk(score_logp[worst_i].exp(), k=5)
        toks = [tok.decode([j]) for j in topk.indices.tolist()]
        probs = topk.values.tolist()
        top5 = ", ".join(f"{repr(t)}={pp:.3f}" for t, pp in zip(toks, probs))
        logger.warning(
            f"pmass<0.5 on {len(low_pmass)}/{len(results)} rows in this batch; "
            f"worst={worst_pm:.3f} top-5: {top5}"
        )
    return results


@dataclass
class MultiBoolResult:
    user_prompt: str
    think_text: str
    logratios: dict[str, float]      # foundation -> 0.5*(lr_violation - lr_ok); positive = violates
    pmass_format: dict[str, float]   # foundation -> avg(pm_violation, pm_ok); ≈1 if formatting holds
    lr_violation: dict[str, float] = field(default_factory=dict)  # diagnostic: per-frame raw logratios
    lr_ok: dict[str, float] = field(default_factory=dict)
    think_tokens: int = 0
    emitted_close: bool = False


_DEFAULT_MULTIBOOL_HINT: str = (
    'After thinking, output JSON: {"is_violation": {<foundation>: true|false}} '
    "marking which Moral Foundations Theory foundation(s) the scenario violates."
)

_DEFAULT_FOUNDATIONS: tuple[str, ...] = (
    "care", "fairness", "loyalty", "authority", "sanctity", "liberty",
)


def _assert_full_attention(model) -> None:
    """Hybrid models (Qwen3.5 linear attention, Mamba SSM, RWKV) carry conv/
    recurrent state that doesn't round-trip through past_key_values without
    fast-path libs (causal-conv1d, flash-linear-attention). Result: KV-fork
    suffix logits diverge from a flat forward by 0.1-0.7 nats per slot.
    Fail loud rather than silently produce wrong logratios."""
    layer_types = getattr(model.config, "layer_types", None)
    if layer_types is None:
        return
    non_full = sorted({t for t in layer_types if t != "full_attention"})
    assert not non_full, (
        f"guided_rollout_multibool requires full-attention layers; "
        f"model.config.layer_types contains {non_full}. Hybrid models lose "
        f"conv/SSM state through past_key_values, breaking KV-fork. "
        f"Use a full-attention model (e.g. Qwen/Qwen3-0.6B) instead."
    )


@torch.no_grad()
def guided_rollout_multibool(
    model, tok,
    user_prompts: list[str],
    foundations: list[str] | None = None,
    *,
    max_think_tokens: int = 256,
    schema_hint: str = _DEFAULT_MULTIBOOL_HINT,
) -> list[MultiBoolResult]:
    """Score per-foundation violation logratio via 12 KV-forked single-slot completions.

    For each prompt:
      1. Batched think generate (≤max_think_tokens).
      2. Cache scoring_prefix = chat + think + </think>  (one batched fwd).
      3. For each (frame, foundation) ∈ {is_violation, is_ok} × foundations,
         fork from the cache and complete with `\\n{"<frame>": {"<f>":`. Read
         logits at the LAST suffix token (predicting ` true|false`).
      4. final[f] = 0.5 * (lr_violation[f] - lr_ok[f]). Flipping the framing
         flips which token means "violates", so averaging cancels per-key
         priors (e.g. JSON-true bias) without the chained-slot causality of
         the all-true / all-false fill design.

    Cost: 1 prefix forward + 12 small suffix forwards (each ≤14 tokens),
    all batched over B prompts.

    Requires a full-attention model -- asserts at entry."""
    _assert_full_attention(model)
    if tok.padding_side != "left":
        raise ValueError("tok.padding_side must be 'left' for batched rollout")
    if foundations is None:
        foundations = list(_DEFAULT_FOUNDATIONS)
    device = next(model.parameters()).device
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id

    # === Phase 1: think generation (batched) ===
    chats = []
    for up in user_prompts:
        full = f"{up}\n\n{schema_hint}" if schema_hint else up
        msgs = [{"role": "user", "content": full}]
        try:
            p = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        except TypeError:
            p = tok.apply_chat_template(msgs, tokenize=False)
        chats.append(p + "<think>\n")

    think_end_id = tok.convert_tokens_to_ids("</think>")
    if think_end_id in (None, getattr(tok, "unk_token_id", None)):
        think_end_id = tok.eos_token_id

    enc = tok(chats, return_tensors="pt", padding=True).to(device)
    prompt_len = enc.input_ids.shape[1]
    phase1 = model.generate(
        **enc,
        max_new_tokens=max_think_tokens,
        do_sample=False,
        eos_token_id=think_end_id,
        pad_token_id=pad_id,
    )

    # === Build per-row scoring_prefix ===
    sp_per_row: list[str] = []
    sp_ids_per_row: list[list[int]] = []
    think_per_row: list[tuple[str, int, bool]] = []  # (think_text, n_think, emitted_close)
    for i, p in enumerate(chats):
        gen_ids = phase1[i, prompt_len:]
        keep = gen_ids != pad_id
        gen_ids = gen_ids[keep] if keep.any() else gen_ids[:0]
        gen_text = tok.decode(gen_ids, skip_special_tokens=True)
        n_think = int(gen_ids.shape[0])
        emitted_close = _CLOSE_MARKER in gen_text
        if emitted_close:
            think_text, _ = gen_text.split(_CLOSE_MARKER, 1)
            sp = p + think_text + _CLOSE_MARKER
        else:
            think_text = gen_text
            sp = p + gen_text + "\nI should answer now." + _CLOSE_MARKER
        sp_per_row.append(sp)
        sp_ids_per_row.append(tok(sp, add_special_tokens=False)["input_ids"])
        think_per_row.append((think_text, n_think, emitted_close))

    B = len(sp_per_row)
    P_max = max(len(s) for s in sp_ids_per_row)

    # === Phase 2a: batched prefix forward (left-padded) ===
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

    # === Phase 2b: 12 forked suffix forwards. Cache is left-padded, so
    # position_ids must use the per-row REAL prefix length (not P_max).
    a_ids, b_ids = choice_token_ids_tf(tok)
    a_t = torch.tensor(a_ids, device=device, dtype=torch.long)
    b_t = torch.tensor(b_ids, device=device, dtype=torch.long)
    all_ids = torch.tensor(a_ids + b_ids, device=device, dtype=torch.long)

    def fork(suffixes: list[list[int]]) -> torch.Tensor:
        """Run one suffix forward with deepcopied cache. Returns [B, V] logp at
        the LAST real suffix token per row."""
        J_max = max(len(s) for s in suffixes)
        suf_input = torch.full((B, J_max), pad_id, dtype=torch.long, device=device)
        suf_mask = torch.zeros((B, J_max), dtype=torch.long, device=device)
        last_pos = torch.zeros(B, dtype=torch.long, device=device)
        for i, s in enumerate(suffixes):
            L = len(s)
            suf_input[i, :L] = torch.tensor(s, device=device)  # right-pad
            suf_mask[i, :L] = 1
            last_pos[i] = L - 1
        full_attn = torch.cat([pref_attn, suf_mask], dim=1)
        position_ids = pref_real[:, None] + torch.arange(J_max, device=device)[None, :]
        forked = copy.deepcopy(cache)
        out = model(
            input_ids=suf_input,
            attention_mask=full_attn,
            position_ids=position_ids,
            past_key_values=forked,
            use_cache=False,
        )
        logp = F.log_softmax(out.logits.float(), dim=-1)
        return logp[torch.arange(B, device=device), last_pos]  # [B, V]

    def suf_ids_for(frame: str, foundation: str) -> list[list[int]]:
        suf_text = f'\n{{"{frame}": {{"{foundation}":'
        return [tok(sp + suf_text, add_special_tokens=False)["input_ids"][len(sp_ids):]
                for sp, sp_ids in zip(sp_per_row, sp_ids_per_row)]

    lr_per: dict[tuple[str, str], torch.Tensor] = {}
    pm_per: dict[tuple[str, str], torch.Tensor] = {}
    for frame in ("is_violation", "is_ok"):
        for f in foundations:
            lp = fork(suf_ids_for(frame, f))  # [B, V]
            la = torch.logsumexp(lp[:, a_t], dim=-1)
            lb = torch.logsumexp(lp[:, b_t], dim=-1)
            lr_per[(frame, f)] = (la - lb).cpu()
            pm_per[(frame, f)] = lp[:, all_ids].exp().sum(-1).cpu()

    # === Aggregate: final = 0.5*(lr_violation - lr_ok), pmass = avg ===
    results: list[MultiBoolResult] = []
    for i, (think_text, n_think, emitted_close) in enumerate(think_per_row):
        per_logr: dict[str, float] = {}
        per_pm: dict[str, float] = {}
        per_lr_v: dict[str, float] = {}
        per_lr_o: dict[str, float] = {}
        for f in foundations:
            lr_v = float(lr_per[("is_violation", f)][i].item())
            lr_o = float(lr_per[("is_ok", f)][i].item())
            per_lr_v[f] = lr_v
            per_lr_o[f] = lr_o
            per_logr[f] = 0.5 * (lr_v - lr_o)
            pm_v = float(pm_per[("is_violation", f)][i].item())
            pm_o = float(pm_per[("is_ok", f)][i].item())
            per_pm[f] = 0.5 * (pm_v + pm_o)

        results.append(MultiBoolResult(
            user_prompt=user_prompts[i],
            think_text=think_text,
            logratios=per_logr,
            pmass_format=per_pm,
            lr_violation=per_lr_v,
            lr_ok=per_lr_o,
            think_tokens=n_think,
            emitted_close=emitted_close,
        ))

    # SHOULD: pmass≈1 at every (foundation, frame). Aggregate-once warning.
    worst = min(
        ((i, f, r.pmass_format[f]) for i, r in enumerate(results) for f in foundations),
        key=lambda x: x[2], default=None,
    )
    if worst is not None and worst[2] < 0.5:
        logger.warning(
            f"multibool pmass<0.5 at {worst[1]} on row {worst[0]} (pm={worst[2]:.3f}); "
            "schema may be drifting under steering"
        )
    return results


def choice_token_ids_tf(tok) -> list[list[int]]:
    def _variants(words):
        seen = []
        for s in words:
            tid = tok.encode(s, add_special_tokens=False)[-1]
            if tid not in seen:
                seen.append(tid)
        return seen
    return [_variants(["true", " true", "\ntrue", "True", " True", "\nTrue", "1"]),
            _variants(["false", " false", "\nfalse", "False", " False", "\nFalse", "0"])]
