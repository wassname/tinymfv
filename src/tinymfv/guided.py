from __future__ import annotations

import contextlib
from dataclasses import dataclass

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

    logits = model(score_ids).logits[0, -1].float()
    logp = F.log_softmax(logits, dim=-1)

    if (len(choice_token_ids) == 2 and all(isinstance(x, (list, tuple)) for x in choice_token_ids)):
        a_ids, b_ids = list(choice_token_ids[0]), list(choice_token_ids[1])
    else:
        a_ids, b_ids = list(choice_token_ids), []
        
    all_ids = torch.tensor(a_ids + b_ids, device=device, dtype=torch.long)
    pmass_format = float(logp[all_ids].exp().sum().item())

    # SHOULD: pmass≈1 (model picks one of the JSON-bool tokens). pmass<0.9
    # means the model is leaking probability to other tokens -> the schema
    # is being ignored or the steering vector has pushed the model OOD.
    if pmass_format < 0.9:
        topk = torch.topk(logp.exp(), k=5)
        toks = [tok.decode([i]) for i in topk.indices.tolist()]
        probs = topk.values.tolist()
        top5 = ", ".join(f"{repr(t)}={p:.3f}" for t, p in zip(toks, probs))
        logger.warning(f"pmass={pmass_format:.3f}<0.9 — top-5: {top5}")

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
    score_logits = model(**score_enc).logits[:, -1].float()
    score_logp = F.log_softmax(score_logits, dim=-1)

    if (len(choice_token_ids) == 2 and all(isinstance(x, (list, tuple)) for x in choice_token_ids)):
        a_ids, b_ids = list(choice_token_ids[0]), list(choice_token_ids[1])
    else:
        a_ids, b_ids = list(choice_token_ids), []
    all_ids = torch.tensor(a_ids + b_ids, device=device, dtype=torch.long)
    a_t = torch.tensor(a_ids, device=device, dtype=torch.long) if a_ids else None
    b_t = torch.tensor(b_ids, device=device, dtype=torch.long) if b_ids else None

    results = []
    for i, (up, (think_text, emitted_close, emitted_prefill, n_think)) in enumerate(zip(user_prompts, per_row)):
        logp = score_logp[i]
        pmass_format = float(logp[all_ids].exp().sum().item())
        if pmass_format < 0.9:
            topk = torch.topk(logp.exp(), k=5)
            toks = [tok.decode([j]) for j in topk.indices.tolist()]
            probs = topk.values.tolist()
            top5 = ", ".join(f"{repr(t)}={pp:.3f}" for t, pp in zip(toks, probs))
            logger.warning(f"pmass={pmass_format:.3f}<0.9 — top-5: {top5}")
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
        ))
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
