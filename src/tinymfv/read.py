"""Answer-token readout: administer an Instrument to a local LLM by reading the next-token
distribution at a prefilled answer slot.

Ported from the weight_steer_honesty experiment (`mft_honesty.admin.score_framing`) and
generalized to any ordinal `Instrument`: instead of hard-coding digits 1-5, we gather the mass
on the instrument's `answer_space` tokens after its `prefill`. Per InstrItem:

  build the chat prompt (user turn = schema_hint + statement, assistant prefill = instr.prefill),
  forward once, take the next-token log-probs, gather the prob on each answer token:

  - p        = renormalized distribution over answer_space (sums to 1). This is the per-(item,frame)
               categorical that `instrument.per_item_categorical` canonicalizes + averages.
  - pmass    = sum of raw (full-vocab) mass on the answer tokens: the coherence canary. Drops when
               the model leaks to refusals / prose / gibberish, independent of which option it picks.

Framing (forward / inverted / negated) is carried by each InstrItem.frame; canonicalization to a
single forward orientation happens downstream in `per_item_categorical`, NOT here. The reader is
frame-agnostic: it just reports the presented-orientation distribution.

Single-token requirement: every answer token must encode to exactly one id given the tokenizer,
and they must be distinct. Verified for Qwen ('(1' -> ['(','1']). A pmass collapse (not an error
here) is the canary that the prefill merged with the option and the readout went blind.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from loguru import logger

from .instrument import Instrument, InstrItem


def resolve_answer_ids(tok, answer_space: list[str]) -> list[int]:
    ids = [tok.encode(a, add_special_tokens=False) for a in answer_space]
    assert all(len(x) == 1 for x in ids), f"answer token not single-token: {list(zip(answer_space, ids))}"
    flat = [x[0] for x in ids]
    assert len(set(flat)) == len(flat), f"answer token id collision: {dict(zip(answer_space, flat))}"
    return flat


def build_prompt(tok, instr: Instrument, item: InstrItem) -> str:
    """Chat-templated user turn + assistant prefill that forces the answer slot.

    The user turn is `task\\n\\nStatement: <prompt>`. For ordinal surveys the task (response-scale
    legend) is FRAME-SPECIFIC -- inverted reverses the legend, negated negates the content -- so it
    travels per-item in `meta['task']`. Falls back to the instrument-level `schema_hint`, or the
    bare prompt (nominal vignettes)."""
    task = item.meta.get("task") or instr.schema_hint
    content = f"{task}\n\nStatement: {item.prompt}" if task else item.prompt
    text = tok.apply_chat_template(
        [{"role": "user", "content": content}],
        tokenize=False, add_generation_prompt=True, enable_thinking=False,
    )
    return text + instr.prefill


@torch.no_grad()
def read_items(model, tok, instr: Instrument, items: list[InstrItem], answer_ids: list[int],
               *, batch_size: int = 36, verbose_first: bool = False) -> list[dict]:
    """Score a list of InstrItems (one frame's worth, or any subset). Returns per-item rows with
    the keys `per_item_categorical` consumes: id, frame, p, pmass_allowed, dimension, sign, human_label."""
    device = next(model.parameters()).device
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    gid = torch.tensor(answer_ids, device=device)

    out: list[dict] = []
    for i in range(0, len(items), batch_size):
        chunk = items[i:i + batch_size]
        texts = [build_prompt(tok, instr, it) for it in chunk]
        enc = tok(texts, return_tensors="pt", padding=True, add_special_tokens=False).to(device)
        logits = model(**enc).logits[:, -1, :].float()        # [B, V] next-token
        logp = F.log_softmax(logits, dim=-1)
        p_a = logp[:, gid].exp()                              # [B, A] prob on each answer token
        pmass = p_a.sum(dim=-1)                               # [B]
        p_norm = p_a / pmass[:, None]                         # [B, A] within allowed
        for j, it in enumerate(chunk):
            out.append({
                "id": it.id, "frame": it.frame,
                "p": p_norm[j].cpu().numpy(),
                "pmass_allowed": float(pmass[j]),
                "dimension": it.dimension, "sign": it.sign,
                "human_label": it.human_label,
            })
        if verbose_first and i == 0:
            top = logp[0].topk(10)
            toks = " ".join(f"{tok.decode([int(t)])!r}:{float(p.exp()):.3f}"
                            for t, p in zip(top.indices, top.values))
            logger.debug(
                f"\n=== TRACE read first item ({instr.name}, special tokens on) ===\n"
                f"--- PROMPT+PREFILL ---\n{texts[0]}\n"
                f"--- top-10 next tokens ---\n{toks}\n"
                f"SHOULD: top tokens are the answer_space {instr.answer_space} (format locked by the "
                f"{instr.prefill!r} prefill); pmass_allowed={float(pmass[0]):.3f} near 1.0 -> coherent. "
                f"ELSE the prefill or chat template is off.\n")
    return out
