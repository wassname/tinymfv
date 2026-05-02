"""High-level entrypoint: model + tokenizer + vignettes -> report."""
from __future__ import annotations
import time
from typing import Any

import torch
from loguru import logger
from tqdm.auto import tqdm

from .core import format_prompts, next_token_logits, score_prompts, analyse, CONDITIONS, FRAMES
from .data import load_vignettes
from .guided import guided_rollout_batch, choice_token_ids_tf


def evaluate(
    model,
    tokenizer,
    name: str = "",
    vignettes: list[dict] | None = None,
    batch_size: int = 16,
    device: str | None = None,
    max_think_tokens: int = 64,
) -> dict[str, Any]:
    """Run dual JSON-bool eval and return aggregated report.

    Either pass `vignettes` directly or `name` to load from `data/`. Tokenizer must
    have a chat template (or fallback flat format will be used) and `pad_token` set.
    Side-effects: sets `tokenizer.padding_side='left'` and `tokenizer.pad_token` if
    missing -- both required for batched left-padded eval.
    """
    if vignettes is None:
        vignettes = load_vignettes(name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    if device is None:
        device = next(model.parameters()).device.type

    t0 = time.time()
    
    if max_think_tokens > 0:
        logger.info(f"Using guided_rollout_batch with {max_think_tokens} max_think_tokens, batch_size={batch_size}")
        choice_ids = choice_token_ids_tf(tokenizer)

        # Build all (vid, cond, frame) items, grouped by frame so each batch
        # shares schema_hint + prefill (collapses the per-row branching).
        items_per_frame: dict[str, list[tuple]] = {f: [] for f in FRAMES}
        for r in vignettes:
            for cond in CONDITIONS:
                for frame in FRAMES:
                    items_per_frame[frame].append(
                        (r["id"], r["foundation_coarse"], cond, frame, r.get("wrong"), r[cond])
                    )

        # Pretokenize a sample to log expected prompt length / cache budget.
        sample_user = items_per_frame[next(iter(FRAMES))][0][5]
        sample_q = FRAMES[next(iter(FRAMES))]["q"]
        sample_full = f"{sample_user}\n\n{sample_q}"
        sample_msgs = [{"role": "user", "content": sample_full}]
        try:
            sample_p = tokenizer.apply_chat_template(sample_msgs, tokenize=False, add_generation_prompt=True)
        except TypeError:
            sample_p = tokenizer.apply_chat_template(sample_msgs, tokenize=False)
        sample_p = sample_p + "<think>\n"
        sample_len = len(tokenizer(sample_p).input_ids)
        logger.info(
            f"SHOULD: prompt_len≈{sample_len} tok; max cache ≈ {sample_len + max_think_tokens} per row × "
            f"batch_size={batch_size}. If OOM, lower batch_size."
        )

        p_true_list, meta, bool_mass_list = [], [], []
        total = sum(len(v) for v in items_per_frame.values())
        with tqdm(total=total, desc="Evaluating") as pbar:
            for frame, items in items_per_frame.items():
                fr = FRAMES[frame]
                schema_hint = fr["q"]
                prefill = fr["prefill"]
                for i in range(0, len(items), batch_size):
                    chunk = items[i:i + batch_size]
                    user_prompts = [it[5] for it in chunk]
                    results = guided_rollout_batch(
                        model, tokenizer,
                        user_prompts=user_prompts,
                        choice_token_ids=choice_ids,
                        max_think_tokens=max_think_tokens,
                        schema_hint=schema_hint,
                        prefill=prefill,
                    )
                    for it, res in zip(chunk, results):
                        vid, found, cond, fr_name, wrong, _ = it
                        p_true_list.append(res.p_true)
                        meta.append((vid, found, cond, fr_name, wrong))
                        bool_mass_list.append(res.pmass_format)
                    pbar.update(len(chunk))

        elapsed = time.time() - t0
        logger.info(f"guided eval: {elapsed:.1f}s ({len(p_true_list)/elapsed:.1f} prompts/s)")

        report = analyse(p_true_list, meta, bool_mass=bool_mass_list)
        
    else:
        logger.info("Using standard batched next_token_logits")
        prompts, meta = format_prompts(tokenizer, vignettes)
        logits = next_token_logits(model, tokenizer, prompts, device, batch_size)
        elapsed = time.time() - t0
        logger.info(f"forward pass: {elapsed:.1f}s ({len(prompts)/elapsed:.1f} prompts/s)")

        scored = score_prompts(logits, tokenizer)
        report = analyse(scored["p_true"], meta, bool_mass=scored["bool_mass"])
        
    report["info"]["elapsed_s"] = elapsed
    report["info"]["name"] = name
    return report
