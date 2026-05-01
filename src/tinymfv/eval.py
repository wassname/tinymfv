"""High-level entrypoint: model + tokenizer + vignettes -> report."""
from __future__ import annotations
import time
from typing import Any

import torch
from loguru import logger
from tqdm.auto import tqdm

from .core import format_prompts, next_token_logits, score_prompts, analyse, CONDITIONS, FRAMES
from .data import load_vignettes
from .guided import guided_rollout, choice_token_ids_tf


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
        logger.info(f"Using guided_rollout with {max_think_tokens} max_think_tokens (sequential)")
        p_true_list = []
        meta = []
        bool_mass_list = []
        choice_ids = choice_token_ids_tf(tokenizer)
        
        for r in tqdm(vignettes, desc="Evaluating"):
            for cond in CONDITIONS:
                for frame, fr in FRAMES.items():
                    user_prompt = f"{r[cond]}"
                    schema_hint = fr["q"]
                    prefill = fr["prefill"]
                    
                    res = guided_rollout(
                        model, tokenizer,
                        user_prompt=user_prompt,
                        choice_token_ids=choice_ids,
                        max_think_tokens=max_think_tokens,
                        schema_hint=schema_hint,
                        prefill=prefill,
                        verbose=False
                    )
                    
                    p_true_list.append(res.p_true)
                    meta.append((r["id"], r["foundation_coarse"], cond, frame, r.get("wrong")))
                    bool_mass_list.append(res.pmass_format)
        
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
