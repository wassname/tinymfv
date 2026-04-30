"""High-level entrypoint: model + tokenizer + vignettes -> report."""
from __future__ import annotations
import time
from typing import Any

import torch
from loguru import logger

from .core import format_prompts, next_token_logits, score_prompts, analyse
from .data import load_vignettes


def evaluate(
    model,
    tokenizer,
    name: str = "",
    vignettes: list[dict] | None = None,
    batch_size: int = 16,
    device: str | None = None,
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

    prompts, meta = format_prompts(tokenizer, vignettes)
    t0 = time.time()
    logits = next_token_logits(model, tokenizer, prompts, device, batch_size)
    elapsed = time.time() - t0
    logger.info(f"forward pass: {elapsed:.1f}s ({len(prompts)/elapsed:.1f} prompts/s)")

    scored = score_prompts(logits, tokenizer)
    report = analyse(scored["p_true"], meta, bool_mass=scored["bool_mass"])
    report["info"]["elapsed_s"] = elapsed
    report["info"]["name"] = name
    return report
