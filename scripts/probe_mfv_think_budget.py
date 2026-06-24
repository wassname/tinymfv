"""Diagnose the MFV (nominal forced-choice) coherence collapse vs think budget.

Job 183 showed MFV pmass=0.166, top1=0.24 at max_think_tokens=256, while the
ordinal path (same model, same _rollout core) reads at pmass 0.997. The aux
stats explain it exactly: emitted_close=220/264 and 44/264 = 0.1667 == pmass.
The collapse is guided.py case (c): when the model closes </think> inside the
budget and no natural prefill window is found, the forced read is discarded
(pmass=0.0, NaN) rather than used, because stacking a second </think>+prefill
on a completed generation pollutes the slot.

If that is the whole story, a SHORT think budget (where the model rarely closes
think, like the ordinal path's 64) should recover pmass. This probe sweeps the
budget and prints coherence per setting. No researched semantics touched.

  uv run --extra benchmark python scripts/probe_mfv_think_budget.py --model Qwen/Qwen3.5-4B
"""
from __future__ import annotations

import argparse

import torch
from loguru import logger
from transformers import AutoModelForCausalLM, AutoTokenizer

from tinymfv import evaluate


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3.5-4B")
    ap.add_argument("--budgets", type=int, nargs="*", default=[8, 16, 64, 256])
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--n-samples", type=int, default=1,
                    help="BMA over N stochastic think traces (needs temperature>0 to differ)")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16).to(args.device).eval()

    logger.info(f"SHOULD: pmass climbs as the budget shrinks (fewer </think> closes -> "
                f"fewer case-(c) collapses). If pmass stays ~0.17 even at budget=8, the bug "
                f"is NOT just case (c) and needs the code fix, not a config change.")
    logger.info(f"n_samples={args.n_samples} temperature={args.temperature}")
    logger.info("budget  pmass   top1   emitted_close")
    for b in args.budgets:
        rep = evaluate(model, tok, name="classic", max_think_tokens=b,
                       batch_size=args.batch_size, verbose=0,
                       n_samples=args.n_samples, temperature=args.temperature)
        logger.info(f"{b:>6}  {rep['mean_pmass_allowed']:.3f}  {rep['top1_acc']:.3f}")


if __name__ == "__main__":
    main()
