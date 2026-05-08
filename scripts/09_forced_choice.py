"""Run forced-choice 7-way primary-foundation probe over a vignette set.

Wraps `tinymfv.evaluate()`. Reports the AI-vs-label distribution match:
    top1_acc       argmax model == argmax label
    mean_js        Jensen-Shannon (model || label), nats; uniform baseline
                   ~ ln 7 = 1.95, max = ln 2 = 0.693
    pearson[f]     cross-vignette Pearson(model_p[f], label_p[f]) on
                   labeled rows (other_violate condition).

Labels:
    classic: human_*    (Clifford 2015 % distributions)
    (paraphrased sets carry the same `human_*` as their classic parent;
     `ai_*` columns are available for cross-source diagnostics)

Usage:
    python scripts/09_forced_choice.py --model Qwen/Qwen3-0.6B
    python scripts/09_forced_choice.py --model Qwen/Qwen3-4B --name ai-actor
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from loguru import logger
from tabulate import tabulate
from transformers import AutoModelForCausalLM, AutoTokenizer

from tinymfv import evaluate, load_vignettes
from tinymfv.guided import _DEFAULT_FORCED_FOUNDATIONS

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--name", default="classic", help="dataset config (classic/scifi/ai-actor)")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--max-think-tokens", type=int, default=128)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--dtype", default="bfloat16", choices=["float32", "float16", "bfloat16"])
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    vig = load_vignettes(args.name)
    if args.limit:
        vig = vig[: args.limit]
    logger.info(f"loaded {len(vig)} {args.name} vignettes")

    dtype = getattr(torch, args.dtype)
    logger.info(f"loading {args.model} on {args.device} dtype={args.dtype}")
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=dtype).to(args.device).eval()

    # Diagnostic: show first-token resolution.
    print("\n=== first-token resolution ===")
    for f in _DEFAULT_FORCED_FOUNDATIONS:
        ids = tok.encode(f, add_special_tokens=False)
        print(f"  {f!r:>14} -> {ids[0]:>6} {tok.decode([ids[0]])!r}  (full: {ids})")

    out = evaluate(
        model, tok, args.name, vignettes=vig,
        batch_size=args.batch_size,
        max_think_tokens=args.max_think_tokens,
        return_per_row=True,
    )

    # Persist per-row predictions for downstream analysis.
    out_path = Path(args.out) if args.out else (
        ROOT / "data" / "results" / f"forced_choice_{args.name}.jsonl")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        for r in out["per_row"]:
            rec = {
                "id": r["id"],
                "condition": r["condition"],
                "foundation_coarse": r["foundation_coarse"],
                "p": {f: float(r["p"][i]) for i, f in enumerate(_DEFAULT_FORCED_FOUNDATIONS)},
                "label": (None if r["label"] is None
                          else {f: float(r["label"][i]) for i, f in enumerate(_DEFAULT_FORCED_FOUNDATIONS)}),
                "top1": r["top1"],
                "margin": float(r["margin"]),
            }
            f.write(json.dumps(rec) + "\n")
    logger.info(f"wrote {len(out['per_row'])} rows to {out_path}")

    # === Per-foundation table ===
    print(f"\n=== per-foundation aggregates on {args.name} ===")
    print("SHOULD: pearson_label > 0.5 on most foundations for a well-calibrated model")
    print(tabulate(out["table"], headers="keys", tablefmt="pipe", floatfmt=".3f", showindex=False))

    # === Headline scalars ===
    print(f"\n=== AI-vs-label headlines on {args.name} (n={len(out['per_row'])}) ===")
    print("SHOULD: top1_acc >> 1/7=0.14 (uniform); mean_js << ln 7 = 1.95 (uniform vs label)")
    print(f"  top1_acc = {out['top1_acc']}")
    print(f"  mean_js  = {out['mean_js']}    (max possible = ln 2 = 0.693)")

    # Confidence calibration
    p_top1 = np.array([float(r["p"].max()) for r in out["per_row"]])
    print(f"\n  p_top1 min/median/mean/max: {p_top1.min():.3f} / "
          f"{np.median(p_top1):.3f} / {p_top1.mean():.3f} / {p_top1.max():.3f}")
    print("  SHOULD: median > 0.4 (clear winner per row); <0.2 -> probe broken")


if __name__ == "__main__":
    main()
