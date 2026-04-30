"""CLI wrapper around `tinymcf.evaluate`. Dual Y/N probe per vignette x condition.

See `src/tinymcf/core.py` for the scoring logic. This script just loads the model,
runs `evaluate(...)`, prints the table, and writes a JSON summary.

Usage:
    python scripts/03_eval.py --model Qwen/Qwen3-0.6B
    python scripts/03_eval.py --model Qwen/Qwen3-0.6B --name scifi --tag step_500
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import torch
from loguru import logger
from tabulate import tabulate
from transformers import AutoModelForCausalLM, AutoTokenizer

from tinymcf import evaluate, format_prompt, FRAMES
from tinymcf.core import next_token_logits  # for sanity sample
from tinymcf.data import load_vignettes

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "data" / "results"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--name", default="", help="config; '' = clifford default")
    ap.add_argument("--tag", default="", help="label for output file")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--dtype", default="bfloat16", choices=["float32", "float16", "bfloat16"])
    args = ap.parse_args()

    rows = load_vignettes(args.name)
    if args.limit:
        rows = rows[: args.limit]
    logger.info(f"{len(rows)} vignettes loaded")

    dtype = getattr(torch, args.dtype)
    logger.info(f"loading {args.model} on {args.device} dtype={args.dtype}")
    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=dtype).to(args.device)
    model.eval()

    # SHOULD: top-10 next tokens for sample include 'Yes' / 'No' in positions 1-3.
    # ELSE prompt format is broken -- model is not completing "A: ___".
    sample = format_prompt(tok, rows[0]["other_violate"], FRAMES["wrong"])
    enc = tok(sample, return_tensors="pt").to(args.device)
    with torch.inference_mode():
        out = model(**enc)
    probs = out.logits[0, -1].float().softmax(-1)
    topk = torch.topk(probs, 10)
    logger.info("--- top-10 next tokens for sample (Q_wrong) ---")
    for p, i in zip(topk.values, topk.indices):
        logger.info(f"  {tok.decode([int(i)])!r:>15}  p={float(p):.3f}")

    report = evaluate(model, tok, name=args.name, vignettes=rows, batch_size=args.batch_size, device=args.device)
    df = report["table"]

    print(tabulate(df, headers="keys", floatfmt="+.3f", tablefmt="pipe", showindex=False))
    print()
    info = report["info"]
    print(f"yn_mass mean={info['yn_mass_mean']:.3f} (>0.5 -> Yes/No dominate; <0.1 -> prompt broken)")
    print(f"inter-frame agreement (corr p_yes_wrong vs 1-p_yes_accept) = {info['interframe_agreement_corr']:+.3f} (negative -> yes-bias dominates raw signal; OK because dual-frame cancels in delta)")
    if info.get("human_corr") is not None:
        print(f"per-vignette corr(s_other_violate, human Wrong) = {info['human_corr']:+.3f} (want > 0.4 on clifford; meaningless for hand-labeled configs)")
    print()
    print(f"HEADLINE  align_other(real)={report['score']:+.3f}  self_other_gap(real)={report['gap']:+.3f}  align_other(SocialNorms control)={report['sn']:+.3f}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tag = args.tag or args.model.replace("/", "_")
    name_suf = f"_{args.name}" if args.name else ""
    out = OUT_DIR / f"eval{name_suf}_{tag}.json"
    out.write_text(json.dumps({
        "model": args.model,
        "name": args.name,
        "tag": args.tag,
        "frames": FRAMES,
        "headline_align_other": report["score"],
        "headline_gap": report["gap"],
        "social_norms_align": report["sn"],
        "by_foundation": df.to_dict(orient="records"),
        **info,
    }, indent=2))
    logger.info(f"wrote {out}")


if __name__ == "__main__":
    main()
