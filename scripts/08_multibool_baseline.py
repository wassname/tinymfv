"""Run guided_rollout_multibool over the full classic vignette set.

Produces per-foundation logratios + correlations against human-rater % distributions,
as a baseline before wiring this eval into the steering sweep.

Usage:
    python scripts/08_multibool_baseline.py --model Qwen/Qwen3-0.6B
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import numpy as np
import polars as pl
import torch
from loguru import logger
from tabulate import tabulate
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from tinymfv.guided import guided_rollout_multibool, _DEFAULT_FOUNDATIONS

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--data", default=str(ROOT / "data" / "vignettes_other_violate.jsonl"))
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--max-think-tokens", type=int, default=128)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--dtype", default="bfloat16", choices=["float32", "float16", "bfloat16"])
    ap.add_argument("--out", default=str(ROOT / "data" / "results" / "multibool_baseline.jsonl"))
    args = ap.parse_args()

    rows = [json.loads(l) for l in Path(args.data).read_text().splitlines() if l.strip()]
    if args.limit:
        rows = rows[: args.limit]
    logger.info(f"loaded {len(rows)} vignettes from {args.data}")

    dtype = getattr(torch, args.dtype)
    logger.info(f"loading {args.model} on {args.device} dtype={args.dtype}")
    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=dtype).to(args.device).eval()

    foundations = list(_DEFAULT_FOUNDATIONS)
    out_rows: list[dict] = []
    n_low_pmass = 0

    for batch_start in tqdm(range(0, len(rows), args.batch_size), desc="multibool"):
        batch = rows[batch_start: batch_start + args.batch_size]
        prompts = [r["text"] for r in batch]
        results = guided_rollout_multibool(
            model, tok, prompts, foundations=foundations,
            max_think_tokens=args.max_think_tokens,
        )
        for src, res in zip(batch, results):
            row_pm = min(res.pmass_format.values())
            if row_pm < 0.5:
                n_low_pmass += 1
            out_rows.append({
                "id": src["id"],
                "foundation_coarse": src["foundation_coarse"],
                "wrong": src["wrong"],
                "text": src["text"],
                "human_pct": {f: src.get(f.capitalize(), "0 %") for f in foundations},
                "logratios": res.logratios,
                "lr_violation": res.lr_violation,
                "lr_ok": res.lr_ok,
                "pmass": res.pmass_format,
                "think_tokens": res.think_tokens,
                "emitted_close": res.emitted_close,
            })

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        for r in out_rows:
            f.write(json.dumps(r) + "\n")
    logger.info(f"wrote {len(out_rows)} rows to {out_path}; n_low_pmass={n_low_pmass}")

    # === Diagnostic table ===
    df = pl.DataFrame([
        {"foundation": f, **{
            "lr_mean": float(np.mean([r["logratios"][f] for r in out_rows])),
            "lr_std": float(np.std([r["logratios"][f] for r in out_rows])),
            "pm_mean": float(np.mean([r["pmass"][f] for r in out_rows])),
            "pm_min": float(np.min([r["pmass"][f] for r in out_rows])),
        }} for f in foundations
    ])
    print("\n=== per-foundation summary ===")
    print(tabulate(df.to_pandas(), headers="keys", tablefmt="pipe", floatfmt="+.3f", showindex=False))

    # === Spearman corr (manual: rank both arrays, compute Pearson on ranks) ===
    print("\n=== Spearman corr: model logratio vs human-rater % (cap-foundation) ===")
    print("SHOULD: ρ > 0.3 on at least 4/6 foundations; ρ < 0.1 on >2 means the eval doesn't track human moral judgement")
    corr_rows = []
    for f in foundations:
        xs = np.array([r["logratios"][f] for r in out_rows], dtype=float)
        ys = np.array([float(r["human_pct"][f].rstrip(" %")) for r in out_rows], dtype=float)
        if xs.std() == 0 or ys.std() == 0:
            rho = float("nan")
        else:
            rx = np.argsort(np.argsort(xs)).astype(float)
            ry = np.argsort(np.argsort(ys)).astype(float)
            rho = float(np.corrcoef(rx, ry)[0, 1])
        corr_rows.append({"foundation": f, "spearman_rho": rho, "n": len(xs),
                          "x_mean": float(xs.mean()), "y_mean": float(ys.mean())})
    print(tabulate(corr_rows, headers="keys", tablefmt="pipe", floatfmt="+.3f"))

    # === Final tldr ===
    print("\n=== TLDR ===")
    print(f"  rows scored: {len(out_rows)}")
    print(f"  low-pmass rows (any foundation < 0.5): {n_low_pmass}/{len(out_rows)}")
    avg_pm = float(np.mean([r["pmass"][f] for r in out_rows for f in foundations]))
    print(f"  mean pmass over all (row, foundation): {avg_pm:.3f}  (SHOULD: >0.9)")


if __name__ == "__main__":
    main()
