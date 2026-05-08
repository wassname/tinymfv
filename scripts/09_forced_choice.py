"""Run guided_rollout_forced_choice over a vignette set.

Single-token K-way primary-foundation probe. Each row gets a softmax over the
seven foundation first-tokens, averaged across n_permutations of the listed
order. Reports per-foundation top1 recall against `foundation_coarse`.

Usage:
    python scripts/09_forced_choice.py --model Qwen/Qwen3-0.6B --limit 32
    python scripts/09_forced_choice.py --model Qwen/Qwen3-0.6B --name clifford_ai
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from loguru import logger
from tabulate import tabulate
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from tinymfv.data import load_vignettes
from tinymfv.guided import guided_rollout_forced_choice, _DEFAULT_FORCED_FOUNDATIONS

ROOT = Path(__file__).resolve().parents[1]

# Map "social" (probe word) <-> "SocialNorms" (dataset coarse label).
_PROBE_TO_COARSE = {
    "care": "Care", "fairness": "Fairness", "loyalty": "Loyalty",
    "authority": "Authority", "sanctity": "Sanctity", "liberty": "Liberty",
    "social": "SocialNorms",
}
# Some Clifford rows use "Social Norms" with a space; normalise.
_COARSE_NORM = {"Social Norms": "SocialNorms"}


def _norm_coarse(s: str) -> str:
    return _COARSE_NORM.get(s, s)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--name", default="classic", help="dataset config (classic/scifi/clifford_ai)")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--max-think-tokens", type=int, default=128)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--dtype", default="bfloat16", choices=["float32", "float16", "bfloat16"])
    ap.add_argument("--cond", default="other_violate", choices=["other_violate", "self_violate"],
                    help="condition / framing axis (3rd vs 1st person)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    rows = load_vignettes(args.name)
    if args.limit:
        rows = rows[: args.limit]
    logger.info(f"loaded {len(rows)} {args.name} vignettes")

    dtype = getattr(torch, args.dtype)
    logger.info(f"loading {args.model} on {args.device} dtype={args.dtype}")
    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=dtype).to(args.device).eval()

    foundations = list(_DEFAULT_FORCED_FOUNDATIONS)

    # Diagnostic: show first-token resolution.
    print("\n=== first-token resolution ===")
    for f in foundations:
        ids = tok.encode(f, add_special_tokens=False)
        print(f"  {f!r:>14} -> {ids[0]:>6} {tok.decode([ids[0]])!r}  (full: {ids})")
    first_ids = [tok.encode(f, add_special_tokens=False)[0] for f in foundations]
    assert len(set(first_ids)) == len(first_ids), "first-token collision"
    print(f"  unique: yes ({len(set(first_ids))}/{len(foundations)})")

    out_rows: list[dict] = []
    for batch_start in tqdm(range(0, len(rows), args.batch_size), desc=f"forced-choice {args.name}"):
        batch = rows[batch_start: batch_start + args.batch_size]
        prompts = [r[args.cond] for r in batch]
        results = guided_rollout_forced_choice(
            model, tok, prompts, foundations=foundations,
            max_think_tokens=args.max_think_tokens,
        )
        for src, res in zip(batch, results):
            out_rows.append({
                "id": src["id"],
                "foundation_coarse": _norm_coarse(src["foundation_coarse"]),
                "wrong": src.get("wrong", True),
                "lp_fwd": res.lp_fwd,
                "lp_rev": res.lp_rev,
                "think_text": res.think_text,
                "think_text_rev": res.think_text_rev,
                "score": res.score,
                "p": res.p,
                "top1": res.top1,
                "top1_coarse": _PROBE_TO_COARSE[res.top1],
                "margin": res.margin,
                "think_tokens": res.think_tokens,
                "emitted_close": res.emitted_close,
            })

    out_path = Path(args.out) if args.out else (
        ROOT / "data" / "results" / f"forced_choice_{args.name}.jsonl")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        for r in out_rows:
            f.write(json.dumps(r) + "\n")
    logger.info(f"wrote {len(out_rows)} rows to {out_path}")

    # === Per-class recall ===
    coarse_set = sorted({_PROBE_TO_COARSE[f] for f in foundations})
    rec_rows = []
    correct_total = 0
    for coarse in coarse_set:
        items = [r for r in out_rows if r["foundation_coarse"] == coarse]
        if not items:
            continue
        n_correct = sum(1 for r in items if r["top1_coarse"] == coarse)
        correct_total += n_correct
        rec_rows.append({
            "foundation": coarse,
            "n": len(items),
            "recall": n_correct / len(items),
            "mean_p_true": float(np.mean([r["p"][_coarse_to_probe(coarse)] for r in items])),
            "mean_margin": float(np.mean([r["margin"] for r in items])),
        })
    print(f"\n=== per-class top1 recall on {args.name} (n={len(out_rows)}) ===")
    print("SHOULD: macro_recall >= 0.70 on classic for Qwen3-0.6B; "
          "comparable to panel 0.97 on bigger models")
    print(tabulate(rec_rows, headers="keys", tablefmt="pipe", floatfmt=".3f"))

    macro = float(np.mean([r["recall"] for r in rec_rows]))
    micro = correct_total / len(out_rows)
    print(f"\nmacro_recall = {macro:.3f}   micro_recall = {micro:.3f}")

    # Confusion summary: mass on the right foundation class (calibration check).
    print("\n=== p_top1 distribution (calibration of confidence) ===")
    p_top1 = np.array([max(r["p"].values()) for r in out_rows])
    print(f"  p_top1 min/median/mean/max: {p_top1.min():.3f} / "
          f"{np.median(p_top1):.3f} / {p_top1.mean():.3f} / {p_top1.max():.3f}")
    # SHOULD: median > 0.4 (clear winner). If <0.2, model is uniform -> probe broken.


def _coarse_to_probe(coarse: str) -> str:
    """Inverse of _PROBE_TO_COARSE."""
    inv = {v: k for k, v in _PROBE_TO_COARSE.items()}
    return inv[coarse]


if __name__ == "__main__":
    main()
