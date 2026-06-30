from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
from tabulate import tabulate

from tinymfv.instrument import canonicalize_to_forward
from tinymfv.readouts import logit_contrast


def _rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _logmeanexp(x: np.ndarray, axis: int) -> np.ndarray:
    m = np.max(x, axis=axis, keepdims=True)
    return np.squeeze(m, axis=axis) + np.log(np.mean(np.exp(x - m), axis=axis))


def _profile_c(rows: list[dict], sample_idx_by_row: list[np.ndarray]) -> dict[str, float]:
    by_id: dict[str, list[dict]] = defaultdict(list)
    for row, sample_idx in zip(rows, sample_idx_by_row, strict=True):
        sample_lp = np.asarray(row["sample_lp"], dtype=float)
        lp = _logmeanexp(sample_lp[sample_idx], axis=0)
        lp_fwd = canonicalize_to_forward(lp, row["framing"], "ordinal")
        by_id[row["id"]].append({
            "foundation": row["foundation"],
            "sign": int(row["sign"]),
            "lp": lp_fwd,
            "scale_max": int(row["scale_max"]),
        })

    by_foundation: dict[str, list[float]] = defaultdict(list)
    for item_id, item_rows in by_id.items():
        assert len({r["foundation"] for r in item_rows}) == 1, item_id
        assert len({r["sign"] for r in item_rows}) == 1, item_id
        assert len({r["scale_max"] for r in item_rows}) == 1, item_id
        lp_item = np.stack([r["lp"] for r in item_rows]).mean(axis=0)
        sign = item_rows[0]["sign"]
        c = logit_contrast(lp_item, item_rows[0]["scale_max"])
        by_foundation[item_rows[0]["foundation"]].append(-c if sign < 0 else c)

    return {foundation: float(np.mean(vals)) for foundation, vals in by_foundation.items()}


def _sample_idx(rows: list[dict], n: int, rng: np.random.Generator) -> list[np.ndarray]:
    out = []
    for row in rows:
        sample_count = len(row["sample_lp"])
        out.append(rng.choice(sample_count, size=n, replace=True))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", type=Path, required=True)
    ap.add_argument("--instrument", default="mfq2")
    ap.add_argument("--foundation", default="authority")
    ap.add_argument("--subset-n", default="1,2,4,8")
    ap.add_argument("--boot", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    files = sorted(args.run_dir.glob(f"{args.instrument}_*_samples.jsonl"))
    assert files, f"no {args.instrument}_*_samples.jsonl files in {args.run_dir}"
    by_c = {float(_rows(path)[0]["c"]): _rows(path) for path in files}
    assert 0.0 in by_c, sorted(by_c)

    sample_counts = {
        len(row["sample_lp"])
        for rows in by_c.values()
        for row in rows
    }
    assert len(sample_counts) == 1, sample_counts
    max_n = next(iter(sample_counts))

    subset_ns = [int(x) for x in args.subset_n.split(",")]
    assert all(1 <= n <= max_n for n in subset_ns), (subset_ns, max_n)

    rng = np.random.default_rng(args.seed)
    out_rows = []
    for n in subset_ns:
        deltas_by_c: dict[float, list[float]] = {c: [] for c in sorted(by_c) if c != 0.0}
        for _ in range(args.boot):
            base = _profile_c(by_c[0.0], _sample_idx(by_c[0.0], n, rng))[args.foundation]
            for c, rows in by_c.items():
                if c == 0.0:
                    continue
                prof = _profile_c(rows, _sample_idx(rows, n, rng))[args.foundation]
                deltas_by_c[c].append(prof - base)
        for c, vals in deltas_by_c.items():
            arr = np.asarray(vals, dtype=float)
            expected_sign = 1.0 if c > 0 else -1.0
            out_rows.append({
                "instrument": args.instrument,
                "foundation": args.foundation,
                "subset_n": n,
                "max_n": max_n,
                "c": c,
                "mean_delta_C": float(arr.mean()),
                "std_delta_C": float(arr.std(ddof=1)),
                "ci95_lo": float(np.percentile(arr, 2.5)),
                "ci95_hi": float(np.percentile(arr, 97.5)),
                "signed_rate": float(np.mean(np.sign(arr) == expected_sign)),
                "boot": args.boot,
            })

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(out_rows[0]))
        writer.writeheader()
        writer.writerows(out_rows)

    print(tabulate(out_rows, headers="keys", tablefmt="pipe", floatfmt="+.3f"))


if __name__ == "__main__":
    main()
