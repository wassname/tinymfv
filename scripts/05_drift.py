"""Per-vignette drift analysis: where verbatim is judged correctly but a rewrite isn't.

Reads validation jsonl produced by 04_validate.py. The verbatim `other_violate` slot
sets the per-vignette ceiling -- if it fails, that's a Clifford-label vs modern-judge
disagreement (not the rewriter's fault). If it succeeds but a rewrite slot fails,
that's rewriter drift and a candidate for re-rewriting.

Output: counts split into (judge_disagrees, rewriter_drifts) and the actionable
rewriter_drift rows printed for review.
"""
from __future__ import annotations
import argparse
import json
from collections import defaultdict
from pathlib import Path

from tabulate import tabulate

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="")
    args = ap.parse_args()

    suf = f"_{args.name}" if args.name else ""
    path = ROOT / "data" / f"validation{suf}.jsonl"
    rows = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]

    by_id: dict[str, dict[str, dict]] = defaultdict(dict)
    for r in rows:
        by_id[r["id"]][r["condition"]] = r

    judge_disagree, rewriter_drift = [], []
    for vid, conds in by_id.items():
        ov = conds.get("other_violate")
        if ov is None:
            continue
        ov_ok = ov["foundation_match"] and ov["valence_match"]
        for cond in ["other_uphold", "self_violate", "self_uphold"]:
            r = conds.get(cond)
            if r is None:
                continue
            r_ok = r["foundation_match"] and r["valence_match"]
            if r_ok:
                continue
            if not ov_ok:
                judge_disagree.append(r)
            else:
                rewriter_drift.append(r)

    n_total_rewrites = len(by_id) * 3
    print(f"\n{path.name}: {len(by_id)} vignettes, {n_total_rewrites} rewrite-slot judgments")
    print(f"  judge disagrees on the original too:  {len(judge_disagree):3d}  ({100*len(judge_disagree)/n_total_rewrites:.1f}%, not actionable)")
    print(f"  rewriter drift (verbatim ok, rewrite not): {len(rewriter_drift):3d}  ({100*len(rewriter_drift)/n_total_rewrites:.1f}%, actionable)")

    drift_by_cond: dict[str, int] = defaultdict(int)
    drift_by_kind: dict[str, int] = defaultdict(int)
    for r in rewriter_drift:
        drift_by_cond[r["condition"]] += 1
        if not r["foundation_match"] and not r["valence_match"]:
            drift_by_kind["both"] += 1
        elif not r["foundation_match"]:
            drift_by_kind["foundation_only"] += 1
        else:
            drift_by_kind["valence_only"] += 1

    print("\ndrift by condition:")
    print(tabulate([{"condition": k, "n": v} for k, v in drift_by_cond.items()], headers="keys", tablefmt="pipe"))
    print("\ndrift by kind:")
    print(tabulate([{"kind": k, "n": v} for k, v in drift_by_kind.items()], headers="keys", tablefmt="pipe"))

    print("\nrewriter drift rows:")
    for r in rewriter_drift:
        f_str = f"{r['labeled_foundation']}->{r['judged_foundation']}"
        v_str = f"{r['expected_valence']}->{r['judged_valence']}"
        print(f"  {r['id'][:10]} {r['condition']:15} [{f_str}] ({v_str}) {r['scenario'][:80]}")


if __name__ == "__main__":
    main()
