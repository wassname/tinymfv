"""P4 data foundation: what the GlobalOpinionQA WVS subset gives us for a WVS/Inglehart-Welzel map.

Anthropic/llm_global_opinions is MC questions with per-country human answer distributions -- almost
exactly tinymfv's instrument shape (allowed answer tokens + human anchors). This probe quantifies
the WVS subset's coverage and prints the parse recipe, so we know a country x question matrix is
dense enough to place models among human societies. No model runs here.

  uv run python scripts/probe_wvs_foundation.py
"""
from __future__ import annotations

import ast
import re
from collections import Counter

import numpy as np
from datasets import load_dataset


def parse_selections(s: str) -> dict[str, list[float]]:
    """`selections` ships as a repr of a defaultdict; pull the dict literal out and eval it safely."""
    return ast.literal_eval(re.search(r"\{.*\}", s, re.S).group(0))


def main() -> None:
    ds = load_dataset("Anthropic/llm_global_opinions", split="train")
    by_source = Counter(ds["source"])
    wvs = [r for r in ds if r["source"] == "WVS"]
    ncountry, allc = [], Counter()
    for r in wvs:
        sel = parse_selections(r["selections"])
        ncountry.append(len(sel))
        allc.update(sel.keys())
    print(f"source split: {dict(by_source)}")
    print(f"WVS questions: {len(wvs)}")
    print(f"distinct countries: {len(allc)}")
    print(f"countries/question min/median/max: {min(ncountry)}/{int(np.median(ncountry))}/{max(ncountry)}")
    print(f"questions with >=40 countries: {sum(n >= 40 for n in ncountry)}")
    print(f"top countries by coverage: {allc.most_common(12)}")
    r = wvs[0]
    sel = parse_selections(r["selections"])
    c0 = next(iter(sel))
    print("\nexample question:", repr(r["question"])[:160])
    print("options:", ast.literal_eval(r["options"]) if isinstance(r["options"], str) else r["options"])
    print(f"human dist for {c0}:", sel[c0])


if __name__ == "__main__":
    main()
