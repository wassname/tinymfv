"""Dump FULL (untruncated) text + options + coverage for WVS questions matching each IW theme, so
the precise per-item selector and axis-positive pole can be locked by eye. -- authored by Claude

  uv run python scripts/probe_iw_themes.py
"""
from __future__ import annotations

import ast
import re

from datasets import load_dataset

from tinymfv.zones import zone_of

SKIP = re.compile(r"don'?t know|no answer|refus|decline|none of|not applicable|^other|missing|inap",
                  re.I)

THEMES = {
    "Y_religion": r"how important it is in your life|believe in any|importance of god|attend religious",
    "Y_abortion": r"\babortion\b",
    "Y_childaut": r"encouraged to learn at home",
    "X_homosex": r"homosexual",
    "X_trust": r"most people can be trusted",
    "X_politaction": r"forms of political action",
}


def main() -> None:
    ds = load_dataset("Anthropic/llm_global_opinions", split="train")
    recs = []
    for r in ds:
        if r["source"] != "WVS" or not r["question"]:
            continue
        opts = ast.literal_eval(r["options"]) if isinstance(r["options"], str) else r["options"]
        keep = [i for i, o in enumerate(opts) if not SKIP.search(o)]
        sel = ast.literal_eval(re.search(r"\{.*\}", r["selections"], re.S).group(0))
        ncov = sum(1 for c in sel if zone_of(c) and sum(sel[c][i] for i in keep) > 0)
        recs.append({"q": r["question"], "opts": [opts[i] for i in keep], "ncov": ncov})

    for theme, pat in THEMES.items():
        rx = re.compile(pat, re.I)
        hits = sorted([r for r in recs if rx.search(r["q"])], key=lambda r: -r["ncov"])
        print(f"\n{'='*90}\n{theme}: {len(hits)} matches\n{'='*90}")
        for r in hits:
            if r["ncov"] < 40:
                continue
            tail = r["q"][-90:].replace("\n", " / ")
            print(f"[cov {r['ncov']:3d}] n_opts={len(r['opts'])} ...TAIL: {tail!r}")
            print(f"          opts: {r['opts']}")


if __name__ == "__main__":
    main()
