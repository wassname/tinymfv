"""LLM-judge consistency check for vignette rewrites.

For each (vignette x condition), ask a strong cheap LLM (default grok-4-fast):
- Which Moral Foundations Theory foundation is most relevant?
- Did the actor violate or uphold it?

`other_violate` is the verbatim Clifford original, so its judge accuracy is
the ceiling -- the other 3 slots can't beat it. The diff is rewriter drift.

Async via openrouter_wrapper for retries + concurrency.

Outputs:
- data/validation.jsonl  -- one row per (vignette_id, condition)
- printed: foundation/valence accuracy split by slot, confusion matrix, flags
"""
from __future__ import annotations
import argparse
import asyncio
import hashlib
import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

from dotenv import load_dotenv
from loguru import logger
from openrouter_wrapper.retry import openrouter_request
from tabulate import tabulate
from tqdm.asyncio import tqdm as atqdm

ROOT = Path(__file__).resolve().parents[1]


def paths(name: str) -> tuple[Path, Path, Path]:
    suf = f"_{name}" if name else ""
    sub = f"validation_{name}" if name else "validation"
    return (ROOT / "data" / f"vignettes{suf}_rewritten.jsonl",
            ROOT / "data" / f"validation{suf}.jsonl",
            ROOT / "data" / "cache" / sub)

FOUNDATIONS = ["Care", "Fairness", "Loyalty", "Authority", "Sanctity", "Liberty", "Social Norms"]
CONDITIONS = ["other_violate", "other_uphold", "self_violate", "self_uphold"]
EXPECTED_VALENCE = {c: c.split("_")[1] for c in CONDITIONS}

JUDGE_PROMPT = """Classify this scenario by Moral Foundations Theory.

Scenario: {scenario}

Output strict JSON with these fields:
- "foundation": one of {foundations} (which foundation the action engages; "Social Norms" if none/weak)
- "valence": "violate" if the actor breaches the foundation, "uphold" if they honor it, "neutral" otherwise
- "reason": one short sentence

JSON only, no prose, no fences."""


def hkey(text: str) -> str:
    return hashlib.md5(text.encode("utf-8")).hexdigest()[:16]


def parse_json(s: str) -> dict:
    s = s.strip()
    if s.startswith("```"):
        s = re.sub(r"^```(?:json)?\s*|\s*```$", "", s, flags=re.MULTILINE)
    m = re.search(r"\{.*\}", s, flags=re.DOTALL)
    if m:
        s = m.group(0)
    return json.loads(s)


async def judge_one(model: str, scenario: str, sem: asyncio.Semaphore) -> dict:
    async with sem:
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": JUDGE_PROMPT.format(
                scenario=scenario, foundations=FOUNDATIONS)}],
            "temperature": 0.0,
            "max_tokens": 200,
        }
        data = await openrouter_request(payload)
        text = data["choices"][0]["message"]["content"]
        obj = parse_json(text)
        if "foundation" not in obj or "valence" not in obj:
            raise ValueError(f"missing fields in {obj}")
        return obj


async def judge_or_cache(cache: Path, model: str, scenario: str, ckey: str, sem: asyncio.Semaphore) -> tuple[str, dict | None]:
    cf = cache / f"{ckey}.json"
    if cf.exists():
        return ckey, json.loads(cf.read_text())
    try:
        judged = await judge_one(model, scenario, sem)
        cf.write_text(json.dumps(judged))
        return ckey, judged
    except Exception as e:
        logger.warning(f"{ckey}: {e}")
        return ckey, None


async def amain(args) -> None:
    in_path, out, cache = paths(args.name)
    cache.mkdir(parents=True, exist_ok=True)
    rows = [json.loads(l) for l in in_path.read_text().splitlines() if l.strip()]
    if args.limit:
        rows = rows[: args.limit]
    logger.info(f"{len(rows)} vignettes x 4 conditions = {len(rows)*4} judgments via {args.model} (concurrency={args.concurrency})")

    sem = asyncio.Semaphore(args.concurrency)
    tasks, lookup = [], {}
    for r in rows:
        for cond in CONDITIONS:
            ckey = f"{r['id']}_{cond}_{hkey(args.model)[:8]}"
            lookup[ckey] = (r, cond)
            tasks.append(judge_or_cache(cache, args.model, r[cond], ckey, sem))

    results: dict[str, dict | None] = {}
    for fut in atqdm.as_completed(tasks, total=len(tasks)):
        ckey, judged = await fut
        results[ckey] = judged

    # tally + write in fixed order
    confusion: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    flagged: list[dict] = []
    by_slot: dict[str, dict[str, int]] = defaultdict(lambda: {"f": 0, "v": 0, "n": 0})
    n_f = n_v = n_total = n_fail = 0

    with out.open("w") as fh:
        for r in rows:
            for cond in CONDITIONS:
                ckey = f"{r['id']}_{cond}_{hkey(args.model)[:8]}"
                judged = results.get(ckey)
                if judged is None:
                    n_fail += 1
                    continue
                f_match = judged["foundation"] == r["foundation_coarse"]
                v_match = judged["valence"] == EXPECTED_VALENCE[cond]
                n_total += 1
                n_f += int(f_match)
                n_v += int(v_match)
                by_slot[cond]["n"] += 1
                by_slot[cond]["f"] += int(f_match)
                by_slot[cond]["v"] += int(v_match)
                confusion[r["foundation_coarse"]][judged["foundation"]] += 1
                rec = {
                    "id": r["id"], "condition": cond, "scenario": r[cond],
                    "labeled_foundation": r["foundation_coarse"],
                    "judged_foundation": judged["foundation"],
                    "expected_valence": EXPECTED_VALENCE[cond],
                    "judged_valence": judged["valence"],
                    "foundation_match": f_match,
                    "valence_match": v_match,
                    "reason": judged.get("reason", ""),
                }
                fh.write(json.dumps(rec) + "\n")
                if not f_match or not v_match:
                    flagged.append(rec)

    print(f"\nfoundation accuracy: {n_f}/{n_total} = {100*n_f/n_total:.1f}%")
    print(f"valence accuracy:    {n_v}/{n_total} = {100*n_v/n_total:.1f}%")
    print(f"failures: {n_fail}")

    # SHOULD: other_violate >= the 3 rewrites on both metrics; if not, judge or original-label is the bottleneck
    print("\nby slot (other_violate = verbatim original = ceiling):")
    slot_rows = []
    for c in CONDITIONS:
        s = by_slot[c]
        slot_rows.append({
            "slot": c, "n": s["n"],
            "foundation%": f"{100*s['f']/s['n']:.1f}" if s["n"] else "-",
            "valence%": f"{100*s['v']/s['n']:.1f}" if s["n"] else "-",
        })
    print(tabulate(slot_rows, headers="keys", tablefmt="pipe"))

    print("\nconfusion (rows=labeled, cols=judged):")
    cm = []
    for f in FOUNDATIONS:
        row = {"labeled": f}
        for g in FOUNDATIONS:
            row[g] = confusion[f].get(g, 0)
        cm.append(row)
    print(tabulate(cm, headers="keys", tablefmt="pipe"))

    per_vig: dict[str, list[bool]] = defaultdict(list)
    for line in out.read_text().splitlines():
        rec = json.loads(line)
        per_vig[rec["id"]].append(rec["foundation_match"])
    bad_vigs = [vid for vid, ms in per_vig.items() if sum(ms) <= 1]
    print(f"\nvignettes with <=1/4 foundation matches: {len(bad_vigs)}/{len(per_vig)}")

    print(f"\n{len(flagged)} flagged condition-rows in {out}")
    print("first 8 flags:")
    for fl in flagged[:8]:
        print(f"  [{fl['labeled_foundation']}->{fl['judged_foundation']}] "
              f"({fl['expected_valence']}->{fl['judged_valence']}) "
              f"{fl['scenario'][:90]}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="x-ai/grok-4-fast")
    ap.add_argument("--name", default="", help="config name; '' = clifford default, else reads vignettes_<name>_rewritten.jsonl")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--concurrency", type=int, default=16)
    args = ap.parse_args()

    load_dotenv(ROOT / ".env")
    load_dotenv(ROOT.parent / "daily-dilemmas-self" / ".env")
    if not os.environ.get("OPENROUTER_API_KEY"):
        logger.error("OPENROUTER_API_KEY not set")
        sys.exit(1)
    asyncio.run(amain(args))


if __name__ == "__main__":
    main()
