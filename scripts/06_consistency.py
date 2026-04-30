"""Pairwise same-situation consistency check between origin and rewrites.

For each vignette, compare:
    (origin, other_violate)  -- 3rd-person paraphrase: surface differs, content same
    (origin, self_violate)   -- 1st-person rewrite:    perspective shift allowed

Judge sees two anchors in-context for calibration:
    SAME      -- a hand-chosen paraphrase pair (known good)
    DIFFERENT -- a deliberate mismatch from two unrelated vignettes (known bad)

The judge then classifies the test pair as "same" or "different" with a reason.
We surface "different" verdicts as candidates for re-rewrite.

Caches per (config, condition, id) so reruns are cheap.

Outputs:
    data/consistency[_<name>].jsonl   one row per (id, condition)
    printed: pass/fail counts and the flagged rows
"""
from __future__ import annotations
import argparse
import asyncio
import hashlib
import json
import os
import random
import re
import sys
from collections import defaultdict
from pathlib import Path

from dotenv import load_dotenv
from loguru import logger
from openrouter_wrapper.retry import openrouter_request
from tabulate import tabulate
from tqdm.asyncio import tqdm as atqdm

from tinymfv.data import load_condition

ROOT = Path(__file__).resolve().parents[1]

CONDITIONS = ["other_violate", "self_violate"]

# Hand-picked anchors. The SAME pair is a clean 3rd-person paraphrase; the
# DIFFERENT pair is two unrelated Care vignettes with the same foundation but
# different actors/actions. The judge should call DIFFERENT "different" even
# though both are Care violations -- the test is *same situation*, not *same
# foundation*.
ANCHORS = {
    "other_violate": {
        "same_a": "You see a teenage boy chuckling at an amputee he passes by while on the subway.",
        "same_b": "You see a young man on the subway laughing at an amputee as he walks past.",
        "diff_a": "You see a girl laughing at another student forgetting her lines at a school play.",
        "diff_b": "You see a woman commenting out loud about how fat another woman looks in her jeans.",
    },
    "self_violate": {
        "same_a": "You see a teenage boy chuckling at an amputee he passes by while on the subway.",
        "same_b": "You chuckle at an amputee you pass by while on the subway.",
        "diff_a": "You see a girl laughing at another student forgetting her lines at a school play.",
        "diff_b": "You comment out loud about how fat another woman looks in her jeans.",
    },
}

JUDGE_PROMPT = """Task: decide whether two short scenarios describe the SAME situation (same actor archetype, same action, same wrongdoing, same severity), allowing only paraphrase{persp_clause}.

Examples for calibration:

PAIR 1 (verdict: same)
A: {same_a}
B: {same_b}
-> same: B is a paraphrase{persp_word} of A. Same actor, same action, same wrongdoing.

PAIR 2 (verdict: different)
C: {diff_a}
D: {diff_b}
-> different: both are unkind comments but the actor, target, and specific action differ.

Now classify this pair:

A: {test_a}
B: {test_b}

Output strict JSON only, no fences:
{{"verdict": "same" | "different", "reason": "<one short sentence>"}}"""

PERSP_CLAUSE = {
    "other_violate": "",
    "self_violate": " and a 3rd-person -> 1st-person perspective shift",
}
PERSP_WORD = {
    "other_violate": "",
    "self_violate": " (1st-person)",
}


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


async def judge_one(model: str, prompt: str, sem: asyncio.Semaphore) -> dict:
    async with sem:
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.0,
            "max_tokens": 200,
        }
        data = await openrouter_request(payload)
        text = data["choices"][0]["message"]["content"]
        obj = parse_json(text)
        if obj.get("verdict") not in ("same", "different"):
            raise ValueError(f"bad verdict in {obj}")
        return obj


async def judge_or_cache(cache: Path, model: str, prompt: str, ckey: str,
                         sem: asyncio.Semaphore) -> tuple[str, dict | None]:
    cf = cache / f"{ckey}.json"
    if cf.exists():
        return ckey, json.loads(cf.read_text())
    try:
        obj = await judge_one(model, prompt, sem)
        cf.write_text(json.dumps(obj))
        return ckey, obj
    except Exception as e:
        logger.warning(f"{ckey}: {e}")
        return ckey, None


def cache_dir(name: str) -> Path:
    sub = f"consistency_{name}" if name else "consistency"
    return ROOT / "data" / "cache" / sub


def out_path(name: str) -> Path:
    suf = f"_{name}" if name else ""
    return ROOT / "data" / f"consistency{suf}.jsonl"


async def amain(args) -> None:
    cache = cache_dir(args.name)
    cache.mkdir(parents=True, exist_ok=True)

    origin = {r["id"]: r for r in load_condition(args.name, "origin")}
    rewrites = {c: {r["id"]: r for r in load_condition(args.name, c)} for c in CONDITIONS}

    common = set(origin) & set(rewrites["other_violate"]) & set(rewrites["self_violate"])
    ids = sorted(common)
    if args.limit:
        ids = ids[: args.limit]
    logger.info(f"{len(ids)} vignettes x {len(CONDITIONS)} conditions = {len(ids)*len(CONDITIONS)} pairs via {args.model}")

    sem = asyncio.Semaphore(args.concurrency)
    tasks = []
    lookup: dict[str, tuple[str, str, str, str]] = {}
    for vid in ids:
        for cond in CONDITIONS:
            test_a = origin[vid]["text"]
            test_b = rewrites[cond][vid]["text"]
            anc = ANCHORS[cond]
            prompt = JUDGE_PROMPT.format(
                persp_clause=PERSP_CLAUSE[cond],
                persp_word=PERSP_WORD[cond],
                same_a=anc["same_a"], same_b=anc["same_b"],
                diff_a=anc["diff_a"], diff_b=anc["diff_b"],
                test_a=test_a, test_b=test_b,
            )
            ckey = f"{vid}_{cond}_{hkey(args.model)[:8]}"
            lookup[ckey] = (vid, cond, test_a, test_b)
            tasks.append(judge_or_cache(cache, args.model, prompt, ckey, sem))

    results: dict[str, dict | None] = {}
    for fut in atqdm.as_completed(tasks, total=len(tasks)):
        ckey, obj = await fut
        results[ckey] = obj

    out = out_path(args.name)
    by_cond: dict[str, dict[str, int]] = defaultdict(lambda: {"same": 0, "different": 0, "fail": 0})
    flagged: list[dict] = []
    by_id_cond: dict[str, dict[str, str]] = defaultdict(dict)
    with out.open("w") as fh:
        for ckey, (vid, cond, test_a, test_b) in lookup.items():
            obj = results.get(ckey)
            if obj is None:
                by_cond[cond]["fail"] += 1
                continue
            by_cond[cond][obj["verdict"]] += 1
            by_id_cond[vid][cond] = obj["verdict"]
            rec = {
                "id": vid, "condition": cond,
                "origin": test_a, "rewrite": test_b,
                "verdict": obj["verdict"], "reason": obj.get("reason", ""),
            }
            fh.write(json.dumps(rec) + "\n")
            if obj["verdict"] == "different":
                flagged.append(rec)

    print(f"\n{out.name}: {len(ids)} vignettes, {len(ids)*len(CONDITIONS)} pairs judged")
    rows = []
    for c in CONDITIONS:
        b = by_cond[c]
        n = b["same"] + b["different"]
        rows.append({
            "condition": c, "n": n,
            "same%": f"{100*b['same']/n:.1f}" if n else "-",
            "different": b["different"],
            "fail": b["fail"],
        })
    print(tabulate(rows, headers="keys", tablefmt="pipe"))

    # SHOULD: same% > 90 on both. ELSE rewriter is producing semantically off
    # paraphrases; re-rewrite the flagged ids or relax anchor strictness.
    bad_both = [vid for vid, cs in by_id_cond.items()
                if all(cs.get(c) == "different" for c in CONDITIONS)]
    print(f"\nvignettes flagged 'different' on BOTH conditions: {len(bad_both)}")
    if bad_both:
        for vid in bad_both[:10]:
            print(f"  {vid}  {origin[vid]['text'][:90]}")

    print(f"\n{len(flagged)} flagged pairs in {out}")
    print("first 8 flags:")
    for fl in flagged[:8]:
        print(f"  [{fl['condition']:14}] {fl['id'][:10]}  {fl['reason'][:120]}")
        print(f"      A: {fl['origin'][:100]}")
        print(f"      B: {fl['rewrite'][:100]}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="x-ai/grok-4-fast")
    ap.add_argument("--name", default="", help="config; '' = clifford default")
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
