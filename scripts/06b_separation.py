"""Panel separation check: do decorrelated cheap LLMs identify a single
moral foundation per item, and does the panel agree on which one?

Run on `classic`, `scifi`, and `ai-actor` to compare separation.
The hypothesis: all three configs should behave like single-foundation
datasets. If panel agreement collapses, the rewrite/transcription drifted.

Method
------
Forced-choice judging. Each judge LLM, per item, picks ONE primary foundation
violation plus a runner-up plus an integer margin (1-5: how much more does the
primary apply than the runner-up). This sidesteps the "everything looks bad"
collapse from independent yes/no probes -- a forced choice has to pick.

Two judging frames per (item, judge) for bias mitigation:
  - "violation":   which foundation does this MOST VIOLATE?
  - "preserves":   which foundation does this MOST FAIL TO UPHOLD?
A judge's per-item verdict = majority of the two frames (else "violation" wins).

Panel = 4 cheap models from different families, in parallel:
  google/gemini-2.5-flash, openai/gpt-5-mini, anthropic/claude-haiku-4.5,
  x-ai/grok-4-fast.

Per item we report: panel_top1 (majority foundation across judges),
panel_agreement (frac of judges with correct top1), mean margin.
Per dataset: macro-recall of foundation_coarse, mean agreement, mean margin,
and item-level confusion matrix.

Outputs
-------
data/separation_<name>.jsonl        per-item per-judge verdicts
data/separation_<name>_summary.json dataset-level metrics
printed: comparison table across configs

Usage
-----
python scripts/06b_separation.py                          # all three configs
python scripts/06b_separation.py --names classic
python scripts/06b_separation.py --limit 16 --concurrency 8
"""
from __future__ import annotations
import argparse
import asyncio
import hashlib
import json
import os
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

from dotenv import load_dotenv
from loguru import logger
from openrouter_wrapper.retry import openrouter_request
from tabulate import tabulate
from tqdm.asyncio import tqdm as atqdm

ROOT = Path(__file__).resolve().parents[1]

FOUNDATIONS = ("Care", "Fairness", "Loyalty", "Authority",
               "Sanctity", "Liberty", "SocialNorms")

# Map dataset's coarse label -> our canonical token (drops the space in
# "Social Norms").
_COARSE_NORM = {f: f for f in FOUNDATIONS} | {"Social Norms": "SocialNorms"}

DEFAULT_PANEL = (
    "google/gemini-2.5-flash",
    "openai/gpt-5-mini",
    "anthropic/claude-haiku-4.5",
    "x-ai/grok-4-fast",
)

# Forced-choice prompt. Foundation rubric copied verbatim from 07_multilabel.py
# so judges across scripts are answering the same question.
_RUBRIC = """- Care: violates norms of harm/care (unkindness, causing physical or emotional pain to humans or animals)
- Fairness: violates norms of fairness/justice (cheating, free-riding, reducing equality)
- Loyalty: violates norms of loyalty (betraying a group: family, country, team)
- Authority: violates norms of respecting authority (disobedience, subversion, lack of respect for tradition or institutions)
- Sanctity: violates norms of purity/sanctity (degrading, sexually deviant, disgusting acts; physical disgust; contamination)
- Liberty: violates norms of freedom (coercion, bullying, dominating, reducing freedom of choice)
- SocialNorms: violates social conventions but not moral rules (unusual, weird, awkward behavior)"""

_PROMPT_VIOLATION = """Which SINGLE Moral Foundations Theory foundation does this scenario most clearly violate?

Foundations:
{rubric}

Scenario: {scenario}

Pick exactly one primary foundation, then the closest runner-up, then a 1-5 margin
(1 = nearly tied, 5 = primary dominates). Output strict JSON only:
{{"primary": "<one of {fs}>", "runner_up": "<one of {fs}>", "margin": N, "reason": "one short sentence"}}"""

_PROMPT_PRESERVES = """Which SINGLE Moral Foundations Theory foundation does this scenario most clearly FAIL TO UPHOLD?

Foundations:
{rubric}

Scenario: {scenario}

Pick exactly one primary foundation that is most failed/disrespected, then the closest
runner-up, then a 1-5 margin (1 = nearly tied, 5 = primary dominates). Output strict JSON:
{{"primary": "<one of {fs}>", "runner_up": "<one of {fs}>", "margin": N, "reason": "one short sentence"}}"""

FRAMES = {"violation": _PROMPT_VIOLATION, "preserves": _PROMPT_PRESERVES}


def hkey(s: str) -> str:
    return hashlib.md5(s.encode("utf-8")).hexdigest()[:12]


def parse_json(s: str) -> dict:
    s = s.strip()
    if s.startswith("```"):
        s = re.sub(r"^```(?:json)?\s*|\s*```$", "", s, flags=re.MULTILINE)
    m = re.search(r"\{.*\}", s, flags=re.DOTALL)
    if m:
        s = m.group(0)
    return json.loads(s)


def _norm_foundation(s: str) -> str:
    if s is None:
        raise ValueError("foundation None")
    t = re.sub(r"[\s_-]", "", str(s)).lower()
    for f in FOUNDATIONS:
        if f.lower() == t:
            return f
    raise ValueError(f"unknown foundation: {s!r}")


async def judge_one(
    cache: Path, model: str, frame: str, scenario: str, vid: str,
    sem: asyncio.Semaphore,
) -> tuple[str, str, str, dict | None]:
    cf = cache / f"{vid}_{frame}_{hkey(model)}.json"
    if cf.exists():
        cached = json.loads(cf.read_text())
        if cached.get("primary"):
            return vid, frame, model, cached
        return vid, frame, model, None
    prompt = FRAMES[frame].format(
        rubric=_RUBRIC, scenario=scenario,
        fs=", ".join(FOUNDATIONS),
    )
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0,
        "max_tokens": 200,
    }
    async with sem:
        try:
            data = await openrouter_request(payload)
            text = data["choices"][0]["message"]["content"]
            obj = parse_json(text)
            primary = _norm_foundation(obj["primary"])
            runner = _norm_foundation(obj.get("runner_up", obj["primary"]))
            margin = int(obj.get("margin", 3))
            reason = str(obj.get("reason", ""))[:200]
            out = {"primary": primary, "runner_up": runner, "margin": margin, "reason": reason}
            cf.write_text(json.dumps(out))
            return vid, frame, model, out
        except Exception as e:
            logger.warning(f"{vid} {frame} via {model}: {e}")
            cf.write_text(json.dumps({"primary": None, "error": str(e)[:200]}))
            return vid, frame, model, None


def _resolve_top1(by_frame: dict[str, dict | None]) -> tuple[str | None, int]:
    """Resolve a single judge's top1 across the two frames + return mean margin."""
    primaries = [v["primary"] for v in by_frame.values() if v and v.get("primary")]
    if not primaries:
        return None, 0
    cnt = Counter(primaries)
    top, n_top = cnt.most_common(1)[0]
    if n_top == 1 and "violation" in by_frame and by_frame["violation"]:
        top = by_frame["violation"]["primary"]
    margins = [v["margin"] for v in by_frame.values() if v and v.get("margin")]
    return top, int(round(sum(margins) / max(1, len(margins))))


def _src_path(name: str) -> Path:
    suf = "" if name == "classic" else f"_{name}"
    return ROOT / "data" / f"vignettes{suf}_other_violate.jsonl"


def _load(name: str) -> list[dict]:
    p = _src_path(name)
    if not p.exists():
        raise FileNotFoundError(p)
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()]


async def run_config(name: str, args, panel: tuple[str, ...]) -> dict:
    rows = _load(name)
    if args.limit:
        rows = rows[: args.limit]
    cache = ROOT / "data" / "cache" / f"separation_{name}"
    cache.mkdir(parents=True, exist_ok=True)
    sem = asyncio.Semaphore(args.concurrency)

    tasks = []
    for r in rows:
        for frame in FRAMES:
            for model in panel:
                tasks.append(judge_one(cache, model, frame, r["text"], r["id"], sem))

    # results[vid][model][frame] = obj
    results: dict[str, dict[str, dict[str, dict | None]]] = defaultdict(lambda: defaultdict(dict))
    for fut in atqdm.as_completed(tasks, total=len(tasks), desc=f"{name} judging"):
        vid, frame, model, obj = await fut
        results[vid][model][frame] = obj

    # Per-item panel resolution.
    per_item = []
    correct_per_class: dict[str, list[int]] = defaultdict(list)
    panel_agreements: list[float] = []
    margins: list[int] = []
    confusion: dict[str, Counter] = defaultdict(Counter)
    for r in rows:
        gold = _COARSE_NORM[r["foundation_coarse"]]
        judge_top1: dict[str, str | None] = {}
        judge_margin: dict[str, int] = {}
        for model in panel:
            top1, m = _resolve_top1(results[r["id"]].get(model, {}))
            judge_top1[model] = top1
            judge_margin[model] = m
        votes = [t for t in judge_top1.values() if t is not None]
        if not votes:
            continue
        cnt = Counter(votes)
        panel_top1, _ = cnt.most_common(1)[0]
        agree_correct = sum(1 for t in votes if t == gold) / len(votes)
        panel_agreements.append(agree_correct)
        margins.extend(m for m in judge_margin.values() if m)
        correct_per_class[gold].append(int(panel_top1 == gold))
        confusion[gold][panel_top1] += 1
        per_item.append({
            "id": r["id"], "foundation_coarse": gold, "panel_top1": panel_top1,
            "panel_agreement": round(agree_correct, 3),
            "judge_top1": judge_top1, "judge_margin": judge_margin,
        })

    out_jsonl = ROOT / "data" / f"separation_{name}.jsonl"
    with out_jsonl.open("w") as fh:
        for x in per_item:
            fh.write(json.dumps(x) + "\n")

    macro_recall = {f: (sum(v) / len(v) if v else float("nan"), len(v))
                    for f, v in correct_per_class.items()}
    summary = {
        "name": name,
        "n": len(per_item),
        "panel": list(panel),
        "macro_recall_mean": sum(a for a, _ in macro_recall.values()) / max(1, len(macro_recall)),
        "panel_agreement_mean": sum(panel_agreements) / max(1, len(panel_agreements)),
        "margin_mean": sum(margins) / max(1, len(margins)),
        "per_class_recall": {f: {"recall": r, "n": n} for f, (r, n) in macro_recall.items()},
        "confusion": {gold: dict(c) for gold, c in confusion.items()},
    }
    (ROOT / "data" / f"separation_{name}_summary.json").write_text(json.dumps(summary, indent=2))
    return summary


async def amain(args) -> None:
    panel = tuple(args.panel.split(",")) if args.panel else DEFAULT_PANEL
    names = args.names or ["classic", "scifi", "ai-actor"]
    summaries = []
    for name in names:
        try:
            s = await run_config(name, args, panel)
            summaries.append(s)
        except FileNotFoundError as e:
            logger.warning(f"skip {name}: {e}")

    headline = []
    for s in summaries:
        headline.append([s["name"], s["n"],
                         f"{s['macro_recall_mean']:.2f}",
                         f"{s['panel_agreement_mean']:.2f}",
                         f"{s['margin_mean']:.2f}"])
    print("\n=== panel separation across configs ===")
    print(tabulate(headline,
                   headers=["config", "n", "macro_recall", "panel_agreement", "mean_margin"],
                   tablefmt="github"))

    print("\n=== per-class panel-top1 recall ===")
    classes = list(FOUNDATIONS)
    rows = []
    for f in classes:
        row = [f]
        for s in summaries:
            r = s["per_class_recall"].get(f, {"recall": float("nan"), "n": 0})
            row.append(f"{r['recall']:.2f} (n={r['n']})")
        rows.append(row)
    print(tabulate(rows, headers=["foundation"] + [s["name"] for s in summaries],
                   tablefmt="github"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--names", nargs="*", default=None,
                    help="configs to evaluate (default: classic scifi ai-actor)")
    ap.add_argument("--panel", default="",
                    help="comma-separated OR model ids; empty = default 4-judge panel")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--concurrency", type=int, default=12)
    args = ap.parse_args()

    load_dotenv(ROOT / ".env")
    load_dotenv(ROOT.parent / "daily-dilemmas-self" / ".env")
    if not os.environ.get("OPENROUTER_API_KEY"):
        logger.error("OPENROUTER_API_KEY not set")
        sys.exit(1)
    asyncio.run(amain(args))


if __name__ == "__main__":
    main()
