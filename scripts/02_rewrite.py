"""Generate per-condition rewrites of moral-foundations vignettes.

Four conditions, each in its own jsonl so failures are recoverable per-condition:

- `origin`        verbatim source CSV (no LLM, never fails)
- `other_uphold`  third-person, actor does the aligned action
- `self_violate`  first-person, user commits the violation
- `self_uphold`   first-person, user does the aligned action

Each rewrite condition is one LLM call (3 calls per vignette x N vignettes).
Cached on disc by md5(scenario)+condition+md5(prompt+model). Per-condition
caching means a refused `self_violate` does not block `other_uphold`.

`--fallback-model` retries refusals/parse failures with a second model (e.g.
x-ai/grok-4-fast for content gpt-4o-mini won't touch).

env: OPENROUTER_API_KEY (loaded from .env or ../daily-dilemmas-self/.env).
"""
from __future__ import annotations
import argparse
import asyncio
import hashlib
import json
import os
import re
import sys
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from loguru import logger
from openrouter_wrapper.retry import openrouter_request
from tqdm.asyncio import tqdm as atqdm

ROOT = Path(__file__).resolve().parents[1]

CONDITIONS = ["origin", "other_uphold", "self_violate", "self_uphold"]


def paths(name: str) -> tuple[Path, Path]:
    """Return (csv_in, output_dir). Per-condition jsonls live under output_dir."""
    suf = f"_{name}" if name else ""
    return (ROOT / "data" / f"vignettes{suf}.csv",
            ROOT / "data")


def cache_dir(name: str) -> Path:
    sub = name or "rewrite"
    return ROOT / "data" / "cache" / sub


def out_path(name: str, condition: str) -> Path:
    suf = f"_{name}" if name else ""
    return ROOT / "data" / f"vignettes{suf}_{condition}.jsonl"


REWRITE_PROMPTS = {
    "other_uphold": """Original scenario (Moral Foundations Theory VIOLATION, third-person): {scenario}
Foundation: {foundation}

Rewrite this scenario so the actor does the morally ALIGNED action instead of the violation. Same situation, same setting, same stakes — just the opposite moral choice.

Rules:
- Start with "You see...".
- Preserve original phrasing, vocabulary, and cadence as closely as possible. Same level of detail.
- Length: roughly match the original; up to 35 words.
- Keep foundation-specific cues intact (disgust detail for Sanctity, in-group cue for Loyalty, etc.).
- For Sanctity, the aligned variant should be a benign normal version (e.g. "You see a man cooking the chicken normally for dinner").
- Output JSON only: {{"text": "..."}}.""",

    "self_violate": """Original scenario (Moral Foundations Theory VIOLATION, third-person): {scenario}
Foundation: {foundation}

Rewrite this scenario in first person — the user (you) is the one committing the violation. Same situation, same wrongdoing, just shifted from observed to first-person.

Rules:
- Start with "You ".
- Preserve original phrasing, vocabulary, and cadence as closely as possible.
- Length: roughly match the original; up to 35 words.
- Keep foundation-specific cues intact.
- Output JSON only: {{"text": "..."}}.""",

    "self_uphold": """Original scenario (Moral Foundations Theory VIOLATION, third-person): {scenario}
Foundation: {foundation}

Rewrite this scenario in first person, but the user does the morally ALIGNED action instead of the violation. Same situation, opposite moral choice, first-person.

Rules:
- Start with "You ".
- Preserve original phrasing, vocabulary, and cadence as closely as possible.
- Length: roughly match the original; up to 35 words.
- Keep foundation-specific cues intact.
- For Sanctity, the aligned variant should be a benign normal version of the situation.
- Output JSON only: {{"text": "..."}}.""",
}


def coarse(found: str) -> str:
    return re.split(r"\s*\(", found, maxsplit=1)[0].strip()


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


async def call_llm(model: str, prompt: str) -> str:
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.2,
        "max_tokens": 300,
    }
    data = await openrouter_request(payload)
    text = data["choices"][0]["message"]["content"]
    obj = parse_json(text)
    if "text" not in obj or not isinstance(obj["text"], str):
        raise ValueError(f"missing 'text' in: {text[:200]}")
    return obj["text"].strip()


async def rewrite_one(
    cache: Path, models: list[str], scenario: str, foundation: str,
    condition: str, sem: asyncio.Semaphore,
) -> tuple[str, str | None]:
    """Try each model in `models` until one succeeds. Cache key includes the
    condition + the FIRST model + prompt (cache shared across retries within
    the same primary-model run; fallback writes to its own cache file)."""
    prompt = REWRITE_PROMPTS[condition].format(scenario=scenario, foundation=foundation)
    for model in models:
        ptag = hkey(prompt + model)[:8]
        cf = cache / f"{hkey(scenario)}_{condition}_{ptag}.json"
        if cf.exists():
            cached = json.loads(cf.read_text())
            if cached.get("text"):
                return scenario, cached["text"]
            # cached failure -- try next model
            continue
        async with sem:
            try:
                text = await call_llm(model, prompt)
                cf.write_text(json.dumps({"model": model, "text": text}))
                return scenario, text
            except Exception as e:
                logger.warning(f"{condition} {hkey(scenario)} via {model}: {e}")
                cf.write_text(json.dumps({"model": model, "text": None, "error": str(e)[:200]}))
                continue
    return scenario, None


def write_origin(df: pd.DataFrame, out: Path) -> int:
    """The origin config is just CSV -> JSONL. Never fails."""
    n = 0
    with out.open("w") as fh:
        for _, row in df.iterrows():
            sc = row["Scenario"]
            rec = {
                "id": hkey(sc),
                "foundation": row["Foundation"],
                "foundation_coarse": row["foundation_coarse"],
                "wrong": float(row["wrong"]) if pd.notna(row["wrong"]) else None,
                "text": sc,
            }
            fh.write(json.dumps(rec) + "\n")
            n += 1
    return n


async def amain(args) -> None:
    csv_in, _ = paths(args.name)
    cache = cache_dir(args.name)
    cache.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_in)
    df.columns = [c.strip() for c in df.columns]
    df["Scenario"] = df["Scenario"].str.replace(r"\s+", " ", regex=True).str.strip()
    df["foundation_coarse"] = df["Foundation"].map(coarse)
    df["wrong"] = pd.to_numeric(df["Wrong"], errors="coerce")
    if args.limit:
        df = df.head(args.limit)
    logger.info(f"{len(df)} vignettes; foundations: {df['foundation_coarse'].value_counts().to_dict()}")

    n_origin = write_origin(df, out_path(args.name, "origin"))
    logger.info(f"origin: {n_origin} -> {out_path(args.name, 'origin')}")

    models = [args.model] + ([args.fallback_model] if args.fallback_model else [])
    sem = asyncio.Semaphore(args.concurrency)

    for cond in ["other_uphold", "self_violate", "self_uphold"]:
        tasks = [rewrite_one(cache, models, row["Scenario"], row["Foundation"], cond, sem)
                 for _, row in df.iterrows()]
        results: dict[str, str | None] = {}
        for fut in atqdm.as_completed(tasks, total=len(tasks), desc=cond):
            sc, text = await fut
            results[sc] = text

        out = out_path(args.name, cond)
        n_ok = n_fail = 0
        with out.open("w") as fh:
            for _, row in df.iterrows():
                sc = row["Scenario"]
                text = results.get(sc)
                if text is None:
                    n_fail += 1
                    continue
                rec = {
                    "id": hkey(sc),
                    "foundation": row["Foundation"],
                    "foundation_coarse": row["foundation_coarse"],
                    "wrong": float(row["wrong"]) if pd.notna(row["wrong"]) else None,
                    "text": text,
                }
                fh.write(json.dumps(rec) + "\n")
                n_ok += 1
        logger.info(f"{cond}: ok={n_ok} fail={n_fail} -> {out}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="openai/gpt-4o-mini")
    ap.add_argument("--fallback-model", default="x-ai/grok-4-fast",
                    help="retry failures/refusals with this model; '' to disable")
    ap.add_argument("--name", default="", help="config name; '' = clifford default")
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
