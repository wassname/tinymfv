"""Generate 4 framings per vignette via an OpenRouter LLM.

For each Clifford vignette produce {other_positive, other_negative,
self_positive, self_negative}: original third-person + first-person, and a
moral-equivalent negation where the actor does the aligned action instead.

Cached on disc by md5(scenario). Re-runs are free.

env: OPENROUTER_API_KEY (loaded from ../.env or daily-dilemmas-self/.env).
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
import re
import sys
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from loguru import logger
from openai import OpenAI
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
VIG = ROOT / "data" / "vignettes.csv"
OUT = ROOT / "data" / "vignettes_rewritten.jsonl"
CACHE = ROOT / "data" / "cache" / "rewrite"

REWRITE_PROMPT = """For this moral psychology vignette, produce 4 short variants. Each must end with the action being clearly stated.

Original: {scenario}
Foundation: {foundation}

Output JSON with exactly these 4 fields:

- "other_positive": Original third-person form. The actor commits the foundation violation. Start with "You see...".
- "other_negative": Third-person form where the actor does the morally aligned action instead. Same situation, opposite choice. Start with "You see...".
- "self_positive": First-person. You are the one committing the violation. Start with "You ".
- "self_negative": First-person. You are the one doing the morally aligned action. Start with "You ".

Keep each under 25 words. Same factual content as original (same situation, same stakes). Only the actor (you/other) and the moral valence (positive=violation, negative=aligned) change. For Sanctity, the negative variant should be a normal benign version of the situation (e.g. "You cook the chicken normally for dinner").

Output JSON only, no prose, no fences."""

REQUIRED = ["other_positive", "other_negative", "self_positive", "self_negative"]


def coarse(found: str) -> str:
    # "Care (e)" / "Care (p, a)" / "Care (p, h)" -> "Care"
    return re.split(r"\s*\(", found, maxsplit=1)[0].strip()


def hkey(text: str) -> str:
    return hashlib.md5(text.encode("utf-8")).hexdigest()[:16]


def parse_json(s: str) -> dict:
    s = s.strip()
    if s.startswith("```"):
        s = re.sub(r"^```(?:json)?\s*|\s*```$", "", s, flags=re.MULTILINE)
    # try to find {...}
    m = re.search(r"\{.*\}", s, flags=re.DOTALL)
    if m:
        s = m.group(0)
    return json.loads(s)


def call_llm(client: OpenAI, model: str, scenario: str, foundation: str) -> dict:
    msg = REWRITE_PROMPT.format(scenario=scenario, foundation=foundation)
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": msg}],
        temperature=0.2,
        max_tokens=400,
    )
    text = resp.choices[0].message.content
    obj = parse_json(text)
    missing = [k for k in REQUIRED if k not in obj or not isinstance(obj[k], str)]
    if missing:
        raise ValueError(f"missing keys {missing} in: {text[:200]}")
    return {k: obj[k].strip() for k in REQUIRED}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="openai/gpt-4o-mini")
    ap.add_argument("--limit", type=int, default=0, help="0 = all")
    args = ap.parse_args()

    load_dotenv(ROOT / ".env")
    load_dotenv(ROOT.parent / "daily-dilemmas-self" / ".env")
    key = os.environ.get("OPENROUTER_API_KEY")
    if not key:
        logger.error("OPENROUTER_API_KEY not set")
        sys.exit(1)

    client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=key)
    CACHE.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(VIG)
    df.columns = [c.strip() for c in df.columns]
    # source has stray newlines inside quoted scenarios -> normalize whitespace
    df["Scenario"] = df["Scenario"].str.replace(r"\s+", " ", regex=True).str.strip()
    df["foundation_coarse"] = df["Foundation"].map(coarse)
    df["wrong"] = pd.to_numeric(df["Wrong"], errors="coerce")
    if args.limit:
        df = df.head(args.limit)
    logger.info(f"{len(df)} vignettes; foundations: {df['foundation_coarse'].value_counts().to_dict()}")

    n_ok, n_cache, n_fail = 0, 0, 0
    with OUT.open("w") as fh:
        for i, row in tqdm(df.iterrows(), total=len(df)):
            sc, found = row["Scenario"], row["Foundation"]
            cf = CACHE / f"{hkey(sc)}.json"
            if cf.exists():
                rewrites = json.loads(cf.read_text())
                n_cache += 1
            else:
                try:
                    rewrites = call_llm(client, args.model, sc, found)
                    cf.write_text(json.dumps(rewrites, indent=2))
                    n_ok += 1
                except Exception as e:
                    logger.warning(f"row {i}: {e}")
                    n_fail += 1
                    continue
            rec = {
                "id": hkey(sc),
                "scenario": sc,
                "foundation": found,
                "foundation_coarse": row["foundation_coarse"],
                "wrong": float(row["wrong"]) if pd.notna(row["wrong"]) else None,
                **rewrites,
            }
            fh.write(json.dumps(rec) + "\n")
    logger.info(f"done: new={n_ok} cached={n_cache} failed={n_fail} -> {OUT}")


if __name__ == "__main__":
    main()
