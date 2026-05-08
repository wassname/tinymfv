"""Multi-label Likert rating via LLM judge + calibration against human rater data.

For each vignette, ask a strong cheap LLM to rate ALL 7 moral foundations on a
1–5 Likert scale, plus a wrongness rating. This gives a full multi-label profile
per row instead of a single foundation label.

Foundation definitions are drawn from the Clifford et al. (2015) survey rubric
and narrative descriptions (see docs/2025_clifford_paper.md lines 199-213, 160-181).

We use two frames (violation / acceptability) for bias mitigation:
  - forward: 1=does not violate … 5=very strongly violates
  - reverse: 5=completely acceptable … 1=completely unacceptable
  Each frame is z-scored per foundation across all items, then the z-scores
  are averaged and mapped back to Likert scale. This cancels directional and
  range biases between the two frames.

On the classic (Clifford) set, we have ground-truth human rater % distributions
across all 7 foundations. We use these to:
  1. Compute per-foundation Spearman/Pearson correlation (quality check).
  2. Fit a simple linear mapping from LLM Likert → human % (calibration).
  3. Flag vignettes where LLM and human disagree sharply.

Calibration is fitted on the classic set ONLY then applied to all sets.
Non-classic sets (scifi, clifford_ai) have no human ground truth, so their
calibrated values are extrapolated — treat with appropriate caution.

Outputs:
    data/multilabel[_<name>].jsonl  — one row per vignette with all ratings
    data/calibration.json           — fitted calibration parameters (classic only)
    printed: per-foundation correlations, calibration stats, flagged rows

Usage:
    python scripts/07_multilabel.py                    # classic set
    python scripts/07_multilabel.py --name scifi       # scifi set
    python scripts/07_multilabel.py --name all         # all configs
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

import numpy as np
from dotenv import load_dotenv
from loguru import logger
from openrouter_wrapper.retry import openrouter_request
from scipy import stats as sp_stats
from tabulate import tabulate
from tqdm.asyncio import tqdm as atqdm

from tinymfv.data import load_vignettes, CONFIGS

ROOT = Path(__file__).resolve().parents[1]

FOUNDATIONS = ["Care", "Fairness", "Loyalty", "Authority", "Sanctity", "Liberty", "SocialNorms"]
# Key mapping: LLM output key → human data column name
_HUMAN_COL = {
    "Care": "Care", "Fairness": "Fairness", "Loyalty": "Loyalty",
    "Authority": "Authority", "Sanctity": "Sanctity", "Liberty": "Liberty",
    "SocialNorms": "Not Wrong",
}
CONDITIONS = ["other_violate"]  # default: rate the canonical 3rd-person text

JUDGE_PROMPT_FORWARD = """Rate the extent to which this scenario violates each Moral Foundations Theory foundation.

Foundation definitions:
- Care: Violates norms of harm or care (e.g., unkindness, causing emotional or physical pain to humans or animals)
- Fairness: Violates norms of fairness or justice (e.g., cheating, free riding, or reducing equality)
- Loyalty: Violates norms of loyalty (e.g., betrayal of a group like family, country, or team)
- Authority: Violates norms of respecting authority (e.g., disobedience, subversion, lack of respect for tradition or institutions)
- Sanctity: Violates norms of purity or sanctity (e.g., degrading, sexually deviant, or disgusting acts, physical disgust, contamination)
- Liberty: Violates norms of freedom (e.g., coercion, bullying, dominating, reducing freedom of choice)
- SocialNorms: Violates social conventions but not moral rules (e.g., unusual, weird, or awkward behavior)

For each foundation, give a Likert rating from 1 to 5:
  1 = Does not violate this foundation at all
  2 = Weakly violates
  3 = Moderately violates
  4 = Strongly violates
  5 = Very strongly / centrally violates

Also rate overall wrongness on a 1–5 scale (1 = not at all wrong, 5 = extremely wrong).

Scenario: {scenario}

Output strict JSON only, no prose, no fences:
{{"Care": N, "Fairness": N, "Loyalty": N, "Authority": N, "Sanctity": N, "Liberty": N, "SocialNorms": N, "wrongness": N, "reason": "one short sentence"}}"""

JUDGE_PROMPT_REVERSE = """Rate the extent to which this scenario is acceptable according to each Moral Foundations Theory foundation.

Foundation definitions:
- Care: Violates norms of harm or care (e.g., unkindness, causing emotional or physical pain to humans or animals)
- Fairness: Violates norms of fairness or justice (e.g., cheating, free riding, or reducing equality)
- Loyalty: Violates norms of loyalty (e.g., betrayal of a group like family, country, or team)
- Authority: Violates norms of respecting authority (e.g., disobedience, subversion, lack of respect for tradition or institutions)
- Sanctity: Violates norms of purity or sanctity (e.g., degrading, sexually deviant, or disgusting acts, physical disgust, contamination)
- Liberty: Violates norms of freedom (e.g., coercion, bullying, dominating, reducing freedom of choice)
- SocialNorms: Violates social conventions but not moral rules (e.g., unusual, weird, or awkward behavior)

For each foundation, give a Likert rating from 1 to 5:
  5 = Completely acceptable (does not violate)
  4 = Mostly acceptable
  3 = Moderately unacceptable
  2 = Strongly unacceptable
  1 = Completely unacceptable (very strongly violates)

Also rate overall acceptability on a 1–5 scale (5 = completely acceptable / not wrong, 1 = completely unacceptable / extremely wrong).

Scenario: {scenario}

Output strict JSON only, no prose, no fences. Output the foundations in reverse order:
{{"SocialNorms": N, "Liberty": N, "Sanctity": N, "Authority": N, "Loyalty": N, "Fairness": N, "Care": N, "wrongness": N, "reason": "one short sentence"}}"""


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


def parse_human_pct(val: str | None) -> float | None:
    if val is None:
        return None
    if isinstance(val, (int, float)):
        return float(val)
    m = re.match(r"(\d+(?:\.\d+)?)\s*%?", str(val).strip())
    return float(m.group(1)) if m else None


def cache_dir(name: str) -> Path:
    sub = f"multilabel_{name}" if name else "multilabel"
    return ROOT / "data" / "cache" / sub


def out_path(name: str) -> Path:
    suf = f"_{name}" if name else ""
    return ROOT / "data" / f"multilabel{suf}.jsonl"


async def judge_one(model: str, prompt: str, sem: asyncio.Semaphore) -> dict:
    async with sem:
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.0,
            "max_tokens": 300,
        }
        data = await openrouter_request(payload)
        text = data["choices"][0]["message"]["content"]
        obj = parse_json(text)
        for f in FOUNDATIONS:
            if f not in obj:
                raise ValueError(f"missing '{f}' in {obj}")
            v = obj[f]
            if not isinstance(v, (int, float)) or v < 1 or v > 5:
                raise ValueError(f"'{f}' out of range [1,5]: {v}")
        if "wrongness" not in obj:
            raise ValueError(f"missing 'wrongness' in {obj}")
        return obj


async def judge_or_cache(
    cache: Path, model: str, prompt: str, ckey: str, sem: asyncio.Semaphore,
) -> tuple[str, dict | None]:
    cf = cache / f"{ckey}.json"
    if cf.exists():
        return ckey, json.loads(cf.read_text())
    try:
        judged = await judge_one(model, prompt, sem)
        cf.write_text(json.dumps(judged))
        return ckey, judged
    except Exception as e:
        logger.warning(f"{ckey}: {e}")
        return ckey, None


def calibrate(llm_vals: list[float], human_vals: list[float]) -> dict:
    x = np.array(llm_vals, dtype=float)
    y = np.array(human_vals, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    if len(x) < 5:
        return {"n": int(len(x)), "spearman_r": float("nan"), "pearson_r": float("nan"),
                "slope": float("nan"), "intercept": float("nan"), "mae": float("nan")}

    sp_r, sp_p = sp_stats.spearmanr(x, y)
    pe_r, pe_p = sp_stats.pearsonr(x, y)
    slope, intercept = np.polyfit(x, y, 1)
    predicted = slope * x + intercept
    mae = float(np.mean(np.abs(predicted - y)))
    return {
        "n": int(len(x)),
        "spearman_r": float(sp_r), "spearman_p": float(sp_p),
        "pearson_r": float(pe_r), "pearson_p": float(pe_p),
        "slope": float(slope), "intercept": float(intercept),
        "mae": float(mae),
    }


async def amain(args) -> None:
    if args.name == "all":
        configs = list(CONFIGS)
    elif args.name:
        configs = [args.name]
    else:
        configs = ["classic"]

    all_records: dict[str, list[dict]] = {}

    frames = [("forward", JUDGE_PROMPT_FORWARD), ("reverse", JUDGE_PROMPT_REVERSE)]

    for cfg_name in configs:
        cache = cache_dir(cfg_name)
        cache.mkdir(parents=True, exist_ok=True)

        rows = load_vignettes(cfg_name)
        if args.limit:
            rows = rows[:args.limit]

        conds = args.conditions.split(",")
        n_judgments = len(rows) * len(conds) * len(frames)
        logger.info(f"[{cfg_name}] {len(rows)} vignettes × {len(conds)} conditions × {len(frames)} frames = "
                     f"{n_judgments} judgments via {args.model}")

        sem = asyncio.Semaphore(args.concurrency)
        tasks, lookup = [], {}
        for r in rows:
            for cond in conds:
                for frame_name, prompt_template in frames:
                    prompt = prompt_template.format(scenario=r[cond])
                    ckey = f"{r['id']}_{cond}_{frame_name}_{hkey(args.model)[:8]}_{hkey(prompt_template)[:4]}"
                    lookup[ckey] = (r, cond, frame_name)
                    tasks.append(judge_or_cache(cache, args.model, prompt, ckey, sem))

        results: dict[str, dict | None] = {}
        for fut in atqdm.as_completed(tasks, total=len(tasks), desc=cfg_name):
            ckey, judged = await fut
            results[ckey] = judged

        # ── Pass 1: collect raw scores from both frames ──
        raw_items = []   # list of (row, cond, judged_fwd, judged_rev)
        n_fail = 0
        for r in rows:
            for cond in conds:
                fwd_pt = JUDGE_PROMPT_FORWARD
                rev_pt = JUDGE_PROMPT_REVERSE
                ckey_fwd = f"{r['id']}_{cond}_forward_{hkey(args.model)[:8]}_{hkey(fwd_pt)[:4]}"
                ckey_rev = f"{r['id']}_{cond}_reverse_{hkey(args.model)[:8]}_{hkey(rev_pt)[:4]}"
                judged_fwd = results.get(ckey_fwd)
                judged_rev = results.get(ckey_rev)

                if judged_fwd is None or judged_rev is None:
                    n_fail += 1
                    continue
                raw_items.append((r, cond, judged_fwd, judged_rev))

        if n_fail:
            logger.warning(f"[{cfg_name}] {n_fail} missing pairs (failures)")

        # ── Collect per-foundation raw vectors for z-scoring ──
        fwd_vecs: dict[str, list[float]] = defaultdict(list)  # foundation → [scores]
        rev_vecs: dict[str, list[float]] = defaultdict(list)
        w_fwd_vec: list[float] = []
        w_rev_vec: list[float] = []
        for _r, _cond, jf, jr in raw_items:
            for f in FOUNDATIONS:
                fwd_vecs[f].append(float(jf[f]))
                rev_vecs[f].append(float(6 - jr[f]))  # flip to violation scale
            w_fwd_vec.append(float(jf.get("wrongness", 3)))
            w_rev_vec.append(float(6 - jr.get("wrongness", 3)))

        # ── Per-foundation z-score parameters ──
        def _zparams(vals: list[float]) -> tuple[float, float]:
            a = np.array(vals, dtype=float)
            return float(a.mean()), float(max(a.std(), 1e-6))

        zp: dict[str, tuple[float, float, float, float]] = {}  # f → (fwd_mu, fwd_sd, rev_mu, rev_sd)
        for f in FOUNDATIONS:
            fm, fs = _zparams(fwd_vecs[f])
            rm, rs = _zparams(rev_vecs[f])
            zp[f] = (fm, fs, rm, rs)
        wf_mu, wf_sd = _zparams(w_fwd_vec)
        wr_mu, wr_sd = _zparams(w_rev_vec)

        # Log frame consistency (before z-scoring)
        fwd_flat = [v for f in FOUNDATIONS for v in fwd_vecs[f]]
        rev_flat = [v for f in FOUNDATIONS for v in rev_vecs[f]]
        if fwd_flat and rev_flat:
            cons_r, _ = sp_stats.pearsonr(fwd_flat, rev_flat)
            logger.info(f"[{cfg_name}] Frame consistency (pearson r between fwd and 6-rev): {cons_r:+.3f}")

        # ── Pass 2: z-score, average, map back to Likert 1-5 ──
        records = []
        for idx, (r, cond, jf, jr) in enumerate(raw_items):
            rec = {
                "id": r["id"],
                "set": r.get("set", cfg_name),
                "condition": cond,
                "foundation_coarse": r["foundation_coarse"],
                "scenario": r[cond],
            }

            llm_scores = {}
            for f in FOUNDATIONS:
                fm, fs, rm, rs = zp[f]
                z_fwd = (fwd_vecs[f][idx] - fm) / fs
                z_rev = (rev_vecs[f][idx] - rm) / rs
                avg_z = (z_fwd + z_rev) / 2.0
                # Map back to Likert scale using pooled mean/std
                pooled_mu = (fm + rm) / 2.0
                pooled_sd = (fs + rs) / 2.0
                llm_v = float(np.clip(avg_z * pooled_sd + pooled_mu, 1.0, 5.0))
                llm_scores[f] = llm_v
                rec[f"llm_{f}"] = round(llm_v, 3)

            # Wrongness: same z-score treatment
            z_wf = (w_fwd_vec[idx] - wf_mu) / wf_sd
            z_wr = (w_rev_vec[idx] - wr_mu) / wr_sd
            avg_wz = (z_wf + z_wr) / 2.0
            pooled_wmu = (wf_mu + wr_mu) / 2.0
            pooled_wsd = (wf_sd + wr_sd) / 2.0
            rec["llm_wrongness"] = round(float(np.clip(avg_wz * pooled_wsd + pooled_wmu, 1.0, 5.0)), 3)

            rec["reason_fwd"] = jf.get("reason", "")
            rec["reason_rev"] = jr.get("reason", "")

            for f in FOUNDATIONS:
                human_col = _HUMAN_COL[f]
                rec[f"human_{f}"] = parse_human_pct(r.get(human_col))
            rec["human_wrongness"] = r.get("wrong")

            rec["llm_dominant"] = max(llm_scores, key=llm_scores.get)
            rec["dominant_match"] = rec["llm_dominant"] == r["foundation_coarse"] or (
                rec["llm_dominant"] == "SocialNorms" and r["foundation_coarse"] == "Social Norms"
            )
            records.append(rec)

        all_records[cfg_name] = records

    # ── Calibration ──
    cal_out = ROOT / "data" / "calibration.json"
    cal_results = {}
    
    classic_records = all_records.get("classic", [])
    has_human = any(rec.get("human_Care") is not None for rec in classic_records)

    if classic_records and has_human:
        print("\n" + "=" * 60)
        print("CALIBRATION: LLM Likert vs Human Rater % (classic set)")
        print("=" * 60)

        cal_rows = []
        for f in FOUNDATIONS:
            llm_vals = [rec[f"llm_{f}"] for rec in classic_records]
            human_vals = [rec[f"human_{f}"] for rec in classic_records]
            cal = calibrate(llm_vals, human_vals)
            cal_results[f] = cal
            cal_rows.append({
                "foundation": f,
                "n": cal["n"],
                "spearman_r": f"{cal['spearman_r']:+.3f}",
                "pearson_r": f"{cal['pearson_r']:+.3f}",
                "slope": f"{cal['slope']:.2f}",
                "intercept": f"{cal['intercept']:.2f}",
                "mae": f"{cal['mae']:.1f}%",
            })

        print("\nPer-foundation correlation (LLM Likert 1-5 vs Human %):")
        print(tabulate(cal_rows, headers="keys", tablefmt="pipe"))

        llm_w = [rec["llm_wrongness"] for rec in classic_records if rec.get("llm_wrongness") is not None]
        human_w = [rec["human_wrongness"] for rec in classic_records if rec.get("human_wrongness") is not None]
        if llm_w and human_w and len(llm_w) == len(human_w):
            wcal = calibrate(llm_w, human_w)
            print(f"\nWrongness calibration: spearman_r={wcal['spearman_r']:+.3f}, "
                  f"pearson_r={wcal['pearson_r']:+.3f}, MAE={wcal['mae']:.2f}")
            cal_results["wrongness"] = wcal

        n_dom = sum(1 for rec in classic_records if rec.get("dominant_match"))
        n_total = len(classic_records)
        print(f"\nDominant-foundation accuracy (argmax): {n_dom}/{n_total} = "
              f"{100*n_dom/n_total:.1f}%")

        conf: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
        for rec in classic_records:
            conf[rec["foundation_coarse"]][rec["llm_dominant"]] += 1
        all_founds = sorted(set(
            list(conf.keys()) + [f for d in conf.values() for f in d.keys()]
        ))
        print("\nConfusion (rows=human, cols=LLM dominant):")
        cm = []
        for f in all_founds:
            row = {"human": f}
            for g in all_founds:
                row[g] = conf[f].get(g, 0)
            cm.append(row)
        print(tabulate(cm, headers="keys", tablefmt="pipe"))

        flagged = []
        for rec in classic_records:
            for f in FOUNDATIONS:
                llm_v = rec[f"llm_{f}"]
                human_v = rec[f"human_{f}"]
                if human_v is None:
                    continue
                # Flag if LLM says strongly relevant (≥4) but human says <10%,
                # or LLM says not relevant (≤2) but human says ≥50%
                if (llm_v >= 4 and human_v < 10) or (llm_v <= 2 and human_v >= 50):
                    flagged.append({
                        "id": rec["id"],
                        "foundation": f,
                        "llm": llm_v,
                        "human": f"{human_v:.0f}%",
                        "scenario": rec["scenario"][:90],
                    })

        print(f"\n{len(flagged)} sharp disagreements (LLM≥4 & human<10%, or LLM≤2 & human≥50%):")
        for fl in flagged[:10]:
            print(f"  {fl['foundation']:12s} llm={fl['llm']} human={fl['human']:>4s}  {fl['scenario']}")

        cal_out.write_text(json.dumps({
            "model": args.model,
            "foundations": cal_results,
            "dominant_accuracy": n_dom / n_total if n_total else 0,
            "n_vignettes": n_total,
        }, indent=2))
        logger.info(f"wrote calibration to {cal_out}")
    elif cal_out.exists():
        cal_results = json.loads(cal_out.read_text()).get("foundations", {})
        logger.info(f"Loaded calibration from {cal_out}")

    # ── Apply Calibration & Write Output ──
    for cfg_name, records in all_records.items():
        if not records:
            continue
        if cfg_name != "classic" and cal_results:
            logger.warning(f"[{cfg_name}] Calibration was fitted on classic set only — "
                           f"calibrated values for '{cfg_name}' are extrapolated")
            
        for rec in records:
            for f in FOUNDATIONS:
                llm_v = rec.get(f"llm_{f}")
                cal = cal_results.get(f)
                if llm_v is not None and cal and not np.isnan(cal.get("slope", float("nan"))):
                    cal_v = cal["slope"] * llm_v + cal["intercept"]
                    rec[f"calibrated_{f}"] = round(max(0.0, min(100.0, float(cal_v))), 1)
            
            w_v = rec.get("llm_wrongness")
            w_cal = cal_results.get("wrongness")
            if w_v is not None and w_cal and not np.isnan(w_cal.get("slope", float("nan"))):
                cal_w = w_cal["slope"] * w_v + w_cal["intercept"]
                rec["calibrated_wrongness"] = round(max(1.0, min(5.0, float(cal_w))), 2)

        out = out_path(cfg_name if cfg_name != "classic" else "")
        with out.open("w") as fh:
            for rec in records:
                fh.write(json.dumps(rec) + "\n")
        logger.info(f"[{cfg_name}] wrote {len(records)} records (with calibrated labels) to {out}")

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for cfg_name, records in all_records.items():
        if not records:
            continue
        n = len(records)
        n_dom = sum(1 for r in records if r.get("dominant_match"))
        mean_w = np.mean([r["llm_wrongness"] for r in records if r.get("llm_wrongness") is not None])
        print(f"  {cfg_name:8s}: {n:3d} rows, dominant-foundation match={n_dom}/{n} ({100*n_dom/n:.0f}%), "
              f"mean wrongness={mean_w:.2f}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="x-ai/grok-4-fast")
    ap.add_argument("--name", default="classic",
                    help="config: 'classic', 'scifi', 'clifford_ai', or 'all'")
    ap.add_argument("--conditions", default="other_violate",
                    help="comma-separated conditions to rate (default: other_violate)")
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
