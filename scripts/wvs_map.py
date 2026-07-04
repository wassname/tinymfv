"""WVS Inglehart-Welzel culture map with LABELED axes: place LLMs among human societies (the
Economist chart), on the two named IW dimensions instead of a blind PCA.

  X = Survival <-> Self-expression        (homosexuality tolerance, interpersonal trust, political action)
  Y = Traditional <-> Secular-Rational    (religion importance + belief, abortion, child autonomy)

Each axis is a small hand-picked battery of GlobalOpinionQA WVS items (tinymfv.iw_axes), every item
oriented to its axis-positive pole by reading the option order. A country's coordinate is the mean
`positiveness` (0-1) over that axis's items from the human WVS distribution; a model's coordinate is
the SAME items administered through the answer-token reader (open models: read_items; logprob-less
API models: read_items_sampled), reduced identically. This is an APPROXIMATE IW (3 themes/axis, not
the canonical 5 -- national pride/authority/materialism are absent from GlobalOpinionQA), not a
verbatim reproduction; the caveat is printed on the figure.

  uv run python scripts/wvs_map.py --local-model Qwen/Qwen3-0.6B \
      --api-models meta-llama/llama-3.1-8b-instruct openai/gpt-4o-mini
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import re
from pathlib import Path

import dotenv
import numpy as np
import torch
from loguru import logger

dotenv.load_dotenv()
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from tinymfv import maps
from tinymfv.zones import zones_for, zone_of
from tinymfv.instrument import Instrument, InstrItem
from tinymfv.read import read_items, resolve_answer_ids
from tinymfv.read_api import read_items_sampled
from tinymfv.iw_axes import AXIS_ITEMS, X_AXIS, Y_AXIS, SKIP, resolve_items, positiveness

# option labels are single digits 0..n-1 -- single-token (unlike '10' on the justifiable scale) and
# the format the answer-token reader is tuned for (a bare digit, not a letter the model ignores in
# favour of the option word).
DIGITS = "0123456789"


def load_wvs_all() -> list[dict]:
    """Every WVS question with its substantive options (DK/refusal/Missing/INAP dropped) and each
    zone-mapped country's distribution renormalized over those options."""
    ds = load_dataset("Anthropic/llm_global_opinions", split="train")
    out = []
    for r in ds:
        if r["source"] != "WVS" or not r["question"]:
            continue
        opts = ast.literal_eval(r["options"]) if isinstance(r["options"], str) else r["options"]
        keep = [i for i, o in enumerate(opts) if not SKIP.search(o)]
        if len(keep) < 2:
            continue
        sel = ast.literal_eval(re.search(r"\{.*\}", r["selections"], re.S).group(0))
        dist = {}
        for c, ps in sel.items():
            if not zone_of(c):
                continue
            v = np.array([ps[i] for i in keep], float)
            if v.sum() > 0:
                dist[c] = v / v.sum()
        out.append({"q": r["question"], "opts": [opts[i] for i in keep], "dist": dist})
    return out


def human_axis_scores(resolved: dict[str, list[dict]]) -> tuple[list[str], np.ndarray]:
    """Per-country (X, Y). A country is kept if it covers at least half of each axis's items; its
    axis value is the mean positiveness over the items it does cover."""
    countries = sorted({c for items in resolved.values() for it in items for c in it["rec"]["dist"]})
    rows, keep = [], []
    for c in countries:
        xy, ok = [], True
        for axis in (X_AXIS, Y_AXIS):
            vals = [positiveness(it["rec"]["dist"][c], it["pole_idx"], it["n"])
                    for it in resolved[axis] if c in it["rec"]["dist"]]
            if len(vals) < (len(resolved[axis]) + 1) // 2:
                ok = False
                break
            xy.append(float(np.mean(vals)))
        if ok:
            keep.append(c)
            rows.append(xy)
    return keep, np.array(rows)


def build_instruments(resolved: dict[str, list[dict]]) -> tuple[list[Instrument], dict[str, dict]]:
    """One nominal Instrument per distinct option-count (answer_space = single letters), covering the
    union of both axes' items. Returns the instruments + a {suffix: {pole_idx, n, axis}} index."""
    items_by_n: dict[int, list[InstrItem]] = {}
    meta: dict[str, dict] = {}
    seen: set[str] = set()
    for axis, items in resolved.items():
        for it in items:
            s = it["suffix"]
            meta[s] = {"pole_idx": it["pole_idx"], "n": it["n"], "axis": axis}
            if s in seen:
                continue
            seen.add(s)
            n, opts = it["n"], it["rec"]["opts"]
            legend = "; ".join(f"{DIGITS[k]}) {o}" for k, o in enumerate(opts))
            task = f"Answer options: {legend}. Respond with only the number."
            items_by_n.setdefault(n, []).append(
                InstrItem(id=s, prompt=it["rec"]["q"], dimension="iw", sign=1,
                          frame="forward", meta={"task": task}))
    instrs = [Instrument(name=f"wvs_iw_n{n}", construct="opinion", kind="nominal",
                         answer_space=list(DIGITS[:n]), dimensions=["iw"], items=its,
                         prefill="(", display="WVS-IW")
              for n, its in sorted(items_by_n.items())]
    return instrs, meta


def model_axis_scores(vecs: dict[str, np.ndarray], meta: dict[str, dict],
                      resolved: dict[str, list[dict]]) -> tuple[float, float]:
    """(X, Y) for one model from its per-item p vectors (suffix -> p over options)."""
    xy = []
    for axis in (X_AXIS, Y_AXIS):
        vals = [positiveness(vecs[it["suffix"]], it["pole_idx"], it["n"]) for it in resolved[axis]]
        xy.append(float(np.mean(vals)))
    return xy[0], xy[1]


def read_model(rows: list[dict], meta: dict[str, dict]) -> dict[str, np.ndarray]:
    """rows from read_items / read_items_sampled -> {suffix: p over that item's options}. NaN p (read
    collapse) fails loud later via positiveness rather than being imputed."""
    return {r["id"]: np.asarray(r["p"], float)[: meta[r["id"]]["n"]] for r in rows}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--local-model", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--api-models", nargs="*", default=[])
    ap.add_argument("--api-samples", type=int, default=20)
    ap.add_argument("--max-think-tokens", type=int, default=64)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", default="/tmp/claude-1000/wvs_map_iw.png")
    ap.add_argument("--cache", default="/tmp/claude-1000/wvs_iw_vectors.json",
                    help="cache model per-item p vectors so re-styling skips the API/model calls")
    args = ap.parse_args()

    recs = load_wvs_all()
    resolved = resolve_items(recs)
    for axis, items in resolved.items():
        logger.info(f"{axis}: " + ", ".join(f"{it['suffix']}[n{it['n']},pole{it['pole_idx']}]"
                                             for it in items))
    countries, P = human_axis_scores(resolved)
    logger.info(f"{len(recs)} WVS questions -> {len(countries)} countries on 2 IW axes")

    instrs, meta = build_instruments(resolved)
    # DETERMINISTIC key over the item set: Python's builtin hash() is salted per process
    # (PYTHONHASHSEED), so it changes every run and the cache never hits across processes -- costing a
    # fresh API call each time. hashlib is stable.
    sig = hashlib.md5(repr(sorted((s, m["n"]) for s, m in meta.items())).encode()).hexdigest()[:8]
    cpath = Path(args.cache)
    cpath.parent.mkdir(parents=True, exist_ok=True)
    cache = json.loads(cpath.read_text()).get(sig, {}) if cpath.exists() else {}
    vecs: dict[str, dict[str, np.ndarray]] = {k: {s: np.array(p) for s, p in v.items()}
                                              for k, v in cache.items()}

    def save_cache() -> None:
        """Persist after EACH model so a killed run (session teardown) keeps finished models -- the
        write-once-at-end version silently discarded every sampled model when interrupted."""
        allc = json.loads(cpath.read_text()) if cpath.exists() else {}
        allc[sig] = {k: {s: p.tolist() for s, p in v.items()} for k, v in vecs.items()}
        cpath.write_text(json.dumps(allc))

    if args.local_model:
        key = args.local_model.split("/")[-1] + " (lp)"
        if key not in vecs:
            tok = AutoTokenizer.from_pretrained(args.local_model)
            if tok.pad_token is None:
                tok.pad_token = tok.eos_token
            tok.padding_side = "left"
            lm = AutoModelForCausalLM.from_pretrained(args.local_model, dtype=torch.bfloat16).to(args.device).eval()
            rows = []
            for k, instr in enumerate(instrs):
                rows += read_items(lm, tok, instr, instr.items,
                                   resolve_answer_ids(tok, instr.answer_space),
                                   max_think_tokens=args.max_think_tokens, batch_size=16,
                                   verbose_first=(k == 0))
            vecs[key] = read_model(rows, meta)
            save_cache()
    for m in args.api_models:
        key = m.split("/")[-1] + " (sampled)"
        if key not in vecs:
            rows = []
            for k, instr in enumerate(instrs):
                rows += read_items_sampled(m, instr, instr.items, n_samples=args.api_samples,
                                           verbose_first=(k == 0))
            vecs[key] = read_model(rows, meta)
            save_cache()                    # persist this model before starting the next (kill-safe)
            logger.info(f"cached {key}")

    models = {k: model_axis_scores(v, meta, resolved) for k, v in vecs.items()}

    # Render through the SHARED value-map renderer (same one the instrument value maps use): pole
    # signposts through the human median, 4 auto-selected zone hulls, textalloc labels, model stars.
    _, emph = zones_for(countries)
    fig = maps.plot_value_map(
        "WVS Inglehart-Welzel", countries, P,
        ("Survival", "Self-expression", "Traditional", "Secular-Rational"),
        models=models, emphasize=emph,
        title=f"WVS Inglehart-Welzel map: LLMs among {len(countries)} human societies (approximate IW axes)",
        note=("Approximate IW: axes built from GlobalOpinionQA WVS items (3 themes/axis, not the\n"
              "canonical 5; national pride / authority / materialism absent). Not a verbatim WVS factor score."))
    fig.savefig(args.out, dpi=200, bbox_inches="tight")
    logger.info(f"wrote {args.out}")


if __name__ == "__main__":
    main()
