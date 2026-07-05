"""WVS Inglehart-Welzel culture map with LABELED axes: place LLMs among human societies (the
Economist chart), on the two named IW dimensions instead of a blind PCA.

  X = Survival <-> Self-expression        (homosexuality tolerance, interpersonal trust, political action)
  Y = Traditional <-> Secular-Rational    (religion importance + belief, abortion, child autonomy)

Each axis is a small hand-picked battery of GlobalOpinionQA WVS items (tinymfv.iw_axes), every item
oriented to its axis-positive pole by reading the option order. A country's coordinate is the mean
`positiveness` (0-1) over that axis's items from the human WVS choice frequencies. A model's
coordinate is the SAME items administered as a dense Likert readout (read_items_rated: rate every
option 1-5 as JSON, binary options order-permuted to cancel positional bias, N samples, normalized
mean rating -> distribution), reduced identically; open models can instead use the answer-token
logprob reader (read_items). Each model also carries a bootstrap 95% CI (over items + samples) drawn
as error bars, so mushy / uncertain placements read as uncertain rather than confident dots. NB the
model coordinate is a rating-derived pseudo-distribution while the human one is a real choice
frequency -- a documented proxy. This is an APPROXIMATE IW (3 themes/axis, not the canonical 5 --
national pride/authority/materialism are absent from GlobalOpinionQA), not a verbatim reproduction.

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
from tinymfv.read_api import read_items_rated
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
    """rows from read_items -> {suffix: p over that item's options}. NaN p (read collapse) fails loud
    later via positiveness rather than being imputed."""
    return {r["id"]: np.asarray(r["p"], float)[: meta[r["id"]]["n"]] for r in rows}


def model_coord_ci(psamples: dict[str, np.ndarray], resolved: dict[str, list[dict]],
                   rng: np.random.Generator, B: int = 500) -> tuple[float, float, float, float]:
    """(x, y, x_se, y_se). Point estimate = mean positiveness over each axis's items on the mean-over-
    samples p. The SE is a bootstrap over BOTH noise sources: resample the axis's items with
    replacement (item-set noise, only ~3-7 items/axis) and, per item, draw one of its N rating samples
    (readout noise). std over B replicates -> the CI drawn as error bars on the map."""
    def axis_coords(getp) -> list[float]:
        return [float(np.mean([positiveness(getp(it), it["pole_idx"], it["n"]) for it in resolved[axis]]))
                for axis in (X_AXIS, Y_AXIS)]
    x, y = axis_coords(lambda it: psamples[it["suffix"]].mean(0))
    bx, by = [], []
    for _ in range(B):
        xy = []
        for axis in (X_AXIS, Y_AXIS):
            items = resolved[axis]
            idx = rng.integers(0, len(items), len(items))
            vals = []
            for j in idx:
                it = items[j]
                ps = psamples[it["suffix"]]
                vals.append(positiveness(ps[int(rng.integers(0, len(ps)))], it["pole_idx"], it["n"]))
            xy.append(float(np.mean(vals)))
        bx.append(xy[0]); by.append(xy[1])
    return x, y, float(np.std(bx)), float(np.std(by))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--local-model", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--api-models", nargs="*", default=[])
    ap.add_argument("--api-samples", type=int, default=12,
                    help="rating samples per item (each dense: every option rated), binary items order-balanced")
    ap.add_argument("--api-max-tokens", type=int, default=1024,
                    help="output budget per rating call; large enough that a reasoning model finishes the JSON")
    ap.add_argument("--max-think-tokens", type=int, default=64)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", default="/tmp/claude-1000/wvs_map_iw.png")
    ap.add_argument("--cache", default="/tmp/claude-1000/wvs_iw_rated.json",
                    help="cache model (x,y[,x_se,y_se]) coords so re-styling skips the API/model calls")
    ap.add_argument("--responses", default="/tmp/claude-1000/wvs_iw_rated_responses.jsonl",
                    help="append every raw model response here (audit trail; API calls cost money)")
    args = ap.parse_args()

    recs = load_wvs_all()
    resolved = resolve_items(recs)
    for axis, items in resolved.items():
        logger.info(f"{axis}: " + ", ".join(f"{it['suffix']}[n{it['n']},pole{it['pole_idx']}]"
                                             for it in items))
    countries, P = human_axis_scores(resolved)
    logger.info(f"{len(recs)} WVS questions -> {len(countries)} countries on 2 IW axes")

    # One rated item per distinct WVS question (canonical option order), administered to every API model.
    rated_items, seen = [], set()
    for axis in (X_AXIS, Y_AXIS):
        for it in resolved[axis]:
            if it["suffix"] in seen:
                continue
            seen.add(it["suffix"])
            rated_items.append({"id": it["suffix"], "question": it["rec"]["q"],
                                "options": it["rec"]["opts"], "n": it["n"]})

    # DETERMINISTIC cache key over the item set: Python's builtin hash() is salted per process
    # (PYTHONHASHSEED), so it changes every run and the cache never hits -- costing a fresh API call
    # each time. hashlib is stable. cache value = (x, y[, x_se, y_se]) per model.
    sig = hashlib.md5(repr(sorted((it["id"], it["n"]) for it in rated_items)).encode()).hexdigest()[:8]
    cpath = Path(args.cache)
    cpath.parent.mkdir(parents=True, exist_ok=True)
    cache = json.loads(cpath.read_text()).get(sig, {}) if cpath.exists() else {}
    models: dict[str, tuple] = {k: tuple(v) for k, v in cache.items()}

    def save_cache() -> None:
        """Persist after EACH model so a killed run keeps every finished model (kill-safe)."""
        allc = json.loads(cpath.read_text()) if cpath.exists() else {}
        allc[sig] = {k: list(v) for k, v in models.items()}
        cpath.write_text(json.dumps(allc))

    rpath = Path(args.responses)
    rpath.parent.mkdir(parents=True, exist_ok=True)

    def save_responses(key: str, rows: list[dict]) -> None:
        """Append every raw rated response (audit trail -- these API calls cost money)."""
        with rpath.open("a") as fh:
            for r in rows:
                fh.write(json.dumps({"model": key, "sig": sig, "item": r["id"],
                                     "prompt": r.get("prompt"), "texts": r.get("texts"),
                                     "p": np.asarray(r["p"]).tolist(), "pmass": r["pmass_allowed"]}) + "\n")

    rng = np.random.default_rng(0)                       # deterministic bootstrap

    # Open local model: answer-token logprob reader (single-choice categorical) -> (x, y), no CI.
    if args.local_model:
        key = args.local_model.split("/")[-1] + " (lp)"
        if key not in models:
            instrs, meta = build_instruments(resolved)
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
            models[key] = model_axis_scores(read_model(rows, meta), meta, resolved)
            save_cache()

    # API models: dense rated readout -> (x, y, x_se, y_se) with bootstrap CI.
    for m in args.api_models:
        key = m.split("/")[-1] + " (rated)"
        if key in models:
            continue
        try:                                # one flaky provider / network blip must not abort the panel
            rows = read_items_rated(m, rated_items, n_samples=args.api_samples,
                                    max_tokens=args.api_max_tokens, verbose_first=True)
        except Exception as e:
            logger.warning(f"{key}: read failed ({type(e).__name__}: {e}) -> skipping (not cached)")
            continue
        save_responses(key, rows)           # raw answers first (before reducing)
        psamples = {r["id"]: np.array(r["p_samples"]) for r in rows}
        collapsed = [k for k, v in psamples.items() if v.size == 0]
        if collapsed:                       # a refusing / off-format model: skip, keep the panel going
            logger.warning(f"{key}: parse collapse on {collapsed} -> skipping (not cached)")
            continue
        models[key] = model_coord_ci(psamples, resolved, rng)
        save_cache()                        # persist this model before the next (kill-safe)
        x, y, xs, ys = models[key]
        logger.info(f"cached {key}: ({x:.2f}, {y:.2f}) +-({1.96*xs:.02f}, {1.96*ys:.02f}) 95% CI")

    # Uncertainty as a TABLE, not whiskers on the map (the CI crosses overlap into noise with a dozen+
    # models). Sorted widest-first so the mushy models are obvious. The CI is bootstrap over items +
    # samples; for the wide ones it's item-disagreement (the model rates different WVS items
    # inconsistently on an axis), which more samples will NOT shrink -- see the readme/journal note.
    ci_rows = [(k, v[0], v[1], 1.96 * v[2], 1.96 * v[3]) for k, v in models.items() if len(v) > 3]
    if ci_rows:
        from tabulate import tabulate
        ci_rows.sort(key=lambda r: -(r[3] + r[4]))
        table = tabulate(ci_rows, headers=["model", "x self-expr", "y secular", "x 95%CI", "y 95%CI"],
                         tablefmt="pipe", floatfmt="+.2f")
        Path(args.out).with_name("wvs_model_ci.md").write_text(table + "\n")
        logger.info("model coords + 95% CI (widest first):\n" + table)

    # Render through the SHARED value-map renderer (same one the instrument value maps use): pole
    # signposts through the human median, 4 auto-selected zone hulls, textalloc labels, model stars.
    _, emph = zones_for(countries)
    # Title + caption live in the README (nicer voice, editable), not baked into the figure.
    # Drop the " (rated)" readout tag from the on-map labels (the cache/CI-table keep it) -- the map is
    # crowded and every model here is rated, so the tag adds nothing.
    plot_models = {k.replace(" (rated)", ""): v for k, v in models.items()}
    fig = maps.plot_value_map(
        "WVS Inglehart-Welzel", countries, P,
        ("Survival", "Self-expression", "Traditional", "Secular-Rational"),
        models=plot_models, emphasize=emph)
    fig.savefig(args.out, dpi=200, bbox_inches="tight")
    logger.info(f"wrote {args.out}")


if __name__ == "__main__":
    main()
