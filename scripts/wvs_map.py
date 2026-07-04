"""WVS / Inglehart-Welzel style culture map: place LLMs among human societies (the Economist chart).

Runs tinymfv on the WVS subset of Anthropic/llm_global_opinions "like the others": each 4-option WVS
question is one ordinal item; a country x question matrix of expected-score E is ipsative-PCA'd (the
same maps.ipsative_pca the instrument maps use) into 2 axes, with IW zone ellipses. Each model is
administered the SAME questions -- open models via the logprob reader (read_items), logprob-less API
models via the sampling reader (read_items_sampled) -- reduced to an E-vector and projected onto the
country axes as a labelled dot.

  uv run python scripts/wvs_map.py --local-model Qwen/Qwen3-0.6B \
      --api-models meta-llama/llama-3.1-8b-instruct openai/gpt-4o-mini
"""
from __future__ import annotations

import argparse
import ast
import json
import re
from collections import Counter
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
from tinymfv.zones import zones_for, zone_of, ECONOMIST_OUTLIERS
from tinymfv.instrument import Instrument, InstrItem
from tinymfv.read import read_items, resolve_answer_ids
from tinymfv.read_api import read_items_sampled
from tinymfv.readouts import expected_score

SKIP = re.compile(r"don'?t know|no answer|refus|decline|none of|not applicable|^other", re.I)
MODEL_COLORS = ["#c0392b", "#8e44ad", "#16a085", "#d35400", "#2980b9", "#c2185b"]


def load_wvs(n_opts: int) -> list[dict]:
    """WVS questions with exactly `n_opts` substantive options (DK/No-answer dropped), each carrying
    its per-country human distribution renormalized over the substantive options."""
    ds = load_dataset("Anthropic/llm_global_opinions", split="train")
    out = []
    for r in ds:
        if r["source"] != "WVS":
            continue
        opts = ast.literal_eval(r["options"]) if isinstance(r["options"], str) else r["options"]
        keep = [i for i, o in enumerate(opts) if not SKIP.search(o)]
        if len(keep) != n_opts:
            continue
        sel = ast.literal_eval(re.search(r"\{.*\}", r["selections"], re.S).group(0))
        dist = {}
        for c, ps in sel.items():
            v = np.array([ps[i] for i in keep], float)
            if v.sum() > 0:
                dist[c] = v / v.sum()
        out.append({"q": r["question"], "opts": [opts[i] for i in keep], "dist": dist})
    return out


def dense_block(recs: list[dict]) -> tuple[list[str], list[dict]]:
    """Largest complete country x question block (no imputation): start from every zone-mapped
    country x every question, then greedily drop whichever row or column has the most missing cells
    until no cell is missing. WVS coverage is patchy per-question, so a complete block needs this
    trim rather than a fixed coverage threshold."""
    countries = sorted({c for r in recs for c in r["dist"] if zone_of(c)})
    A = np.array([[c in r["dist"] for r in recs] for c in countries])   # C x Q presence
    rows, cols = list(range(len(countries))), list(range(len(recs)))
    while True:
        sub = A[np.ix_(rows, cols)]
        if sub.all():
            break
        frac_r, frac_c = (~sub).mean(1), (~sub).mean(0)   # missing FRACTION per country / per question
        if frac_r.max() >= frac_c.max():
            rows.pop(int(np.argmax(frac_r)))
        else:
            cols.pop(int(np.argmax(frac_c)))
    return [countries[i] for i in rows], [recs[j] for j in cols]


def human_matrix(countries: list[str], block: list[dict], n_opts: int) -> np.ndarray:
    """countries x questions matrix of expected-score E as a 0-1 fraction (E-1)/(n_opts-1)."""
    w = np.arange(1, n_opts + 1)
    M = np.array([[float((block[q]["dist"][c] * w).sum()) for q in range(len(block))] for c in countries])
    return (M - 1) / (n_opts - 1)


def build_instrument(block: list[dict], n_opts: int) -> Instrument:
    """One ordinal Instrument over the WVS questions: answer_space digits 1..n_opts, each item the
    question + its numbered options legend."""
    items = []
    for i, r in enumerate(block):
        legend = "; ".join(f"{k+1}) {o}" for k, o in enumerate(r["opts"]))
        task = f"Answer options: {legend}. Respond with only the number."
        items.append(InstrItem(id=str(i), prompt=r["q"], dimension="wvs", sign=1,
                               frame="forward", meta={"task": task}))
    return Instrument(name="wvs", construct="opinion", kind="ordinal",
                      answer_space=[str(i) for i in range(1, n_opts + 1)],
                      dimensions=["wvs"], items=items, prefill="(", scale_max=n_opts,
                      human_scale_max=n_opts, display="WVS")


def model_vector(rows: list[dict], n: int, n_opts: int) -> np.ndarray:
    """Per-question E as a 0-1 fraction, in item order. NaN if a question's read collapsed."""
    by_id = {int(r["id"]): r for r in rows}
    out = np.full(n, np.nan)
    for i in range(n):
        p = np.asarray(by_id[i]["p"], float)
        if np.isfinite(p).all() and abs(p.sum() - 1) < 1e-6:
            out[i] = (expected_score(p, n_opts) - 1) / (n_opts - 1)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-opts", type=int, default=4)
    ap.add_argument("--local-model", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--api-models", nargs="*", default=[])
    ap.add_argument("--api-samples", type=int, default=20)
    ap.add_argument("--max-think-tokens", type=int, default=64)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", default="/tmp/claude-1000/wvs_map.png")
    ap.add_argument("--cache", default="/tmp/claude-1000/wvs_model_vectors.json",
                    help="cache model E-vectors so re-plotting (new zone style) skips the API calls")
    args = ap.parse_args()

    recs = load_wvs(args.n_opts)
    countries, block = dense_block(recs)
    logger.info(f"{len(recs)} {args.n_opts}-option WVS questions -> dense block "
                f"{len(countries)} countries x {len(block)} questions")
    Hfrac = human_matrix(countries, block, args.n_opts)
    P, Vt, var, mu, Pc = maps.ipsative_pca(Hfrac)
    instr = build_instrument(block, args.n_opts)

    # cache keyed by the exact question block: same block -> reuse E-vectors, only re-projecting +
    # re-drawing (so iterating on the zone style costs no API calls).
    sig = f"{args.n_opts}:{len(block)}:{hash(tuple(r['q'] for r in block)) & 0xffffffff}"
    cpath = Path(args.cache)
    cache = json.loads(cpath.read_text()).get(sig, {}) if cpath.exists() else {}
    vecs: dict[str, np.ndarray] = {k: np.array(v) for k, v in cache.items()}

    if args.local_model:
        key = args.local_model.split("/")[-1] + " (lp)"
        if key not in vecs:
            tok = AutoTokenizer.from_pretrained(args.local_model)
            if tok.pad_token is None:
                tok.pad_token = tok.eos_token
            tok.padding_side = "left"
            lm = AutoModelForCausalLM.from_pretrained(args.local_model, dtype=torch.bfloat16).to(args.device).eval()
            rows = read_items(lm, tok, instr, instr.items, resolve_answer_ids(tok, instr.answer_space),
                              max_think_tokens=args.max_think_tokens, batch_size=16, verbose_first=True)
            vecs[key] = model_vector(rows, len(block), args.n_opts)
    for m in args.api_models:
        key = m.split("/")[-1] + " (sampled)"
        if key not in vecs:
            rows = read_items_sampled(m, instr, instr.items, n_samples=args.api_samples, verbose_first=True)
            vecs[key] = model_vector(rows, len(block), args.n_opts)

    cpath.parent.mkdir(parents=True, exist_ok=True)
    allc = json.loads(cpath.read_text()) if cpath.exists() else {}
    allc[sig] = {k: v.tolist() for k, v in vecs.items()}
    cpath.write_text(json.dumps(allc))
    hmean = Hfrac.mean(0)      # a question the model didn't answer coherently -> the human average there
    models = {k: ((np.where(np.isnan(v), hmean, v) @ Pc) - mu) @ Vt[:2].T for k, v in vecs.items()}

    zones, emph = zones_for(countries)
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.set_facecolor("#faf8f2")
    ax.grid(True, color="#eceadf", lw=0.3, zorder=0)
    maps.draw_zone_regions(ax, P, countries, zones)
    ax.scatter(P[:, 0], P[:, 1], s=22, c=maps.C_HUM, alpha=0.7, edgecolors="white", linewidths=0.4, zorder=3)
    for i, c in enumerate(countries):
        is_e = c in emph
        ax.annotate(c, (P[i, 0], P[i, 1]), fontsize=7.5 if is_e else 6, xytext=(3, 2),
                    textcoords="offset points", color="#111" if is_e else "#666",
                    fontweight="bold" if is_e else "normal", zorder=6)
    for (name, pt), col in zip(models.items(), MODEL_COLORS):
        ax.scatter(*pt, s=120, marker="*", c=col, edgecolors="white", linewidths=1.0, zorder=8)
        ax.annotate(name, pt, xytext=(6, 4), textcoords="offset points", fontsize=9,
                    fontweight="bold", color=col, zorder=9)
    ax.set_xlabel(f"PC1 ({var[0]*100:.0f}% var)")
    ax.set_ylabel(f"PC2 ({var[1]*100:.0f}% var)")
    ax.set_title(f"WVS values map: LLMs among {len(countries)} human societies "
                 f"({len(block)} WVS questions, ipsative PCA)", fontsize=11)
    fig.tight_layout()
    fig.savefig(args.out, dpi=200, bbox_inches="tight")
    logger.info(f"wrote {args.out}")


if __name__ == "__main__":
    main()
