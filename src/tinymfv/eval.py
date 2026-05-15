"""High-level entrypoint: forced-choice 7-way moral-foundation probe.

For each vignette+condition, ask the model to pick which foundation is violated
(or "social" = morally fine, just unusual). Returns the per-row 7-vec
distribution plus aggregates against the label distribution.

Labels:
- `human_*` columns. On `classic` these are direct Clifford et al. (2015) %
    distributions. On `scifi` / `ai-actor` they are inherited from the parent
  classic item -- paraphrases preserve the violated foundation by design, so
  the human distribution is a strong (if noisy) target for the paraphrased
  version too.
- `ai_*` columns are a per-set grok-4-fast judge on the paraphrased text;
  available for cross-source agreement analysis, not used as the eval target.

Headline metrics:
- top1_acc:  argmax model == argmax label, fraction of rows.
- mean_nll:  mean soft cross-entropy -sum_f p_human[f] log p_model[f], in nats.
- mean_nll_T: same metric after fitting one temperature on the scored set.
- mean_js:   legacy mean Jensen-Shannon divergence between model and label dist
             (in nats, max = ln 2 ≈ 0.693).
- gap[f]:    per-foundation perspective gap = mean p[f] (other_violate)
             - mean p[f] (self_violate). Detects perspective bias.
"""
from __future__ import annotations
import math
import time
from typing import Any

import numpy as np
import pandas as pd
import torch
from loguru import logger
from tqdm.auto import tqdm

from .data import load_vignettes, ConfigName
from .guided import (
    guided_rollout_forced_choice,
    _DEFAULT_FORCED_FOUNDATIONS,
)

CONDITIONS = ("other_violate", "self_violate")

# Probe word -> dataset coarse label.
_PROBE_TO_COARSE: dict[str, str] = {
    "care": "Care", "fairness": "Fairness", "loyalty": "Loyalty",
    "authority": "Authority", "sanctity": "Sanctity", "liberty": "Liberty",
    "social": "SocialNorms",
}
_COARSE_TO_PROBE: dict[str, str] = {v: k for k, v in _PROBE_TO_COARSE.items()}
# Some Clifford rows use "Social Norms" with a space; normalise.
_COARSE_NORM = {"Social Norms": "SocialNorms"}


def _label_dist(row: dict, foundations: list[str]) -> np.ndarray | None:
    """Build the 7-vec human label distribution for a vignette.

    Order matches `foundations` (probe-word order: care, fairness, ..., social).
    Reads `human_*` -- on `classic` these are direct Clifford et al. (2015) %
    distributions; on `scifi` and `ai-actor` they are inherited from the
    parent classic item (paraphrases preserve the violated foundation by
    design). Returns None if any column missing or row sums to 0.
    """
    coarse = [_PROBE_TO_COARSE[f] for f in foundations]
    vals = [row.get(f"human_{c}") for c in coarse]
    if any(v is None for v in vals):
        return None
    arr = np.array(vals, dtype=float)
    s = float(arr.sum())
    if s <= 0:
        return None
    return arr / s


def _js_divergence(p: np.ndarray, q: np.ndarray) -> float:
    """Jensen-Shannon divergence in nats. Symmetric, bounded by ln 2."""
    p = p + 1e-12; q = q + 1e-12
    p = p / p.sum(); q = q / q.sum()
    m = 0.5 * (p + q)
    kl_pm = float((p * np.log(p / m)).sum())
    kl_qm = float((q * np.log(q / m)).sum())
    return 0.5 * kl_pm + 0.5 * kl_qm


def _soft_nll(p_human: np.ndarray, p_model: np.ndarray) -> float:
    """Soft cross-entropy: -sum_f p_human[f] log p_model[f], in nats.

    Standard quantity for matching a predicted distribution to a soft-labelled
    target. Unbounded; sensitive to confident-wrong rows.
    """
    p_model = np.clip(p_model, 1e-12, 1.0)
    return float(-(p_human * np.log(p_model)).sum())


def _fit_temperature(
    scores: np.ndarray,           # [N, K] pre-softmax scores (sum of fwd+rev logprobs / 2)
    p_human: np.ndarray,          # [N, K] target distributions, rows sum to 1
    grid: np.ndarray | None = None,
) -> float:
    """Fit one temperature `T>0` minimizing mean soft-NLL of `p_human` under
    softmax(scores/T). Coarse-then-fine 1D search; cheap, deterministic, no
    autograd. Returns T*.

    Forced-choice probes are typically far overconfident (T*>>1); we search
    on a log-grid `[0.1, 50]` then refine around the minimum.
    """
    if grid is None:
        grid = np.logspace(np.log10(0.1), np.log10(50.0), 41)

    def mean_nll_at(T: float) -> float:
        s = scores / T
        s = s - s.max(axis=1, keepdims=True)
        logp = s - np.log(np.exp(s).sum(axis=1, keepdims=True))
        return float(-(p_human * logp).sum(axis=1).mean())

    coarse = np.array([mean_nll_at(float(T)) for T in grid])
    i = int(np.argmin(coarse))
    lo = grid[max(0, i - 1)]; hi = grid[min(len(grid) - 1, i + 1)]
    fine = np.linspace(lo, hi, 41)
    fine_vals = np.array([mean_nll_at(float(T)) for T in fine])
    return float(fine[int(np.argmin(fine_vals))])


def evaluate(
    model,
    tokenizer,
    name: ConfigName = "classic",
    vignettes: list[dict] | None = None,
    *,
    conditions: tuple[str, ...] = CONDITIONS,
    max_think_tokens: int = 256,
    batch_size: int = 8,
    device: str | None = None,
    return_per_row: bool = False,
) -> dict[str, Any]:
    """Run forced-choice 7-way probe per (vignette, condition).

    Args:
        model, tokenizer: HuggingFace causal LM + matching tokenizer with chat template.
        name: dataset config (`classic` / `scifi` / `ai-actor`).
        vignettes: optional pre-loaded list (overrides `name`).
        conditions: which condition strings to score. Default = both.
        max_think_tokens: think budget per (row, frame). Two frames per row.
        batch_size: rows per forced-choice call (KV cache = batch * 2 * max_think_tokens).
        return_per_row: if True, include the per-row 7-vec p in the result.

    Returns:
        Dict with `table`, `profile`, `mean_js`, `mean_nll`, `mean_nll_T`,
        `median_nll_T`, `T`, `top1_acc`, and `info`. If `return_per_row=True`,
        also includes `per_row` with the row-level distributions and scores.
    """
    if vignettes is None:
        vignettes = load_vignettes(name)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    if device is None:
        device = next(model.parameters()).device.type

    foundations = list(_DEFAULT_FORCED_FOUNDATIONS)

    t0 = time.time()
    per_row: list[dict] = []
    total_calls = len(vignettes) * len(conditions)
    with tqdm(total=total_calls, desc=f"forced-choice {name}", mininterval=60, maxinterval=120) as pbar:
        for cond in conditions:
            for i in range(0, len(vignettes), batch_size):
                chunk = vignettes[i: i + batch_size]
                user_prompts = [r[cond] for r in chunk]
                results = guided_rollout_forced_choice(
                    model, tokenizer, user_prompts,
                    foundations=foundations,
                    max_think_tokens=max_think_tokens,
                )
                for src, res in zip(chunk, results):
                    p_vec = np.array([res.p[f] for f in foundations], dtype=float)
                    score_vec = np.array([res.score[f] for f in foundations], dtype=float)
                    label = _label_dist(src, foundations)
                    coarse = _COARSE_NORM.get(src["foundation_coarse"], src["foundation_coarse"])
                    per_row.append({
                        "id": src["id"],
                        "condition": cond,
                        "foundation_coarse": coarse,
                        "p": p_vec,
                        "score": score_vec,  # pre-softmax averaged logprobs, for temperature fit
                        "label": label,  # may be None on unlabeled rows
                        "top1": res.top1,
                        "margin": res.margin,
                        "nll_prompt": res.nll_prompt,
                    })
                pbar.update(len(chunk))

    elapsed = time.time() - t0
    n_rows = len(per_row)
    n_labeled = sum(1 for r in per_row if r["label"] is not None)
    logger.info(
        f"{name}: {n_rows} rows in {elapsed:.1f}s ({n_rows/elapsed:.1f} rows/s); "
        f"{n_labeled}/{n_rows} have label dist"
    )

    # === per-foundation aggregates ===
    rows = []
    for fi, fname in enumerate(foundations):
        coarse = _PROBE_TO_COARSE[fname]
        ov = [r["p"][fi] for r in per_row if r["condition"] == "other_violate"]
        sv = [r["p"][fi] for r in per_row if r["condition"] == "self_violate"]
        # Cross-vignette Pearson with label[fi] on labeled rows.
        labeled = [r for r in per_row if r["label"] is not None and r["condition"] == "other_violate"]
        if len(labeled) >= 5:
            x = np.array([r["p"][fi] for r in labeled])
            y = np.array([r["label"][fi] for r in labeled])
            if x.std() > 0 and y.std() > 0:
                pr = float(np.corrcoef(x, y)[0, 1])
            else:
                pr = float("nan")
        else:
            pr = float("nan")
        rows.append({
            "foundation": coarse,
            "n": len(ov),
            "mean_p_other": float(np.mean(ov)) if ov else float("nan"),
            "mean_p_self": float(np.mean(sv)) if sv else float("nan"),
            "gap": (float(np.mean(ov)) - float(np.mean(sv))) if ov and sv else float("nan"),
            "pearson_label": pr,
        })
    table = pd.DataFrame(rows)

    # === headline scalars (need labels) ===
    labeled_rows = [r for r in per_row if r["label"] is not None]
    if labeled_rows:
        js_vals = np.array([_js_divergence(r["p"], r["label"]) for r in labeled_rows])
        mean_js = float(js_vals.mean())
        median_js = float(np.median(js_vals))
        top1_acc = float(np.mean([
            np.argmax(r["p"]) == np.argmax(r["label"]) for r in labeled_rows
        ]))

        # Soft NLL at T=1 (raw, overconfident-dominated).
        nll_raw = np.array([_soft_nll(r["label"], r["p"]) for r in labeled_rows])
        mean_nll = float(nll_raw.mean())
        median_nll = float(np.median(nll_raw))

        # Temperature scaling: fit one T* on `classic` (or whatever set we're on)
        # to minimise mean soft NLL. T cancels for steering deltas; only matters
        # for absolute model-vs-human comparison.
        scores = np.stack([r["score"] for r in labeled_rows])
        humans = np.stack([r["label"] for r in labeled_rows])
        T = _fit_temperature(scores, humans)
        s_scaled = scores / T
        s_scaled = s_scaled - s_scaled.max(axis=1, keepdims=True)
        p_scaled = np.exp(s_scaled)
        p_scaled = p_scaled / p_scaled.sum(axis=1, keepdims=True)
        nll_T = -(humans * np.log(np.clip(p_scaled, 1e-12, 1.0))).sum(axis=1)
        mean_nll_T = float(nll_T.mean())
        median_nll_T = float(np.median(nll_T))

        # Mean profile across vignettes (model and human), each a 7-vec on the
        # simplex. The natural object to compare moral character across models.
        model_profile = np.stack([r["p"] for r in labeled_rows]).mean(axis=0)
        human_profile = humans.mean(axis=0)
        profile = pd.DataFrame({
            "foundation": [_PROBE_TO_COARSE[f] for f in foundations],
            "human": human_profile,
            "model": model_profile,
            "model_T": p_scaled.mean(axis=0),
        })
    else:
        mean_js = median_js = top1_acc = None
        mean_nll = median_nll = mean_nll_T = median_nll_T = None
        T = None
        profile = None

    info = {
        "name": name,
        "n_rows": n_rows,
        "n_labeled": n_labeled,
        "elapsed_s": elapsed,
        "median_js": median_js,
        "max_js": math.log(2),
        "mean_nll": mean_nll,
        "median_nll": median_nll,
        "median_nll_T": median_nll_T,
        # Mean prompt NLL across rows (nats/token, teacher-forcing on the
        # rendered chat). Free degradation probe; unsteered baseline gives
        # the model's "natural" surprise on prompt text.
        "mean_nll_prompt": float(np.mean([r["nll_prompt"] for r in per_row]))
            if per_row else None,
    }

    out: dict[str, Any] = {
        "table": table,
        "profile": profile,    # 7-row DataFrame: foundation, human, model, model_T
        "mean_js": mean_js,
        "mean_nll": mean_nll,
        "mean_nll_T": mean_nll_T,  # temperature-scaled soft cross-entropy in nats
        "median_nll_T": median_nll_T,
        "T": T,                # fitted temperature (>1 = model is overconfident)
        "top1_acc": top1_acc,
        "info": info,
    }
    if return_per_row:
        out["per_row"] = per_row
    return out

