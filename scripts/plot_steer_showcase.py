"""Showcase tinymfv's plotting on a real steering run (the dogfood before publishing the lib).

Consumes a steering-lite `run_allinstr_showcase.py` output dir (one calibrated
activation-steering vector administered across every instrument over a signed
c-sweep) and renders the SAME two figures for every instrument, uniformly:

  - map  : ipsative culture map (PCA), AI base + steer trajectory vs the human cloud.
  - range: per-factor range, AI base dot + +c/-c arrows vs the human society strip.

Ordinal instruments (mfq2/big5/16pf/humor_styles) read <name>_profiles.csv; nominal
MFV reads mfv.json and is projected into z-scored relative-emphasis space (its
logit-violation units cannot share a raw axis with 1-5 wrongness), but it goes
through the same plot_ipsative_pca / plot_range and yields the same two figures.

cs are SIGNED multipliers of the calibrated coefficient C (0 = base); the real C
is in the title, the legend shows only the multiplier (c=+1, c=-2, ...).

  uv run python scripts/plot_steer_showcase.py \
    --run-dir ../steering-lite/outputs/allinstr_qwen35_4b --out docs/img/showcase
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from types import SimpleNamespace

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import tinymfv as T
from tinymfv import get_instrument

ORDINAL = ["mfq2", "big5", "16pf", "humor_styles"]


def _frac(x, scale_max: int) -> np.ndarray:
    return (np.asarray(x, float) - 1) / (scale_max - 1)


def read_human_csv(path: str) -> dict[tuple[str, str], float]:
    """{(country, foundation): mean} from a tinymfv human_<instrument>.csv."""
    out: dict[tuple[str, str], float] = {}
    with open(path, newline="") as fh:
        for r in csv.DictReader(fh):
            out[(r["country"], r["foundation"])] = float(r["mean"])
    return out


def human_matrix(instr) -> tuple[list[str], np.ndarray]:
    """(countries, M[countries x factors] as 0-1 fraction). Mirrors mft_honesty.maps.human_matrix."""
    dims = instr.dimensions
    h = read_human_csv(instr.human_csv)
    countries = sorted({c for (c, _f) in h})
    raw = np.array([[h[(c, f)] for f in dims] for c in countries])
    return countries, _frac(raw, instr.human_scale_max)


def human_haze(instr, n_per_country: int = 200, seed: int = 0) -> np.ndarray:
    """Synthetic individual-respondent cloud (n x K, 0-1 fraction) for instruments that ship only
    society-level stats (big5/16pf/humor: no raw per-person data like mfq2's Atari file). For each
    (country, factor) we resample n Normal(mean, sd) draws from the published country mean+sd, so the
    cloud carries BOTH between-country (different means) and within-country (sd) human spread. Caveat:
    factors are drawn independently, so this marginal resample loses the cross-factor correlation a
    real respondent matrix has -- it is a backdrop envelope, not a covariance estimate, and is NOT
    used as the PCA basis (that stays the society means M)."""
    dims = instr.dimensions
    rng = np.random.default_rng(seed)
    stats: dict[tuple[str, str], tuple[float, float]] = {}
    with open(instr.human_csv, newline="") as fh:
        for r in csv.DictReader(fh):
            stats[(r["country"], r["foundation"])] = (float(r["mean"]), float(r["sd"]))
    countries = sorted({c for (c, _f) in stats})
    blocks = []
    for c in countries:
        cols = [rng.normal(stats[(c, f)][0], stats[(c, f)][1], n_per_country) for f in dims]
        blocks.append(np.clip(np.stack(cols, axis=1), 1.0, instr.human_scale_max))
    return _frac(np.concatenate(blocks, axis=0), instr.human_scale_max)


def human_strip(instr) -> dict[str, list[tuple[str, float]]]:
    """{factor: [(country, mean_on_model_scale)]}. Human 1-H rescaled to model 1-M for the range."""
    h = read_human_csv(instr.human_csv)
    H, M = instr.human_scale_max, instr.scale_max
    def rescale(v: float) -> float:
        return 1.0 + (v - 1.0) / (H - 1) * (M - 1) if H != M else v
    strip: dict[str, list[tuple[str, float]]] = {}
    for f in instr.dimensions:
        strip[f] = sorted(((c, rescale(v)) for (c, ff), v in h.items() if ff == f),
                          key=lambda t: t[1])
    return strip


def read_profiles(run_dir: Path, name: str, dims: list[str], value_col: str = "mean"
                  ) -> tuple[dict[float, np.ndarray], dict[float, float]]:
    """({c: profile-vector in factor order}, {c: pmass}) from <name>_profiles.csv. `c` is the signed
    multiplier of calibrated C (0 = base); a single-multiplier run yields just {-1, 0, +1}.
    value_col selects the readout: 'mean' = E (human-comparable, for the map/range vs human band);
    'C' = the rank-centered logit contrast (the steer-legible signal, for the steer-effect plot)."""
    by_c: dict[float, dict[str, float]] = {}
    pmass: dict[float, float] = {}
    with open(run_dir / f"{name}_profiles.csv", newline="") as fh:
        for r in csv.DictReader(fh):
            c = float(r["c"])
            by_c.setdefault(c, {})[r["foundation"]] = float(r[value_col])
            pmass[c] = float(r["pmass"])
    return {c: np.array([d[f] for f in dims]) for c, d in by_c.items()}, pmass


def plot_ordinal(run_dir: Path, out: Path, name: str, vec_label: str, C: float) -> list[Path]:
    instr = get_instrument(name)
    dims = instr.dimensions
    prof_c, pmass = read_profiles(run_dir, name, dims)
    cs = sorted(prof_c)
    base = prof_c[0.0]
    # headline arrows = the calibrated coefficient (c=+-1); the trajectory dots at |c|>1 extend
    # BEYOND the arrowheads, so a multi-C run shows deployment point + where stronger steer drifts.
    pos = prof_c[1.0] if 1.0 in prof_c else prof_c[max(cs)]
    neg = prof_c[-1.0] if -1.0 in prof_c else prof_c[min(cs)]
    humans = human_strip(instr)
    prof = prof_c

    countries, Mfrac = human_matrix(instr)
    labels = (f"base (c=0)", f"+C={C:+.2f}", f"-C={-C:+.2f}")
    # mfq2 has per-respondent Atari data -> scatter the REAL individual cloud behind the societies AND
    # fit the ipsative PCA on it (better-conditioned, the true envelope). Other instruments have no raw
    # per-person data, so scatter a marginal resample from each country's published mean+sd as the haze
    # while keeping the PCA basis on the society means M.
    if name == "mfq2":
        respondents, haze = T.maps.respondent_profiles(dims, instr.scale_max), None
    else:
        respondents, haze = None, human_haze(instr)
    # trajectory overlay only when the run swept more than the 3-point base/+-C (else the arrows suffice).
    # Coherence gate is RELATIVE: keep a c only if its pmass stays within 95% of the base (c=0) pmass;
    # below that the readout has degraded enough that the profile is not comparable, so drop it entirely.
    base_pm = pmass[0.0]
    coh_cs = [c for c in cs if pmass[c] >= 0.95 * base_pm]
    traj = {c: _frac(prof_c[c], instr.scale_max) for c in coh_cs} if len(coh_cs) > 3 else None
    traj_inco = None  # excluded (not drawn hollow) per the 95%-of-base coherence gate
    figm = T.maps.plot_ipsative_pca(instr, dims, countries, Mfrac,
                                    _frac(base, instr.scale_max), _frac(pos, instr.scale_max),
                                    _frac(neg, instr.scale_max), respondents=respondents, haze=haze,
                                    traj=traj, traj_incoherent=traj_inco, labels=labels)
    paths = [T.maps.save_both(figm, out / name, "map_pca_ipsative")]
    plt.close(figm)

    # Range renders the SAME coherence-gated c-points the map uses (coh_cs), so the two figures agree
    # on which steer multipliers are valid. Without this the map drops incoherent/NaN poles while the
    # range still plots them (GPT-5.5 code review). Base (c=0) is always in coh_cs (pmass==base_pm).
    assert 0.0 in coh_cs, f"{name}: base c=0 dropped by coherence gate, pmass={pmass}"
    prof_coh = {c: prof_c[c] for c in coh_cs}
    figr = T.maps.plot_range(instr, dims, coh_cs, prof_coh, humans, None, vec_label)
    paths.append(T.maps.save_both(figr, out / name, "range"))
    plt.close(figr)
    return paths


def _zscore(v: np.ndarray) -> np.ndarray:
    """Relative emphasis: centre and scale a profile across foundations, so a logit profile (model)
    and a 1-5 wrongness profile (human cultures) are comparable by PATTERN regardless of units."""
    return (v - v.mean()) / (v.std() + 1e-9)


def read_human_mfv() -> tuple[list[str], dict[str, dict[str, float]]]:
    """(countries, {country: {foundation: mean_1to5}}) from the bundled MFV human norms.
    JimenezLeal2025 (LatAm) + Yamada2025 (MFV-J): 5 countries x 6 foundations (no Social Norms)."""
    path = T.maps.DATA / "human" / "mfv_country_factors.csv"
    by_country: dict[str, dict[str, float]] = {}
    with open(path, newline="") as fh:
        for r in csv.DictReader(fh):
            by_country.setdefault(r["country"], {})[r["foundation"]] = float(r["mean"])
    return sorted(by_country), by_country


def _mfv_zspace(run_dir: Path):
    """Shared MFV adapter -> the common coordinate system the map AND range both consume: z-scored
    relative-emphasis profiles (model base / +C / -C) + the human MFV culture matrix in the same
    space. MFV is nominal (model emits logit(violation) per foundation, humans rate wrongness 1-5),
    so absolute scales differ; z-scoring each profile ACROSS foundations compares the PATTERN -- which
    foundations a reader weights as more violation-worthy than their own average -- which is exactly
    what the steer moves. Social Norms is dropped (no human MFV norm), asserted so a taxonomy change
    fails loud. Returns (founds, countries, M_z[countries x founds], base_z, posz, negz)."""
    d = json.loads((run_dir / "mfv.json").read_text())
    base_l = d["base_logit_per_foundation"]
    pos_dl, neg_dl = d["pos"]["dlogit_per_foundation"], d["neg"]["dlogit_per_foundation"]
    countries, human = read_human_mfv()
    hfounds = set(next(iter(human.values())))
    founds = [f for f in d["foundation_order"] if f.lower() in hfounds]   # shared, model order
    dropped = [f for f in d["foundation_order"] if f.lower() not in hfounds]
    assert dropped == ["Social Norms"], f"unexpected MFV foundations without a human norm: {dropped}"
    fl = [f.lower() for f in founds]
    base = _zscore(np.array([base_l[f]["mean"] for f in founds]))
    posz = _zscore(np.array([base_l[f]["mean"] + pos_dl[f]["mean"] for f in founds]))
    negz = _zscore(np.array([base_l[f]["mean"] + neg_dl[f]["mean"] for f in founds]))
    M = np.array([_zscore(np.array([human[c][f] for f in fl])) for c in countries])
    return founds, countries, M, base, posz, negz


# MFV has no ordinal Instrument (it goes through evaluate_multibool, not administer), but the shared
# plotters only read .name/.display off it -- a shim supplies those. The y-values are z-scores, not a
# 1-M scale, so it carries no scale_max and the range passes its own ylabel.
_MFV_INSTR = SimpleNamespace(name="mfv", display="MFV vignettes")
_MFV_YLABEL = "relative emphasis  (z across foundations)"


def plot_mfv_map(run_dir: Path, out: Path, vec_label: str, C: float) -> Path:
    """MFV ipsative culture map via the SAME plot_ipsative_pca the ordinal instruments use, in the
    z-scored relative-emphasis space (logit-violation and 1-5 wrongness cannot share a raw axis).
    base->+C (red) / base->-C (blue) arrows show where the steer moves the AI among human cultures."""
    founds, countries, M, base, posz, negz = _mfv_zspace(run_dir)
    labels = ("base (c=0)", f"+C={C:+.2f}", f"-C={-C:+.2f}")
    fig = T.maps.plot_ipsative_pca(_MFV_INSTR, founds, countries, M, base, posz, negz, labels=labels)
    path = T.maps.save_both(fig, out / "mfv", "map_pca_ipsative")
    plt.close(fig)
    return path


def plot_mfv_range(run_dir: Path, out: Path, vec_label: str, C: float) -> Path:
    """MFV range via the SAME plot_range the ordinal instruments use, in z relative-emphasis space.
    Only base/+C/-C (the MFV eval is a 3-point sweep, not a multi-C grid like the ordinal admin)."""
    founds, countries, M, base, posz, negz = _mfv_zspace(run_dir)
    cs = [-1.0, 0.0, 1.0]
    prof = {-1.0: negz, 0.0: base, 1.0: posz}
    humans = {f: sorted(((countries[ci], float(M[ci, fi])) for ci in range(len(countries))), key=lambda t: t[1])
              for fi, f in enumerate(founds)}
    fig = T.maps.plot_range(_MFV_INSTR, founds, cs, prof, humans, None, vec_label, ylabel=_MFV_YLABEL)
    path = T.maps.save_both(fig, out / "mfv", "range")
    plt.close(fig)
    return path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("docs/img/showcase"))
    args = ap.parse_args()
    summary = json.loads((args.run_dir / "summary.json").read_text())
    C = float(summary["calibrated_C"])
    method = summary["method"]
    vec_label = summary.get("vec_label", f"{method} (Authority/Care axis)")
    args.out.mkdir(parents=True, exist_ok=True)

    written: list[str] = []
    for name in ORDINAL:
        if (args.run_dir / f"{name}_profiles.csv").exists():
            written += [str(p) for p in plot_ordinal(args.run_dir, args.out, name, vec_label, C)]
    if (args.run_dir / "mfv.json").exists():
        written.append(str(plot_mfv_map(args.run_dir, args.out, vec_label, C)))    # shared ipsative map (z-space)
        written.append(str(plot_mfv_range(args.run_dir, args.out, vec_label, C)))  # shared range (z-space)
    print(f"wrote {len(written)} figures under {args.out}:")
    for w in written:
        print(" ", w)


if __name__ == "__main__":
    main()
