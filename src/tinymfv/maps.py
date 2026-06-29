"""Instrument-parameterized culture-map + range viz: the visual proof of the survey eval.

Two plot families, both pure (numpy arrays + an Instrument in, matplotlib Figure out). All data
loading -- LLM profiles, calibration windows, the human cross-cultural CSVs -- stays in the calling
experiment; this module only draws, so it serves every instrument (MFQ-2 / Big5 / 16PF / HSQ) the
same way.

  plot_ipsative_pca : where the model sits among human cultures. Each society's profile is
                      row-centred (its overall endorsement level removed) then PCA'd, so the axes
                      are RELATIVE emphasis across factors, not acquiescence. The model's base +
                      steered poles are projected into the same space; a compass inset shows how
                      each factor loads.
  plot_range        : per-factor, the steer as a directed c-sweep (tail at the -c pole, single
                      arrowhead at the +c pole) against the strip of human societies. With a zoomed
                      small-multiple companion (plot_range_zoom), one panel per factor on its own
                      y-axis.

Ported from the weight_steer_honesty experiment (mft_honesty.maps / mapviz + scripts/
fig_profile_sweeps.py). The experiment's MFQ-2-specific overlays (MFT theory axes, delta table)
stay there -- they are about the steering vectors, not the instrument.
"""
from __future__ import annotations
import csv
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from .instrument import Instrument

DATA = Path(__file__).resolve().parent / "data"

# Official MFQ-2 final 6-factor keying, verbatim from Atari et al. Code_Study2.R `model_6factors`.
# Drives respondent_profiles below (the individual cloud behind the ipsative map). MFQ-2 only.
MFQ2_FOUNDATION_ITEMS = {
    "care":            ["care1", "care3", "care11", "care12", "care13", "care14"],
    "equality":        ["equalFairness6", "equalFairness10", "equality2", "equality4", "equality6", "equality10"],
    "proportionality": ["propFairness1", "propFairness3", "proportionality5", "proportionality9", "proportionality12", "proportionality17"],
    "loyalty":         ["loyalty5", "loyalty6", "loyalty12", "loyalty13", "loyalty14", "loyalty16"],
    "authority":       ["authority6", "authority8", "authority11", "authority14", "authority18", "authority20"],
    "purity":          ["purity2", "purity3", "purity6", "purity9", "purity13", "purity17"],
}


def respondent_profiles(foundations: list[str], scale_max: int = 5) -> np.ndarray:
    """Per-respondent MFQ-2 6-foundation profiles (each foundation = mean of its 6 raw 1-5 items),
    returned as 0-1 FRACTION (same scale as `M` in plot_ipsative_pca) in `foundations` order.
    Rows with any NA in the keyed items are dropped (fail-fast, no imputation). MFQ-2 only --
    keying is MFQ2_FOUNDATION_ITEMS. Source: data/atari_study2_raw.csv (Atari et al. 2023 Study 2)."""
    raw_path = DATA / "atari_study2_raw.csv"
    rows = list(csv.DictReader(raw_path.open(newline="")))
    cols = rows[0].keys()
    missing = [c for items in MFQ2_FOUNDATION_ITEMS.values() for c in items if c not in cols]
    if missing:
        raise KeyError(f"keying columns absent from {raw_path.name}: {missing}")
    def cell(v: str) -> float:
        return np.nan if v in ("NA", "", "-99") else float(v)
    out = []
    for r in rows:
        prof = [np.mean([cell(r[c]) for c in MFQ2_FOUNDATION_ITEMS[f]]) for f in foundations]
        if not np.isnan(prof).any():
            out.append(prof)
    X = np.array(out)                                   # (n_resp x K), raw 1-5
    return (X - 1) / (scale_max - 1)                    # -> 0-1 fraction

# range-plot palette + geometry (shared with the experiment's prior fig_profile_sweeps look)
CLOUD_GREY = "0.78"      # individual respondents (subtle backdrop)
COUNTRY_GREY = "0.38"    # society means (the named dots)
MEDIAN_GREY = "0.15"
POS_COL, NEG_COL = "#c0392b", "#2c6fbb"   # +c side (red) / -c side (blue) of the sweep
DX_HUMAN, DX_STEER = -0.17, 0.18
GROUP_PITCH = 1.55       # x-distance between factors; > pair width so each (societies, steer) reads as one unit
# ipsative-map palette. Keep steer colors identical to the range plots:
# base is neutral, +c is red, -c is blue, human societies are grey.
C_BASE, C_HON, C_DIS, C_HUM = "#111111", POS_COL, NEG_COL, "#888888"


def save_both(fig, fig_dir: Path, stem: str, dpi: int = 200) -> Path:
    fig_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_dir / f"{stem}.png", dpi=dpi, bbox_inches="tight")
    fig.savefig(fig_dir / f"{stem}.svg", bbox_inches="tight")
    return fig_dir / f"{stem}.png"


# --- ipsative PCA culture map -------------------------------------------------------------------

def row_centre_op(K: int) -> np.ndarray:
    """Ipsative operator: M @ Pc subtracts each row's own mean across the K factors, removing the
    overall-endorsement/acquiescence level that would otherwise dominate PC1."""
    return np.eye(K) - np.ones((K, K)) / K


def ipsative_pca(M: np.ndarray, k: int = 2):
    """Row-centre each row of M (societies x K), then PCA across rows.
    Returns (P, Vt, var, mu, Pc); project a new point v via ((v @ Pc) - mu) @ Vt[:k].T.
    SVD signs are stabilized here (PC1 loads + on factor 0, PC2 on factor 1) so this helper and
    any plot built on it share ONE orientation -- otherwise saved coords could mirror the figure."""
    K = M.shape[1]
    Pc = row_centre_op(K)
    Mp = M @ Pc
    mu = Mp.mean(axis=0)
    Mc = Mp - mu
    _, S, Vt = np.linalg.svd(Mc, full_matrices=False)
    if Vt[0, 0] < 0:
        Vt[0] = -Vt[0]
    if Vt.shape[0] > 1 and Vt[1, 1 % K] < 0:
        Vt[1] = -Vt[1]
    var = (S ** 2) / (S ** 2).sum()
    return Mc @ Vt[:k].T, Vt, var, mu, Pc


def compass(ax_main, L: np.ndarray, labels: list[str], title: str = "compass",
            box=(0.62, 0.70, 0.30, 0.27), color: str = "#3a6b35") -> None:
    """Compass-rose inset (axes-fraction): each factor is an arrow (direction = 2D loading, length =
    magnitude); reference circle at 80% of the shortest arrow so labels sit at the tips."""
    lens = np.linalg.norm(L, axis=1)
    L = L / (lens.max() + 1e-9)
    circle_r = 0.8 * lens.min() / lens.max()
    cax = ax_main.inset_axes(list(box))
    cax.patch.set_facecolor("#faf8f2"); cax.patch.set_alpha(0.92)   # opaque: sits in the padded legend strip
    cax.add_patch(plt.Circle((0, 0), circle_r, fill=False, color="#bbbbbb", lw=0.7))
    tx, ty, tips_x, tips_y = [], [], [], []
    for j, lab in enumerate(labels):
        x, y = L[j]
        cax.annotate("", xy=(x, y), xytext=(0, 0), arrowprops=dict(arrowstyle="->", color=color, lw=1.1))
        r = np.hypot(x, y)
        tx.append(x / r * (r + 0.07)); ty.append(y / r * (r + 0.07)); tips_x.append(x); tips_y.append(y)
    cax.set_xlim(-1.5, 1.5); cax.set_ylim(-1.5, 1.5)
    placed = False
    try:                                                       # textalloc spreads colliding tip labels
        import textalloc as ta
        ta.allocate_text(ax_main.figure, cax, tx, ty, [l.capitalize() for l in labels],
                         x_scatter=tips_x + [0], y_scatter=tips_y + [0], textsize=7.5,
                         linecolor=color, linewidth=0.5, textcolor=color, draw_lines=True)
        placed = True
    except Exception:
        placed = False
    if not placed:
        for j, lab in enumerate(labels):
            x, y = L[j]; r = np.hypot(x, y)
            cax.text(x / r * (r + 0.07), y / r * (r + 0.07), lab.capitalize(), fontsize=7.5,
                     fontweight="bold", color=color, ha="left" if x >= 0 else "right",
                     va="bottom" if y >= 0 else "top", clip_on=False)
    cax.set_aspect("equal"); cax.axis("off")
    cax.set_title(title, fontsize=10, fontweight="bold", color=color, pad=3)


def _minimap(ax_main, cloud_full: np.ndarray, societies: np.ndarray, base_pt, view, box) -> None:
    """Macro overview inset: the FULL human cloud + all societies + the model base, with a red
    rectangle marking the zoomed main frame -- so a tightly-cropped map still shows where its
    window sits in the whole space (and any off-frame society stays visible here). No labels: the
    inset is too small to letter without clutter; orientation comes from the rectangle alone."""
    from matplotlib.patches import Rectangle
    xlo, xhi, ylo, yhi = view
    mm = ax_main.inset_axes(list(box))
    mm.set_facecolor("#faf8f2")                        # opaque: sits in the padded legend strip
    mm.scatter(cloud_full[:, 0], cloud_full[:, 1], s=2, c="#8f8a7e", alpha=0.12,
               edgecolors="none", rasterized=True, zorder=1)
    mm.scatter(societies[:, 0], societies[:, 1], s=5, c=C_HUM, alpha=0.8, edgecolors="none", zorder=2)
    if base_pt is not None:
        mm.plot(base_pt[0], base_pt[1], "o", ms=3, color=C_BASE, zorder=3)
    mm.add_patch(Rectangle((xlo, ylo), xhi - xlo, yhi - ylo, fill=False, ec=POS_COL, lw=1.0, zorder=4))
    mm.set_xticks([]); mm.set_yticks([])
    mm.set_title("all human respondents", fontsize=6.0, color="0.4", pad=2)
    for s in mm.spines.values():
        s.set_color("0.7"); s.set_linewidth(0.5)


def _axis_gloss(load1: np.ndarray, dims: list[str], n: int = 2) -> str:
    """One-line interpretation of a PC from its loadings: top-n +loading factors vs top-n -loading,
    e.g. 'loyalty/authority (+) vs equality/care (-)'. So the axis is readable without the compass."""
    order = np.argsort(load1)
    neg = "/".join(dims[i] for i in order[:n])
    pos = "/".join(dims[i] for i in order[::-1][:n])
    return f"{pos} (+) vs {neg} (-)"


def plot_ipsative_pca(instr: Instrument, dims: list[str], countries: list[str], M: np.ndarray,
                      base: np.ndarray, pos: np.ndarray | None, neg: np.ndarray | None,
                      *, respondents: np.ndarray | None = None, haze: np.ndarray | None = None,
                      traj: dict[float, np.ndarray] | None = None, traj_incoherent: set | None = None,
                      boots: dict | None = None,
                      labels: tuple[str, str, str] = ("baseline (c=0)", "honest (c=+2)", "dishonest (c=-2)")):
    """Ipsative culture map. M is societies x K (0-1 fraction); base / pos / neg are the length-K
    fraction vectors for the base model and its two steer poles (or None). `labels` is the legend
    text (base, +pole, -pole) -- override it for a non-honesty steer or a different coefficient.
    `respondents` (n_resp x K, SAME 0-1 fraction scale as M) is the PCA BASIS when given: fitting on
    people (thousands of points, real covariance) beats fitting on a handful of noisy society means.
    `haze` (n x K, same fraction scale) is the human cloud SCATTERED behind the societies and used
    for the crop -- separate from the fit so instruments with only society-level mean+sd (big5/16pf/
    humor: a marginal resample) get a backdrop without that resample dictating the axes. mfq2 passes
    real `respondents` (also used as the haze when `haze` is None). With neither, fit on M, pad-crop,
    no backdrop. `traj` (signed c-multiplier -> length-K fraction vector) draws the full steer SWEEP
    as a connected path through PC space, so a multi-C run shows where the steer leaves the human
    cloud and curves into incoherence (the base/pos/neg arrows stay as the headline +-C anchors).
    `traj_incoherent` is the subset of those c whose admin pmass fell below the coherence floor --
    drawn hollow. `boots` optionally maps 'base'/'honest'/'dis' -> (n x K) bootstrap matrices. Returns
    the Figure."""
    try:
        import textalloc as ta
    except ImportError:
        ta = None
    fit_on = respondents if respondents is not None else M
    _, Vt, var, mu, Pc = ipsative_pca(fit_on)          # signs already stabilized inside the helper
    P = (M @ Pc - mu) @ Vt[:2].T
    cloud = haze if haze is not None else respondents  # what we scatter + crop to (fit is separate)

    def proj(v):
        return ((v @ Pc) - mu) @ Vt[:2].T if v is not None else None
    pb, ph, pf = proj(base), proj(pos), proj(neg)

    fig, ax = plt.subplots(figsize=(8.5, 7.5))
    ax.set_facecolor("#faf8f2")
    ax.grid(True, color="#eceadf", lw=0.3, zorder=0)
    if cloud is not None:                              # grey haze = human respondents (rasterized; SVG-safe)
        Pi = (cloud @ Pc - mu) @ Vt[:2].T
        ax.scatter(Pi[:, 0], Pi[:, 1], s=4, c="#8f8a7e", alpha=0.14, edgecolors="none",
                   zorder=1, rasterized=True)
    ax.scatter(P[:, 0], P[:, 1], s=26, c=C_HUM, alpha=0.7, edgecolors="white", linewidths=0.5, zorder=3)
    # Society labels: each 2-letter ISO code is pinned RIGHT NEXT to its dot (small fixed offset, no
    # leader line). A label is dropped entirely if its box would collide with an already-placed one --
    # better an omitted code than one flung far from its point. No relocation, no arrows.
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    placed_boxes = []
    for i in np.argsort(P[:, 0]):                      # left-to-right; leftmost wins contested space
        t = ax.annotate(countries[i], (P[i, 0], P[i, 1]), fontsize=7, color="#555555",
                        xytext=(3, 2), textcoords="offset points", zorder=6)
        bb = t.get_window_extent(renderer)
        if any(bb.overlaps(b) for b in placed_boxes):
            t.remove()
        else:
            placed_boxes.append(bb)
    for key, col, pt in [("base", C_BASE, pb), ("honest", C_HON, ph), ("dis", C_DIS, pf)]:
        if boots and key in boots and pt is not None:
            bp = (np.asarray(boots[key]) @ Pc - mu) @ Vt[:2].T
            e1, e2 = 1.96 * bp.std(0)
            ax.errorbar(pt[0], pt[1], xerr=e1, yerr=e2, fmt="none", ecolor=col,
                        elinewidth=0.7, alpha=0.55, capsize=2.5, capthick=0.8, zorder=4)
    base_lab, pos_lab, neg_lab = labels
    for pt, col, lab, dxy, ha in [(ph, C_HON, pos_lab, (9, 9), "left"),
                                  (pf, C_DIS, neg_lab, (-9, -1), "right"),
                                  (pb, C_BASE, base_lab, (9, -13), "left")]:
        if pt is None:
            continue
        if pt is not pb:
            ax.annotate("", xy=pt, xytext=pb, arrowprops=dict(arrowstyle="-|>", color=col, lw=2.0), zorder=5)
        ax.scatter(*pt, s=120, c=col, marker="o", edgecolors="white", linewidths=1.2, zorder=7)
        ax.annotate(lab, pt, xytext=dxy, textcoords="offset points", fontsize=9, color=col,
                    fontweight="bold", ha=ha, va="center", zorder=8)
    traj_pts = None
    if traj:
        inco = traj_incoherent or set()
        cs_sorted = sorted(traj)
        cmax = max(abs(c) for c in cs_sorted) or 1.0
        traj_pts = np.array([proj(traj[c]) for c in cs_sorted])
        # two arms fanning from base (c=0): +c red, -c blue. Marker grows with |c|; a point whose
        # admin pmass fell below the coherence floor is hollow (the steer is no longer measuring).
        for lo, hi in [(0.0, max(cs_sorted)), (min(cs_sorted), 0.0)]:
            arm = [(c, proj(traj[c])) for c in cs_sorted if lo <= c <= hi]
            if len(arm) < 2:
                continue
            xy = np.array([p for _, p in arm])
            ax.plot(xy[:, 0], xy[:, 1], "-", color="0.55", lw=0.9, zorder=4, alpha=0.8)
            for c, p in arm:
                if c == 0:
                    continue
                col = POS_COL if c > 0 else NEG_COL
                ax.scatter(p[0], p[1], s=14 + 26 * abs(c) / cmax, c="none" if c in inco else col,
                           edgecolors=col, linewidths=1.0, zorder=6)
            cend, pend = arm[-1] if hi > 0 else arm[0]
            ax.annotate(f"c={cend:+.0f}", pend, xytext=(4, 4), textcoords="offset points",
                        fontsize=7, color=POS_COL if cend > 0 else NEG_COL, zorder=8,
                        bbox=dict(boxstyle="round,pad=0.1", fc="#faf8f2", ec="none", alpha=0.7))
    # Crop to the SOCIETIES + steer anchors for EVERY instrument (the human cloud is far wider and would
    # bury them in a central blob; it stays a clipped backdrop). Then PAD THE BOTTOM to reserve a clean
    # strip for the legend insets -- deterministic placement, identical on every plot, no overlap with
    # data or labels regardless of where the trajectory heads.
    PAD_B = 0.62                                          # bottom legend strip = PAD_B of the data y-range
    if cloud is not None:
        anc_extra = [traj_pts] if traj_pts is not None else []
        anc = np.vstack([P] + [p for p in (pb, ph, pf) if p is not None] + anc_extra)
        cx, cy = np.percentile(P[:, 0], [2, 98]), np.percentile(P[:, 1], [2, 98])
        wx0, wx1 = min(cx[0], np.nanmin(anc[:, 0])), max(cx[1], np.nanmax(anc[:, 0]))
        wy0, wy1 = min(cy[0], np.nanmin(anc[:, 1])), max(cy[1], np.nanmax(anc[:, 1]))
        sx, sy = wx1 - wx0, wy1 - wy0
        dview = (wx0 - 0.05 * sx, wx1 + 0.05 * sx, wy0 - 0.05 * sy, wy1 + 0.05 * sy)  # data frame, no pad
    else:
        x0, x1 = ax.get_xlim(); y0, y1 = ax.get_ylim()
        dview = (x0, x1, y0, y1); sy = y1 - y0
    ax.set_xlim(dview[0], dview[1])
    ax.set_ylim(dview[2] - PAD_B * sy, dview[3])
    # legend strip along the padded bottom: minimap bottom-right (it reads like a small map), compass
    # bottom-left. Both insets are opaque (set in compass()/_minimap) so the faint haze stays behind them.
    if cloud is not None:
        _minimap(ax, Pi, P, pb, dview, box=(0.71, 0.01, 0.27, 0.27))
    compass(ax, Vt[:2].T, dims, title=f"{instr.display} compass", box=(0.0, 0.01, 0.40, 0.26))
    ax.set_xlabel(f"PC1 ({var[0]*100:.0f}% var) · {_axis_gloss(Vt[0], dims)}")
    ax.set_ylabel(f"PC2 ({var[1]*100:.0f}% var) · {_axis_gloss(Vt[1], dims)}")
    ax.set_title(f"{instr.name}: ipsative culture map ({len(countries)} societies)", fontsize=10)
    return fig


# --- foundation scatter-plot matrix (SPLOM) -----------------------------------------------------

def plot_splom(instr: Instrument, dims: list[str], cloud: np.ndarray, M: np.ndarray,
               base: np.ndarray, prof_by_c: dict[float, np.ndarray], *,
               select: int | None = None, zoom: bool = False, vec_label: str = ""):
    """Scatter-plot matrix of the foundations: the ipsative map's 2-PC projection seen pair by pair,
    so the steer's path is read in each raw foundation-plane, not just the top-2 PCs.

      lower triangle : human joint scatter (REAL covariance) -- respondents recede (light), society
                       means sit darker, and the AI base -> +-c trajectory dominates (saturated).
      diagonal       : that foundation's human marginal (hist) with the AI base/+-c as vertical rules.
      upper triangle : Pearson r as a number sized+coloured by |r| (red +, blue -) -- the correlation
                       the lower scatter shows, given once as a value rather than a duplicate cloud.

    Foundations are ordered by ipsative PC1 loading so correlated factors sit adjacent (block
    structure) and the order matches plot_ipsative_pca. `cloud`/`M`/`base`/`prof_by_c` are 0-1
    fraction; displayed on the native 1..scale_max scale. `select` keeps the N foundations the steer
    moves most (largest |+c - -c| span); None keeps all. `zoom=True` frames each axis to the AI
    trajectory +-margin (micro: the small steer move); False frames to the human 2-98 pct (macro:
    AI vs the whole human spread). mfq2 ONLY -- the other instruments ship an independent-marginal
    haze (no real joint), so their off-diagonals would FABRICATE the correlation structure."""
    cs = sorted(prof_by_c)
    K = len(dims)
    _, Vt, *_ = ipsative_pca(cloud if cloud is not None else M)
    order = list(np.argsort(Vt[0])[::-1])                      # high +PC1 first (matches the map axis)
    if select is not None and select < K:
        span = np.abs(prof_by_c[cs[-1]] - prof_by_c[cs[0]])
        keep = set(np.argsort(span)[::-1][:select].tolist())
        order = [i for i in order if i in keep]
    n = len(order)
    smax = instr.scale_max
    nat = lambda fr: 1.0 + np.asarray(fr) * (smax - 1)
    cloud_n, M_n = nat(cloud), nat(M)
    prof_n = {c: nat(prof_by_c[c]) for c in cs}

    # per-foundation display range (shared down each col / across each row)
    # NaN-safe: a collapsed pole reads NaN ("do not compare"); nan-reduce the trajectory and fall
    # back to the human spread when a foundation's whole steer arm collapsed.
    rng: dict[int, tuple[float, float]] = {}
    for f in order:
        tv = np.array([prof_n[c][f] for c in cs])
        tlo, thi = (np.nanmin(tv), np.nanmax(tv)) if np.isfinite(tv).any() else (np.nan, np.nan)
        hlo, hhi = np.percentile(cloud_n[:, f], [2, 98])
        if zoom and np.isfinite(tlo):
            lo, hi = tlo, thi
            m = max(0.20 * (hi - lo), 0.06 * (smax - 1))
        else:
            lo = np.nanmin([hlo, tlo]); hi = np.nanmax([hhi, thi])
            m = 0.04 * (hi - lo)
        rng[f] = (lo - m, hi + m)

    fig, axes = plt.subplots(n, n, figsize=(1.55 * n + 0.6, 1.55 * n + 0.6), squeeze=False)
    rngen = np.random.default_rng(0)
    for r in range(n):
        fr = order[r]
        for c in range(n):
            fc = order[c]
            ax = axes[r][c]
            ax.tick_params(labelsize=6, length=2)
            if r == c:                                          # human distribution (violin) + AI steer
                # horizontal violin = human spread of this foundation; AI base/+C/-C ride on it as the
                # same trajectory dots used off-diagonal (1-D analogue), so the diagonal reads the same way.
                vp = ax.violinplot(cloud_n[:, fr], positions=[0.5], vert=False, widths=0.9,
                                   showextrema=False)
                for b in vp["bodies"]:
                    b.set_facecolor(CLOUD_GREY); b.set_alpha(0.5); b.set_edgecolor("none")
                ax.plot([prof_n[cc][fr] for cc in cs], [0.5] * len(cs), "-", color="0.5", lw=0.7, zorder=4)
                cmax = max(abs(cc) for cc in cs) or 1.0
                for cc in cs:
                    if not np.isfinite(prof_n[cc][fr]):
                        continue
                    col = "black" if cc == 0 else (POS_COL if cc > 0 else NEG_COL)
                    ax.scatter(prof_n[cc][fr], 0.5, s=(34 if cc == 0 else 14 + 18 * abs(cc) / cmax),
                               color=col, edgecolors="white", linewidths=0.4, zorder=6)
                ax.set_xlim(*rng[fr]); ax.set_ylim(0, 1); ax.set_yticks([])
                ax.text(0.5, 0.94, dims[fr], transform=ax.transAxes, ha="center", va="top",
                        fontsize=7.5, fontweight="bold", color="0.25")
            elif r > c:                                         # joint scatter + trajectory
                # jitter the respondent cloud: MFQ-2 per-foundation scores are ordinal, so raw points
                # land on a lattice that reads as confusing grid-dots; jitter softens it to a density.
                jit = 0.05 * (smax - 1)
                ax.scatter(cloud_n[:, fc] + rngen.uniform(-jit, jit, cloud_n.shape[0]),
                           cloud_n[:, fr] + rngen.uniform(-jit, jit, cloud_n.shape[0]),
                           s=3, color=CLOUD_GREY, alpha=0.10,
                           edgecolors="none", zorder=1, rasterized=True)
                ax.scatter(M_n[:, fc], M_n[:, fr], s=9, color=COUNTRY_GREY, alpha=0.75,
                           edgecolors="none", zorder=2)
                px = [prof_n[cc][fc] for cc in cs]; py = [prof_n[cc][fr] for cc in cs]
                ax.plot(px, py, "-", color="0.5", lw=0.7, zorder=4)
                cmax = max(abs(cc) for cc in cs) or 1.0
                for cc in cs:
                    col = "black" if cc == 0 else (POS_COL if cc > 0 else NEG_COL)
                    ax.scatter(prof_n[cc][fc], prof_n[cc][fr], s=(34 if cc == 0 else 14 + 18 * abs(cc) / cmax),
                               color=col, edgecolors="white", linewidths=0.4, zorder=6)
                ax.set_xlim(*rng[fc]); ax.set_ylim(*rng[fr])
            else:                                               # upper: correlation as a value
                rp = float(np.corrcoef(cloud_n[:, fc], cloud_n[:, fr])[0, 1])
                ax.text(0.5, 0.5, f"{rp:+.2f}", transform=ax.transAxes, ha="center", va="center",
                        fontsize=7 + 11 * abs(rp), color=POS_COL if rp > 0 else NEG_COL)
                ax.set_xticks([]); ax.set_yticks([])
                ax.spines[:].set_visible(False)
            if r != c:
                ax.spines[["top", "right"]].set_visible(False)
            if c != 0 or r == 0:
                ax.set_yticklabels([])
            if r != n - 1:
                ax.set_xticklabels([])
            if r == n - 1:
                ax.set_xlabel(dims[fc], fontsize=7)
            if c == 0 and r != 0:
                ax.set_ylabel(dims[fr], fontsize=7)
    scope = "zoomed to AI steer +-margin" if zoom else "full human range (2-98 pct)"
    fig.suptitle(f"{instr.display} foundation pairs: human joint (grey) vs AI base->steer ({vec_label})\n"
                 f"lower=scatter, diag=marginal, upper=Pearson r · {scope}", fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    return fig


# --- per-vector steer range ---------------------------------------------------------------------

def draw_steer(ax, xs: float, cs: list[float], yv: np.ndarray, base_y: float,
               ms: float = 3.5, lw: float = 2.0, head: float = 7.0, dx: float = 0.06,
               dots: bool = True) -> None:
    """Draw the steer c-sweep at column x=xs as TWO arms fanning out FROM the base (the unsteered
    model at c=0): a red arm to the +c pole, a blue arm to the -c pole. Each arm is a line plus a
    triangle HEAD MARKER at the pole pointing AWAY from base (^ if the pole is above base, v if below),
    so the head flips to point down on factors the steer lowers. This matches plot_ipsative_pca's
    base->pole arrows -- the unsteered model is the origin, NOT the -c pole.

    The head is the ONLY marker at each pole (no dot under it) and is a constant-size marker, NOT a
    FancyArrow: FancyArrowPatch shrinks and can flip its head once the shaft is shorter than the head,
    which is exactly the near-collapsed-pole case here. `dots` adds the intermediate per-c step dots
    (the zoom labels them c=X and wants them; the main range does not -- there they only clutter a short
    arm into a blob). The +c arm is nudged +dx in x and the -c arm -dx, so a NON-bidirectional steer
    (both poles the same side of base) reads as two short PARALLEL arms rather than overlapping."""
    assert list(cs) == sorted(cs) and 0.0 in cs, f"draw_steer needs sorted cs with c=0 (yv[-1]=+pole, yv[0]=-pole), got {cs}"
    ax.plot(xs, base_y, "o", ms=ms, color="black", zorder=7)            # base: the unsteered model
    if dots:
        for c, y in zip(cs, yv):
            if c == 0 or c == cs[0] or c == cs[-1]:                     # base drawn above; poles = head only
                continue
            ax.plot(xs + (dx if c > 0 else -dx), float(y), "o", ms=ms * 0.6, zorder=7,
                    color=POS_COL if c > 0 else NEG_COL)
    for pole_y, col, xo in [(float(yv[-1]), POS_COL, xs + dx), (float(yv[0]), NEG_COL, xs - dx)]:
        if abs(pole_y - base_y) > 1e-9:
            ax.plot([xo, xo], [base_y, pole_y], color=col, lw=lw, zorder=6, solid_capstyle="round")
            ax.plot(xo, pole_y, marker=("^" if pole_y >= base_y else "v"), color=col, ms=head,
                    markeredgecolor="none", zorder=8)


def draw_range_panel(ax, instr: Instrument, dims: list[str], cs: list[float], prof: dict,
                     humans: dict, cloud: np.ndarray | None, vec: str) -> tuple[float, float]:
    """One vector's range panel: respondent cloud + named society dots + the steer c-sweep drawn as
    two arrows fanning out from the base dot (red to the +c pole, blue to the -c pole). The widest
    steer carries a '-N {vec} / base / +N {vec}' in-plot key. Returns the cropped (ymin, ymax)."""
    rng = np.random.default_rng(0)
    ys: list[float] = []
    spans = [float(np.ptp([prof[c][i] for c in cs])) for i in range(len(dims))]
    label_i = int(np.argmax(spans)) if len(cs) >= 2 else -1

    for i, d in enumerate(dims):
        gx = i * GROUP_PITCH
        xh = gx + DX_HUMAN
        if cloud is not None:
            col = cloud[:, i]
            jit = (rng.random(len(col)) - 0.5) * 0.30
            yj = (rng.random(len(col)) - 0.5) * 0.17
            ax.scatter(xh + jit, col + yj, s=2, color=CLOUD_GREY, alpha=0.13, edgecolor="none", zorder=0)
        means = np.array([m for _, m in humans[d]])
        names = [c for c, _ in humans[d]]
        jit2 = (rng.random(len(means)) - 0.5) * 0.16
        ax.scatter(xh + jit2, means, s=24, color=COUNTRY_GREY, alpha=0.9,
                   edgecolor="white", linewidth=0.3, zorder=3)
        ax.plot([xh - 0.15, xh + 0.15], [np.median(means)] * 2, color=MEDIAN_GREY, lw=1.5, zorder=4)
        ys += means.tolist()
        for idx, va, dy in [(int(means.argmax()), "bottom", 0.03), (int(means.argmin()), "top", -0.03)]:
            ax.annotate(names[idx], (xh, means[idx] + dy), fontsize=7, ha="center", va=va, color="0.25", zorder=5)

        xs = gx + DX_STEER
        yv = np.array([prof[c][i] for c in cs])
        ys += yv.tolist()
        base_y = float(yv[list(cs).index(0.0)])
        draw_steer(ax, xs, cs, yv, base_y, dots=False)
        if i == label_i:
            # Only the two pole labels, to the RIGHT of the steer column. No 'base' tag: the black dot
            # between the two coloured arms is self-evidently the unsteered model, and on a near-collapsed
            # pole (e.g. humor affiliative, +c ~ base) a 'base' tag overprints the +c tag.
            for c_end, y_end, col in [(cs[-1], yv[-1], POS_COL), (cs[0], yv[0], NEG_COL)]:
                ax.annotate(f"c={int(c_end):+d}", (xs + 0.30, y_end), fontsize=6.8,
                            ha="left", va="center", color=col, zorder=9)

    pad = 0.10 * (max(ys) - min(ys))
    ax.set_ylim(min(ys) - pad, max(ys) + 1.7 * pad)               # top headroom for the human/AI strip
    # Direct labels (show, not tell): bracket the FIRST group's two columns as the human society strip
    # vs the AI steer, so the grey-dots / arms encoding is read once off the data rather than a caption.
    # Pinned to a fixed strip at the top of the panel (axes-fraction y) so it always clears the data.
    from matplotlib.transforms import blended_transform_factory
    trans = blended_transform_factory(ax.transData, ax.transAxes)
    ybar = 0.965
    for x0, txt, col, half in [(DX_HUMAN, "human", COUNTRY_GREY, 0.20), (DX_STEER, "AI", MEDIAN_GREY, 0.10)]:
        ax.plot([x0 - half, x0 - half, x0 + half, x0 + half], [ybar - 0.03, ybar, ybar, ybar - 0.03],
                transform=trans, color=col, lw=0.8, clip_on=False, zorder=10)
        ax.text(x0, ybar + 0.012, txt, transform=trans, fontsize=7.5, ha="center", va="bottom",
                fontweight="bold", color=col, zorder=10, clip_on=False)
    ax.set_xticks(np.arange(len(dims)) * GROUP_PITCH)
    ax.set_xticklabels(dims, rotation=30, ha="right", fontsize=8)
    ax.set_xlim(-0.6, (len(dims) - 1) * GROUP_PITCH + 0.6)
    ax.grid(axis="y", alpha=0.15)
    ax.spines[["top", "right"]].set_visible(False)
    return min(ys) - pad, max(ys) + pad


def plot_range(instr: Instrument, dims: list[str], cs: list[float], prof: dict, humans: dict,
               cloud: np.ndarray | None, vec: str, *, ylabel: str | None = None):
    """Single-axes range figure for one steering vector. Returns the Figure.

    The dot/line encoding legend belongs in the figure CAPTION, not the image: a paragraph of
    legend text in the suptitle forces the figure wide to fit one line and squashes the panel.
    The axes title already names the geometry (tail=-c, arrow=+c).

    `ylabel` overrides the default ordinal "<instr> mean (1-M)" -- nominal MFV passes its own
    "relative emphasis (z across foundations)" since its y-values are z-scores, not a 1-M scale."""
    figw = max(6.4, 1.5 * len(dims) + 1.5)
    fig, ax = plt.subplots(figsize=(figw, 4.8))
    draw_range_panel(ax, instr, dims, cs, prof, humans, cloud, vec)
    ax.set_ylabel(ylabel or f"{instr.display} mean (1-{instr.scale_max})")
    fig.suptitle(f"Steered {instr.display}: {vec}", fontsize=11)
    fig.tight_layout()
    return fig


def plot_range_zoom(instr: Instrument, dims: list[str], cs: list[float], prof: dict, humans: dict, vec: str):
    """Zoomed companion: one subplot per factor with its OWN y-axis, so the steer (small vs the
    human spread) is legible. Societies near the steer named; off-range extremes in the corners.
    Returns the Figure."""
    try:
        import textalloc as ta
    except ImportError:
        ta = None
    n = len(dims)
    ncol = min(3, n)
    nrow = (n + ncol - 1) // ncol
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.9 * ncol, 3.2 * nrow), squeeze=False)
    rng = np.random.default_rng(0)
    for i, d in enumerate(dims):
        ax = axes[i // ncol][i % ncol]
        yv = np.array([prof[c][i] for c in cs])
        soc = humans[d]
        soc_vals = np.array([m for _, m in soc])
        q1, q3 = np.percentile(soc_vals, [25, 75])
        # nanmin/nanmax: a collapsed pole reads NaN (read.py NaN-at-collapse, "do not compare"). The
        # base (c=0) is always finite, so the axis still frames the un-collapsed cells; draw_steer
        # skips the NaN arm on its own (the abs(NaN-base) test is False).
        lo, hi = min(np.nanmin(yv), q1), max(np.nanmax(yv), q3)
        m = max(0.10, 0.30 * (hi - lo))
        ylo, yhi = lo - m, hi + m
        near = [(name, v) for name, v in soc if ylo <= v <= yhi]
        soc_x = (-0.42 + (rng.random(len(near)) - 0.5) * 0.16) if near else np.array([])
        if near:
            ax.scatter(soc_x, [v for _, v in near], s=20, color=COUNTRY_GREY,
                       edgecolor="white", linewidth=0.3, zorder=3)
        med = float(np.median(soc_vals))
        if ylo <= med <= yhi:
            ax.plot([-0.60, -0.24], [med, med], color=MEDIAN_GREY, lw=1.2, zorder=4)

        xs = 0.30
        base_y = float(yv[list(cs).index(0.0)])
        draw_steer(ax, xs, cs, yv, base_y, ms=4.5, lw=2.4, head=9.0, dx=0.10)
        named = {}
        if near:
            name_xy = {nm: (float(x), v) for (nm, v), x in zip(near, soc_x)}
            for ref in (base_y, float(yv.max()), float(yv.min())):
                named[min(near, key=lambda t: abs(t[1] - ref))[0]] = ref
        tx = [xs] * len(cs) + [name_xy[nm][0] for nm in named] if near else [xs] * len(cs)
        ty = list(map(float, yv)) + [name_xy[nm][1] for nm in named] if near else list(map(float, yv))
        txt = [("c=0" if c == 0 else f"c={int(c):+d}") for c in cs] + list(named)
        dot_x = [xs] * len(cs) + (list(soc_x) if near else [])
        dot_y = list(map(float, yv)) + ([v for _, v in near] if near else [])
        placed = False
        if ta is not None:
            try:
                ta.allocate_text(fig, ax, tx, ty, txt, x_scatter=dot_x, y_scatter=dot_y,
                                 textsize=6.5, linecolor="#bbbbbb", linewidth=0.4, textcolor="#333333")
                placed = True
            except Exception:
                placed = False
        if not placed:
            for x, y, t in zip(tx, ty, txt):
                ax.annotate(t, (x + (0.12 if x >= 0 else -0.12), y), fontsize=6.5,
                            ha="left" if x >= 0 else "right", va="center", color="#333333")
        mx_name, mx_val = max(soc, key=lambda t: t[1])
        mn_name, mn_val = min(soc, key=lambda t: t[1])
        if mx_val > yhi:
            ax.text(0.02, 0.99, f"↑ {mx_name} {mx_val:.2f}", transform=ax.transAxes,
                    fontsize=6.3, ha="left", va="top", color=MEDIAN_GREY)
        if mn_val < ylo:
            ax.text(0.02, 0.01, f"↓ {mn_name} {mn_val:.2f}", transform=ax.transAxes,
                    fontsize=6.3, ha="left", va="bottom", color=MEDIAN_GREY)
        ax.set_title(d, fontsize=9)
        ax.set_xlim(-1.05, 1.25)
        ax.set_ylim(ylo, yhi)
        ax.set_xticks([])
        ax.tick_params(labelsize=7)
        ax.spines[["top", "right", "bottom"]].set_visible(False)
    for k in range(n, nrow * ncol):
        axes[k // ncol][k % ncol].axis("off")
    fig.suptitle(f"Steered {instr.display}, zoomed per subscale: {vec}", fontsize=11)
    fig.tight_layout()
    return fig
