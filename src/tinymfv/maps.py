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
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from .instrument import Instrument

# range-plot palette + geometry (shared with the experiment's prior fig_profile_sweeps look)
CLOUD_GREY = "0.78"      # individual respondents (subtle backdrop)
COUNTRY_GREY = "0.38"    # society means (the named dots)
MEDIAN_GREY = "0.15"
POS_COL, NEG_COL = "#c0392b", "#2c6fbb"   # +c side (red) / -c side (blue) of the sweep
DX_HUMAN, DX_STEER = -0.17, 0.18
GROUP_PITCH = 1.55       # x-distance between factors; > pair width so each (societies, steer) reads as one unit
# ipsative-map palette
C_BASE, C_HON, C_DIS, C_HUM = "#33688f", "#c2702f", "#8a5a9c", "#888888"


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
    Returns (P, Vt, var, mu, Pc); project a new point v via ((v @ Pc) - mu) @ Vt[:k].T."""
    Pc = row_centre_op(M.shape[1])
    Mp = M @ Pc
    mu = Mp.mean(axis=0)
    Mc = Mp - mu
    _, S, Vt = np.linalg.svd(Mc, full_matrices=False)
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
    cax.patch.set_alpha(0.0)
    cax.add_patch(plt.Circle((0, 0), circle_r, fill=False, color="#bbbbbb", lw=0.7))
    for j, lab in enumerate(labels):
        x, y = L[j]
        cax.annotate("", xy=(x, y), xytext=(0, 0), arrowprops=dict(arrowstyle="->", color=color, lw=1.1))
        r = np.hypot(x, y)
        cax.text(x / r * (r + 0.07), y / r * (r + 0.07), lab.capitalize(), fontsize=8,
                 fontweight="bold", color=color, ha="left" if x >= 0 else "right",
                 va="bottom" if y >= 0 else "top", clip_on=False)
    cax.set_xlim(-1.5, 1.5); cax.set_ylim(-1.5, 1.5)
    cax.set_aspect("equal"); cax.axis("off")
    cax.set_title(title, fontsize=10, fontweight="bold", color=color, pad=3)


def plot_ipsative_pca(instr: Instrument, dims: list[str], countries: list[str], M: np.ndarray,
                      base: np.ndarray, hon: np.ndarray | None, dis: np.ndarray | None,
                      *, boots: dict | None = None, pad=(0.18, 0.16)):
    """Ipsative culture map. M is societies x K (0-1 fraction); base/hon/dis are length-K fraction
    vectors (or None). `boots` optionally maps 'base'/'honest'/'dis' -> (n x K) bootstrap fraction
    matrices for the uncertainty cross. Returns the Figure."""
    try:
        import textalloc as ta
    except ImportError:
        ta = None
    P, Vt, var, mu, Pc = ipsative_pca(M)
    if Vt[0, 0] < 0:                                    # stabilise SVD sign: factor[0] loads +PC1
        Vt[0] = -Vt[0]
    if Vt[1, 1 % len(dims)] < 0:
        Vt[1] = -Vt[1]
    P = (M @ Pc - mu) @ Vt[:2].T

    def proj(v):
        return ((v @ Pc) - mu) @ Vt[:2].T if v is not None else None
    pb, ph, pf = proj(base), proj(hon), proj(dis)

    fig, ax = plt.subplots(figsize=(8.5, 7.5))
    ax.set_facecolor("#faf8f2")
    ax.grid(True, color="#eceadf", lw=0.3, zorder=0)
    ax.scatter(P[:, 0], P[:, 1], s=26, c=C_HUM, alpha=0.7, edgecolors="white", linewidths=0.5, zorder=3)
    if ta is not None:
        try:
            ta.allocate_text(fig, ax, P[:, 0], P[:, 1], countries, x_scatter=P[:, 0], y_scatter=P[:, 1],
                             textsize=7.5, linecolor="#c2bca8", linewidth=0.5, textcolor="#555555")
        except Exception:
            ta = None
    if ta is None:
        for i, c in enumerate(countries):
            ax.annotate(c, (P[i, 0], P[i, 1]), fontsize=7, color="#555555",
                        xytext=(3, 2), textcoords="offset points")
    for key, col, pt in [("base", C_BASE, pb), ("honest", C_HON, ph), ("dis", C_DIS, pf)]:
        if boots and key in boots and pt is not None:
            bp = (np.asarray(boots[key]) @ Pc - mu) @ Vt[:2].T
            e1, e2 = 1.96 * bp.std(0)
            ax.errorbar(pt[0], pt[1], xerr=e1, yerr=e2, fmt="none", ecolor=col,
                        elinewidth=0.7, alpha=0.55, capsize=2.5, capthick=0.8, zorder=4)
    for pt, col, mk, lab, dxy, ha in [(ph, C_HON, "s", "honest (c=+2)", (9, 9), "left"),
                                      (pf, C_DIS, "^", "dishonest (c=-2)", (-9, -1), "right"),
                                      (pb, C_BASE, "o", "baseline (c=0)", (9, -13), "left")]:
        if pt is None:
            continue
        if pt is not pb:
            ax.annotate("", xy=pt, xytext=pb, arrowprops=dict(arrowstyle="-|>", color=col, lw=2.0), zorder=5)
        ax.scatter(*pt, s=120, c=col, marker=mk, edgecolors="white", linewidths=1.2, zorder=7)
        ax.annotate(lab, pt, xytext=dxy, textcoords="offset points", fontsize=9, color=col,
                    fontweight="bold", ha=ha, va="center", zorder=8)
    compass(ax, Vt[:2].T, dims, title=f"{instr.display} compass")
    x0, x1 = ax.get_xlim(); y0, y1 = ax.get_ylim()      # modest top-right headroom for the compass inset
    ax.set_xlim(x0, x1 + pad[0] * (x1 - x0)); ax.set_ylim(y0, y1 + pad[1] * (y1 - y0))
    ax.set_xlabel(f"PC1 ({var[0]*100:.0f}% var, relative emphasis)")
    ax.set_ylabel(f"PC2 ({var[1]*100:.0f}% var, relative emphasis)")
    ax.set_title(f"{instr.name}: ipsative culture map ({len(countries)} societies)", fontsize=10)
    return fig


# --- per-vector steer range ---------------------------------------------------------------------

def draw_range_panel(ax, instr: Instrument, dims: list[str], cs: list[float], prof: dict,
                     humans: dict, cloud: np.ndarray | None, vec: str) -> tuple[float, float]:
    """One vector's range panel: respondent cloud + named society dots + the steer c-sweep drawn as
    a directed -c->+c axis (tail dot at -c, interior dots, single arrowhead at +c). The widest steer
    carries a '-N {vec} / base / +N {vec}' in-plot key. Returns the cropped (ymin, ymax)."""
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
        for ca, cb, ya, yb in zip(cs[:-1], cs[1:], yv[:-1], yv[1:]):
            ax.plot([xs, xs], [ya, yb], color=(NEG_COL if cb <= 0 else POS_COL),
                    lw=2.0, zorder=6, solid_capstyle="round")
        for k, (c, y) in enumerate(zip(cs, yv)):                   # tail + interior get dots; +c end is the arrowhead
            if k == len(cs) - 1:
                continue
            ax.plot(xs, y, "o", ms=3.5, zorder=7,
                    color="black" if c == 0 else (POS_COL if c > 0 else NEG_COL))
        if len(cs) >= 2:
            ax.plot(xs, yv[-1], marker=("^" if yv[-1] >= yv[-2] else "v"),
                    color=POS_COL, ms=6.5, markeredgecolor="none", zorder=8)
        if i == label_i:
            base_y = float(yv[list(cs).index(0.0)])
            for c_end, y_end in [(cs[0], yv[0]), (0.0, base_y), (cs[-1], yv[-1])]:
                txt = "base" if c_end == 0 else f"{int(c_end):+d} {vec}"
                ax.annotate(txt, (xs + 0.10, y_end), fontsize=6.8, ha="left", va="center",
                            color=(MEDIAN_GREY if c_end == 0 else (POS_COL if c_end > 0 else NEG_COL)), zorder=9)

    pad = 0.10 * (max(ys) - min(ys))
    ax.set_ylim(min(ys) - pad, max(ys) + pad)
    ax.set_title(f"steer c-sweep {int(min(cs)):+d}..{int(max(cs)):+d} (coherent only); "
                 f"tail = -c, arrow = +c (more {vec})", fontsize=9)
    ax.set_xticks(np.arange(len(dims)) * GROUP_PITCH)
    ax.set_xticklabels(dims, rotation=30, ha="right", fontsize=8)
    ax.set_xlim(-0.6, (len(dims) - 1) * GROUP_PITCH + 0.6)
    ax.grid(axis="y", alpha=0.15)
    ax.spines[["top", "right"]].set_visible(False)
    return min(ys) - pad, max(ys) + pad


def plot_range(instr: Instrument, dims: list[str], cs: list[float], prof: dict, humans: dict,
               cloud: np.ndarray | None, vec: str, key_text: str):
    """Single-axes range figure for one steering vector. Returns the Figure."""
    figw = max(6.4, 1.5 * len(dims) + 1.5)
    fig, ax = plt.subplots(figsize=(figw, 4.8))
    draw_range_panel(ax, instr, dims, cs, prof, humans, cloud, vec)
    ax.set_ylabel(f"{instr.display} mean (1-{instr.scale_max})")
    fig.suptitle(f"Steered {instr.display} range: {vec}\n{key_text}", fontsize=7.5, y=1.0)
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
        lo, hi = min(yv.min(), q1), max(yv.max(), q3)
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
        for ca, cb, ya, yb in zip(cs[:-1], cs[1:], yv[:-1], yv[1:]):
            ax.plot([xs, xs], [ya, yb], color=(NEG_COL if cb <= 0 else POS_COL), lw=2.4, zorder=6, solid_capstyle="round")
        for k, (c, y) in enumerate(zip(cs, yv)):                   # tail + interior get dots; +c end is the arrowhead
            if k != len(cs) - 1:
                ax.plot(xs, y, "o", ms=4.5, color="black" if c == 0 else (POS_COL if c > 0 else NEG_COL), zorder=7)
        if len(cs) >= 2:
            ax.plot(xs, yv[-1], marker=("^" if yv[-1] >= yv[-2] else "v"), color=POS_COL, ms=7, markeredgecolor="none", zorder=8)

        base_y = float(yv[list(cs).index(0.0)])
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
    fig.suptitle(f"Steered {instr.display}, zoomed per subscale: {vec}\n"
                 f"grey = nearby societies (named: nearest the base / +end / -end; corner ^v = the 2 off-range "
                 f"extremes); coloured line = steer (tail = -c, arrow = +c); each panel has its OWN y-axis", fontsize=8)
    fig.tight_layout()
    return fig
