"""Instrument-parameterized culture-map + range viz: the visual proof of the survey eval.

Two plot families, both pure (numpy arrays + an Instrument in, matplotlib Figure out). All data
loading -- LLM profiles, calibration windows, the human cross-cultural CSVs -- stays in the calling
experiment; this module only draws, so it serves every instrument (MFQ-2 / Big5 / 16PF / HSQ) the
same way.

  plot_ipsative_pca : where the model sits among human cultures. Each society's profile is
                      row-centred (its overall endorsement level removed) then PCA'd, so the axes
                      are RELATIVE emphasis across factors, not acquiescence. The model's base and
                      steered endpoint profiles are projected into the same space; a compass inset
                      shows how each factor loads.
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
from matplotlib.patches import Ellipse

from .instrument import Instrument
from .labelplace import allocate_labels, densify_polygon

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


def respondent_profiles(foundations: list[str], scale_max: int = 5) -> tuple[list[str], np.ndarray]:
    """Per-respondent MFQ-2 6-foundation profiles (each foundation = mean of its 6 raw 1-5 items),
    returned as (countries, X) where X is a 0-1 FRACTION matrix (same scale as `M` in
    plot_ipsative_pca) in `foundations` order and `countries[i]` is respondent i's country. Rows with
    any NA in the keyed items are dropped (fail-fast, no imputation). MFQ-2 only -- keying is
    MFQ2_FOUNDATION_ITEMS. Source: data/atari_study2_raw.csv (Atari et al. 2023 Study 2)."""
    raw_path = DATA / "atari_study2_raw.csv"
    rows = list(csv.DictReader(raw_path.open(newline="")))
    cols = rows[0].keys()
    missing = [c for items in MFQ2_FOUNDATION_ITEMS.values() for c in items if c not in cols]
    if missing:
        raise KeyError(f"keying columns absent from {raw_path.name}: {missing}")
    def cell(v: str) -> float:
        return np.nan if v in ("NA", "", "-99") else float(v)
    out, countries = [], []
    for r in rows:
        prof = [np.mean([cell(r[c]) for c in MFQ2_FOUNDATION_ITEMS[f]]) for f in foundations]
        if not np.isnan(prof).any():
            out.append(prof)
            countries.append(r["country"])
    X = np.array(out)                                   # (n_resp x K), raw 1-5
    return countries, (X - 1) / (scale_max - 1)         # -> 0-1 fraction

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

# Stable per-zone fill colors so the SAME Inglehart-Welzel zone reads the same across every
# instrument's map (research consistency). Keyed by the zone names the caller passes; an unlisted
# zone falls back to grey. -- added by Claude
ZONE_COLORS = {
    # macro zones (map default)
    "West":              "#4e79a7",
    "East Asia":         "#e15759",
    # fine zones (macro=False)
    "English-Speaking":  "#4e79a7",
    "Protestant Europe": "#59a14f",
    "Catholic Europe":   "#8cd17d",
    "Baltic":            "#499894",
    "Confucian":         "#e15759",
    # shared by both groupings
    "Orthodox":          "#b6992d",
    "Latin America":     "#f28e2b",
    "African-Islamic":   "#9c755f",
    "South Asia":        "#b07aa1",
}


def _country_region(cen: np.ndarray, pts: np.ndarray | None, sigma: float, r_fixed: float):
    """One country's territory as a shapely polygon: its real p75-ish covariance ellipse from its
    projected respondent/haze cloud `pts` (natural shape + size, scaled by `sigma`), or a fixed disc
    of radius `r_fixed` when there is no within-country cloud (MFV / WVS have only country means)."""
    from shapely.geometry import Point, Polygon as ShapelyPolygon
    if pts is not None and len(pts) >= 10:
        evals, evecs = np.linalg.eigh(np.cov(pts.T))
        evals = np.maximum(evals, (0.35 * r_fixed) ** 2)   # floor so a low-spread country stays visible
        th = np.linspace(0, 2 * np.pi, 48)
        xy = (evecs @ (sigma * np.sqrt(evals)[:, None] * np.stack([np.cos(th), np.sin(th)]))).T + cen
        return ShapelyPolygon(xy)
    return Point(*cen).buffer(r_fixed, quad_segs=24)


def _zone_hull(P: np.ndarray, cidx: dict, members: list[str], buf: float):
    """Buffered convex hull of a zone's country-mean points, or None if <2 members (can't contour)."""
    from shapely.geometry import MultiPoint
    pts = [tuple(P[cidx[c]]) for c in members if c in cidx]
    return MultiPoint(pts).convex_hull.buffer(buf, quad_segs=16) if len(pts) >= 2 else None


def select_spread_zones(P: np.ndarray, countries: list[str], zones: dict[str, list[str]],
                        n: int = 4, pad: float = 0.022) -> dict[str, list[str]]:
    """The `n` zones that COVER THE MOST SEPARATE SPACE -- greedy max-coverage on the actual hull
    areas: seed with the largest-area zone, then repeatedly add whichever zone contributes the most
    NEW (non-overlapping) area to the union. Central zones whose hull is already covered by the picks
    (Orthodox sitting inside West+East-Asia) add little and are dropped; corner cultures win. Purely
    geometric, so it works identically on any map. Zones with <2 members can't be contoured and are
    skipped."""
    cidx = {c: i for i, c in enumerate(countries)}
    buf = pad * float(np.hypot(*(P.max(0) - P.min(0))))
    hulls = {z: h for z, m in zones.items() if (h := _zone_hull(P, cidx, m, buf)) is not None}
    if len(hulls) <= n:
        return {z: zones[z] for z in hulls}
    sel = [max(hulls, key=lambda z: hulls[z].area)]
    union = hulls[sel[0]]
    while len(sel) < n:
        best = max((z for z in hulls if z not in sel), key=lambda z: hulls[z].difference(union).area)
        sel.append(best)
        union = union.union(hulls[best])
    return {z: zones[z] for z in sel}


def outlying_countries(P: np.ndarray, countries: list[str], n: int = 4) -> set[str]:
    """The corner-most country in each direction: the society that projects farthest along each of n
    compass directions from the centroid (n=4 -> the four diagonal CORNERS top-right/bottom-left/
    top-left/bottom-right; n=8 adds the axis extremes). Unlike 'n farthest from the centroid' (which
    can bunch all on one side and miss a corner), this guarantees the top-right-most, bottom-left-most
    etc. are each labelled -- the extremes a reader's eye goes to, on ANY map."""
    Pc = P - P.mean(0)
    dirs = [(1, 1), (-1, -1), (-1, 1), (1, -1), (1, 0), (-1, 0), (0, 1), (0, -1)][:n]
    return {countries[int(np.argmax(Pc[:, 0] * dx + Pc[:, 1] * dy))] for dx, dy in dirs}


def orient_geographic(P: np.ndarray, countries: list[str],
                      zones: dict[str, list[str]] | None) -> tuple[float, float]:
    """Per-axis sign (+1 / -1) that puts the cultural West to the WEST (small x, left) and the
    African-Islamic global South to the SOUTH (small y, bottom), so every map shares one orientation
    and a reader never re-learns left/right (the Economist/WVS convention). Anchors on each zone's
    country-mean centroid vs the map centroid; a missing zone leaves that axis unflipped. The two
    anchors pin two different axes, so they don't fight. -- authored by Claude"""
    if not zones:
        return 1.0, 1.0
    cidx = {c: i for i, c in enumerate(countries)}
    ctr = P.mean(0)

    def centroid(zname: str):
        idx = [cidx[c] for c in zones.get(zname, []) if c in cidx]
        return P[idx].mean(0) if idx else None

    west, south = centroid("West"), centroid("African-Islamic")
    sx = -1.0 if (west is not None and west[0] > ctr[0]) else 1.0
    sy = -1.0 if (south is not None and south[1] > ctr[1]) else 1.0
    return sx, sy


def _map_annotations(P: np.ndarray, countries: list[str], zones_all: dict[str, list[str]] | None,
                     emphasize: set[str] | None, grey: str):
    """Shared map policy for plot_value_map + plot_ipsative_pca -- one copy so the two renderers can't
    silently diverge: pick the 4 most-separate zones, colour each dot by its drawn zone (`grey` if
    ungrouped), and build label_set = landmarks (emphasize) + 4 corner outliers + one central
    representative per drawn zone. Returns (selected_zones, dot_cols, label_set)."""
    zones = select_spread_zones(P, countries, zones_all, 4) if zones_all else {}
    zone_of_c = {c: z for z, ms in zones.items() for c in ms}
    dot_cols = [ZONE_COLORS.get(zone_of_c.get(c), grey) for c in countries]
    cidx = {c: i for i, c in enumerate(countries)}
    emph = emphasize or set()
    reps = set()
    for ms in zones.values():
        if emph & set(ms):                          # zone already has a landmark labelled -> no rep
            continue                                # (else e.g. Macau SAR reps East Asia next to Japan/China)
        mem = [c for c in ms if c in cidx]
        mp = P[[cidx[c] for c in mem]]
        reps.add(mem[int(np.argmin(np.hypot(*(mp - mp.mean(0)).T)))])
    label_set = emph | outlying_countries(P, countries, 4) | reps
    return zones, dot_cols, label_set


def draw_zone_hulls(ax, P: np.ndarray, countries: list[str], zones: dict[str, list[str]],
                    pad: float = 0.022, label: bool = True) -> list[tuple[str, tuple[float, float], str]]:
    """Economist-style zone outline: the tight CONVEX HULL of a zone's country-mean points, rounded and
    slightly inflated (shapely buffer), drawn as a coloured EDGE ONLY (no fill, so overlapping zones
    don't muddy). A 1- or 2-country zone degenerates to a rounded disc/capsule via the same buffer.
    Cleaner than a union of per-country discs when the axes already separate the countries (WVS IW
    map); the disc-union `draw_zone_regions` stays for the instrument maps that overlay within-country
    respondent spread.

    Returns [(zone_name, (anchor_x, anchor_y), colour)] anchored at the hull's TOP vertex. With
    label=True (ipsative maps) the label is drawn here, nailed to that vertex. With label=False
    (value maps) it is NOT drawn -- the caller feeds these specs to the label allocator so the zone
    name gets MOVED to open space with a leader line, same as the country/model labels, instead of
    landing in a crowded spot."""
    import matplotlib.patheffects as pe
    from shapely.geometry import MultiPoint
    from matplotlib.patches import Polygon as MplPolygon
    cidx = {c: i for i, c in enumerate(countries)}
    buf = pad * float(np.hypot(*(P.max(0) - P.min(0))))
    specs: list[tuple[str, tuple[float, float], str]] = []
    for zname, members in zones.items():
        pts = np.array([P[cidx[c]] for c in members if c in cidx])
        if len(pts) < 2:                                # convex hull needs 2+ members to contour
            continue
        geom = MultiPoint([tuple(p) for p in pts]).convex_hull.buffer(buf, quad_segs=16)
        coords = np.asarray(geom.exterior.coords)
        zcol = ZONE_COLORS.get(zname, "#888888")
        ax.add_patch(MplPolygon(coords, closed=True, facecolor="none", edgecolor=zcol,
                                lw=1.8, alpha=0.9, zorder=1.5))
        if label:
            top = coords[np.argmax(coords[:, 1])]       # ipsative maps draw it at the hull top vertex
            ax.text(top[0], top[1], zname, fontsize=10, color=zcol, ha="center", va="bottom",
                    style="italic", fontweight="bold", zorder=5,
                    path_effects=[pe.withStroke(linewidth=3.0, foreground="white")])
        specs.append((zname, coords, zcol))
    return specs


def _pole_signposts(ax, med_x: float, med_y: float, poles: tuple[str, str, str, str]) -> None:
    """Four arrowed pole signposts sitting ON the median crosshairs (x=med_x vertical, y=med_y
    horizontal), pointing out to each pole, in the padded inner margin. poles = (x_neg, x_pos, y_neg,
    y_pos). All labels horizontal."""
    import matplotlib.patheffects as pe
    from matplotlib.transforms import blended_transform_factory
    xn, xp, yn, yp = poles
    tX = blended_transform_factory(ax.transData, ax.transAxes)   # x=data (on x=med line), y=axes frac
    tY = blended_transform_factory(ax.transAxes, ax.transData)   # x=axes frac, y=data (on y=med line)
    kw = dict(fontsize=11, fontweight="bold", color="#555", zorder=10, ha="center", va="center",
              path_effects=[pe.withStroke(linewidth=3.0, foreground="white")])
    awp = dict(arrowstyle="-|>", color="#999", lw=1.3)
    ax.annotate(yp, xy=(med_x, 0.995), xytext=(med_x, 0.945), xycoords=tX, arrowprops=awp, **kw)
    ax.annotate(yn, xy=(med_x, 0.005), xytext=(med_x, 0.055), xycoords=tX, arrowprops=awp, **kw)
    ax.annotate(xn, xy=(0.006, med_y), xytext=(0.085, med_y), xycoords=tY, arrowprops=awp, **kw)
    ax.annotate(xp, xy=(0.994, med_y), xytext=(0.9, med_y), xycoords=tY, arrowprops=awp, **kw)


# Economist convention: every model is the SAME bold red, told apart by its label. We keep MODEL_RED
# as the fallback but colour each model STAR by its lab FAMILY, with the hue chosen to echo the lab's
# home region (Chinese labs warm / near the East-Asia red; US labs cool blue-purple; Europe green) so a
# family clusters by colour at a glance, not just by reading labels. -- added by Claude
MODEL_RED = "#d0021b"
# Hues spread ~30-50 deg apart so no two families read alike (the earlier set had grok~gpt and
# llama~gemini~gemma colliding). Chinese labs stay warm (pink/orange, near the East-Asia red), Google's
# two share a teal/sea-blue sibling pair, grok is xAI-brand near-black to sit clear of gpt's blue.
MODEL_FAMILY_COLORS = {
    "deepseek": "#ec4899",   # DeepSeek (China) -> pink
    "qwen":     "#f97316",   # Qwen / Alibaba (China) -> orange
    "mistral":  "#2ca02c",   # Mistral (France / Europe) -> green
    "gemma":    "#14b8a6",   # Gemma / Google -> teal
    "gemini":   "#0ea5e9",   # Gemini / Google -> sea blue (sibling of gemma, bluer)
    "gpt":      "#2563eb",   # OpenAI -> blue
    "llama":    "#6d5ae0",   # Llama / Meta -> indigo
    "claude":   "#c026d3",   # Anthropic -> purple / magenta
    "grok":     "#2b2d42",   # Grok / xAI -> near-black (brand), well clear of gpt blue
}


def model_family_color(name: str) -> str:
    """The lab-family colour for a model key (substring match on the family name), MODEL_RED if none."""
    key = name.lower()
    for fam, col in MODEL_FAMILY_COLORS.items():
        if fam in key:
            return col
    return MODEL_RED


def plot_value_map(display: str, countries: list[str], P: np.ndarray,
                   poles: tuple[str, str, str, str], *, models: dict[str, tuple[float, float]] | None = None,
                   model_labels: dict[str, str] | None = None,
                   steer: dict[str, tuple[float, float, str]] | None = None,
                   emphasize: set[str] | None = None,
                   title: str | None = None, note: str | None = None):
    """The interpretable "4-value map": two NAMED axes with four pole signposts through the human
    MEDIAN crosshair, Economist-style zone hulls (the 4 most-separate zones), zone-coloured dots, and
    auto-placed labels (landmarks + corner outliers + one representative per zone + any models; see
    labelplace.allocate_labels). NO compass / minimap / ticks -- the alternative to plot_ipsative_pca.

    P is countries x 2 already in the named-axis space (see value_axes.value_coords / iw_axes).

    Two ways to overlay AI (mutually exclusive):
    - `models`: name -> (x, y[, x_se, y_se]) INDEPENDENT points (the WVS panel), each a family-coloured
      star with an auto-placed label (CI lives in the companion table, not on the figure).
    - `steer`: {"base": (x, y, label), "pos": (...), "neg": (...)} ONE model's steer, drawn as a
      CONNECTED path (black base, red +c arm, blue -c arm) with directly-placed pole labels -- the
      same visual language as plot_ipsative_pca's trajectory, so the two map families read alike.
    Returns the Figure."""
    from .zones import zones_for
    zones_all, emph = zones_for(countries)
    emph = (emphasize or set()) | emph
    # Anchor orientation geographically (West -> west, African-Islamic -> south) so every map reads the
    # same way. Flip the coords and swap the pole labels for any flipped axis; models/steer live in the
    # same space, so flip them too. Reflection leaves zone selection + corner outliers unchanged.
    P = np.asarray(P, float).copy()
    sgx, sgy = orient_geographic(P, countries, zones_all)
    xn, xp, yn, yp = poles
    if sgx < 0:
        P[:, 0] *= -1; xn, xp = xp, xn
    if sgy < 0:
        P[:, 1] *= -1; yn, yp = yp, yn
    poles = (xn, xp, yn, yp)
    flip = lambda t: (t[0] * sgx, t[1] * sgy, *t[2:])
    if models:
        models = {k: flip(v) for k, v in models.items()}
    if steer:
        steer = {k: flip(v) for k, v in steer.items()}
    zones, dot_cols, label_set = _map_annotations(P, countries, zones_all, emph, "#888888")

    med_x, med_y = float(np.median(P[:, 0])), float(np.median(P[:, 1]))
    fig, ax = plt.subplots(figsize=(10.5, 8.5))
    ax.set_facecolor("#faf8f2")
    ax.grid(True, color="#eceadf", lw=0.3, zorder=0)
    ax.axhline(med_y, color="#c9c4b4", lw=1.0, zorder=1)
    ax.axvline(med_x, color="#c9c4b4", lw=1.0, zorder=1)
    zone_specs = draw_zone_hulls(ax, P, countries, zones, label=False)   # labels go through the allocator
    ax.scatter(P[:, 0], P[:, 1], s=26, c=dot_cols, alpha=0.85, edgecolors="white", linewidths=0.5, zorder=3)

    # Two-stage placement. The few big ZONE labels get a dedicated emptiest-hull-edge search first, then
    # the point labels (country / model / steer) go through allocate_labels, which dodges dots + hull
    # edges + those zone labels. obs_x/obs_y are the dots+stars every label must dodge.
    obs_x, obs_y = list(P[:, 0]), list(P[:, 1])
    # each marker label spec: (x, y, text, colour, weight, marker_pad_px). pad clears the marker glyph.
    lab_specs = [(P[i, 0], P[i, 1], countries[i], "#111", "normal", 5.0)
                 for i, c in enumerate(countries) if c in label_set]
    if models:
        # each model is a STAR coloured by lab family. Every model is plotted, but only `model_labels`
        # get a text label (default: all) -- the WVS panel labels ONE representative per family and lets
        # colour + legend carry the rest. The 95% CI is not drawn (it lives in the companion table).
        mnames = list(models)
        mx = np.array([models[k][0] for k in mnames]); my = np.array([models[k][1] for k in mnames])
        mcols = [model_family_color(k) for k in mnames]
        ax.scatter(mx, my, s=230, marker="*", c=mcols, edgecolors="white", linewidths=0.8, zorder=8)
        for k, x, y, col in zip(mnames, mx, my, mcols):
            disp = k if model_labels is None else model_labels.get(k)
            if disp:
                lab_specs.append((x, y, disp, col, "bold", 13.0))    # star is big -> larger marker pad
        obs_x += list(mx); obs_y += list(my)
    if steer:
        bx, by, blab = steer["base"]
        for key, col in [("pos", POS_COL), ("neg", NEG_COL)]:
            if key not in steer:
                continue
            ex, ey, elab = steer[key]
            ax.plot([bx, ex], [by, ey], "-", color=col, lw=1.6, alpha=0.85, zorder=7)   # connected arm
            ax.scatter(ex, ey, s=90, c=col, edgecolors="white", linewidths=1.0, zorder=8)
            lab_specs.append((ex, ey, elab, col, "bold", 8.0))
            obs_x.append(ex); obs_y.append(ey)
        ax.scatter(bx, by, s=90, c=C_BASE, edgecolors="white", linewidths=1.0, zorder=8)
        lab_specs.append((bx, by, blab, C_BASE, "bold", 8.0))
        obs_x.append(bx); obs_y.append(by)

    ax.margins(0.17)                                     # roomy edges: the pole signposts sit in the inner
    ax.autoscale(False)                                  # margin and edge labels (Serbia, Adaptive) need to fit.
    # Orientation is already baked into the coords (orient_geographic above), so the pixel-space label
    # allocator sees the final left/right; no ax.invert_xaxis() that would mirror every placed label.
    # ONE placement pass for everything (see labelplace.allocate_labels). Each zone name is a REGION
    # label whose candidate anchors are its whole densified hull perimeter -- so it seats itself in the
    # emptiest open air outside the hull, no white box, no leader. Country/model/steer names are MARKER
    # labels sitting adjacent to their point. hard_pts = dots + stars every label dodges; soft_pts = all
    # hull edges, which only the region labels avoid (marker labels wear a white outline and may cross).
    step = 0.02 * float(np.mean(P.max(0) - P.min(0)))
    zone_perims = [densify_polygon(coords, step) for _, coords, _ in zone_specs]
    soft_pts = np.vstack(zone_perims) if zone_perims else np.empty((0, 2))
    # the four pole signposts (drawn later, ON the crosshair edges) become obstacles so a zone/point
    # label near the frame edge won't collide with 'Adaptive' / 'Self-directed' etc. Each pole is a
    # short keep-out BAND (a few points inward along its axis), since the label text is wide.
    fxm, fym = ax.transLimits.transform((med_x, med_y))
    a2d = lambda fx, fy: tuple(ax.transData.inverted().transform(ax.transAxes.transform((fx, fy))))
    pole_pts = ([a2d(fxm, f) for f in (0.945, 0.88)] + [a2d(fxm, f) for f in (0.055, 0.12)] +
                [a2d(f, fym) for f in (0.085, 0.15, 0.21)] + [a2d(f, fym) for f in (0.9, 0.84, 0.78)])
    hard_pts = np.array(list(zip(obs_x, obs_y)) + pole_pts)
    anchor_sets = zone_perims + [np.array([[s[0], s[1]]]) for s in lab_specs]
    texts = [zn for zn, _, _ in zone_specs] + [s[2] for s in lab_specs]
    colors = [zc for _, _, zc in zone_specs] + [s[3] for s in lab_specs]
    weights = ["bold"] * len(zone_specs) + [s[4] for s in lab_specs]
    fontsizes = [10.0] * len(zone_specs) + [9.0] * len(lab_specs)
    styles = ["italic"] * len(zone_specs) + ["normal"] * len(lab_specs)
    region = [True] * len(zone_specs) + [False] * len(lab_specs)
    anchor_pad = [3.0] * len(zone_specs) + [s[5] for s in lab_specs]
    allocate_labels(ax, anchor_sets, texts, colors, weights, hard_pts,
                    soft_pts=soft_pts, region=region, fontsizes=fontsizes, styles=styles,
                    anchor_pad=anchor_pad)
    _pole_signposts(ax, med_x, med_y, poles)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_xlabel(""); ax.set_ylabel("")
    # NO legend: each model family already has a directly-placed, family-coloured label on the map, so a
    # swatch legend would just duplicate that ink (Tufte eraser test). Grey dots read as societies from
    # the labelled examples (Sweden, Japan...); the count lives in the README caption.
    # Title + caption are OFF by default -- the README carries the headline + sources (nicer voice
    # there than baked jargon). Pass title/note only for a standalone figure.
    if title:
        ax.set_title(title, fontsize=12, loc="left")
    if note:
        fig.text(0.02, 0.015, note, ha="left", va="bottom", fontsize=7.5, color="#999")
    return fig


def draw_zone_regions(ax, P: np.ndarray, countries: list[str], zones: dict[str, list[str]],
                      cloud_P: np.ndarray | None = None, cloud_countries: list[str] | None = None,
                      sigma: float = 1.0) -> None:
    """Each zone is the UNION (one merged shape, uniform alpha -- not stacked translucent discs) of a
    per-country region: that country's real spread ellipse when a cloud is given, else a fixed disc.
    Every country therefore sits inside its own zone territory, a 2-country zone still has an area,
    and clustered members merge into one blob. P is countries x 2; `cloud_P`/`cloud_countries` are the
    projected respondent/haze cloud and its per-row country. Shared by the instrument + WVS maps."""
    from shapely.ops import unary_union
    from matplotlib.patches import Polygon as MplPolygon
    cidx = {c: i for i, c in enumerate(countries)}
    r_fixed = 0.09 * float(np.hypot(*(P.max(0) - P.min(0))))   # ~50% bigger than the first pass
    by_country: dict[str, np.ndarray] = {}
    if cloud_P is not None and cloud_countries is not None:
        cc = np.asarray(cloud_countries)
        for c in set(cloud_countries):
            by_country[c] = cloud_P[cc == c]
    # Real per-country individual spread is far larger than the between-country dot spacing (within >>
    # between variance), so raw ellipses fill the plot. Rescale so the MEDIAN country ellipse ~ r_fixed
    # -- keeps each country's natural shape/orientation and relative size, at a legible overall scale.
    radii = [np.sqrt(max(np.linalg.eigvalsh(np.cov(pts.T)).mean(), 1e-12))
             for pts in by_country.values() if len(pts) >= 10]
    sigma *= (r_fixed / float(np.median(radii))) if radii else 1.0
    for zname, members in zones.items():
        idxs = [cidx[c] for c in members if c in cidx]
        if not idxs:
            continue
        regions = [_country_region(P[cidx[c]], by_country.get(c), sigma, r_fixed)
                   for c in members if c in cidx]
        union = unary_union(regions)
        geoms = list(union.geoms) if union.geom_type == "MultiPolygon" else [union]
        zcol = ZONE_COLORS.get(zname, "#888888")
        for g in geoms:
            ax.add_patch(MplPolygon(np.asarray(g.exterior.coords), closed=True, facecolor=zcol,
                         edgecolor=zcol, alpha=0.15, lw=1.1, zorder=1.5))
        cen = P[idxs].mean(0)
        # white halo so the zone-coloured label reads on top of the same-coloured blob (no contrast
        # otherwise). Darkened text + heavier stroke over the pale fill.
        import matplotlib.patheffects as pe
        ax.text(cen[0], cen[1], zname, fontsize=9.5, color=zcol, ha="center", va="center",
                style="italic", fontweight="bold", zorder=5,
                path_effects=[pe.withStroke(linewidth=3.0, foreground="white")])




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
    for x, y in L:
        cax.annotate("", xy=(x, y), xytext=(0, 0), arrowprops=dict(arrowstyle="->", color=color, lw=1.1))
    cax.set_xlim(-1.5, 1.5); cax.set_ylim(-1.5, 1.5)
    cax.set_aspect("equal")
    # A compass is a RADIAL layout, not a map -- the general placer's 'nearest clear slot' fights the
    # rose. Each factor name sits just beyond its own arrow tip, pushed straight OUT from the origin,
    # with outward ha/va so long names lean away from the centre. -- authored by Claude
    for (x, y), lab in zip(L, labels):
        r = np.hypot(x, y) or 1.0
        ux, uy = x / r, y / r
        ha = "left" if ux > 0.25 else "right" if ux < -0.25 else "center"
        va = "bottom" if uy > 0.25 else "top" if uy < -0.25 else "center"
        cax.text(x + ux * 0.16, y + uy * 0.16, lab.capitalize(), fontsize=7, color=color,
                 ha=ha, va=va, zorder=10)
    cax.axis("off")
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
                      emphasize: set[str] | None = None,
                      zones: dict[str, list[str]] | None = None, cloud_countries: list[str] | None = None,
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
    no backdrop. `traj` (signed c-multiplier -> length-K fraction vector) draws the coherent steer
    path through PC space. Public README plots pass only the coherent prefix; incoherent c values
    are omitted.
    `traj_incoherent` is the subset of those c whose admin pmass fell below the coherence floor --
    drawn hollow. `boots` optionally maps 'base'/'honest'/'dis' -> (n x K) bootstrap matrices.
    `zones` maps an Inglehart-Welzel zone name to the subset of `countries` (verbatim strings) in it;
    each zone with >=3 members gets a shaded convex hull (echoes the Economist WVS map's zone blobs),
    `emphasize` is a subset of
    `countries` labelled bold-first so named outliers (China, US, Sweden...) always survive the
    label-collision drop. `zones` maps an IW zone name to its member `countries`; each becomes a
    covariance ellipse over that zone's COUNTRY-MEAN points (between-country spread, so zones stay
    separate -- contouring individual respondents instead gives huge overlap since within-culture
    variance dominates). An eigenvalue floor gives a 1- or 2-country zone a visible blob. The PCA is
    fit on the country means M; `respondents`/`haze` only scatter + set the crop. Returns the
    Figure."""
    _, Vt, var, mu, Pc = ipsative_pca(M)               # fit on country means: between-country axes
    P = (M @ Pc - mu) @ Vt[:2].T
    # Anchor orientation geographically (West -> west, African-Islamic -> south) by folding the axis
    # signs into Vt itself, so EVERY downstream projection (dots, haze, steer path, compass loadings,
    # axis gloss) inherits the flip from one place and cannot mirror-diverge. -- authored by Claude
    sgx, sgy = orient_geographic(P, countries, zones)
    Vt = Vt.copy(); Vt[0] *= sgx; Vt[1] *= sgy
    P = (M @ Pc - mu) @ Vt[:2].T
    cloud = haze if haze is not None else respondents  # what we scatter + crop to (fit is separate)

    def proj(v):
        return ((v @ Pc) - mu) @ Vt[:2].T if v is not None else None
    pb, ph, pf = proj(base), proj(pos), proj(neg)

    fig, ax = plt.subplots(figsize=(8.5, 7.5))
    ax.set_facecolor("#faf8f2")
    ax.grid(True, color="#eceadf", lw=0.3, zorder=0)
    Pi = None
    if cloud is not None:                              # grey haze/respondents (rasterized; SVG-safe)
        Pi = (cloud @ Pc - mu) @ Vt[:2].T
        ax.scatter(Pi[:, 0], Pi[:, 1], s=4, c="#8f8a7e", alpha=0.14, edgecolors="none",
                   zorder=1, rasterized=True)
    # Same clean treatment as the WVS map (shared policy via _map_annotations): draw only the 4 zones
    # covering the most separate space, as edge-only hulls, and colour each dot by its drawn zone.
    sel_zones, dot_cols, label_set = _map_annotations(P, countries, zones, emphasize, C_HUM)
    zone_specs = draw_zone_hulls(ax, P, countries, sel_zones, label=False) if sel_zones else []
    ax.scatter(P[:, 0], P[:, 1], s=26, c=dot_cols, alpha=0.75, edgecolors="white", linewidths=0.5, zorder=3)
    # Labels go through labelplace.allocate_labels (the same placer as the WVS/value maps), so the
    # ipsative map reads alike: zone names hug their hull, society codes sit adjacent to their dot with
    # a short leader only when crowded, emphasize (landmark) countries are bold+dark. Collect the specs
    # here; PLACE them after the crop below, so the pixel-space allocator sees the final axis window.
    # Label only landmarks (emphasize) + 4 most-outlying + one central rep per drawn zone (de-clutter).
    emph = emphasize or set()
    mk_specs = [(P[i, 0], P[i, 1], countries[i],
                 "#111111" if countries[i] in emph else "#555555",
                 "bold" if countries[i] in emph else "normal",
                 8.5 if countries[i] in emph else 7.0, 5.0)     # (x,y,text,colour,weight,fontsize,pad)
                for i in range(len(countries)) if countries[i] in label_set]
    for key, col, pt in [("base", C_BASE, pb), ("honest", C_HON, ph), ("dis", C_DIS, pf)]:
        if boots and key in boots and pt is not None:
            bp = (np.asarray(boots[key]) @ Pc - mu) @ Vt[:2].T
            e1, e2 = 1.96 * bp.std(0)
            ax.errorbar(pt[0], pt[1], xerr=e1, yerr=e2, fmt="none", ecolor=col,
                        elinewidth=0.7, alpha=0.55, capsize=2.5, capthick=0.8, zorder=4)
    base_lab, pos_lab, neg_lab = labels
    pt_specs = list(mk_specs)                              # (x,y,text,colour,weight,fontsize,pad); placed post-crop
    if pb is not None:
        ax.scatter(*pb, s=72, c=C_BASE, marker="o", edgecolors="white", linewidths=1.0, zorder=7)
        pt_specs.append((pb[0], pb[1], base_lab, C_BASE, "bold", 9.0, 8.0))
    traj_pts = None
    if traj:
        inco = traj_incoherent or set()
        cs_sorted = sorted(traj)
        traj_pts = np.array([proj(traj[c]) for c in cs_sorted])
        # Two arms from base (c=0): +c red, -c blue. Marker size is constant so path length, not ink
        # area, carries the coefficient change.
        for lo, hi in [(0.0, max(cs_sorted)), (min(cs_sorted), 0.0)]:
            arm = [(c, proj(traj[c])) for c in cs_sorted if lo <= c <= hi]
            if len(arm) < 2:
                continue
            xy = np.array([p for _, p in arm])
            col = POS_COL if hi > 0 else NEG_COL
            ax.plot(xy[:, 0], xy[:, 1], "-", color=col, lw=1.1, zorder=4, alpha=0.75)
            c_end = max((c for c, _p in arm), key=abs)
            for c, p in arm:
                if c == 0:
                    continue
                fill = col if c == c_end and c not in inco else "none"
                ax.scatter(p[0], p[1], s=42, c=fill, edgecolors=col,
                           linewidths=1.25, zorder=6)
                if c == c_end:
                    lab = pos_lab if c > 0 else neg_lab
                    pt_specs.append((p[0], p[1], lab, col, "bold", 9.0, 8.0))
    else:
        for pt, col, lab in [(ph, C_HON, pos_lab), (pf, C_DIS, neg_lab)]:
            if pt is None:
                continue
            ax.scatter(*pt, s=42, c=col, marker="o", edgecolors="white", linewidths=1.0, zorder=7)
            pt_specs.append((pt[0], pt[1], lab, col, "bold", 9.0, 8.0))
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
        dview = (wx0 - 0.12 * sx, wx1 + 0.12 * sx, wy0 - 0.12 * sy, wy1 + 0.12 * sy)  # data frame, no legend pad
    else:
        x0, x1 = ax.get_xlim(); y0, y1 = ax.get_ylim()
        dview = (x0, x1, y0, y1); sy = y1 - y0
    ax.set_xlim(dview[0], dview[1])
    ax.set_ylim(dview[2] - PAD_B * sy, dview[3])
    # ONE placement pass, now that the axis window is final (see labelplace.allocate_labels). Zone names
    # are REGION labels hugging their hull; society codes + base/steer are MARKER labels. hard_pts = all
    # society dots (+ steer anchors); soft_pts = hull edges (only zone names dodge them).
    step = 0.02 * float(np.mean(P.max(0) - P.min(0)))
    zone_perims = [densify_polygon(coords, step) for _, coords, _ in zone_specs]
    soft_pts = np.vstack(zone_perims) if zone_perims else np.empty((0, 2))
    steer_pts = [p for p in (pb, ph, pf) if p is not None] + ([traj_pts] if traj_pts is not None else [])
    hard_pts = np.vstack([P] + [np.atleast_2d(p) for p in steer_pts]) if steer_pts else P
    anchor_sets = zone_perims + [np.array([[s[0], s[1]]]) for s in pt_specs]
    allocate_labels(
        ax, anchor_sets,
        [zn for zn, _, _ in zone_specs] + [s[2] for s in pt_specs],
        [zc for _, _, zc in zone_specs] + [s[3] for s in pt_specs],
        ["bold"] * len(zone_specs) + [s[4] for s in pt_specs], hard_pts, soft_pts=soft_pts,
        region=[True] * len(zone_specs) + [False] * len(pt_specs),
        fontsizes=[10.0] * len(zone_specs) + [s[5] for s in pt_specs],
        styles=["italic"] * len(zone_specs) + ["normal"] * len(pt_specs),
        anchor_pad=[3.0] * len(zone_specs) + [s[6] for s in pt_specs])
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

def draw_steer(ax, xs: float, cs: list[float], yv: np.ndarray, lw: float = 2.0,
               lane_dx: float = 0.14) -> None:
    """Draw one coherent AI c-path. X is only the sign lane: negative, base, positive."""
    assert list(cs) == sorted(cs) and 0.0 in cs, f"draw_steer needs sorted cs with c=0 (yv[-1]=+pole, yv[0]=-pole), got {cs}"
    xs_by_c = {c: xs + lane_dx * np.sign(c) for c in cs}
    for side_cs, col in [([c for c in cs if c <= 0.0], NEG_COL), ([c for c in cs if c >= 0.0], POS_COL)]:
        if len(side_cs) < 2:
            continue
        x_path = [xs_by_c[c] for c in side_cs]
        y_path = [float(yv[cs.index(c)]) for c in side_cs]
        ax.plot(x_path, y_path, color=col, lw=lw, alpha=0.9, zorder=6, solid_capstyle="round")
    for c, y in zip(cs, yv):
        if c == 0.0:
            col = "black"
        else:
            col = POS_COL if c > 0 else NEG_COL
        x = xs_by_c[c]
        ax.scatter(x, float(y), s=18, color=col, edgecolors="white", linewidths=0.45, zorder=8)


def draw_range_panel(ax, instr: Instrument, dims: list[str], cs: list[float], prof: dict,
                     humans: dict, cloud: np.ndarray | None, vec: str) -> tuple[float, float]:
    """One vector's range panel: human society dots plus one AI steer column per factor."""
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
        draw_steer(ax, xs, cs, yv)
        if i == label_i:
            # Only the two pole labels, to the RIGHT of the steer column. No 'base' tag: the black dot
            # between the two coloured arms is self-evidently the unsteered model, and on a near-collapsed
            # pole (e.g. humor affiliative, +c ~ base) a 'base' tag overprints the +c tag.
            for c_end, y_end, col in [(cs[-1], yv[-1], POS_COL), (cs[0], yv[0], NEG_COL)]:
                ax.annotate(f"c={c_end:+g}", (xs + 0.30, y_end), fontsize=6.8,
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
        # base (c=0) is always finite, so the axis still frames the un-collapsed cells.
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
        draw_steer(ax, xs, cs, yv, lw=2.4)
        named = {}
        if near:
            name_xy = {nm: (float(x), v) for (nm, v), x in zip(near, soc_x)}
            for ref in (base_y, float(yv.max()), float(yv.min())):
                named[min(near, key=lambda t: abs(t[1] - ref))[0]] = ref
        tx = [xs] * len(cs) + [name_xy[nm][0] for nm in named] if near else [xs] * len(cs)
        ty = list(map(float, yv)) + [name_xy[nm][1] for nm in named] if near else list(map(float, yv))
        txt = [("c=0" if c == 0 else f"c={c:+g}") for c in cs] + list(named)
        dot_x = [xs] * len(cs) + (list(soc_x) if near else [])
        dot_y = list(map(float, yv)) + ([v for _, v in near] if near else [])
        allocate_labels(ax, [np.array([[x, y]]) for x, y in zip(tx, ty)], txt,
                        ["#333333"] * len(txt), ["normal"] * len(txt),
                        np.column_stack([dot_x, dot_y]) if dot_x else np.empty((0, 2)),
                        fontsize=6.5, anchor_pad=[4.0] * len(txt), linecolor="#bbbbbb", linewidth=0.4)
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
