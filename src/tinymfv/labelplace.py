"""General candidate-slot label placement for matplotlib: polygon-aware, short-leader, gist-ready.

The problem: matplotlib has no label placer that (a) tries every side of a marker and keeps the
nearest clear slot, (b) draws a leader line ONLY when the label had to move far, and (c) treats a
filled polygon (a convex-hull region) as something to avoid. adjustText (Phlya/adjustText) relaxes
by force and parks labels in local minima; textalloc (ckjellson/textalloc) does candidate placement
but only against points/lines/other-text, and always/never draws lines. This is a small placer that
does all three, generalised so ONE call handles both marker labels and region labels.

The generalisation (wassname's idea): every label owns a SET of 1..N candidate anchor points, and we
search placements around all of them.
  - a MARKER label (country / model dot) passes its single point -> the box sits adjacent to it.
  - a REGION label (a zone name over a convex hull) passes its whole densified perimeter -> the box
    can attach ANYWHERE along the hull edge, so it has tons of options and never needs to overlap.

Two obstacle classes:
  - HARD points (markers, and every already-placed label box): no label may cover these.
  - SOFT points (polygon edges, via densify_polygon): only REGION labels avoid these. A marker label
    wears a thin white outline, so it may cross a hull line and stay readable (cheaper than contorting
    every country label around the zone boundaries).

Selection differs by label kind, which is the whole point of the 1..N anchor-set framing:
  - region labels MAXIMISE clearance over all (perimeter-anchor x slot) candidates -> the emptiest arc
    of their own hull, in the open, no white box and no leader.
  - marker labels take the NEAREST clear slot (adjacent reads as attached), with a ~half-character gap
    from every obstacle and a leader line only when the slot is far or contested.

Runs in PIXEL space (measures real rendered text extents), so call it AFTER the axes are at their
final limits and orientation -- e.g. after ax.invert_xaxis() -- otherwise the 'try every side'
geometry is mirrored and every label drifts one way.

Principles (Tufte, verified by reading each rendered PNG -- code/SVG alone can't judge a plot):
  - Direct labels over legends: a swatch legend duplicating on-plot labels fails the eraser test; drop
    it. Colour + a placed name carry the identity.
  - A leader line is a FALLBACK, not decoration: draw one only when a label had to move far. A good
    adjacent placement needs none, so most labels have no line.
  - Nearest clear slot, not farthest empty space: labels hug their marker / hull so the eye pairs them
    without tracing. (Maximising clearance sends a label fleeing to the void -- the opposite of what you
    want.)
  - Anisotropic spacing: text stacks tighter vertically than horizontally, so pull labels in more on y
    than x. Keep a ~half-character gap from every obstacle for legibility.
  - Collision test on the real box: promote polygon PATHS to sampled points (densify_polygon) so a wide
    label box actually feels a hull edge it would cross, not just the corners.

Adapted from textalloc (ckjellson/textalloc, MIT) and wassname's plotly placer
(gist b0b34492cd1679f1daeb5892ef714dce). -- authored by Claude
"""
from __future__ import annotations

import numpy as np
import matplotlib.patheffects as pe


def densify_polygon(coords: np.ndarray, step: float) -> np.ndarray:
    """A convex hull is stored as ~6 CORNER vertices; the long straight edges between them carry no
    points, so a label can sit ON an edge and 'see' nothing to dodge. Sample points every `step` data
    units ALONG each closed edge, promoting the polygon PATH (not just its corners) to a point cloud
    the box-collision test can feel."""
    coords = np.asarray(coords, float)
    pts = []
    for i in range(len(coords)):
        a, b = coords[i], coords[(i + 1) % len(coords)]
        n = max(2, int(np.hypot(*(b - a)) / step) + 1)
        pts.extend(a + t * (b - a) for t in np.linspace(0, 1, n, endpoint=False))
    return np.array(pts) if pts else np.empty((0, 2))


def _box_metrics(box, pts):
    """(count of pts inside box, distance from box to nearest pt) for a padded AABB and a point cloud."""
    if not len(pts):
        return 0, np.inf
    x0, y0, x1, y1 = box
    dx = np.maximum(0.0, np.maximum(x0 - pts[:, 0], pts[:, 0] - x1))
    dy = np.maximum(0.0, np.maximum(y0 - pts[:, 1], pts[:, 1] - y1))
    d = np.hypot(dx, dy)
    return int(np.count_nonzero(d == 0.0)), float(d.min())


# candidate directions in priority order: right, left, under, up (horizontal reads best, 'under'
# before 'over'), then the four diagonals. y is UP in matplotlib display space.
_ANGLES = np.deg2rad([0, 180, 270, 90, 315, 225, 45, 135])
_DIRS = np.column_stack([np.cos(_ANGLES), np.sin(_ANGLES)])


def allocate_labels(ax, anchor_sets: list[np.ndarray], texts: list[str], colors: list[str],
                    weights: list[str], hard_pts: np.ndarray, *, soft_pts: np.ndarray | None = None,
                    region: list[bool] | None = None, fontsize: float = 9.0,
                    fontsizes: list[float] | None = None, styles: list[str] | None = None,
                    anchor_pad: list[float] | None = None, gap_frac: float = 0.28,
                    spacing_x: float = 0.4, spacing_y: float = 0.3, edge_pad: float = 4.0,
                    stroke: float = 2.0, linecolor: str = "#9a958a", linewidth: float = 0.6):
    """Place N labels. See the module docstring for the model. Draws directly onto `ax`.

    anchor_sets : per label, an (Ki, 2) array of candidate attachment points (data coords).
    hard_pts    : (M, 2) markers no label may cover; placed label boxes are added to this as we go.
    soft_pts    : (P, 2) polygon-edge points; only `region` labels avoid them.
    region[i]   : True  -> multi-anchor, nearest CLEAR ring (hugs the hull), avoid soft points, no leader
                  False -> nearest clear slot, hard points only, thin white outline, leader if far.
    anchor_pad[i]: px radius of label i's own marker, so the box clears a big star as well as the gap.
    gap_frac    : gap kept from every obstacle, as a fraction of text height (~half a character).
    spacing_x/_y: scale the label<->own-marker spacing beyond the marker (anisotropic: text stacks
                  tighter vertically than horizontally, so y is pulled in more than x).
    """
    n = len(texts)
    region = region or [False] * n
    fs = fontsizes or [fontsize] * n
    st = styles or ["normal"] * n
    pad0 = anchor_pad or [4.0] * n
    fig = ax.figure
    fig.canvas.draw()                                        # freeze limits + get a live renderer
    rend = fig.canvas.get_renderer()
    to_px = ax.transData.transform
    to_data = ax.transData.inverted().transform
    A_px = [to_px(np.asarray(a, float).reshape(-1, 2)) for a in anchor_sets]
    hard = to_px(np.asarray(hard_pts, float)) if len(hard_pts) else np.empty((0, 2))
    soft = to_px(np.asarray(soft_pts, float)) if (soft_pts is not None and len(soft_pts)) else np.empty((0, 2))
    abox = ax.get_window_extent()
    wh = []                                                  # measured (w, h) px per label
    for t, w, s, z in zip(texts, weights, st, fs):
        h = ax.text(0, 0, t, fontsize=z, fontweight=w, fontstyle=s, ha="left", va="bottom")
        e = h.get_window_extent(rend); wh.append((e.width, e.height)); h.remove()
    placed = []                                              # settled label boxes -> hard obstacles
    order = sorted(range(n), key=lambda i: not region[i])    # region labels first, so markers dodge them
    for i in order:
        w_i, h_i = wh[i]
        gap = gap_frac * h_i                                 # ~half a character clear of every obstacle
        obstacles = np.vstack([hard, soft]) if region[i] and len(soft) else hard
        # reach = extra spacing rings (in text-heights) tried NEAREST-first, so a label hugs its marker /
        # hull. Spacing beyond the marker is anisotropic (spacing_x/_y): wider left-right than up-down,
        # since stacked text crowds vertically. pad0 (marker radius) is NOT scaled, so nothing lands on
        # its own glyph.
        reach = (0.0, 0.7, 1.4, 2.2, 3.0) if region[i] else (0.0, 0.9, 1.8, 2.8, 4.0)
        best = None                                          # global fallback: lowest penalty seen
        pick = None                                          # accepted clear slot: (box, anchor, k)
        for k in reach:
            ring = None                                      # best clear candidate at THIS ring
            for anc in A_px[i]:
                ax0, ay0 = anc
                ext = gap + k * h_i
                rx, ry = pad0[i] + spacing_x * ext, pad0[i] + spacing_y * ext
                for ux, uy in _DIRS:
                    cx, cy = ax0 + ux * (rx + w_i / 2), ay0 + uy * (ry + h_i / 2)
                    box = (cx - w_i / 2 - gap, cy - h_i / 2 - gap, cx + w_i / 2 + gap, cy + h_i / 2 + gap)
                    pen = 0.0
                    if (box[0] < abox.x0 + edge_pad or box[2] > abox.x1 - edge_pad or
                            box[1] < abox.y0 + edge_pad or box[3] > abox.y1 - edge_pad):
                        pen += 1000.0                        # off-canvas / flush-to-frame: last resort
                    inside, clear = _box_metrics(box, obstacles)
                    pen += 50.0 * inside
                    for pb in placed:                        # overlap area with settled labels
                        ox = max(0.0, min(box[2], pb[2]) - max(box[0], pb[0]))
                        oy = max(0.0, min(box[3], pb[3]) - max(box[1], pb[1]))
                        pen += 0.02 * ox * oy
                    if best is None or (pen, -clear) < best[0]:
                        best = ((pen, -clear), box, (ax0, ay0), k)
                    if pen == 0.0:
                        if not region[i]:                    # marker: first clear slot (nearest) wins
                            ring = (box, (ax0, ay0), k, clear); break
                        if ring is None or clear > ring[3]:  # region: emptiest slot on this NEAREST ring
                            ring = (box, (ax0, ay0), k, clear)
                if ring is not None and not region[i]:
                    break
            if ring is not None:
                pick = ring; break                           # nearest clear ring wins -> label hugs marker/hull
        box, (ax0, ay0), k = (pick[0], pick[1], pick[2]) if pick else (best[1], best[2], best[3])
        placed.append(box)
        cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
        # leader line: marker labels only, when the slot is far or contested. Same-colour (model/steer)
        # labels get a looser threshold since their colour already ties them to the marker.
        if not region[i]:
            thr = 2.6 if colors[i] != "#111" else 1.15       # in text-heights of reach
            if k > thr or pick is None:
                nx, ny = min(max(ax0, box[0]), box[2]), min(max(ay0, box[1]), box[3])
                (lx0, ly0), (lx1, ly1) = to_data((ax0, ay0)), to_data((nx, ny))
                ax.plot([lx0, lx1], [ly0, ly1], "-", color=linecolor, lw=linewidth, zorder=2.5)
        dx, dy = to_data((cx, cy))
        ax.text(dx, dy, texts[i], color=colors[i], fontsize=fs[i], fontweight=weights[i], fontstyle=st[i],
                ha="center", va="center", zorder=10,
                path_effects=[pe.withStroke(linewidth=stroke, foreground="white")])
