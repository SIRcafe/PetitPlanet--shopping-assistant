"""Stuck diagnosis: what is the character pressed against, and how to get out.

The previous rule was a blind "turn right 100 deg, walk 2 s", and it is kept as
the ``unknown`` fallback.  This module splits that rule into three questions:

    features(img) -> numeric description of the band in front of the feet
    classify(f)   -> wall | tree | water | railing | step | open | unknown
    plan(kind)    -> an ordered list of actions to execute

Pure numpy, no models, no network.  Labels are learned offline (the vision
tools act as the annotator on frames collected with `scripts/collect_stuck.py`)
and frozen into the weights below, so the runtime stays local.

An action is one of::

    {"do": "back",  "ms": 350}      # S for a moment, to un-stick the collision
    {"do": "turn",  "px": 2222}     # chunked by the caller (see Runner.turn_px)
    {"do": "walk",  "ms": 2000}     # hold W
    {"do": "jump"}                  # Space

Calibration status (2026-10-18 night): weights are a first guess from the
existing fixtures; `scripts/escape_probe.py` compares them against labelled
frames, and only after that comparison are they to be trusted.
"""

from __future__ import annotations

import numpy as np

from . import imgproc as ip
from . import spec


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _up(v: float, lo: float, hi: float) -> float:
    """Ramp 0->1 as v goes lo->hi (lo > hi mirrors it, i.e. a falling ramp)."""
    if hi == lo:
        return 1.0 if v >= hi else 0.0
    return float(np.clip((v - lo) / (hi - lo), 0.0, 1.0))


def _bell(v: float, lo: float, hi: float, soft: float = 0.15) -> float:
    """1 inside [lo, hi], tapering to 0 over `soft` outside it."""
    if lo <= v <= hi:
        return 1.0
    d = (lo - v) if v < lo else (v - hi)
    return float(max(0.0, 1.0 - d / max(1e-6, soft)))


def _crop(img: np.ndarray, roi: dict) -> np.ndarray:
    h, w = img.shape[:2]
    return img[int(roi["y0"] * h):int(roi["y1"] * h),
               int(roi["x0"] * w):int(roi["x1"] * w)]


# --------------------------------------------------------------------------- #
# features
# --------------------------------------------------------------------------- #
def front_features(img: np.ndarray, roi: dict | None = None,
                   char_box="default", width: int = 320) -> dict:
    """Colour / texture statistics of the ground band in front of the character.

    The character's own silhouette is masked out: clothes and hair are the
    loudest colours in the frame and would otherwise dominate every fraction
    (a white shirt reads as a "neutral wall" all by itself).  Pass
    ``char_box=False`` to keep it (used by the tests).
    """
    roi = roi or spec.STUCK_ROI
    if char_box == "default":
        char_box = spec.STUCK_CHAR_BOX
    crop = _crop(img, roi)
    if crop.size == 0 or crop.shape[0] < 8 or crop.shape[1] < 8:
        return {}

    ch, cw = crop.shape[:2]
    if cw > width:                                    # 320 px is plenty and fast
        crop = ip.resize_bilinear(crop, width, max(8, int(round(ch * width / cw))))
    ch, cw = crop.shape[:2]

    rgb = crop.astype(np.float32)
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    gray = ip.to_gray(crop)
    sat = rgb.max(axis=2) - rgb.min(axis=2)

    keep = np.ones((ch, cw), bool)
    if char_box:
        span_x = max(1e-6, roi["x1"] - roi["x0"])
        span_y = max(1e-6, roi["y1"] - roi["y0"])
        cx0 = int((char_box[0] - roi["x0"]) / span_x * cw)
        cx1 = int((char_box[2] - roi["x0"]) / span_x * cw)
        cy0 = int((char_box[1] - roi["y0"]) / span_y * ch)
        cy1 = int((char_box[3] - roi["y0"]) / span_y * ch)
        keep[max(0, cy0):max(0, cy1), max(0, cx0):max(0, cx1)] = False
    if keep.sum() < 400:                              # degenerate box -> ignore it
        keep[:] = True
    n = float(keep.sum())

    water = ((b > r + 18) & (b > g + 6) & (sat > 22)) & keep
    foliage = ((g > r + 12) & (g > b + 4) & (sat > 25)) & keep
    warm = ((r > g + 8) & (g > b) & (sat > 30)) & keep
    rail = ip.color_mask(crop, spec.RAIL_BLUE, spec.STUCK_RAIL_TOL) & keep
    # a railing is thin vertical bars, water is a broad field: keep only the
    # rail pixels that sit in columns which are *mostly* rail (bar columns)
    col_keep = keep.sum(axis=0)
    col_rail = rail.sum(axis=0) / np.maximum(1, col_keep)
    bar_cols = (col_rail >= 0.55) & (col_keep > 0)
    rail_bar = float((rail & bar_cols[None, :]).sum() / n)
    neutral = (sat < 22) & keep
    dark = (gray < 65) & keep
    bright = (gray > 190) & keep

    # gradients over valid neighbour pairs only, so masked pixels do not leak
    gy_abs = np.abs(np.diff(gray, axis=0))
    gx_abs = np.abs(np.diff(gray, axis=1))
    ky = keep[1:] & keep[:-1]
    kx = keep[:, 1:] & keep[:, :-1]
    edge_y = float((gy_abs * ky).sum() / max(1.0, ky.sum()))
    edge_x = float((gx_abs * kx).sum() / max(1.0, kx.sum()))
    row_edge = (gy_abs * ky).sum(axis=1) / np.maximum(1.0, ky.sum(axis=1))

    # high-frequency energy: a canopy / hedge / gravel is detailed, water and
    # painted walls are not  (4-neighbour Laplacian, masked)
    lap = np.abs(4.0 * gray[1:-1, 1:-1] - gray[:-2, 1:-1] - gray[2:, 1:-1]
                 - gray[1:-1, :-2] - gray[1:-1, 2:]) if ch > 2 and cw > 2 else np.zeros((1, 1))
    kc = keep[1:-1, 1:-1] if ch > 2 and cw > 2 else np.ones((1, 1), bool)
    lap_energy = float((lap * kc).sum() / max(1.0, kc.sum()))

    # upper 40% of the band = what is *in front of the face*, not the ground
    up_rows = max(1, int(ch * 0.40))
    up_keep = keep[:up_rows]
    upper_foliage = float((foliage[:up_rows]).sum() / max(1.0, up_keep.sum()))
    upper_water = float((water[:up_rows]).sum() / max(1.0, up_keep.sum()))

    # the strongest horizontal edge line (curb / step / railing top) and where
    lo = int(len(row_edge) * 0.25)
    hi = int(len(row_edge) * 0.92)
    if hi <= lo:
        edge_row_max, edge_row_pos = 0.0, 0.0
    else:
        seg = row_edge[lo:hi]
        k = int(np.argmax(seg))
        edge_row_max = float(seg[k])
        edge_row_pos = float((lo + k) / max(1, len(row_edge) - 1))

    flat = crop.reshape(-1, 3)[keep.ravel()]
    med = np.median(flat, axis=0)
    uniform = float((np.abs(flat - med).max(axis=1) <= 14).mean())

    return dict(
        n_px=int(n),
        water=float(water.sum() / n),
        foliage=float(foliage.sum() / n),
        warm=float(warm.sum() / n),
        rail=float(rail.sum() / n),
        rail_bar=rail_bar,
        neutral=float(neutral.sum() / n),
        dark=float(dark.sum() / n),
        bright=float(bright.sum() / n),
        uniform=uniform,
        gray_mean=float(gray.mean()),
        gray_std=float(gray.std()),
        sat_mean=float(sat.mean()),
        edge_x=edge_x,
        edge_y=edge_y,
        lap=lap_energy,
        upper_foliage=upper_foliage,
        upper_water=upper_water,
        edge_row_max=edge_row_max,
        edge_row_pos=edge_row_pos,
    )


# --------------------------------------------------------------------------- #
# classification
# --------------------------------------------------------------------------- #
def scores(f: dict) -> dict[str, float]:
    """How much the features look like each kind of obstacle (0..1 each).

    Design notes:
      * water and rialing share almost the same blue, so the railing score is
        suppressed when the blue is a broad flat field (water covers the band,
        a railing is thin bars);
      * a wall is one colour that does not change from row to row, and it is
        smooth;
      * a step is a strong horizontal edge line inside the band;
      * "open" means nothing specific - the block is beside us, probe with a
        small angle.
    """
    empty = dict(wall=0.0, tree=0.0, water=0.0, railing=0.0, step=0.0, open=0.0)
    if not f:
        return empty

    water = _up(f["water"], 0.15, 0.42)
    tree = _up(f["upper_foliage"], 0.24, 0.52) * _up(f["lap"], 5.0, 12.0)
    # a wall is one colour, identical on every row, smooth ... and not blue,
    # not foliage (otherwise a lake or a lawn reads as a painted wall)
    wall = (_up(f["uniform"], 0.45, 0.75)
            * _up(1.0 - min(1.0, f["edge_y"] / 26.0), 0.40, 0.75)
            * (1.0 - _up(f["water"], 0.30, 0.60))
            * (1.0 - _up(f["foliage"], 0.40, 0.75)))
    railing = _up(f["rail_bar"], 0.015, 0.050)
    step = (_up(f["edge_row_max"], 12.0, 26.0)
            * _bell(f["edge_row_pos"], 0.30, 0.82)
            * _up(f["uniform"], 0.28, 0.52))
    open_ = (_up(f["lap"], 12.0, 30.0)
             * _up(0.45 - f["uniform"], 0.0, 0.30)
             * (1.0 - _up(f["upper_foliage"], 0.20, 0.50)))
    return dict(wall=wall, tree=tree, water=water, railing=railing, step=step, open=open_)


def classify(f: dict) -> dict:
    """Pick the most likely obstacle kind.

    `rail` is a single very specific cue (a fixed brand colour, RAIL_BLUE), so
    it wins over the argmax whenever it is present and the blue is not a broad
    water field; everything else is a plain argmax with a confidence floor.
    """
    s = scores(f)
    order = sorted(s.items(), key=lambda kv: kv[1], reverse=True)
    kind, best = order[0]
    second = order[1][1] if len(order) > 1 else 0.0

    if f and f["rail_bar"] >= spec.STUCK_RAIL_MIN and f["water"] < 0.30:
        return dict(kind="railing", confidence=float(s["railing"]),
                    margin=float(s["railing"] - second), scores=s, rule="rail-priority")
    # a flat black or white frame is a lost-focus / loading screen, not a wall
    # (HANDOFF pit 8: screenshot mean ~248 while the game is unfocused)
    if f and f["gray_std"] < 5.0 and (f["gray_mean"] < 14.0 or f["gray_mean"] > 242.0):
        return dict(kind="unknown", confidence=0.0, margin=0.0, scores=s, rule="blank-frame")
    if best < spec.STUCK_CONF_MIN:
        return dict(kind="unknown", confidence=float(best), margin=float(best - second),
                    scores=s, rule="below-floor")
    return dict(kind=kind, confidence=float(best), margin=float(best - second),
                scores=s, rule="argmax")


# --------------------------------------------------------------------------- #
# escape plans
# --------------------------------------------------------------------------- #
# 2222 px == ~100 deg, 1000 px == ~45 deg, 3000 px == ~135 deg (see HANDOFF 2.6)
PLANS: dict[str, list[dict]] = {
    # a building/rock face is long: follow it - back off, turn, walk along it
    "wall": [{"do": "back", "ms": 350}, {"do": "turn", "px": 2222},
             {"do": "walk", "ms": 2000}],
    # a tree trunk is narrow: a small turn clears it
    "tree": [{"do": "back", "ms": 300}, {"do": "turn", "px": 1000},
             {"do": "walk", "ms": 1500}],
    # water has to be walked around, not through: sharper turn, short step
    "water": [{"do": "back", "ms": 250}, {"do": "turn", "px": 3000},
              {"do": "walk", "ms": 1600}],
    # railing: never zig-zag into it - turn and go along
    "railing": [{"do": "back", "ms": 300}, {"do": "turn", "px": 1000},
                {"do": "walk", "ms": 1600}],
    # a step/curb needs a hop, then commit forward
    "step": [{"do": "back", "ms": 200}, {"do": "jump"}, {"do": "walk", "ms": 700}],
    # nothing in front -> the block is beside us; probe with a small angle
    "open": [{"do": "turn", "px": 1000}, {"do": "walk", "ms": 1500}],
    # the validated field rule (HANDOFF 2.7)
    "unknown": [{"do": "turn", "px": 2222}, {"do": "walk", "ms": 2000}],
}


def plan(kind: str, mirrored: bool = False) -> list[dict]:
    """Actions for one escape attempt; `mirrored` flips every turn (try left)."""
    out = [dict(a) for a in PLANS.get(kind, PLANS["unknown"])]
    if mirrored:
        for a in out:
            if a["do"] == "turn":
                a["px"] = -a["px"]
    return out


def kinds() -> tuple[str, ...]:
    return tuple(PLANS)


def describe(f: dict) -> str:
    """One-line feature summary for the log."""
    if not f:
        return "(no features)"
    return (f"water={f['water']:.2f} tree={f['foliage']:.2f}(up {f['upper_foliage']:.2f}) "
            f"rail={f['rail']:.3f}(bar {f['rail_bar']:.3f}) unif={f['uniform']:.2f} "
            f"std={f['gray_std']:.0f} lap={f['lap']:.1f} eX={f['edge_x']:.1f} "
            f"eY={f['edge_y']:.1f} row={f['edge_row_max']:.1f}@{f['edge_row_pos']:.2f}")
