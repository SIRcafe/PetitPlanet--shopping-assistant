"""Walkability in front of the character, for going *around* objects.

The game makes every prop solid, so any object standing on the path stops the
character.  We cannot name the object (no model, and nothing to train on), but
we can measure the thing that matters: **is the ground in that direction still
the ground?**

Reference: the colour of the ground next to the feet (sampled left and right of
the character, so its own silhouette is not used).  A direction is walkable when
its lower half is mostly that colour, is not water, and has no hard vertical
structure standing in it.  A prop in front shows up as a patch that differs from
the ground, with crisp edges, usually sitting in the middle of the screen.

`profile()` returns one score per angular bin, left to right; the caller then
nudges the heading toward the freer side while keeping the blue dot as the goal.
"""

from __future__ import annotations

import numpy as np

from . import imgproc as ip
from . import spec

GROUND_TOL = 0.10          # chroma distance (see _bin_score) that still is ground
NEAR_Y = (0.60, 0.88)      # band above the feet: what is in front of us
FEET_Y = (0.88, 0.96)      # band next to the feet: the walkable reference
FEET_X = ((0.26, 0.38), (0.62, 0.74))
WATER_VETO = 0.55          # strict water coverage that makes a bin unusable
MIN_FREE = 0.40            # below this a bin counts as blocked
SIDE_FREE = 0.55           # a neighbour must be this free to steer toward it
NUDGE_PX = 150             # how hard to angle around an obstacle


def _strip(img: np.ndarray, x0: float, x1: float, y0: float, y1: float,
           width: int = 40) -> np.ndarray:
    h, w = img.shape[:2]
    crop = img[int(y0 * h):int(y1 * h), int(x0 * w):int(x1 * w)]
    if crop.size == 0:
        return crop
    if crop.shape[1] > width:
        crop = ip.resize_bilinear(crop, width, max(4, int(crop.shape[0] * width / crop.shape[1])))
    return crop


def ground_reference(img: np.ndarray) -> np.ndarray:
    """Median colour of the ground beside the character's feet."""
    samples = [_strip(img, x0, x1, FEET_Y[0], FEET_Y[1]).reshape(-1, 3)
               for x0, x1 in FEET_X]
    flat = np.concatenate([s for s in samples if s.size], axis=0)
    if flat.size == 0:
        return np.array((128, 128, 128), np.float32)
    return np.median(flat.astype(np.float32), axis=0)


def _chroma(rgb: np.ndarray) -> np.ndarray:
    """Normalised colour: r/(r+g+b) style, so shading keeps the same value.

    Comparing raw RGB against a single reference fails as soon as the ground is
    lit unevenly (a plaza at night is blue-ish, grass in sun is bright), which
    made every bin look blocked.  Dividing by the sum removes brightness, so
    only *material* differences remain.
    """
    total = rgb.sum(axis=-1, keepdims=True)
    return rgb / np.maximum(total, 1.0)


def _bin_score(crop: np.ndarray, ref: np.ndarray) -> dict:
    if crop.size == 0:
        return dict(free=0.0, ground=0.0, water=0.0, edge=0.0)
    rgb = crop.astype(np.float32)
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    sat = rgb.max(axis=2) - rgb.min(axis=2)
    ref_chroma = _chroma(ref.reshape(1, 1, 3))[0, 0]
    dist = np.abs(_chroma(rgb) - ref_chroma[None, None, :]).max(axis=2)
    ground = float((dist <= GROUND_TOL).mean())
    # strict water: a saturated blue that is definitely not the ground colour
    water = float(((b > r + 35) & (b > g + 12) & (sat > 45)
                   & (dist > GROUND_TOL)).mean())
    gray = ip.to_gray(crop)
    edge = float(np.abs(np.diff(gray, axis=1)).mean()) if gray.shape[1] > 1 else 0.0
    free = ground
    if water > WATER_VETO:
        free *= 1.0 - water
    return dict(free=float(np.clip(free, 0.0, 1.0)), ground=ground, water=water, edge=edge)


def profile(img: np.ndarray, bins: int = 9, span: tuple[float, float] = (0.16, 0.84)) -> list[dict]:
    """Left-to-right walkability bins across the lower-middle of the frame."""
    if img.ndim != 3 or img.shape[0] < 40 or img.shape[1] < 40:
        return []
    ref = ground_reference(img)
    width = (span[1] - span[0]) / bins
    out = []
    for i in range(bins):
        x0 = span[0] + i * width
        x1 = x0 + width
        crop = _strip(img, x0, x1, NEAR_Y[0], NEAR_Y[1])
        score = _bin_score(crop, ref)
        score.update(cx=(x0 + x1) / 2.0 * spec.DESIGN_W)
        out.append(score)
    return out


def center_index(bins: int = 9) -> int:
    return bins // 2


def steer(prof: list[dict], dot_dx: float | None = None,
          min_free: float = MIN_FREE) -> dict:
    """Suggest a camera nudge when something stands in the path.

    Returns {'nudge': px, 'why': str, 'frees': [...]}.  `nudge` uses the same
    units as the dot controller (positive = turn right) so it can simply be
    added to the dot term.
    """
    if not prof:
        return dict(nudge=0, why="no profile", frees=[])
    n = len(prof)
    c = center_index(n)
    frees = [p["free"] for p in prof]
    if frees[c] >= min_free:
        return dict(nudge=0, why=f"centre clear ({frees[c]:.2f})", frees=frees)

    # nearest clearly-free neighbour, preferring the side the dot is on
    def rank(item):
        dist, i = item
        on_target_side = dot_dx is None or i_side(i, c, dot_dx)
        return (dist, 0 if on_target_side else 1)

    order = sorted(((abs(i - c), i) for i in range(n) if i != c), key=rank)
    for _dist, i in order:
        if frees[i] >= SIDE_FREE:
            side = 1 if i > c else -1
            return dict(nudge=side * NUDGE_PX,
                        why=f"centre blocked ({frees[c]:.2f}) -> bin {i} "
                            f"free ({frees[i]:.2f})",
                        frees=frees)
    return dict(nudge=0, why=f"everything looks blocked (min {min(frees):.2f})",
                frees=frees)


def i_side(i: int, c: int, dot_dx: float) -> bool:
    """True when bin i lies on the same side as the target."""
    return (i - c) * dot_dx >= 0
