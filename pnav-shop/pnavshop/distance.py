r"""Read the "N米" distance under the blue navigation dot.

Why bother: the user's rule (2026-10-18 night) is that **a distance that does
not change means we are stuck**.  Comparing bitmap signatures (see
`detect.distance_signature`) cannot tell "42米 -> 41米" from "42米 -> 42米" once
the glyphs are slightly re-rendered, and it cannot tell us *how much* closer we
got.  Reading the integer converts navigation into an objective we can check
after every step.

The digits are recognised with the same template bank as the UID (`uid.py`),
so this stays local, offline and free: segment the near-white glyphs under the
dot, drop the trailing 「米」, normalise each glyph to 16x24 and run the bank.

Usage:
    bank = uid.load_bank(spec.TEMPLATE_DIR)
    info = distance.read(frame, dot, bank)     # {'meters': 42, ...}
"""

from __future__ import annotations

import numpy as np

from . import imgproc as ip
from . import uid

DIST_RGB = (240, 238, 225)      # the distance text is near-white
DIST_TOL = 45
BAND_DX = 80                    # horizontal half-width around the dot
BAND_DY = (10, 70)              # below the dot centre
MIN_GLYPH_H = 10
MAX_GLYPH_H = 40


def band(img: np.ndarray, dot: dict, dx: int = BAND_DX,
         dy: tuple[int, int] = BAND_DY) -> np.ndarray:
    cx, cy = int(dot["cx"]), int(dot["cy"])
    y0 = max(0, cy + dy[0])
    y1 = min(img.shape[0], cy + dy[1])
    x0 = max(0, cx - dx)
    x1 = min(img.shape[1], cx + dx)
    return img[y0:y1, x0:x1]


def glyph_boxes(crop: np.ndarray, tol: int = DIST_TOL) -> list[dict]:
    """Left-to-right glyph boxes of the distance line (digits + 米)."""
    mask = ip.color_mask(crop, DIST_RGB, tol)
    comps = [c for c in ip.components(mask, min_area=10)
             if MIN_GLYPH_H <= c["h"] <= MAX_GLYPH_H and 2 <= c["w"] <= 48]
    if not comps:
        return []
    mid = sorted(c["y0"] + c["h"] / 2 for c in comps)[len(comps) // 2]
    row = [c for c in comps if abs(c["y0"] + c["h"] / 2 - mid) <= max(8, mid * 0.35)]
    row.sort(key=lambda c: c["x0"])
    return row


def _normalize(crop: np.ndarray, box: dict, tol: int = DIST_TOL) -> np.ndarray:
    pad = 1
    y0 = max(0, box["y0"] - pad)
    x0 = max(0, box["x0"] - pad)
    y1 = min(crop.shape[0], box["y0"] + box["h"] + pad)
    x1 = min(crop.shape[1], box["x0"] + box["w"] + pad)
    patch = crop[y0:y1, x0:x1]
    mask = ip.color_mask(patch, DIST_RGB, tol).astype(np.float32)
    small = ip.resize_bilinear(np.dstack([mask * 255] * 3),
                               uid.GLYPH_W, uid.GLYPH_H)[..., 0]
    return (small / 255.0).astype(np.float32)


def digit_boxes(boxes: list[dict], max_ratio: float = 0.80) -> list[dict]:
    """Keep the leading run of digit-shaped glyphs (the 米 is wide)."""
    out: list[dict] = []
    for b in boxes:
        if b["w"] / max(1, b["h"]) > max_ratio:
            break
        out.append(b)
    return out


def read(img: np.ndarray, dot: dict | None, bank: dict, tol: int = DIST_TOL) -> dict:
    """{'meters': int|None, 'text': str, 'confidence': float, 'n_glyphs': int}"""
    empty = dict(meters=None, text="", confidence=0.0, n_glyphs=0)
    if not dot or not dot.get("found") or not bank:
        return empty
    crop = band(img, dot)
    if crop.size == 0:
        return empty
    boxes = glyph_boxes(crop, tol)
    digits = digit_boxes(boxes)
    if not digits:
        return empty
    bitmaps = [_normalize(crop, b, tol) for b in digits]
    text, conf, margin = uid.classify(bitmaps, bank)
    meters = int(text) if text.isdigit() else None
    return dict(meters=meters, text=text, confidence=round(conf, 3),
                margin=round(margin, 3), n_glyphs=len(boxes))
