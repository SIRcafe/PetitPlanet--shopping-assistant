r"""UID OCR for the three-dot popup: glyph segmentation + template bank.

The popup shows one line:  `• UID: 1031164`.  Digits are ~18 px tall and the
string length is fixed (7), so we segment the row into glyph boxes and match
each to a small bitmap bank.  The bank is built from labelled samples
(`scripts/uid_collect.py` produces the strips, the user labels them once).
"""

from __future__ import annotations

import json

import numpy as np

from . import imgproc as ip

# generous region around the "UID: NNNNNNN" line at 1920x1080
UID_BAND = (1190, 720, 300, 44)
TEXT = (218, 215, 196)
GLYPH_W, GLYPH_H = 16, 24
BANK_PATH_NAME = "uid_digits.json"


def find_band(frame: np.ndarray) -> np.ndarray:
    x, y, w, h = UID_BAND
    return frame[y:y + h, x:x + w]


def glyph_boxes(band: np.ndarray, tol: int = 30) -> list[dict]:
    """Component boxes on the UID line, left to right."""
    mask = ip.color_mask(band, TEXT, tol)
    comps = [
        c for c in ip.components(mask, min_area=18)
        if 4 <= c["w"] <= 28 and 9 <= c["h"] <= 28
    ]
    if not comps:
        return []
    # keep the dominant row only
    mid = sorted(c["y0"] + c["h"] / 2 for c in comps)[len(comps) // 2]
    row = [c for c in comps if abs(c["y0"] + c["h"] / 2 - mid) <= 8]
    row.sort(key=lambda c: c["x0"])
    return row


def normalize(band: np.ndarray, box: dict, tol: int = 30) -> np.ndarray:
    """Grayscale 16x24 glyph bitmap in 0..1 (kept continuous on purpose:
    binarising at this size throws away too much for a reliable match)."""
    pad = 1
    y0 = max(0, box["y0"] - pad)
    x0 = max(0, box["x0"] - pad)
    y1 = min(band.shape[0], box["y0"] + box["h"] + pad)
    x1 = min(band.shape[1], box["x0"] + box["w"] + pad)
    patch = band[y0:y1, x0:x1]
    mask = ip.color_mask(patch, TEXT, tol).astype(np.float32) * 1.0
    small = ip.resize_bilinear(np.dstack([mask * 255] * 3), GLYPH_W, GLYPH_H)[..., 0]
    return (small / 255.0).astype(np.float32)


def segment(frame: np.ndarray, tol: int = 30) -> list[np.ndarray]:
    """Return the glyph bitmaps of the UID line, left to right."""
    band = find_band(frame)
    return [normalize(band, b, tol) for b in glyph_boxes(band, tol)]


def _ncc(a: np.ndarray, b: np.ndarray) -> float:
    a0 = a - a.mean()
    b0 = b - b.mean()
    den = float(np.sqrt((a0 * a0).sum() * (b0 * b0).sum()))
    return float((a0 * b0).sum() / den) if den > 1e-6 else 0.0


def _as_bitmap(v) -> np.ndarray:
    """Accept a JSON list of floats, a '0/1' string, or a flat array."""
    if isinstance(v, str):
        arr = np.array([1.0 if c == "1" else 0.0 for c in v], np.float32)
    else:
        arr = np.asarray(v, np.float32).ravel()
    return arr.reshape(GLYPH_H, GLYPH_W)


def classify(glyphs: list[np.ndarray], bank: dict) -> tuple[str, float, float]:
    """Greedy per-glyph nearest template (normalised cross-correlation).

    Returns (digits, mean confidence, min margin); margin is best minus
    runner-up score, i.e. how far the read is from flipping to another digit.
    """
    if not bank:
        return "", 0.0, 0.0
    tmpl = [(d, _as_bitmap(v)) for d, v in bank.items()]
    out, confs, margins = [], [], []
    for g in glyphs:
        scored = sorted(((_ncc(g, t), d) for d, t in tmpl), reverse=True)
        best_s, best_d = scored[0]
        second = scored[1][0] if len(scored) > 1 else 0.0
        out.append(best_d)
        confs.append(best_s)
        margins.append(best_s - second)
    return ("".join(out),
            sum(confs) / len(confs) if confs else 0.0,
            min(margins) if margins else 0.0)


def read_uid(frame: np.ndarray, bank: dict, drop_prefix: int = 2) -> dict:
    """Segment, drop the `U`/`D` prefix glyphs, classify the 7 digits.

    Measured on 2026-10-18: the row segments as `U`, `D`, then 7 digits
    (the bullet, the `I` and the `:` either merge or fall below min area).
    """
    glyphs = segment(frame)
    digits = glyphs[drop_prefix:] if len(glyphs) > drop_prefix else glyphs
    if len(digits) > 7:
        digits = digits[-7:]
    text, conf, margin = classify(digits, bank)
    return dict(uid=text, confidence=round(conf, 3), margin=round(margin, 3),
                n_glyphs=len(glyphs))


def load_bank(assets_dir) -> dict:
    p = assets_dir / BANK_PATH_NAME
    if not p.exists():
        return {}
    return json.loads(p.read_text(encoding="utf-8"))


def save_bank(assets_dir, digits: dict) -> None:
    """`digits` maps digit character -> flat list/array of floats (0..1)."""
    p = assets_dir / BANK_PATH_NAME
    payload = {}
    for d, v in digits.items():
        if isinstance(v, str):
            payload[d] = [1 if c == "1" else 0 for c in v]
        elif hasattr(v, "ravel"):
            payload[d] = [round(float(x), 3) for x in v.ravel()]
        else:
            payload[d] = [round(float(x), 3) for x in v]
    p.write_text(json.dumps(payload, indent=0), encoding="utf-8")
