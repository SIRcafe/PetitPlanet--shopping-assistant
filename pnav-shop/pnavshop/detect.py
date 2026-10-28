"""Detectors for the Gardien-finding pipeline (T1: offline verification only).

Every detector takes an (H, W, 3) uint8 RGB frame and returns a dict or None:

    {name, found, score, cx, cy, w, h, extra...}

Thresholds live in `spec.py` so the offline report can print the margin
between the best positive score and the worst negative score.
"""

from __future__ import annotations

import numpy as np

from . import imgproc as ip
from . import spec


# --------------------------------------------------------------------------- #
# blue navigation marker (the "blue dot")
# --------------------------------------------------------------------------- #
def _patch_mean(rgb: np.ndarray, cx: float, cy: float, k: int = 1) -> np.ndarray:
    h, w = rgb.shape[:2]
    x0 = max(0, int(round(cx)) - k)
    y0 = max(0, int(round(cy)) - k)
    x1 = min(w, int(round(cx)) + k + 1)
    y1 = min(h, int(round(cy)) + k + 1)
    return rgb[y0:y1, x0:x1].reshape(-1, 3).astype(np.float32).mean(axis=0)


def _ring_mean(rgb: np.ndarray, cx: float, cy: float, radius: float, n: int = 8) -> np.ndarray:
    ang = np.linspace(0, 2 * np.pi, n, endpoint=False)
    pts = np.stack([cx + radius * np.cos(ang), cy + radius * np.sin(ang)], axis=1)
    rgb_mean = np.zeros(3, np.float32)
    for px, py in pts:
        rgb_mean += _patch_mean(rgb, px, py, k=0)
    return rgb_mean / n


def _ring_hits(rgb: np.ndarray, cx: float, cy: float, radius: float, target, tol: int, n: int = 16) -> int:
    """How many of n evenly spaced samples on a circle match `target`."""
    ang = np.linspace(0, 2 * np.pi, n, endpoint=False)
    tgt = np.asarray(target, np.float32)
    hits = 0
    for a in ang:
        px = cx + radius * np.cos(a)
        py = cy + radius * np.sin(a)
        if np.abs(_patch_mean(rgb, px, py, k=0) - tgt).max() <= tol:
            hits += 1
    return hits


def _verdict_blue(rgb: np.ndarray, cx: float, cy: float, r: float) -> tuple[float, dict]:
    """Check the three-ring structure: cream core, dark disc, teal ring.

    Tightened 2026-10-18: a HUD icon (teal ring + pale centre) in the top-left
    corner passed the loose version, so now the core must be *cream* (not just
    bright) and at least 12 of 16 outer samples must sit on the exact ring
    colour.
    """
    core = _patch_mean(rgb, cx, cy, k=1)
    mid = _ring_mean(rgb, cx, cy, r * 0.45)
    outer = _ring_mean(rgb, cx, cy, r * 0.85)
    outer_hits = _ring_hits(rgb, cx, cy, r * 0.85, spec.BLUE_RING, 26, 16)
    mid_dark = _ring_hits(rgb, cx, cy, r * 0.40, (72, 70, 65), 34, 12)

    core_ok = bool(np.abs(core - np.array(spec.BLUE_CORE, np.float32)).max() <= 34)
    mid_ok = 35.0 <= float(mid.mean()) <= 105.0 and mid_dark >= 9
    outer_ok = outer_hits >= 12

    checks = [core_ok, mid_ok, outer_ok]
    score = sum(checks) / 3.0
    if all(checks):
        score = 1.0
    return score, dict(core=core.tolist(), mid=mid.tolist(), outer=outer.tolist(),
                       checks=checks, outer_hits=outer_hits, mid_dark=mid_dark)


def _in_hud(cx: float, cy: float, w: int, h: int) -> bool:
    """HUD strips where the marker never legitimately appears."""
    zones = [
        (0.00, 0.00, 0.13, 0.19),      # top-left clock / 1w / exp bar
        (0.58, 0.00, 1.00, 0.13),      # top-right F1..E icon row
        (0.00, 0.86, 0.17, 1.00),      # bottom-left Q/C/Enter
        (0.83, 0.72, 1.00, 1.00),      # bottom-right Tab
        (0.00, 0.03, 1.00, 0.055),     # top status strip
    ]
    for x0, y0, x1, y1 in zones:
        if x0 * w <= cx <= x1 * w and y0 * h <= cy <= y1 * h:
            return True
    return False


def _cream_cores(rgb: np.ndarray) -> list[tuple[float, float]]:
    """Candidate marker centres: the tiny cream dot inside the marker.

    Anchoring on the core instead of the teal ring is what lets the detector
    survive the marker being clamped against blue scenery (railing / water):
    there the teal ring merges with the background blob and its bounding-box
    centre drifts by tens of pixels, which used to make the ring checks fail.
    """
    mask = ip.color_mask(rgb, spec.BLUE_CORE, 22)
    out = []
    for c in ip.components(mask, min_area=20):
        if 4 <= c["w"] <= 18 and 4 <= c["h"] <= 18 and 20 <= c["area"] <= 260:
            out.append(((c["x0"] + c["x1"]) / 2.0, (c["y0"] + c["y1"]) / 2.0))
    return out


def detect_blue_marker(rgb: np.ndarray, roi: dict | None = None) -> dict | None:
    """Find the blue navigation dot.  `roi` is fractional and optional; the
    dot is a fixed-colour UI overlay, so a limited ROI is nearly always safe
    and cuts cost by the area ratio."""
    ox = oy = 0
    frame = rgb
    if roi is not None:
        h, w = rgb.shape[:2]
        ox, oy = int(roi["x0"] * w), int(roi["y0"] * h)
        frame = rgb[oy:int(roi["y1"] * h), ox:int(roi["x1"] * w)]

    # --- path 1: anchor on the cream core, then verify the three rings ------
    for cx, cy in _cream_cores(frame):
        if _in_hud(cx, cy, frame.shape[1], frame.shape[0]):
            continue
        for r in range(16, 26):
            vscore, detail = _verdict_blue(frame, cx, cy, float(r))
            if vscore >= 0.99:
                return dict(name="blue_marker", found=True, score=vscore,
                            cx=cx + ox, cy=cy + oy, r=float(r),
                            w=2 * r, h=2 * r, area=0,
                            extra={**detail, "anchor": "core"})

    # --- path 2: teal blob, with a small centre grid refinement -------------
    mask = ip.color_mask(frame, spec.BLUE_RING, spec.BLUE_RING_TOL)
    comps = ip.components(mask, min_area=spec.BLUE_MIN_AREA)
    best = None
    for c in comps:
        if not (spec.BLUE_MIN_DIAM - 8 <= max(c["w"], c["h"]) <= spec.BLUE_MAX_DIAM + 40):
            continue
        cx0 = (c["x0"] + c["x1"]) / 2.0
        cy0 = (c["y0"] + c["y1"]) / 2.0
        span_x = max(1, c["w"] // 2)
        span_y = max(1, c["h"] // 2)
        for dy in range(-span_y, span_y + 1, 4):
            for dx in range(-span_x, span_x + 1, 4):
                for r in range(15, 27, 2):
                    vscore, detail = _verdict_blue(frame, cx0 + dx, cy0 + dy, float(r))
                    if vscore < 0.99:
                        continue
                    cand = dict(name="blue_marker", found=True, score=vscore,
                                cx=cx0 + dx + ox, cy=cy0 + dy + oy, r=float(r),
                                w=2 * r, h=2 * r, area=c["area"],
                                extra={**detail, "anchor": "teal"})
                    if best is None:
                        best = cand
                    break
    return best


# --------------------------------------------------------------------------- #
# template based UI detectors
# --------------------------------------------------------------------------- #
def _match_multiscale(rgb: np.ndarray, template: np.ndarray, roi: dict, scales) -> dict:
    """Greyscale NCC inside a fractional ROI, over a list of template scales."""
    h, w = rgb.shape[:2]
    x0 = int(roi["x0"] * w)
    x1 = int(roi["x1"] * w)
    y0 = int(roi["y0"] * h)
    y1 = int(roi["y1"] * h)
    patch = rgb[y0:y1, x0:x1]
    patch_gray = ip.to_gray(patch)

    best = None
    for s in scales:
        tw = max(8, int(round(template.shape[1] * s)))
        th = max(8, int(round(template.shape[0] * s)))
        if tw >= patch_gray.shape[1] or th >= patch_gray.shape[0]:
            continue
        tpl = template if (tw, th) == (template.shape[1], template.shape[0]) else ip.resize_bilinear(template, tw, th)
        score_map = ip.ncc_map(patch_gray, ip.to_gray(tpl))
        m = ip.best_match(score_map, (th, tw))
        # runner-up peak, measured outside a window around the best hit
        r = max(6, min(th, tw) // 2)
        tmp = score_map.copy()
        tmp[max(0, m["y"] - r):m["y"] + r + 1, max(0, m["x"] - r):m["x"] + r + 1] = -1.0
        second = float(tmp.max()) if tmp.size else -1.0
        cand = dict(
            score=m["score"],
            margin=m["score"] - second,
            x=x0 + m["x"],
            y=y0 + m["y"],
            cx=x0 + m["cx"],
            cy=y0 + m["cy"],
            w=tw,
            h=th,
            scale=s,
        )
        if best is None or cand["score"] > best["score"]:
            best = cand
    if best is None:
        return dict(score=0.0, margin=0.0, cx=-1, cy=-1, w=0, h=0, scale=0.0, color_rmse=999.0)
    # colour sanity check at the best location
    loc = rgb[best["y"]:best["y"] + best["h"], best["x"]:best["x"] + best["w"]]
    tpl_at_scale = template if (best["w"], best["h"]) == (template.shape[1], template.shape[0]) \
        else ip.resize_bilinear(template, best["w"], best["h"])
    best["color_rmse"] = ip.color_rmse(loc, tpl_at_scale) if loc.shape == tpl_at_scale.shape else 999.0
    return best


def detect_avatar_candidates(rgb: np.ndarray, template: np.ndarray, top: int = 5,
                             roi: dict | None = None, scales=None,
                             min_score: float = 0.70, spacing: int = 45) -> list[dict]:
    """All plausible avatar positions, best first.

    The map's avatar row is a line of similar-looking animal badges, and the
    格连 template happily scores 0.97 on the *neighbour* badge (measured on
    2026-10-18: the detector picked x=780 while 格连 sat at x=625).  Clicking
    the single best match therefore selects the wrong character and the
    「寻找友邻」 button never appears — so the caller must be able to try the
    next candidate instead of trusting argmax.
    """
    roi = roi or spec.ROI_AVATAR
    scales = scales or spec.AVATAR_SCALES
    h, w = rgb.shape[:2]
    x0, x1 = int(roi["x0"] * w), int(roi["x1"] * w)
    y0, y1 = int(roi["y0"] * h), int(roi["y1"] * h)
    patch = ip.to_gray(rgb[y0:y1, x0:x1])

    found: list[dict] = []
    for s in scales:
        tw = max(8, int(round(template.shape[1] * s)))
        th = max(8, int(round(template.shape[0] * s)))
        if tw >= patch.shape[1] or th >= patch.shape[0]:
            continue
        tpl = template if (tw, th) == (template.shape[1], template.shape[0]) \
            else ip.resize_bilinear(template, tw, th)
        score_map = ip.ncc_map(patch, ip.to_gray(tpl))
        work = score_map.copy()
        for _ in range(top):
            idx = int(np.argmax(work))
            yy, xx = np.unravel_index(idx, work.shape)
            score = float(work[yy, xx])
            if score < min_score:
                break
            found.append(dict(score=score, cx=x0 + xx + tw / 2.0, cy=y0 + yy + th / 2.0,
                              w=tw, h=th, scale=s))
            for dy in (-spacing, 0, spacing):
                for dx in (-spacing, 0, spacing):
                    ya, yb = max(0, yy + dy - spacing // 2), min(work.shape[0], yy + dy + spacing)
                    xa, xb = max(0, xx + dx - spacing // 2), min(work.shape[1], xx + dx + spacing)
                    work[ya:yb, xa:xb] = -1.0

    found.sort(key=lambda c: c["score"], reverse=True)
    kept: list[dict] = []
    for c in found:
        if all((c["cx"] - k["cx"]) ** 2 + (c["cy"] - k["cy"]) ** 2 > spacing ** 2
               for k in kept):
            kept.append(c)
        if len(kept) >= top:
            break
    return kept


def detect_avatar(rgb: np.ndarray, template: np.ndarray) -> dict:
    best = _match_multiscale(rgb, template, spec.ROI_AVATAR, spec.AVATAR_SCALES)
    found = (
        best["score"] >= spec.AVATAR_SCORE_MIN
        and best.get("color_rmse", 999) <= spec.AVATAR_COLOR_MAX
        and best.get("margin", 0.0) >= spec.AVATAR_MARGIN_MIN
    )
    return dict(name="avatar_gelian", found=found, **best)


def detect_find_button(rgb: np.ndarray, template: np.ndarray) -> dict:
    best = _match_multiscale(rgb, template, spec.ROI_FIND_BTN, spec.UI_SCALES)
    found = best["score"] >= spec.FIND_BTN_SCORE_MIN and best.get("color_rmse", 999) <= spec.UI_COLOR_MAX
    return dict(name="find_neighbor_btn", found=found, **best)


def detect_f_prompt(rgb: np.ndarray, template: np.ndarray) -> dict:
    best = _match_multiscale(rgb, template, spec.ROI_F_PROMPT, spec.UI_SCALES)
    found = best["score"] >= spec.F_PROMPT_SCORE_MIN and best.get("color_rmse", 999) <= spec.UI_COLOR_MAX
    return dict(name="f_prompt_gelian", found=found, **best)


def detect_map_open(rgb: np.ndarray, avatar_template: np.ndarray) -> dict:
    """The avatar row only exists on the map screen, so match one avatar."""
    best = _match_multiscale(rgb, avatar_template, spec.ROI_AVATAR, spec.AVATAR_SCALES)
    found = best["score"] >= spec.MAP_OPEN_SCORE_MIN
    return dict(name="map_open", found=found, **best)


# --------------------------------------------------------------------------- #
# bridge / railing detection (image 14)
# --------------------------------------------------------------------------- #
RAIL_BLUE = spec.RAIL_BLUE
RAIL_TOL = spec.RAIL_TOL
BRIDGE_RATIO = 0.030        # blue coverage of the lower half while on the bridge


def bridge_ratio(rgb: np.ndarray) -> float:
    h = rgb.shape[0]
    lower = rgb[int(h * 0.5):]
    mask = ip.color_mask(lower, RAIL_BLUE, RAIL_TOL)
    return float(mask.sum()) / float(lower.shape[0] * lower.shape[1])


def on_bridge(rgb: np.ndarray) -> tuple[bool, float]:
    r = bridge_ratio(rgb)
    return r >= BRIDGE_RATIO, r


# --------------------------------------------------------------------------- #
# jump / climb badge (pic1/16): golden round badge + key cap on the right edge
# --------------------------------------------------------------------------- #
def jump_badge(img: np.ndarray, thresh: float = 0.70) -> tuple[bool, int]:
    """Golden jump/climb badge at the right edge (template from pic1/16).

    Colour counting is unusable: a field of yellow flowers scored 1500+ "gold"
    pixels and made the walker jump forever.  This is a real template match on
    the badge glyph (gold ring + dark running figure + key cap).
    """
    tpl = _jump_template()
    if tpl is None:
        return False, 0
    h, w = img.shape[:2]
    band = img[int(h * 0.20):int(h * 0.95), int(w * 0.80):]
    band_gray = ip.to_gray(band)
    best = 0.0
    for sc in (0.90, 1.00, 1.10):
        tw = max(8, int(round(tpl.shape[1] * sc)))
        th = max(8, int(round(tpl.shape[0] * sc)))
        if tw >= band_gray.shape[1] or th >= band_gray.shape[0]:
            continue
        t = tpl if (tw, th) == (tpl.shape[1], tpl.shape[0]) else ip.resize_bilinear(tpl, tw, th)
        m = ip.ncc_map(band_gray, ip.to_gray(t))
        best = max(best, float(m.max()))
    return best >= thresh, int(best * 1000)


_JUMP_TPL = None


def _jump_template():
    global _JUMP_TPL
    if _JUMP_TPL is None:
        from . import pngio, spec
        p = spec.TEMPLATE_DIR / "jump_badge.png"
        if not p.exists():
            return None
        _JUMP_TPL = pngio.load_image(p, cache_dir=spec.CACHE_DIR)
    return _JUMP_TPL


# --------------------------------------------------------------------------- #
# distance readout under the blue dot ("N米")
# --------------------------------------------------------------------------- #
def distance_signature(img: np.ndarray, dot: dict) -> np.ndarray | None:
    """Bitmap of the "N米" text under the marker (no OCR needed).

    The escape rule only needs "did the distance change?", so comparing these
    bitmaps is enough: identical bitmap == we did not get any closer.
    """
    if not dot or not dot.get("found"):
        return None
    cx, cy = int(dot["cx"]), int(dot["cy"])
    y0, y1 = cy + 16, cy + 56
    x0, x1 = max(0, cx - 60), min(img.shape[1], cx + 60)
    if y0 >= img.shape[0] or x1 <= x0:
        return None
    crop = img[y0:y1, x0:x1]
    mask = ip.color_mask(crop, (240, 238, 225), 45).astype(np.float32)
    small = ip.resize_bilinear(np.dstack([mask * 255] * 3), 60, 20)[..., 0]
    return (small > 100).astype(np.float32)


def distance_changed(a, b, iou_thresh: float = 0.88) -> bool:
    if a is None or b is None:
        return False
    inter = float((a * b).sum())
    union = float(((a + b) > 0).sum())
    if union <= 0:
        return False
    return (inter / union) < iou_thresh
