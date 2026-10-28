"""Image ops on numpy only: colour masks, connected components, resize, FFT NCC.

The FFT normalised cross-correlation reproduces OpenCV's TM_CCOEFF_NORMED,
so the detector code can swap in cv2.matchTemplate later without changing
semantics.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np

GRAY_WEIGHTS = np.array([0.299, 0.587, 0.114], dtype=np.float32)


# --------------------------------------------------------------------------- #
# basics
# --------------------------------------------------------------------------- #
def to_gray(rgb: np.ndarray) -> np.ndarray:
    return (rgb.astype(np.float32) @ GRAY_WEIGHTS).astype(np.float32)


def subimage(rgb: np.ndarray, x0: int, y0: int, w: int, h: int) -> np.ndarray:
    return rgb[y0:y0 + h, x0:x0 + w]


def color_mask(rgb: np.ndarray, target: Iterable[int], tol: int) -> np.ndarray:
    diff = np.abs(rgb.astype(np.int16) - np.asarray(list(target), np.int16))
    return np.all(diff <= tol, axis=2)


def color_distance(rgb: np.ndarray, target: Iterable[int]) -> np.ndarray:
    diff = rgb.astype(np.int16) - np.asarray(list(target), np.int16)
    return np.sqrt((diff.astype(np.float32) ** 2).sum(axis=2))


def resize_bilinear(rgb: np.ndarray, out_w: int, out_h: int) -> np.ndarray:
    src_h, src_w = rgb.shape[:2]
    if (out_w, out_h) == (src_w, src_h):
        return rgb
    ys = (np.arange(out_h, dtype=np.float32) + 0.5) * (src_h / out_h) - 0.5
    xs = (np.arange(out_w, dtype=np.float32) + 0.5) * (src_w / out_w) - 0.5
    ys = np.clip(ys, 0, src_h - 1)
    xs = np.clip(xs, 0, src_w - 1)
    y0 = np.floor(ys).astype(np.int32)
    x0 = np.floor(xs).astype(np.int32)
    y1 = np.minimum(y0 + 1, src_h - 1)
    x1 = np.minimum(x0 + 1, src_w - 1)
    wy = (ys - y0)[:, None, None]
    wx = (xs - x0)[None, :, None]
    src = rgb.astype(np.float32)
    top = src[y0][:, x0] * (1 - wx) + src[y0][:, x1] * wx
    bot = src[y1][:, x0] * (1 - wx) + src[y1][:, x1] * wx
    return np.clip(top * (1 - wy) + bot * wy, 0, 255).astype(np.uint8)


# --------------------------------------------------------------------------- #
# connected components (8-connectivity) on a sparse boolean mask
# --------------------------------------------------------------------------- #
def components(mask: np.ndarray, min_area: int = 1) -> list[dict]:
    ys, xs = np.nonzero(mask)
    remaining = set(zip(ys.tolist(), xs.tolist()))
    out: list[dict] = []
    while remaining:
        seed = remaining.pop()
        stack = [seed]
        comp = [seed]
        while stack:
            cy, cx = stack.pop()
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    if dy == 0 and dx == 0:
                        continue
                    nb = (cy + dy, cx + dx)
                    if nb in remaining:
                        remaining.discard(nb)
                        stack.append(nb)
                        comp.append(nb)
        if len(comp) < min_area:
            continue
        cy = np.fromiter((p[0] for p in comp), np.int32, len(comp))
        cx = np.fromiter((p[1] for p in comp), np.int32, len(comp))
        out.append(
            dict(
                x0=int(cx.min()),
                y0=int(cy.min()),
                x1=int(cx.max()),
                y1=int(cy.max()),
                w=int(cx.max() - cx.min() + 1),
                h=int(cy.max() - cy.min() + 1),
                area=len(comp),
                cx=float(cx.mean()),
                cy=float(cy.mean()),
            )
        )
    out.sort(key=lambda c: c["area"], reverse=True)
    return out


# --------------------------------------------------------------------------- #
# normalised cross correlation via FFT (== cv2.TM_CCOEFF_NORMED)
# --------------------------------------------------------------------------- #
def _correlate(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    ah, aw = a.shape
    bh, bw = b.shape
    shape = (ah + bh - 1, aw + bw - 1)
    fa = np.fft.rfft2(a.astype(np.float32), s=shape)
    fb = np.fft.rfft2(b.astype(np.float32), s=shape)
    full = np.fft.irfft2(fa * np.conj(fb), s=shape)
    return full[: ah - bh + 1, : aw - bw + 1].astype(np.float32)


def ncc_map(image: np.ndarray, template: np.ndarray) -> np.ndarray:
    """Score map over valid offsets; image/template are 2-D float arrays."""
    ih, iw = image.shape
    th, tw = template.shape
    if th > ih or tw > iw:
        raise ValueError(f"template {tw}x{th} larger than image {iw}x{ih}")
    n = float(th * tw)
    t_sum = float(template.sum(dtype=np.float64))
    t_sq = float((template.astype(np.float64) ** 2).sum())
    t_var = t_sq - t_sum * t_sum / n
    if t_var <= 1e-6:
        raise ValueError("degenerate (constant) template")

    ones = np.ones((th, tw), np.float32)
    i_sum = _correlate(image, ones)
    i_sq = _correlate(image * image, ones)
    i_t = _correlate(image, template)

    num = i_t - i_sum * (t_sum / n)
    i_var = np.maximum(i_sq - i_sum * i_sum / n, 0.0)
    den = np.sqrt(i_var * t_var)
    with np.errstate(divide="ignore", invalid="ignore"):
        score = np.where(den > 1e-6, num / den, 0.0)
    return np.clip(score, -1.0, 1.0)


def best_match(score: np.ndarray, template_shape: tuple[int, int]) -> dict:
    idx = int(np.argmax(score))
    y, x = np.unravel_index(idx, score.shape)
    th, tw = template_shape
    return dict(
        score=float(score[y, x]),
        x=int(x),
        y=int(y),
        cx=int(x + tw // 2),
        cy=int(y + th // 2),
        w=int(tw),
        h=int(th),
    )


def ncc_best(image: np.ndarray, template: np.ndarray) -> dict:
    return best_match(ncc_map(to_gray(image), to_gray(template)), template.shape[:2])


def color_rmse(a: np.ndarray, b: np.ndarray) -> float:
    """Mean per-channel absolute colour difference (0..255) between equal shapes."""
    if a.shape != b.shape:
        raise ValueError(f"shape mismatch {a.shape} vs {b.shape}")
    return float(np.abs(a.astype(np.int16) - b.astype(np.int16)).mean())
