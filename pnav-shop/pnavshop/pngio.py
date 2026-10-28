"""Minimal PNG codec (stdlib + numpy), plus an .npy cache.

Why hand-rolled: this machine cannot install packages (pip hangs), so the
offline recognition harness must run on bare Python + numpy.  The product
build will use OpenCV when it becomes available; nothing here blocks that.

Supported PNG: 8-bit, non-interlaced, colour type 2 (RGB) / 6 (RGBA).
"""

from __future__ import annotations

import binascii
import pathlib
import struct
import zlib

import numpy as np

PNG_SIG = b"\x89PNG\r\n\x1a\n"
_CHANNELS = {0: 1, 2: 3, 4: 2, 6: 4}


# --------------------------------------------------------------------------- #
# decode
# --------------------------------------------------------------------------- #
def _paeth(a: int, b: int, c: int) -> int:
    p = a + b - c
    pa, pb, pc = abs(p - a), abs(p - b), abs(p - c)
    if pa <= pb and pa <= pc:
        return a
    if pb <= pc:
        return b
    return c


def _filter_sub(line: np.ndarray, bpp: int) -> np.ndarray:
    a = line.reshape(-1, bpp).astype(np.uint32)
    return (np.cumsum(a, axis=0) & 0xFF).astype(np.uint8).reshape(-1)


def _filter_average(line: np.ndarray, prev: np.ndarray, bpp: int) -> np.ndarray:
    n = line.shape[0]
    ln, pv = line.tobytes(), prev.tobytes()
    out = bytearray(n)
    for i in range(bpp):
        out[i] = (ln[i] + (pv[i] >> 1)) & 0xFF
    for i in range(bpp, n):
        out[i] = (ln[i] + ((out[i - bpp] + pv[i]) >> 1)) & 0xFF
    return np.frombuffer(bytes(out), np.uint8)


def _filter_paeth(line: np.ndarray, prev: np.ndarray, bpp: int) -> np.ndarray:
    n = line.shape[0]
    ln, pv = line.tobytes(), prev.tobytes()
    out = bytearray(n)
    for i in range(bpp):
        out[i] = ln[i]
    for i in range(bpp, n):
        a = out[i - bpp]
        b = pv[i]
        c = pv[i - bpp]
        out[i] = (ln[i] + _paeth(a, b, c)) & 0xFF
    return np.frombuffer(bytes(out), np.uint8)


def _unfilter(raw: bytes, width: int, height: int, channels: int) -> np.ndarray:
    bpp = channels
    stride = width * bpp
    out = np.empty((height, stride), np.uint8)
    prev = np.zeros(stride, np.uint8)
    pos = 0
    for y in range(height):
        ftype = raw[pos]
        pos += 1
        line = np.frombuffer(raw, np.uint8, stride, pos)
        pos += stride
        if ftype == 0:
            cur = np.array(line, copy=True)
        elif ftype == 1:
            cur = _filter_sub(line, bpp)
        elif ftype == 2:
            cur = (line.astype(np.uint16) + prev).astype(np.uint8)
        elif ftype == 3:
            cur = _filter_average(line, prev, bpp)
        elif ftype == 4:
            cur = _filter_paeth(line, prev, bpp)
        else:
            raise ValueError(f"unsupported PNG filter type {ftype} on row {y}")
        out[y] = cur
        prev = cur
    return out.reshape(height, width, channels)


def decode_png(path: str | pathlib.Path,
               composite: tuple[int, int, int] = (255, 255, 255)) -> np.ndarray:
    """Return an (H, W, 3) uint8 RGB array (alpha -> `composite`)."""
    data = pathlib.Path(path).read_bytes()
    if data[:8] != PNG_SIG:
        raise ValueError(f"not a PNG: {path}")
    pos, idat, header, palette = 8, [], None, None
    while pos < len(data):
        (length,) = struct.unpack(">I", data[pos:pos + 4])
        ctype = data[pos + 4:pos + 8]
        body = data[pos + 8:pos + 8 + length]
        if ctype == b"IHDR":
            w, h, depth, color, _comp, _filt, inter = struct.unpack(">IIBBBBB", body)
            if depth != 8:
                raise ValueError(f"unsupported bit depth {depth}")
            if inter != 0:
                raise ValueError("interlaced PNG unsupported")
            header = (w, h, color)
        elif ctype == b"PLTE":
            palette = np.frombuffer(body, np.uint8).reshape(-1, 3)
        elif ctype == b"IDAT":
            idat.append(body)
        elif ctype == b"IEND":
            break
        pos += 12 + length
    if header is None:
        raise ValueError("missing IHDR")
    w, h, color = header
    if color == 3:                      # indexed colour (palette PNG)
        if palette is None:
            raise ValueError("palette PNG without PLTE")
        idx = _unfilter(zlib.decompress(b"".join(idat)), w, h, 1)[..., 0]
        return np.ascontiguousarray(palette[idx])
    ch = _CHANNELS[color]
    img = _unfilter(zlib.decompress(b"".join(idat)), w, h, ch)
    if ch == 4:
        alpha = img[..., 3:4].astype(np.uint16)
        rgb = img[..., :3].astype(np.uint16)
        # composite over the caller's background colour (white keeps UI captures
        # looking the same; dark backgrounds need their own colour)
        back = np.array(composite, dtype=np.uint16)[None, None, :]
        rgb = (rgb * alpha + back * (255 - alpha)) // 255
        return rgb.astype(np.uint8)
    if ch == 1 or ch == 2:
        img = np.repeat(img[..., :1], 3, axis=2)
    return np.ascontiguousarray(img[..., :3])


# --------------------------------------------------------------------------- #
# encode
# --------------------------------------------------------------------------- #
def _chunk(ctype: bytes, body: bytes) -> bytes:
    crc = binascii.crc32(ctype + body) & 0xFFFFFFFF
    return struct.pack(">I", len(body)) + ctype + body + struct.pack(">I", crc)


def save_png(path: str | pathlib.Path, arr: np.ndarray) -> None:
    """Write an (H, W, 3) uint8 RGB array as PNG (filter 0)."""
    a = np.ascontiguousarray(arr.astype(np.uint8))
    h, w, _ = a.shape
    raw = b"".join(b"\x00" + a[y].tobytes() for y in range(h))
    ihdr = struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0)
    png = PNG_SIG + _chunk(b"IHDR", ihdr) + _chunk(b"IDAT", zlib.compress(raw, 6)) + _chunk(b"IEND", b"")
    pathlib.Path(path).write_bytes(png)


# --------------------------------------------------------------------------- #
# cached loader
# --------------------------------------------------------------------------- #
def load_image(path: str | pathlib.Path,
               cache_dir: str | pathlib.Path | None = None,
               composite: tuple[int, int, int] = (255, 255, 255)) -> np.ndarray:
    """Decode a PNG, caching the decoded array as .npy (decoding is slow)."""
    p = pathlib.Path(path)
    if cache_dir is None:
        return decode_png(p, composite=composite)
    tag = "%02x%02x%02x" % tuple(int(max(0, min(255, v))) for v in composite)
    cache = pathlib.Path(cache_dir) / (p.stem + "." + tag + ".npy")
    if cache.exists() and cache.stat().st_mtime >= p.stat().st_mtime:
        return np.load(cache)
    arr = decode_png(p, composite=composite)
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache, arr)
    return arr

def to_photoimage(rgb, master=None):
    """numpy RGB array -> tk.PhotoImage via a cached temp PNG."""
    import hashlib
    import tempfile
    import tkinter as tk

    rgb = np.ascontiguousarray(rgb[:, :, :3])
    folder = pathlib.Path(tempfile.gettempdir()) / "pnav_ui"
    folder.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha1(rgb.tobytes()).hexdigest()[:16]
    path = folder / ("view_" + digest + ".png")
    if not path.exists():
        save_png(path, rgb)
    return tk.PhotoImage(master=master, file=str(path))

def decode_rgba(path: str | pathlib.Path):
    """Decode a PNG keeping alpha when present -> (h, w, 4) uint8, else None."""
    import struct as _struct
    import zlib as _zlib

    data = pathlib.Path(path).read_bytes()
    if data[:8] != PNG_SIG:
        return None
    pos, idat, header = 8, [], None
    while pos < len(data):
        (length,) = _struct.unpack(">I", data[pos:pos + 4])
        ctype = data[pos + 4:pos + 8]
        if ctype == b"IHDR":
            w, h, depth, color, _c, _f, inter = _struct.unpack(">IIBBBBB", data[pos + 8:pos + 21])
            if depth != 8 or color != 6 or inter != 0:
                return None
            header = (w, h)
        elif ctype == b"IDAT":
            idat.append(data[pos + 8:pos + 8 + length])
        elif ctype == b"IEND":
            break
        pos += 12 + length
    if header is None:
        return None
    return _unfilter(_zlib.decompress(b"".join(idat)), header[0], header[1], 4)
