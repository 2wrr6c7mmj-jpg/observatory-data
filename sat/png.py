"""Minimal PNG reader/writer (numpy + zlib only).

Reads non-interlaced PNGs of colour types 0 (grey), 2 (RGB), 3 (palette, 8 bit),
4 (grey+alpha) and 6 (RGBA) at bit depth 8 or 16; this covers the UINT16
multi-band PNGs returned by the Sentinel Hub Process API. Writes 8-bit grey,
RGB or palette images.
"""
from __future__ import annotations

import struct
import zlib

import numpy as np

SIGNATURE = b"\x89PNG\r\n\x1a\n"
CHANNELS = {0: 1, 2: 3, 3: 1, 4: 2, 6: 4}


class PNGError(ValueError):
    pass


def _chunks(data: bytes):
    if not data.startswith(SIGNATURE):
        raise PNGError("not a PNG file")
    pos = len(SIGNATURE)
    while pos + 8 <= len(data):
        (length,) = struct.unpack(">I", data[pos:pos + 4])
        ctype = data[pos + 4:pos + 8]
        body = data[pos + 8:pos + 8 + length]
        if len(body) != length:
            raise PNGError("truncated chunk")
        yield ctype, body
        pos += 12 + length
        if ctype == b"IEND":
            return
    raise PNGError("missing IEND")


def _paeth_row(f: bytearray, prior: bytes, bpp: int) -> bytearray:
    out = bytearray(f)
    for i in range(len(out)):
        a = out[i - bpp] if i >= bpp else 0
        b = prior[i]
        c = prior[i - bpp] if i >= bpp else 0
        p = a + b - c
        pa, pb, pc = abs(p - a), abs(p - b), abs(p - c)
        if pa <= pb and pa <= pc:
            pred = a
        elif pb <= pc:
            pred = b
        else:
            pred = c
        out[i] = (out[i] + pred) & 0xFF
    return out


def _avg_row(f: bytearray, prior: bytes, bpp: int) -> bytearray:
    out = bytearray(f)
    for i in range(len(out)):
        a = out[i - bpp] if i >= bpp else 0
        out[i] = (out[i] + ((a + prior[i]) >> 1)) & 0xFF
    return out


def _unfilter(raw: bytes, height: int, stride: int, bpp: int) -> np.ndarray:
    if len(raw) < height * (stride + 1):
        raise PNGError("image data too short")
    out = np.zeros((height, stride), dtype=np.uint8)
    prior = np.zeros(stride, dtype=np.uint8)
    for r in range(height):
        start = r * (stride + 1)
        ftype = raw[start]
        line = raw[start + 1:start + 1 + stride]
        f = np.frombuffer(line, dtype=np.uint8)
        if ftype == 0:
            cur = f.copy()
        elif ftype == 1:  # Sub: cumulative sum per byte position within a pixel
            cur = (np.cumsum(f.reshape(-1, bpp).astype(np.int64), axis=0) % 256).astype(np.uint8).ravel()
        elif ftype == 2:
            cur = (f.astype(np.uint16) + prior).astype(np.uint8)
        elif ftype == 3:
            cur = np.frombuffer(bytes(_avg_row(bytearray(line), prior.tobytes(), bpp)), dtype=np.uint8)
        elif ftype == 4:
            cur = np.frombuffer(bytes(_paeth_row(bytearray(line), prior.tobytes(), bpp)), dtype=np.uint8)
        else:
            raise PNGError(f"bad filter type {ftype}")
        out[r] = cur
        prior = out[r]
    return out


def decode(data: bytes) -> np.ndarray:
    """Return an array of shape (h, w) or (h, w, c); uint8 or uint16."""
    ihdr = None
    idat = []
    for ctype, body in _chunks(data):
        if ctype == b"IHDR":
            ihdr = struct.unpack(">IIBBBBB", body)
        elif ctype == b"IDAT":
            idat.append(body)
    if ihdr is None:
        raise PNGError("missing IHDR")
    width, height, depth, ctype, _comp, _filt, interlace = ihdr
    if interlace:
        raise PNGError("interlaced PNG not supported")
    if ctype not in CHANNELS or depth not in (8, 16) or (ctype == 3 and depth != 8):
        raise PNGError(f"unsupported PNG (colour type {ctype}, depth {depth})")
    ch = CHANNELS[ctype]
    bpp = ch * depth // 8
    raw = zlib.decompress(b"".join(idat))
    img = _unfilter(raw, height, width * bpp, bpp)
    if depth == 16:
        arr = img.reshape(height, width * ch, 2)
        arr = (arr[..., 0].astype(np.uint16) << 8) | arr[..., 1]
    else:
        arr = img
    arr = arr.reshape(height, width, ch)
    return arr[..., 0] if ch == 1 else arr


def _chunk(ctype: bytes, body: bytes) -> bytes:
    return struct.pack(">I", len(body)) + ctype + body + struct.pack(">I", zlib.crc32(ctype + body) & 0xFFFFFFFF)


def encode(arr: np.ndarray, palette: list[tuple[int, int, int]] | None = None,
           filters: list[int] | None = None, depth: int = 8) -> bytes:
    """Encode uint8 grey (h, w), RGB (h, w, 3) or palette (h, w) + palette.

    ``filters`` (per row, 0..4) is for tests; the default writes filter 0.
    ``depth=16`` writes uint16 data (grey or multi-channel), used by tests.
    """
    arr = np.asarray(arr)
    if arr.ndim == 2:
        h, w = arr.shape
        ch = 1
    else:
        h, w, ch = arr.shape
    if palette is not None:
        ctype = 3
    else:
        ctype = {1: 0, 2: 4, 3: 2, 4: 6}[ch]
    if depth == 16:
        a16 = arr.astype(">u2").reshape(h, w * ch)
        rows = a16.view(np.uint8).reshape(h, w * ch * 2)
    else:
        rows = arr.astype(np.uint8).reshape(h, w * ch)
    bpp = ch * depth // 8
    out = bytearray()
    prior = np.zeros(rows.shape[1], dtype=np.int32)
    for r in range(h):
        cur = rows[r].astype(np.int32)
        ft = filters[r % len(filters)] if filters else 0
        if ft == 0:
            f = cur
        elif ft == 1:
            left = np.concatenate([np.zeros(bpp, np.int32), cur[:-bpp]])
            f = cur - left
        elif ft == 2:
            f = cur - prior
        elif ft == 3:
            left = np.concatenate([np.zeros(bpp, np.int32), cur[:-bpp]])
            f = cur - ((left + prior) >> 1)
        else:
            left = np.concatenate([np.zeros(bpp, np.int32), cur[:-bpp]])
            ul = np.concatenate([np.zeros(bpp, np.int32), prior[:-bpp]])
            p = left + prior - ul
            pa, pb, pc = np.abs(p - left), np.abs(p - prior), np.abs(p - ul)
            pred = np.where((pa <= pb) & (pa <= pc), left, np.where(pb <= pc, prior, ul))
            f = cur - pred
        out.append(ft)
        out += (f % 256).astype(np.uint8).tobytes()
        prior = cur
    body = SIGNATURE + _chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, depth, ctype, 0, 0, 0))
    if palette is not None:
        body += _chunk(b"PLTE", b"".join(bytes(c) for c in palette))
    body += _chunk(b"IDAT", zlib.compress(bytes(out), 9))
    return body + _chunk(b"IEND", b"")
