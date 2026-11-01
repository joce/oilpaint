"""Minimal PNG writer so saving images needs nothing beyond numpy."""

from __future__ import annotations

import struct
import zlib

import numpy as np


def write_png(path: str, img: np.ndarray) -> None:
    img = np.ascontiguousarray(img, dtype=np.uint8)
    if img.ndim == 2:
        img = img[..., None]
    h, w, c = img.shape
    color_type = {1: 0, 3: 2, 4: 6}[c]
    raw = b"".join(b"\x00" + img[y].tobytes() for y in range(h))

    def chunk(tag: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)

    with open(path, "wb") as f:
        f.write(b"\x89PNG\r\n\x1a\n")
        f.write(chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, color_type, 0, 0, 0)))
        f.write(chunk(b"IDAT", zlib.compress(raw, 6)))
        f.write(chunk(b"IEND", b""))
