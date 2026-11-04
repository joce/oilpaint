"""A study after James McNeill Whistler's *Nocturne in Black and Gold: The
Falling Rocket* (1875).

Painted from a photograph of the painting (Detroit Institute of Arts; public
domain; ``examples/reference``) the way Whistler made his nocturnes: very
thin, fluid paint (his "sauce") swept on in broad washes over a dark ground,
smoke and lights brushed into it wet, a few dark figures, and the sparks of
the falling rocket flicked on: spattered from a loaded brush, so that they
fall in showers. This is the picture John Ruskin said flung "a pot of paint
in the public's face", and the libel suit Whistler brought over it.

    python examples/whistler_rocket.py [output.png]
"""

import os
import sys
import time

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from oilpaint import Canvas, Paint, Reference, brushes, surfaces
from oilpaint import palette as P

HERE = os.path.dirname(__file__)
REF = os.path.join(HERE, "reference", "whistler_falling_rocket.jpg")
W = 760
rng = np.random.default_rng(1875)

GOLD = Paint.mix((P.CADMIUM_YELLOW, 2), (P.YELLOW_OCHRE, 1), (P.TITANIUM_WHITE, 0.5))
FIRE = Paint.mix((P.CADMIUM_ORANGE, 2), (P.CADMIUM_YELLOW, 1))


_PAINTS: dict = {}


def sauce(rgb, levels=24):
    """Paint thinned to a fluid wash (one per small step of colour, cached)."""
    key = tuple(np.round(np.clip(rgb, 0, 1) * levels).astype(int))
    if key not in _PAINTS:
        _PAINTS[key] = Paint(np.asarray(key) / levels, body=0.15)
    return _PAINTS[key]


def grid(c, step, jitter=0.5):
    ys, xs = np.mgrid[step / 2 : c.height : step, step / 2 : c.width : step]
    xs = xs.ravel() + rng.uniform(-jitter, jitter, xs.size) * step
    ys = ys.ravel() + rng.uniform(-jitter, jitter, ys.size) * step
    order = rng.permutation(xs.size)
    return xs[order], ys[order]


def along(ref, x, y, length, scale):
    angle, _ = ref.direction(x, y, scale=scale)
    d = 0.5 * length
    return [(x - d * np.cos(angle), y - d * np.sin(angle)),
            (x + d * np.cos(angle), y + d * np.sin(angle))]  # fmt: skip


def washes(c, ref, step, length, scale, brush):
    for x, y in zip(*grid(c, step), strict=True):
        paint = sauce(ref.color(x, y, scale=scale))
        c.stroke(
            along(ref, x, y, length, scale), paint, amount=4, brush=brush, pressure=0.4
        )


def figures(c, ref):
    """The dark figures and the dark accents of the trees and the barge."""
    n = 0
    for x, y in zip(*grid(c, 5), strict=True):
        if ref.value(x, y, scale=1) > 0.2 or ref.detail(x, y, scale=2) < 0.07:
            continue
        c.stroke(along(ref, x, y, 7, 2), sauce(ref.color(x, y, scale=0.8)), amount=6,
                 brush=brushes.ROUND.sized(4), pressure=0.5)  # fmt: skip
        n += 1
    return n


def sparks(c, ref, cell=36):
    """Where the photograph shows sparks (gold points brighter than the night
    around them), flick paint: each patch of sparks is a spatter falling from
    just above it."""
    img = ref.image.astype(float) / (255.0 if ref.image.dtype == np.uint8 else 1.0)
    glow = img[..., 0] * 0.6 + img[..., 1] * 0.4 - img[..., 2] * 0.5
    ys = np.arange(c.height)[:, None]
    around = np.array([[ref.value(x, y, scale=8) for x in range(0, c.width, 4)]
                       for y in range(0, c.height, 4)])  # fmt: skip
    around = np.kron(around, np.ones((4, 4)))[: c.height, : c.width]
    # small gold points clearly brighter than the night around them, above the
    # horizon (the warm ground below is not sparks)
    lit = (glow > 0.3) & (img.mean(-1) > around + 0.16) & (ys < 0.72 * c.height)
    flicks = drops = 0
    for y0 in range(0, c.height, cell):
        for x0 in range(0, c.width, cell):
            m = lit[y0 : y0 + cell, x0 : x0 + cell]
            n = int(m.sum())
            if n < 3:
                continue
            py, px = np.nonzero(m)
            cx, cy = x0 + px.mean(), y0 + py.mean()
            spread_px = max(px.std(), py.std(), 4.0)
            reach = 2.5 * spread_px + 10
            paint = FIRE if img[y0 + py, x0 + px].mean(0)[2] < 0.2 else GOLD
            drops += c.spatter(
                (cx, cy - reach),
                paint,
                amount=min(0.004 * n, 0.4),  # about one drop per spark
                direction=90,
                spread=np.degrees(np.arctan2(spread_px, reach)),
                reach=reach,
                size=1.3 if n < 40 else 1.7,
                seed=int(rng.integers(1 << 31)),
            )
            flicks += 1
    return flicks, drops


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def paint_it(path):
    t0 = time.time()
    with Image.open(REF) as im:
        h = round(W * im.height / im.width)
    ref = Reference(REF, W, h)
    c = Canvas(W, h, surface=surfaces.CANVAS.with_(ground="#1e2420"), seed=1875)
    log("washes")
    washes(c, ref, 30, 130, 26, brushes.MOP.sized(80))
    c.dry()
    log("smoke and lights")
    washes(c, ref, 16, 44, 9, brushes.MOP.sized(34))
    c.dry()
    log("figures")
    f = figures(c, ref)
    c.dry()
    log("sparks")
    flicks, drops = sparks(c, ref)
    c.save(path)
    print(f"wrote {path}: {f} dark touches, {flicks} flicks, {drops} drops, "
          f"in {time.time() - t0:.0f} s")  # fmt: skip


if __name__ == "__main__":
    paint_it(
        sys.argv[1]
        if len(sys.argv) > 1
        else os.path.join(HERE, "output", "whistler_rocket.png")
    )
