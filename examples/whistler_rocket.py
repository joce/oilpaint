"""A study after James McNeill Whistler's *Nocturne in Black and Gold: The
Falling Rocket* (1875).

Painted from a photograph of the painting (Detroit Institute of Arts; public
domain; ``examples/reference``) the way Whistler made his nocturnes: very
thin, fluid paint (his "sauce") swept on in broad washes over a dark ground,
the smoke and the lights brushed into it, a few dark figures, and the sparks
of the falling rocket flicked on: spattered from a loaded brush, so that they
fall in showers. This is the picture John Ruskin said flung "a pot of paint
in the public's face", and the libel suit Whistler brought over it. Each
layer looks at the canvas so far and paints only where it still differs from
the photograph (see ``_studies``).

    python examples/whistler_rocket.py [output.png]
"""

import os
import sys
import time

import numpy as np
from _studies import Study, box_mean
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from oilpaint import Canvas, Paint, Reference, brushes, surfaces
from oilpaint import palette as P

HERE = os.path.dirname(__file__)
REF = os.path.join(HERE, "reference", "whistler_falling_rocket.jpg")
W = 760

# the sparks: a dull gold, and a few orange
GOLD = Paint.mix((P.YELLOW_OCHRE, 2), (P.CADMIUM_YELLOW, 1), (P.RAW_UMBER, 0.3))
FIRE = Paint.mix((P.CADMIUM_ORANGE, 2), (P.YELLOW_OCHRE, 1))


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def spark_points(study):
    """Where the photograph shows sparks: small gold points clearly brighter
    than the night around them, above the horizon (the warm ground below is
    not sparks)."""
    img = study.ref.image.astype(float)
    img /= 255.0 if study.ref.image.dtype == np.uint8 else 1.0
    glow = img[..., 0] * 0.6 + img[..., 1] * 0.4 - img[..., 2] * 0.5
    ys = np.arange(study.c.height)[:, None]
    around = box_mean(img.mean(-1), 8)
    return (glow > 0.3) & (img.mean(-1) > around + 0.16) & (ys < 0.72 * study.c.height)


def sparks(study, lit, cell=36):
    """Where the photograph shows sparks (``lit``), flick paint: each patch of
    sparks is a spatter falling from just above it."""
    c = study.c
    img = study.ref.image.astype(float)
    img /= 255.0 if study.ref.image.dtype == np.uint8 else 1.0
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
                amount=min(0.003 * n, 0.3),  # about one drop per spark
                direction=90,
                spread=np.degrees(np.arctan2(spread_px, reach)),
                reach=reach,
                size=0.9,
                seed=int(study.rng.integers(1 << 31)),
            )
            flicks += 1
    return flicks, drops


def paint_it(path):
    t0 = time.time()
    with Image.open(REF) as im:
        h = round(W * im.height / im.width)
    ref = Reference(REF, W, h)
    c = Canvas(W, h, surface=surfaces.CANVAS.with_(ground="#1e2420"), seed=1875)
    study = Study(c, ref, seed=1875, body=0.15)  # thin, fluid "sauce"
    lit = spark_points(study)
    # the brushes leave the sparks alone: they are flicked on at the end
    night = box_mean(lit.astype(float), 6) < 1e-6
    log("washes")
    # long washes, up and down the canvas, then narrower sweeps the same way
    for size, threshold in ((80, 0.0), (40, 4.0), (20, 5.0)):
        study.layer(size, brushes.MOP, angle=np.pi / 2, threshold=threshold,
                    load=0.5, pressure=0.4, correct=0.8)  # fmt: skip
    c.dry()
    log("smoke and lights")
    # soft: the mop again, smaller, following the drifts of smoke
    study.layer(14, brushes.MOP, threshold=6.0, load=0.6, pressure=0.35, where=night)
    log("figures, the barge, the lights")
    study.layer(6, brushes.ROUND, threshold=9.0, length=3, load=0.8, pressure=0.45,
                where=night)  # fmt: skip
    c.dry()
    log("sparks")
    flicks, drops = sparks(study, lit)
    c.save(path, relief=0.3, gloss=0.4)  # thin paint, a matte night
    print(
        f"wrote {path}: {study.strokes} brush strokes, {flicks} flicks, "
        f"{drops} drops, in {time.time() - t0:.0f} s"
    )


if __name__ == "__main__":
    paint_it(
        sys.argv[1]
        if len(sys.argv) > 1
        else os.path.join(HERE, "output", "whistler_rocket.png")
    )
