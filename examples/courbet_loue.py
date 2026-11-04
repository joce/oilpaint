"""A study after Gustave Courbet's *The Source of the Loue* (1864).

Painted from a photograph of the painting (The Metropolitan Museum of Art,
New York; public domain; ``examples/reference``) the way Courbet painted his
rocks: on a dark ground, the masses and the dark clefts laid in with a broad
brush; then, once that is dry, the stone built up light over dark with a
sponge, loaded lightly and dabbed on in layers of value, each left to dry:
the middle tones over the rock, lighter prints only on its lit planes, and
the brightest pressed most lightly of all, so that the darker layers show
through the sponge's holes as the grain of the stone. The falling water is
laid on with the palette knife, and the darkest cracks drawn again with a
small brush. The brushwork and the knife go only where the canvas still
differs from the photograph (see ``_studies``).

    python examples/courbet_loue.py [output.png]
"""

import os
import sys
import time
from collections import defaultdict

import numpy as np
from _studies import Study
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from oilpaint import Canvas, Reference, brushes, surfaces

HERE = os.path.dirname(__file__)
REF = os.path.join(HERE, "reference", "courbet_source_of_the_loue.jpg")
W = 1000


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def sponged(study, size, spacing, scale, where, pressure, lift=1.0, load=0.6):
    """A layer of sponge prints about ``spacing`` apart over ``where``. The
    sponge is loaded lightly (dabbed off on the palette first: ``load`` per
    print) with the photograph's colour there, nudged towards what the canvas
    still lacks and ``lift``-ed (the darker layers show through its holes),
    and a few prints are made from each loading, the sponge turned each
    time."""
    target, lacking, _ = study.gap(size, scale)
    xs, ys = study.points(spacing, where)
    groups = defaultdict(list)
    paints = {}
    for x, y in zip(xs, ys, strict=True):
        yi, xi = int(y), int(x)
        rgb = (
            (target[yi, xi] + 0.4 * lacking[yi, xi])
            * lift
            * study.rng.uniform(0.95, 1.05)
        )
        paint = study.paint(rgb)
        groups[id(paint)].append((x, y))
        paints[id(paint)] = paint
    batches = []
    for key, pts in groups.items():
        batches.extend((paints[key], pts[k : k + 5]) for k in range(0, len(pts), 5))
    sponge = brushes.SPONGE.sized(size)
    for k in study.rng.permutation(len(batches)):
        paint, pts = batches[k]
        jitter = study.rng.uniform(0.85, 1.15, len(pts))
        study.c.stipple(
            pts, paint, amount=load * len(pts), pressure=pressure * jitter, brush=sponge
        )
    return len(xs)


def foam(study, where, threshold=5.0):
    """The falling water and its foam: small strokes of the palette knife,
    following the flow."""
    target, lacking, err = study.gap(10, 1.5)
    xs, ys = study.points(6, where & (err > threshold))
    knife = brushes.KNIFE_SMALL
    for x, y in zip(xs, ys, strict=True):
        a, _ = study.ref.direction(x, y, scale=4)
        d = study.rng.uniform(5, 10)
        pts = [
            (x - d * np.cos(a), y - d * np.sin(a)),
            (x + d * np.cos(a), y + d * np.sin(a)),
        ]
        yi, xi = int(y), int(x)
        paint = study.paint(target[yi, xi] + 0.6 * lacking[yi, xi])
        study.c.stroke(pts, paint, amount=5, brush=knife, pressure=0.5, angle=90)
    return len(xs)


def paint_it(path):
    t0 = time.time()
    with Image.open(REF) as im:
        h = round(W * im.height / im.width)
    ref = Reference(REF, W, h)
    # Courbet worked up from a dark brown ground
    c = Canvas(W, h, surface=surfaces.LINEN_FINE.with_(ground="#2e2620"), seed=1864)
    study = Study(c, ref, seed=1864, body=0.55)
    ys, xs = np.mgrid[0:h, 0:W]
    value = np.asarray(ref.value(xs, ys, scale=3))
    around = np.asarray(ref.value(xs, ys, scale=12))
    water = (value > 0.6) & (ys > 0.4 * h) & (xs > 0.2 * W)
    rock = ~water
    log("lay-in: the masses and the clefts")
    study.layer(56, brushes.FLAT_HOG, straight=True, load=0.5, pressure=0.5)
    study.layer(28, brushes.FLAT_HOG, threshold=5.0, load=0.5, pressure=0.5)
    c.dry()
    # the sponge, in layers of value, each let dry: the middle tones over the
    # rock, then lighter prints only where the stone catches the light (lighter
    # than around it), then the brightest, pressed most lightly
    n = 0
    layers = (
        ("middle tones", 40, 12, 5, rock & (value > 0.22), 0.35, 1.0),
        ("middle tones", 28, 9, 3, rock & (value > 0.3), 0.3, 1.0),
        ("the lit planes", 26, 8, 2.5, rock & (value > 0.42) & (value > around), 0.24,
         1.06),
        ("the lights", 18, 6, 1.5, rock & (value > 0.58) & (value > around + 0.04),
         0.17, 1.12),
    )  # fmt: skip
    for name, size, spacing, scale, where, pressure, lift in layers:
        log(f"sponge {size}px: {name} ({n} prints so far)")
        n += sponged(study, size, spacing, scale, where, pressure, lift)
        c.dry()
    log("knife: the falling water")
    k = foam(study, water, threshold=3.0)
    c.dry()
    log("the darkest cracks")
    study.layer(4, brushes.ROUND, threshold=10.0, length=4, load=0.6, pressure=0.45,
                where=value < 0.28)  # fmt: skip
    c.save(path, relief=0.45, gloss=0.5)
    print(f"wrote {path}: {study.strokes} brush strokes, {n} sponge prints, "
          f"{k} knife strokes, in {time.time() - t0:.0f} s")  # fmt: skip


if __name__ == "__main__":
    paint_it(
        sys.argv[1]
        if len(sys.argv) > 1
        else os.path.join(HERE, "output", "courbet_loue.png")
    )
