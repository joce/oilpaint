"""A study after Gustave Courbet's *The Source of the Loue* (1864).

Painted from a photograph of the painting (The Metropolitan Museum of Art,
New York; public domain; ``examples/reference``) the way Courbet painted his
rocks, light over dark: on a dark ground, the whole picture laid in with a
broad brush in its shadow tone (the lights brought down, the darks left as
they are); then, once that is dry, the lights brought up with a sponge,
loaded lightly and dabbed on in layers, each left to dry: broad prints on
the broad lit planes, smaller ones on the narrower lights, the brightest
pressed most lightly of all. A print goes only where all of it falls on
light, so the clefts and the shadows stay the dark lay-in, and the dark
shows through the sponge's holes as the grain of the stone. The foam on
the water is sponged too, and its brightest parts laid on with the palette
knife. The brushwork goes only where the canvas still differs from the
photograph (see ``_studies``).

    python examples/courbet_loue.py [output.png]
"""

import os
import sys
import time
from collections import defaultdict

import numpy as np
from _studies import Study
from PIL import Image, ImageFilter

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from oilpaint import Canvas, Reference, brushes, surfaces

HERE = os.path.dirname(__file__)
REF = os.path.join(HERE, "reference", "courbet_source_of_the_loue.jpg")
W = 1000


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def shadow_tone(img):
    """The photograph in its shadow tone: the lights brought down, the darks
    left about as they are."""
    lightness = np.clip(img.mean(-1, keepdims=True) / 0.5, 0.0, 1.0)
    return img * (1.0 - 0.55 * lightness)


def darkest(value, size):
    """The darkest value within about ``size`` px of each point."""
    k = max(int(size) // 2 * 2 + 1, 3)
    im = Image.fromarray(np.round(np.clip(value, 0, 1) * 255).astype(np.uint8))
    return np.asarray(im.filter(ImageFilter.MinFilter(k)), dtype=float) / 255.0


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
        d = study.rng.uniform(3, 7)
        pts = [
            (x - d * np.cos(a), y - d * np.sin(a)),
            (x + d * np.cos(a), y + d * np.sin(a)),
        ]
        yi, xi = int(y), int(x)
        paint = study.paint(target[yi, xi] + 0.6 * lacking[yi, xi])
        study.c.stroke(pts, paint, amount=3, brush=knife, pressure=0.5, angle=90)
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
    value = np.asarray(ref.value(xs, ys, scale=2))
    water = (value > 0.55) & (ys > 0.4 * h) & (xs > 0.2 * W)
    log("lay-in, in the shadow tone")
    study.tone = shadow_tone
    study.layer(56, brushes.FLAT_HOG, straight=True, load=0.5, pressure=0.5)
    study.layer(28, brushes.FILBERT_HOG, threshold=5.0, load=0.5, pressure=0.5)
    study.layer(14, brushes.FILBERT_HOG, threshold=7.0, load=0.5, pressure=0.5)
    study.tone = None
    c.dry()
    # the sponge brings up the lights, in layers, each let dry: a print only
    # where all of it falls on light enough stone (the darkest point under it
    # is light enough), so the clefts stay dark; each layer lighter, smaller
    # and pressed more lightly than the one before, on less of the rock (the
    # levels are the photograph's own: so much of the rock is lighter)
    rock = ~water

    def level(share):
        """The value that ``share`` of the rock is lighter than."""
        return float(np.quantile(value[rock], 1.0 - share))

    n = 0
    layers = (
        (36, 9, 4.0, 0.85, 0.75, 0.35, 1.05),
        (24, 7, 2.5, 0.70, 0.50, 0.30, 1.12),
        (16, 5, 1.5, 0.55, 0.30, 0.25, 1.2),
        (10, 4, 1.0, 0.40, 0.15, 0.20, 1.28),
    )
    for size, spacing, scale, floor, light, pressure, lift in layers:
        log(f"sponge {size}px: the lights ({n} prints so far)")
        under = darkest(value, 0.6 * size)
        where = rock & (under > level(floor)) & (value > level(light))
        n += sponged(study, size, spacing, scale, where, pressure, lift)
        c.dry()
    log("sponge: the foam")
    n += sponged(study, 22, 7, 2.0, water & (value > 0.6), 0.3, 1.05)
    n += sponged(study, 14, 5, 1.2, water & (value > 0.7), 0.22, 1.1)
    c.dry()
    log("knife: the brightest water")
    k = foam(study, water & (value > 0.85), threshold=6.0)
    c.save(path, relief=0.45, gloss=0.5)
    print(f"wrote {path}: {study.strokes} brush strokes, {n} sponge prints, "
          f"{k} knife strokes, in {time.time() - t0:.0f} s")  # fmt: skip


if __name__ == "__main__":
    paint_it(
        sys.argv[1]
        if len(sys.argv) > 1
        else os.path.join(HERE, "output", "courbet_loue.png")
    )
