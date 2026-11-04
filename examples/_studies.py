"""Shared by the studies after the masters: painting up a picture from a
photograph of it, coarse to fine, the way the Rembrandt study does (see
rembrandt_self_portrait.py). Each layer looks at the canvas so far and puts
strokes only where it still differs from the photograph, each following the
forms, in the colour that brings the canvas closer to it.
"""

from __future__ import annotations

import os
import sys
from collections.abc import Callable

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from oilpaint import Brush, Paint
from oilpaint import spectral as sp


def box_mean(img, r):
    """Mean over a (2r+1)-wide box (edges clamped), for (H, W) or (H, W, C)."""
    r = max(int(round(r)), 0)
    if r == 0:
        return img.astype(np.float64)
    out = img.astype(np.float64)
    for axis in (0, 1):
        pad = [(0, 0)] * out.ndim
        pad[axis] = (r + 1, r)
        c = np.cumsum(np.pad(out, pad, mode="edge"), axis=axis)
        hi = np.take(c, np.arange(2 * r + 1, c.shape[axis]), axis=axis)
        lo = np.take(c, np.arange(0, c.shape[axis] - 2 * r - 1), axis=axis)
        out = (hi - lo) / (2 * r + 1)
    return out


class Study:
    """A canvas painted up from a reference photograph. ``tone``, when set, is
    a function of an (H, W, 3) image that the photograph's colours pass
    through first: to lay a picture in darker than it is, say, and bring up
    the lights afterwards."""

    def __init__(self, canvas, ref, seed, body=0.6, levels=28):
        self.c, self.ref = canvas, ref
        self.rng = np.random.default_rng(seed)
        self.body, self.levels = body, levels
        self.paints: dict = {}
        self.strokes = 0
        self.tone: Callable[[np.ndarray], np.ndarray] | None = None
        ys, xs = np.mgrid[0 : canvas.height, 0 : canvas.width]
        self._xy = (xs, ys)

    def paint(self, rgb):
        """A paint for this colour (one per small step of colour)."""
        key = tuple(np.round(np.clip(rgb, 0, 1) * self.levels).astype(int))
        if key not in self.paints:
            self.paints[key] = Paint(np.asarray(key) / self.levels, body=self.body)
        return self.paints[key]

    def target(self, scale):
        """The photograph softened to ``scale``, as an (H, W, 3) image (passed
        through ``tone``, if set)."""
        img = np.asarray(self.ref.color(*self._xy, scale=scale), dtype=np.float64)
        return img if self.tone is None else self.tone(img)

    def gap(self, size, scale):
        """What the canvas still lacks, at the scale of a brush ``size`` wide:
        (target colours, the colour push towards them, the difference in delta E).
        The canvas is compared with the photograph as that brush sees it (both
        averaged over its width), so a crack thinner than the brush does not
        count as a difference the brush should paint."""
        target = self.target(scale)
        r = size / 2.0
        now = box_mean(self.c.render(lighting=False), r)
        seen = box_mean(target, r)
        err = np.linalg.norm(sp.srgb_to_lab(now) - sp.srgb_to_lab(seen), axis=-1)
        lacking = seen - now
        norm = np.linalg.norm(lacking, axis=-1, keepdims=True)
        lacking *= np.minimum(1.0, 0.15 / np.maximum(norm, 1e-9))
        return target, lacking, box_mean(err, r)

    def points(self, step, keep=None):
        """Jittered points about ``step`` apart, in random order."""
        H, W = self.c.height, self.c.width
        gy, gx = np.mgrid[step / 2 : H : step, step / 2 : W : step]
        px = np.clip(
            gx.ravel() + self.rng.uniform(-0.45, 0.45, gx.size) * step, 0, W - 1
        )
        py = np.clip(
            gy.ravel() + self.rng.uniform(-0.45, 0.45, gy.size) * step, 0, H - 1
        )
        ok = (
            np.ones(px.size, bool)
            if keep is None
            else keep[py.astype(int), px.astype(int)]
        )
        order = self.rng.permutation(np.flatnonzero(ok))
        return px[order], py[order]

    def layer(
        self,
        size,
        tool,
        *,
        spacing=0.7,
        scale=None,
        threshold=0.0,
        length=2.5,
        load=0.8,
        pressure=0.5,
        correct=0.6,
        where=None,
        tolerance=0.08,
        straight=False,
        colour=None,
        angle=None,
    ):
        """One layer of strokes from a ``tool`` ``size`` px wide: wherever the
        canvas differs from the photograph by more than ``threshold`` (delta E)
        at this scale, a stroke up to ``length`` widths long that follows the
        forms, in the photograph's colour pushed ``correct`` times past what the
        canvas lacks there (thin paint takes on the colour beneath it).
        ``where`` limits it to a mask; ``colour(rgb, x, y)`` can alter the
        colour; ``angle`` (radians) makes every stroke a straight one in that
        direction instead. Returns the number of strokes."""
        scale = max(size / 3.0, 1.0) if scale is None else scale
        target, lacking, err = self.gap(size, scale)
        keep = err > threshold
        if where is not None:
            keep &= where
        px, py = self.points(size * spacing, keep)
        H, W = self.c.height, self.c.width
        brush_type = tool.sized(size)
        for x, y in zip(px, py, strict=True):
            if angle is not None:
                a = angle + self.rng.normal(0, 0.12)
                d = np.array([np.cos(a), np.sin(a)]) * 0.5 * length * size
                path = np.array([[x, y] - d, [x, y] + d])
            else:
                path = self.ref.trace(
                    x, y, length * size, scale=max(scale, 1.5), tolerance=tolerance
                )
            if len(path) < 2 or np.linalg.norm(path[-1] - path[0]) < 0.6 * size:
                a, _ = self.ref.direction(x, y, max(scale, 1.5))
                a = float(a) + self.rng.normal(0, 0.3)
                d = np.array([np.cos(a), np.sin(a)]) * size * 0.45
                path = np.array([[x, y] - d, [x, y] + d])
            if straight:
                path = path[[0, -1]]
            yi, xi = int(min(y, H - 1)), int(min(x, W - 1))
            rgb = np.clip(target[yi, xi] + correct * lacking[yi, xi], 0.0, 1.0)
            if colour is not None:
                rgb = colour(rgb, x, y)
            run = float(np.linalg.norm(np.diff(path, axis=0), axis=1).sum())
            brush = Brush(brush_type).load(self.paint(rgb), load * (run / size + 1.0))
            self.c.stroke([tuple(p) for p in path], brush=brush, pressure=pressure)
            self.strokes += 1
        return len(px)
