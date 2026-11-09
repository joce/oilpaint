"""Painting from a reference image (a photo, a sketch, a painting to study).

A :class:`Reference` is the image scaled to the canvas. It answers what a
painter looks for before each stroke, at the scale of the brush in hand:

* :meth:`Reference.color` and :meth:`Reference.value`: the colour and the
  lightness there, softened to that scale (a big brush sees big shapes);
* :meth:`Reference.direction`: which way the forms run there (along the
  edges and the lines of equal light), so strokes can follow the form;
* :meth:`Reference.detail`: how much is going on there (a small brush, or a
  big one);
* :meth:`Reference.trace`: a curved stroke path from a point, following the
  forms for as long as the colour stays the same.

All take arrays of points as well as single ones. Coordinates are canvas
pixels. Only numpy and Pillow are needed::

    ref = Reference("portrait.jpg", canvas.width, canvas.height)
    x, y = 420, 310
    path = ref.trace(x, y, length=60, scale=6)
    canvas.stroke(path, paint=ref.color(x, y, scale=6), width=12)
"""

from __future__ import annotations

from collections import OrderedDict

import numpy as np

from . import spectral as sp


def _box(a, r, axis):
    """Box blur of radius r along an axis (edges repeat the border value)."""
    if r <= 0:
        return a
    pad = [(0, 0)] * a.ndim
    pad[axis] = (r + 1, r)
    c = np.cumsum(np.pad(a, pad, mode="edge"), axis=axis, dtype=np.float64)
    n = a.shape[axis]
    hi = np.take(c, np.arange(2 * r + 1, 2 * r + 1 + n), axis=axis)
    lo = np.take(c, np.arange(0, n), axis=axis)
    return ((hi - lo) / (2 * r + 1)).astype(a.dtype)


def _blur(a, sigma):
    """Gaussian blur (three box passes) over the first two axes."""
    if sigma <= 0.3:
        return a
    # box radii whose three passes add up to the Gaussian's variance
    r = max(int(round((np.sqrt(4.0 * sigma * sigma + 1.0) - 1.0) / 2.0)), 1)
    for _ in range(3):
        a = _box(_box(a, r, 0), r, 1)
    return a


def _sample(img, x, y):
    """Bilinear sample of img (H, W, ...) at points x, y."""
    H, W = img.shape[:2]
    x = np.clip(np.asarray(x, np.float64), 0, W - 1)
    y = np.clip(np.asarray(y, np.float64), 0, H - 1)
    x0, y0 = np.minimum(x.astype(int), W - 2), np.minimum(y.astype(int), H - 2)
    fx, fy = x - x0, y - y0
    if img.ndim == 3:
        fx, fy = fx[..., None], fy[..., None]
    return (img[y0, x0] * (1 - fx) + img[y0, x0 + 1] * fx) * (1 - fy) + (
        img[y0 + 1, x0] * (1 - fx) + img[y0 + 1, x0 + 1] * fx
    ) * fy


class Reference:
    """A reference image fitted to a ``width`` x ``height`` canvas.

    ``image`` is a path, a PIL image or an (H, W, 3) array (0..1 floats or
    0..255 ints). ``box`` optionally crops it first: (left, top, right,
    bottom) in the image's own pixels.

    Each softened map (one per kind of query and scale) is the size of the
    canvas, so only the ``max_maps`` used most recently are kept; one asked
    for again after that is made again. A painting works at a few scales at
    a time, so that is rare."""

    #: How many softened maps to keep (each one canvas-sized).
    max_maps: int = 16

    def __init__(self, image, width: int, height: int, box=None):
        from PIL import Image

        if isinstance(image, np.ndarray):
            arr = image.astype(np.float64)
            image = Image.fromarray(
                np.clip(arr * (255 if arr.max() <= 1.0 else 1), 0, 255).astype(np.uint8)
            )
        elif not hasattr(image, "convert"):
            image = Image.open(image)
        image = image.convert("RGB")
        if box is not None:
            image = image.crop((int(box[0]), int(box[1]), int(box[2]), int(box[3])))
        self.width, self.height = int(width), int(height)
        self.image = (
            np.asarray(
                image.resize((self.width, self.height), Image.Resampling.LANCZOS),
                np.float32,
            )
            / 255.0
        )
        lab = sp.srgb_to_lab(self.image.astype(np.float64))
        # perceptual, 0..1
        self.lightness = (lab[..., 0] / 100.0).astype(np.float32)
        self._cache: OrderedDict = OrderedDict()
        # the scale each map was first made at (a few floats per scale used)
        self._made_at: dict = {}

    def __setstate__(self, state) -> None:
        # References pickled before the maps were limited kept them in a plain
        # dict, and had no record of the scales they were made at.
        self.__dict__.update(state)
        self._cache = OrderedDict(self.__dict__.get("_cache", {}))
        self.__dict__.setdefault("_made_at", {})

    # ------------------------------------------------------------------ maps
    def _map(self, name, scale):
        key = (name, round(float(scale), 2))
        m = self._cache.get(key)
        if m is None:
            # Scales that round alike share a map. Made again after being
            # dropped, it is made at the same scale as the first time, so it is
            # the same map whatever was asked for in between.
            scale = self._made_at.setdefault(key, scale)
            m = self._cache[key] = getattr(self, "_make_" + name)(scale)
            while len(self._cache) > max(int(self.max_maps), 1):
                self._cache.popitem(last=False)
        else:
            self._cache.move_to_end(key)
        return m

    def _make_color(self, scale):
        return _blur(self.image, scale)

    def _make_value(self, scale):
        return _blur(self.lightness, scale)

    def _make_tensor(self, scale):
        L = _blur(self.lightness, max(scale, 1.0))
        gy, gx = np.gradient(L.astype(np.float64))
        # structure tensor, pooled over a neighbourhood twice the scale
        J = _blur(np.stack([gx * gx, gx * gy, gy * gy], axis=-1), 2.0 * max(scale, 1.0))
        jxx, jxy, jyy = J[..., 0], J[..., 1], J[..., 2]
        # the way the light changes fastest
        across = 0.5 * np.arctan2(2.0 * jxy, jxx - jyy)
        root = np.sqrt((jxx - jyy) ** 2 + 4.0 * jxy * jxy)
        coherence = (root / np.maximum(jxx + jyy, 1e-12)) ** 2
        return np.stack(
            [np.cos(across + np.pi / 2), np.sin(across + np.pi / 2), coherence], axis=-1
        )

    def _make_detail(self, scale):
        L = self.lightness.astype(np.float64)
        m = _blur(L, scale)
        return np.sqrt(np.maximum(_blur(L * L, scale) - m * m, 0.0)).astype(np.float32)

    # ------------------------------------------------------------------ queries
    def color(self, x, y, scale: float = 0.0):
        """sRGB colour (0..1) at (x, y), softened over about ``scale`` pixels."""
        return _sample(self._map("color", scale), x, y)

    def value(self, x, y, scale: float = 0.0):
        """Lightness (CIE L*, 0 = black .. 1 = white) at (x, y)."""
        return _sample(self._map("value", scale), x, y)

    def direction(self, x, y, scale: float = 4.0):
        """(angle, strength) of the forms at (x, y): the angle in radians of the
        direction along edges and lines of equal light (strokes laid this way
        follow the form), and how clear that direction is (0 = none, e.g. a
        flat area; 1 = a clean edge)."""
        t = _sample(self._map("tensor", scale), x, y)
        return np.arctan2(t[..., 1], t[..., 0]), t[..., 2]

    def detail(self, x, y, scale: float = 4.0):
        """How much the lightness varies around (x, y), within ``scale``
        pixels (0 = flat; ~0.1 and up = busy: eyes, edges, folds)."""
        return _sample(self._map("detail", scale), x, y)

    def trace(
        self,
        x,
        y,
        length: float,
        scale: float = 4.0,
        step: float | None = None,
        tolerance: float = 0.08,
        bend: float = 1.0,
    ):
        """A stroke path from (x, y) that follows the forms: it steps along
        :meth:`direction` (both ways, so the start ends up in the middle) for up
        to ``length`` pixels in all, stopping where the colour (at ``scale``)
        departs from the colour at the start by more than ``tolerance`` (sRGB
        distance) or the stroke would turn back on itself. ``bend`` < 1
        straightens it. Returns an (n, 2) array of points."""
        step = step or max(scale, 2.0)
        start_col = self.color(x, y, scale)
        a0, _ = self.direction(x, y, scale)
        d0 = np.array([np.cos(a0), np.sin(a0)])
        halves = []
        for sign in (1.0, -1.0):
            p, d, pts, run = np.array([x, y], np.float64), sign * d0, [], 0.0
            while run < 0.5 * length:
                a, _ = self.direction(p[0], p[1], scale)
                nd = np.array([np.cos(a), np.sin(a)])
                if nd @ d < 0:
                    nd = -nd
                nd = (1 - bend) * d + bend * nd
                nd /= max(np.linalg.norm(nd), 1e-9)
                if nd @ d < 0.5:  # too sharp a turn
                    break
                q = p + step * nd
                if not (0 <= q[0] < self.width and 0 <= q[1] < self.height):
                    break
                if (
                    np.linalg.norm(self.color(q[0], q[1], scale) - start_col)
                    > tolerance
                ):
                    break
                p, d, run = q, nd, run + step
                pts.append(p.copy())
            halves.append(pts)
        return np.array(halves[1][::-1] + [[float(x), float(y)]] + halves[0])
