"""The canvas: paint state, drying and rendering."""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

from . import spectral as sp
from .brush import Brush, Stroke, apply_stroke
from .paint import ColorLike
from .png import write_png
from .surface import DEFAULT_SURFACE, Surface


class Canvas:
    """A primed support you can paint on.

    Parameters
    ----------
    width, height:
        Size in pixels.
    ground:
        Color of the priming, or an sRGB image array (height, width, 3) in
        0..1 to paint over (a toned ground, or a photo as underpainting).
        Overrides the surface's own ground.
    texture:
        Shortcut for the surface's ``relief`` (0 = flat).
    weave:
        Shortcut for the surface's ``scale`` (thread spacing in pixels).
    seed:
        Random seed for the surface and stroke textures.
    surface:
        A :class:`oilpaint.Surface`, e.g. ``surfaces.LINEN_FINE`` or
        ``surfaces.GESSO_PANEL``: texture, tooth, absorbency, slickness and
        priming. Default: a generic primed canvas.

    Memory: about 0.3 KB per pixel (24 spectral bands for the dry ground and
    for each of the two wet sublayers).
    """

    def __init__(self, width: int, height: int, ground: Optional[ColorLike] = None,
                 texture: Optional[float] = None, weave: Optional[float] = None, seed: Optional[int] = None,
                 surface: Optional[Surface] = None):
        self.width, self.height = int(width), int(height)
        if self.width <= 0 or self.height <= 0:
            raise ValueError("canvas size must be positive")
        self._rng = np.random.default_rng(seed)
        surf = surface or DEFAULT_SURFACE
        image = None
        changes = {}
        if isinstance(ground, np.ndarray) and ground.ndim == 3:
            image = ground
        elif ground is not None:
            changes["ground"] = ground
        if texture is not None:
            changes["relief"] = float(texture)
        if weave is not None:
            changes["scale"] = float(weave)
        self.surface = surf.with_(**changes) if changes else surf
        H, W, N = self.height, self.width, sp.N_BANDS
        f32 = np.float32
        if image is not None and image.shape[:2] != (H, W):
            raise ValueError(f"ground image is {image.shape[1]}x{image.shape[0]}, canvas is {W}x{H}")
        self.ground = self.surface.ground_reflectance(H, W, self._rng, image=image)
        # Wet sublayers, packed per pixel as [K (N bands), S, volume], all
        # extensive (optical thickness). The *_k/_s/_v names are views.
        self.top = np.zeros((H, W, N + 2), f32)
        self.und = np.zeros((H, W, N + 2), f32)
        self._make_views()
        # Fraction of each pixel the wet paint covers (anti-aliased edges).
        self.top_a = np.zeros((H, W), f32)
        self.und_a = np.zeros((H, W), f32)
        self.top_id = np.zeros((H, W), np.int32)
        self.dry_height = np.zeros((H, W), f32)
        self.texture = float(self.surface.relief)
        self.grain = self.surface.height_map(H, W, self._rng).astype(f32)
        self._stroke_counter = 0
        self._update_tooth()

    def _make_views(self):
        N = sp.N_BANDS
        self.top_k, self.top_s, self.top_v = self.top[..., :N], self.top[..., N], self.top[..., N + 1]
        self.und_k, self.und_s, self.und_v = self.und[..., :N], self.und[..., N], self.und[..., N + 1]

    def __getstate__(self):
        state = self.__dict__.copy()
        for name in ("top_k", "top_s", "top_v", "und_k", "und_s", "und_v"):
            state.pop(name, None)
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        for name in ("top_a", "und_a"):          # canvases saved before coverage existed
            if name not in self.__dict__:
                setattr(self, name, np.zeros((self.height, self.width), np.float32))
        self._make_views()

    # ------------------------------------------------------------------
    # painting
    # ------------------------------------------------------------------
    def stroke(self, path, paint: Optional[ColorLike] = None, amount: float = 6.0,
               width=None, pressure=0.5, *, brush=None, **options) -> Brush:
        """Paint one stroke. See :class:`oilpaint.Stroke` for all parameters.

        ``path`` is a list of (x, y) points (a smooth spline is passed through
        them) or a :mod:`oilpaint.spline` path. ``brush`` is a
        :class:`oilpaint.BrushType` (e.g. ``brushes.FLAT_HOG``) or a
        :class:`oilpaint.Brush` carrying paint between strokes; ``width``
        defaults to the brush's own size. Returns the :class:`Brush`
        afterwards, which you can keep painting with.
        """
        return apply_stroke(self, Stroke(path=path, paint=paint, amount=amount, width=width,
                                         pressure=pressure, brush=brush, **options))

    def paint(self, *strokes: Stroke) -> None:
        """Apply one or more :class:`Stroke` objects, in order."""
        for s in strokes:
            apply_stroke(self, s)

    def dry(self) -> None:
        """Let all wet paint dry. Later strokes no longer mix with it; they
        layer over it optically (glazes show the dry paint through)."""
        for s in self._row_chunks():
            self.ground[s] = self._reflectance_rows(s)
        self.dry_height += self.top_v + self.und_v
        self.top.fill(0)
        self.und.fill(0)
        self.top_a.fill(0)
        self.und_a.fill(0)
        self.top_id.fill(0)
        self._update_tooth()

    def _new_stroke_id(self) -> int:
        self._stroke_counter += 1
        return self._stroke_counter

    def _update_tooth(self):
        # What a nearly-dry brush catches on: the weave, plus ridges of dry paint.
        relief = self.dry_height - _box_blur(self.dry_height, 6)
        tooth = self.texture * self.grain + 1.5 * relief
        tooth = tooth - _box_blur(tooth, 12) + 0.5 * self.texture
        lo, hi = np.percentile(tooth, [1, 99])
        self._tooth = np.clip((tooth - lo) / max(hi - lo, 1e-6), 0, 1).astype(np.float32)

    # ------------------------------------------------------------------
    # inspection
    # ------------------------------------------------------------------
    @property
    def wet_thickness(self) -> np.ndarray:
        """Wet paint thickness per pixel (1 = a normal covering layer)."""
        return self.top_v + self.und_v

    @property
    def thickness(self) -> np.ndarray:
        """Total paint thickness per pixel, wet and dry."""
        return self.top_v + self.und_v + self.dry_height

    def reflectance(self) -> np.ndarray:
        """Spectral reflectance of the painting, shape (H, W, N_BANDS)."""
        return self._reflectance_rows(slice(None))

    def _reflectance_rows(self, s) -> np.ndarray:
        R = _composite(self.und_k[s], self.und_s[s], self.und_v[s], self.und_a[s], self.ground[s])
        return _composite(self.top_k[s], self.top_s[s], self.top_v[s], self.top_a[s], R)

    def _row_chunks(self):
        rows = max(1, 2_000_000 // max(self.width * sp.N_BANDS, 1))
        return [slice(y, y + rows) for y in range(0, self.height, rows)]

    def color_at(self, x: int, y: int) -> str:
        """Hex color of the paint at pixel (x, y), without lighting."""
        R = _composite(self.und_k[y, x], self.und_s[y, x], self.und_v[y, x], self.und_a[y, x], self.ground[y, x])
        R = _composite(self.top_k[y, x], self.top_s[y, x], self.top_v[y, x], self.top_a[y, x], R)
        return sp.to_hex(sp.reflectance_to_srgb(R))

    # ------------------------------------------------------------------
    # rendering
    # ------------------------------------------------------------------
    def render(self, lighting: bool = True, light: Sequence[float] = (-0.6, -0.8, 1.1),
               relief: float = 1.0, gloss: float = 1.0) -> np.ndarray:
        """Render to an sRGB float image (H, W, 3) in 0..1.

        ``lighting`` shades the surface relief (canvas weave and impasto) with
        a raking light from ``light`` (x, y, z; y points down the image).
        ``relief`` scales the apparent height of paint, ``gloss`` the oily
        highlights.
        """
        lin = self._render_linear()
        matte = self._matte()
        if matte is not None:
            # Paint sunk into an absorbent ground loses its gloss: the ~4% of light
            # a glossy surface reflects away is scattered back diffusely instead,
            # so darks look lighter and duller.
            lin = lin + (0.04 * matte)[..., None] * (1.0 - lin)
        if lighting:
            lin = self._light(lin, np.asarray(light, float), relief, gloss, matte)
        return sp.linear_to_srgb(lin).astype(np.float32)

    def _matte(self):
        """Per-pixel mattness of paint sunk into an absorbent ground (None if not absorbent)."""
        a = self.surface.absorbency
        if a <= 0:
            return None
        t = self.top_v + self.und_v + self.dry_height
        coverage = np.clip(t / MIN_FILM, 0.0, 1.0)
        return (a * np.exp(-t / 2.0) * coverage).astype(np.float32)

    def _render_linear(self) -> np.ndarray:
        out = np.empty((self.height, self.width, 3), np.float32)
        for s in self._row_chunks():
            out[s] = sp.reflectance_to_linear_rgb(self._reflectance_rows(s))
        return out

    def _light(self, lin, light, relief, gloss, matte=None):
        paint_h = self.top_v + self.und_v + self.dry_height
        # Thin paint follows the weave; thick paint buries it.
        weave = 0.35 * self.texture * self.grain / (1.0 + 4.0 * paint_h)
        # Paint body with soft bevels at stroke edges, plus the fine brush marks.
        body = _box_blur(_box_blur(paint_h, 2), 2)
        marks = paint_h - _box_blur(paint_h, 1)
        h = weave + relief * (0.5 * body + 0.4 * marks)
        gy, gx = np.gradient(h)
        n = np.stack([-gx, -gy, np.ones_like(h)], -1)
        n /= np.linalg.norm(n, axis=-1, keepdims=True)
        L = light / np.linalg.norm(light)
        flat = L[2]
        ndotl = np.clip(n @ L, 0, None)
        diffuse = 1.0 + 0.8 * (ndotl - flat) / flat
        Hv = L + np.array([0.0, 0.0, 1.0])
        Hv /= np.linalg.norm(Hv)
        ndoth = np.clip(n @ Hv, 0, None)
        coverage = np.clip(paint_h / 0.5, 0, 1)
        wet = np.clip((self.top_v + self.und_v) / 0.5, 0, 1)
        flat_h = Hv[2]
        spec = gloss * (0.03 + 0.07 * wet) * coverage * np.clip((ndoth - flat_h) / (1 - flat_h), 0, None) ** 3
        if matte is not None:
            spec = spec * (1.0 - 0.9 * np.clip(matte / max(self.surface.absorbency, 1e-6), 0, 1)
                           * self.surface.absorbency)
        return np.clip(lin * diffuse[..., None] + spec[..., None], 0, None)

    def save(self, path: str, **render_options) -> None:
        """Render and save as PNG."""
        img = self.render(**render_options)
        write_png(path, np.round(img * 255).astype(np.uint8))

    def to_pil(self, **render_options):
        from PIL import Image  # optional dependency
        return Image.fromarray(np.round(self.render(**render_options) * 255).astype(np.uint8))


#: Paint thinner than this cannot form a continuous film; it is rendered as
#: partial coverage of a film this thick (dry-brush specks, stroke edges).
MIN_FILM = 0.15


def _composite(kappa, sigma, volume, alpha, below):
    """Reflectance of a wet sublayer over ``below``.

    ``alpha`` is the fraction of the pixel the brush covered (anti-aliased
    stroke edges): the paint forms a film of thickness volume / alpha over
    that fraction. Paint too thin to form a continuous film (dry-brush
    specks, smears) is also treated as partial coverage."""
    volume = np.asarray(volume, dtype=np.float32)
    alpha = np.asarray(alpha, dtype=np.float32)
    by_volume = volume / MIN_FILM
    cov = np.where(alpha > 0, np.minimum(alpha, by_volume), by_volume)
    cov = np.clip(cov, 1e-6, 1.0)
    film = sp.km_layer(kappa / cov[..., None], sigma / cov, below)
    cov = cov[..., None]
    return cov * film + (1.0 - cov) * below


def _box_blur(a: np.ndarray, r: int) -> np.ndarray:
    """Separable box blur with edge clamping (radius r)."""
    if r <= 0:
        return a
    out = a.astype(np.float64)
    for axis in (0, 1):
        pad = [(0, 0), (0, 0)]
        pad[axis] = (r + 1, r)
        c = np.cumsum(np.pad(out, pad, mode="edge"), axis=axis)
        n = out.shape[axis]
        hi = np.take(c, np.arange(2 * r + 1, 2 * r + 1 + n), axis=axis)
        lo = np.take(c, np.arange(0, n), axis=axis)
        out = (hi - lo) / (2 * r + 1)
    return out.astype(a.dtype)
