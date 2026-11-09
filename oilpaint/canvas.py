"""The canvas: paint state, drying and rendering."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from . import _engine as eng
from . import brushes
from . import spectral as sp
from ._filters import _box_blur, _smooth121
from .brush import (
    IB,
    PACKED,
    Brush,
    BrushType,
    Stroke,
    StrokePaint,
    apply_stroke,
    load_paint,
)
from .paint import DEFAULT_BODY, ColorLike
from .png import write_png
from .spline import Profile
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
    reverse:
        Paint on the back of a clear support (``surfaces.GLASS``) and look
        at it through the front: a reverse glass painting. The first paint
        laid is then in front: the outlines and details go on first, the
        background last. The brushes work as usual (on the back); the picture
        is seen through flat glass, without brush relief. Coordinates are as
        seen from the front, so nothing has to be painted mirrored. ``ground``
        is what shows through where there is no paint.

    Memory: about 0.3 KB per pixel (24 spectral bands for the dry ground and
    for each of the two wet sublayers); 0.5 KB painted in reverse.
    """

    def __init__(
        self,
        width: int,
        height: int,
        ground: ColorLike | None = None,
        texture: float | None = None,
        weave: float | None = None,
        seed: int | None = None,
        surface: Surface | None = None,
        reverse: bool = False,
    ):
        self.width, self.height = int(width), int(height)
        if self.width <= 0 or self.height <= 0:
            raise ValueError("canvas size must be positive")
        if reverse and not (surface or DEFAULT_SURFACE).transparent:
            raise ValueError(
                "reverse painting needs a clear support, such as surfaces.GLASS"
            )
        self.reverse = bool(reverse)
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
        H, W = self.height, self.width
        f32 = np.float32
        if image is not None and image.shape[:2] != (H, W):
            raise ValueError(
                f"ground image is {image.shape[1]}x{image.shape[0]}, canvas is {W}x{H}"
            )
        self.ground = self.surface.ground_reflectance(H, W, self._rng, image=image)
        if self.reverse:
            # The dried paint on the back of the glass, as one stack: its
            # reflectance seen through the glass and seen from behind, and
            # the share of light that gets through it and back out (see
            # _stack). Clear glass to begin with. The ground is behind it all.
            N = sp.N_BANDS
            self.glass_r = np.zeros((H, W, N), f32)
            self.glass_rb = np.zeros((H, W, N), f32)
            self.glass_t2 = np.ones((H, W, N), f32)
        # Wet sublayers, packed per pixel as [K (N bands), S, volume, body x
        # volume], all extensive (summed over the paint present). The
        # *_k/_s/_v names are views.
        self.top = np.zeros((H, W, PACKED), f32)
        self.und = np.zeros((H, W, PACKED), f32)
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
        self.top_k, self.top_s, self.top_v = (
            self.top[..., :N],
            self.top[..., N],
            self.top[..., N + 1],
        )
        self.und_k, self.und_s, self.und_v = (
            self.und[..., :N],
            self.und[..., N],
            self.und[..., N + 1],
        )

    def __getstate__(self):
        state = self.__dict__.copy()
        for name in ("top_k", "top_s", "top_v", "und_k", "und_s", "und_v", "_pending"):
            state.pop(name, None)
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.__dict__.setdefault("reverse", False)  # canvases saved before glass
        # canvases saved before coverage existed
        for name in ("top_a", "und_a"):
            if name not in self.__dict__:
                setattr(self, name, np.zeros((self.height, self.width), np.float32))
        for name in ("top", "und"):  # ... or before paint had a body
            layer = self.__dict__[name]
            if layer.shape[-1] < PACKED:
                body = (DEFAULT_BODY * layer[..., IB - 1])[..., None]
                self.__dict__[name] = np.concatenate(
                    [layer, body.astype(layer.dtype)], axis=-1
                )
        self._make_views()
        if not all(
            hasattr(self, n)
            for n in ("_floor", "_hollows", "_peaks_fine", "_peaks_wide")
        ):
            # maps derived from the surface, added in later versions
            self._update_tooth()

    # ------------------------------------------------------------------
    # painting
    # ------------------------------------------------------------------
    def stroke(
        self,
        path,
        paint: StrokePaint | None = None,
        amount: float = 6.0,
        width: Profile | None = None,
        pressure: Profile = 0.5,
        *,
        brush=None,
        **options,
    ) -> Brush:
        """Paint one stroke. See :class:`oilpaint.Stroke` for all parameters.

        ``path`` is a list of (x, y) points (a smooth spline is passed through
        them) or a :mod:`oilpaint.spline` path. ``brush`` is a
        :class:`oilpaint.BrushType` (e.g. ``brushes.FLAT_HOG``) or a
        :class:`oilpaint.Brush` carrying paint between strokes; ``width``
        defaults to the brush's own size. Returns the :class:`Brush`
        afterwards, which you can keep painting with.
        """
        return apply_stroke(
            self,
            Stroke(
                path=path,
                paint=paint,
                amount=amount,
                width=width,
                pressure=pressure,
                brush=brush,
                **options,
            ),
        )

    def stipple(
        self,
        points,
        paint: StrokePaint | None = None,
        amount: float = 6.0,
        width: float | None = None,
        pressure: Profile = 0.3,
        *,
        brush=None,
        **options,
    ) -> Brush:
        """Tap a brush straight down at each of ``points`` in turn, without
        reloading it: stippling (foliage, textures, broken colour, the dots of
        pointillism). The brush (``brushes.STIPPLER`` by default) turns a
        little on every tap, so no two prints are the same, and it runs out as
        it goes. ``pressure`` is one value, or one per point: a light tap
        prints sparse dots, a firmer one a fuller mark. Returns the brush."""
        pts = np.asarray(points, dtype=float).reshape(-1, 2)
        if isinstance(brush, Brush):
            tool = brush
        else:
            tool = Brush(brush if isinstance(brush, BrushType) else brushes.STIPPLER)
        if width is not None:  # the paint it takes depends on its size
            tool.width = float(width)
        load_paint(tool, paint, amount)
        press = np.broadcast_to(np.asarray(pressure, dtype=float), (len(pts),))
        for (x, y), p in zip(pts, press, strict=True):
            self.stroke([(x, y)], brush=tool, width=width, pressure=float(p), **options)
        return tool

    def spatter(
        self,
        at,
        paint: ColorLike,
        amount: float = 1.0,
        direction: float = 0.0,
        spread: float = 25.0,
        reach: float = 120.0,
        size: float = 3.0,
        seed: int | None = None,
    ) -> int:
        """Flick paint off a loaded brush: a spray of drops.

        ``at`` is where the paint leaves the brush, ``(x, y)``, or a path of
        points it leaves from along the way (a brush swept as it is flicked).
        The drops fly ``direction`` degrees (0 to the right, 90 down the
        canvas), fanned out by about ``spread`` degrees either way, and land
        about ``reach`` px away, thinning out further off. ``size`` is a
        typical drop's diameter in px: most are smaller, a few much bigger
        (with little satellite drops). Drops landing head-on stay round; those
        flying further come in at a slant and stretch into teardrops, their
        narrow tails pointing the way they flew. They sit on top of wet paint.
        ``amount`` sets how much paint flies: 1 is a good flick, about 120
        drops. Returns the number of drops."""
        rng = np.random.default_rng(
            seed if seed is not None else self._rng.integers(1 << 31)
        )
        path = np.asarray(at, dtype=float).reshape(-1, 2)
        # distance along the path the brush sweeps (a single point: none)
        arc = np.r_[0.0, np.cumsum(np.hypot(*np.diff(path, axis=0).T))]
        n = round(120 * amount)
        if n <= 0:  # nothing flicked
            return 0
        heading = np.deg2rad(direction)
        drops = 0
        for _ in range(n):
            if len(path) > 1:
                s = rng.uniform(0.0, arc[-1])
                x0 = float(np.interp(s, arc, path[:, 0]))
                y0 = float(np.interp(s, arc, path[:, 1]))
            else:
                x0, y0 = path[0]
            a = heading + np.deg2rad(spread) * rng.normal(0.0, 0.6)
            dist = reach * rng.gamma(2.0, 0.5)
            d = float(np.clip(size * rng.lognormal(0.0, 0.6), 0.7, 8.0 * size))
            ux, uy = np.cos(a), np.sin(a)
            x, y = x0 + dist * ux, y0 + dist * uy
            # flying further, a drop comes in at more of a slant and stretches
            stretch = 1.0 + 1.6 * min(dist / reach, 2.5) * rng.uniform(0.2, 1.0)
            drops += self._drop(x, y, d, stretch, ux, uy, paint, rng)
            if d > 2.0 * size:  # a big drop throws off little satellites
                for _ in range(int(rng.integers(1, 5))):
                    t = rng.uniform(1.0, 3.5) * d
                    side = rng.normal(0.0, 0.6) * d
                    drops += self._drop(
                        x + t * ux - side * uy,
                        y + t * uy + side * ux,
                        max(0.7, d * rng.uniform(0.15, 0.35)),
                        1.0,
                        ux,
                        uy,
                        paint,
                        rng,
                    )
        return drops

    def _drop(self, x, y, d, stretch, ux, uy, paint, rng) -> int:
        """One drop of paint landing at (x, y), ``d`` px across; stretched into
        a teardrop along (ux, uy) when ``stretch`` > 1."""
        if not (-d < x < self.width + d and -d < y < self.height + d):
            return 0
        kw = {
            "amount": 4.0 + 0.4 * d,
            "brush": _DROP.sized(d),
            "pressure": 0.6,
            "seed": int(rng.integers(1 << 31)),
        }
        if stretch < 1.25 or d < 1.5:
            self.stroke([(x, y)], paint, width=d, **kw)
        else:
            tail = d * (stretch - 1.0) * 0.9
            pts = [
                (x, y),
                (x + 0.5 * tail * ux, y + 0.5 * tail * uy),
                (x + tail * ux, y + tail * uy),
            ]
            self.stroke(pts, paint, width=[d, 0.7 * d, 0.25 * d], **kw)
        return 1

    def paint(self, *strokes: Stroke) -> None:
        """Apply one or more :class:`Stroke` objects, in order."""
        for s in strokes:
            apply_stroke(self, s)

    def dry(self) -> None:
        """Let all wet paint dry. Later strokes no longer mix with it; they
        layer over it optically (glazes show the dry paint through)."""
        for s in self._row_chunks():
            if self.reverse:
                self.glass_r[s], self.glass_rb[s], self.glass_t2[s] = self._stack(s)
            else:
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
        self._tooth = np.clip((tooth - lo) / max(hi - lo, 1e-6), 0, 1).astype(
            np.float32
        )
        # The solid surface under the wet paint, in units of paint thickness:
        # the weave plus dried paint. A rigid blade pressed on it rests on its
        # high points: the tops of the threads (a flexible blade) or the
        # highest ridges around (a stiff one).
        self._floor = (
            WEAVE_DEPTH * self.texture * self.grain + self.dry_height
        ).astype(np.float32)
        # How far each point lies below the local average surface (the hollows
        # of the weave and of dried paint): thick wet paint fills these first.
        r_mean = max(2, int(round(self.surface.scale)))
        self._hollows = (_box_blur(self._floor, r_mean) - self._floor).astype(
            np.float32
        )
        r_fine = max(1, int(round(0.5 * self.surface.scale)))
        self._peaks_fine = _box_blur(eng.max_filter(self._floor, r_fine), 1)
        self._peaks_wide = _box_blur(eng.max_filter(self._floor, 3 * r_fine + 3), 2)

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
        if self.reverse:
            # through the glass: the dried paint first, then the wet paint
            # behind it (the newest, ``top``, furthest back), then the ground
            rf, rb, t2 = self._stack(s)
            g = self.ground[s]
            return rf + t2 * g / (1.0 - rb * g)
        R = _composite(
            self.und_k[s], self.und_s[s], self.und_v[s], self.und_a[s], self.ground[s]
        )
        return _composite(self.top_k[s], self.top_s[s], self.top_v[s], self.top_a[s], R)

    def _stack(self, s):
        """Reverse painting: the paint on the back of the glass, seen through it,
        as ``(reflectance from the front, reflectance from behind, round trip
        through it)``. The dried stack, then each wet sublayer behind it."""
        rf, rb, t2 = self.glass_r[s], self.glass_rb[s], self.glass_t2[s]
        for k, sg, v, a in (
            (self.und_k[s], self.und_s[s], self.und_v[s], self.und_a[s]),
            (self.top_k[s], self.top_s[s], self.top_v[s], self.top_a[s]),
        ):
            r, tt = _film(k, sg, v, a)
            # light reflected back and forth between the stack and the film
            bounce = 1.0 / (1.0 - rb * r)
            rf, rb, t2 = (
                rf + t2 * r * bounce,
                r + tt * rb * bounce,
                t2 * tt * bounce * bounce,
            )
        return rf, rb, t2

    def _row_chunks(self):
        rows = max(1, 2_000_000 // max(self.width * sp.N_BANDS, 1))
        return [slice(y, y + rows) for y in range(0, self.height, rows)]

    def color_at(self, x: int, y: int) -> str:
        """Hex color of the paint at pixel (x, y), without lighting."""
        if self.reverse:
            R = self._reflectance_rows((slice(y, y + 1), slice(x, x + 1)))[0, 0]
            return sp.to_hex(sp.reflectance_to_srgb(R))
        R = _composite(
            self.und_k[y, x],
            self.und_s[y, x],
            self.und_v[y, x],
            self.und_a[y, x],
            self.ground[y, x],
        )
        R = _composite(
            self.top_k[y, x], self.top_s[y, x], self.top_v[y, x], self.top_a[y, x], R
        )
        return sp.to_hex(sp.reflectance_to_srgb(R))

    # ------------------------------------------------------------------
    # rendering
    # ------------------------------------------------------------------
    def render(
        self,
        lighting: bool = True,
        light: Sequence[float] = (-0.6, -0.8, 1.1),
        relief: float = 1.0,
        gloss: float = 1.0,
    ) -> np.ndarray:
        """Render to an sRGB float image (H, W, 3) in 0..1.

        ``lighting`` shades the surface relief (canvas weave and impasto) with
        a raking light from ``light`` (x, y, z; y points down the image).
        ``relief`` scales the apparent height of paint, ``gloss`` the oily
        highlights. A reverse painting is seen through flat glass: no relief.
        """
        lin = self._render_linear()
        matte = self._matte()
        if matte is not None:
            # Paint sunk into an absorbent ground loses its gloss: the ~4% of light
            # a glossy surface reflects away is scattered back diffusely instead,
            # so darks look lighter and duller.
            lin = lin + (0.04 * matte)[..., None] * (1.0 - lin)
        if lighting and not self.reverse:
            lin = self._light(lin, np.asarray(light, float), relief, gloss, matte)
        return sp.linear_to_srgb(lin).astype(np.float32)

    def _matte(self):
        """Per-pixel mattness of paint sunk into an absorbent ground (None if not
        absorbent)."""
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

    def surface_height(self, relief: float = 1.0) -> np.ndarray:
        """Height of the painted surface in pixels, as used for lighting: the
        same surface the paint is pushed around on (the canvas weave, dried
        paint and wet paint)."""
        paint = self.top_v + self.und_v + self.dry_height
        surface = self._floor + self.top_v + self.und_v
        # Thin strokes thin out towards their edges (soft shoulders, low
        # relief); impasto keeps every ridge, groove and sharp edge, and bare
        # canvas its weave.
        broad = _box_blur(_box_blur(surface, 2), 2)
        fine = _smooth121(surface)
        detail = 0.4 + 0.6 * _smoothstep(0.6, 2.0, _box_blur(np.abs(fine - broad), 2))
        detail = np.maximum(detail, np.exp(-paint / 0.3))
        return (HEIGHT_PX * relief) * (broad + detail * (fine - broad))

    def _light(self, lin, light, relief, gloss, matte=None):
        h = self.surface_height(relief)
        gy, gx = np.gradient(h)
        n = np.stack([-gx, -gy, np.ones_like(h)], -1)
        n /= np.linalg.norm(n, axis=-1, keepdims=True)
        L = light / np.linalg.norm(light)
        flat = L[2]
        ndotl = np.clip(n @ L, 0, None)
        # Raking light: impasto ridges cast shadows on the paint behind them...
        shadow = _cast_shadow(h, L)
        # ...and deep grooves and hollows get less light from all around.
        cavity = np.clip(_box_blur(h, 3) - h, 0.0, None)
        ambient = 1.0 - 0.35 * _smoothstep(0.3, 2.0, cavity)
        direct = (ndotl / flat) * (1.0 - 0.6 * shadow)
        diffuse = ambient * (0.2 + 0.8 * direct)
        # Oil gloss: a broad sheen plus sharp highlights on the slopes facing
        # the light (stronger while wet). Flat paint seen from the front shows none.
        Hv = L + np.array([0.0, 0.0, 1.0])
        Hv /= np.linalg.norm(Hv)
        flat_h = Hv[2]
        tilt = np.clip((np.clip(n @ Hv, 0, None) - flat_h) / (1 - flat_h), 0, None)
        paint_h = self.top_v + self.und_v + self.dry_height
        coverage = np.clip(paint_h / 0.5, 0, 1)
        wet = np.clip((self.top_v + self.und_v) / 0.5, 0, 1)
        spec = (
            gloss
            * coverage
            * (1.0 - shadow)
            * ((0.03 + 0.07 * wet) * tilt**3 + (0.05 + 0.15 * wet) * tilt**8)
        )
        if matte is not None:
            spec = spec * (
                1.0
                - 0.9
                * np.clip(matte / max(self.surface.absorbency, 1e-6), 0, 1)
                * self.surface.absorbency
            )
        return np.clip(lin * diffuse[..., None] + spec[..., None], 0, None)

    def save(self, path: str, **render_options) -> None:
        """Render and save as PNG."""
        img = self.render(**render_options)
        write_png(path, np.round(img * 255).astype(np.uint8))

    def to_pil(self, **render_options):
        from PIL import Image  # optional dependency

        return Image.fromarray(
            np.round(self.render(**render_options) * 255).astype(np.uint8)
        )


#: Rendered height of one unit of paint thickness, in pixels (``relief`` in
#: :meth:`Canvas.render` scales it).
HEIGHT_PX = 0.6

#: Height of the canvas weave, in units of paint thickness: paint about this
#: thick fills the valleys between the threads.
WEAVE_DEPTH = 0.7

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


#: A drop of paint landing (see Canvas.spatter): no bristles, it neither drags
#: nor picks up the paint it lands on, and sits on top of it.
_DROP = BrushType(
    shape="round",
    width=4.0,
    stiffness=0.5,
    bristles=1.0,
    streaks=0.0,
    pickup=0.0,
    drag=0.0,
    mixing=0.03,
    rough=0.0,
    name="Drop",
)


def _film(kappa, sigma, volume, alpha):
    """A wet sublayer as one film of a stack (reverse painting): its
    reflectance (the same from both sides) and the share of light that gets
    through it and back.

    Where it covers only part of the pixel (see _composite), the rest of the
    pixel is clear, and the stack treats the pixel as one averaged film. That
    average also bounces light back and forth with what lies behind it, as if
    the clear part reflected too; the light through the clear part is cut by
    as much, so a pixel never reflects more than comes in. Exact for a full
    film, and for an opaque one over black or white."""
    volume = np.asarray(volume, dtype=np.float32)
    alpha = np.asarray(alpha, dtype=np.float32)
    by_volume = volume / MIN_FILM
    cov = np.where(alpha > 0, np.minimum(alpha, by_volume), by_volume)
    cov = np.clip(cov, 1e-6, 1.0)
    r, t = sp.km_film(kappa / cov[..., None], sigma / cov)
    cov = cov[..., None]
    r_avg = cov * r
    return r_avg, cov * t * t + (1.0 - cov) * (1.0 - r_avg)


def _cast_shadow(h: np.ndarray, L: np.ndarray, max_steps: int = 32) -> np.ndarray:
    """0..1: how much each pixel is shadowed from the light direction ``L``
    by higher paint (heights ``h`` in pixels) between it and the light."""
    horiz = float(np.hypot(L[0], L[1]))
    if horiz < 1e-6:
        return np.zeros_like(h)
    lx, ly = L[0] / horiz, L[1] / horiz
    rise = L[2] / horiz  # how fast the light ray climbs per pixel
    span = float(h.max() - h.min())
    steps = int(min(max_steps, np.ceil(span / max(rise, 1e-3)) + 1))
    if steps < 1 or span < 0.05:
        return np.zeros_like(h)
    H, W = h.shape
    blocked = np.zeros_like(h)
    for t in range(1, steps + 1):
        dx, dy = int(round(t * lx)), int(round(t * ly))
        ys = np.clip(np.arange(H) + dy, 0, H - 1)
        xs = np.clip(np.arange(W) + dx, 0, W - 1)
        np.maximum(blocked, h[ys][:, xs] - h - t * rise, out=blocked)
    return np.clip(blocked / 0.8, 0.0, 1.0).astype(np.float32)


def _smoothstep(e0, e1, x):
    t = np.clip((x - e0) / (e1 - e0), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)
