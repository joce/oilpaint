"""The canvas: paint state, drying and rendering."""

from __future__ import annotations

import copy
import numbers
from collections.abc import Sequence

import numpy as np

from . import _engine as eng
from . import _render, brushes
from . import spectral as sp
from ._filters import _box_blur, _smooth121
from .brush import (
    IB,
    PACKED,
    Brush,
    BrushType,
    StrokePaint,
    _check_seed,
    _check_tool,
    _checked_loads,
    _number,
    _width,
    load_paint,
)
from .paint import DEFAULT_BODY, ColorLike, as_paint
from .png import write_png
from .spline import Profile
from .stroke import Stroke, apply_stroke
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

    Raises ValueError or TypeError unless ``width`` and ``height`` are whole
    numbers above 0, ``texture`` (if given) a finite number >= 0, ``weave``
    a finite number above 0, and ``seed`` something ``np.random.default_rng``
    takes (None, a whole number >= 0, ...).
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
        self.width, self.height = _size("width", width), _size("height", height)
        if texture is not None and _number("texture", texture) < 0:
            raise ValueError(f"texture must be 0 (flat) or more, got {texture}")
        if weave is not None:
            _width("weave", weave)
        _check_seed("seed", seed)
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

    def copy(self) -> Canvas:
        """An independent copy of the canvas: its paint, wet and dry, its
        surface and where its random numbers have got to, so the copy paints
        on exactly as the original would. Painting on one, or letting it
        dry, leaves the other as it was. ``copy.copy(canvas)`` does the same
        (a shallow copy would share the paint, which painting changes in
        place)."""
        return copy.deepcopy(self)

    __copy__ = copy

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
        prints sparse dots, a firmer one a fuller mark. Returns the brush.

        Raises ValueError or TypeError, before any paint is loaded, unless
        ``points`` are (x, y) pairs of finite numbers, ``pressure`` one
        finite number or one per point, ``width`` (if given) a finite number
        above 0, and ``brush`` None, a BrushType or a Brush."""
        _check_tool(brush)
        pts = _points("points", points, empty=True)
        press = np.asarray(pressure, dtype=float).reshape(-1)
        if press.size not in (1, len(pts)):
            raise ValueError(
                f"pressure must be one number or one per point ({len(pts)}), "
                f"got {press.size}"
            )
        if not np.isfinite(press).all():
            raise ValueError(f"pressure must be finite, got {pressure!r:.80}")
        if width is not None:
            _width("width", width)
        _number("amount", amount)
        if isinstance(brush, Brush):
            tool = brush
        else:
            tool = Brush(brush if isinstance(brush, BrushType) else brushes.STIPPLER)
        _checked_loads(tool, paint, amount)  # (before its size changes)
        if width is not None:  # the paint it takes depends on its size
            tool.width = float(width)
        load_paint(tool, paint, amount)
        press = np.broadcast_to(press, (len(pts),))
        for (x, y), p in zip(pts, press, strict=True):
            self.stroke([(x, y)], brush=tool, width=width, pressure=float(p), **options)
        return tool

    def spatter(
        self,
        at,
        paint: ColorLike | None,
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
        drops. Returns the number of drops laid (those landing on or by the
        canvas).

        With no paint (``paint`` None, or ``amount`` 0 or less) nothing flies
        and it returns 0. Raises ValueError or TypeError, before anything is
        painted, unless ``at`` is one or more (x, y) points of finite
        numbers, ``amount``, ``direction`` and ``spread`` finite numbers,
        ``reach`` and ``size`` finite numbers above 0, and ``seed`` something
        ``np.random.default_rng`` takes."""
        path = _points("at", at)
        # (checked, then used as given: as numpy numbers, if they are)
        _number("amount", amount)
        _number("direction", direction)
        _number("spread", spread)
        _width("reach", reach)
        _width("size", size)
        _check_seed("seed", seed)
        if paint is None:  # nothing to flick
            return 0
        as_paint(paint)
        rng = np.random.default_rng(
            seed if seed is not None else self._rng.integers(1 << 31)
        )
        n = round(120 * amount)
        if n <= 0:  # nothing flicked
            return 0
        # distance along the path the brush sweeps (a single point: none)
        arc = np.r_[0.0, np.cumsum(np.hypot(*np.diff(path, axis=0).T))]
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
        """Hex color of the paint film at pixel (x, y): the colour of the
        paint itself over the ground, without lighting, and without the
        dullness of paint sunk into an absorbent ground that
        ``render(lighting=False)`` shows (the two differ there in the darks).

        ``x`` and ``y`` are whole numbers, 0 <= x < width and 0 <= y < height
        (no counting from the end); anything else raises ValueError or
        TypeError."""
        x, y = _pixel("x", x, self.width), _pixel("y", y, self.height)
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
        light: Sequence[float] | None = (-0.6, -0.8, 1.1),
        relief: float = 1.0,
        gloss: float = 1.0,
    ) -> np.ndarray:
        """Render to an sRGB float image (H, W, 3) in 0..1.

        ``lighting`` shades the surface relief (canvas weave and impasto) with
        a raking light from ``light`` (x, y, z; y points down the image).
        ``relief`` scales the apparent height of paint, ``gloss`` the oily
        highlights. A reverse painting is seen through flat glass: no relief.

        When lit, raises ValueError, before any work, unless ``light`` is
        three finite numbers with z > 0 (the lamp in front of the picture),
        and ``relief`` and ``gloss`` are finite and not negative. Unlit (or
        reverse), they are not used and not checked.
        """
        lamp = None
        if lighting and not self.reverse:  # (else none of these is used)
            lamp = _check_light(light)
            for name, value in (("relief", relief), ("gloss", gloss)):
                if not (np.isfinite(value) and value >= 0):
                    raise ValueError(f"{name} must be finite and >= 0, got {value!r}")
        lin = self._render_linear()
        matte = self._matte()
        if matte is not None:
            # Paint sunk into an absorbent ground loses its gloss: the ~4% of light
            # a glossy surface reflects away is scattered back diffusely instead,
            # so darks look lighter and duller.
            lin = lin + (OIL_F0 * matte)[..., None] * (1.0 - lin)
        if lamp is not None:
            lin = self._light(lin, lamp, float(relief), float(gloss), matte)
        # to sRGB in place (lin is our own float32 array), in float64 a few
        # rows at a time
        for s in self._row_chunks():
            lin[s] = sp.linear_to_srgb(lin[s])
        return lin

    def _matte(self):
        """Per-pixel mattness of paint sunk into an absorbent ground (None if not
        absorbent)."""
        a = self.surface.absorbency
        if a <= 0:
            return None
        t = self.top_v + self.und_v + self.dry_height
        coverage = np.clip(t / MIN_FILM, 0.0, 1.0)
        return (a * np.exp(-t / SINK_DEPTH) * coverage).astype(np.float32)

    def _render_linear(self) -> np.ndarray:
        out = np.empty((self.height, self.width, 3), np.float32)
        if self.reverse:
            for s in self._row_chunks():
                out[s] = sp.reflectance_to_linear_rgb(self._reflectance_rows(s))
        else:
            _render.paint_rgb(
                self.top,
                self.und,
                self.top_a,
                self.und_a,
                self.ground,
                sp.SPECTRUM_TO_RGB,
                MIN_FILM,
                out,
            )
        return out

    def surface_height(self, relief: float = 1.0) -> np.ndarray:
        """Height of the painted surface in pixels, as used for lighting: the
        same surface the paint is pushed around on (the canvas weave, dried
        paint and wet paint)."""
        wet = self.top_v + self.und_v
        paint = wet + self.dry_height
        surface = self._floor + wet
        # Thin strokes thin out towards their edges (soft shoulders, low
        # relief); impasto keeps every ridge, groove and sharp edge, and bare
        # canvas its weave.
        broad = _blur(_blur(surface, 2), 2)
        fine = _smooth121(surface)
        detail = DETAIL_MIN + (1.0 - DETAIL_MIN) * _smoothstep(
            DETAIL_VAR[0], DETAIL_VAR[1], _blur(np.abs(fine - broad), 2)
        )
        detail = np.maximum(detail, np.exp(-paint / WEAVE_FILM))
        return ((HEIGHT_PX * relief) * (broad + detail * (fine - broad))).astype(
            np.float32
        )

    def _light(self, lin, light, relief, gloss, matte=None):
        """Light the linear colour ``lin`` (H, W, 3) as a painting under a
        studio lamp in the direction ``light``, seen from straight in front.
        Shaded with the colour multiplied, so flat paint lit by the lamp keeps
        exactly its colour:

        * the lamp: Lambert on the slope of each pixel (normalised so that
          flat paint gets 1), less where higher paint between it and the lamp
          casts a shadow. The lamp has a size (LIGHT_SIZE), so shadows are
          sharp at their foot and soften further from what casts them, and
          some of its light still gets round (SHADOW_DEPTH);
        * the room (AMBIENT): light from all around, less in grooves and
          hollows and at the foot of thick paint, from how high the surface
          stands around each pixel at a few scales (AO_SCALES); fine
          crevices also see less of the lamp (CAVITY);
        * gloss: the surface of the oil mirrors the lamp (Fresnel 4%) on the
          slopes that face halfway between the lamp and the viewer: a sharp
          highlight (GGX, ROUGH_WET; weaker and a little broader once dry) on
          the crests and lips of thick paint, spread by the curvature inside
          each pixel. Only facets near that mirror slope shine (GLOSS_CUT),
          less what flat paint shows, so paint that does not mirror the lamp
          keeps its colour: no veil of reflected light over it. Thin, lean
          paint barely shines (GLOSS_FILM), paint sunk into an absorbent
          ground less;
        * brighter-than-paint values roll off towards white rather than clip.
        """
        h = np.ascontiguousarray(self.surface_height(relief), dtype=np.float32)
        L = light / np.linalg.norm(light)
        lx, ly, lz = (float(v) for v in L)
        vis = np.empty_like(h)
        _render.shadows(h, lx, ly, lz, LIGHT_SIZE, SHADOW_REACH, vis)
        aos = np.empty((len(AO_SCALES),) + h.shape, np.float32)
        for k, (r, _) in enumerate(AO_SCALES):
            _render.box_blur(h, r, aos[k])
        wet = self.top_v + self.und_v
        out = np.empty_like(lin)
        _render.shade(
            h=h,
            vis=vis,
            aos=aos,
            ao=np.array(AO_SCALES, np.float64).reshape(-1, 2),
            lin=np.ascontiguousarray(lin, dtype=np.float32),
            paint=wet + self.dry_height,
            wet=wet,
            matte=matte if matte is not None else np.zeros((0, 0), np.float32),
            lx3=lx,
            ly3=ly,
            lz3=lz,
            ambient=AMBIENT,
            shadow_depth=SHADOW_DEPTH,
            cavity=CAVITY,
            ao_eps=AO_EPS,
            footprint=FOOTPRINT,
            oil_f0=OIL_F0,
            rough_wet=ROUGH_WET,
            rough_dry=ROUGH_DRY,
            gloss_wet=gloss * GLOSS_WET,
            gloss_dry=gloss * GLOSS_DRY,
            gloss_taper=GLOSS_TAPER,
            gloss_cut=GLOSS_CUT,
            film0=GLOSS_FILM[0],
            film1=GLOSS_FILM[1],
            dry0=GLOSS_DRY_FILM[0],
            dry1=GLOSS_DRY_FILM[1],
            out=out,
        )
        return out

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

#: Lighting (see Canvas._light). Share of the light that comes from all
#: around (the room) rather than from the lamp.
AMBIENT = 0.2
#: How much of the lamp's light a cast shadow takes away (the rest gets
#: round the paint, from the width of the lamp and the room).
SHADOW_DEPTH = 0.75
#: Angular radius of the lamp, in radians: the softness of cast shadows.
LIGHT_SIZE = 0.1
#: Furthest a cast shadow is looked for, in pixels: as far as the tallest
#: relief could cast one at the lamp's elevation, but no further than this
#: (beyond it, shadows fade out rather than stop).
SHADOW_REACH = 256.0
#: Darkening of fine crevices (the first AO scale) for all light, not only
#: the ambient share.
CAVITY = 0.8
#: Ambient occlusion: (radius in pixels, weight) of each scale.
AO_SCALES = ((2, 1.0), (8, 0.8), (24, 0.5))
#: Hollows shallower than this (times the radius) are not occluded: the
#: weave of bare canvas stays even.
AO_EPS = 0.04
#: Reflectance of the oil surface seen head-on (refractive index ~1.5).
OIL_F0 = 0.04
#: Roughness (GGX alpha) of wet and of dried oil paint.
ROUGH_WET, ROUGH_DRY = 0.08, 0.15
#: Strength of the highlight on wet and on dried paint (dried oil has sunk
#: and is far less glossy).
GLOSS_WET, GLOSS_DRY = 1.5, 0.25
#: The lamp is mirrored only by facets within GLOSS_CUT alphas (roughness)
#: of the mirror slope, tapering off smoothly from GLOSS_TAPER: the oil shows
#: highlights without hard edges, never a veil of reflected light over paint
#: that does not face the lamp's mirror direction.
GLOSS_TAPER, GLOSS_CUT = 0.5, 2.0
#: Wet paint thickness (in units of paint thickness) from which it starts to
#: shine, and from which it is fully glossy; the same for dried paint.
GLOSS_FILM = (0.5, 2.5)
GLOSS_DRY_FILM = (2.0, 6.0)
#: How much the spread of slopes across a pixel and its neighbours (its
#: curvature) widens its highlight: enough that a highlight on a steep ridge
#: fades out over a pixel or two instead of stopping between neighbours.
FOOTPRINT = 2.0

#: Share of the fine relief kept on gently modelled paint, and the local
#: relief (in units of paint thickness) from which all of it is kept.
DETAIL_MIN = 0.4
DETAIL_VAR = (0.15, 0.8)

#: Paint thinner than this (units of paint thickness) keeps the full detail
#: of the weave under it: a film that thin follows the threads.
WEAVE_FILM = 0.3
#: Paint about this thick (units of paint thickness) is mostly out of reach
#: of an absorbent ground: thinner paint sinks in and goes matte.
SINK_DEPTH = 2.0

#: Height of the canvas weave, in units of paint thickness: paint about this
#: thick fills the valleys between the threads.
WEAVE_DEPTH = 0.7

#: Paint thinner than this cannot form a continuous film; it is rendered as
#: partial coverage of a film this thick (dry-brush specks, stroke edges).
MIN_FILM = 0.15


def _size(name: str, value) -> int:
    """A canvas side in pixels: a whole number above 0."""
    v = _number(name, value)
    if v != int(v) or v <= 0:
        raise ValueError(
            f"canvas {name} must be a whole number of pixels above 0, got {value}"
        )
    return int(v)


def _pixel(name: str, value, size: int) -> int:
    """A pixel coordinate on a side ``size`` pixels long."""
    if not isinstance(value, numbers.Integral) or isinstance(value, bool):
        v = _number(name, value)
        if v != int(v):
            raise TypeError(f"{name} must be a whole number (a pixel), got {value}")
    value = int(value)
    if not 0 <= value < size:
        raise ValueError(f"{name} must be from 0 to {size - 1}, got {value}")
    return value


def _points(name: str, points, empty: bool = False) -> np.ndarray:
    """(x, y) points of finite numbers, as an (n, 2) array (n > 0 unless
    ``empty`` is allowed)."""
    try:
        pts = np.asarray(points, dtype=float)
    except (TypeError, ValueError):
        raise TypeError(f"{name} must be (x, y) points, got {points!r:.80}") from None
    if pts.size % 2 or (pts.ndim > 1 and pts.shape[-1] != 2):
        raise ValueError(f"{name} must be (x, y) points, got shape {pts.shape}")
    pts = pts.reshape(-1, 2)
    if not empty and len(pts) == 0:
        raise ValueError(f"{name} needs at least one (x, y) point")
    if not np.isfinite(pts).all():
        raise ValueError(f"{name} must be finite numbers, got {points!r:.80}")
    return pts


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


#: The lamp must stand at least this far above the picture's plane (the z of
#: its unit direction): any lower and its light would reach no flat paint.
MIN_LIGHT_Z = 1e-6


def _check_light(light) -> np.ndarray:
    """``light`` as a unit vector (float array of 3), or ValueError."""
    try:
        L = np.asarray(light, dtype=np.float64)
    except (TypeError, ValueError) as e:
        raise ValueError(f"light must be three numbers (x, y, z), got {light!r}") from e
    if L.shape != (3,) or not np.isfinite(L).all():
        raise ValueError(f"light must be three finite numbers (x, y, z): {light!r}")
    big = np.abs(L).max()
    if big > 0:
        L = L / big  # (scaled first, so the norm neither overflows nor underflows)
        L = L / np.linalg.norm(L)
    if not L[2] >= MIN_LIGHT_Z:
        raise ValueError(
            f"light must shine from in front of the picture (z > 0), got {light!r}"
        )
    return L


def _blur(a: np.ndarray, r: int) -> np.ndarray:
    """Box blur of a float32 map (compiled ``_box_blur``)."""
    a = np.ascontiguousarray(a, dtype=np.float32)
    out = np.empty_like(a)
    _render.box_blur(a, r, out)
    return out


def _smoothstep(e0, e1, x):
    t = np.clip((x - e0) / (e1 - e0), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)
