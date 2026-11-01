"""Brush model and the stroke simulation.

The brush is not simulated bristle by bristle. It is a round footprint that
carries paint in two compartments:

* a **reservoir** (the paint loaded deep in the brush), and
* a small **tip** buffer that touches the canvas.

At every step along the stroke the brush pushes some of the wet paint under it
forward, the tip lifts some of it, deposits a mix of what it holds, and is
refilled from the reservoir. Paint picked up from the canvas is therefore
smeared forward and fades out over a short distance, while a little of it
leaks into the reservoir (a dirty brush).

On the canvas, each pixel has a dry ground plus two wet sublayers: ``under``
(older wet paint, well mixed) and ``top`` (the paint of the most recent
stroke). Brush pressure stirs the two together; that is the difference between
laying color *over* wet paint and churning it *into* wet paint.

Paint on the brush is held in narrow bristle *columns* (:data:`COLUMNS`),
each with three depths: the tip that touches the canvas, the belly, and the
heel near the ferrule. On flat, filbert and fan brushes the columns run
across the width; on a round they go around the bundle, so rolling the
brush between the fingers turns different bristles to the canvas. That lets a
brush carry several paints side by side (double loading), on its tip or deep
in its heel, or only partly mixed; the paints blend into each other as the
brush is worked, at a rate set by :attr:`BrushType.blending`.

Paint quantities are stored *packed*: a vector of the absorption spectrum K
(N_BANDS values), then scattering S, then volume V, all extensive (summed over
the paint present). Mixing paint is then just adding vectors.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Optional, Union

import numpy as np

from . import spectral as sp
from ._filters import _box_blur, _max_filter
from .noise import aniso_noise, fractal_noise_2d, value_noise_1d
from .paint import ColorLike, Paint, as_paint
from .spline import Path, Profile, as_path, evaluate_profile

N = sp.N_BANDS
IS, IV, IB = N, N + 1, N + 2      # packed indices of scattering, volume and body (stiffness x volume)
PACKED = N + 3

# Width of the footprint's soft (anti-aliased) edge in pixels. Kept narrow:
# a wide soft edge would read as a thin film of paint, which tints differently.
_EDGE_PX = 1.5
# Paint volume (in brush-width squares of unit-thickness paint) per unit of
# `amount`, calibrated so that `amount` ~ brush widths of solid stroke.
_VOLUME_PER_AMOUNT = 0.9
# Paint a brush holds on to and never releases (in width squares).
_RESIDUAL = 0.03
#: Number of bristle columns across a brush. Each holds its own paint.
COLUMNS = 24
_COL_X = (np.arange(COLUMNS) + 0.5) / COLUMNS          # column centres, 0 = left edge, 1 = right edge
# On a round brush the columns go around the bundle: column b sits at angle
# _RING_A[b] (0 = facing the direction of travel, pi/2 = the brush's left side).
# Pressed down, the bundle splays its bristles evenly across the mark, the
# front and back of the bundle overlapping: a bristle at angle a lands at
# u = -(2/pi) asin(sin a) across the mark (-1 = left edge, +1 = right edge).
_RING_A = 2.0 * np.pi * (np.arange(COLUMNS) + 0.5) / COLUMNS
_RING_X = (1.0 - (2.0 / np.pi) * np.arcsin(np.sin(_RING_A))) / 2.0      # where across the mark each one lands
# Depth layers: tip [0, 0.12], belly [0.12, 0.6], heel [0.6, 1] of the bristle length.
_LAYER_EDGES = (0.0, 0.12, 0.6, 1.0)
# Paint up in the heel only comes down when the brush is pressed: at
# touchdown (more with more pressure), and whenever it is pressed harder
# during the stroke. It does not trickle out on its own.
_HEEL_SQUEEZE = 0.15
# Diffusion between neighbouring columns per brush width of travel, at blending 1 and full pressure.
_BLEND_RATE = 2.3

# --- impasto -------------------------------------------------------------------
# Paint (in load units: brush-width squares) the hair holds inside at medium
# pressure, per unit of capacity, for a brush 20 px wide or more (smaller
# brushes hold more for their size). Paint loaded beyond that sits on the
# outside as a glob: it goes on thick, and pressing squeezes it out at the
# edges of the stroke.
_HOLD = 11.0
# Extra layer thickness per load unit of glob, for paint of body 1.
_IMPASTO = 0.2
# How high a knife blade rides above the surface with no pressure (paint thickness units).
_KNIFE_GAP = 16.0
# Paint a knife carries on its blade beyond what fits under it (load units).
_KNIFE_CARRY = 1.5
# A knife rides no higher than this times the paint film under its blade.
_KNIFE_FILM = 0.5
# Radius (px) over which wet paint dragged by a blade moves together (viscosity).
_VISCOUS = 3
# How deep a knife sinks into wet paint already on the canvas, as a share of
# its thickness: pressure ** _KNIFE_SINK (light touch: skims; hard: nearly all).
_KNIFE_SINK = 1.8
# Share of an overload squeezed out at the edges per width of travel (knife, brush).
_SQUEEZE = {True: 0.03, False: 0.04}
# Steepest slope (paint thickness per pixel) that fluid (body 0) and very
# stiff (body 1) paint can hold before it flows.
_SLOPE_FLUID, _SLOPE_STIFF = 0.05, 4.0


def _deposit_thickness(load: float) -> float:
    """Layer thickness laid down by a brush holding ``load`` width-squares of paint.

    Roughly 1 (a normal, covering layer) for a well-loaded brush, thicker for a
    heavily loaded one (impasto), thinning out as the brush runs dry.
    """
    free = np.maximum(load - _RESIDUAL, 0.0)
    return free / (free + 0.3) * (1.0 + 0.3 * np.log1p(free / 6.0))


def _pack(paint: Paint, volume: float) -> np.ndarray:
    v = np.empty(PACKED)
    v[:N] = paint.K * volume
    v[IS] = paint.S * volume
    v[IV] = volume
    v[IB] = paint.body * volume
    return v


#: Blade outlines of palette and painting knives (see :class:`BrushType`).
KNIFE_SHAPES = ("trowel", "pear", "straight", "angled", "spoon")
SHAPES = ("round", "flat", "filbert", "fan") + KNIFE_SHAPES


@dataclass(frozen=True)
class BrushType:
    """The physical tool: shape, size and bristle character of a brush (or knife).

    Ready-made tools live in :mod:`oilpaint.brushes`; build your own or tweak a
    preset with :meth:`with_` / :meth:`sized`::

        from oilpaint import BrushType, brushes
        big_flat = brushes.FLAT_HOG.sized(48)
        my_brush = BrushType(shape="filbert", width=26, aspect=0.45, stiffness=0.6)

    Parameters
    ----------
    shape:
        Brushes: ``"round"``, ``"flat"`` (square ends), ``"filbert"`` (a flat
        with an oval tip) or ``"fan"`` (a thin spread of separate bristle
        clumps). Knives (a rigid steel blade, see :data:`KNIFE_SHAPES`):
        ``"trowel"`` (diamond, pointed tip), ``"pear"`` (teardrop, round
        tip), ``"straight"`` (long blade, rounded end), ``"angled"`` (end cut
        on a slant) or ``"spoon"`` (oval).
    width:
        Default width in pixels (a stroke's own ``width`` overrides it). For
        a knife, the length of the blade.
    aspect:
        Thickness of the footprint relative to its width: 1 for a round,
        about 0.2-0.3 for a flat. For a knife, the blade's width relative to
        its length. Thin flats lay crisp ribbons; thick ones are chunkier.
    angle:
        Degrees between the brush's long side and the stroke's cross
        direction: 0 drags a flat broadside (wide stroke), 90 drags it on its
        edge (thin line). Strokes can vary it along their length. A knife at
        0 is pulled sideways, the whole blade spreading paint; at 90 it is
        pulled along its length (a narrow mark the width of the blade).
    follow_path:
        If True the angle is relative to the stroke's direction (the brush
        turns with the hand). If False it is fixed on the canvas, measured from
        horizontal, like a calligraphy nib: the stroke thickens and thins as it
        changes direction.
    stiffness:
        0 = very soft (sable, squirrel) to 1 = very stiff (hog bristle, knife).
        Stiff tools leave stronger ridges and ragged edges, push more paint
        and blend less; soft ones lay smooth paint and blend gently. For a
        knife: a flexible blade (~0.5) bends under pressure, scrapes closer
        and spreads thinner; a stiff one (1) rides on the highest points.
    bristles:
        Bristle density: 0 = sparse and coarse (bold streaks, dry-brush
        breakup), 1 = dense and fine (smooth).
    capacity:
        How much paint sits in the tip of the brush: above 1 ("fat") the brush
        keeps colour it picked up and smears it further; below 1 it is quickly
        cleaned by its own fresh paint. How far a stroke reaches is set by the
        stroke's ``amount``.
    spread:
        How much the brush widens under pressure (0..1). Soft rounds flatten a
        lot; flats very little.
    clumps:
        Number of separate bristle clumps (fan brushes), or of teeth on a
        comb or serrated knife; 0 = continuous.
    slant:
        Bristles cut on a bias (0..1): angled brushes (~0.5), daggers (~0.9),
        deer foot (~0.3). The toe touches first: light pressure paints only
        with the toe side, heavier pressure brings the heel down.
    softness:
        Softness of the stroke edges as a fraction of the brush width (mops,
        soft blenders); 0 = crisp.
    deposits:
        False for tools that only lift paint (a rag wiping back to the ground).
    blending:
        0..1: how readily different paints loaded on the brush merge into each
        other as it is used. Low (soft synthetic flats) keeps a double-loaded
        brush crisply two-toned for a long stroke; high (scrubby hog) blends
        them within a few widths. Pressure speeds it up. Default: derived from
        stiffness.
    taper:
        Default feathered ends for strokes made with this brush.
    streaks, pickup, drag, mixing, rough:
        Explicit physics overrides. ``None`` derives them from stiffness and
        bristles (``rough`` is the raggedness of the stroke edges).
    """

    shape: str = "round"
    width: float = 20.0
    aspect: float = 1.0
    angle: float = 0.0
    follow_path: bool = True
    stiffness: float = 0.5
    bristles: float = 0.5
    capacity: float = 1.0
    spread: float = 0.0
    clumps: int = 0
    slant: float = 0.0
    softness: float = 0.0
    deposits: bool = True
    blending: Optional[float] = None
    taper: float = 0.0
    streaks: Optional[float] = None
    pickup: Optional[float] = None
    drag: Optional[float] = None
    mixing: Optional[float] = None
    rough: Optional[float] = None
    name: str = "brush"

    def __post_init__(self):
        if self.shape not in SHAPES:
            raise ValueError(f"unknown brush shape {self.shape!r}; use one of {SHAPES}")
        if self.width <= 0 or self.aspect <= 0:
            raise ValueError("width and aspect must be positive")

    def with_(self, **changes) -> "BrushType":
        """A copy with some properties changed."""
        return replace(self, **changes)

    def sized(self, width: float) -> "BrushType":
        """The same brush in another size."""
        return replace(self, width=float(width))

    @property
    def rigid(self) -> bool:
        """True for knives: a rigid blade rather than bristles."""
        return self.shape in KNIFE_SHAPES

    def physics(self) -> dict:
        """Stroke behaviour derived from the brush's character."""
        st, br = self.stiffness - 0.5, self.bristles - 0.5
        if self.rigid:      # smooth steel: no streaks, lays paint on top, little mixing
            derived = dict(streaks=0.03, pickup=0.05, drag=0.05, mixing=0.12, rough=0.04)
        else:
            derived = dict(
                streaks=0.35 + 0.5 * st - 0.3 * br,
                pickup=0.35 - 0.2 * st,
                drag=0.2 + 0.3 * st,
                mixing=0.35 + 0.2 * st,
                rough=0.1 + 0.2 * st - 0.1 * br,
            )
        for key in derived:
            override = getattr(self, key)
            derived[key] = float(np.clip(override if override is not None else derived[key], 0.0, 2.0))
        derived["smear"] = max(self.capacity, 0.05)
        blend = self.blending if self.blending is not None else (0.12 if self.rigid else 0.4 + 0.5 * st)
        derived["blending"] = float(np.clip(blend, 0.0, 1.0))
        # a slight build-up of paint at the edges of a brushed stroke
        derived["ridge"] = 0.0 if self.rigid else 0.3 + 0.4 * st
        derived["lift"] = 0.6 if self.rigid else 0.0
        return derived


DEFAULT_BRUSH = BrushType(name="round")


def _depth_weights(depth) -> np.ndarray:
    """Share of a load that goes to the tip, the belly and the heel."""
    if isinstance(depth, str):
        named = {"tip": (1.0, 0.0, 0.0), "belly": (0.0, 1.0, 0.0), "heel": (0.0, 0.0, 1.0)}
        if depth not in named:
            raise ValueError("depth must be 'tip', 'belly', 'heel' or a dip depth from 0 to 1")
        return np.array(named[depth])
    d = float(depth)
    if not 0.0 < d <= 1.0:
        raise ValueError("a dip depth must be between 0 (not dipped) and 1 (up to the ferrule)")
    e = _LAYER_EDGES
    w = np.array([max(0.0, min(d, e[k + 1]) - e[k]) for k in range(3)])
    return w / w.sum()


def _across_weights(across, ring: bool = False) -> np.ndarray:
    """Share of a load that goes into each bristle column (sums to 1).

    ``across`` says where on the brush the paint goes, left to right as seen
    looking along the stroke: ``None``/``"all"``, ``"left"``, ``"right"``,
    ``"center"``, ``"edges"``; ``("left", 0.3)`` = the leftmost 30%;
    ``(a, b)`` = from a to b (0 = left edge, 1 = right edge); a function of
    x in 0..1; or one weight per column.
    """
    x = _RING_X if ring else _COL_X
    soft = 0.06

    def band(a, b):
        return _smoothstep(a - soft, a + soft, x) * _smoothstep(b + soft, b - soft, x)

    if across is None or (isinstance(across, str) and across == "all"):
        w = np.ones(COLUMNS)
    elif isinstance(across, str):
        named = {"left": (0.0 - 1, 0.5), "right": (0.5, 1.0 + 1), "center": (0.3, 0.7), "middle": (0.3, 0.7)}
        if across == "edges":
            w = band(-1, 0.25) + band(0.75, 2)
        elif across in named:
            w = band(*named[across])
        else:
            raise ValueError(f"unknown side {across!r}; use left, right, center, edges or all")
    elif callable(across):
        w = np.asarray([float(across(v)) for v in x])
    elif isinstance(across, tuple) and len(across) == 2 and isinstance(across[0], str):
        side, frac = across[0], float(across[1])
        if side == "left":
            w = band(-1, frac)
        elif side == "right":
            w = band(1.0 - frac, 2)
        else:
            raise ValueError("use ('left', fraction) or ('right', fraction)")
    else:
        arr = np.asarray(across, dtype=np.float64).reshape(-1)
        if arr.size == 2:
            w = band(float(arr[0]), float(arr[1]))
        elif arr.size == COLUMNS:
            w = arr
        else:
            raise ValueError(f"across: give (start, end) in 0..1 or {COLUMNS} column weights")
    w = np.clip(w, 0.0, None)
    if w.sum() <= 0:
        raise ValueError("that part of the brush is empty: no column gets any paint")
    return w / w.sum()


def _diffuse(cols: np.ndarray, k: float, ring: bool = False) -> None:
    """Spread paint between neighbouring columns (in place, conserving it).
    On a ring (a round brush) the last column neighbours the first."""
    if k <= 0:
        return
    n_sub = int(np.ceil(k / 0.24))
    kk = k / n_sub
    for _ in range(n_sub):
        if ring:
            cols += kk * (np.roll(cols, -1, axis=0) - 2.0 * cols + np.roll(cols, 1, axis=0))
        else:
            flow = np.diff(cols, axis=0)          # from column b+1 into b
            cols[:-1] += kk * flow
            cols[1:] -= kk * flow


def _seep(brush, rate: float) -> None:
    """Squeeze paint from the heel down into the belly (more when the belly is emptier)."""
    hv, bv = brush.heel[:, IV], brush.reservoir[:, IV]
    move_v = rate * np.maximum(hv - bv, 0.0) * 0.5
    ok = (move_v > 0) & (hv > 0)
    if not ok.any():
        return
    moved = brush.heel * np.where(ok, move_v / np.where(hv > 0, hv, 1.0), 0.0)[:, None]
    brush.heel -= moved
    brush.reservoir += moved


def _swap_layers(tip: np.ndarray, belly: np.ndarray, rate: float) -> None:
    """Exchange paint between the tip and the belly of each column (in place)."""
    if rate <= 0:
        return
    tv, bv = tip[:, IV], belly[:, IV]
    e = rate * np.minimum(tv, bv)
    ok = e > 0
    if not ok.any():
        return
    ft = np.where(ok, e / np.where(tv > 0, tv, 1.0), 0.0)[:, None]
    fb = np.where(ok, e / np.where(bv > 0, bv, 1.0), 0.0)[:, None]
    from_tip, from_belly = tip * ft, belly * fb
    tip += from_belly - from_tip
    belly += from_tip - from_belly


class Brush:
    """A brush with paint on it, which it keeps (with any dirt) between strokes.

    Paint sits in bristle columns across the brush, so a brush can carry
    several paints at once::

        brush = Brush(brushes.FLAT_SABLE)
        brush.load(P.CADMIUM_RED, amount=6)                      # the whole brush
        brush.load(P.TITANIUM_WHITE, amount=2, across="left")    # double-load one side
        brush.load(P.CADMIUM_YELLOW, amount=0.5, depth="tip")    # just the tips
        brush.load(P.BURNT_UMBER, amount=3, depth=1.0)           # dipped right up to the ferrule
        brush.work(1)                                            # stroke it on the palette
        canvas.stroke(path, brush=brush)
        canvas.stroke(path2, brush=brush)                        # continues with what's left
        brush.clean()

    Paint in the heel doesn't flow to the tip by itself: some of it is
    squeezed down when the brush touches the canvas (more when pressed hard)
    and when the pressure increases during a stroke. The rest stays in the
    brush, tinting later loads, until it is cleaned.
    """

    def __init__(self, type: Optional[BrushType] = None):
        self.type = type or DEFAULT_BRUSH
        self.reservoir = np.zeros((COLUMNS, PACKED))   # the belly of each column
        self.tip = np.zeros((COLUMNS, PACKED))         # paint at the tips of each column
        self.heel = np.zeros((COLUMNS, PACKED))        # paint deep in the heel, by the ferrule
        self.width = float(self.type.width)
        self.wet = 0.0          # how coated the bristles are from recent wet contact

    # ------------------------------------------------------------------ loading
    def load(self, paint: ColorLike, amount: float, width: Optional[float] = None, across=None,
             depth: Union[str, float] = "belly") -> "Brush":
        """Add ``amount`` of paint (same units as :attr:`Stroke.amount`).

        ``across`` puts it on part of the brush: ``"left"``, ``"right"``,
        ``"center"``, ``"edges"``, ``("left", 0.3)``, ``(0.2, 0.6)``, a
        function of x in 0..1 (left to right), or column weights. Left and
        right are the brush's own sides, as seen looking along the stroke
        (on a round, the sides of the bundle).

        ``depth`` is where in the bristles it goes: ``"belly"`` (dipped in:
        the bulk of the load, the default), ``"tip"`` (just touched to the
        paint: comes off first), ``"heel"`` (worked up to the ferrule: only
        squeezed out by pressing, see :class:`Brush`), or a dip depth from 0
        to 1 (fraction of the bristle length), spread over the layers it
        reaches: 0.1 is the tip, 0.5 tip and belly, 1.0 all three.
        """
        if width is not None:
            self.width = float(width)
        volume = max(float(amount), 0.0) * _VOLUME_PER_AMOUNT * self.width ** 2
        if volume <= 0:
            return self
        cols = self._spread_load(across)[:, None] * _pack(as_paint(paint), volume)[None, :]
        for layer, w in zip(self._layers(), _depth_weights(depth)):
            if w > 0:
                layer += w * cols
        return self

    def load_mix(self, parts, amount: float, mixing: float = 0.5, across=None, depth: Union[str, float] = "belly",
                 seed: Optional[int] = None, width: Optional[float] = None) -> "Brush":
        """Pick up several paints at once, only partly mixed on the palette.

        ``parts`` is a list of ``(paint, proportion)``. ``mixing`` = 1 gives a
        uniform mixture (like :meth:`Paint.mix`); lower values leave each
        paint in irregular streaks across the bristles, so the stroke comes
        out marbled; 0 = barely touched together.
        """
        if width is not None:
            self.width = float(width)
        paints = [as_paint(p) for p, _ in parts]
        props = np.array([max(float(w), 0.0) for _, w in parts])
        if props.sum() <= 0:
            raise ValueError("load_mix needs positive proportions")
        props = props / props.sum()
        mixing = float(np.clip(mixing, 0.0, 1.0))
        rng = np.random.default_rng(seed)
        cell = COLUMNS / (2.0 + 5.0 * (1.0 - mixing))         # less mixed: more, narrower streaks
        noise = np.stack([value_noise_1d(COLUMNS, cell, rng) for _ in paints])
        logits = np.log(props)[:, None] + 12.0 * (1.0 - mixing) * (noise - 0.5)
        w = np.exp(logits - logits.max(axis=0, keepdims=True))
        w /= w.sum(axis=0, keepdims=True)
        for _ in range(20):      # keep each paint's overall share while keeping the streaks
            w *= (props * COLUMNS / np.maximum(w.sum(axis=1), 1e-12))[:, None]
            w /= w.sum(axis=0, keepdims=True)
        w = mixing * props[:, None] + (1.0 - mixing) * w
        volume = max(float(amount), 0.0) * _VOLUME_PER_AMOUNT * self.width ** 2
        col_vol = volume * self._spread_load(across)
        for layer, dw in zip(self._layers(), _depth_weights(depth)):
            if dw <= 0:
                continue
            for i, paint in enumerate(paints):
                layer += (dw * w[i] * col_vol)[:, None] * _pack(paint, 1.0)[None, :]
        return self

    def work(self, passes: float = 1.0, pressure: float = 0.5) -> "Brush":
        """Stroke the brush back and forth on the palette without painting:
        the paints on it blend into each other (side to side, and tip with
        belly) as they would over a few widths of real strokes."""
        blend = self.type.physics()["blending"]
        widths = 3.0 * max(float(passes), 0.0)
        k = _BLEND_RATE * blend * (0.2 + pressure) ** 2 * widths
        for layer in self._layers():
            _diffuse(layer, k, ring=self.ring)
        for _ in range(int(np.ceil(widths))):
            _swap_layers(self.tip, self.reservoir, min(0.375 * blend, 0.5))
            _swap_layers(self.reservoir, self.heel, min(0.1 * blend, 0.5))
        return self

    def clean(self) -> "Brush":
        self.__init__(self.type)
        return self

    def _spread_load(self, across) -> np.ndarray:
        """Share of a load each column gets."""
        return _across_weights(across, ring=self.ring)

    @property
    def ring(self) -> bool:
        """True for a round brush, whose bristle columns go around the bundle."""
        return self.type.shape == "round"

    def _layers(self):
        return (self.tip, self.reservoir, self.heel)

    # ------------------------------------------------------------------ inspection
    @property
    def volume(self) -> float:
        """Total paint volume on the brush."""
        return float(self.reservoir[:, IV].sum() + self.tip[:, IV].sum() + self.heel[:, IV].sum())

    @property
    def amount(self) -> float:
        """Paint left on the brush, in the units of :attr:`Stroke.amount`."""
        return self.volume / (_VOLUME_PER_AMOUNT * self.width ** 2)

    @property
    def paint(self) -> Optional[Paint]:
        """The overall mixture of paint on the brush."""
        total = self.reservoir.sum(axis=0) + self.tip.sum(axis=0) + self.heel.sum(axis=0)
        if total[IV] <= 0:
            return None
        return Paint.from_optics(total[:N] / total[IV], total[IS] / total[IV],
                                 body=float(np.clip(total[IB] / total[IV], 0.0, 1.0)))

    def colors(self, depth: str = "all") -> list:
        """Hex colour of each bristle column (None where empty): left to right
        on a flat; around the bundle on a round, starting from its front and
        turning towards its left side. ``depth`` is ``"tip"``, ``"belly"``,
        ``"heel"`` or ``"all"``."""
        cols = {"tip": self.tip, "belly": self.reservoir, "heel": self.heel,
                "all": self.tip + self.reservoir + self.heel}[depth]
        out = []
        for c in cols:
            if c[IV] <= 1e-9:
                out.append(None)
            else:
                out.append(Paint.from_optics(c[:N] / c[IV], c[IS] / c[IV]).hex)
        return out


@dataclass
class Stroke:
    """One brush stroke.

    Parameters
    ----------
    path:
        The stroke's spine. A list of (x, y) points (interpolated with a smooth
        Catmull-Rom spline that passes through them) or any
        :class:`oilpaint.spline.Path` (``CatmullRom``, ``Bezier``,
        ``Polyline``). Pixel coordinates, origin top-left. A single point
        makes a dab.
    paint:
        A :class:`Paint`, or a plain color (``"#rrggbb"`` / RGB triple).
        ``None`` paints with a clean (or the given) brush without loading
        paint, which just drags and blends the wet paint. A *list* loads
        several paints: each item is ``(paint, amount[, across[, depth]])``
        as in :meth:`Brush.load`, e.g. ``[(red, 6), (white, 2, "left")]``.
    amount:
        Paint loaded on the brush, measured in *brush-width squares at normal
        thickness*. Roughly: ``amount=6`` paints about 6 brush widths of
        solid stroke before the brush starts running dry. More paint also
        means a thicker layer (impasto).
    width:
        Brush width in pixels: a number, one value per control point, or a
        function of u (0..1 along the stroke). Defaults to the brush's width.
    pressure:
        0..1 (default 0.5), same forms as ``width``. More pressure releases
        more paint and churns it into the wet paint underneath; light
        pressure lays paint on top. An overloaded brush pressed hard
        squeezes paint out into ridges along the edges. A knife rides on
        the paint: light pressure leaves a thick slab, heavy pressure
        scrapes close to the canvas, leaving paint only in the weave.
    brush:
        The tool: a :class:`BrushType` (e.g. ``brushes.FLAT_HOG``), or a
        :class:`Brush` to keep paint and dirt between strokes (``paint`` and
        ``amount``, when given, are added to it first). Default: a medium round.
    angle:
        Override of the brush angle in degrees (see :class:`BrushType`); a
        number, per control point, or a function of u. Turning a flat during
        the stroke twists it from a wide band into a thin ribbon.
    follow_path:
        Override of :attr:`BrushType.follow_path`.
    roll:
        Rolling a round brush between the fingers, in degrees of rotation
        about its handle: a number, per control point, or a function of u
        (``lambda u: 360 * u`` rolls it one full turn over the stroke). It
        turns other bristles, and the paint on them, to the canvas. (Flat
        brushes are turned with ``angle`` instead.)
    mixing:
        How readily the stroke churns into wet paint below (0 = sits on top,
        1 = stirs in strongly). Scaled by pressure.
    pickup:
        How much wet paint the brush lifts and carries along (0..1).
    drag:
        How much wet paint the brush pushes along with it (0..1).
    streaks:
        Strength of the brush-hair streak texture (0..1).
    smear:
        How long paint picked up from the canvas persists on the brush tip,
        roughly in brush widths.
    blending:
        How readily the paints on the brush merge into each other during the
        stroke (see :attr:`BrushType.blending`).
    taper:
        Fraction of the stroke length over which the brush is lowered at the
        start and lifted at the end (narrower, feathered ends). A number, or
        a (start, end) pair. 0 gives full ends.
    lift:
        0..1: how the tool leaves the paint at the end. 0 slides off cleanly;
        higher pulls the paint up into little peaks and torn ridges, as a
        knife lifted straight off thick paint does (default 0.6 for knives,
        0 for brushes). Stiffer paint holds sharper peaks.

        ``mixing``, ``pickup``, ``drag``, ``streaks``, ``smear``,
        ``blending``, ``taper`` and ``lift`` default to values derived from
        the brush type.
    seed:
        Random seed for the stroke's texture.
    """

    path: object
    paint: Optional[ColorLike] = None
    amount: float = 6.0
    width: Optional[Profile] = None
    pressure: Profile = 0.5
    brush: Union[Brush, BrushType, None] = None
    angle: Optional[Profile] = None
    follow_path: Optional[bool] = None
    roll: Optional[Profile] = None
    mixing: Optional[float] = None
    pickup: Optional[float] = None
    drag: Optional[float] = None
    streaks: Optional[float] = None
    smear: Optional[float] = None
    blending: Optional[float] = None
    taper: object = None
    lift: Optional[float] = None
    seed: Optional[int] = None


def _resolve(stroke: Stroke):
    """The Brush to paint with, its type, and the effective stroke settings."""
    tool = stroke.brush
    if isinstance(tool, Brush):
        brush = tool
    else:
        brush = Brush(tool if isinstance(tool, BrushType) else None)
    bt = brush.type
    phys = bt.physics()
    eff = {k: (getattr(stroke, k) if getattr(stroke, k) is not None else phys[k])
           for k in ("mixing", "pickup", "drag", "streaks", "smear", "blending")}
    eff["rough"] = phys["rough"]
    eff["ridge"] = phys["ridge"]
    eff["lift"] = float(np.clip(stroke.lift if stroke.lift is not None else phys["lift"], 0.0, 1.0))
    eff["width"] = stroke.width if stroke.width is not None else bt.width
    eff["taper"] = stroke.taper if stroke.taper is not None else bt.taper
    eff["angle"] = stroke.angle if stroke.angle is not None else bt.angle
    eff["follow_path"] = stroke.follow_path if stroke.follow_path is not None else bt.follow_path
    return brush, bt, eff


_CHORD_CACHE: dict = {}
_SUPERELLIPSE = {"flat": 10.0, "filbert": 3.0, "fan": 6.0, "round": 2.0, "spoon": 2.0}

# Knife blades as polygons: (u along the blade from heel -1 to tip +1, v
# across it in units of the blade's half-width, i.e. scaled by aspect).
_BLADES = {
    # a diamond trowel: widest a little behind the middle, pointed tip
    "trowel": [(-1.0, -0.3), (-0.25, -1.0), (1.0, 0.0), (-0.25, 1.0), (-1.0, 0.3)],
    # a pear / teardrop: tapering from a narrow heel to a round tip
    "pear": [(-1.0, -0.35)] + [(0.3 + 0.7 * np.cos(t), -np.sin(t)) for t in np.linspace(np.pi / 2, -np.pi / 2, 15)][::-1]
            + [(-1.0, 0.35)],
    # a long straight blade with a rounded end
    "straight": [(-1.0, -1.0)] + [(0.82 + 0.18 * np.cos(t), np.sin(t)) for t in np.linspace(-np.pi / 2, np.pi / 2, 11)]
                + [(-1.0, 1.0)],
    # a straight blade whose end is cut on a slant (a sharp toe)
    "angled": [(-1.0, -1.0), (1.0, -1.0), (0.55, 1.0), (-1.0, 1.0)],
}


def _outline(shape: str, aspect: float, n: int = 400):
    """Points (u, v) around the footprint outline, u in -1..1 along the brush's
    width (the blade's length), v across it."""
    if shape in _SUPERELLIPSE:
        e = _SUPERELLIPSE[shape]
        th = np.linspace(0, 2 * np.pi, n, endpoint=False)
        cu, sv = np.cos(th), np.sin(th)
        return np.sign(cu) * np.abs(cu) ** (2 / e), aspect * np.sign(sv) * np.abs(sv) ** (2 / e)
    poly = np.asarray(_BLADES[shape], float)
    poly = np.vstack([poly, poly[:1]])
    seg = np.linalg.norm(np.diff(poly * [1.0, aspect], axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    t = np.linspace(0, s[-1], n, endpoint=False)
    return np.interp(t, s, poly[:, 0]), aspect * np.interp(t, s, poly[:, 1])


def _footprint_area(shape: str, aspect: float) -> float:
    """Area of the footprint, in units of the half-width squared."""
    u, v = _outline(shape, aspect)
    return float(0.5 * abs(np.dot(u, np.roll(v, -1)) - np.dot(v, np.roll(u, -1))))


def _chord_table(shape: str, aspect: float, phi: float):
    """Geometry of a footprint swept along +t, as lookup tables over the
    lateral coordinate (all in units of the brush half-width).

    The footprint is a superellipse with half-axes 1 (along the brush's width,
    rotated by ``phi`` from the lateral direction) and ``aspect``. For each
    lateral offset l, the line through it parallel to the motion crosses the
    footprint over [c - h, c + h]. Returns (l grid, c, h, lateral half-extent).
    """
    key = (shape, round(aspect, 3), round(phi, 2))
    tab = _CHORD_CACHE.get(key)
    if tab is not None:
        return tab
    phi = key[2]
    u, v = _outline(shape, aspect)
    lat = u * np.cos(phi) - v * np.sin(phi)
    al = u * np.sin(phi) + v * np.cos(phi)
    lmax = float(np.abs(lat).max())
    grid = np.linspace(-lmax, lmax, 257)
    l0, l1 = lat[None, :], np.roll(lat, -1)[None, :]
    a0, a1 = al[None, :], np.roll(al, -1)[None, :]
    den = np.where(np.abs(l1 - l0) < 1e-12, 1e-12, l1 - l0)
    t = (grid[:, None] - l0) / den
    ok = (t >= 0) & (t <= 1)
    a = a0 + t * (a1 - a0)
    lo = np.where(ok, a, np.inf).min(axis=1)
    hi = np.where(ok, a, -np.inf).max(axis=1)
    hit = np.isfinite(lo) & np.isfinite(hi)
    lo, hi = np.where(hit, lo, 0.0), np.where(hit, hi, 0.0)
    c = (lo + hi) / 2
    h = (hi - lo) / 2
    tab = (grid, c, h, lmax)
    if len(_CHORD_CACHE) > 4000:
        _CHORD_CACHE.clear()
    _CHORD_CACHE[key] = tab
    return tab


def _signed_floor(x, floor):
    return np.copysign(max(abs(x), floor), x if x != 0 else 1.0)


def apply_stroke(canvas, stroke: Stroke) -> Brush:
    """Run the stroke simulation on ``canvas``. Returns the brush used.

    The brush moves along the path in short straight segments. For every
    pixel we compute its *exposure* during a segment: the fraction of a full
    brush pass over that pixel that happens during this segment (exact for a
    convex footprint sweeping a straight segment). Exposures over a whole pass
    add up to the brush's lateral profile, so the result is smooth and does
    not depend on the step size.
    """
    path: Path = as_path(stroke.path)
    rng = np.random.default_rng(stroke.seed if stroke.seed is not None else canvas._rng.integers(1 << 31))
    brush, bt, eff = _resolve(stroke)

    n_probe = max(path.n_knots, 2) * 8
    w_probe = evaluate_profile(eff["width"], np.linspace(0, 1, n_probe),
                               np.linspace(0, max(path.n_knots - 1, 0), n_probe), path.n_knots)
    w_typ = max(float(np.mean(w_probe)), 1.0)
    spacing = float(np.clip(0.25 * w_typ, 0.5, 8.0))
    taper = eff["taper"]
    t0, t1 = (taper, taper) if np.isscalar(taper) else taper
    circle = bt.shape == "round" and abs(bt.aspect - 1.0) < 1e-6

    def sample(spacing):
        pts, _, u, knot = path.sample(spacing)
        widths = evaluate_profile(eff["width"], u, knot, path.n_knots)
        if t0 > 0:
            widths = widths * (0.25 + 0.75 * _smoothstep(0.0, t0, u))
        if t1 > 0:
            widths = widths * (0.25 + 0.75 * _smoothstep(1.0, 1.0 - t1, u))
        widths = np.maximum(widths, 1.0)
        pressure = np.clip(evaluate_profile(stroke.pressure, u, knot, path.n_knots), 0.0, 1.0)
        angles = np.radians(evaluate_profile(eff["angle"], u, knot, path.n_knots))
        rolls = np.radians(evaluate_profile(stroke.roll if stroke.roll is not None else 0.0, u, knot,
                                            path.n_knots))
        return pts, widths, pressure, angles, rolls

    pts, widths, pressure, angles, rolls = sample(spacing)
    if not circle and len(pts) > 2:
        # An elongated footprint that turns between two steps leaves a thin
        # wedge uncovered at the outside of the turn. Step finely enough that
        # no turn moves the brush's edge by more than about half a pixel.
        seg = np.diff(pts, axis=0)
        heading = np.arctan2(seg[:, 1], seg[:, 0])
        turn = np.abs(np.angle(np.exp(1j * np.diff(heading))))
        twist = np.abs(np.diff(angles))[1:] if len(angles) > 2 else 0.0
        reach_px = 0.5 * float(widths.max()) * (1.0 + bt.spread)
        gap = float(np.max(turn + twist)) * reach_px
        if gap > 0.5:
            pts, widths, pressure, angles, rolls = sample(max(spacing * 0.5 / gap, 0.35))
    is_dab = len(pts) == 1
    if is_dab:  # a dab: press the brush down with a tiny movement (small next to its narrow side)
        nudge = max(0.05 * widths[0] * min(1.0, bt.aspect), 0.25)
        pts = np.vstack([pts[0] - [nudge, 0], pts[0] + [nudge, 0]])
        widths = np.repeat(widths, 2)
        pressure = np.repeat(pressure, 2)
        angles = np.repeat(angles, 2)
        rolls = np.repeat(rolls, 2)

    brush.width = float(np.mean(widths))
    if isinstance(stroke.paint, list):          # several paints: [(paint, amount[, across[, depth]]), ...]
        for item in stroke.paint:
            if isinstance(item, dict):
                brush.load(**item)
            else:
                item = tuple(item)
                brush.load(item[0], item[1], across=item[2] if len(item) > 2 else None,
                           depth=item[3] if len(item) > 3 else "belly")
    elif stroke.paint is not None:
        brush.load(stroke.paint, stroke.amount)

    # Hair streaks: a fixed 1-D pattern across the brush (in units of its
    # half-width), so streaks run along the stroke; coarser bristles give
    # fewer, bolder streaks. Plus a slow wobble of the width.
    n_tex = 512
    density = 2.0 ** (2.0 * (bt.bristles - 0.5))
    tex = (0.6 * value_noise_1d(n_tex, n_tex / (14.0 * density), rng)
           + 0.4 * value_noise_1d(n_tex, n_tex / (45.0 * density), rng))
    tex = (tex - tex.min()) / max(np.ptp(tex), 1e-9)
    tex_x = np.linspace(-1.3, 1.3, n_tex)
    wobble = value_noise_1d(len(pts) + 2, max(2.0, 0.8 * w_typ / spacing), rng)[:len(pts)] - 0.5
    # a knife's height wavers as it is pushed along (faint ripples across the
    # slab: chatter), and a hand never presses both ends of the blade quite
    # equally (the slab is thicker towards one side)
    arc_all = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))])
    period = max(0.4 * w_typ, 2.0)          # ripples about half a blade width apart
    ripple = value_noise_1d(int(arc_all[-1] / period) + 4, 1.0, rng)
    chatter = np.interp(arc_all / period, np.arange(len(ripple)), ripple) - 0.5
    swell = value_noise_1d(int(arc_all[-1] / (2.0 * period)) + 4, 1.0, rng)
    lip = np.interp(arc_all / (2.0 * period), np.arange(len(swell)), swell)
    blade_tilt = float(np.clip(rng.normal(0.0, 0.08), -0.2, 0.2))
    if is_dab:   # bristle tips pressed straight down: a blotchy 2-D pattern (stippling)
        tex2 = fractal_noise_2d(96, 96, 96 / (9.0 * density), rng, octaves=2)
        tex2 = (tex2 - tex2.min()) / max(np.ptp(tex2), 1e-9)
    aspect = 1.0 if circle else bt.aspect
    reach = 1.0 if circle else float(np.hypot(1.0, aspect))
    streaks, rough, ridge = eff["streaks"], eff["rough"], eff["ridge"]

    # Moving paint around (impasto). The tool's underside sits at some level
    # above the canvas: a knife blade rides at a height set by pressure,
    # bristles comb through at about the thickness they lay. Wet paint above
    # that level is shifted: partly onto the tool, partly out to the sides
    # (ridges), the rest pushed ahead of it (and left as a ridge where the
    # tool lifts).
    rigid = bt.rigid
    lift = eff["lift"]
    area_unit = _footprint_area("round" if circle else bt.shape, aspect)    # in half-widths squared
    plough_k = 4.0 if rigid else 0.8 * bt.stiffness ** 2
    # A blade carries what it scrapes as a bow wave in front of its edge (only a
    # little sticks to it): dropped into the low spots it passes, piled where it lifts.
    share_pick = 0.08 if rigid else 0.5 * min(eff["pickup"], 1.0)
    share_side = 0.3 if rigid else 0.35
    stiff_blade = float(np.clip((bt.stiffness - 0.4) / 0.6, 0.0, 1.0))
    col_u = (_RING_X if brush.ring else _COL_X) * 2.0 - 1.0    # where each column sits across the mark
    # squeezed-out ridges are at most this wide (px); they lie just outside the mark
    margin = float(np.clip(0.12 * w_typ * (1.0 + bt.spread), 0.6, 7.0)) + 1.0

    # brush half-width at every sample: size, a slow wobble, and how far the
    # hairs splay under pressure (a pointed round paints thin with a light touch)
    radii = 0.5 * widths * (1.0 + 0.06 * wobble)
    nominal = radii.copy()               # the bristle bundle's size; pressure only splays it
    if bt.spread > 0:
        radii = radii * ((1.0 - bt.spread) + bt.spread * (np.maximum(pressure, 0.02) / 0.5) ** 0.8)
    arc = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))])
    feather = bt.softness * 3.0          # soft brushes fade in and out over this many widths

    surf = canvas.surface
    surface_acts = surf.affects_paint
    # how much the surface texture (vs. the hair pattern) decides where a dry brush catches
    tooth_w = float(np.clip(0.3 + (surf.tooth - 0.5), 0.0, 0.85))

    stroke_id = canvas._new_stroke_id()
    H, W = canvas.height, canvas.width
    dry = False
    brush.wet = 0.0
    for i in range(1, len(pts)):
        (ax, ay), (cx, cy) = pts[i - 1], pts[i]
        ds = float(np.hypot(cx - ax, cy - ay))
        if ds < 1e-9:
            continue
        tx, ty = (cx - ax) / ds, (cy - ay) / ds
        p = 0.5 * (pressure[i] + pressure[i - 1])
        r0, r1 = float(radii[i - 1]), float(radii[i])
        r = 0.5 * (r0 + r1)
        rmax = max(r0, r1)
        Wd = 2.0 * r
        Wn = float(nominal[i - 1] + nominal[i])      # how loaded the brush is is relative to its real size
        # orientation of the brush's long side relative to the stroke's cross direction
        ang = 0.5 * (angles[i] + angles[i - 1])
        phi = ang if eff["follow_path"] else ang - np.arctan2(tx, -ty)
        side = 1.0 if np.cos(phi) >= 0 else -1.0      # a brush turned past 90 degrees has its sides swapped
        phi = (phi + np.pi / 2) % np.pi - np.pi / 2

        # --- footprint geometry on a window around the swept segment -------
        R = rmax * reach + (3.0 if rigid else 0.0)
        x0, x1 = int(np.floor(min(ax, cx) - R - 1)), int(np.ceil(max(ax, cx) + R + 1)) + 1
        y0, y1 = int(np.floor(min(ay, cy) - R - 1)), int(np.ceil(max(ay, cy) + R + 1)) + 1
        cx0, cx1, cy0, cy1 = max(x0, 0), min(x1, W), max(y0, 0), min(y1, H)
        if cx0 >= cx1 or cy0 >= cy1:
            _use_paint_off_canvas(brush, Wd, ds, p)
            continue
        gx = (np.arange(x0, x1) + 0.5 - cx)[None, :]
        gy = (np.arange(y0, y1) + 0.5 - cy)[:, None]
        along = gx * tx + gy * ty                  # <= 0 behind the brush centre
        lat_px = -gx * ty + gy * tx
        # the brush size changes smoothly through the segment (no steps when a flat tapers)
        r_px = r1 + (r0 - r1) * np.clip(-along / ds, 0.0, 1.0) if r0 != r1 else r
        lat = lat_px / r_px                        # signed, in brush half-widths
        if circle:
            lmax = 1.0
            ucoord = lat * side
        else:
            grid, c_tab, h_tab, lmax = _chord_table(bt.shape, aspect, phi)
            ucoord = side * lat / _signed_floor(np.cos(phi), aspect)       # position across the bristles
        hair = np.interp(ucoord, tex_x, tex)       # 0..1
        if is_dab:
            ix = np.clip(((ucoord + 1.3) / 2.6 * 95).astype(int), 0, 95)
            iy = np.clip(((along / r + 1.3) / 2.6 * 95).astype(int), 0, 95)
            hair = tex2[iy, ix]
        lr = np.abs(lat) / lmax * (1.0 + rough * (hair - 0.5))
        # Anti-aliasing: how much of each pixel the footprint covers across its
        # edge (a pixel-wide ramp centred on the true edge). This is geometry:
        # it sets the pixel's coverage, not the paint's thickness.
        edge_px = (1.0 - lr) * (r_px * lmax)
        aa = _smoothstep(-0.5 * _EDGE_PX, 0.5 * _EDGE_PX, edge_px)
        # Flat-topped profile; pressing harder squeezes paint into low ridges at the edges.
        falloff = aa
        if bt.softness > 0:    # a long, feathery falloff that stays visible even under thick paint
            falloff = falloff * _smoothstep(1.0, 1.0 - bt.softness, lr) ** (1.0 + 4.0 * bt.softness)
        profile = falloff * (0.92 + ridge * p * _smoothstep(0.55, 0.95, lr))
        if bt.slant > 0:       # bias-cut bristles: the toe (u = -1) touches first
            press = 0.25 + 0.9 * p
            uc = np.clip(ucoord, -1.0, 1.0)
            profile = profile * np.clip((press - bt.slant * (1.0 + uc) / 2.0) / 0.3, 0.0, 1.0)
        if bt.clumps > 0:      # separate clumps of bristles (fan brushes), or the teeth of a comb
            m = 0.5 + 0.5 * np.cos(np.pi * bt.clumps * (np.clip(ucoord, -1, 1) + 1.0) + 2.0 * (hair - 0.5))
            profile = profile * _smoothstep(0.3, 0.7, m)
            between = falloff * _smoothstep(0.6, 0.2, m)       # the gaps between them
        # Chord of the footprint through each pixel along the motion, and the
        # part of it swept during this segment.
        if circle:
            c = 0.0
            h = np.maximum(r_px * np.sqrt(np.clip(1.0 - lr * lr, 0.0, None)), 0.5)
        else:
            c = np.interp(lat, grid, c_tab) * r_px
            h = np.maximum(np.interp(lat, grid, h_tab) * r_px, 0.5)
        rel = along - c
        overlap = np.clip(np.minimum(0.0, rel + h) - np.maximum(-ds, rel - h), 0.0, None)
        # Pixels under the brush where it touches down / lifts off get topped
        # up to a full pass, so strokes have solid ends.
        # Soft brushes are lowered and lifted gently, so their ends feather.
        end_k = 1.0 if is_dab else max(0.0, 1.0 - 1.5 * bt.softness)
        half_aa = 0.5 * _EDGE_PX
        if i == 1 and end_k > 0:
            a0 = rel + ds
            overlap += end_k * _smoothstep(-h - half_aa, -h + half_aa, a0) * np.clip(h - a0, 0.0, 2.0 * h)
        if i == len(pts) - 1 and end_k > 0:
            overlap += end_k * _smoothstep(h + half_aa, h - half_aa, rel) * np.clip(h + rel, 0.0, 2.0 * h)
        swept = overlap / (2.0 * h)                 # fraction of a full pass
        expo = profile * swept
        cover = aa * swept                          # geometric coverage added during this segment
        if feather > 0 and not is_dab:
            s_pix = arc[i] + along                  # distance along the stroke
            span = max(feather * 2.0 * r, 1.0)
            expo = expo * _smoothstep(-r, -r + span, s_pix) * _smoothstep(-r, -r + span, arc[-1] - s_pix)
        streak = np.clip(1.0 + streaks * 1.6 * (hair - 0.5), 0.0, None)

        sl = (slice(cy0, cy1), slice(cx0, cx1))
        wsl = (slice(cy0 - y0, cy1 - y0), slice(cx0 - x0, cx1 - x0))
        expo_c = expo[wsl]
        if not expo_c.any():
            continue
        streak_c = streak[wsl]
        contact = expo_c * streak_c
        top, und = canvas.top[sl], canvas.und[sl]
        if surface_acts:
            # Where the brush works directly on the priming (little paint
            # below), the surface itself decides how paint grips and slides.
            wet = top[..., IV] + und[..., IV]
            bare = np.exp(-(wet + canvas.dry_height[sl]) / 0.4)
            # Lifting wet paint depends on how it adheres to what is beneath: it
            # wipes cleanly off a slick surface but stays stained into an
            # absorbent one. Dry paint below behaves like a neutral surface.
            on_ground = np.exp(-wet / 1.5) * np.exp(-canvas.dry_height[sl] / 0.4)
            f_pick = np.clip(1.0 + on_ground * (2.0 * (surf.slickness - 0.5) - 0.8 * surf.absorbency), 0.1, None)
            f_dep = np.clip(1.0 + bare * (0.6 * surf.absorbency + 0.4 * (surf.tooth - 0.5)
                                          - 0.4 * (surf.slickness - 0.5)), 0.4, None)
            f_mix = 1.0 - 0.6 * surf.absorbency * bare
            f_drag = 1.0 + (surf.slickness - 0.5) * float(bare.mean())
        else:
            f_pick = f_dep = f_mix = f_drag = 1.0

        # 0) drag: wet paint under the brush is pushed along with it (a knife
        #    pushes paint with its bow wave and smears it instead, below)
        if eff["drag"] > 0 and not rigid:
            f = eff["drag"] * f_drag * (0.5 + p) * profile * np.clip(overlap / ds, 0.0, 1.0)
            _drag(canvas, f, x0, y0, tx * ds, ty * ds, p)

        # Which bristle columns are above each pixel: (column, weight) taps,
        # blending linearly between the nearest columns. On a thin brush one
        # pixel spans several columns: sample across it so that every column
        # reaches the canvas (else paint in the columns between pixel centres
        # would only come out by diffusion, and thin strokes would run long).
        u_px = ucoord[wsl]
        span = 1.0 / max(r * (1.0 if circle else abs(_signed_floor(np.cos(phi), aspect))), 1e-6)
        span_cols = span * COLUMNS * (0.25 if brush.ring else 0.5)
        nsub = int(min(max(np.ceil(span_cols / 2.0 - 1e-9), 1), 8))
        rho = 0.5 * float(rolls[i] + rolls[i - 1])
        taps = []
        for k_sub in range(nsub):
            uc = np.clip(u_px + span * ((k_sub + 0.5) / nsub - 0.5), -1.0, 1.0) if nsub > 1 else \
                np.clip(u_px, -1.0, 1.0)
            if brush.ring:
                # Around a round bundle, a point across the mark is touched by the
                # bristles at the front and the back of the bundle, at that side;
                # rolling the brush turns the whole ring.
                for theta in (-uc * np.pi / 2.0, np.pi + uc * np.pi / 2.0):     # front and back of the bundle
                    pos = np.mod((theta - rho) / (2.0 * np.pi) * COLUMNS - 0.5, COLUMNS)
                    c0 = np.minimum(pos.astype(int), COLUMNS - 1)
                    fr = pos - c0
                    taps += [(c0, 0.5 * (1.0 - fr) / nsub), ((c0 + 1) % COLUMNS, 0.5 * fr / nsub)]
            else:
                # column b covers an equal slice of the brush, centred at _COL_X[b]
                colf = (uc + 1.0) * 0.5 * COLUMNS - 0.5
                c0 = np.clip(np.floor(colf), 0, COLUMNS - 2).astype(int)
                fr = np.clip(colf - c0, 0.0, 1.0)
                taps += [(c0, (1.0 - fr) / nsub), (c0 + 1, fr / nsub)]
        npx = u_px.size
        assign = np.zeros((COLUMNS, npx))                       # pixel -> column weights
        flat_i = np.arange(npx)
        for idx, wt in taps:
            assign[idx.ravel(), flat_i] += wt.ravel()

        def at_pixels(per_column):
            return sum(wt * per_column[idx] for idx, wt in taps)

        # 1) refill each column's tip from its belly; slow exchange between the
        #    two; the paints on the brush blend sideways as the bristles work
        capacity = max(eff["smear"], 0.05) * 0.8 * Wd * Wd
        if rigid:
            # The paint at the blade's edge is what goes down next: enough for a
            # few steps of slab. Paint it scrapes up joins it there and is laid
            # again within a short distance (drag streaks), the rest of the
            # load waits further back on the blade.
            gap_now = _KNIFE_GAP * (1.0 - 0.5 * (pressure[i] + pressure[i - 1])) ** 1.6 + 1.0
            capacity = capacity + 3.0 * gap_now * ds * Wd
        _refill(brush, capacity)
        if i == 1:      # pressing the brush down squeezes some paint from the heel towards the tip
            _seep(brush, min(_HEEL_SQUEEZE * (0.3 + 2.8 * p * p), 1.0))
        elif pressure[i] > pressure[i - 1]:      # ... and so does bearing down harder
            _seep(brush, min(_HEEL_SQUEEZE * 5.0 * (pressure[i] - pressure[i - 1]), 1.0))
        blend = eff["blending"]
        _swap_layers(brush.tip, brush.reservoir, 0.375 * blend * ds / Wd)
        _swap_layers(brush.reservoir, brush.heel, 0.1 * blend * ds / Wd)
        k = _BLEND_RATE * blend * (0.2 + p) ** 2 * ds / Wd
        for layer in (brush.tip, brush.reservoir, brush.heel):
            _diffuse(layer, k, ring=brush.ring)

        # 2) pick up wet paint. Bristles plough through it: they lift paint
        #    along the hair lines and leave it between them (brush marks).
        #    Thick fresh paint shields the older layer below. What is lifted
        #    goes into the bristles that touched it.
        plough = expo_c * np.clip(2.0 - streak_c, 0.0, None)
        a_top = 1.0 - np.exp(-eff["pickup"] * 1.2 * plough * f_pick)
        a_und = (1.0 - np.exp(-eff["pickup"] * 1.2 * p * plough * f_pick)) * np.exp(-top[..., IV] / 1.5)
        for layer, a in ((top, a_top), (und, a_und)):
            lifted = a.astype(np.float32)[..., None] * layer
            brush.tip += assign @ lifted.reshape(npx, PACKED).astype(np.float64)
            layer -= lifted

        # 3) how much paint the tool carries, per column (load units: width
        #    squares). Paint beyond what the hair holds is a glob on the
        #    outside: it goes on thick, the more so the stiffer the paint.
        #    (Paint deep in the heel only counts once squeezed down, above.)
        col_vol = brush.reservoir[:, IV] + brush.tip[:, IV]
        load = float(col_vol.sum()) / (Wn * Wn)
        col_load = col_vol * COLUMNS / (Wn * Wn)
        col_b = brush.reservoir[:, IB] + brush.tip[:, IB]
        col_body = np.where(col_vol > 0, col_b / np.maximum(col_vol, 1e-30), 0.0)
        body_all = float(col_b.sum() / max(col_vol.sum(), 1e-30))
        if rigid:
            gap = _KNIFE_GAP * (1.0 - p) ** 1.6 * (0.55 + 0.45 * bt.stiffness) + 0.05
            if bt.clumps > 0:       # the teeth of a comb dig in, down to the threads
                gap *= 0.03
            hold = gap * area_unit / 4.0 + _KNIFE_CARRY
            thick = gap
        else:
            # small brushes have long hair for their width: they hold far more for their
            # size (a rigger holds enough for a line hundreds of widths long)
            hold = _HOLD * bt.capacity * (1.3 - 0.6 * p) * max(1.0, 20.0 / Wn) ** 1.5
            glob_gain = _IMPASTO * (1.2 - 0.5 * p)
            thick = _deposit_thickness(load) * (0.5 + p) + glob_gain * body_all * max(load - hold, 0.0)
            col_glob = glob_gain * col_body * np.maximum(col_load - hold, 0.0)
            col_thick = _deposit_thickness(col_load) * (0.5 + p) + col_glob
            col_fill = np.clip((col_load - _RESIDUAL) / 0.6, 0.0, 1.0)

        # 4) displace wet paint that stands above the tool's underside
        floor = canvas._floor[sl]
        uv = und[..., IV]
        touchdown = i == 1 or is_dab
        ring_pk = np.zeros(PACKED)          # paint squeezed out all around the footprint
        side_pk = [np.zeros(PACKED), np.zeros(PACKED)]     # ... or out at the left / right edge
        excess = None
        if rigid:   # the blade rests on the high points of the surface below, plus the paint film,
            # and shoves everything above it, its own trail included
            rest = canvas._peaks_fine[sl] + stiff_blade * (canvas._peaks_wide[sl] - canvas._peaks_fine[sl])
            # Wet paint from earlier strokes holds the blade up too: it rests on
            # its high points and sinks in the harder it is pressed (less into
            # stiff paint). A light touch lays new paint on top of the old.
            old_v = uv + np.where(canvas.top_id[sl] != stroke_id, top[..., IV], 0.0)
            if old_v.max() > 0.05:
                old_body = float((und[..., IB].sum() + np.where(canvas.top_id[sl] != stroke_id, top[..., IB], 0.0).sum())
                                 / max(float(old_v.sum()), 1e-9))
                # How far into it the blade sinks, as a share of its thickness: a light
                # touch skims, a firm one cuts in, pressing hard goes nearly down to
                # the canvas, however thick the paint. Stiff paint resists a little.
                depth = float(np.clip(p ** _KNIFE_SINK * (1.15 - 0.3 * min(old_body, 1.0)), 0.0, 1.0))
                if bt.clumps > 0:       # narrow teeth concentrate the pressure: they cut right through
                    depth = max(depth, 0.9)
                # A rigid blade can't follow bumps much smaller than itself: it rests
                # on the high points under it (not on the pile it pushes ahead).
                under_blade = (aa * np.clip(h - np.abs(rel) + 0.5, 0.0, 1.0))[wsl] > 0.5
                if under_blade.any():
                    # the blade bends to follow the surface over a span set by its
                    # stiffness, resting on the high points under it
                    span = int(np.clip(0.1 * w_typ * (0.5 + stiff_blade), 2, 10))
                    wgt = under_blade.astype(np.float32)
                    hi = _max_filter(floor + old_v, 1)
                    norm = np.maximum(_box_blur(wgt, span), 1e-6)
                    high = np.where(under_blade, _box_blur(hi * wgt, span) / norm, hi)
                    base = np.where(under_blade, _box_blur(rest * wgt, span) / norm, rest)
                    rest = np.maximum(rest, high - depth * np.maximum(high - base, 0.0))
            # The blade rides on the paint under it: as that runs low, it sinks
            # and the slab thins out. Nicks and grit on its edge leave fine
            # scratches along the stroke.
            film = at_pixels(brush.tip[:, IV] + brush.reservoir[:, IV]) * COLUMNS / max(area_unit * r * r, 1e-9)
            ride = np.minimum(gap, _KNIFE_FILM * film) if bt.clumps == 0 else gap
            level = rest + ride * (1.0 + 0.25 * chatter[i] + 0.12 * (hair[wsl] - 0.5)
                                   + blade_tilt * np.clip(ucoord[wsl], -1.0, 1.0))
            tv = top[..., IV]
            excess = np.clip(floor + tv + uv - level, 0.0, None)
        else:       # bristles comb through paint already there (not what they just laid):
            # deeper along the hair lines, and more under pressure
            hb = max(1.25 * thick, 3.5 * (1.15 - p) * (1.3 - 0.8 * bt.stiffness))
            if float((top[..., IV] + uv).max()) > (1.0 - 0.4 * streaks) * hb:
                level = floor + hb * (1.0 + 0.8 * streaks * (hair[wsl] - 0.5))
                tv = np.where(canvas.top_id[sl] != stroke_id, top[..., IV], 0.0)
                excess = np.clip(floor + tv + uv - level, 0.0, None)
        if excess is not None:     # shifted gradually as the tool passes over
            k_scrape = plough_k * expo_c
            if rigid:   # nicks and flex let streaks of what the blade pushes slip under it and stay
                k_scrape = k_scrape * (0.15 + 1.7 * hair[wsl] ** 2)
            excess *= 1.0 - np.exp(-k_scrape)
        flowing = rigid and bt.clumps == 0 and not touchdown
        if flowing and excess is not None:
            # under a knife, what stands above the blade is bulldozed ahead by the
            # flow (below); only what spills out at the sides or sticks to the
            # blade is taken out here
            excess *= share_pick + share_side
        if excess is not None and excess.max() > 1e-3:
            from_top = np.minimum(excess, tv)
            from_und = np.clip(excess - from_top, 0.0, uv)
            ft = (from_top / np.where(tv > 0, tv, 1.0)).astype(np.float32)[..., None]
            fu = (from_und / np.where(uv > 0, uv, 1.0)).astype(np.float32)[..., None]
            moved = top * ft + und * fu
            top -= top * ft
            und -= und * fu
            moved = moved.astype(np.float64)
            pick = share_pick / (share_pick + share_side) if flowing else share_pick
            brush.tip += assign @ (pick * moved).reshape(npx, PACKED)
            moved *= 1.0 - pick
            if flowing:
                right = (lat_px[wsl] > 0)[..., None]
                side_pk[1] += (moved * right).sum(axis=(0, 1))
                side_pk[0] += (moved * ~right).sum(axis=(0, 1))
            elif rigid and bt.clumps > 0:
                # a comb's teeth push the paint they cut aside, into ridges between them
                fresh = between * _smoothstep(-ds - 1.0, 0.0, along - c) * _smoothstep(1.0, -1.0, along - c - h)
                _spread(canvas, sl, fresh[wsl], moved.sum(axis=(0, 1)), stroke_id, brush,
                        np.zeros(fresh[wsl].shape, np.float32))
            elif touchdown:
                ring_pk += moved.sum(axis=(0, 1))
            else:
                to_side = share_side / (1.0 - share_pick)
                right = (lat_px[wsl] > 0)[..., None]
                side_pk[1] += to_side * (moved * right).sum(axis=(0, 1))
                side_pk[0] += to_side * (moved * ~right).sum(axis=(0, 1))
                _place(canvas, (1.0 - to_side) * moved, cx0, cy0, tx * ds, ty * ds, stroke_id)
        if flowing:
            # The blade drags the wet paint it touches, as a thin-film flow: between
            # the moving blade and the still canvas the paint shears, so paint near
            # the blade travels almost with it and paint at the canvas hardly moves;
            # what stands above the blade is bulldozed ahead of it. Each wet layer
            # moves at its own speed, so their order is kept (white dragged over
            # purple stays on top). The blade glides on its own fresh paint.
            _flow_under_blade(canvas, sl, stroke_id, floor, level, aa[wsl], along[wsl],
                              (c[wsl] if not circle else 0.0), h[wsl], ds, tx, ty,
                              np.clip((0.5 + 0.45 * p) * (0.6 + 0.8 * hair[wsl]), 0.0, 1.0))

        # an overloaded tool pressed down squeezes paint out at its edges
        over = np.maximum(col_load - hold, 0.0)
        if rigid:   # only the paint under the blade gets squeezed out, not the whole glob on it
            over = np.minimum(over, hold + 1.0)
        if bt.deposits and over.any():
            # unevenly: the lips along a stroke swell and break
            rate = _SQUEEZE[rigid] * (0.2 + p * p) * (0.5 + bt.stiffness) * ds / Wd * (0.2 + 1.6 * lip[i])
            if touchdown:
                rate += 0.03 * (0.3 + p)
            take = np.minimum(over * (Wn * Wn / COLUMNS) * rate, col_vol)
            frac = np.where(col_vol > 0, take / np.maximum(col_vol, 1e-30), 0.0)[:, None]
            out = (brush.tip + brush.reservoir) * frac
            brush.tip -= brush.tip * frac
            brush.reservoir -= brush.reservoir * frac
            if touchdown:
                ring_pk += out.sum(axis=0)
            else:
                w_right = _smoothstep(-0.3, 0.3, col_u * side)[:, None]
                side_pk[1] += (out * w_right).sum(axis=0)
                side_pk[0] += (out * (1.0 - w_right)).sum(axis=0)
        if ring_pk[IV] > 0 or side_pk[0][IV] > 0 or side_pk[1][IV] > 0:
            # the ridges lie just outside the footprint: work on a slightly larger window
            m_px = int(np.ceil(margin))
            ex0, ex1, ey0, ey1 = x0 - m_px, x1 + m_px, y0 - m_px, y1 + m_px
            esl = (slice(max(ey0, 0), min(ey1, H)), slice(max(ex0, 0), min(ex1, W)))
            ewsl = (slice(esl[0].start - ey0, esl[0].stop - ey0), slice(esl[1].start - ex0, esl[1].stop - ex0))
            egx = (np.arange(ex0, ex1) + 0.5 - cx)[None, :]
            egy = (np.arange(ey0, ey1) + 0.5 - cy)[:, None]
            e_along = egx * tx + egy * ty
            e_lat = -egx * ty + egy * tx
            e_r = r1 + (r0 - r1) * np.clip(-e_along / ds, 0.0, 1.0) if r0 != r1 else r

            ridge_w = float(np.clip(0.24 * r * lmax, 0.6, margin - 1.0))   # scales with the mark's width

            def tent(e):        # cross-section of a squeezed-out ridge, just outside the edge
                return np.clip(1.0 - np.abs(e - 0.35 * ridge_w) / (0.65 * ridge_w + 0.5), 0.0, 1.0)

            def geo(e):         # how much of each pixel the ridge covers (anti-aliased outer edge)
                return np.clip(ridge_w - e, 0.0, 1.0) * (e > -0.3 * ridge_w)
            e_side = np.abs(e_lat) - e_r * lmax
            if ring_pk[IV] > 0:
                # all around the footprint: as wide as its narrow side allows
                ridge_w = float(np.clip(0.24 * r * min(lmax, 1.0 if circle else float(h_tab.max())), 0.6,
                                        margin - 1.0))
                if circle:
                    e_ring = np.sqrt(e_lat * e_lat + e_along * e_along) - e_r
                else:
                    el = e_lat / e_r
                    e_c = np.interp(el, grid, c_tab) * e_r
                    e_h = np.interp(el, grid, h_tab, left=0.0, right=0.0) * e_r
                    e_ring = np.maximum(e_side, np.abs(e_along - e_c) - e_h)
                _spread(canvas, esl, tent(e_ring)[ewsl], ring_pk, stroke_id, brush, geo(e_ring)[ewsl])
            if side_pk[0][IV] > 0 or side_pk[1][IV] > 0:
                ridge_w = float(np.clip(0.24 * r * lmax, 0.6, margin - 1.0))
                ce = 0.0 if circle else np.where(e_lat > 0, c_tab[-1], c_tab[0]) * e_r
                rel_e = e_along - ce
                # soft-ended, so that the ridges laid at successive steps run together smoothly
                ramp = 0.5 * ds + 0.7
                box = _smoothstep(-ds - ramp, -ds + ramp, rel_e) * _smoothstep(ramp, -ramp, rel_e)
                band, g = tent(e_side) * box, geo(e_side) * box
                for k_side, mask in ((0, e_lat < 0), (1, e_lat > 0)):
                    _spread(canvas, esl, (band * mask)[ewsl], side_pk[k_side], stroke_id, brush,
                            (g * mask)[ewsl])

        # 5) deposit from the tips
        tip_v = brush.tip[:, IV]
        d = None
        if rigid:
            # A knife fills the space under the blade, up to its level, from
            # the film of paint under it: with plenty of paint, a smooth slab;
            # running low, it only reaches the high points (broken colour).
            if not dry and tip_v.sum() > 0 and bt.deposits:
                surf_h = floor + top[..., IV] + und[..., IV]
                gap_px = np.clip(level - surf_h, 0.0, None)
                film = at_pixels(tip_v + brush.reservoir[:, IV]) * COLUMNS / max(area_unit * r * r, 1e-9)
                reach_f = _smoothstep(1.3, 0.7, gap_px / np.maximum(film, 1e-9))
                d = gap_px * reach_f * (1.0 - np.exp(-4.0 * expo_c))
        else:
            # Each column's load sets how thick its paint goes on and when it
            # starts to run dry. Bristles that were just in wet paint are
            # coated and keep laying a continuous smear for a short distance,
            # even with little load.
            wet_contact = _smoothstep(0.05, 0.35, top[..., IV] + und[..., IV])
            wet_frac = float((contact * wet_contact).sum()) / max(float(contact.sum()), 1e-12)
            brush.wet = max(wet_frac, brush.wet * np.exp(-ds / Wd))
            fill = np.clip(np.maximum(at_pixels(col_fill), 0.9 * brush.wet), 0.0, 1.0)
            # Dry brush: only the canvas tooth (and hair lines) catch paint. Over
            # wet paint the bristles are in full contact anyway.
            thr = 1.5 * (1.0 - fill) ** 0.8 - 0.35
            # a toothy surface breaks up a lightly pressed stroke even when the brush
            # is loaded (paint only touches the tops of the threads: scumbling)
            rough_t = surf.tooth - 0.5
            thr = thr + (3.2 if rough_t > 0 else 0.6) * rough_t * (1.0 - p) ** 1.2
            tooth = tooth_w * canvas._tooth[sl] + (1.0 - tooth_w) * hair[wsl]
            catch = np.maximum(_smoothstep(thr - 0.08, thr + 0.08, tooth), wet_contact)
            q = contact * catch
            # When only the tooth catches, paint concentrates on those peaks
            # instead of staying on the brush (so the brush still empties)...
            caught = float(q.sum()) / max(float(contact.sum()), 1e-12)
            q /= max(caught, 0.2)
            # ...and once the brush as a whole is effectively dry, the rest stays
            # in the bristles until a real reload (e.g. dragging through wet paint).
            fill_all = max(float(np.clip((load - _RESIDUAL) / 0.6, 0.0, 1.0)), 0.9 * brush.wet)
            thr_all = 1.5 * (1.0 - fill_all) ** 0.8 - 0.35 + (thr - (1.5 * (1.0 - fill) ** 0.8 - 0.35))
            catch_all = np.maximum(_smoothstep(thr_all - 0.08, thr_all + 0.08, tooth), wet_contact)
            caught_all = float((contact * catch_all).sum()) / max(float(contact.sum()), 1e-12)
            if dry:   # a brush that has run dry only comes back to life by dragging through wet
                # paint from earlier strokes (not the trace it is leaving itself)
                other = und[..., IV] + np.where(canvas.top_id[sl] != stroke_id, top[..., IV], 0.0)
                reload = _smoothstep(0.05, 0.35, other)
                dry = float((contact * reload).sum()) / max(float(contact.sum()), 1e-12) < 0.12
            else:
                dry = caught_all < 0.04
            if not dry and tip_v.sum() > 0 and bt.deposits:
                d = at_pixels(col_thick) * q * f_dep      # absorbent / toothy grounds pull paint off the brush
                # thick paint from a glob is combed by the bristles into deep grooves and ridges
                glob_px = at_pixels(col_glob) / np.maximum(at_pixels(col_thick), 1e-9)
                d *= np.clip(1.0 + glob_px * 2.0 * streaks * (2.0 * hair[wsl] - 1.0), 0.15, None)
        def lay(d, cover_px, moved=False):
            """Lay volume ``d`` per pixel from the tips of the columns above it;
            each column can only give what it holds."""
            tip_v = brush.tip[:, IV]
            demand = assign @ d.ravel()
            give = np.where(demand > tip_v, tip_v / np.maximum(demand, 1e-30), 1.0)
            comp = brush.tip / np.where(tip_v > 0, tip_v, 1.0)[:, None]     # paint per unit volume
            parts = [(idx, d * wt * give[idx]) for idx, wt in taps]       # volume from each column
            d_px = sum(v for _, v in parts)
            if d_px.sum() > 0:
                dep = sum(v[..., None] * comp[idx] for idx, v in parts)
                _deposit(canvas, sl, dep, d_px, stroke_id, cover_px, moved=moved)
                used = sum(np.bincount(idx.ravel(), v.ravel(), COLUMNS) for idx, v in parts)
                brush.tip -= used[:, None] * comp
                np.maximum(brush.tip[:, IV], 0.0, out=brush.tip[:, IV])

        if d is not None:
            # A knife's paint goes on top; if that makes three wet layers, the two most
            # alike merge (a dirty knife's smear joins the paint it came from, a new
            # colour stays on top of the others).
            lay(d, cover[wsl], moved=rigid)
        if d is not None:
            if not rigid:
                off_canvas = thick * float((expo * streak).sum() - contact.sum())
                left = brush.tip[:, IV].sum()
                if off_canvas > 0 and left > 0:
                    brush.tip *= max(1.0 - off_canvas / left, 0.0)
        # A knife lifted off thick paint pulls it up into little peaks and torn ridges.
        if rigid and lift > 0 and (i == len(pts) - 1):
            under = (aa * np.clip((h - np.abs(rel)) / 1.5, 0.0, 1.0))[wsl]
            _lift_off(canvas, sl, under, lift * (0.5 + 0.8 * body_all), max(1.5, 0.05 * Wd),
                      np.degrees(np.arctan2(ty, tx)), rng)

        # 6) overflow back into the belly (this is how the brush gets dirty)
        cap_col = capacity / COLUMNS
        over = brush.tip[:, IV] - cap_col
        if (over > 0).any():
            frac = np.where(over > 0, over / np.maximum(brush.tip[:, IV], 1e-30), 0.0)[:, None]
            moved = brush.tip * frac
            brush.tip -= moved
            brush.reservoir += moved

        # 7) stir: pressure churns the fresh top layer and the wet paint below
        #    into each other
        m = 1.0 - np.exp(-eff["mixing"] * 3.0 * p * p * contact * f_mix)
        if rigid:
            # A blade only shears the paint it cuts through: riding within the top
            # layer it leaves the one below alone; cutting into it, it folds the two
            # into each other in streaks (marbling), not evenly.
            into_under = np.clip(canvas._floor[sl] + und[..., IV] + 0.5 - level, 0.0, 1.0)
            m = np.clip(m * (0.1 + 1.8 * hair[wsl]) * into_under, 0.0, 1.0)
        _stir(top, und, m.astype(np.float32), canvas.top_a[sl], canvas.und_a[sl])

    # The paint settles: fluid paint levels out, sheer edges of thick paint slump.
    pad = float(radii.max()) * reach + margin + 2.0
    bx0, by0 = int(max(np.floor(pts[:, 0].min() - pad), 0)), int(max(np.floor(pts[:, 1].min() - pad), 0))
    bx1, by1 = int(min(np.ceil(pts[:, 0].max() + pad) + 1, W)), int(min(np.ceil(pts[:, 1].max() + pad) + 1, H))
    if bx0 < bx1 and by0 < by1:
        _level(canvas, (slice(by0, by1), slice(bx0, bx1)))
    return brush


# ----------------------------------------------------------------------------
# helpers


def _smoothstep(e0, e1, x):
    t = np.clip((x - e0) / (e1 - e0), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def _use_paint_off_canvas(brush, Wd, ds, p):
    """Painting past the edge of the canvas still uses up paint."""
    total = brush.volume
    if total <= 0:
        return
    used = _deposit_thickness(total / (Wd * Wd)) * (0.5 + p) * ds * Wd * 0.9
    g = max(1.0 - used / total, 0.0)
    brush.tip *= g
    brush.reservoir *= g


def _refill(brush, capacity):
    """Top up each column's tip from its own belly."""
    need = capacity / COLUMNS - brush.tip[:, IV]
    have = brush.reservoir[:, IV]
    f = np.where((need > 0) & (have > 0), np.minimum(need / np.where(have > 0, have, 1.0), 1.0), 0.0)
    if not f.any():
        return
    moved = brush.reservoir * f[:, None]
    brush.tip += moved
    brush.reservoir -= moved


def _alike(a, b):
    """How different two packed paints are (0 = the same paint), per pixel."""
    fa = np.log1p(a[..., :IV] / np.maximum(a[..., IV:IV + 1], 1e-9))
    fb = np.log1p(b[..., :IV] / np.maximum(b[..., IV:IV + 1], 1e-9))
    return np.abs(fa - fb).mean(axis=-1)


def _deposit(canvas, sl, dep, d, stroke_id, cover, moved=False):
    # ``moved`` may also be a per-pixel mask
    """Lay paint ``dep`` (packed, per pixel; volume ``d``) on top of the canvas.

    Each pixel holds two wet layers. New paint goes on top and the old top
    joins the layer below. Paint that is only being shoved around (``moved``:
    pushed ahead of a knife, squeezed into a ridge) instead merges with
    whichever layer it is most like, so that, say, white pushed over white
    stays a white skin over the purple beneath it."""
    top, und = canvas.top[sl], canvas.und[sl]
    top_a, und_a = canvas.top_a[sl], canvas.und_a[sl]
    # Paint from an earlier stroke stops being "top" once new paint covers it.
    ids = canvas.top_id[sl]
    touched = d > 1e-7
    old = touched & (ids != stroke_id)
    if np.any(moved) and old.any():
        tv, uv = top[..., IV], und[..., IV]
        three = old & (tv > 1e-6) & (uv > 1e-6) & moved
        if three.any():
            # three layers in two places: merge the two most alike
            keep = np.zeros_like(old)
            keep[three] = _alike(top[three], dep[three]) < _alike(und[three], top[three])
            top[keep] += dep[keep].astype(np.float32)
            dep = np.where(keep[..., None], 0.0, dep)
            d = np.where(keep, 0.0, d)
            touched = d > 1e-7
            old = touched & (ids != stroke_id)
    if old.any():
        und[old] += top[old]
        und_a[old] = np.maximum(und_a[old], top_a[old])
        top[old] = 0
        top_a[old] = 0
        ids[old] = stroke_id
    top_a += np.where(touched, cover, 0.0).astype(np.float32)
    np.minimum(top_a, 1.0, out=top_a)
    top += dep.astype(np.float32)


def _spread(canvas, sl, w, pk, stroke_id, brush, cover):
    """Lay the packed paint ``pk`` over the window ``sl`` in proportion to the
    weights ``w`` (a squeezed-out ridge covering ``cover`` of each pixel). If
    none of it lands on the canvas, it stays on the tool."""
    if pk[IV] <= 0:
        return
    tot = float(w.sum())
    if tot <= 1e-9:
        brush.tip += pk / COLUMNS
        return
    share = w / tot
    d = share * pk[IV]
    _deposit(canvas, sl, share[..., None] * pk[None, None, :], d, stroke_id, cover, moved=True)


def _layer_speed(z0, z1, gap, slip):
    """Speed, as a fraction of the blade's, of a wet layer between heights z0
    and z1 above the canvas, with the blade at height ``gap``: below the blade
    the paint shears (Couette flow, speed rising with height, times ``slip``);
    above it, it is pushed along with the blade."""
    thick = np.maximum(z1 - z0, 1e-9)
    lo, hi = np.minimum(z0, gap), np.minimum(z1, gap)
    sheared = np.maximum(hi - lo, 0.0)
    plug = np.maximum(z1 - np.maximum(z0, gap), 0.0)
    return (sheared * slip * 0.5 * (hi + lo) / gap + plug) / thick


def _flow_under_blade(canvas, sl, stroke_id, floor, level, lateral, along, c, half, ds, tx, ty, slip):
    """Move the wet paint a blade touches as it travels ``ds`` px along (tx, ty)
    in the window ``sl`` (thin-film shear and bulldozing, see ``_layer_speed``).

    The blade's footprint at each sub-step is where ``lateral`` > 0 and the
    pixel's position ``along`` the motion lies within ``half`` of the chord
    centre ``c`` (relative to the blade's position at the end of the step).
    Conservative upwind fluxes, in sub-steps short enough to be stable."""
    top, und = canvas.top[sl], canvas.und[sl]
    own = canvas.top_id[sl] == stroke_id
    # under paint, the blade doesn't feel the individual threads of the canvas
    gap = _box_blur(np.maximum(level - floor, 0.2), 2)
    # the paint under the blade must reach it somewhere for anything to happen
    reach = und[..., IV] + np.where(own, 0.0, top[..., IV])
    if not ((reach > gap - 1.0) & (lateral > 0)).any():
        return
    # two wet layers of the same paint can merge without any visible change:
    # do so, so that paint dragged in from elsewhere lands on top of them
    both = ~own & (top[..., IV] > 1e-6) & (und[..., IV] > 1e-6)
    if both.any():
        same = np.zeros_like(both)
        same[both] = _alike(top[both], und[both]) < 0.25
        und[same] += top[same]
        top[same] = 0.0
    ax, ay = abs(tx), abs(ty)
    n_sub = max(int(np.ceil(ds * (ax + ay))), 1)
    step = ds / n_sub
    xdir, ydir = (1 if tx >= 0 else -1), (1 if ty >= 0 else -1)
    for k in range(1, n_sub + 1):
        pos = -ds + k * step                       # the blade's position along the motion
        inside = lateral * np.clip(half - np.abs(along - pos - c) + 0.5, 0.0, 1.0)
        if not inside.any():
            continue
        # the flow is smooth over a pixel or two: speeds from lightly smoothed heights
        u = _box_blur(und[..., IV], 1)
        t = _box_blur(np.where(own, 0.0, top[..., IV]), 1)
        shielded = own & (top[..., IV] > 0.3)       # the blade glides on its own fresh paint
        contact = inside * np.clip(u + t - gap + 1.0, 0.0, 1.0) * ~shielded
        if not contact.any():
            continue
        # viscosity: neighbouring paint drags each other along, so the speed varies
        # smoothly (without it, the paint tears into foam)
        frac_u = np.clip(_box_blur(contact * _layer_speed(0.0, u, gap, slip), _VISCOUS), 0.0, 1.0) * step
        frac_t = np.clip(_box_blur(contact * _layer_speed(u, u + t, gap, slip), _VISCOUS), 0.0, 1.0) * step * ~own
        for layer, frac in ((und, frac_u), (top, frac_t)):
            fx = (frac * ax).astype(np.float32)[..., None]
            fy = (frac * ay).astype(np.float32)[..., None]
            mx, my = layer * fx, layer * fy
            layer -= mx + my
            if xdir > 0:
                layer[:, 1:] += mx[:, :-1]
            else:
                layer[:, :-1] += mx[:, 1:]
            if ydir > 0:
                layer[1:] += my[:-1]
            else:
                layer[:-1] += my[1:]


def _place(canvas, pk, x0, y0, sx, sy, stroke_id, layer="top"):
    """Put the packed paint ``pk`` (a window whose top-left pixel is (x0, y0))
    back on the canvas, moved by (sx, sy) pixels: paint pushed ahead of the
    tool. ``layer="und"`` slides it along within the lower wet layer."""
    H, W = canvas.height, canvas.width
    m = int(np.ceil(max(abs(sx), abs(sy)))) + 1
    h, w = pk.shape[:2]
    rx0, ry0 = max(x0 - m, 0), max(y0 - m, 0)
    rx1, ry1 = min(x0 + w + m, W), min(y0 + h + m, H)
    buf = np.zeros((ry1 - ry0, rx1 - rx0, PACKED), np.float32)
    buf[y0 - ry0:y0 - ry0 + h, x0 - rx0:x0 - rx0 + w] = pk
    moved = _shift(buf, sx, sy)
    d = moved[..., IV]
    if d.max() <= 0:
        return
    if layer == "und":
        canvas.und[ry0:ry1, rx0:rx1] += moved
        return
    # moved paint doesn't change how much of each pixel is covered
    _deposit(canvas, (slice(ry0, ry1), slice(rx0, rx1)), moved, d, stroke_id, np.zeros_like(d), moved=True)


def _lift_off(canvas, sl, under, amp, cell, direction, rng):
    """Paint under a tool lifted off it is pulled up into torn crests and
    peaks, stretched along the way the tool was moving (volume-conserving)."""
    if amp <= 0 or not under.any():
        return
    top = canvas.top[sl]
    V = top[..., IV].astype(np.float64)
    wv = under * V
    if wv.sum() <= 1e-9:
        return
    hh, ww = under.shape
    n = 0.7 * aniso_noise(hh, ww, 4.0 * cell, cell, direction, rng, octaves=3) \
        + 0.3 * fractal_noise_2d(hh, ww, cell, rng, octaves=2)
    n = (n - n.min()) / max(np.ptp(n), 1e-9)
    crest = (1.0 - np.abs(2.0 * n - 1.0)) ** 3
    m = (wv * crest).sum() / wv.sum()
    fac = np.clip(1.0 + 2.0 * amp * under * (crest - m), 0.05, None)
    fac *= V.sum() / max((V * fac).sum(), 1e-12)
    top *= fac.astype(np.float32)[..., None]


def _level(canvas, region, iters: int = 12):
    """Let wet paint settle. Oil paint holds its shape up to a point: it only
    flows where it is steeper than its stiffness (body) can hold. Fluid paint
    levels out almost flat (its brush marks and ridges disappear); stiff paint
    keeps them and only slumps a little at sheer edges. Thin, fluid paint
    stays within the area it covers; thick paint can slump beyond it."""
    top, und = canvas.top[region], canvas.und[region]
    V = top[..., IV] + und[..., IV]
    if float(V.max()) < 0.3:
        return
    body = np.where(V > 1e-6, (top[..., IB] + und[..., IB]) / np.maximum(V, 1e-6), 1.0)
    crit = _SLOPE_FLUID + (_SLOPE_STIFF - _SLOPE_FLUID) * body ** 3     # steepest slope it holds, per px
    S = canvas.dry_height[region] + V
    steep = np.zeros(V.shape, bool)
    steep[:, :-1] |= np.abs(np.diff(S, axis=1)) > crit[:, :-1]
    steep[:, 1:] |= np.abs(np.diff(S, axis=1)) > crit[:, 1:]
    steep[:-1] |= np.abs(np.diff(S, axis=0)) > crit[:-1]
    steep[1:] |= np.abs(np.diff(S, axis=0)) > crit[1:]
    steep &= V > 0.05
    if not steep.any():
        return
    ys, xs = np.nonzero(steep.any(axis=1))[0], np.nonzero(steep.any(axis=0))[0]
    pad = iters + 2
    y0, y1 = max(ys[0] - pad, 0), min(ys[-1] + pad + 1, V.shape[0])
    x0, x1 = max(xs[0] - pad, 0), min(xs[-1] + pad + 1, V.shape[1])
    top, und, crit = top[y0:y1, x0:x1], und[y0:y1, x0:x1], crit[y0:y1, x0:x1]
    base = canvas.dry_height[region][y0:y1, x0:x1]
    for _ in range(iters):
        for axis in (0, 1):
            V = top[..., IV] + und[..., IV]
            S = base + V
            a = (slice(None, -1), slice(None)) if axis == 0 else (slice(None), slice(None, -1))
            b = (slice(1, None), slice(None)) if axis == 0 else (slice(None), slice(1, None))
            dS = S[a] - S[b]
            # downhill flow beyond what the paint can hold; onto bare canvas only from a sheer edge
            qa = np.clip(dS - crit[a], 0.0, None) * ((V[b] > 0.1) | (dS > 2.0))
            qb = np.clip(-dS - crit[b], 0.0, None) * ((V[a] > 0.1) | (-dS > 2.0))
            fa = np.clip(0.25 * qa / np.maximum(V[a], 1e-6), 0.0, 0.25)
            fb = np.clip(0.25 * qb / np.maximum(V[b], 1e-6), 0.0, 0.25)
            for layer in (top, und):
                ma = layer[a] * fa[..., None]
                mb = layer[b] * fb[..., None]
                layer[a] += mb - ma
                layer[b] += ma - mb


def _stir(top, und, m, top_a=None, und_a=None):
    """Homogenise the two wet sublayers by a fraction ``m`` (0 = untouched,
    1 = both become the same mixture). Each sublayer keeps its volume."""
    tv, uv = top[..., IV], und[..., IV]
    m = np.where((tv > 0) & (uv > 0), m, 0.0).astype(np.float32)
    if not m.any():
        return
    if top_a is not None:
        np.maximum(und_a, np.where(m > 0.01, top_a, 0.0), out=und_a)
    share = tv / np.maximum(tv + uv, 1e-12)
    # top <- (1 - m) * top + m * share * (top + under); under keeps the rest
    total = top + und
    new_top = (1.0 - m)[..., None] * top + (m * share)[..., None] * total
    top[...] = new_top
    und[...] = total - new_top


def _drag(canvas, f, x0, y0, sx, sy, p):
    """Move a fraction ``f`` of the wet paint by (sx, sy) pixels. ``f`` is
    defined on the footprint window whose top-left pixel is (x0, y0). Paint is
    conserved except what is pushed off the canvas. The older layer moves less."""
    H, W = canvas.height, canvas.width
    m = int(np.ceil(max(abs(sx), abs(sy)))) + 1
    rx0, ry0 = max(x0 - m, 0), max(y0 - m, 0)
    rx1, ry1 = min(x0 + f.shape[1] + m, W), min(y0 + f.shape[0] + m, H)
    fx0, fy0 = max(x0, 0), max(y0, 0)
    fx1, fy1 = min(x0 + f.shape[1], W), min(y0 + f.shape[0], H)
    if fx0 >= fx1 or fy0 >= fy1:
        return
    frac = np.zeros((ry1 - ry0, rx1 - rx0), np.float32)
    frac[fy0 - ry0:fy1 - ry0, fx0 - rx0:fx1 - rx0] = f[fy0 - y0:fy1 - y0, fx0 - x0:fx1 - x0]
    region = (slice(ry0, ry1), slice(rx0, rx1))
    for layer, scale in ((canvas.top[region], 1.0), (canvas.und[region], p)):
        if scale <= 0 or not layer[..., IV].any():
            continue
        moved = (frac * np.float32(scale))[..., None] * layer
        layer += _shift(moved, sx, sy) - moved


def _shift(a, sx, sy):
    """Bilinear translation: out[y, x] = a[y - sy, x - sx], zero outside."""
    ix, iy = int(np.floor(sx)), int(np.floor(sy))
    fx, fy = sx - ix, sy - iy
    h, w = a.shape[:2]
    out = np.zeros_like(a)
    for dx, wx in ((ix, 1.0 - fx), (ix + 1, fx)):
        for dy, wy in ((iy, 1.0 - fy), (iy + 1, fy)):
            wgt = wx * wy
            if wgt < 1e-6 or abs(dx) >= w or abs(dy) >= h:
                continue
            ys, yd = (slice(0, h - dy), slice(dy, h)) if dy >= 0 else (slice(-dy, h), slice(0, h + dy))
            xs, xd = (slice(0, w - dx), slice(dx, w)) if dx >= 0 else (slice(-dx, w), slice(0, w + dx))
            out[yd, xd] += np.float32(wgt) * a[ys, xs]
    return out
