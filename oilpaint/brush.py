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
from .noise import fractal_noise_2d, value_noise_1d
from .paint import ColorLike, Paint, as_paint
from .spline import Path, Profile, as_path, evaluate_profile

N = sp.N_BANDS
IS, IV = N, N + 1          # packed indices of scattering and volume
PACKED = N + 2

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
    return v


SHAPES = ("round", "flat", "filbert", "fan")


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
        ``"round"``, ``"flat"`` (square ends), ``"filbert"`` (a flat with an
        oval tip) or ``"fan"`` (a thin spread of separate bristle clumps).
    width:
        Default width in pixels (a stroke's own ``width`` overrides it).
    aspect:
        Thickness of the footprint relative to its width: 1 for a round,
        about 0.2-0.3 for a flat, 0.05 for a knife blade. Thin flats lay crisp
        ribbons; thick ones are chunkier.
    angle:
        Degrees between the brush's long side and the stroke's cross
        direction: 0 drags a flat broadside (wide stroke), 90 drags it on its
        edge (thin line). Strokes can vary it along their length.
    follow_path:
        If True the angle is relative to the stroke's direction (the brush
        turns with the hand). If False it is fixed on the canvas, measured from
        horizontal, like a calligraphy nib: the stroke thickens and thins as it
        changes direction.
    stiffness:
        0 = very soft (sable, squirrel) to 1 = very stiff (hog bristle, knife).
        Stiff tools leave stronger ridges and ragged edges, push more paint
        and blend less; soft ones lay smooth paint and blend gently.
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
        Number of separate bristle clumps (fan brushes), 0 = continuous.
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

    def physics(self) -> dict:
        """Stroke behaviour derived from the brush's character."""
        st, br = self.stiffness - 0.5, self.bristles - 0.5
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
        blend = self.blending if self.blending is not None else 0.4 + 0.5 * st
        derived["blending"] = float(np.clip(blend, 0.0, 1.0))
        derived["ridge"] = 0.3 + 0.4 * st          # paint squeezed to the edges by pressure
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
        cols = _across_weights(across, ring=self.ring)[:, None] * _pack(as_paint(paint), volume)[None, :]
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
        col_vol = volume * _across_weights(across, ring=self.ring)
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
        return Paint.from_optics(total[:N] / total[IV], total[IS] / total[IV])

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
        pressure lays paint on top.
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

        ``mixing``, ``pickup``, ``drag``, ``streaks``, ``smear``,
        ``blending`` and ``taper`` default to values derived from the brush
        type.
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
    eff["width"] = stroke.width if stroke.width is not None else bt.width
    eff["taper"] = stroke.taper if stroke.taper is not None else bt.taper
    eff["angle"] = stroke.angle if stroke.angle is not None else bt.angle
    eff["follow_path"] = stroke.follow_path if stroke.follow_path is not None else bt.follow_path
    return brush, bt, eff


_CHORD_CACHE: dict = {}
_SUPERELLIPSE = {"flat": 10.0, "filbert": 3.0, "fan": 6.0, "round": 2.0}


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
    n = _SUPERELLIPSE[shape]
    th = np.linspace(0, 2 * np.pi, 400, endpoint=False)
    cu, sv = np.cos(th), np.sin(th)
    u = np.sign(cu) * np.abs(cu) ** (2 / n)
    v = aspect * np.sign(sv) * np.abs(sv) ** (2 / n)
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
    if is_dab:  # a dab: press the brush down with a tiny movement
        pts = np.vstack([pts[0] - [0.05 * widths[0], 0], pts[0] + [0.05 * widths[0], 0]])
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
    if is_dab:   # bristle tips pressed straight down: a blotchy 2-D pattern (stippling)
        tex2 = fractal_noise_2d(96, 96, 96 / (9.0 * density), rng, octaves=2)
        tex2 = (tex2 - tex2.min()) / max(np.ptp(tex2), 1e-9)
    aspect = 1.0 if circle else bt.aspect
    reach = 1.0 if circle else float(np.hypot(1.0, aspect))
    streaks, rough, ridge = eff["streaks"], eff["rough"], eff["ridge"]

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
        R = rmax * reach
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
        if bt.clumps > 0:      # separate clumps of bristles (fan brushes)
            m = 0.5 + 0.5 * np.cos(np.pi * bt.clumps * (np.clip(ucoord, -1, 1) + 1.0) + 2.0 * (hair - 0.5))
            profile = profile * _smoothstep(0.3, 0.7, m)
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

        # 0) drag: wet paint under the brush is pushed along with it
        if eff["drag"] > 0:
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
                colf = (uc + 1.0) * 0.5 * (COLUMNS - 1)
                c0 = np.minimum(colf.astype(int), COLUMNS - 2)
                fr = colf - c0
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

        # 3) deposit from the tips. Each column's load sets how thick its
        #    paint goes on and when it starts to run dry.
        # paint deep in the heel only reaches the tips by seeping down (above)
        col_vol = brush.reservoir[:, IV] + brush.tip[:, IV]
        load = float(col_vol.sum()) / (Wn * Wn)
        thick = _deposit_thickness(load) * (0.5 + p)
        col_load = col_vol * COLUMNS / (Wn * Wn)
        col_thick = _deposit_thickness(col_load) * (0.5 + p)
        col_fill = np.clip((col_load - _RESIDUAL) / 0.6, 0.0, 1.0)
        # Bristles that were just in wet paint are coated and keep laying a
        # continuous smear for a short distance, even with little load.
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
        tip_v = brush.tip[:, IV]
        if not dry and tip_v.sum() > 0 and bt.deposits:
            d = at_pixels(col_thick) * q * f_dep      # absorbent / toothy grounds pull paint off the brush
            # each column can only give what it holds
            demand = assign @ d.ravel()
            give = np.where(demand > tip_v, tip_v / np.maximum(demand, 1e-30), 1.0)
            comp = brush.tip / np.where(tip_v > 0, tip_v, 1.0)[:, None]     # paint per unit volume
            parts = [(idx, d * wt * give[idx]) for idx, wt in taps]       # volume from each column
            d_px = sum(v for _, v in parts)
            if d_px.sum() > 0:
                dep = sum(v[..., None] * comp[idx] for idx, v in parts)
                _deposit(canvas, sl, dep, d_px, stroke_id, cover[wsl])
                used = sum(np.bincount(idx.ravel(), v.ravel(), COLUMNS) for idx, v in parts)
                brush.tip -= used[:, None] * comp
                np.maximum(brush.tip[:, IV], 0.0, out=brush.tip[:, IV])
            off_canvas = thick * float((expo * streak).sum() - contact.sum())
            left = brush.tip[:, IV].sum()
            if off_canvas > 0 and left > 0:
                brush.tip *= max(1.0 - off_canvas / left, 0.0)

        # 4) overflow back into the belly (this is how the brush gets dirty)
        cap_col = capacity / COLUMNS
        over = brush.tip[:, IV] - cap_col
        if (over > 0).any():
            frac = np.where(over > 0, over / np.maximum(brush.tip[:, IV], 1e-30), 0.0)[:, None]
            moved = brush.tip * frac
            brush.tip -= moved
            brush.reservoir += moved

        # 5) stir: pressure churns the fresh top layer and the wet paint below
        #    into each other
        m = 1.0 - np.exp(-eff["mixing"] * 3.0 * p * p * contact * f_mix)
        _stir(top, und, m.astype(np.float32), canvas.top_a[sl], canvas.und_a[sl])

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


def _deposit(canvas, sl, dep, d, stroke_id, cover):
    """Lay paint ``dep`` (packed, per pixel; volume ``d``) on top of the canvas."""
    top, und = canvas.top[sl], canvas.und[sl]
    top_a, und_a = canvas.top_a[sl], canvas.und_a[sl]
    # Paint from an earlier stroke stops being "top" once new paint covers it.
    ids = canvas.top_id[sl]
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
