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
from .noise import aniso_noise, fractal_noise_2d, value_noise_1d, value_noise_2d
from .paint import ColorLike, Paint, as_paint
from .spline import Path, Profile, as_path, evaluate_profile

N = sp.N_BANDS
IS, IV, IB = N, N + 1, N + 2      # packed indices of scattering, volume and body (stiffness x volume)
PACKED = N + 3

# Two paints this alike (see _alike) count as much the same paint.
_SAME_PAINT = 0.6
# A new film of paint this thick covers the wet paint under it about two-thirds
# of the way: it buries that much of it (see _deposit) ...
_COVERING_FILM = 0.15
# ... and once it would bury this much, it buries it all (see _Pending).
_BURIED = 0.9
# Where the body of a brush passes over (a full pass at this much of its
# full-strength profile), even a light touch lays its paint on top.
_BODY = (0.5, 0.8)
# Two wet layers this alike look the same mixed or one over the other (a sky
# worked in several strokes: ~0.1; white over that sky: ~0.4 and up).
_SAME_LAYER = 0.2
# Paint thinner than this can't form a continuous film (as canvas.MIN_FILM).
_MIN_FILM = 0.15
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
# How far the colour bands of a loaded knife wander sideways (in blade half-lengths).
_KNIFE_BAND_WANDER = 0.12
# Share of the wet paint standing above a moving blade near the edges of its
# mark that spills out sideways as lips (the rest is bulldozed ahead), and how
# near (px): a narrow mark spills almost all of it, a wide scraper only at its ends.
_KNIFE_SPILL = 0.3
_KNIFE_SPILL_PX = 8.0
# Below about this pressure a knife skims: only the bead of paint on its edge
# touches the high points of the surface.
_KNIFE_SKIM = 0.3
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
    dig:
        0..1: how deep a point cuts into the wet paint under it, as a share
        of that paint (1: down to the dry paint or the canvas beneath), pushing
        it aside into low ridges. The handle of a brush scratched through wet
        paint (sgraffito); see :data:`oilpaint.brushes.BRUSH_HANDLE`.
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
    dig: float = 0.0
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
            # (a knife drags wet paint rather than picking it up: pickup and drag unused)
            derived = dict(streaks=0.03, pickup=0.0, drag=0.0, mixing=0.12, rough=0.04)
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
        derived["lift"] = 0.3 if self.rigid else 0.0
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
        Brushes only: a knife drags wet paint instead (a thin-film flow).
    drag:
        How much wet paint the brush pushes along with it (0..1). Brushes only.
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

    The tool moves along the path in short straight segments. For every pixel
    we compute its *exposure* during a segment: the fraction of a full pass
    of the footprint over that pixel that happens during this segment (exact
    for a convex footprint sweeping a straight segment). Exposures over a
    whole pass add up to the tool's lateral profile, so the result is smooth
    and does not depend on the step size. Each segment is then one step of
    :func:`_brush_step` or :func:`_knife_step`.
    """
    run = _Run(canvas, stroke)
    step = _knife_step if run.bt.rigid else _brush_step
    pending = None
    if not run.bt.rigid:
        y0, y1, x0, x1 = run.bounds()
        if y0 < y1 and x0 < x1:
            pending = canvas._pending = _Pending(canvas, run.stroke_id, y0, y1, x0, x1)
    try:
        for i in range(1, len(run.pts)):
            s = run.footprint(i)
            if s is not None:
                step(run, s)
    finally:
        canvas._pending = None
    if pending is not None:
        pending.settle()
    run.settle()
    return run.brush


class _Run:
    """One stroke in progress: the tool, the sampled path, and the textures
    and constants that stay the same along the stroke."""

    def __init__(self, canvas, stroke: Stroke):
        self.canvas = canvas
        path: Path = as_path(stroke.path)
        rng = self.rng = np.random.default_rng(
            stroke.seed if stroke.seed is not None else canvas._rng.integers(1 << 31))
        brush, bt, eff = self.brush, self.bt, self.eff = _resolve(stroke)

        n_probe = max(path.n_knots, 2) * 8
        w_probe = evaluate_profile(eff["width"], np.linspace(0, 1, n_probe),
                                   np.linspace(0, max(path.n_knots - 1, 0), n_probe), path.n_knots)
        w_typ = self.w_typ = max(float(np.mean(w_probe)), 1.0)
        spacing = float(np.clip(0.25 * w_typ, 0.5, 8.0))
        taper = eff["taper"]
        t0, t1 = (taper, taper) if np.isscalar(taper) else taper
        circle = self.circle = bt.shape == "round" and abs(bt.aspect - 1.0) < 1e-6

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
        is_dab = self.is_dab = len(pts) == 1
        if is_dab:  # a dab: press the brush down with a tiny movement (small next to its narrow side)
            nudge = max(0.05 * widths[0] * min(1.0, bt.aspect), 0.25)
            pts = np.vstack([pts[0] - [nudge, 0], pts[0] + [nudge, 0]])
            widths = np.repeat(widths, 2)
            pressure = np.repeat(pressure, 2)
            angles = np.repeat(angles, 2)
            rolls = np.repeat(rolls, 2)
        self.pts, self.pressure, self.angles, self.rolls = pts, pressure, angles, rolls

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
        self.tex = (tex - tex.min()) / max(np.ptp(tex), 1e-9)
        self.tex_x = np.linspace(-1.3, 1.3, n_tex)
        wobble = value_noise_1d(len(pts) + 2, max(2.0, 0.8 * w_typ / spacing), rng)[:len(pts)] - 0.5
        # a knife's height wavers as it is pushed along (faint ripples across the
        # slab: chatter), and a hand never presses both ends of the blade quite
        # equally (the slab is thicker towards one side); the lips squeezed out
        # along a stroke swell and break
        arc = self.arc = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))])
        period = max(0.4 * w_typ, 2.0)          # ripples about half a blade width apart
        ripple = value_noise_1d(int(arc[-1] / period) + 4, 1.0, rng)
        self.chatter = np.interp(arc / period, np.arange(len(ripple)), ripple) - 0.5
        swell = value_noise_1d(int(arc[-1] / (2.0 * period)) + 4, 1.0, rng)
        self.lip = np.interp(arc / (2.0 * period), np.arange(len(swell)), swell)
        self.blade_tilt = float(np.clip(rng.normal(0.0, 0.08), -0.2, 0.2))
        if is_dab:   # bristle tips pressed straight down: a blotchy 2-D pattern (stippling)
            tex2 = fractal_noise_2d(96, 96, 96 / (9.0 * density), rng, octaves=2)
            self.tex2 = (tex2 - tex2.min()) / max(np.ptp(tex2), 1e-9)
        aspect = self.aspect = 1.0 if circle else bt.aspect
        self.reach = 1.0 if circle else float(np.hypot(1.0, aspect))
        self.area_unit = _footprint_area("round" if circle else bt.shape, aspect)    # in half-widths squared
        self.stiff_blade = float(np.clip((bt.stiffness - 0.4) / 0.6, 0.0, 1.0))
        self.col_u = (_RING_X if brush.ring else _COL_X) * 2.0 - 1.0    # where each column sits across the mark
        # squeezed-out ridges are at most this wide (px); they lie just outside the mark
        self.margin = float(np.clip(0.12 * w_typ * (1.0 + bt.spread), 0.6, 7.0)) + 1.0

        # brush half-width at every sample: size, a slow wobble, and how far the
        # hairs splay under pressure (a pointed round paints thin with a light touch)
        radii = 0.5 * widths * (1.0 + 0.06 * wobble)
        self.nominal = radii.copy()          # the bristle bundle's size; pressure only splays it
        if bt.spread > 0:
            radii = radii * ((1.0 - bt.spread) + bt.spread * (np.maximum(pressure, 0.02) / 0.5) ** 0.8)
        self.radii = radii
        self.feather = bt.softness * 3.0     # soft brushes fade in and out over this many widths

        surf = canvas.surface
        self.surface_acts = surf.affects_paint
        # how much the surface texture (vs. the hair pattern) decides where a dry brush catches
        self.tooth_w = float(np.clip(0.3 + (surf.tooth - 0.5), 0.0, 0.85))

        if bt.rigid:
            # The wet paint lying on the canvas before the stroke: the blade's ends
            # push it aside into lips (a pile it is carrying mostly rides along).
            self.wet_before = canvas.top_v + canvas.und_v
            # Paint on a blade isn't held in lanes like paint in bristles: its
            # colour bands wander a little from side to side along the stroke.
            period = max(0.5 * w_typ, 4.0)
            n_s = int(arc[-1] / period) + 3
            self.bands = value_noise_2d(8 * n_s, 64, 8.0, rng)
            self.band_period = period

        self.stroke_id = canvas._new_stroke_id()
        self.dry = False
        brush.wet = 0.0

    # ------------------------------------------------------------------
    def footprint(self, i: int) -> Optional["_Step"]:
        """Where the tool touches the canvas while it moves along segment
        ``i`` (from point i-1 to point i), or None if it touches nothing."""
        canvas, brush, bt, eff = self.canvas, self.brush, self.bt, self.eff
        pts, pressure, angles = self.pts, self.pressure, self.angles
        circle, aspect, rough, ridge = self.circle, self.aspect, eff["rough"], eff["ridge"]
        H, W = canvas.height, canvas.width
        (ax, ay), (cx, cy) = pts[i - 1], pts[i]
        ds = float(np.hypot(cx - ax, cy - ay))
        if ds < 1e-9:
            return None
        s = _Step()
        s.i, s.ds, s.cx, s.cy = i, ds, cx, cy
        tx, ty = s.tx, s.ty = (cx - ax) / ds, (cy - ay) / ds
        p = s.p = 0.5 * (pressure[i] + pressure[i - 1])
        r0, r1 = s.r0, s.r1 = float(self.radii[i - 1]), float(self.radii[i])
        r = s.r = 0.5 * (r0 + r1)
        rmax = max(r0, r1)
        s.Wd = 2.0 * r
        s.Wn = float(self.nominal[i - 1] + self.nominal[i])   # how loaded the brush is is relative to its real size
        # orientation of the brush's long side relative to the stroke's cross direction
        ang = 0.5 * (angles[i] + angles[i - 1])
        phi = ang if eff["follow_path"] else ang - np.arctan2(tx, -ty)
        side = s.side = 1.0 if np.cos(phi) >= 0 else -1.0    # a brush turned past 90 degrees has its sides swapped
        phi = s.phi = (phi + np.pi / 2) % np.pi - np.pi / 2

        # --- footprint geometry on a window around the swept segment -------
        R = rmax * self.reach + (3.0 if bt.rigid else 0.0)
        x0, x1 = int(np.floor(min(ax, cx) - R - 1)), int(np.ceil(max(ax, cx) + R + 1)) + 1
        y0, y1 = int(np.floor(min(ay, cy) - R - 1)), int(np.ceil(max(ay, cy) + R + 1)) + 1
        cx0, cx1, cy0, cy1 = max(x0, 0), min(x1, W), max(y0, 0), min(y1, H)
        if cx0 >= cx1 or cy0 >= cy1:
            _use_paint_off_canvas(brush, s.Wd, ds, p)
            return None
        s.x0, s.x1, s.y0, s.y1, s.cx0, s.cy0 = x0, x1, y0, y1, cx0, cy0
        gx = (np.arange(x0, x1) + 0.5 - cx)[None, :]
        gy = (np.arange(y0, y1) + 0.5 - cy)[:, None]
        along = s.along = gx * tx + gy * ty         # <= 0 behind the brush centre
        lat_px = s.lat_px = -gx * ty + gy * tx
        # the brush size changes smoothly through the segment (no steps when a flat tapers)
        r_px = r1 + (r0 - r1) * np.clip(-along / ds, 0.0, 1.0) if r0 != r1 else r
        lat = lat_px / r_px                         # signed, in brush half-widths
        if circle:
            lmax = 1.0
            ucoord = lat * side
        else:
            s.grid, s.c_tab, s.h_tab, lmax = _chord_table(bt.shape, aspect, phi)
            ucoord = side * lat / _signed_floor(np.cos(phi), aspect)       # position across the bristles
        s.lmax, s.ucoord = lmax, ucoord
        hair = np.interp(ucoord, self.tex_x, self.tex)       # 0..1
        if self.is_dab:
            ix = np.clip(((ucoord + 1.3) / 2.6 * 95).astype(int), 0, 95)
            iy = np.clip(((along / r + 1.3) / 2.6 * 95).astype(int), 0, 95)
            hair = self.tex2[iy, ix]
        s.hair = hair
        lr = np.abs(lat) / lmax * (1.0 + rough * (hair - 0.5))
        # Anti-aliasing: how much of each pixel the footprint covers across its
        # edge (a pixel-wide ramp centred on the true edge). This is geometry:
        # it sets the pixel's coverage, not the paint's thickness.
        edge_px = (1.0 - lr) * (r_px * lmax)
        aa = s.aa = _smoothstep(-0.5 * _EDGE_PX, 0.5 * _EDGE_PX, edge_px)
        # Flat-topped profile; pressing harder squeezes paint into low ridges at the edges.
        falloff = aa
        if bt.softness > 0 and not self.is_dab:    # a long, feathery falloff that stays visible even under thick paint
            falloff = falloff * _smoothstep(1.0, 1.0 - bt.softness, lr) ** (1.0 + 4.0 * bt.softness)
        profile = falloff * (0.92 + ridge * p * _smoothstep(0.55, 0.95, lr))
        if bt.slant > 0:       # bias-cut bristles: the toe (u = -1) touches first
            press = 0.25 + 0.9 * p
            profile = profile * np.clip((press - bt.slant * (1.0 + np.clip(ucoord, -1.0, 1.0)) / 2.0) / 0.3,
                                        0.0, 1.0)
        if bt.clumps > 0:      # separate clumps of bristles (fan brushes), or the teeth of a comb
            m = 0.5 + 0.5 * np.cos(np.pi * bt.clumps * (np.clip(ucoord, -1, 1) + 1.0) + 2.0 * (hair - 0.5))
            profile = profile * _smoothstep(0.3, 0.7, m)
            s.between = falloff * _smoothstep(0.6, 0.2, m)       # the gaps between them
        s.profile = profile
        # Chord of the footprint through each pixel along the motion, and the
        # part of it swept during this segment.
        if circle:
            c = 0.0
            h = np.maximum(r_px * np.sqrt(np.clip(1.0 - lr * lr, 0.0, None)), 0.5)
        else:
            c = np.interp(lat, s.grid, s.c_tab) * r_px
            h = np.maximum(np.interp(lat, s.grid, s.h_tab) * r_px, 0.5)
        s.c, s.h = c, h
        rel = s.rel = along - c
        if self.is_dab:
            if bt.softness > 0:    # a soft brush pressed down: feathered all round, not just at its sides
                # distance from the centre in footprint radii (exact for an ellipse)
                rr = np.sqrt(lr * lr + np.clip(rel / h, -1.0, 1.0) ** 2 * np.clip(1.0 - lr * lr, 0.0, None))
                profile = profile * _smoothstep(1.0, 1.0 - bt.softness, rr) ** (1.0 + 4.0 * bt.softness)
            # Tapped lightly, only the tips of the bristles touch: a stippled,
            # broken mark (pressed harder, the whole brush comes down)
            tip_thr = 0.75 - 1.5 * p
            if tip_thr > 0:
                profile = profile * _smoothstep(tip_thr - 0.12, tip_thr + 0.12, hair)
            s.profile = profile
        overlap = np.clip(np.minimum(0.0, rel + h) - np.maximum(-ds, rel - h), 0.0, None)
        # Pixels under the brush where it touches down / lifts off get topped
        # up to a full pass, so strokes have solid ends.
        # Soft brushes are lowered and lifted gently, so their ends feather.
        end_k = 1.0 if self.is_dab else max(0.0, 1.0 - 1.5 * bt.softness)
        half_aa = 0.5 * _EDGE_PX
        if i == 1 and end_k > 0:
            a0 = rel + ds
            overlap += end_k * _smoothstep(-h - half_aa, -h + half_aa, a0) * np.clip(h - a0, 0.0, 2.0 * h)
        if i == len(pts) - 1 and end_k > 0:
            overlap += end_k * _smoothstep(h + half_aa, h - half_aa, rel) * np.clip(h + rel, 0.0, 2.0 * h)
        s.overlap = overlap
        swept = overlap / (2.0 * h)                 # fraction of a full pass
        expo = profile * swept
        s.cover = aa * swept                        # geometric coverage added during this segment
        if self.feather > 0 and not self.is_dab:
            s_pix = self.arc[i] + along             # distance along the stroke
            span = max(self.feather * 2.0 * r, 1.0)
            expo = expo * _smoothstep(-r, -r + span, s_pix) * _smoothstep(-r, -r + span, self.arc[-1] - s_pix)
        s.expo = expo
        s.streak = np.clip(1.0 + eff["streaks"] * 1.6 * (hair - 0.5), 0.0, None)

        sl = s.sl = (slice(cy0, cy1), slice(cx0, cx1))
        wsl = s.wsl = (slice(cy0 - y0, cy1 - y0), slice(cx0 - x0, cx1 - x0))
        s.expo_c = expo[wsl]
        if not s.expo_c.any():
            return None
        s.streak_c = s.streak[wsl]
        s.contact = s.expo_c * s.streak_c
        s.top, s.und = canvas.top[sl], canvas.und[sl]
        s.floor = canvas._floor[sl]
        s.touchdown = i == 1 or self.is_dab
        self._surface_factors(s)
        self._bristle_taps(s)
        return s

    def _surface_factors(self, s):
        """How the bare surface changes pickup, deposit, mixing and drag."""
        surf = self.canvas.surface
        if self.surface_acts:
            # Where the brush works directly on the priming (little paint
            # below), the surface itself decides how paint grips and slides.
            dry_h = self.canvas.dry_height[s.sl]
            wet = s.top[..., IV] + s.und[..., IV]
            bare = np.exp(-(wet + dry_h) / 0.4)
            # Lifting wet paint depends on how it adheres to what is beneath: it
            # wipes cleanly off a slick surface but stays stained into an
            # absorbent one. Dry paint below behaves like a neutral surface.
            on_ground = np.exp(-wet / 1.5) * np.exp(-dry_h / 0.4)
            s.f_pick = np.clip(1.0 + on_ground * (2.0 * (surf.slickness - 0.5) - 0.8 * surf.absorbency), 0.1, None)
            s.f_dep = np.clip(1.0 + bare * (0.6 * surf.absorbency + 0.4 * (surf.tooth - 0.5)
                                            - 0.4 * (surf.slickness - 0.5)), 0.4, None)
            s.f_mix = 1.0 - 0.6 * surf.absorbency * bare
            s.f_drag = 1.0 + (surf.slickness - 0.5) * float(bare.mean())
        else:
            s.f_pick = s.f_dep = s.f_mix = s.f_drag = 1.0

    def _bristle_taps(self, s):
        """Which bristle columns are above each pixel: (column, weight) taps,
        blending linearly between the nearest columns. On a thin brush one
        pixel spans several columns: sample across it so that every column
        reaches the canvas (else paint in the columns between pixel centres
        would only come out by diffusion, and thin strokes would run long)."""
        brush = self.brush
        u_px = s.ucoord[s.wsl]
        if self.bt.rigid:
            along_stroke = np.clip((self.arc[s.i] + s.along[s.wsl]) / self.band_period * 8.0, 0.0,
                                   self.bands.shape[0] - 1.001)
            across = np.clip((u_px + 1.0) * 0.5 * 63.0, 0.0, 62.999)
            u_px = u_px + _KNIFE_BAND_WANDER * (2.0 * _bilinear(self.bands, along_stroke, across) - 1.0)
        span = 1.0 / max(s.r * (1.0 if self.circle else abs(_signed_floor(np.cos(s.phi), self.aspect))), 1e-6)
        span_cols = span * COLUMNS * (0.25 if brush.ring else 0.5)
        nsub = int(min(max(np.ceil(span_cols / 2.0 - 1e-9), 1), 8))
        rho = 0.5 * float(self.rolls[s.i] + self.rolls[s.i - 1])
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
        s.npx = u_px.size
        s.assign = np.zeros((COLUMNS, s.npx))                   # pixel -> column weights
        flat_i = np.arange(s.npx)
        for idx, wt in taps:
            s.assign[idx.ravel(), flat_i] += wt.ravel()
        s.taps = taps

    # ------------------------------------------------------------------
    def bounds(self):
        """(y0, y1, x0, x1): the part of the canvas the stroke can reach."""
        canvas, pts = self.canvas, self.pts
        H, W = canvas.height, canvas.width
        pad = float(self.radii.max()) * self.reach + self.margin + 2.0
        bx0, by0 = int(max(np.floor(pts[:, 0].min() - pad), 0)), int(max(np.floor(pts[:, 1].min() - pad), 0))
        bx1, by1 = int(min(np.ceil(pts[:, 0].max() + pad) + 1, W)), int(min(np.ceil(pts[:, 1].max() + pad) + 1, H))
        return by0, by1, bx0, bx1

    def settle(self):
        """The paint settles: fluid paint levels out, sheer edges of thick paint slump."""
        canvas = self.canvas
        by0, by1, bx0, bx1 = self.bounds()
        if bx0 < bx1 and by0 < by1:
            region = (slice(by0, by1), slice(bx0, bx1))
            _level(canvas, region)
            _fix_coverage(canvas, region)


class _Step:
    """The footprint of one segment of a stroke, on a window of the canvas.

    Full-window arrays (``along``, ``lat_px``, ``hair``, ``aa``, ``c``, ``h``,
    ``rel``, ``overlap``, ``expo``, ...) cover the window around the swept
    segment, which may stick out of the canvas; ``wsl`` crops them to the
    canvas part ``sl``. ``expo_c`` (exposure), ``contact`` (exposure times
    hair streaks) and ``top``/``und``/``floor`` are already cropped.
    ``taps`` and ``assign`` map bristle columns to pixels."""

    def at_pixels(self, per_column):
        """A per-column quantity, as seen at each pixel under the brush."""
        return sum(wt * per_column[idx] for idx, wt in self.taps)


# ----------------------------------------------------------------------------
# one step of a brush, one step of a knife


def _brush_step(run: _Run, s: _Step) -> None:
    canvas, brush, bt, eff = run.canvas, run.brush, run.bt, run.eff
    p, top, und, wsl = s.p, s.top, s.und, s.wsl

    # 0) drag: wet paint under the brush is pushed along with it
    if eff["drag"] > 0:
        f = eff["drag"] * s.f_drag * (0.5 + p) * s.profile * np.clip(s.overlap / s.ds, 0.0, 1.0)
        _drag(canvas, f, s.x0, s.y0, s.tx * s.ds, s.ty * s.ds, p)

    # 1) refill the tips, work the paints on the brush, pick up wet paint
    capacity = max(eff["smear"], 0.05) * 0.8 * s.Wd * s.Wd
    _work_load(run, s, capacity)
    _pick_up(run, s)

    # 2) how much paint the brush carries, per column (load units: width
    #    squares). Paint beyond what the hair holds is a glob on the outside:
    #    it goes on thick, the more so the stiffer the paint.
    ld = _Load(brush, s.Wn)
    # small brushes have long hair for their width: they hold far more for their
    # size (a rigger holds enough for a line hundreds of widths long)
    hold = _HOLD * bt.capacity * (1.3 - 0.6 * p) * max(1.0, 20.0 / s.Wn) ** 1.5
    glob_gain = _IMPASTO * (1.2 - 0.5 * p)
    thick = _deposit_thickness(ld.load) * (0.5 + p) + glob_gain * ld.body * max(ld.load - hold, 0.0)
    col_glob = glob_gain * ld.col_body * np.maximum(ld.col_load - hold, 0.0)
    col_thick = _deposit_thickness(ld.col_load) * (0.5 + p) + col_glob
    col_fill = np.clip((ld.col_load - _RESIDUAL) / 0.6, 0.0, 1.0)

    # 3) the bristles comb through thick wet paint already there (not what they
    #    just laid), deeper along the hair lines and under pressure: what stands
    #    above them is shifted, partly onto the brush, partly out to the sides
    #    (ridges), the rest pushed ahead (left as a ridge where the brush lifts)
    ridges = _Ridges()
    hb = max(1.25 * thick, 3.5 * (1.15 - p) * (1.3 - 0.8 * bt.stiffness))
    digging = bt.dig > 0
    if digging or float((top[..., IV] + und[..., IV]).max()) > (1.0 - 0.4 * eff["streaks"]) * hb:
        tv = np.where(canvas.top_id[s.sl] != run.stroke_id, top[..., IV], 0.0)
        if digging:     # a point cut into the paint: a groove through a share of all of it
            depth = bt.dig * _smoothstep(0.05, 0.6, p)
            excess = (tv + und[..., IV]) * depth * (1.0 - np.exp(-6.0 * s.expo_c))
        else:
            level = s.floor + hb * (1.0 + 0.8 * eff["streaks"] * (s.hair[wsl] - 0.5))
            excess = np.clip(s.floor + tv + und[..., IV] - level, 0.0, None)
            excess *= 1.0 - np.exp(-0.8 * bt.stiffness ** 2 * s.expo_c)     # shifted gradually as it passes
            if run.is_dab:      # a light tap only touches with the bristle tips: it doesn't plough
                excess *= _smoothstep(0.4, 0.8, p)
        if excess.max() > 1e-3:
            moved = _take(top, und, excess, tv)
            share_pick = 0.5 * min(eff["pickup"], 1.0)
            brush.tip += s.assign @ (share_pick * moved).reshape(s.npx, PACKED)
            moved *= 1.0 - share_pick
            if s.touchdown:
                ridges.ring += moved.sum(axis=(0, 1))
            else:
                to_side = 1.0 if digging else 0.35 / (1.0 - share_pick)     # a point pushes it all aside
                ridges.add_sides(s, to_side * moved)
                if to_side < 1.0:
                    _place(canvas, (1.0 - to_side) * moved, s.cx0, s.cy0, s.tx * s.ds, s.ty * s.ds, run.stroke_id)
    _squeeze_out(run, s, ld, hold, ridges)
    ridges.lay(run, s)

    # 4) deposit from the tips. Each column's load sets how thick its paint
    #    goes on and when it starts to run dry. Bristles that were just in wet
    #    paint are coated and keep laying a continuous smear for a short
    #    distance, even with little load.
    surf = canvas.surface
    wet_contact = _smoothstep(0.05, 0.35, top[..., IV] + und[..., IV])
    wet_frac = float((s.contact * wet_contact).sum()) / max(float(s.contact.sum()), 1e-12)
    brush.wet = max(wet_frac, brush.wet * np.exp(-s.ds / s.Wd))
    fill = np.clip(np.maximum(s.at_pixels(col_fill), 0.9 * brush.wet), 0.0, 1.0)
    # Dry brush: only the canvas tooth (and hair lines) catch paint. Over
    # wet paint the bristles are in full contact anyway.
    thr = 1.5 * (1.0 - fill) ** 0.8 - 0.35
    # a toothy surface breaks up a lightly pressed stroke even when the brush
    # is loaded (paint only touches the tops of the threads: scumbling)
    rough_t = surf.tooth - 0.5
    thr = thr + (3.2 if rough_t > 0 else 0.6) * rough_t * (1.0 - p) ** 1.2
    tooth = run.tooth_w * canvas._tooth[s.sl] + (1.0 - run.tooth_w) * s.hair[wsl]
    catch = np.maximum(_smoothstep(thr - 0.08, thr + 0.08, tooth), wet_contact)
    q = s.contact * catch
    # When only the tooth catches, paint concentrates on those peaks
    # instead of staying on the brush (so the brush still empties)...
    caught = float(q.sum()) / max(float(s.contact.sum()), 1e-12)
    q /= max(caught, 0.2)
    # ...and once the brush as a whole is effectively dry, the rest stays
    # in the bristles until a real reload (e.g. dragging through wet paint).
    fill_all = max(float(np.clip((ld.load - _RESIDUAL) / 0.6, 0.0, 1.0)), 0.9 * brush.wet)
    thr_all = 1.5 * (1.0 - fill_all) ** 0.8 - 0.35 + (thr - (1.5 * (1.0 - fill) ** 0.8 - 0.35))
    catch_all = np.maximum(_smoothstep(thr_all - 0.08, thr_all + 0.08, tooth), wet_contact)
    caught_all = float((s.contact * catch_all).sum()) / max(float(s.contact.sum()), 1e-12)
    if run.dry:   # a brush that has run dry only comes back to life by dragging through wet
        # paint from earlier strokes (not the trace it is leaving itself)
        other = und[..., IV] + np.where(canvas.top_id[s.sl] != run.stroke_id, top[..., IV], 0.0)
        reload = _smoothstep(0.05, 0.35, other)
        run.dry = float((s.contact * reload).sum()) / max(float(s.contact.sum()), 1e-12) < 0.12
    else:
        run.dry = caught_all < 0.04
    if not run.dry and brush.tip[:, IV].sum() > 0 and bt.deposits:
        d = s.at_pixels(col_thick) * q * s.f_dep      # absorbent / toothy grounds pull paint off the brush
        # thick paint from a glob is combed by the bristles into deep grooves and ridges
        glob_px = s.at_pixels(col_glob) / np.maximum(s.at_pixels(col_thick), 1e-9)
        d *= np.clip(1.0 + glob_px * 2.0 * eff["streaks"] * (2.0 * s.hair[wsl] - 1.0), 0.15, None)
        # Paint laid on thick fills the hollows of the weave (and of dried paint)
        # under it rather than coating them evenly, so it buries the texture.
        thick_px = s.at_pixels(col_thick)
        d *= np.clip(1.0 + _smoothstep(0.3, 2.0, thick_px) * canvas._hollows[s.sl] / np.maximum(thick_px, 1e-6),
                     0.0, None)
        _lay(run, s, d, s.cover[wsl], by_film=True, reach=s.expo_c)
        # the sides of the brush brush past the canvas edge
        off_canvas = thick * float((s.expo * s.streak).sum() - s.contact.sum())
        left = brush.tip[:, IV].sum()
        if off_canvas > 0 and left > 0:
            brush.tip *= max(1.0 - off_canvas / left, 0.0)

    _overflow(brush, capacity)
    # 5) stir: pressure churns the fresh top layer and the wet paint below into each other
    m = 1.0 - np.exp(-eff["mixing"] * 3.0 * p * p * s.contact * s.f_mix)
    _stir(top, und, m.astype(np.float32), canvas.top_a[s.sl], canvas.und_a[s.sl])


def _knife_step(run: _Run, s: _Step) -> None:
    canvas, brush, bt, eff = run.canvas, run.brush, run.bt, run.eff
    p, top, und, wsl, floor = s.p, s.top, s.und, s.wsl, s.floor

    # 1) All the paint on a knife is on the blade, ready to go down; several
    #    paints loaded side by side blend slowly as it works. (A knife doesn't
    #    pick wet paint up as it goes: it drags it, step 5.)
    _work_load(run, s, np.inf)

    # 2) The blade rides above the surface at a gap set by pressure (all its
    #    paint is a film under it; what doesn't fit under it rides along on it).
    ld = _Load(brush, s.Wn)
    gap = _KNIFE_GAP * (1.0 - p) ** 1.6 * (0.55 + 0.45 * bt.stiffness) + 0.05
    if bt.clumps > 0:       # the teeth of a comb dig in, down to the threads
        gap *= 0.03
    hold = gap * run.area_unit / 4.0 + _KNIFE_CARRY

    # 3) Where the blade sits: on the high points of the surface below (thread
    #    tops for a flexible blade, the highest ridges around for a stiff one).
    rest = canvas._peaks_fine[s.sl] + run.stiff_blade * (canvas._peaks_wide[s.sl] - canvas._peaks_fine[s.sl])
    # Wet paint from earlier strokes holds the blade up too, the heap it pushes
    # included (a knife rides on a wave of paint): it rests on its high points
    # and sinks in the harder it is pressed (less into stiff paint). A light
    # touch lays new paint on top of the old.
    earlier = canvas.top_id[s.sl] != run.stroke_id
    old_v = und[..., IV] + np.where(earlier, top[..., IV], 0.0)
    if old_v.max() > 0.05:
        old_body = float((und[..., IB].sum() + np.where(earlier, top[..., IB], 0.0).sum())
                         / max(float(old_v.sum()), 1e-9))
        # How far into it the blade sinks, as a share of its thickness: a light
        # touch skims, a firm one cuts in, pressing hard goes nearly down to
        # the canvas, however thick the paint. Stiff paint resists a little.
        depth = float(np.clip(p ** _KNIFE_SINK * (1.15 - 0.3 * min(old_body, 1.0)), 0.0, 1.0))
        if bt.clumps > 0:       # narrow teeth concentrate the pressure: they cut right through
            depth = max(depth, 0.9)
        # The blade is rigid along the direction it moves: its height varies only
        # along its length. It rests on the high points under it (not on the pile
        # it pushes ahead), bending to follow them over a span set by its stiffness.
        under_blade = (s.aa * np.clip(s.h - np.abs(s.rel) + 0.5, 0.0, 1.0))[wsl] > 0.5
        if under_blade.any():
            span = int(np.clip(0.1 * run.w_typ * (0.5 + run.stiff_blade), 2, 10))
            hi = _max_filter(floor + old_v, 1)
            high = _along_blade(s.lat_px[wsl], under_blade, hi, span)
            base = _along_blade(s.lat_px[wsl], under_blade, rest, span)
            rest = np.maximum(rest, high - depth * np.maximum(high - base, 0.0))
    # The blade rides on the paint under it: as that runs low, it sinks and the
    # slab thins out. Nicks and grit on its edge leave fine scratches along the
    # stroke; it wavers as it is pushed (chatter) and is never pressed quite
    # evenly along its length (tilt).
    film = s.at_pixels(brush.tip[:, IV] + brush.reservoir[:, IV]) * COLUMNS / max(run.area_unit * s.r * s.r, 1e-9)
    ride = np.minimum(gap, _KNIFE_FILM * film) if bt.clumps == 0 else gap
    # (skimming, the blade glides: no chatter; its nicks still streak the paint)
    chat = 0.25 * (1.0 - float(_smoothstep(_KNIFE_SKIM, 0.05, p)))
    level = rest + ride * (1.0 + chat * run.chatter[s.i] + 0.12 * (s.hair[wsl] - 0.5)
                           + run.blade_tilt * np.clip(s.ucoord[wsl], -1.0, 1.0))

    # 4) Wet paint standing above the blade is bulldozed ahead by the flow (5);
    #    near the edges of the mark, some of the paint that was lying there spills
    #    out sideways (lips; the pile the blade carries rides along with it). Nicks
    #    and flex let streaks of it slip under the blade and stay. A comb's teeth
    #    push what they cut into ridges between them.
    ridges = _Ridges()
    flowing = bt.clumps == 0
    tv = top[..., IV]
    excess = np.clip(floor + tv + und[..., IV] - level, 0.0, None)
    excess *= 1.0 - np.exp(-4.0 * s.expo_c * (0.15 + 1.7 * s.hair[wsl] ** 2))
    if flowing:
        half = max(s.r * s.lmax, 1e-6)                    # the mark's half-width, px
        from_edge = half - np.abs(s.lat_px[wsl])          # distance in from the mark's edge, px
        near = _smoothstep(_KNIFE_SPILL_PX, 0.0, from_edge)
        excess = _KNIFE_SPILL * near * np.minimum(excess, run.wet_before[s.sl])
    if excess.max() > 1e-3:
        moved = _take(top, und, excess, tv)
        if flowing:
            ridges.add_sides(s, moved)
        else:
            fresh = (s.between * _smoothstep(-s.ds - 1.0, 0.0, s.along - s.c)
                     * _smoothstep(1.0, -1.0, s.along - s.c - s.h))[wsl]
            _spread(canvas, s.sl, fresh, moved.sum(axis=(0, 1)), run.stroke_id, brush,
                    np.zeros(fresh.shape, np.float32))
    # 5) The blade drags the wet paint it touches, as a thin-film flow: between
    #    the moving blade and the still canvas the paint shears, so paint near
    #    the blade travels almost with it and paint at the canvas hardly moves;
    #    what stands above the blade is bulldozed ahead of it. Each wet layer
    #    moves at its own speed, so their order is kept (white dragged over
    #    purple stays on top). The blade glides on its own fresh paint.
    if flowing:
        _flow_under_blade(canvas, s.sl, run.stroke_id, floor, level, s.aa[wsl], s.along[wsl],
                          (s.c[wsl] if not run.circle else 0.0), s.h[wsl], s.ds, s.tx, s.ty,
                          np.clip((0.5 + 0.45 * p) * (0.6 + 0.8 * s.hair[wsl]), 0.0, 1.0))
    _squeeze_out(run, s, ld, hold, ridges)
    ridges.lay(run, s)

    # 6) The blade fills the space under it, up to its level, from the film of
    #    paint it carries: with plenty of paint, a smooth slab; running low, it
    #    only reaches the high points (broken colour). If that makes three wet
    #    layers, the two most alike merge (white laid on white-over-purple joins
    #    the white; a new colour stays on top of the others).
    #    Barely touching, the blade isn't spreading a film any more: only the
    #    bead of paint on its edge reaches down, and it touches the high points
    #    of the surface (thread tops, ridges of paint) and leaves paint on them,
    #    nothing in the hollows: broken colour ("let it break").
    tip_v = brush.tip[:, IV]
    if tip_v.sum() > 0 and bt.deposits:
        surf = floor + top[..., IV] + und[..., IV]
        gap_px = np.clip(level - surf, 0.0, None)
        film = s.at_pixels(tip_v + brush.reservoir[:, IV]) * COLUMNS / max(run.area_unit * s.r * s.r, 1e-9)
        reach_f = _smoothstep(1.3, 0.7, gap_px / np.maximum(film, 1e-9))
        d = gap_px * reach_f
        skim = float(_smoothstep(_KNIFE_SKIM, 0.05, p))
        if skim > 0 and bt.clumps == 0:
            bead = np.minimum(film, 1.2)
            touch = np.clip((surf - (level - bead)) / np.maximum(bead, 1e-6), 0.0, 1.0)
            # the bead is uneven and the paint below still follows the threads
            # of the canvas: barely touching, it catches in streaks along the
            # blade's nicks and on the thread tops, the fewer the lighter the touch
            grain = 0.55 * s.hair[wsl] + 0.45 * canvas._tooth[s.sl]
            thr = 0.6 - 2.2 * p
            touch = touch * _smoothstep(thr - 0.1, thr + 0.1, grain)
            d = (1.0 - skim) * d + skim * bead * touch ** 0.7
        _lay(run, s, d * (1.0 - np.exp(-4.0 * s.expo_c)), s.cover[wsl], moved=True)
    # A knife lifted off thick paint pulls it up into little peaks and torn ridges.
    if eff["lift"] > 0 and s.i == len(run.pts) - 1:
        under = (s.aa * np.clip((s.h - np.abs(s.rel)) / 1.5, 0.0, 1.0))[wsl]
        _lift_off(canvas, s.sl, under, eff["lift"] * (0.5 + 0.8 * ld.body), max(2.5, 0.08 * s.Wd),
                  np.degrees(np.arctan2(s.ty, s.tx)), run.rng)

    # 7) A blade only shears the paint it cuts through: riding within the top
    #    layer it leaves the one below alone; cutting into it, it folds the two
    #    into each other in streaks (marbling), not evenly.
    m = 1.0 - np.exp(-eff["mixing"] * 3.0 * p * p * s.contact * s.f_mix)
    into_under = np.clip(floor + und[..., IV] + 0.5 - level, 0.0, 1.0)
    m = np.clip(m * (0.1 + 1.8 * s.hair[wsl]) * into_under, 0.0, 1.0)
    _stir(top, und, m.astype(np.float32), canvas.top_a[s.sl], canvas.und_a[s.sl])


# ----------------------------------------------------------------------------
# pieces shared by brushes and knives


def _bilinear(grid, y, x):
    """Sample ``grid`` at fractional indices (y, x) (inside its bounds)."""
    y0, x0 = y.astype(int), x.astype(int)
    fy, fx = y - y0, x - x0
    return ((grid[y0, x0] * (1 - fx) + grid[y0, x0 + 1] * fx) * (1 - fy)
            + (grid[y0 + 1, x0] * (1 - fx) + grid[y0 + 1, x0 + 1] * fx) * fy)


def _along_blade(lat, under, values, span):
    """Average ``values`` over the pixels ``under`` the blade at each position
    along its length (``lat``, px), smoothed over ``span`` px, and spread back
    to every pixel by its position along the blade."""
    b = np.round(lat - lat.min()).astype(int)
    n = int(b.max()) + 1
    cnt = np.bincount(b[under], minlength=n).astype(np.float64)
    tot = np.bincount(b[under], values[under], minlength=n)
    kernel = np.ones(2 * span + 1)
    cnt_s = np.convolve(cnt, kernel, mode="same")
    tot_s = np.convolve(tot, kernel, mode="same")
    prof = np.where(cnt_s > 0, tot_s / np.maximum(cnt_s, 1e-9), 0.0)
    return np.where(cnt_s[b] > 0, prof[b], values)


class _Load:
    """How much paint a tool carries, per column, in load units (width squares)."""

    def __init__(self, brush, Wn):
        # (paint deep in the heel only counts once squeezed down)
        self.col_vol = brush.reservoir[:, IV] + brush.tip[:, IV]
        self.load = float(self.col_vol.sum()) / (Wn * Wn)
        self.col_load = self.col_vol * COLUMNS / (Wn * Wn)
        col_b = brush.reservoir[:, IB] + brush.tip[:, IB]
        self.col_body = np.where(self.col_vol > 0, col_b / np.maximum(self.col_vol, 1e-30), 0.0)
        self.body = float(col_b.sum() / max(self.col_vol.sum(), 1e-30))
        self.Wn = Wn


def _work_load(run, s, capacity):
    """Refill each column's tip from its belly; squeeze the heel when the tool
    is pressed down or harder; the paints on the tool blend tip with belly and
    side to side as it works."""
    brush, p = run.brush, s.p
    _refill(brush, capacity)
    if s.i == 1:      # pressing the brush down squeezes some paint from the heel towards the tip
        _seep(brush, min(_HEEL_SQUEEZE * (0.3 + 2.8 * p * p), 1.0))
    elif run.pressure[s.i] > run.pressure[s.i - 1]:      # ... and so does bearing down harder
        _seep(brush, min(_HEEL_SQUEEZE * 5.0 * (run.pressure[s.i] - run.pressure[s.i - 1]), 1.0))
    blend = run.eff["blending"]
    _swap_layers(brush.tip, brush.reservoir, 0.375 * blend * s.ds / s.Wd)
    _swap_layers(brush.reservoir, brush.heel, 0.1 * blend * s.ds / s.Wd)
    k = _BLEND_RATE * blend * (0.2 + p) ** 2 * s.ds / s.Wd
    for layer in (brush.tip, brush.reservoir, brush.heel):
        _diffuse(layer, k, ring=brush.ring)


def _pick_up(run, s):
    """The tool lifts wet paint. Bristles plough through it: they lift paint
    along the hair lines and leave it between them (brush marks). Thick fresh
    paint shields the older layer below. What is lifted goes into the
    columns that touched it."""
    plough = s.expo_c * np.clip(2.0 - s.streak_c, 0.0, None)
    # (a brush barely touching lifts much less than one pressed into the paint)
    k = run.eff["pickup"] * 1.2 * plough * s.f_pick * (0.25 + 1.5 * min(s.p, 1.0))
    a_top = 1.0 - np.exp(-k)
    a_und = (1.0 - np.exp(-k * s.p)) * np.exp(-s.top[..., IV] / 1.5)
    for layer, a in ((s.top, a_top), (s.und, a_und)):
        lifted = a.astype(np.float32)[..., None] * layer
        run.brush.tip += s.assign @ lifted.reshape(s.npx, PACKED).astype(np.float64)
        layer -= lifted


def _take(top, und, excess, tv):
    """Take volume ``excess`` per pixel off the wet paint, from the top layer
    (of volume ``tv``) first, and return it (packed)."""
    uv = und[..., IV]
    from_top = np.minimum(excess, tv)
    from_und = np.clip(excess - from_top, 0.0, uv)
    ft = (from_top / np.where(tv > 0, tv, 1.0)).astype(np.float32)[..., None]
    fu = (from_und / np.where(uv > 0, uv, 1.0)).astype(np.float32)[..., None]
    moved = top * ft + und * fu
    top -= top * ft
    und -= und * fu
    return moved.astype(np.float64)


def _squeeze_out(run, s, ld, hold, ridges):
    """An overloaded tool pressed down squeezes paint out at its edges,
    unevenly: the lips along a stroke swell and break. Only the paint under a
    knife's blade gets squeezed, not the whole glob riding on it."""
    brush, bt, p = run.brush, run.bt, s.p
    over = np.maximum(ld.col_load - hold, 0.0)
    if bt.rigid:
        over = np.minimum(over, hold + 1.0)
    if not (bt.deposits and over.any()):
        return
    rate = _SQUEEZE[bt.rigid] * (0.2 + p * p) * (0.5 + bt.stiffness) * s.ds / s.Wd * (0.2 + 1.6 * run.lip[s.i])
    if s.touchdown:
        rate += 0.03 * (0.3 + p)
    take = np.minimum(over * (ld.Wn * ld.Wn / COLUMNS) * rate, ld.col_vol)
    frac = np.where(ld.col_vol > 0, take / np.maximum(ld.col_vol, 1e-30), 0.0)[:, None]
    out = (brush.tip + brush.reservoir) * frac
    brush.tip -= brush.tip * frac
    brush.reservoir -= brush.reservoir * frac
    if s.touchdown:
        ridges.ring += out.sum(axis=0)
    else:
        w_right = _smoothstep(-0.3, 0.3, run.col_u * s.side)[:, None]
        ridges.right += (out * w_right).sum(axis=0)
        ridges.left += (out * (1.0 - w_right)).sum(axis=0)


class _Ridges:
    """Paint pushed out of a footprint during one step: all around it (when the
    tool is set down) or out at its left and right edges. ``lay`` puts it on
    the canvas as low ridges just outside the edges."""

    def __init__(self):
        self.ring = np.zeros(PACKED)
        self.left = np.zeros(PACKED)
        self.right = np.zeros(PACKED)

    def add_sides(self, s, moved):
        right = (s.lat_px[s.wsl] > 0)[..., None]
        self.right += (moved * right).sum(axis=(0, 1))
        self.left += (moved * ~right).sum(axis=(0, 1))

    def lay(self, run, s):
        if not (self.ring[IV] > 0 or self.left[IV] > 0 or self.right[IV] > 0):
            return
        canvas, circle, margin = run.canvas, run.circle, run.margin
        H, W = canvas.height, canvas.width
        # the ridges lie just outside the footprint: work on a slightly larger window
        m_px = int(np.ceil(margin))
        ex0, ex1, ey0, ey1 = s.x0 - m_px, s.x1 + m_px, s.y0 - m_px, s.y1 + m_px
        esl = (slice(max(ey0, 0), min(ey1, H)), slice(max(ex0, 0), min(ex1, W)))
        ewsl = (slice(esl[0].start - ey0, esl[0].stop - ey0), slice(esl[1].start - ex0, esl[1].stop - ex0))
        egx = (np.arange(ex0, ex1) + 0.5 - s.cx)[None, :]
        egy = (np.arange(ey0, ey1) + 0.5 - s.cy)[:, None]
        e_along = egx * s.tx + egy * s.ty
        e_lat = -egx * s.ty + egy * s.tx
        e_r = s.r1 + (s.r0 - s.r1) * np.clip(-e_along / s.ds, 0.0, 1.0) if s.r0 != s.r1 else s.r
        e_side = np.abs(e_lat) - e_r * s.lmax

        def tent(e, w):         # cross-section of a squeezed-out ridge of width w, just outside the edge
            return np.clip(1.0 - np.abs(e - 0.35 * w) / (0.65 * w + 0.5), 0.0, 1.0)

        def geo(e, w):          # how much of each pixel the ridge covers (anti-aliased outer edge)
            return np.clip(w - e, 0.0, 1.0) * (e > -0.3 * w)

        if self.ring[IV] > 0:
            # all around the footprint: as wide as its narrow side allows
            w = float(np.clip(0.24 * s.r * min(s.lmax, 1.0 if circle else float(s.h_tab.max())), 0.6,
                              margin - 1.0))
            if circle:
                e_ring = np.sqrt(e_lat * e_lat + e_along * e_along) - e_r
            else:
                el = e_lat / e_r
                e_c = np.interp(el, s.grid, s.c_tab) * e_r
                e_h = np.interp(el, s.grid, s.h_tab, left=0.0, right=0.0) * e_r
                e_ring = np.maximum(e_side, np.abs(e_along - e_c) - e_h)
            _spread(canvas, esl, tent(e_ring, w)[ewsl], self.ring, run.stroke_id, run.brush, geo(e_ring, w)[ewsl],
                    by_film=not run.bt.rigid)
        if self.left[IV] > 0 or self.right[IV] > 0:
            w = float(np.clip(0.24 * s.r * s.lmax, 0.6, margin - 1.0))   # scales with the mark's width
            ce = 0.0 if circle else np.where(e_lat > 0, s.c_tab[-1], s.c_tab[0]) * e_r
            rel_e = e_along - ce
            # soft-ended, so that the ridges laid at successive steps run together smoothly
            ramp = 0.5 * s.ds + 0.7
            box = _smoothstep(-s.ds - ramp, -s.ds + ramp, rel_e) * _smoothstep(ramp, -ramp, rel_e)
            band, g = tent(e_side, w) * box, geo(e_side, w) * box
            for pk, mask in ((self.left, e_lat < 0), (self.right, e_lat > 0)):
                _spread(canvas, esl, (band * mask)[ewsl], pk, run.stroke_id, run.brush, (g * mask)[ewsl],
                        by_film=not run.bt.rigid)


def _lay(run, s, d, cover_px, moved=False, by_film=False, reach=None):
    """Lay volume ``d`` per pixel from the tips of the columns above it;
    each column can only give what it holds. ``by_film``, ``reach``: see :func:`_deposit`."""
    brush = run.brush
    tip_v = brush.tip[:, IV]
    demand = s.assign @ d.ravel()
    give = np.where(demand > tip_v, tip_v / np.maximum(demand, 1e-30), 1.0)
    comp = brush.tip / np.where(tip_v > 0, tip_v, 1.0)[:, None]     # paint per unit volume
    parts = [(idx, d * wt * give[idx]) for idx, wt in s.taps]       # volume from each column
    d_px = sum(v for _, v in parts)
    if d_px.sum() > 0:
        dep = sum(v[..., None] * comp[idx] for idx, v in parts)
        _deposit(run.canvas, s.sl, dep, d_px, run.stroke_id, cover_px, moved=moved, by_film=by_film, reach=reach)
        used = sum(np.bincount(idx.ravel(), v.ravel(), COLUMNS) for idx, v in parts)
        brush.tip -= used[:, None] * comp
        np.maximum(brush.tip[:, IV], 0.0, out=brush.tip[:, IV])


def _overflow(brush, capacity):
    """What the tips hold beyond their capacity goes back into the belly
    (this is how the brush gets dirty)."""
    cap_col = capacity / COLUMNS
    over = brush.tip[:, IV] - cap_col
    if (over > 0).any():
        frac = np.where(over > 0, over / np.maximum(brush.tip[:, IV], 1e-30), 0.0)[:, None]
        moved = brush.tip * frac
        brush.tip -= moved
        brush.reservoir += moved


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


def _deposit(canvas, sl, dep, d, stroke_id, cover, moved=False, by_film=False, reach=None):
    # ``moved`` may also be a per-pixel mask
    """Lay paint ``dep`` (packed, per pixel; volume ``d``) on top of the canvas.

    Each pixel holds two wet layers. New paint goes on top and the old top
    joins the layer below. Paint that is only being shoved around (``moved``:
    pushed ahead of a knife, squeezed into a ridge) instead merges with
    whichever layer it is most like, so that, say, white pushed over white
    stays a white skin over the purple beneath it.

    With ``by_film``, how much of the old top goes under depends on how well
    the new paint covers it (:func:`_buries`): a covering film buries it, a
    faint trace (the feathered edge of a soft brush) only stirs into it.
    Buried, it would mix with the layer below and show through the faint new
    film: white over wet blue would turn blue along every soft edge. Where the
    body of the brush passes over (``reach``: this step's share of a full pass
    at full strength), even a thin film goes on top: a light touch lays paint
    over wet paint. (Where the two old layers are much the same paint it goes
    under anyway.) During a brushstroke this waits until the stroke has passed
    (see :class:`_Pending`): a pixel first grazed by the edge of a looping
    stroke may later get its body."""
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
            # three layers in two places: merge the two most alike. The new paint
            # joins the top layer only if it really is much the same paint (white
            # pushed onto white); a different colour stays on top of the others.
            keep = np.zeros_like(old)
            near = _alike(top[three], dep[three])
            keep[three] = (near < _SAME_PAINT) & (near < _alike(und[three], top[three]))
            top[keep] += dep[keep].astype(np.float32)
            dep = np.where(keep[..., None], 0.0, dep)
            d = np.where(keep, 0.0, d)
            touched = d > 1e-7
            old = touched & (ids != stroke_id)
    pending = getattr(canvas, "_pending", None)
    if by_film and not np.any(moved) and pending is not None and old.any():
        dep, d, cover, touched, old = pending.hold(sl, dep, d, cover, old, reach)
    if old.any():
        if not by_film:
            und[old] += top[old]
            und_a[old] = np.maximum(und_a[old], top_a[old])
            top[old] = 0
            top_a[old] = 0
        else:
            k = _buries(d[old])
            if reach is not None:
                k = np.maximum(k, _smoothstep(*_BODY, np.broadcast_to(reach, d.shape)[old]))
            k = np.where(_harmless(top[old], und[old]), 1.0, k).astype(np.float32)
            going = k[:, None] * top[old]
            und[old] += going
            und_a[old] = np.where(k > 0.01, np.maximum(und_a[old], top_a[old]), und_a[old])
            top[old] -= going
            top_a[old] *= 1.0 - k
        ids[old] = stroke_id
    # The tool covers what it passes over with this stroke's paint, also where it
    # adds nothing new in this step (it already laid paint there).
    covered = touched | ((ids == stroke_id) & (top[..., IV] > 1e-6))
    top_a += np.where(covered, cover, 0.0).astype(np.float32)
    np.minimum(top_a, 1.0, out=top_a)
    top += dep.astype(np.float32)


def _buries(film):
    """How much of the wet top layer a new film of paint this thick buries (see
    :func:`_deposit`): a covering film all of it, a faint trace hardly any."""
    return 1.0 - np.exp(-np.asarray(film) / _COVERING_FILM)


def _harmless(top, und):
    """Where burying the top layer changes nothing you could see: the layer
    below is much the same paint (or there is none)."""
    return (und[..., IV] < 1e-4) | (_alike(np.maximum(top, 0.0), np.maximum(und, 0.0)) < _SAME_LAYER)


class _Pending:
    """New paint from the brushstroke in progress, held back where it lands on
    wet paint from earlier strokes until we know how well the stroke covers it.

    Once the body of the brush has passed over a pixel, or the stroke has
    laid a covering film on it, the old top layer goes under and the held
    paint becomes the new top. Pixels only ever grazed by the brush's
    feathered edge are settled when the stroke ends: the old top keeps its
    place, the share the faint new film would bury (:func:`_buries`) goes
    under, and the new paint stirs into the rest."""

    def __init__(self, canvas, stroke_id, y0, y1, x0, x1):
        self.canvas, self.stroke_id = canvas, stroke_id
        self.y0, self.y1, self.x0, self.x1 = y0, y1, x0, x1
        h, w = y1 - y0, x1 - x0
        self.paint = np.zeros((h, w, PACKED), np.float32)
        self.cover = np.zeros((h, w), np.float32)
        self.reach = np.zeros((h, w), np.float32)
        self.on = np.zeros((h, w), bool)

    def hold(self, sl, dep, d, cover, old, reach=None):
        """Take this step's paint at pixels with an earlier top layer. Returns
        (dep, d, cover, touched, old) for what is left to lay down now."""
        canvas = self.canvas
        ys, xs = sl
        # the part of the window inside the held area
        wy0, wy1 = max(ys.start, self.y0), min(ys.stop, self.y1)
        wx0, wx1 = max(xs.start, self.x0), min(xs.stop, self.x1)
        if wy0 >= wy1 or wx0 >= wx1:
            return dep, d, cover, d > 1e-7, old
        win = (slice(wy0 - ys.start, wy1 - ys.start), slice(wx0 - xs.start, wx1 - xs.start))
        mine = (slice(wy0 - self.y0, wy1 - self.y0), slice(wx0 - self.x0, wx1 - self.x0))
        top = canvas.top[sl][win]
        # an earlier top layer worth keeping out of the layer below
        h = old[win] & (top[..., IV] > 1e-4)
        if h.any():
            h[h] = ~_harmless(top[h], canvas.und[sl][win][h])
        if not h.any():
            return dep, d, cover, d > 1e-7, old
        here = np.zeros_like(old)
        here[win] = h
        cover = np.broadcast_to(np.asarray(cover, np.float32), d.shape)
        paint, pcov, on = self.paint[mine], self.cover[mine], self.on[mine]
        paint[h] += dep[win][h].astype(np.float32)
        pcov[h] += cover[win][h]
        rch = self.reach[mine]
        if reach is not None:
            rch[h] += np.broadcast_to(reach, d.shape)[win][h]
        on[h] = True
        # the body of the brush has been over these, or it laid a covering film: bury the old top now
        done = h & (self.buried(paint[..., IV], rch) >= _BURIED)
        if done.any():
            und, top_a = canvas.und[sl][win], canvas.top_a[sl][win]
            und_a, ids = canvas.und_a[sl][win], canvas.top_id[sl][win]
            und[done] += top[done]
            und_a[done] = np.maximum(und_a[done], top_a[done])
            top[done] = paint[done]
            top_a[done] = np.minimum(pcov[done], 1.0)
            ids[done] = self.stroke_id
            paint[done] = 0.0
            pcov[done] = 0.0
            rch[done] = 0.0
            on[done] = False
        # nothing more to lay at the held pixels in this step
        dep = np.where(here[..., None], 0.0, dep)
        d = np.where(here, 0.0, d)
        return dep, d, cover, d > 1e-7, old & ~here

    @staticmethod
    def buried(film, reach):
        """How much of the old top layer goes under: all of it where the body of
        the brush went over it or it laid a covering film, less elsewhere."""
        return np.maximum(_buries(film), _smoothstep(*_BODY, reach))

    def settle(self):
        """The stroke has passed: settle the pixels its edge only grazed."""
        on = self.on
        if not on.any():
            return
        canvas = self.canvas
        sl = (slice(self.y0, self.y1), slice(self.x0, self.x1))
        top, und = canvas.top[sl], canvas.und[sl]
        top_a, und_a, ids = canvas.top_a[sl], canvas.und_a[sl], canvas.top_id[sl]
        k = self.buried(self.paint[on][:, IV], self.reach[on])
        k = np.where(_harmless(top[on], und[on]), 1.0, k).astype(np.float32)
        going = k[:, None] * top[on]
        und[on] += going
        und_a[on] = np.where(k > 0.01, np.maximum(und_a[on], top_a[on]), und_a[on])
        top[on] += self.paint[on] - going
        top_a[on] = np.maximum(top_a[on] * (1.0 - k), np.minimum(self.cover[on], 1.0))
        ids[on] = self.stroke_id


def _spread(canvas, sl, w, pk, stroke_id, brush, cover, by_film=False):
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
    _deposit(canvas, sl, share[..., None] * pk[None, None, :], d, stroke_id, cover, moved=True, by_film=by_film)


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
        und_a, top_a = canvas.und_a[sl], canvas.top_a[sl]
        und_a[same] = np.maximum(und_a[same], top_a[same])
        top_a[same] = 0.0
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


def _place(canvas, pk, x0, y0, sx, sy, stroke_id):
    """Put the packed paint ``pk`` (a window whose top-left pixel is (x0, y0))
    back on the canvas, moved by (sx, sy) pixels: paint pushed ahead of the tool."""
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
    # moved paint doesn't change how much of each pixel is covered
    _deposit(canvas, (slice(ry0, ry1), slice(rx0, rx1)), moved, d, stroke_id, np.zeros_like(d), moved=True,
             by_film=True)


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
    n = aniso_noise(hh, ww, 6.0 * cell, cell, direction, rng, octaves=2)
    n = (n - n.min()) / max(np.ptp(n), 1e-9)
    crest = (1.0 - np.abs(2.0 * n - 1.0)) ** 2
    m = (wv * crest).sum() / wv.sum()
    fac = np.clip(1.0 + 2.0 * amp * under * (crest - m), 0.05, None)
    fac *= V.sum() / max((V * fac).sum(), 1e-12)
    top *= fac.astype(np.float32)[..., None]


def _fix_coverage(canvas, region):
    """Each pixel records how much of it its wet paint covers (anti-aliased
    edges): a film of thickness volume / coverage over that part. Paint pushed
    into a pixel (dragged, slumped, squeezed out) doesn't update that record,
    so make it consistent. A pixel with paint all around it isn't at an edge:
    it is fully covered. Elsewhere, no pixel holds a film much thicker than
    the mostly-covered pixels around it; where it would, the paint covers
    more of it."""
    total = canvas.top_v[region] + canvas.und_v[region]
    inside = -_max_filter(-total, 1) > _MIN_FILM     # every neighbour carries a film of paint
    for V, A in ((canvas.top_v[region], canvas.top_a[region]), (canvas.und_v[region], canvas.und_a[region])):
        reliable = np.where(A >= 0.5, V / np.maximum(A, 1e-6), 0.0)     # film thickness where it is well defined
        nb = _max_filter(reliable, 2)
        need = np.where((A > 0) & (nb > 0), np.minimum(V / (1.25 * np.maximum(nb, 1e-6)), 1.0), 0.0)
        need = np.where((A > 0) & inside & (V > 0.5 * _MIN_FILM), 1.0, need)
        np.maximum(A, need.astype(A.dtype), out=A)


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
