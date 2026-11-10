"""Brushes and knives, and the paint on them.

A :class:`BrushType` is a tool: the shape of its footprint (a round, flat,
filbert or fan brush, or the steel blade of a knife), its size and the
character of its hair, from which its behaviour in a stroke follows
(:meth:`BrushType.physics`). A :class:`Brush` is one such tool with paint on
it, which it keeps (with any dirt) between strokes. A :class:`Stroke` is one
pass of a tool along a path; :func:`apply_stroke` paints it (both in
:mod:`oilpaint.stroke`, and found here too).

Paint on a brush is held in narrow bristle *columns* (:data:`COLUMNS`),
each with three depths: the tip that touches the canvas, the belly, and the
heel near the ferrule. On flat, filbert and fan brushes the columns run
across the width; on a round they go around the bundle, so rolling the
brush between the fingers turns different bristles to the canvas. That lets a
brush carry several paints side by side (double loading), on its tip or deep
in its heel, or only partly mixed; the paints blend into each other as the
brush is worked, at a rate set by :attr:`BrushType.blending`.

Paint quantities are stored *packed*: a vector of the absorption spectrum K
(N_BANDS values), then scattering S, volume V and body (the paint's
stiffness times V), all extensive (summed over the paint present). Mixing
paint is then just adding vectors.
"""

from __future__ import annotations

import hashlib
import importlib
import math
import numbers
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from itertools import starmap
from typing import TYPE_CHECKING, Any, ClassVar, cast

import numpy as np

from . import _engine as eng
from . import spectral as sp
from ._engine import COLUMNS, IB, IS, IV, PACKED
from .noise import value_noise_1d
from .paint import ColorLike, Paint, as_paint

if TYPE_CHECKING:  # (at run time, see __getattr__ at the end)
    from ._geometry import _chord_table as _chord_table
    from ._geometry import _chord_table_at as _chord_table_at
    from ._hair import _hair_field as _hair_field
    from .stroke import Stroke as Stroke
    from .stroke import _brush_step as _brush_step
    from .stroke import _pending as _pending
    from .stroke import _resolve as _resolve
    from .stroke import _Run as _Run
    from .stroke import apply_stroke as apply_stroke

N = sp.N_BANDS

# Paint volume (in brush-width squares of unit-thickness paint) per unit of
# `amount`, calibrated so that `amount` ~ brush widths of solid stroke.
_VOLUME_PER_AMOUNT = 0.9
# column centres, 0 = left edge, 1 = right edge
_COL_X = (np.arange(COLUMNS) + 0.5) / COLUMNS
# On a round brush the columns go around the bundle: column b sits at angle
# _RING_A[b] (0 = facing the direction of travel, pi/2 = the brush's left side).
# Pressed down, the bundle splays its bristles evenly across the mark, the
# front and back of the bundle overlapping: a bristle at angle a lands at
# u = -(2/pi) asin(sin a) across the mark (-1 = left edge, +1 = right edge).
_RING_A = 2.0 * np.pi * (np.arange(COLUMNS) + 0.5) / COLUMNS
# where across the mark each one lands
_RING_X = (1.0 - (2.0 / np.pi) * np.arcsin(np.sin(_RING_A))) / 2.0
# Depth layers: tip [0, 0.12], belly [0.12, 0.6], heel [0.6, 1] of the bristle length.
_LAYER_EDGES = (0.0, 0.12, 0.6, 1.0)


def _pack(paint: Paint, volume: float) -> np.ndarray:
    v = np.empty(PACKED)
    v[:N] = paint.K * volume
    v[IS] = paint.S * volume
    v[IV] = volume
    v[IB] = paint.body * volume
    return v


# --- checking what callers pass ------------------------------------------------
# A NaN or an infinity handed to the engine spreads through the wet paint (and
# through blurs and shadows in a render) and stays on the brush for good, so
# numbers are checked where they come in, before anything is painted.


def _is_number(value) -> bool:
    """A real number: a Python or numpy int or float, or a 0-d numeric array."""
    if isinstance(value, np.ndarray):
        return value.ndim == 0 and value.dtype.kind in "biuf"
    return isinstance(value, numbers.Real)


def _number(name: str, value) -> float:
    """``value`` as a float: a TypeError unless it is a real number, a
    ValueError unless it is finite (both naming ``name``)."""
    if not _is_number(value):
        raise TypeError(f"{name} must be a number, got {value!r:.80}")
    v = float(value)
    if not math.isfinite(v):
        raise ValueError(f"{name} must be a finite number, got {v}")
    return v


def _width(name: str, value) -> float:
    """A width in pixels: a finite number above 0."""
    v = _number(name, value)
    if v <= 0:
        raise ValueError(f"{name} must be above 0 (pixels), got {v}")
    return v


def _spread(brush: Brush, across) -> np.ndarray:
    """The share of a load each column of ``brush`` gets (see
    :func:`_across_weights`), which must be finite numbers."""
    w = brush._spread_load(across)
    if not np.isfinite(w).all():
        raise ValueError(f"across must give finite weights, got {across!r:.80}")
    return w


def _check_taper(taper, name: str = "taper") -> None:
    """A taper: a share of the stroke from 0 to 1, or a (start, end) pair."""
    if _is_number(taper):
        ends = (taper,)
    elif isinstance(taper, (tuple, list, np.ndarray)) and len(taper) == 2:
        ends = tuple(taper)
    else:
        ends = ()
    if not ends or not all(map(_is_number, ends)):
        raise TypeError(
            f"{name} must be a number from 0 to 1 or a (start, end) pair of "
            f"them, got {taper!r:.80}"
        )
    for t in ends:
        if not 0.0 <= float(t) <= 1.0:  # (False for NaN)
            raise ValueError(
                f"{name} must be from 0 to 1 (a share of the stroke's length), "
                f"got {taper!r:.80}"
            )


_SEEDS = (
    "None, a whole number >= 0, a sequence of them, or a numpy SeedSequence, "
    "BitGenerator or Generator"
)


def _check_seed(name: str, seed) -> None:
    """A random seed: anything ``np.random.default_rng`` takes."""
    rng = np.random
    if seed is None or isinstance(
        seed, (rng.SeedSequence, rng.BitGenerator, rng.Generator)
    ):
        return
    try:
        rng.SeedSequence(seed)
    except ValueError:
        raise ValueError(f"{name} must be {_SEEDS}, got {seed!r:.80}") from None
    except TypeError:
        raise TypeError(f"{name} must be {_SEEDS}, got {seed!r:.80}") from None


def _check_tool(tool) -> None:
    """What a stroke may be painted with."""
    if tool is not None and not isinstance(tool, (BrushType, Brush)):
        raise TypeError(
            "brush must be a BrushType (such as brushes.FLAT_HOG), a Brush or "
            f"None (a medium round), got {tool!r:.80}"
        )


#: Blade outlines of palette and painting knives (see :class:`BrushType`).
KNIFE_SHAPES = ("trowel", "pear", "straight", "angled", "spoon")
SHAPES = ("round", "flat", "filbert", "fan") + KNIFE_SHAPES
#: What a tool's face looks like where it is pressed straight down (a dab).


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
    tips:
        How many separate tips (clumps of bristle ends) a dab shows when the
        brush is tapped lightly: a light tap touches only the longest ones,
        sparse dots; harder taps bring in more until the mark closes up. 0
        works it out from the size and the hair (fine hair has more, smaller
        tips). Each brush keeps its own layout, turned a little on every tap.
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
    blending: float | None = None
    taper: float = 0.0
    tips: int = 0
    streaks: float | None = None
    pickup: float | None = None
    drag: float | None = None
    mixing: float | None = None
    rough: float | None = None
    name: str = "brush"

    # (the fields that must be finite numbers; the optional ones may be None)
    _NUMBERS: ClassVar[tuple[str, ...]] = (
        "width",
        "aspect",
        "angle",
        "stiffness",
        "bristles",
        "capacity",
        "spread",
        "clumps",
        "slant",
        "softness",
        "dig",
        "blending",
        "tips",
        "streaks",
        "pickup",
        "drag",
        "mixing",
        "rough",
    )
    _OPTIONAL: ClassVar[tuple[str, ...]] = (
        "blending",
        "streaks",
        "pickup",
        "drag",
        "mixing",
        "rough",
    )

    def __post_init__(self):
        if self.shape not in SHAPES:
            raise ValueError(f"unknown brush shape {self.shape!r}; use one of {SHAPES}")
        for name in self._NUMBERS:
            value = getattr(self, name)
            if value is not None or name not in self._OPTIONAL:
                _number(f"BrushType {name}", value)
        if self.width <= 0 or self.aspect <= 0:
            raise ValueError(
                f"width and aspect must be positive, got width {self.width} and "
                f"aspect {self.aspect}"
            )
        if self.tips < 0:
            raise ValueError("tips must be 0 (automatic) or more")
        if self.clumps < 0:
            raise ValueError(
                "clumps (the teeth of a comb or serrated knife) must be 0 "
                f"(continuous) or more, got {self.clumps}"
            )
        _check_taper(self.taper, "BrushType taper")

    def with_(self, **changes) -> BrushType:
        """A copy with some properties changed."""
        return replace(self, **changes)

    def sized(self, width: float) -> BrushType:
        """The same brush in another size."""
        return replace(self, width=float(width))

    @property
    def rigid(self) -> bool:
        """True for knives: a rigid blade rather than bristles."""
        return self.shape in KNIFE_SHAPES

    def physics(self) -> dict:
        """Stroke behaviour derived from the brush's character (and its own
        overrides), each clipped to the range the engine works in."""
        return _physics(self)


DEFAULT_BRUSH = BrushType(name="round")


def _depth_weights(depth) -> np.ndarray:
    """Share of a load that goes to the tip, the belly and the heel."""
    if isinstance(depth, str):
        named = {
            "tip": (1.0, 0.0, 0.0),
            "belly": (0.0, 1.0, 0.0),
            "heel": (0.0, 0.0, 1.0),
        }
        if depth not in named:
            raise ValueError(
                "depth must be 'tip', 'belly', 'heel' or a dip depth from 0 to 1"
            )
        return np.array(named[depth])
    d = float(depth)
    if not 0.0 < d <= 1.0:
        raise ValueError(
            "a dip depth must be between 0 (not dipped) and 1 (up to the ferrule)"
        )
    e = _LAYER_EDGES
    w = np.array([max(0.0, min(d, e[k + 1]) - e[k]) for k in range(3)])
    return w / w.sum()


def _across_weights(across: Any, ring: bool = False) -> np.ndarray:
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
        named = {
            "left": (0.0 - 1, 0.5),
            "right": (0.5, 1.0 + 1),
            "center": (0.3, 0.7),
            "middle": (0.3, 0.7),
        }
        if across == "edges":
            w = band(-1, 0.25) + band(0.75, 2)
        elif across in named:
            w = band(*named[across])
        else:
            raise ValueError(
                f"unknown side {across!r}; use left, right, center, edges or all"
            )
    elif callable(across):
        fn = cast(Callable[[float], float], across)
        w = np.asarray([float(fn(v)) for v in x])
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
            raise ValueError(
                f"across: give (start, end) in 0..1 or {COLUMNS} column weights"
            )
    w = _clip(w, 0.0, None)
    if w.sum() <= 0:
        raise ValueError("that part of the brush is empty: no column gets any paint")
    return w / w.sum()


class Brush:
    """A brush with paint on it, which it keeps (with any dirt) between strokes.

    Paint sits in bristle columns across the brush, so a brush can carry
    several paints at once::

        brush = Brush(brushes.FLAT_SABLE)
        brush.load(P.CADMIUM_RED, amount=6)                    # the whole brush
        brush.load(P.TITANIUM_WHITE, amount=2, across="left")  # one side
        brush.load(P.CADMIUM_YELLOW, amount=0.5, depth="tip")  # just the tips
        brush.load(P.BURNT_UMBER, amount=3, depth=1.0)         # up to the ferrule
        brush.work(1)                                          # worked on the palette
        canvas.stroke(path, brush=brush)
        canvas.stroke(path2, brush=brush)                      # with what's left
        brush.clean()

    Paint in the heel doesn't flow to the tip by itself: some of it is
    squeezed down when the brush touches the canvas (more when pressed hard)
    and when the pressure increases during a stroke. The rest stays in the
    brush, tinting later loads, until it is cleaned.
    """

    def __init__(
        self,
        type: BrushType | None = None,  # ruff: ignore[builtin-argument-shadowing]
    ):
        if type is not None and not isinstance(type, BrushType):
            raise TypeError(
                "Brush takes a BrushType (such as brushes.FLAT_HOG) or None (a "
                f"medium round), got {type!r:.80}"
            )
        self.type = type or DEFAULT_BRUSH
        self.reservoir = np.zeros((COLUMNS, PACKED))  # the belly of each column
        # paint at the tips of each column
        self.tip = np.zeros((COLUMNS, PACKED))
        # paint deep in the heel, by the ferrule
        self.heel = np.zeros((COLUMNS, PACKED))
        self.width = float(self.type.width)
        self.wet = 0.0  # how coated the bristles are from recent wet contact

    # ------------------------------------------------------------------ loading
    def load(
        self,
        paint: ColorLike,
        amount: float,
        across=None,
        depth: str | float = "belly",
        *,
        width: float | None = None,
    ) -> Brush:
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

        ``width`` (keyword only) is the stroke width, in pixels, the amount
        is measured for (see :attr:`Stroke.amount`; it is kept in
        :attr:`width` and replaced by the next stroke's mean width). Default:
        the brush's :attr:`width`.

        Raises ValueError or TypeError, and loads nothing, unless ``amount``
        is a finite number (at most 0 loads nothing) and ``width`` a finite
        number above 0.
        """
        if _is_number(across):  # (it used to be the third argument)
            raise TypeError(
                "Brush.load takes width by keyword only, as in load(paint, "
                f"amount, width={float(cast(float, across)):g}); its third "
                "argument is across (where on the brush the paint goes)"
            )
        amount = _number("amount", amount)
        size = self.width if width is None else _width("width", width)
        volume = max(amount, 0.0) * _VOLUME_PER_AMOUNT * size**2
        paint = as_paint(paint)
        spread = _spread(self, across)
        weights = _depth_weights(depth)
        self.width = size
        if volume <= 0:
            return self
        cols = spread[:, None] * _pack(paint, volume)[None, :]
        for layer, w in zip(self._layers(), weights, strict=True):
            if w > 0:
                layer += w * cols
        return self

    def load_mix(
        self,
        parts,
        amount: float,
        mixing: float = 0.5,
        across=None,
        depth: str | float = "belly",
        seed: int | None = None,
        width: float | None = None,
    ) -> Brush:
        """Pick up several paints at once, only partly mixed on the palette.

        ``parts`` is a list of ``(paint, proportion)``. ``mixing`` = 1 gives a
        uniform mixture (like :meth:`Paint.mix`); lower values leave each
        paint in irregular streaks across the bristles, so the stroke comes
        out marbled; 0 = barely touched together. ``amount``, ``across``,
        ``depth`` and ``width`` are as in :meth:`load`.

        ``seed`` lays out the streaks (anything ``np.random.default_rng``
        takes). Without one they follow from the arguments and from what is
        on the brush already, so a painting made again on the same machine
        and numpy build comes out the same; pass seeds to vary otherwise
        identical loads.
        """
        amount = _number("amount", amount)
        mixing = float(_clip(_number("mixing", mixing), 0.0, 1.0))
        _check_seed("seed", seed)
        size = self.width if width is None else _width("width", width)
        if isinstance(parts, (str, Paint)) or not isinstance(parts, Sequence):
            raise TypeError(
                f"load_mix takes a list of (paint, proportion) pairs, got {parts!r:.80}"
            )
        for i, part in enumerate(parts):
            if not isinstance(part, (tuple, list)) or len(part) != 2:
                raise TypeError(
                    "load_mix takes a list of (paint, proportion) pairs; part "
                    f"{i} is {part!r:.80}"
                )
        paints = [as_paint(p) for p, _ in parts]
        props = np.array(
            [
                max(_number(f"proportion of part {i}", w), 0.0)
                for i, (_, w) in enumerate(parts)
            ]
        )
        if props.sum() <= 0:
            raise ValueError("load_mix needs positive proportions")
        spread = _spread(self, across)
        weights = _depth_weights(depth)
        self.width = size
        props = props / props.sum()
        if seed is None:
            seed = self._mix_seed(paints, props, amount, mixing, spread, weights)
        rng = np.random.default_rng(seed)
        # less mixed: more, narrower streaks
        cell = COLUMNS / (2.0 + 5.0 * (1.0 - mixing))
        noise = np.stack([value_noise_1d(COLUMNS, cell, rng) for _ in paints])
        logits = np.log(props)[:, None] + 12.0 * (1.0 - mixing) * (noise - 0.5)
        w = np.exp(logits - logits.max(axis=0, keepdims=True))
        w /= w.sum(axis=0, keepdims=True)
        # keep each paint's overall share while keeping the streaks
        for _ in range(20):
            w *= (props * COLUMNS / np.maximum(w.sum(axis=1), 1e-12))[:, None]
            w /= w.sum(axis=0, keepdims=True)
        w = mixing * props[:, None] + (1.0 - mixing) * w
        volume = max(amount, 0.0) * _VOLUME_PER_AMOUNT * self.width**2
        col_vol = volume * spread
        for layer, dw in zip(self._layers(), weights, strict=True):
            if dw <= 0:
                continue
            for i, paint in enumerate(paints):
                layer += (dw * w[i] * col_vol)[:, None] * _pack(paint, 1.0)[None, :]
        return self

    def _mix_seed(self, paints, props, amount, mixing, spread, weights) -> int:
        """A seed for the streaks of :meth:`load_mix` made from what it loads
        and what the brush holds: the same calls give the same brush."""
        h = hashlib.sha256()
        for paint in paints:
            h.update(_pack(paint, 1.0).tobytes())
        h.update(np.array([*props, amount, mixing, self.width]).tobytes())
        for a in (spread, weights, *self._layers()):
            h.update(np.ascontiguousarray(a, dtype=np.float64).tobytes())
        return int.from_bytes(h.digest()[:8], "little")

    def work(self, passes: float = 1.0, pressure: float = 0.5) -> Brush:
        """Stroke the brush back and forth on the palette without painting:
        the paints on it blend into each other (side to side, and tip with
        belly) as they would over a few widths of real strokes."""
        passes, pressure = _number("passes", passes), _number("pressure", pressure)
        blend = self.type.physics()["blending"]
        widths = 3.0 * max(passes, 0.0)
        k = float(eng.BLEND_RATE * blend * (0.2 + pressure) ** 2 * widths)
        for layer in self._layers():
            eng.diffuse(layer, k, self.ring)
        for _ in range(int(np.ceil(widths))):
            eng.swap_layers(self.tip, self.reservoir, float(min(0.375 * blend, 0.5)))
            eng.swap_layers(self.reservoir, self.heel, float(min(0.1 * blend, 0.5)))
        return self

    def clean(self) -> Brush:
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
        return float(
            self.reservoir[:, IV].sum() + self.tip[:, IV].sum() + self.heel[:, IV].sum()
        )

    @property
    def amount(self) -> float:
        """Paint left on the brush, in the units of :attr:`Stroke.amount`."""
        return self.volume / (_VOLUME_PER_AMOUNT * self.width**2)

    @property
    def paint(self) -> Paint | None:
        """The overall mixture of paint on the brush."""
        total = (
            self.reservoir.sum(axis=0) + self.tip.sum(axis=0) + self.heel.sum(axis=0)
        )
        if total[IV] <= 0:
            return None
        return Paint.from_optics(
            total[:N] / total[IV],
            total[IS] / total[IV],
            body=float(_clip(total[IB] / total[IV], 0.0, 1.0)),
        )

    def colors(self, depth: str = "all") -> list:
        """Hex colour of each bristle column (None where empty): left to right
        on a flat; around the bundle on a round, starting from its front and
        turning towards its left side. ``depth`` is ``"tip"``, ``"belly"``,
        ``"heel"`` or ``"all"``."""
        if depth not in ("tip", "belly", "heel", "all"):
            raise ValueError(
                f"depth must be 'tip', 'belly', 'heel' or 'all', got {depth!r:.80}"
            )
        cols = {
            "tip": self.tip,
            "belly": self.reservoir,
            "heel": self.heel,
            "all": self.tip + self.reservoir + self.heel,
        }[depth]
        out = []
        for c in cols:
            if c[IV] <= 1e-9:
                out.append(None)
            else:
                out.append(Paint.from_optics(c[:N] / c[IV], c[IS] / c[IV]).hex)
        return out


def load_paint(brush: Brush, paint: StrokePaint | None, amount: float) -> None:
    """Load what a stroke asks for onto the brush: one paint (``amount`` of
    it), or several, each as ``(paint, amount[, across[, depth]])`` or as a
    dict of :meth:`Brush.load` keywords (``amount`` is then not used). Every
    load is checked before any goes on the brush."""
    for colour, volume, across, depth, width in _checked_loads(brush, paint, amount):
        brush.load(colour, volume, across=across, depth=depth, width=width)


def _checked_loads(brush: Brush, paint, amount) -> list:
    """What :func:`load_paint` would load on ``brush`` (see :func:`_loads`),
    every part of it checked (on a round brush too)."""
    loads = _loads(paint, amount)
    for _, _, across, _, _ in loads:
        _spread(brush, across)
    return loads


#: What a stroke loads on the brush: one paint, or several, each as
#: ``(paint, amount[, across[, depth]])`` or as :meth:`Brush.load` keywords.
StrokePaint = ColorLike | Sequence[tuple | dict]

_LOAD_FORM = (
    "a load is (paint, amount[, across[, depth]]) or a dict with the keys "
    "paint, amount and optionally across, depth and width"
)


def _is_colour(paint) -> bool:
    """A single colour rather than a list of loads: a Paint, a string, an
    array, or a sequence of three numbers (an RGB triple)."""
    if isinstance(paint, (Paint, str, np.ndarray)):
        return True
    return (
        isinstance(paint, (tuple, list))
        and len(paint) == 3
        and all(map(_is_number, paint))
    )


def _loads(paint, amount) -> list[tuple[Paint, float, Any, Any, float | None]]:
    """What ``paint`` asks to load, as (paint, amount, across, depth, width)
    each, all checked (``across`` as on a flat brush; see :func:`load_paint`)
    so that a stroke can refuse them before it starts."""
    if paint is None:
        return []
    if _is_colour(paint) or not isinstance(paint, (tuple, list)):
        return [(as_paint(paint), _number("amount", amount), None, "belly", None)]
    if paint and all(map(_is_number, paint)):
        raise TypeError(
            f"a colour has three components (r, g, b), got {len(paint)} numbers: "
            f"{paint!r:.80}"
        )
    return list(starmap(_load_item, enumerate(paint)))


def _load_item(i: int, item) -> tuple[Paint, float, Any, Any, float | None]:
    width = None
    if isinstance(item, dict):
        unknown = set(item) - {"paint", "amount", "across", "depth", "width"}
        if unknown or not {"paint", "amount"} <= set(item):
            what = (
                f"has the key {min(unknown)!r}" if unknown else "misses paint or amount"
            )
            raise ValueError(f"load {i} {what}: {_LOAD_FORM}")
        if item.get("width") is not None:
            width = _width(f"width of load {i}", item["width"])
        item = (
            item["paint"],
            item["amount"],
            item.get("across"),
            item.get("depth", "belly"),
        )
    elif not isinstance(item, (tuple, list)) or not 2 <= len(item) <= 4:
        raise TypeError(f"load {i} is {item!r:.80}; {_LOAD_FORM}")
    colour, volume = item[0], item[1]
    across = item[2] if len(item) > 2 else None
    depth = item[3] if len(item) > 3 and item[3] is not None else "belly"
    loaded = as_paint(colour), _number(f"amount of load {i}", volume)
    if not np.isfinite(_across_weights(across)).all():
        raise ValueError(f"across must give finite weights, got {across!r:.80}")
    _depth_weights(depth)
    return *loaded, across, depth, width


#: How a tool behaves in a stroke (see BrushType.physics), and the range each
#: is clipped to; a stroke may override all but rough and ridge.
_PHYSICS = {
    "streaks": (0.0, 2.0),
    "pickup": (0.0, 2.0),
    "drag": (0.0, 2.0),
    "mixing": (0.0, 2.0),
    "rough": (0.0, 2.0),
    "smear": (0.0, None),
    "blending": (0.0, 1.0),
    "ridge": (None, None),
    "lift": (0.0, 1.0),
}


def _physics(bt: BrushType, stroke: Stroke | None = None) -> dict:
    """How tool ``bt`` behaves in a stroke: derived from its character,
    unless the stroke (first) or the tool overrides a value. Each is clipped
    to the range the engine works in (an override out of range would make
    paint negative or blow up); a non-finite override raises ValueError."""
    st, br = bt.stiffness - 0.5, bt.bristles - 0.5
    if bt.rigid:  # smooth steel: no streaks, lays paint on top, little mixing
        # (a knife drags wet paint rather than picking it up: pickup and drag
        # unused)
        derived = {
            "streaks": 0.03,
            "pickup": 0.0,
            "drag": 0.0,
            "mixing": 0.12,
            "rough": 0.04,
            "blending": 0.12,
        }
    else:
        derived = {
            "streaks": 0.35 + 0.5 * st - 0.3 * br,
            "pickup": 0.35 - 0.2 * st,
            "drag": 0.2 + 0.3 * st,
            "mixing": 0.35 + 0.2 * st,
            "rough": 0.1 + 0.2 * st - 0.1 * br,
            "blending": 0.4 + 0.5 * st,
        }
    # (how much paint the tips hold: at least a little)
    derived["smear"] = max(bt.capacity, 0.05)
    # a slight build-up of paint at the edges of a brushed stroke
    derived["ridge"] = 0.0 if bt.rigid else 0.3 + 0.4 * st
    # how a tool leaves the paint: stiff bristles flick off one clump after
    # another, soft hair closes up and slides off clean
    derived["lift"] = (
        0.3 if bt.rigid else float(_clip((bt.stiffness - 0.5) / 0.3, 0.0, 1.0))
    )
    out = {}
    for key, (lo, hi) in _PHYSICS.items():
        value = derived[key]
        for source, prefix in ((stroke, ""), (bt, "BrushType ")):
            given = getattr(source, key, None)
            if given is not None:
                value = _number(prefix + key, given)
                break
        out[key] = value if lo is None else float(_clip(value, lo, hi))
    return out


_clip = np.clip


def _smoothstep(e0, e1, x):
    t = _clip((x - e0) / (e1 - e0), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


#: What a stroke paints with is defined here; the stroke and its simulation
#: are in modules that build on this one. These names are found here too
#: (imported from there when first asked for).
_ELSEWHERE = {
    "Stroke": "stroke",
    "apply_stroke": "stroke",
    "_resolve": "stroke",
    "_Run": "stroke",
    "_pending": "stroke",
    "_brush_step": "stroke",
    "_chord_table": "_geometry",
    "_chord_table_at": "_geometry",
    "_hair_field": "_hair",
}


def __getattr__(name: str) -> Any:
    if name in _ELSEWHERE:
        module = importlib.import_module(f".{_ELSEWHERE[name]}", __package__)
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
