"""Brushes, knives and the stroke simulation.

A :class:`BrushType` is a tool: the shape of its footprint (a round, flat,
filbert or fan brush, or the steel blade of a knife), its size and the
character of its hair, from which its behaviour in a stroke follows
(:meth:`BrushType.physics`). A :class:`Brush` is one such tool with paint on
it, which it keeps (with any dirt) between strokes. A :class:`Stroke` is one
pass of a tool along a path; :func:`apply_stroke` paints it.

Paint on a brush is held in narrow bristle *columns* (:data:`COLUMNS`),
each with three depths: the tip that touches the canvas, the belly, and the
heel near the ferrule. On flat, filbert and fan brushes the columns run
across the width; on a round they go around the bundle, so rolling the
brush between the fingers turns different bristles to the canvas. That lets a
brush carry several paints side by side (double loading), on its tip or deep
in its heel, or only partly mixed; the paints blend into each other as the
brush is worked, at a rate set by :attr:`BrushType.blending`.

At every step along the stroke a brush pushes some of the wet paint under it
forward (drag), its tips lift some of it (pickup) and lay a mix of what they
hold, and they are refilled from the belly. Paint picked up from the canvas
is therefore smeared forward and fades out over a short distance, while some
of it works its way into the belly (a dirty brush). A knife holds its paint on
the blade: it lays it as a slab, at a height set by pressure, and drags the
wet paint under it along as a thin film rather than picking it up.

On the canvas, each pixel has the dry paint (or the ground) plus two wet
sublayers: ``und`` (older wet paint) and ``top`` (the paint of the most recent
strokes). Brush pressure stirs the two together; that is the difference
between laying color *over* wet paint and churning it *into* wet paint.

Paint quantities are stored *packed*: a vector of the absorption spectrum K
(N_BANDS values), then scattering S, volume V and body (the paint's
stiffness times V), all extensive (summed over the paint present). Mixing
paint is then just adding vectors.

This module holds the tools and prepares each step of a stroke; the work on
the pixels under the tool is compiled (numba), in :mod:`oilpaint._engine`.
"""

from __future__ import annotations

import functools
import hashlib
import math
import numbers
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from itertools import starmap
from typing import Any, ClassVar, cast

import numpy as np

from . import _engine as eng
from . import spectral as sp
from ._dab import dab_pattern
from ._engine import COLUMNS, IB, IS, IV, PACKED
from .noise import aniso_noise, value_noise_1d, value_noise_2d
from .paint import ColorLike, Paint, as_paint
from .spline import Path, Profile, as_path, evaluate_profile

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

# --- knives --------------------------------------------------------------------
# How high a knife blade rides above the surface with no pressure (paint thickness
# units).
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
# How much paint a knife presses out beyond the ends of its blade, per unit of
# slab thickness over a pass (times 0.8 + pressure), from how far in from its
# ends (px).
_KNIFE_END = 1.35
_KNIFE_END_PX = 5.0
# How far (px) beyond the end of the blade the paint it presses out lands.
_KNIFE_END_OUT = 2.5
# How much paint the bead on a skimming blade's edge leaves (relative to a
# blade spreading its film), and how high the grain must stand for it to catch
# with no pressure at all.
_KNIFE_BEAD = 1.5
_KNIFE_TOUCH = 0.7
# A knife rides no higher than this times the paint film under its blade.
_KNIFE_FILM = 0.5
# Radius (px) over which wet paint dragged by a blade moves together (viscosity).
_VISCOUS = 3
# How deep a knife sinks into wet paint already on the canvas, as a share of
# its thickness: pressure ** _KNIFE_SINK (light touch: skims; hard: nearly all).
_KNIFE_SINK = 1.8


# --- lifting a brush off (see _Run._lift) ------------------------------------------
# Over how much of the end (px, for a brush 36 px wide, at most) a stiff
# brush's clumps leave the paint one after another, and how far they pull
# thick paint up.
_LIFT_LEN = 21.0
_LIFT_CREST = 4.0

# --- the hair of a brush (see _hair_field) ----------------------------------------
# The most samples a hair pattern takes (rows times columns).
_HAIR_SAMPLES = 8.0e6
# How far the nicks of a knife's blade drift, relative to a brush's clumps.
_NICK_WANDER = 0.15
# Distance between bristle clumps (px), for the finest and the coarsest hair:
# the furrows they comb are a physical size, the same on any width of brush.
_CLUMP_PITCH = (3.0, 7.0)
# How far the clumps drift sideways along a stroke, in clump spacings (stiff
# hair; soft hair half as far).
_HAIR_WANDER = 0.65
# Lattice cells (px) of the drift, the finer drift, the regrouping and the
# bearing-down of the clumps, along the stroke, in brush widths (with a floor
# in px), and across, in clump spacings (with a floor in brush widths).
_HAIR_CELLS = (
    ((2.5, 35.0), (3.0, 0.0)),
    ((0.8, 12.0), (1.5, 0.0)),
    ((4.5, 50.0), (2.0, 0.6)),
    ((1.5, 20.0), (2.0, 0.0)),
)


def _hair_field(
    bt: BrushType,
    width: float,
    half: float,
    s0: float,
    s1: float,
    rng,
    wander: float = 1.0,
):
    """The hair of a brush over one stroke: where its bristle clumps comb the
    paint, as a field over distance along the stroke (``s0``..``s1`` px) and
    position across the bundle (px, out to a little past ``half``, its
    largest half-width), from 0 on a clump's line (a furrow) to 1 between
    clumps (a ridge); see :func:`_engine.hair_field`. Returns the field and
    where it lies: (s at its first row, x at its first column, row spacing,
    column spacing).

    The clumps are a physical size (coarser for coarse hair), not a share of
    the brush's width: a bigger brush has more of them. Along the stroke they
    drift sideways, crowd together and part (furrows merge and split), the
    brush regroups now and then (furrows end and others start), and the
    clumps bear down more or less (furrows deepen and fade): a furrow lasts
    a width or two. The field is fixed in stroke coordinates, so the steps
    of a stroke all comb the same lanes."""
    lo, hi = _CLUMP_PITCH
    pitch = hi - (hi - lo) * float(_clip(bt.bristles, 0.0, 1.0))
    wander *= _HAIR_WANDER * pitch * (0.5 + 0.5 * float(_clip(bt.stiffness, 0.0, 1.0)))
    w = max(width, 12.0)
    x_half = 1.35 * half + 1.0
    length = s1 - s0
    dx = max(0.5, 2.0 * x_half / 400.0)
    cells = np.array(
        [(max(a * w, a0), max(b * pitch, b0 * w)) for (a, a0), (b, b0) in _HAIR_CELLS]
    )
    # rows every 2 px, further apart on a long stroke, but at least four to the
    # shortest cell along it (and no more samples than _HAIR_SAMPLES)
    n_x = int(2.0 * x_half / dx) + 2
    ds = max(2.0, min(length / 1500.0, float(cells[:, 0].min()) / 4.0))
    ds = max(ds, length * n_x / _HAIR_SAMPLES)
    n_s = int(length / ds) + 2
    span_x, span_s = n_x * dx, n_s * ds
    grids = tuple(
        rng.random((int(span_s / cs) + 3, int(span_x / cx) + 3)) for cs, cx in cells
    )
    lane_cells = np.array([0.5 * pitch, max(0.2 * pitch, 1.0)])
    n_lane = int((span_x + 4.0 * wander + 2.0) / lane_cells[1]) + 3
    lanes = rng.random((4, n_lane))
    field = eng.hair_field(n_s, n_x, ds, dx, cells, grids, lanes, lane_cells, wander)
    return field, (float(s0), -x_half, float(ds), float(dx))


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


def _check_profile(name: str, profile, size: bool = False) -> None:
    """A width, pressure, angle or roll: a number, a sequence of numbers or a
    function of u. Numbers must be finite (and not negative for a ``size``);
    a function's values are checked when the stroke samples them
    (:func:`_check_sampled`)."""
    if callable(profile):
        return
    accepted = "a number, a sequence of numbers or a function of u"
    if _is_number(profile):  # (a Fraction, say, too)
        profile = float(profile)
    arr = np.asarray(profile) if not isinstance(profile, str) else None
    if arr is None or arr.dtype.kind not in "biuf":
        raise TypeError(f"{name} must be {accepted}, got {profile!r:.80}")
    if arr.ndim > 1 or arr.size == 0:
        raise ValueError(
            f"{name} must be {accepted}; got an array of shape {arr.shape}"
        )
    arr = arr.reshape(-1).astype(np.float64)
    bad = ~np.isfinite(arr) | ((arr < 0) if size else False)
    if bad.any():
        i = int(np.argmax(bad))
        need = "finite and not negative" if size else "finite"
        where = f" (item {i})" if arr.size > 1 else ""
        raise ValueError(f"{name} must be {need}, got {arr[i]}{where}")


def _check_sampled(u, **samples) -> None:
    """The values a stroke's width, pressure, angle and roll take along it
    (where they are functions of u, nothing else checks them)."""
    for name, values in samples.items():
        bad = ~np.isfinite(values)
        if bad.any():
            i = int(np.argmax(bad))
            raise ValueError(
                f"{name} must be finite along the stroke; it is {values[i]} at "
                f"u = {float(u[i]):.4g}"
            )


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
        as in :meth:`Brush.load`, e.g. ``[(red, 6), (white, 2, "left")]``
        (a tuple of them works too), or a dict of :meth:`Brush.load`'s
        keywords (``paint``, ``amount``, ``across``, ``depth``, ``width``).
        A list or tuple of three numbers is one RGB colour.
    amount:
        Paint loaded on the brush, measured in *squares of the stroke's
        width at normal thickness*. Roughly: ``amount=6`` paints about 6
        widths of solid stroke before the brush starts running dry. More
        paint also means a thicker layer (impasto). The width it is measured
        with is the stroke's *mean* width, after any ``taper`` and along any
        width profile: a tapered or narrowing stroke loads less paint for the
        same ``amount`` (``taper=0.3`` loads about 58% of what the same
        stroke untapered does), and keeps the thickness about the same.
        Not used when ``paint`` is a list of loads (each has its own amount).
    width:
        Brush width in pixels, 0 or more: a number, one value per control
        point, or a function of u (0..1 along the stroke). A function is
        called once with the array of all u; if it only takes one number
        (``lambda u: 30 if u < 0.5 else 10``) it is called for each u in
        turn. Where the width is less than 1 px (in any of these forms), the
        stroke is 1 px wide.
        Defaults to the brush's width.
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
        knife lifted straight off thick paint does. A brush's bristle clumps
        leave the paint one after another over the last stretch of the
        stroke (a ragged end), each pulling thick paint up into a crest.
        Default 0.3 for knives; for brushes, from the stiffness (hog about
        1, sable 0: soft hair slides off clean). Given explicitly it acts on
        any brush that lays paint, on strokes and dabs alike. Stiffer paint
        holds sharper peaks.

        ``mixing``, ``pickup``, ``drag``, ``streaks``, ``smear``,
        ``blending``, ``taper`` and ``lift`` default to values derived from
        the brush type. Given here they are clipped as the brush type's are:
        ``mixing``, ``pickup``, ``drag`` and ``streaks`` to 0..2,
        ``blending`` and ``lift`` to 0..1, ``smear`` to 0 or more.
    seed:
        Random seed for the stroke's texture: None (from the canvas's own
        random numbers), or anything ``np.random.default_rng`` takes (a
        whole number from 0 up, a sequence of them, a SeedSequence, ...).

    A stroke checks its arguments when it is made, and raises ValueError or
    TypeError (naming the argument) unless they have the forms above and
    every number in them is finite (and no width negative). A function of u
    is only called when the stroke is painted: if it gives a value that is
    not finite there, the stroke raises ValueError before any paint is
    laid.
    """

    path: object
    paint: StrokePaint | None = None
    amount: float = 6.0
    width: Profile | None = None
    pressure: Profile = 0.5
    brush: Brush | BrushType | None = None
    angle: Profile | None = None
    follow_path: bool | None = None
    roll: Profile | None = None
    mixing: float | None = None
    pickup: float | None = None
    drag: float | None = None
    streaks: float | None = None
    smear: float | None = None
    blending: float | None = None
    taper: object = None
    lift: float | None = None
    seed: int | None = None

    def __post_init__(self):
        _check_tool(self.brush)
        _number("amount", self.amount)
        _loads(self.paint, self.amount)
        if self.width is not None:
            _check_profile("width", self.width, size=True)
        _check_profile("pressure", self.pressure)
        for name in ("angle", "roll"):
            if getattr(self, name) is not None:
                _check_profile(name, getattr(self, name))
        for name in _PHYSICS:
            if getattr(self, name, None) is not None:
                _number(name, getattr(self, name))
        if self.taper is not None:
            _check_taper(self.taper)
        _check_seed("seed", self.seed)


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


def _finite_width(f: Callable) -> Callable:
    """Width function ``f``, refusing values that are not finite as it gives
    them. (The stroke tries its width out before it samples the path; a NaN
    there would otherwise surface as an error from deep in the sampling.)
    It calls ``f`` exactly as often as the stroke would."""

    def width(u):
        values = f(u)
        arr = np.asarray(values)
        if arr.dtype.kind not in "biuf":
            return values  # (not numbers: refused as such by the stroke)
        if not np.isfinite(arr).all():
            raise ValueError(
                "width must be finite along the stroke; the width function "
                f"gave {arr.flat[int(np.argmax(~np.isfinite(arr)))]}"
            )
        return values

    return width


def _resolve(stroke: Stroke):
    """The Brush to paint with, its type, and the effective stroke settings."""
    tool = stroke.brush
    _check_tool(tool)
    brush = tool if isinstance(tool, Brush) else Brush(tool)
    bt = brush.type
    phys = _physics(bt, stroke)
    eff = {
        k: phys[k] for k in ("mixing", "pickup", "drag", "streaks", "smear", "blending")
    }
    eff["rough"] = phys["rough"]
    eff["ridge"] = phys["ridge"]
    eff["lift"] = phys["lift"]
    eff["width"] = stroke.width if stroke.width is not None else bt.width
    if callable(eff["width"]):
        eff["width"] = _finite_width(eff["width"])
    eff["taper"] = stroke.taper if stroke.taper is not None else bt.taper
    eff["angle"] = stroke.angle if stroke.angle is not None else bt.angle
    eff["follow_path"] = (
        stroke.follow_path if stroke.follow_path is not None else bt.follow_path
    )
    return brush, bt, eff


_SUPERELLIPSE = {"flat": 10.0, "filbert": 3.0, "fan": 6.0, "round": 2.0, "spoon": 2.0}

# Knife blades as polygons: (u along the blade from heel -1 to tip +1, v
# across it in units of the blade's half-width, i.e. scaled by aspect).
_BLADES = {
    # a diamond trowel: widest a little behind the middle, pointed tip
    "trowel": [(-1.0, -0.3), (-0.25, -1.0), (1.0, 0.0), (-0.25, 1.0), (-1.0, 0.3)],
    # a pear / teardrop: tapering from a narrow heel to a round tip
    "pear": [(-1.0, -0.35)]
    + [
        (0.3 + 0.7 * np.cos(t), -np.sin(t))
        for t in np.linspace(np.pi / 2, -np.pi / 2, 15)
    ][::-1]
    + [(-1.0, 0.35)],
    # a long straight blade with a rounded end
    "straight": [(-1.0, -1.0)]
    + [
        (0.82 + 0.18 * np.cos(t), np.sin(t))
        for t in np.linspace(-np.pi / 2, np.pi / 2, 11)
    ]
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
        return np.sign(cu) * np.abs(cu) ** (2 / e), aspect * np.sign(sv) * np.abs(
            sv
        ) ** (2 / e)
    poly = np.asarray(_BLADES[shape], float)
    poly = np.vstack([poly, poly[:1]])
    seg = np.linalg.norm(np.diff(poly * [1.0, aspect], axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    t = np.linspace(0, s[-1], n, endpoint=False)
    return np.interp(t, s, poly[:, 0]), aspect * np.interp(t, s, poly[:, 1])


@functools.lru_cache(maxsize=1024)
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
    footprint over [c - h, c + h]. Returns (l grid, c, h, lateral half-extent, max h).

    Tables are kept for the aspect to 3 decimals and the angle to 2, and
    built for those rounded values: the same tool gets the same table
    whatever was painted before. (The arrays are shared: don't change them.)
    """
    return _chord_table_at(shape, round(aspect, 3), round(phi, 2))


@functools.lru_cache(maxsize=4096)
def _chord_table_at(shape: str, aspect: float, phi: float):
    """:func:`_chord_table` for an aspect and angle already rounded."""
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
    return grid, c, h, lmax, float(h.max())


def _signed_floor(x, floor):
    return math.copysign(max(abs(x), floor), x if x != 0 else 1.0)


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
    if run.bt.rigid:
        for i in range(1, len(run.pts)):
            s = run.footprint(i)
            if s is not None:
                _knife_step(run, s)
    else:
        y0, y1, x0, x1 = run.bounds()
        if y0 < y1 and x0 < x1:
            run.pend = _pending(y0, y1, x0, x1)
        for i in range(1, len(run.pts)):
            got = run.step(i)
            if got is not None:
                _brush_step(run, *got)
    eng.settle_pending(run.cv, run.pend, run.stroke_id)
    if run.is_dab and not run.bt.rigid and run.bt.deposits:
        _dab_lift(run)
    run.settle()
    return run.brush


def _pending(y0, y1, x0, x1):
    """Room for new paint from the brushstroke in progress, held back where it
    lands on wet paint from earlier strokes until we know how well the stroke
    covers it (over the part of the canvas the stroke can reach).

    Once the body of the brush has passed over a pixel, or the stroke has
    laid a covering film on it, the old top layer goes under and the held
    paint becomes the new top. Pixels only ever grazed by the brush's
    feathered edge are settled when the stroke ends: the old top keeps its
    place, the share the faint new film would bury goes under, and the new
    paint stirs into the rest. Returns (paint, cover, reach, held, (on, y0,
    x0, h, w, and the rows y0..y1 and columns x0..x1 of it where any is held)),
    as :func:`_engine.deposit` takes it."""
    h, w = y1 - y0, x1 - x0
    return (
        np.zeros((h, w, PACKED), np.float32),
        np.zeros((h, w), np.float32),
        np.zeros((h, w), np.float32),
        np.zeros((h, w), bool),
        # (and where any is held: none yet)
        np.array([1, y0, x0, h, w, h, 0, w, 0], np.int64),
    )


# (no paint held: knives lay theirs at once)
_NO_PENDING = (
    np.zeros((1, 1, PACKED), np.float32),
    np.zeros((1, 1), np.float32),
    np.zeros((1, 1), np.float32),
    np.zeros((1, 1), bool),
    np.zeros(9, np.int64),
)
# (chord tables of a round footprint: not needed)
_NO_TABLES = (np.zeros(2), np.zeros(2), np.zeros(2))


class _Run:
    """One stroke in progress: the tool, the sampled path, and the textures
    and constants that stay the same along the stroke."""

    def __init__(self, canvas, stroke: Stroke):
        self.canvas = canvas
        path: Path = as_path(stroke.path)
        rng = self.rng = np.random.default_rng(
            stroke.seed if stroke.seed is not None else canvas._rng.integers(1 << 31)
        )
        brush, bt, eff = self.brush, self.bt, self.eff = _resolve(stroke)

        n_probe = max(path.n_knots, 2) * 8
        w_probe = evaluate_profile(
            eff["width"],
            np.linspace(0, 1, n_probe),
            np.linspace(0, max(path.n_knots - 1, 0), n_probe),
            path.n_knots,
        )
        w_typ = self.w_typ = max(float(np.mean(w_probe)), 1.0)
        spacing = float(_clip(0.25 * w_typ, 0.5, 8.0))
        taper = eff["taper"]
        t0, t1 = (taper, taper) if isinstance(taper, (int, float, np.number)) else taper
        circle = self.circle = bt.shape == "round" and abs(bt.aspect - 1.0) < 1e-6

        def sample(spacing):
            pts, _, u, knot = path.sample(spacing)
            widths = evaluate_profile(eff["width"], u, knot, path.n_knots)
            if t0 > 0:
                widths = widths * (0.25 + 0.75 * _smoothstep(0.0, t0, u))
            if t1 > 0:
                widths = widths * (0.25 + 0.75 * _smoothstep(1.0, 1.0 - t1, u))
            widths = np.maximum(widths, 1.0)
            pressure = _clip(
                evaluate_profile(stroke.pressure, u, knot, path.n_knots), 0.0, 1.0
            )
            angles = np.radians(evaluate_profile(eff["angle"], u, knot, path.n_knots))
            rolls = np.radians(
                evaluate_profile(
                    stroke.roll if stroke.roll is not None else 0.0,
                    u,
                    knot,
                    path.n_knots,
                )
            )
            _check_sampled(u, width=widths, pressure=pressure, angle=angles, roll=rolls)
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
                pts, widths, pressure, angles, rolls = sample(
                    max(spacing * 0.5 / gap, 0.35)
                )
        is_dab = self.is_dab = len(pts) == 1
        # a dab: press the brush down with a tiny movement (small next to its narrow
        # side)
        if is_dab:
            nudge = max(0.05 * widths[0] * min(1.0, bt.aspect), 0.25)
            pts = np.vstack([pts[0] - [nudge, 0], pts[0] + [nudge, 0]])
            widths = np.repeat(widths, 2)
            pressure = np.repeat(pressure, 2)
            angles = np.repeat(angles, 2)
            rolls = np.repeat(rolls, 2)
        self.pts, self.pressure, self.angles, self.rolls = pts, pressure, angles, rolls

        brush.width = float(np.mean(widths))
        load_paint(brush, stroke.paint, stroke.amount)
        self.lays_paint = bool(bt.deposits and brush.volume > 0)

        self.tex = _NO_TEX
        # A slow wobble of the width.
        wobble = (
            value_noise_1d(len(pts) + 2, max(2.0, 0.8 * w_typ / spacing), rng)[
                : len(pts)
            ]
            - 0.5
        )
        # a knife's height wavers as it is pushed along (faint ripples across the
        # slab: chatter), and a hand never presses both ends of the blade quite
        # equally (the slab is thicker towards one side); the lips squeezed out
        # along a stroke swell and break
        arc = self.arc = np.concatenate(
            [[0.0], np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))]
        )
        period = max(0.4 * w_typ, 2.0)  # ripples about half a blade width apart
        ripple = value_noise_1d(int(arc[-1] / period) + 4, 1.0, rng)
        self.chatter = np.interp(arc / period, np.arange(len(ripple)), ripple) - 0.5
        swell = value_noise_1d(int(arc[-1] / (2.0 * period)) + 4, 1.0, rng)
        self.lip = np.interp(arc / (2.0 * period), np.arange(len(swell)), swell)
        self.blade_tilt = float(_clip(rng.normal(0.0, 0.08), -0.2, 0.2))
        # the bristle tips a dab prints (see _dab)
        if is_dab:
            self.tex2 = dab_pattern(bt, 0.5 * w_typ, rng)
        aspect = self.aspect = 1.0 if circle else bt.aspect
        self.reach = 1.0 if circle else float(np.hypot(1.0, aspect))
        # in half-widths squared
        self.area_unit = _footprint_area("round" if circle else bt.shape, aspect)
        self.stiff_blade = float(_clip((bt.stiffness - 0.4) / 0.6, 0.0, 1.0))
        # where each column sits across the mark
        self.col_u = (_RING_X if brush.ring else _COL_X) * 2.0 - 1.0
        # squeezed-out ridges are at most this wide (px); they lie just outside the mark
        self.margin = float(_clip(0.12 * w_typ * (1.0 + bt.spread), 0.6, 7.0)) + 1.0

        # brush half-width at every sample: size, a slow wobble, and how far the
        # hairs splay under pressure (a pointed round paints thin with a light touch)
        radii = 0.5 * widths * (1.0 + 0.06 * wobble)
        # the bristle bundle's size; pressure only splays it
        self.nominal = radii.copy()
        if bt.spread > 0:
            radii = radii * (
                (1.0 - bt.spread)
                + bt.spread * (np.maximum(pressure, 0.02) / 0.5) ** 0.8
            )
        self.radii = radii
        # soft brushes fade in and out over this many widths
        self.feather = bt.softness * 3.0
        # lifting a brush off at the end (see _lift): how far before the end it
        # starts, at most (px)
        self.lift_reach = (
            0.0
            if is_dab or bt.rigid or not bt.deposits
            else _LIFT_LEN
            * eff["lift"]
            * 1.5
            * (2.0 * float(radii.max()) / 36.0) ** 0.3
        )
        # (no step further than this from the end can be lifting off)
        steps = np.linalg.norm(np.diff(pts, axis=0), axis=1) if len(pts) > 1 else [0.0]
        self._lift_from = (
            float(arc[-1])
            - self.lift_reach
            - float(np.max(steps))
            - 2.0 * float(radii.max())
            - 1.0
        )

        surf = canvas.surface
        self.surface_acts = surf.affects_paint
        # how much the surface texture (vs. the hair pattern) decides where a dry brush
        # catches
        self.tooth_w = float(_clip(0.3 + (surf.tooth - 0.5), 0.0, 0.85))

        if bt.rigid:
            # The wet paint lying on the canvas before the stroke: the blade's ends
            # push it aside into lips (a pile it is carrying mostly rides along).
            # (Over the part of the canvas the steps' windows cover, from
            # (wet_y0, wet_x0); see step.)
            pad = float(radii.max()) * self.reach + 3.0
            H, W = canvas.height, canvas.width
            self.wet_y0 = min(max(math.floor(float(pts[:, 1].min()) - pad - 1), 0), H)
            self.wet_x0 = min(max(math.floor(float(pts[:, 0].min()) - pad - 1), 0), W)
            wet_y1 = max(math.ceil(float(pts[:, 1].max()) + pad + 1) + 1, self.wet_y0)
            wet_x1 = max(math.ceil(float(pts[:, 0].max()) + pad + 1) + 1, self.wet_x0)
            box = (slice(self.wet_y0, wet_y1), slice(self.wet_x0, wet_x1))
            self.wet_before = canvas.top_v[box] + canvas.und_v[box]
            # Paint on a blade isn't held in lanes like paint in bristles: its
            # colour bands wander a little from side to side along the stroke.
            period = max(0.5 * w_typ, 4.0)
            n_s = int(arc[-1] / period) + 3
            self.bands = value_noise_2d(8 * n_s, 64, 8.0, rng)
            self.band_period = period

        self.stroke_id = canvas._new_stroke_id()
        self.dry = False
        brush.wet = 0.0

        # the samples as plain floats, for the per-step arithmetic
        self._pts = [(float(x), float(y)) for x, y in pts]
        self._pressure, self._angles, self._rolls = (
            pressure.tolist(),
            angles.tolist(),
            rolls.tolist(),
        )
        self._radii, self._nominal = radii.tolist(), self.nominal.tolist()
        self._arc, self._lip = arc.tolist(), self.lip.tolist()

        # what the compiled engine works on
        self.cv = (canvas.top, canvas.und, canvas.top_a, canvas.und_a, canvas.top_id)
        self.aux = (canvas._floor, canvas._tooth, canvas._hollows, canvas.dry_height)
        self.br = (brush.tip, brush.reservoir, brush.heel)
        self.pend = _NO_PENDING
        # the brush's hair: where its bristle clumps run along the stroke (see
        # _hair_field)
        hair_at = (0.0, 0.0, 1.0, 1.0)
        if not is_dab:
            pad = float(radii.max()) * self.reach + 4.0
            # (a blade's nicks score nearly straight lines, which come and go
            # as grit catches and paint closes over them)
            self.tex2, hair_at = _hair_field(
                bt,
                w_typ,
                float(self.nominal.max()),
                -pad,
                float(arc[-1]) + pad,
                rng,
                wander=_NICK_WANDER if bt.rigid else 1.0,
            )
        # how ragged this stroke's end is (each stroke ends its own way)
        self.lift_var = float(0.4 + 1.2 * rng.random()) if self.lift_reach > 0 else 1.0
        self.K = eng.StrokeConsts(
            stroke_id=int(self.stroke_id),
            circle=bool(circle),
            is_dab=bool(is_dab),
            ring=bool(brush.ring),
            rigid=bool(bt.rigid),
            deposits=bool(bt.deposits),
            margin=float(self.margin),
            tooth_w=self.tooth_w,
            drag=float(eff["drag"]),
            pickup=float(eff["pickup"]),
            streaks=float(eff["streaks"]),
            mixing=float(eff["mixing"]),
            smear=float(eff["smear"]),
            blending=float(eff["blending"]),
            rough=float(eff["rough"]),
            ridge=float(eff["ridge"]),
            stiffness=float(bt.stiffness),
            capacity=float(bt.capacity),
            dig=float(bt.dig),
            softness=float(bt.softness),
            slant=float(bt.slant),
            clumps=float(bt.clumps),
            feather=float(self.feather),
            # soft brushes are lowered and lifted gently, so their ends feather
            end_k=1.0 if is_dab else max(0.0, 1.0 - 1.5 * float(bt.softness)),
            surface_acts=bool(self.surface_acts),
            slickness=float(surf.slickness),
            absorbency=float(surf.absorbency),
            surface_tooth=float(surf.tooth),
            arc_end=float(arc[-1]),
            hair_s0=hair_at[0],
            hair_x0=hair_at[1],
            hair_ds=hair_at[2],
            hair_dx=hair_at[3],
            spacing=float(spacing),
            bundle=float(brush.width),
        )

    # ------------------------------------------------------------------
    def step(self, i: int):
        """Step ``i`` of the stroke, from point i-1 to point i: its scalars (an
        :class:`_engine.Step`) and the footprint's chord tables, or None if
        the tool doesn't reach the canvas there."""
        canvas, brush, bt = self.canvas, self.brush, self.bt
        # (plain floats: this runs at every step)
        pts, pressure, angles, radii = (
            self._pts,
            self._pressure,
            self._angles,
            self._radii,
        )
        H, W = canvas.height, canvas.width
        (ax, ay), (cx, cy) = pts[i - 1], pts[i]
        ds = math.hypot(cx - ax, cy - ay)
        if ds < 1e-9:
            return None
        tx, ty = (cx - ax) / ds, (cy - ay) / ds
        p = 0.5 * (pressure[i] + pressure[i - 1])
        r0, r1 = radii[i - 1], radii[i]
        r = 0.5 * (r0 + r1)
        Wd = 2.0 * r
        # orientation of the brush's long side relative to the stroke's cross direction
        ang = 0.5 * (angles[i] + angles[i - 1])
        phi = ang if self.eff["follow_path"] else ang - math.atan2(tx, -ty)
        # a brush turned past 90 degrees has its sides swapped
        side = 1.0 if math.cos(phi) >= 0 else -1.0
        phi = (phi + math.pi / 2) % math.pi - math.pi / 2

        # the window around the swept segment, and its part on the canvas
        R = max(r0, r1) * self.reach + (3.0 if bt.rigid else 0.0)
        x0, x1 = math.floor(min(ax, cx) - R - 1), math.ceil(max(ax, cx) + R + 1) + 1
        y0, y1 = math.floor(min(ay, cy) - R - 1), math.ceil(max(ay, cy) + R + 1) + 1
        cx0, cx1, cy0, cy1 = max(x0, 0), min(x1, W), max(y0, 0), min(y1, H)
        if cx0 >= cx1 or cy0 >= cy1:
            _use_paint_off_canvas(brush, Wd, ds, p)
            return None
        if self.circle:
            tables, lmax, across, h_max, uscale = _NO_TABLES, 1.0, 1.0, 1.0, side
        else:
            grid, c_tab, h_tab, lmax, h_max = _chord_table(bt.shape, self.aspect, phi)
            tables = (grid, c_tab, h_tab)
            sf = _signed_floor(math.cos(phi), self.aspect)
            across, uscale = abs(sf), side / sf
        # On a thin brush one pixel spans several bristle columns: sample across
        # it so that every column reaches the canvas (else paint in the columns
        # between pixel centres would only come out by diffusion, and thin
        # strokes would run long).
        span = 1.0 / max(r * across, 1e-6)
        span_cols = span * COLUMNS * (0.25 if brush.ring else 0.5)
        n = len(pts)
        lift_len, lift_amp = (
            self._lift(i, p, Wd, ds)
            if self.lift_reach > 0 and self._arc[i] >= self._lift_from
            else (0.0, 0.0)
        )
        overload = self._overload(i, p) if self.is_dab else 0.0
        st = eng.Step(
            i=i,
            first=i == 1,
            last=i == n - 1,
            touchdown=i == 1 or self.is_dab,
            x0=x0,
            y0=y0,
            hh=y1 - y0,
            ww=x1 - x0,
            cy0=cy0,
            cx0=cx0,
            ch=cy1 - cy0,
            cw=cx1 - cx0,
            wy0=cy0 - y0,
            wx0=cx0 - x0,
            cx=cx,
            cy=cy,
            tx=tx,
            ty=ty,
            ds=ds,
            p=p,
            p_now=pressure[i],
            p_before=pressure[i - 1],
            r0=r0,
            r1=r1,
            r=r,
            Wd=Wd,
            # how loaded the brush is is relative to its real size
            Wn=self._nominal[i - 1] + self._nominal[i],
            side=side,
            uscale=uscale,
            lmax=lmax,
            h_tab_max=h_max,
            arc_i=self._arc[i],
            lip=self._lip[i],
            wet=float(brush.wet),
            dry=bool(self.dry),
            span=span,
            nsub=int(min(max(math.ceil(span_cols / 2.0 - 1e-9), 1), 8)),
            rho=0.5 * (self._rolls[i] + self._rolls[i - 1]),
            splay=r / max(0.5 * (self._nominal[i - 1] + self._nominal[i]), 1e-9),
            lift_len=lift_len,
            lift_amp=lift_amp,
            overload=overload,
        )
        return st, tables

    def _load(self, i: int, p: float) -> tuple[float, float, float]:
        """How loaded the brush is at step ``i``, as the engine reckons it
        (see _engine.brush_step_on): its load and what its hair holds (in
        load units), and the volume on it."""
        brush, bt = self.brush, self.bt
        Wn = max(self._nominal[i - 1] + self._nominal[i], 0.6 * self.K.bundle)
        vol = float(brush.tip[:, IV].sum() + brush.reservoir[:, IV].sum())
        hold = eng.HOLD * bt.capacity * (1.3 - 0.6 * p) * max(1.0, 20.0 / Wn) ** 1.5
        return vol / (Wn * Wn), hold, vol

    def _overload(self, i: int, p: float) -> float:
        """How far a dab's load goes beyond what its hair holds (0: none)."""
        if not self.bt.deposits or self.bt.rigid:
            return 0.0
        load, hold, _ = self._load(i, p)
        return max(load / hold - 1.0, 0.0)

    def _lift(self, i: int, p: float, Wd: float, ds: float) -> tuple[float, float]:
        """How the brush lifts off at the end of the stroke, as seen at step
        ``i``: over how many px before the end its bristle clumps leave the
        paint one after another, and how far they pull it up as they go.

        Only stiff hair flicks off clump by clump (``lift``, by default from
        the stiffness); soft hair closes up and slides off clean. The more the
        brush is loaded the more it does so: a brush running dry is already
        breaking up, and its tail keeps its length. Pressing hard at the end
        keeps the clumps down longer; a round's point gathers them. On a short
        stroke (a dab-like mark) the clumps leave together, and only thick,
        stiff paint is pulled up into crests."""
        none = (0.0, 0.0)
        arc = self._arc
        if arc[-1] - arc[i] > self.lift_reach + ds + Wd:
            return none  # (not reached: see _lift_from)
        brush, bt = self.brush, self.bt
        load, hold, vol = self._load(i, p)
        loaded = float(_smoothstep(0.15, 0.6, load / hold))
        if loaded <= 0:
            return none
        length = arc[-1]
        short = float(_smoothstep(1.0, 3.0, length / Wd))
        lift = self.eff["lift"]
        # The clumps leave over the distance the bristle tips travel while the
        # brush comes up: a matter of how long the hair is (it grows far more
        # slowly than the width), of how hard it was pressed, and of the hand
        # (each stroke its own).
        L = (
            _LIFT_LEN
            * lift
            * (Wd / 36.0) ** 0.3
            * self.lift_var
            * (1.25 - 0.75 * float(self._pressure[-1]))
            * (0.5 if bt.shape == "round" else 1.0)
            * loaded
            * short
        )
        L = min(L, 0.25 * length)
        # how thick it lays, and how stiff the paint is
        body = float(
            (brush.tip[:, IB].sum() + brush.reservoir[:, IB].sum()) / max(vol, 1e-12)
        )
        thick = float(eng.deposit_thickness(load)) * (0.5 + p) + eng.IMPASTO * (
            1.2 - 0.5 * p
        ) * body * max(load - hold, 0.0)
        # fluid paint closes up behind the clumps as they leave
        L *= float(_smoothstep(0.1, 0.5, body))
        amp = (
            _LIFT_CREST
            * lift
            * float(_smoothstep(0.8, 3.0, thick))
            * float(_smoothstep(0.3, 0.8, body))
        )
        if L < 1.0:
            return none
        return float(L), float(amp)

    def footprint(self, i: int) -> _Step | None:
        """Where the tool touches the canvas while it moves along segment
        ``i`` (from point i-1 to point i), or None if it touches nothing.

        For every pixel we compute its *exposure* (see :func:`apply_stroke`),
        its anti-aliased coverage, and where it sits under the brush (across
        the bristles, along the motion); see :func:`_engine.footprint`. And
        which bristle columns are above each pixel."""
        got = self.step(i)
        if got is None:
            return None
        st, tables = got
        F, hit = eng.footprint(st, self.K, self.tex, self.tex2, *tables)
        if not hit:
            return None
        s = _Step(st, F, tables)
        u_px = s.ucoord[s.wsl]
        if self.bt.rigid:
            # paint on a blade isn't held in lanes: its colour bands wander a little
            along_stroke = _clip(
                (self.arc[i] + s.along[s.wsl]) / self.band_period * 8.0,
                0.0,
                self.bands.shape[0] - 1.001,
            )
            u_band = _clip((u_px + 1.0) * 0.5 * 63.0, 0.0, 62.999)
            u_px = u_px + _KNIFE_BAND_WANDER * (
                2.0 * _bilinear(self.bands, along_stroke, u_band) - 1.0
            )
        s.taps = eng.bristle_taps(u_px, st.span, st.nsub, self.brush.ring, st.rho)
        return s

    def bounds(self):
        """(y0, y1, x0, x1): the part of the canvas the stroke can reach."""
        canvas, pts = self.canvas, self.pts
        H, W = canvas.height, canvas.width
        pad = float(self.radii.max()) * self.reach + self.margin + 2.0
        bx0, by0 = (
            int(max(np.floor(pts[:, 0].min() - pad), 0)),
            int(max(np.floor(pts[:, 1].min() - pad), 0)),
        )
        bx1, by1 = (
            int(min(np.ceil(pts[:, 0].max() + pad) + 1, W)),
            int(min(np.ceil(pts[:, 1].max() + pad) + 1, H)),
        )
        return by0, by1, bx0, bx1

    def settle(self):
        """The paint settles: fluid paint levels out, sheer edges of thick paint
        slump."""
        canvas = self.canvas
        by0, by1, bx0, bx1 = self.bounds()
        if bx0 < bx1 and by0 < by1:
            eng.level(canvas.top, canvas.und, canvas.dry_height, by0, by1, bx0, bx1, 12)
        # (a brush with no paint of its own only moves the wet paint about)
        if bx0 < bx1 and by0 < by1 and not self.bt.rigid and self.lays_paint:
            eng.soften(
                canvas.top,
                canvas.und,
                canvas.dry_height,
                canvas.top_id,
                self.stroke_id,
                by0,
                by1,
                bx0,
                bx1,
                2,
            )
        if bx0 < bx1 and by0 < by1:
            eng.clear_negative(canvas.top, canvas.und, by0, by1, bx0, bx1)
            eng.fix_coverage(
                canvas.top, canvas.und, canvas.top_a, canvas.und_a, by0, by1, bx0, bx1
            )


# (the stippling pattern of a dab or the hair of a brush: not needed)
_NO_TEX2 = np.zeros((1, 1))
# (the nicks of a blade: not needed)
_NO_TEX = np.zeros(2)


class _Step:
    """The footprint of one segment of a stroke.

    ``st`` holds its scalars (an :class:`_engine.Step`), ``F`` the per-pixel
    footprint over the window around the swept segment, which may stick out
    of the canvas: ``along``, ``lat_px``, ``ucoord``, ``hair``, ``aa``,
    ``profile``, ``c``, ``h``, ``rel``, ``overlap``, ``expo``, ``cover``,
    ``streak`` and ``between`` are its planes. ``wsl`` crops them to the
    canvas part ``sl``. ``taps`` maps bristle columns to the pixels of that
    part, ``tables`` are the footprint's chord tables."""

    _PLANES: ClassVar[dict[str, int]] = {
        "along": eng.F_ALONG,
        "lat_px": eng.F_LAT,
        "ucoord": eng.F_U,
        "hair": eng.F_HAIR,
        "aa": eng.F_AA,
        "profile": eng.F_PROFILE,
        "c": eng.F_C,
        "h": eng.F_H,
        "rel": eng.F_REL,
        "overlap": eng.F_OVERLAP,
        "expo": eng.F_EXPO,
        "cover": eng.F_COVER,
        "streak": eng.F_STREAK,
        "between": eng.F_BETWEEN,
        "edge": eng.F_EDGE,
    }

    taps: tuple

    def __init__(self, st, F, tables):
        self.st, self.F, self.tables = st, F, tables
        self.sl = (slice(st.cy0, st.cy0 + st.ch), slice(st.cx0, st.cx0 + st.cw))
        self.wsl = (slice(st.wy0, st.wy0 + st.ch), slice(st.wx0, st.wx0 + st.cw))

    def __getattr__(self, name):
        k = _Step._PLANES.get(name)
        if k is None:
            raise AttributeError(name)
        return self.F[k]


# ----------------------------------------------------------------------------
# one step of a brush, one step of a knife


def _brush_step(run: _Run, st, tables) -> None:
    """One step of a brush (see :func:`_engine.brush_step`): it drags the wet
    paint under it along; refills its tips and picks up wet paint; combs
    through thick paint, pushing it aside into ridges; lays paint from its
    tips (a dry brush only on the tooth of the canvas); and stirs the fresh
    paint into the wet paint below."""
    run.brush.wet, run.dry = eng.brush_step(
        st,
        run.K,
        run.tex,
        run.tex2,
        tables,
        run.cv,
        run.aux,
        run.pend,
        run.br,
        run.col_u,
    )


def _knife_step(run: _Run, s: _Step) -> None:
    canvas, brush, bt, eff, st = run.canvas, run.brush, run.bt, run.eff, s.st
    p, sl, wsl = st.p, s.sl, s.wsl
    top, und, floor = canvas.top[sl], canvas.und[sl], canvas._floor[sl]
    hair, expo_c = s.hair[wsl], s.expo[wsl]
    f_mix = eng.surface_factors(run.cv, run.aux, st, run.K)[2]
    area = max(run.area_unit * st.r * st.r, 1e-9)  # of the blade, px

    # 1) All the paint on a knife is on the blade, ready to go down; several
    #    paints loaded side by side blend slowly as it works. (A knife doesn't
    #    pick wet paint up as it goes: it drags it, step 5.)
    eng.work_load(run.br, run.K, st, np.inf)

    # 2) The blade rides above the surface at a gap set by pressure (all its
    #    paint is a film under it; what doesn't fit under it rides along on it).
    col_vol, _load, col_load, _col_body, body = eng.load_stats(
        brush.tip, brush.reservoir, st.Wn
    )
    gap = _KNIFE_GAP * (1.0 - p) ** 1.6 * (0.55 + 0.45 * bt.stiffness) + 0.05
    if bt.clumps > 0:  # the teeth of a comb dig in, down to the threads
        gap *= 0.03
    hold = gap * run.area_unit / 4.0 + _KNIFE_CARRY

    # 3) Where the blade sits: on the high points of the surface below (thread
    #    tops for a flexible blade, the highest ridges around for a stiff one).
    rest = canvas._peaks_fine[sl] + run.stiff_blade * (
        canvas._peaks_wide[sl] - canvas._peaks_fine[sl]
    )
    # Wet paint from earlier strokes holds the blade up too, the heap it pushes
    # included (a knife rides on a wave of paint): it rests on its high points
    # and sinks in the harder it is pressed (less into stiff paint). A light
    # touch lays new paint on top of the old.
    earlier = canvas.top_id[sl] != run.stroke_id
    old_v = und[..., IV] + np.where(earlier, top[..., IV], 0.0)
    if old_v.max() > 0.05:
        old_body = float(
            (und[..., IB].sum() + np.where(earlier, top[..., IB], 0.0).sum())
            / max(float(old_v.sum()), 1e-9)
        )
        # How far into it the blade sinks, as a share of its thickness: a light
        # touch skims, a firm one cuts in, pressing hard goes nearly down to
        # the canvas, however thick the paint. Stiff paint resists a little.
        depth = float(
            _clip(p**_KNIFE_SINK * (1.15 - 0.3 * min(old_body, 1.0)), 0.0, 1.0)
        )
        # narrow teeth concentrate the pressure: they cut right through
        if bt.clumps > 0:
            depth = max(depth, 0.9)
        # The blade is rigid along the direction it moves: its height varies only
        # along its length. It rests on the high points under it (not on the pile
        # it pushes ahead), bending to follow them over a span set by its stiffness.
        under_blade = (s.aa * _clip(s.h - np.abs(s.rel) + 0.5, 0.0, 1.0))[wsl] > 0.5
        if under_blade.any():
            span = int(_clip(0.1 * run.w_typ * (0.5 + run.stiff_blade), 2, 10))
            hi = eng.max_filter(floor + old_v, 1)
            high = _along_blade(s.lat_px[wsl], under_blade, hi, span)
            base = _along_blade(s.lat_px[wsl], under_blade, rest, span)
            rest = np.maximum(rest, high - depth * np.maximum(high - base, 0.0))
    # The blade rides on the paint under it: as that runs low, it sinks and the
    # slab thins out. Nicks and grit on its edge leave fine scratches along the
    # stroke; it wavers as it is pushed (chatter) and is never pressed quite
    # evenly along its length (tilt).
    film = (
        eng.at_pixels(s.taps, brush.tip[:, IV] + brush.reservoir[:, IV], st.ch, st.cw)
        * COLUMNS
        / area
    )
    ride = np.minimum(gap, _KNIFE_FILM * film) if bt.clumps == 0 else gap
    # (skimming, the blade glides: no chatter; its nicks still streak the paint)
    chat = 0.25 * (1.0 - float(_smoothstep(_KNIFE_SKIM, 0.05, p)))
    level = rest + ride * (
        1.0
        + chat * run.chatter[st.i]
        + 0.12 * (hair - 0.5)
        + run.blade_tilt * _clip(s.ucoord[wsl], -1.0, 1.0)
    )

    # 4) Wet paint standing above the blade is bulldozed ahead by the flow (5);
    #    near the edges of the mark, some of the paint that was lying there spills
    #    out sideways (lips; the pile the blade carries rides along with it). Nicks
    #    and flex let streaks of it slip under the blade and stay. A comb's teeth
    #    push what they cut into ridges between them.
    # all round, left, right
    ridges = (np.zeros(PACKED), np.zeros(PACKED), np.zeros(PACKED))
    flowing = bt.clumps == 0
    tv = top[..., IV].astype(np.float64)
    excess = _clip(floor + tv + und[..., IV] - level, 0.0, None)
    excess *= 1.0 - np.exp(-4.0 * expo_c * (0.15 + 1.7 * hair**2))
    if flowing:
        half = max(st.r * st.lmax, 1e-6)  # the mark's half-width, px
        # distance in from the mark's edge, px
        from_edge = half - np.abs(s.lat_px[wsl])
        near = _smoothstep(_KNIFE_SPILL_PX, 0.0, from_edge)
        wet_sl = (
            slice(st.cy0 - run.wet_y0, st.cy0 - run.wet_y0 + st.ch),
            slice(st.cx0 - run.wet_x0, st.cx0 - run.wet_x0 + st.cw),
        )
        excess = _KNIFE_SPILL * near * np.minimum(excess, run.wet_before[wet_sl])
    if excess.max() > 1e-3:
        moved = eng.take(canvas.top, canvas.und, st.cy0, st.cx0, excess, tv)
        if flowing:
            eng.add_sides(ridges, moved, s.lat_px[wsl], 1.0)
        else:
            fresh = (
                s.between
                * _smoothstep(-st.ds - 1.0, 0.0, s.along - s.c)
                * _smoothstep(1.0, -1.0, s.along - s.c - s.h)
            )[wsl]
            eng.spread(
                run.cv,
                run.pend,
                run.br,
                st.cy0,
                st.cx0,
                fresh,
                moved.sum(axis=(0, 1)),
                np.zeros(fresh.shape),
                run.stroke_id,
                False,
            )
    # 5) The blade drags the wet paint it touches, as a thin-film flow: between
    #    the moving blade and the still canvas the paint shears, so paint near
    #    the blade travels almost with it and paint at the canvas hardly moves;
    #    what stands above the blade is bulldozed ahead of it. Each wet layer
    #    moves at its own speed, so their order is kept (white dragged over
    #    purple stays on top). The blade glides on its own fresh paint.
    if flowing:
        eng.flow_under_blade(
            run.cv,
            st.cy0,
            st.cx0,
            run.stroke_id,
            floor,
            level,
            s.aa[wsl],
            s.along[wsl],
            s.c[wsl],
            s.h[wsl],
            st.ds,
            st.tx,
            st.ty,
            _clip((0.5 + 0.45 * p) * (0.6 + 0.8 * hair), 0.0, 1.0),
            _VISCOUS,
        )
    eng.squeeze_out(run.br, run.K, st, col_vol, col_load, hold, ridges, run.col_u)
    grid, c_tab, h_tab = s.tables
    eng.lay_ridges(
        run.cv, run.pend, run.br, run.K, st, ridges, grid, c_tab, h_tab, _NO_TEX2
    )

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
        gap_px = _clip(level - surf, 0.0, None)
        film = (
            eng.at_pixels(s.taps, tip_v + brush.reservoir[:, IV], st.ch, st.cw)
            * COLUMNS
            / area
        )
        reach_f = _smoothstep(1.3, 0.7, gap_px / np.maximum(film, 1e-9))
        # Barely touching, the blade skims: it fills less of the gap under it.
        skim = float(_smoothstep(_KNIFE_SKIM, 0.05, p))
        # Each step fills the share of the gap left that it passes over: as the
        # gap shrinks step by step, how full a pixel ends up depends only on
        # how much of a full pass went over it, not on how the stroke is cut
        # into steps. (Filling a fixed share of each step's fill instead, as a
        # skimming blade did, a pixel passed over in two steps got more than
        # one passed over in one: a sawtooth along a pointed or rounded blade
        # end, where the blade is shorter along the stroke than a step.) How
        # much a skimming blade fills, and how much its bead leaves, over a
        # whole pass depends on how long the blade is along the stroke: as
        # much as its nominal steps would have, the same on a curve or a
        # pointed end (n: the blade's length along the stroke over a nominal
        # step, as in _engine.own_lift).
        n = 1.0
        if skim > 0:
            n = np.maximum(0.4 + 2.0 * s.h[wsl] / run.K.spacing, 1.0)
            per_step = 1.0 - (1.0 - skim) * (1.0 - np.exp(-4.0 / n))
            k_fill = -n * np.log(np.maximum(per_step, 1e-12))
        else:  # (the same: per_step is e^(-4/n))
            k_fill = 4.0
        d = gap_px * reach_f * (1.0 - np.exp(-k_fill * expo_c))
        if skim > 0 and bt.clumps == 0:
            bead = np.minimum(film, 1.2)
            touch = _clip((surf - (level - bead)) / np.maximum(bead, 1e-6), 0.0, 1.0)
            # the bead is uneven and the paint below still follows the threads
            # of the canvas: barely touching, it catches in streaks along the
            # blade's nicks and on the thread tops, the fewer the lighter the touch
            grain = 0.55 * hair + 0.45 * canvas._tooth[sl]
            thr = _KNIFE_TOUCH - 2.2 * p
            touch = touch * _smoothstep(thr - 0.1, thr + 0.1, grain)
            # the bead leaves paint wherever it passes, in proportion to how
            # much of a pass this step is (for the same reason)
            pass_bead = n * (1.0 - np.exp(-4.0 / n))
            d += skim * _KNIFE_BEAD * pass_bead * bead * touch**0.7 * expo_c
        if bt.clumps == 0:
            # Near its two ends the blade presses paint out sideways, more the
            # harder it is pressed: it stands in a bead on the edge of the slab
            # and beyond each end of the blade.
            lat = s.lat_px[wsl]
            from_edge = max(st.r * st.lmax, 1e-6) - np.abs(lat)
            near = _smoothstep(_KNIFE_END_PX, 0.0, from_edge)
            # (skimming, the blade spreads no film to press out; it takes a slab
            # of some thickness to stand in a bead)
            # (as much as the slab is thick, over each step's share of a pass:
            # not what this step filled, which wavers from step to step with
            # the blade's chatter and would come out as a bead at every step)
            d_end = (
                ride
                * expo_c
                * (_KNIFE_END * (0.8 + p) * (1.0 - skim))
                * near
                * _smoothstep(0.5, 2.0, ride)
            )
            if d_end.max() > 1e-6:
                _press_out(run, st, s.taps, lat, s.along[wsl], d_end)
        eng.lay(
            run.cv,
            run.pend,
            run.br,
            s.taps,
            st.cy0,
            st.cx0,
            d,
            s.cover[wsl],
            run.stroke_id,
            True,
            False,
            expo_c,
            False,
        )
    # A knife lifted off thick paint pulls it up into little peaks and torn ridges.
    if eff["lift"] > 0 and st.i == len(run.pts) - 1:
        under = (s.aa * _clip((s.h - np.abs(s.rel)) / 1.5, 0.0, 1.0))[wsl]
        # most along the trailing edge, the last of the blade to leave the paint
        under = under * (0.3 + 0.7 * _smoothstep(0.3, -0.5, (s.rel / s.h)[wsl]))
        _lift_off(
            canvas,
            sl,
            under,
            eff["lift"] * (0.5 + 0.8 * body),
            max(2.5, 0.08 * st.Wd),
            np.degrees(np.arctan2(st.ty, st.tx)),
            run.rng,
        )

    # 7) A blade only shears the paint it cuts through: riding within the top
    #    layer it leaves the one below alone; cutting into it, it folds the two
    #    into each other in streaks (marbling), not evenly.
    m = 1.0 - np.exp(-eff["mixing"] * 3.0 * p * p * expo_c * s.streak[wsl] * f_mix)
    into_under = _clip(floor + und[..., IV] + 0.5 - level, 0.0, 1.0)
    m = _clip(m * (0.1 + 1.8 * hair) * into_under, 0.0, 1.0)
    eng.stir(run.cv, st.cy0, st.cx0, m)


# ----------------------------------------------------------------------------
# helpers


def _bilinear(grid, y, x):
    """Sample ``grid`` at fractional indices (y, x) (inside its bounds)."""
    y0, x0 = y.astype(int), x.astype(int)
    fy, fx = y - y0, x - x0
    return (grid[y0, x0] * (1 - fx) + grid[y0, x0 + 1] * fx) * (1 - fy) + (
        grid[y0 + 1, x0] * (1 - fx) + grid[y0 + 1, x0 + 1] * fx
    ) * fy


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


_clip = np.clip


def _smoothstep(e0, e1, x):
    t = _clip((x - e0) / (e1 - e0), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def _use_paint_off_canvas(brush, Wd, ds, p):
    """Painting past the edge of the canvas still uses up paint."""
    total = brush.volume
    if total <= 0:
        return
    used = eng.deposit_thickness(total / (Wd * Wd)) * (0.5 + p) * ds * Wd * 0.9
    g = max(1.0 - used / total, 0.0)
    brush.tip *= g
    brush.reservoir *= g


def _press_out(run: _Run, st, taps, lat, along_px, d_end) -> None:
    """Paint pressed out at a blade's ends: ``d_end`` per pixel near each end
    (over the step's part of the canvas) moves out past the end of the blade
    to within _KNIFE_END_OUT px past the end of the blade, as it was spread (so
    it is as even along the stroke as the slab itself), smeared along the
    stroke over two steps (the paint escaping at the end of the blade runs
    together into one bead, whatever was laid in which step), and stands
    there in a bead against the slab's edge. (Landing from where it was
    spread to just beyond the end of the blade: on what the blade has filled,
    it would change what later steps fill.)"""
    H, W = run.canvas.height, run.canvas.width
    m = int(math.ceil(_KNIFE_END_OUT + _KNIFE_END_PX + 2.0 * st.ds)) + 1
    y0, x0 = max(st.cy0 - m, 0), max(st.cx0 - m, 0)
    y1, x1 = min(st.cy0 + st.ch + m, H), min(st.cx0 + st.cw + m, W)
    wl, wr = eng.splat_out(
        d_end,
        lat,
        along_px,
        st.cy0 - y0,
        st.cx0 - x0,
        y1 - y0,
        x1 - x0,
        st.tx,
        st.ty,
        st.ds,
        st.arc_i,
        run.K.arc_end,
        max(st.r * st.lmax, 1e-6),
        _KNIFE_END_OUT,
    )
    for side, w in ((-1.0, wl), (1.0, wr)):
        src = np.where(np.sign(lat) == side, d_end, 0.0)
        if not src.max() > 1e-9:
            continue
        pk = eng.take_tips(run.brush.tip, taps, src)
        eng.spread(
            run.cv,
            run.pend,
            run.br,
            y0,
            x0,
            w,
            pk,
            _smoothstep(0.1, 0.6, w),
            run.stroke_id,
            False,
        )


def _dab_lift(run: _Run) -> None:
    """A loaded brush lifted straight off the mound of paint it pressed out
    pulls it up into a peak or two, a little off centre the way it leaves
    (volume-conserving): more so for thick, stiff paint and stiff hair."""
    canvas, lift = run.canvas, float(run.eff["lift"])
    if lift <= 0:
        return
    (ax, ay), (cx, cy) = run.pts[0], run.pts[-1]
    r = float(run.radii.max())
    R = r * run.reach + 1.0
    H, W = canvas.height, canvas.width
    x0, x1 = max(int(cx - R), 0), min(int(cx + R) + 2, W)
    y0, y1 = max(int(cy - R), 0), min(int(cy + R) + 2, H)
    if x0 >= x1 or y0 >= y1:
        return
    sl = (slice(y0, y1), slice(x0, x1))
    own = canvas.top_id[sl] == run.stroke_id
    top = canvas.top[sl]
    V = np.where(own, top[..., IV], 0.0).astype(np.float64)
    if V.sum() <= 1e-9:
        return
    thick = float(V.sum() / max(float((V > 0.15).sum()), 1.0))
    body = float(np.where(own, top[..., IB], 0.0).sum() / V.sum())
    amp = 1.2 * lift * float(_smoothstep(2.0, 6.0, thick) * _smoothstep(0.3, 0.8, body))
    if amp <= 0:
        return
    rng = run.rng
    yy, xx = np.mgrid[y0:y1, x0:x1] + 0.5
    # the way the brush leaves (it is nudged from the first point to the last)
    d = np.array([cx - ax, cy - ay])
    d = d / max(float(np.hypot(*d)), 1e-9)
    peak = np.zeros_like(V)
    for k in range(int(rng.integers(1, 4))):
        off = (0.15 + 0.2 * rng.random()) * r * d + rng.normal(0.0, 0.15 * r, 2)
        size = (0.18 + 0.12 * rng.random()) * r
        weight = 1.0 if k == 0 else 0.3 + 0.4 * rng.random()
        dist2 = (xx - cx - off[0]) ** 2 + (yy - cy - off[1]) ** 2
        peak += weight * np.exp(-dist2 / (2.0 * size * size))
    peak /= max(float(peak.max()), 1e-9)
    m = (V * peak).sum() / V.sum()
    fac = _clip(1.0 + 2.0 * amp * (peak - m), 0.05, None)
    fac *= V.sum() / max(float((V * fac).sum()), 1e-12)
    fac = np.where(own, fac, 1.0)
    top *= fac.astype(np.float32)[..., None]


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
    fac = _clip(1.0 + 2.0 * amp * under * (crest - m), 0.05, None)
    fac *= V.sum() / max((V * fac).sum(), 1e-12)
    top *= fac.astype(np.float32)[..., None]
