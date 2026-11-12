"""A stroke: one pass of a tool along a path, and the simulation that
paints it.

At every step along the stroke a brush pushes some of the wet paint under it
forward (drag), its tips lift some of it (pickup) and lay a mix of what they
hold, and they are refilled from the belly. Paint picked up from the canvas
is therefore smeared forward and fades out over a short distance, while some
of it works its way into the belly (a dirty brush). A knife holds its paint on
the blade: it lays it as a slab, at a height set by pressure, and drags the
wet paint under it along as a thin film rather than picking it up (see
:mod:`oilpaint._knife`).

On the canvas, each pixel has the dry paint (or the ground) plus two wet
sublayers: ``und`` (older wet paint) and ``top`` (the paint of the most recent
strokes). Brush pressure stirs the two together; that is the difference
between laying color *over* wet paint and churning it *into* wet paint.

This module prepares each step of a stroke; the work on the pixels under the
tool is compiled (numba), in :mod:`oilpaint._engine`.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import NamedTuple, Protocol

import numpy as np

from . import _engine as eng
from ._dab import dab_pattern
from ._engine import COLUMNS, IB, IV, PACKED
from ._filters import _smoothstep
from ._geometry import (
    _NO_TABLES,
    _chord_table,
    _footprint_area,
    _frozen,
    _signed_floor,
    _Step,
    _step_slices,
)
from ._hair import _NICK_WANDER, _NO_TEX, _NO_TEX2, _hair_field
from ._knife import _before_the_blade, _BladeStart, _knife_step, _knife_taps
from .brush import (
    _COL_X,
    _PHYSICS,
    _RING_X,
    Brush,
    BrushType,
    StrokePaint,
    _check_seed,
    _check_taper,
    _check_tool,
    _is_number,
    _loads,
    _number,
    _physics,
    load_paint,
)
from .noise import value_noise_1d
from .spline import Path, Profile, as_path, evaluate_profile
from .surface import Surface

# --- lifting a brush off (see _Run._lift) ------------------------------------------
# Over how much of the end (px, for a brush 36 px wide, at most) a stiff
# brush's clumps leave the paint one after another, and how far they pull
# thick paint up.
_LIFT_LEN = 21.0
_LIFT_CREST = 4.0

# --- the paint settling after a stroke (see _Run.settle) ---------------------------
# How long wet paint flows where it is steeper than it holds (passes over the
# part of the canvas the stroke reached), and how long the crests of the
# stroke's own thick paint round off.
_LEVEL_PASSES = 12
_SOFTEN_PASSES = 2


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
        1 = stirs in strongly). Scaled by pressure. Pressed hard through
        thick wet paint, it also sets how far the old paint folds up over the
        new between the grooves the brush cuts (0: not at all, the stroke is
        ploughed but not marbled).
    pickup:
        How much wet paint the brush lifts and carries along (0..2).
        Brushes only: a knife drags wet paint instead (a thin-film flow).
    drag:
        How much wet paint the brush pushes along with it (0..2). Brushes only.
    streaks:
        Strength of the brush-hair streak texture (0..2).
    smear:
        How long paint picked up from the canvas persists on the brush tip,
        roughly in brush widths.
    blending:
        How readily the paints on the brush merge into each other during the
        stroke (see :attr:`BrushType.blending`).
    taper:
        Fraction of the stroke length over which the brush is lowered at the
        start and lifted at the end (narrower, feathered ends). A number from
        0 to 1, or a (start, end) pair of them. 0 gives full ends.
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


class _Relief(NamedTuple):
    """The surface under the wet paint, as a stroke meets it: the solid
    surface (the weave and dried paint, in units of paint thickness), what a
    nearly dry brush catches on (0..1), how far each point lies below the
    surface around it, and where a blade comes to rest on it (flexible,
    stiff)."""

    floor: np.ndarray
    tooth: np.ndarray
    hollows: np.ndarray
    peaks_fine: np.ndarray
    peaks_wide: np.ndarray


class _Canvas(Protocol):
    """What a stroke paints on (a :class:`oilpaint.Canvas`), and all a
    stroke uses of it. It reads the size, the surface and the dried paint's
    height (``dry_height``), and the volume of the wet paint (``top_v``,
    ``und_v``); it changes the wet paint in place (``top``, ``und``, their
    coverage ``top_a``, ``und_a`` and the stroke ids ``top_id``; see
    :mod:`oilpaint._engine`). The canvas hands it its random numbers, its id
    and the relief under the paint (maps it only reads, the same until the
    paint dries)."""

    width: int
    height: int
    surface: Surface
    top: np.ndarray
    und: np.ndarray
    top_a: np.ndarray
    und_a: np.ndarray
    top_id: np.ndarray
    dry_height: np.ndarray

    @property
    def top_v(self) -> np.ndarray: ...  # (views of the volume in top, und)

    @property
    def und_v(self) -> np.ndarray: ...

    def _stroke_rng(self, seed) -> np.random.Generator: ...

    def _new_stroke_id(self) -> int: ...

    def _relief(self) -> _Relief: ...

    def _weave_hollows(self) -> np.ndarray: ...

    def _beyond_weave(self) -> float: ...


def apply_stroke(canvas: _Canvas, stroke: Stroke) -> Brush:
    """Run the stroke simulation on ``canvas``. Returns the brush used.

    The tool moves along the path in short straight segments. For every pixel
    we compute its *exposure* during a segment: the fraction of a full pass
    of the footprint over that pixel that happens during this segment (exact
    for a convex footprint sweeping a straight segment). Exposures over a
    whole pass add up to the tool's lateral profile, so the result is smooth
    and does not depend on the step size. Each segment is then one step of
    :func:`_brush_step` or :func:`_knife_step`.
    """
    run = _Run.begin(canvas, stroke)
    if run.bt.rigid:
        for st, tables in run.steps():
            s = run.footprint(st, tables, off=True)
            if s is not None:
                _knife_step(run, s)
    else:
        y0, y1, x0, x1 = run.bounds()
        if y0 < y1 and x0 < x1:
            run.pend = _pending(y0, y1, x0, x1)
        for st, tables in run.steps():
            _brush_step(run, st, tables)
    eng.settle_pending(run.cv, run.pend, run.stroke_id)
    eng.fold_pending(run.cv, run.pend)
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
    paint stirs into the rest. It also keeps how far the wet paint the brush
    meets at each pixel can be ploughed (see _engine._meet), and how far the
    old paint there folds up over the new when the stroke has passed (see
    _engine.fold_pending). Returns an
    :class:`_engine.Pending`, as :func:`_engine.deposit` takes it."""
    h, w = y1 - y0, x1 - x0
    where = [0] * 14
    where[eng.PEND_ON] = 1
    where[eng.PEND_Y0], where[eng.PEND_X0] = y0, x0
    where[eng.PEND_H], where[eng.PEND_W] = h, w
    # (the rows and columns where any is held: none yet)
    where[eng.PEND_ROW0], where[eng.PEND_ROW1] = h, 0
    where[eng.PEND_COL0], where[eng.PEND_COL1] = w, 0
    # (where any old paint is to fold up: none yet)
    where[eng.PEND_FOLD_ROW0], where[eng.PEND_FOLD_ROW1] = h, 0
    where[eng.PEND_FOLD_COL0], where[eng.PEND_FOLD_COL1] = w, 0
    return eng.Pending(
        paint=np.zeros((h, w, PACKED), np.float32),
        cover=np.zeros((h, w), np.float32),
        reach=np.zeros((h, w), np.float32),
        held=np.zeros((h, w), bool),
        where=np.array(where, np.int64),
        met=np.full((h, w), -1.0, np.float32),
        folds=np.zeros((h, w), np.float32),
    )


# (no paint held: knives lay theirs at once. Nothing is written to it, but it
# stays writable: the kernels that take it write to the paint held, and numba
# does not compile a write to a read-only array.)
_NO_PENDING = eng.Pending(
    paint=np.zeros((1, 1, PACKED), np.float32),
    cover=np.zeros((1, 1), np.float32),
    reach=np.zeros((1, 1), np.float32),
    held=np.zeros((1, 1), bool),
    where=np.zeros(14, np.int64),
    met=np.full((1, 1), -1.0, np.float32),
    folds=np.zeros((1, 1), np.float32),
)


class _Samples(NamedTuple):
    """A stroke's path sampled into steps: the points, and the tool's width
    (px), pressure, angle and roll (radians) at each."""

    pts: np.ndarray
    widths: np.ndarray
    pressure: np.ndarray
    angles: np.ndarray
    rolls: np.ndarray
    spacing: float  # the nominal step (px)
    w_typ: float  # the stroke's typical width (px)
    is_dab: bool  # a single point, pressed down with a tiny movement


def _circular(bt: BrushType) -> bool:
    """A round footprint: the same whichever way it turns."""
    return bt.shape == "round" and abs(bt.aspect - 1.0) < 1e-6


def _sample_stroke(path: Path, stroke: Stroke, bt: BrushType, eff: dict) -> _Samples:
    """The stroke's path cut into steps about a quarter of its width long
    (shorter where an elongated footprint turns), with the width (tapered at
    the ends), pressure, angle and roll at each point. Raises ValueError if a
    function of u gives a value that is not finite."""
    n_probe = max(path.n_knots, 2) * 8
    w_probe = evaluate_profile(
        eff["width"],
        np.linspace(0, 1, n_probe),
        np.linspace(0, max(path.n_knots - 1, 0), n_probe),
        path.n_knots,
    )
    w_typ = max(float(np.mean(w_probe)), 1.0)
    spacing = float(np.clip(0.25 * w_typ, 0.5, 8.0))
    taper = eff["taper"]
    t0, t1 = (taper, taper) if isinstance(taper, (int, float, np.number)) else taper

    def sample(spacing):
        pts, _, u, knot = path.sample(spacing)
        widths = evaluate_profile(eff["width"], u, knot, path.n_knots)
        if t0 > 0:
            widths = widths * (0.25 + 0.75 * _smoothstep(0.0, t0, u))
        if t1 > 0:
            widths = widths * (0.25 + 0.75 * _smoothstep(1.0, 1.0 - t1, u))
        widths = np.maximum(widths, 1.0)
        pressure = np.clip(
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
    if not _circular(bt) and len(pts) > 2:
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
    is_dab = len(pts) == 1
    # a dab: press the brush down with a tiny movement (small next to its narrow
    # side)
    if is_dab:
        nudge = max(0.05 * widths[0] * min(1.0, bt.aspect), 0.25)
        pts = np.vstack([pts[0] - [nudge, 0], pts[0] + [nudge, 0]])
        widths = np.repeat(widths, 2)
        pressure = np.repeat(pressure, 2)
        angles = np.repeat(angles, 2)
        rolls = np.repeat(rolls, 2)
    return _Samples(pts, widths, pressure, angles, rolls, spacing, w_typ, is_dab)


class _Textures(NamedTuple):
    """How a stroke wavers along its length (one value per point)."""

    wobble: np.ndarray  # the width's slow wobble, -0.5..0.5
    chatter: np.ndarray  # a knife's height, wavering as it is pushed
    lip: np.ndarray  # how much the lips squeezed out along it swell
    blade_tilt: float  # a knife pressed harder at one end of its blade
    dab: np.ndarray  # the bristle tips a dab prints (see _dab)


def _stroke_textures(rng, bt: BrushType, smp: _Samples, arc) -> _Textures:
    """The first textures of a stroke, drawn from its random numbers in this
    order (see :class:`_Run`)."""
    pts, w_typ, spacing = smp.pts, smp.w_typ, smp.spacing
    # A slow wobble of the width.
    wobble = (
        value_noise_1d(len(pts) + 2, max(2.0, 0.8 * w_typ / spacing), rng)[: len(pts)]
        - 0.5
    )
    # a knife's height wavers as it is pushed along (faint ripples across the
    # slab: chatter), and a hand never presses both ends of the blade quite
    # equally (the slab is thicker towards one side); the lips squeezed out
    # along a stroke swell and break
    period = max(0.4 * w_typ, 2.0)  # ripples about half a blade width apart
    ripple = value_noise_1d(int(arc[-1] / period) + 4, 1.0, rng)
    chatter = np.interp(arc / period, np.arange(len(ripple)), ripple) - 0.5
    swell = value_noise_1d(int(arc[-1] / (2.0 * period)) + 4, 1.0, rng)
    lip = np.interp(arc / (2.0 * period), np.arange(len(swell)), swell)
    blade_tilt = float(np.clip(rng.normal(0.0, 0.08), -0.2, 0.2))
    # the bristle tips a dab prints (see _dab)
    dab = _frozen(dab_pattern(bt, 0.5 * w_typ, rng)) if smp.is_dab else _NO_TEX2
    return _Textures(wobble, chatter, lip, blade_tilt, dab)


def _radii(smp: _Samples, wobble, bt: BrushType) -> tuple[np.ndarray, np.ndarray]:
    """The brush's half-width at every sample: the bristle bundle's size with
    a slow wobble (nominal), and that as far as the hairs splay under pressure
    (a pointed round paints thin with a light touch)."""
    radii = 0.5 * smp.widths * (1.0 + 0.06 * wobble)
    # the bristle bundle's size; pressure only splays it
    nominal = radii.copy()
    if bt.spread > 0:
        radii = radii * (
            (1.0 - bt.spread)
            + bt.spread * (np.maximum(smp.pressure, 0.02) / 0.5) ** 0.8
        )
    return nominal, radii


def _lift_reach(bt: BrushType, eff: dict, smp: _Samples, arc, radii):
    """Lifting a brush off at the end (see _Run._lift): how far before the
    end it starts, at most (px), and the arc length from which a step can be
    lifting off."""
    reach = (
        0.0
        if smp.is_dab or bt.rigid or not bt.deposits
        else _LIFT_LEN * eff["lift"] * 1.5 * (2.0 * float(radii.max()) / 36.0) ** 0.3
    )
    pts = smp.pts
    # (no step further than this from the end can be lifting off)
    steps = np.linalg.norm(np.diff(pts, axis=0), axis=1) if len(pts) > 1 else [0.0]
    lift_from = (
        float(arc[-1]) - reach - float(np.max(steps)) - 2.0 * float(radii.max()) - 1.0
    )
    return reach, lift_from


class _Segment(NamedTuple):
    """Step ``i`` of a stroke, from point i-1 to point i."""

    i: int
    ax: float  # where it starts
    ay: float
    cx: float  # where it ends
    cy: float
    tx: float  # the way the tool moves
    ty: float
    ds: float  # how far (px)
    p: float  # pressure
    r0: float  # the tool's half-width at the start, at the end and on average
    r1: float
    r: float
    Wd: float  # its width
    phi: float  # its long side from the cross direction (radians)
    side: float  # -1 for a brush turned past 90 degrees: its sides swapped


class _Window(NamedTuple):
    """The pixels around a step's swept segment, x0..x1 and y0..y1 (which may
    stick out of the canvas), and their part on the canvas."""

    x0: int
    y0: int
    x1: int
    y1: int
    cx0: int
    cy0: int
    cx1: int
    cy1: int

    @property
    def on_canvas(self) -> bool:
        return self.cx0 < self.cx1 and self.cy0 < self.cy1


def _arc_length(pts: np.ndarray) -> np.ndarray:
    """The distance along the path (px) at each of its points ``pts``."""
    return np.concatenate(
        [[0.0], np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))]
    )


class _Shape(NamedTuple):
    """The footprint of a tool, as a stroke works with it."""

    circle: bool  # round: the same whichever way it turns
    aspect: float  # its thickness over its width (1 for a circle)
    reach: float  # how far it reaches from its centre, in half-widths
    area_unit: float  # its area, in half-widths squared


def _shape(bt: BrushType) -> _Shape:
    """The footprint of tool ``bt``."""
    circle = _circular(bt)
    aspect = 1.0 if circle else bt.aspect
    return _Shape(
        circle,
        aspect,
        1.0 if circle else float(np.hypot(1.0, aspect)),
        _footprint_area("round" if circle else bt.shape, aspect),
    )


# (where the hair pattern of a stroke with none lies: see _stroke_hair)
_NO_HAIR = (0.0, 0.0, 1.0, 1.0)


def _stroke_hair(
    bt: BrushType, w_typ: float, half: float, r_max: float, reach: float, arc, rng
):
    """The brush's hair over the stroke: where its bristle clumps run along
    it, or a blade's nicks (which score nearly straight lines, coming and
    going as grit catches and paint closes over them), and where that lies
    (see :func:`_hair_field`), from a little before its start to a little
    past its end. ``half`` is the bristle bundle's largest half-width (px),
    ``r_max`` the largest it is splayed to under pressure."""
    pad = r_max * reach + 4.0
    return _hair_field(
        bt,
        w_typ,
        half,
        -pad,
        float(arc[-1]) + pad,
        rng,
        wander=_NICK_WANDER if bt.rigid else 1.0,
    )


def _stroke_consts(
    bt: BrushType,
    eff: dict,
    brush: Brush,
    surf: Surface,
    *,
    stroke_id: int,
    circle: bool,
    is_dab: bool,
    margin: float,
    room: float,
    arc_end: float,
    hair_at: tuple[float, float, float, float],
    spacing: float,
    max_pressure: float,
    beyond_weave: float,
):
    """The settings of a stroke with tool ``bt`` (settings ``eff``, see
    :func:`_resolve`) on a canvas of surface ``surf`` whose deepest hollow
    lies ``beyond_weave`` deeper than the weave's own (see
    ``Canvas._beyond_weave``), as the engine takes them (an
    :class:`_engine.StrokeConsts`)."""
    return eng.StrokeConsts(
        stroke_id=int(stroke_id),
        circle=bool(circle),
        is_dab=bool(is_dab),
        ring=bool(brush.ring),
        rigid=bool(bt.rigid),
        deposits=bool(bt.deposits),
        margin=float(margin),
        room=float(room),
        # how much the surface texture (vs. the hair pattern) decides
        # where a dry brush catches
        tooth_w=float(np.clip(0.3 + (surf.tooth - 0.5), 0.0, 0.85)),
        drag=float(eff["drag"]),
        pickup=float(eff["pickup"]),
        streaks=float(eff["streaks"]),
        mixing=float(eff["mixing"]),
        smear=float(eff["smear"]),
        blending=float(eff["blending"]),
        rough=float(eff["rough"]),
        ridge=float(eff["ridge"]),
        # (0 to 1: beyond, a soft brush would give back more than it took)
        stiffness=float(np.clip(bt.stiffness, 0.0, 1.0)),
        capacity=float(bt.capacity),
        dig=float(bt.dig),
        softness=float(bt.softness),
        slant=float(bt.slant),
        clumps=float(bt.clumps),
        # soft brushes fade in and out over this many widths
        feather=float(bt.softness * 3.0),
        # soft brushes are lowered and lifted gently, so their ends feather
        end_k=1.0 if is_dab else max(0.0, 1.0 - 1.5 * float(bt.softness)),
        surface_acts=bool(surf.affects_paint),
        slickness=float(surf.slickness),
        absorbency=float(surf.absorbency),
        surface_tooth=float(surf.tooth),
        beyond_weave=float(beyond_weave),
        arc_end=float(arc_end),
        hair_s0=hair_at[0],
        hair_x0=hair_at[1],
        hair_ds=hair_at[2],
        hair_dx=hair_at[3],
        spacing=float(spacing),
        # the brush's own size: how wide it paints on average (px)
        bundle=float(brush.width),
        # whether it may plough anywhere: it has bristles (BrushTypes pickled
        # before the property existed do), and is pressed hard enough
        # somewhere along it (see _engine.PLOUGH_P0; a stroke that is, may
        # come back hard over paint it first met lightly)
        ploughs=bool(getattr(bt, "ploughs", True)) and max_pressure > eng.PLOUGH_P0,
    )


class _Run:
    """One stroke in progress: the tool, the sampled path, and the textures
    and constants that stay the same along the stroke.

    Every texture of a stroke is drawn from its own random numbers (one draw
    from the canvas's seeds them, unless the stroke has a seed), always in
    this order: the wobble of the width, the chatter, the swell of the lips
    and the tilt of a blade (drawn for brushes too, which do not use them),
    a dab's bristle tips, a knife's colour bands, the hair, how ragged the
    end is, and at the end the peaks pulled up as the tool lifts off. Leaving
    one out, or drawing one earlier, would change every stroke.

    A run is made by :meth:`begin`, which sizes and loads the brush first;
    the constructor then works out the rest, takes the stroke's id from the
    canvas and starts the brush's hair dry (``brush.wet``)."""

    blade: _BladeStart  # (knives only)
    # (knives only) the slab laid past the edge of the canvas, in tiles (see
    # _knife._laid_off)
    knife_off: dict[tuple[int, int], np.ndarray] | None = None
    # (brushes only) what stays the same along the stroke, as the engine's
    # brush_step takes it (see _engine.brush_kit and brush_kit below). It
    # holds K, tex, tex2, cv, aux, br and col_u, all fixed once __init__
    # returns, and pend, the one of them that may be rebound: its setter
    # clears the kit, which is made again at the next step.
    _kit: tuple | None = None

    @classmethod
    def begin(cls, canvas: _Canvas, stroke: Stroke) -> _Run:
        """Set ``stroke`` going on ``canvas``: its random numbers, the tool,
        the path sampled, and the brush sized for the stroke and loaded with
        its paint (a load it refuses raises before the stroke has an id)."""
        path: Path = as_path(stroke.path)
        rng = canvas._stroke_rng(stroke.seed)
        brush, bt, eff = _resolve(stroke)
        smp = _sample_stroke(path, stroke, bt, eff)
        # a load is measured in widths of the stroke (see Stroke.amount)
        brush.width = float(np.mean(smp.widths))
        load_paint(brush, stroke.paint, stroke.amount)
        return cls(canvas, rng, brush, bt, eff, smp)

    def __init__(
        self,
        canvas: _Canvas,
        rng,
        brush: Brush,
        bt: BrushType,
        eff: dict,
        smp: _Samples,
    ):
        self.canvas, self.rng = canvas, rng
        self.brush, self.bt, self.eff = brush, bt, eff
        pts, is_dab, w_typ = smp.pts, smp.is_dab, smp.w_typ
        self.pts, self.is_dab, self.w_typ = pts, is_dab, w_typ
        self.lays_paint = bool(bt.deposits and brush.volume > 0)
        arc = self.arc = _arc_length(pts)
        shape = _shape(bt)
        self.circle, self.aspect, self.reach, self.area_unit = shape
        self.stiff_blade = float(np.clip((bt.stiffness - 0.4) / 0.6, 0.0, 1.0))
        # where each column sits across the mark
        self.col_u = (_RING_X if brush.ring else _COL_X) * 2.0 - 1.0
        # squeezed-out ridges lie just outside the mark, as a rule within this
        # many px of it; a brush's side ridges within ``room`` px, which grows
        # with the width of the mark (see lay_ridges_off)
        spill = 0.12 * w_typ * (1.0 + bt.spread)
        self.margin = float(np.clip(spill, 0.6, 7.0)) + 1.0
        self.room = (
            self.margin if (bt.rigid or is_dab) else max(self.margin, spill + 1.0)
        )
        self.tex = _NO_TEX

        # (from here to lift_var, the stroke's textures are drawn from rng in
        # the order the class gives; nothing in between draws any)
        tex = _stroke_textures(rng, bt, smp, arc)
        self.chatter, self.blade_tilt = tex.chatter, tex.blade_tilt
        nominal, radii = _radii(smp, tex.wobble, bt)
        self.radii = radii
        self.lift_reach, self._lift_from = _lift_reach(bt, eff, smp, arc, radii)
        if bt.rigid:
            pad = float(radii.max()) * shape.reach + 3.0
            self.blade = _before_the_blade(canvas, rng, pts, pad, w_typ, arc)
        self.stroke_id = canvas._new_stroke_id()
        if is_dab:
            self.tex2, hair_at = tex.dab, _NO_HAIR
        else:
            self.tex2, hair_at = _stroke_hair(
                bt,
                w_typ,
                float(nominal.max()),
                float(radii.max()),
                shape.reach,
                arc,
                rng,
            )
        # how ragged this stroke's end is (each stroke ends its own way)
        self.lift_var = float(0.4 + 1.2 * rng.random()) if self.lift_reach > 0 else 1.0

        self.dry = False
        brush.wet = 0.0
        # the samples as plain floats, for the per-step arithmetic
        self._pts = [(float(x), float(y)) for x, y in pts]
        self._pressure, self._angles, self._rolls = (
            smp.pressure.tolist(),
            smp.angles.tolist(),
            smp.rolls.tolist(),
        )
        self._radii, self._nominal = radii.tolist(), nominal.tolist()
        self._arc, self._lip = arc.tolist(), tex.lip.tolist()

        # what the compiled engine works on
        self.cv = eng.Layers(
            top=canvas.top,
            und=canvas.und,
            top_a=canvas.top_a,
            und_a=canvas.und_a,
            top_id=canvas.top_id,
        )
        relief = self.relief = canvas._relief()
        beyond_weave = canvas._beyond_weave()
        self.aux = eng.Beneath(
            floor=relief.floor,
            tooth=relief.tooth,
            hollows=relief.hollows,
            dry_height=canvas.dry_height,
            # (read only where something may be out of reach: see
            # _engine._contact_catch_wet_dry)
            weave_hollows=(
                canvas._weave_hollows()
                if beyond_weave > eng.REACH_MIN
                else relief.hollows
            ),
        )
        self.br = eng.Load(tip=brush.tip, belly=brush.reservoir, heel=brush.heel)
        self.pend = _NO_PENDING
        self.K = _stroke_consts(
            bt,
            eff,
            brush,
            canvas.surface,
            stroke_id=self.stroke_id,
            circle=shape.circle,
            is_dab=is_dab,
            margin=self.margin,
            room=self.room,
            arc_end=float(arc[-1]),
            hair_at=hair_at,
            spacing=smp.spacing,
            max_pressure=float(np.max(smp.pressure)),
            beyond_weave=beyond_weave,
        )

    @property
    def pend(self) -> eng.Pending:
        """The paint the stroke holds back (see :func:`_pending`; a knife's
        holds none)."""
        return self._pend

    @pend.setter
    def pend(self, pend: eng.Pending) -> None:
        self._pend = pend
        self._kit = None

    def brush_kit(self) -> tuple:
        """What stays the same along the stroke, as the engine's brush_step
        takes it (see _engine.brush_kit): made at the first step."""
        if self._kit is None:
            self._kit = eng.brush_kit(
                self.K,
                self.tex,
                self.tex2,
                self.cv,
                self.aux,
                self.pend,
                self.br,
                self.col_u,
            )
        return self._kit

    # ------------------------------------------------------------------
    def steps(self):
        """The steps of the stroke, in order, each as :meth:`step` gives it.
        Painting past the edge of the canvas uses up paint all the same: a
        tool's steps off the canvas are taken as on a canvas large enough to
        hold them (see _engine.off_canvas). (Each step is worked out once the
        one before it has been painted: it starts from the brush as that left
        it.)"""
        for i in range(1, len(self._pts)):
            seg = self._segment(i)
            if seg is None:
                continue
            win = self._window(seg)
            # (off the canvas, the tool goes on as on a canvas large enough
            # to hold it: see _engine.off_canvas)
            yield self._step(seg, win)

    def step(self, i: int):
        """Step ``i`` of the stroke, from point i-1 to point i: its scalars (an
        :class:`_engine.Step`) and the footprint's chord tables, or None if
        the tool doesn't move or doesn't reach the canvas there. It changes
        nothing: the paint a brush uses up past the edge of the canvas is
        taken in :meth:`steps` only."""
        seg = self._segment(i)
        if seg is None:
            return None
        win = self._window(seg)
        return self._step(seg, win) if win.on_canvas else None

    def _segment(self, i: int) -> _Segment | None:
        """How the tool moves over step ``i`` (None if it doesn't)."""
        # (plain floats: this runs at every step)
        pts, pressure, angles, radii = (
            self._pts,
            self._pressure,
            self._angles,
            self._radii,
        )
        (ax, ay), (cx, cy) = pts[i - 1], pts[i]
        ds = math.hypot(cx - ax, cy - ay)
        if ds < 1e-9:
            return None
        tx, ty = (cx - ax) / ds, (cy - ay) / ds
        p = 0.5 * (pressure[i] + pressure[i - 1])
        r0, r1 = radii[i - 1], radii[i]
        r = 0.5 * (r0 + r1)
        # orientation of the brush's long side relative to the stroke's cross direction
        ang = 0.5 * (angles[i] + angles[i - 1])
        phi = ang if self.eff["follow_path"] else ang - math.atan2(tx, -ty)
        # a brush turned past 90 degrees has its sides swapped
        side = 1.0 if math.cos(phi) >= 0 else -1.0
        phi = (phi + math.pi / 2) % math.pi - math.pi / 2
        return _Segment(i, ax, ay, cx, cy, tx, ty, ds, p, r0, r1, r, 2.0 * r, phi, side)

    def _window(self, seg: _Segment) -> _Window:
        """The window around the swept segment, and its part on the canvas."""
        H, W = self.canvas.height, self.canvas.width
        ax, ay, cx, cy = seg.ax, seg.ay, seg.cx, seg.cy
        R = max(seg.r0, seg.r1) * self.reach + (3.0 if self.bt.rigid else 0.0)
        x0, x1 = math.floor(min(ax, cx) - R - 1), math.ceil(max(ax, cx) + R + 1) + 1
        y0, y1 = math.floor(min(ay, cy) - R - 1), math.ceil(max(ay, cy) + R + 1) + 1
        return _Window(x0, y0, x1, y1, max(x0, 0), max(y0, 0), min(x1, W), min(y1, H))

    def _step(self, seg: _Segment, win: _Window):
        """Step ``seg`` over the window ``win``, as :meth:`step` gives it."""
        i, p, r, Wd, side = seg.i, seg.p, seg.r, seg.Wd, seg.side
        x0, y0, cx0, cy0 = win.x0, win.y0, win.cx0, win.cy0
        if self.circle:
            tables, lmax, across, h_max, uscale = _NO_TABLES, 1.0, 1.0, 1.0, side
        else:
            grid, c_tab, h_tab, lmax, h_max = _chord_table(
                self.bt.shape, self.aspect, seg.phi
            )
            tables = (grid, c_tab, h_tab)
            sf = _signed_floor(math.cos(seg.phi), self.aspect)
            across, uscale = abs(sf), side / sf
        # On a thin brush one pixel spans several bristle columns: sample across
        # it so that every column reaches the canvas (else paint in the columns
        # between pixel centres would only come out by diffusion, and thin
        # strokes would run long).
        span = 1.0 / max(r * across, 1e-6)
        span_cols = span * COLUMNS * (0.25 if self.brush.ring else 0.5)
        n = len(self._pts)
        lift_len, lift_amp = (
            self._lift(i, p, Wd, seg.ds)
            if self.lift_reach > 0 and self._arc[i] >= self._lift_from
            else (0.0, 0.0)
        )
        overload = self._overload(i, p) if self.is_dab else 0.0
        pressure, nominal = self._pressure, self._nominal
        st = eng.Step(
            win=eng.Window(
                x0=x0,
                y0=y0,
                hh=win.y1 - y0,
                ww=win.x1 - x0,
                cy0=cy0,
                cx0=cx0,
                # (none of it, for a step wholly off the canvas)
                ch=max(win.cy1 - cy0, 0),
                cw=max(win.cx1 - cx0, 0),
                wy0=cy0 - y0,
                wx0=cx0 - x0,
            ),
            motion=eng.Motion(
                i=i,
                first=i == 1,
                last=i == n - 1,
                touchdown=i == 1 or self.is_dab,
                cx=seg.cx,
                cy=seg.cy,
                tx=seg.tx,
                ty=seg.ty,
                ds=seg.ds,
                arc_i=self._arc[i],
                lip=self._lip[i],
            ),
            hold=eng.Hold(
                p=p,
                p_now=pressure[i],
                p_before=pressure[i - 1],
                r0=seg.r0,
                r1=seg.r1,
                r=r,
                Wd=Wd,
                # how loaded the brush is is relative to its real size
                Wn=nominal[i - 1] + nominal[i],
                side=side,
                uscale=uscale,
                lmax=lmax,
                h_tab_max=h_max,
                span=span,
                nsub=int(min(max(math.ceil(span_cols / 2.0 - 1e-9), 1), 8)),
                rho=0.5 * (self._rolls[i] + self._rolls[i - 1]),
                splay=r / max(0.5 * (nominal[i - 1] + nominal[i]), 1e-9),
                lift_len=lift_len,
                lift_amp=lift_amp,
                overload=overload,
            ),
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
        loaded = eng.smoothstep(0.15, 0.6, load / hold)
        if loaded <= 0:
            return none
        length = arc[-1]
        short = eng.smoothstep(1.0, 3.0, length / Wd)
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
        L *= eng.smoothstep(0.1, 0.5, body)
        amp = (
            _LIFT_CREST
            * lift
            * eng.smoothstep(0.8, 3.0, thick)
            * eng.smoothstep(0.3, 0.8, body)
        )
        if L < 1.0:
            return none
        return float(L), float(amp)

    def footprint(self, st, tables, off: bool = False) -> _Step | None:
        """Where the tool touches the canvas while it moves along the step
        ``st`` (with the chord tables ``tables``, see :meth:`step`), or None
        if it touches nothing (with ``off``: nothing on or off the canvas).

        For every pixel we compute its *exposure* (see :func:`apply_stroke`),
        its anti-aliased coverage, and where it sits under the brush (across
        the bristles, along the motion); see :func:`_engine.footprint`. And
        which bristle columns are above each pixel."""
        F, hit = eng.footprint(st, self.K, self.tex, self.tex2, *tables)
        if not hit and not (off and eng._touches_off(st.win, F)):
            return None
        if self.bt.rigid:
            taps, taps_all = _knife_taps(self, st, F)
            return _Step(st, F, tables, taps, taps_all)
        u_px = F[eng.F_U][_step_slices(st)[1]]
        taps = eng.bristle_taps(
            u_px, st.hold.span, st.hold.nsub, self.brush.ring, st.hold.rho
        )
        return _Step(st, F, tables, taps)

    def bounds(self, ridges: float | None = None):
        """(y0, y1, x0, x1): the part of the canvas the stroke can reach,
        its ridges reaching ``ridges`` px past the mark (by default as far as
        they may: its room)."""
        canvas, pts = self.canvas, self.pts
        H, W = canvas.height, canvas.width
        out = self.room if ridges is None else ridges
        pad = float(self.radii.max()) * self.reach + out + 2.0
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
        # (as far as its ridges reached)
        reach = float(self.pend.where[eng.PEND_REACH])
        by0, by1, bx0, bx1 = self.bounds(max(self.margin, min(reach, self.room)))
        if bx0 < bx1 and by0 < by1:
            eng.level(
                canvas.top,
                canvas.und,
                canvas.dry_height,
                by0,
                by1,
                bx0,
                bx1,
                _LEVEL_PASSES,
            )
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
                _SOFTEN_PASSES,
            )
        if bx0 < bx1 and by0 < by1:
            eng.clear_negative(canvas.top, canvas.und, by0, by1, bx0, bx1)
            eng.fix_coverage(
                canvas.top, canvas.und, canvas.top_a, canvas.und_a, by0, by1, bx0, bx1
            )


# ----------------------------------------------------------------------------
# one step of a brush, one step of a knife


def _brush_step(run: _Run, st, tables) -> None:
    """One step of a brush (see :func:`_engine.brush_step`): it drags the wet
    paint under it along; refills its tips and picks up wet paint; combs
    through thick paint, pushing it aside into ridges; lays paint from its
    tips (a dry brush only on the tooth of the canvas); and stirs the fresh
    paint into the wet paint below."""
    run.brush.wet, run.dry = eng.brush_step(
        st, float(run.brush.wet), bool(run.dry), tables, run.brush_kit()
    )


# How far (in brush half-widths) the paint pulled up into a dab's peaks comes
# from (a Gaussian's sigma, see _engine.pull_up): about the size of a peak
# (0.18 to 0.3), so a peak is made mostly of the paint it stands in.
DAB_PULL = 0.3


def _dab_lift(run: _Run) -> None:
    """A loaded brush lifted straight off the mound of paint it pressed out
    pulls it up into a peak or two, a little off centre the way it leaves:
    more so for thick, stiff paint and stiff hair. The peaks are made of the
    paint around them, which falls a little (see _engine.pull_up): nothing
    is made or lost, pigment by pigment."""
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
    amp = (
        1.2 * lift * (eng.smoothstep(2.0, 6.0, thick) * eng.smoothstep(0.3, 0.8, body))
    )
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
    fac = np.clip(1.0 + 2.0 * amp * (peak - m), 0.05, None)
    fac *= V.sum() / max(float((V * fac).sum()), 1e-12)
    fac = np.where(own, fac, 1.0)  # (1: left alone)
    eng.pull_up(top, fac, DAB_PULL * r)
