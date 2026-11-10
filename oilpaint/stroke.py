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
from typing import NamedTuple

import numpy as np

from . import _engine as eng
from ._dab import dab_pattern
from ._engine import COLUMNS, IB, IV, PACKED
from ._geometry import (
    _NO_TABLES,
    _chord_table,
    _footprint_area,
    _signed_floor,
    _Step,
    _step_slices,
)
from ._hair import _NICK_WANDER, _NO_TEX, _NO_TEX2, _hair_field
from ._knife import _before_the_blade, _knife_step, _wandering_bands
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
    _clip,
    _is_number,
    _loads,
    _number,
    _physics,
    _smoothstep,
    load_paint,
)
from .noise import value_noise_1d
from .spline import Path, Profile, as_path, evaluate_profile

# --- lifting a brush off (see _Run._lift) ------------------------------------------
# Over how much of the end (px, for a brush 36 px wide, at most) a stiff
# brush's clumps leave the paint one after another, and how far they pull
# thick paint up.
_LIFT_LEN = 21.0
_LIFT_CREST = 4.0


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
    run = _Run.begin(canvas, stroke)
    if run.bt.rigid:
        for st, tables in run.steps():
            s = run.footprint(st, tables)
            if s is not None:
                _knife_step(run, s)
    else:
        y0, y1, x0, x1 = run.bounds()
        if y0 < y1 and x0 < x1:
            run.pend = _pending(y0, y1, x0, x1)
        for st, tables in run.steps():
            _brush_step(run, st, tables)
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
    spacing = float(_clip(0.25 * w_typ, 0.5, 8.0))
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
    blade_tilt = float(_clip(rng.normal(0.0, 0.08), -0.2, 0.2))
    # the bristle tips a dab prints (see _dab)
    dab = dab_pattern(bt, 0.5 * w_typ, rng) if smp.is_dab else _NO_TEX2
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


class _Run:
    """One stroke in progress: the tool, the sampled path, and the textures
    and constants that stay the same along the stroke.

    Every texture of a stroke is drawn from its own random numbers (one draw
    from the canvas's seeds them, unless the stroke has a seed), always in
    this order: the wobble of the width, the chatter, the swell of the lips
    and the tilt of a blade (drawn for brushes too, which do not use them),
    a dab's bristle tips, a knife's colour bands, the hair, how ragged the
    end is, and at the end the peaks pulled up as the tool lifts off. Leaving
    one out, or drawing one earlier, would change every stroke."""

    @classmethod
    def begin(cls, canvas, stroke: Stroke) -> _Run:
        """Set ``stroke`` going on ``canvas``: its random numbers, the tool,
        the path sampled, and the brush sized for the stroke and loaded with
        its paint (a load it refuses raises before the stroke has an id)."""
        path: Path = as_path(stroke.path)
        rng = np.random.default_rng(
            stroke.seed if stroke.seed is not None else canvas._rng.integers(1 << 31)
        )
        brush, bt, eff = _resolve(stroke)
        smp = _sample_stroke(path, stroke, bt, eff)
        # a load is measured in widths of the stroke (see Stroke.amount)
        brush.width = float(np.mean(smp.widths))
        load_paint(brush, stroke.paint, stroke.amount)
        return cls(canvas, rng, brush, bt, eff, smp)

    def __init__(self, canvas, rng, brush: Brush, bt: BrushType, eff: dict, smp):
        self.canvas, self.rng = canvas, rng
        self.brush, self.bt, self.eff = brush, bt, eff
        pts, is_dab, w_typ = smp.pts, smp.is_dab, smp.w_typ
        self.pts, self.is_dab, self.w_typ = pts, is_dab, w_typ
        self.lays_paint = bool(bt.deposits and brush.volume > 0)
        arc = self.arc = np.concatenate(
            [[0.0], np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))]
        )
        self.tex = _NO_TEX
        tex = _stroke_textures(rng, bt, smp, arc)
        self.chatter, self.blade_tilt = tex.chatter, tex.blade_tilt

        circle = self.circle = _circular(bt)
        aspect = self.aspect = 1.0 if circle else bt.aspect
        self.reach = 1.0 if circle else float(np.hypot(1.0, aspect))
        # in half-widths squared
        self.area_unit = _footprint_area("round" if circle else bt.shape, aspect)
        self.stiff_blade = float(_clip((bt.stiffness - 0.4) / 0.6, 0.0, 1.0))
        # where each column sits across the mark
        self.col_u = (_RING_X if brush.ring else _COL_X) * 2.0 - 1.0
        # squeezed-out ridges are at most this wide (px); they lie just outside the mark
        self.margin = float(_clip(0.12 * w_typ * (1.0 + bt.spread), 0.6, 7.0)) + 1.0
        nominal, radii = _radii(smp, tex.wobble, bt)
        self.radii = radii
        self.lift_reach, self._lift_from = _lift_reach(bt, eff, smp, arc, radii)

        if bt.rigid:
            pad = float(radii.max()) * self.reach + 3.0
            (
                self.wet_y0,
                self.wet_x0,
                self.wet_before,
                self.bands,
                self.band_period,
            ) = _before_the_blade(canvas, rng, pts, pad, w_typ, arc)

        self.stroke_id = canvas._new_stroke_id()
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
        self.cv = (canvas.top, canvas.und, canvas.top_a, canvas.und_a, canvas.top_id)
        self.aux = (canvas._floor, canvas._tooth, canvas._hollows, canvas.dry_height)
        self.br = (brush.tip, brush.reservoir, brush.heel)
        self.pend = _NO_PENDING
        # the brush's hair: where its bristle clumps run along the stroke (see
        # _hair_field)
        hair_at = (0.0, 0.0, 1.0, 1.0)
        if is_dab:
            self.tex2 = tex.dab
        else:
            pad = float(radii.max()) * self.reach + 4.0
            # (a blade's nicks score nearly straight lines, which come and go
            # as grit catches and paint closes over them)
            self.tex2, hair_at = _hair_field(
                bt,
                w_typ,
                float(nominal.max()),
                -pad,
                float(arc[-1]) + pad,
                rng,
                wander=_NICK_WANDER if bt.rigid else 1.0,
            )
        # how ragged this stroke's end is (each stroke ends its own way)
        self.lift_var = float(0.4 + 1.2 * rng.random()) if self.lift_reach > 0 else 1.0
        self.K = self._consts(hair_at, smp.spacing)

    def _consts(self, hair_at, spacing: float):
        """The stroke's settings as the engine takes them."""
        bt, eff, surf = self.bt, self.eff, self.canvas.surface
        return eng.StrokeConsts(
            stroke_id=int(self.stroke_id),
            circle=bool(self.circle),
            is_dab=bool(self.is_dab),
            ring=bool(self.brush.ring),
            rigid=bool(bt.rigid),
            deposits=bool(bt.deposits),
            margin=float(self.margin),
            # how much the surface texture (vs. the hair pattern) decides
            # where a dry brush catches
            tooth_w=float(_clip(0.3 + (surf.tooth - 0.5), 0.0, 0.85)),
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
            # soft brushes fade in and out over this many widths
            feather=float(bt.softness * 3.0),
            # soft brushes are lowered and lifted gently, so their ends feather
            end_k=1.0 if self.is_dab else max(0.0, 1.0 - 1.5 * float(bt.softness)),
            surface_acts=bool(surf.affects_paint),
            slickness=float(surf.slickness),
            absorbency=float(surf.absorbency),
            surface_tooth=float(surf.tooth),
            arc_end=float(self.arc[-1]),
            hair_s0=hair_at[0],
            hair_x0=hair_at[1],
            hair_ds=hair_at[2],
            hair_dx=hair_at[3],
            spacing=float(spacing),
            bundle=float(self.brush.width),
        )

    # ------------------------------------------------------------------
    def steps(self):
        """The steps of the stroke that reach the canvas, in order, each as
        :meth:`step` gives it. Painting past the edge of the canvas uses up
        paint all the same. (Each step is worked out once the one before it
        has been painted: it starts from the brush as that left it.)"""
        for i in range(1, len(self._pts)):
            seg = self._segment(i)
            if seg is None:
                continue
            win = self._window(seg)
            if win.on_canvas:
                yield self._step(seg, win)
            else:
                _use_paint_off_canvas(self.brush, seg.Wd, seg.ds, seg.p)

    def step(self, i: int):
        """Step ``i`` of the stroke, from point i-1 to point i: its scalars (an
        :class:`_engine.Step`) and the footprint's chord tables, or None if
        the tool doesn't move or doesn't reach the canvas there."""
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
            i=i,
            first=i == 1,
            last=i == n - 1,
            touchdown=i == 1 or self.is_dab,
            x0=x0,
            y0=y0,
            hh=win.y1 - y0,
            ww=win.x1 - x0,
            cy0=cy0,
            cx0=cx0,
            ch=win.cy1 - cy0,
            cw=win.cx1 - cx0,
            wy0=cy0 - y0,
            wx0=cx0 - x0,
            cx=seg.cx,
            cy=seg.cy,
            tx=seg.tx,
            ty=seg.ty,
            ds=seg.ds,
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
            arc_i=self._arc[i],
            lip=self._lip[i],
            wet=float(self.brush.wet),
            dry=bool(self.dry),
            span=span,
            nsub=int(min(max(math.ceil(span_cols / 2.0 - 1e-9), 1), 8)),
            rho=0.5 * (self._rolls[i] + self._rolls[i - 1]),
            splay=r / max(0.5 * (nominal[i - 1] + nominal[i]), 1e-9),
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

    def footprint(self, st, tables) -> _Step | None:
        """Where the tool touches the canvas while it moves along the step
        ``st`` (with the chord tables ``tables``, see :meth:`step`), or None
        if it touches nothing.

        For every pixel we compute its *exposure* (see :func:`apply_stroke`),
        its anti-aliased coverage, and where it sits under the brush (across
        the bristles, along the motion); see :func:`_engine.footprint`. And
        which bristle columns are above each pixel."""
        F, hit = eng.footprint(st, self.K, self.tex, self.tex2, *tables)
        if not hit:
            return None
        wsl = _step_slices(st)[1]
        u_px = F[eng.F_U][wsl]
        if self.bt.rigid:
            u_px = _wandering_bands(self, st, F[eng.F_ALONG][wsl], u_px)
        taps = eng.bristle_taps(u_px, st.span, st.nsub, self.brush.ring, st.rho)
        return _Step(st, F, tables, taps)

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


def _use_paint_off_canvas(brush, Wd, ds, p):
    """Painting past the edge of the canvas still uses up paint."""
    total = brush.volume
    if total <= 0:
        return
    used = eng.deposit_thickness(total / (Wd * Wd)) * (0.5 + p) * ds * Wd * 0.9
    g = max(1.0 - used / total, 0.0)
    brush.tip *= g
    brush.reservoir *= g


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
