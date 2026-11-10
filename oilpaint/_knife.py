"""One step of a knife: a rigid blade riding on the paint, laying its load
as a slab, dragging the wet paint under it and pushing it aside, and lifted
off at the end."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, NamedTuple

import numpy as np

from . import _engine as eng
from ._engine import COLUMNS, IB, IV, PACKED
from ._filters import _smoothstep
from ._geometry import _Step
from ._hair import _NO_TEX2
from .noise import aniso_noise, value_noise_2d

if TYPE_CHECKING:
    from .stroke import _Run

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


class _BladeStart(NamedTuple):
    """What a knife stroke starts from (see :func:`_before_the_blade`)."""

    wet_y0: int  # where wet_before lies on the canvas
    wet_x0: int
    wet_before: np.ndarray  # the wet paint there before the stroke
    bands: np.ndarray  # the bands the colours on the blade wander in
    band_period: float  # (px along the stroke)


def _before_the_blade(canvas, rng, pts, pad, w_typ, arc) -> _BladeStart:
    """What a knife stroke starts from: the wet paint lying on the canvas
    before it, which the blade's ends push aside into lips (a pile it is
    carrying mostly rides along), over the part of the canvas within ``pad``
    px of its path, from (wet_y0, wet_x0); and the bands its colours wander
    in (see :func:`_wandering_bands`) and their period along the stroke (px)."""
    H, W = canvas.height, canvas.width
    wet_y0 = min(max(math.floor(float(pts[:, 1].min()) - pad - 1), 0), H)
    wet_x0 = min(max(math.floor(float(pts[:, 0].min()) - pad - 1), 0), W)
    wet_y1 = max(math.ceil(float(pts[:, 1].max()) + pad + 1) + 1, wet_y0)
    wet_x1 = max(math.ceil(float(pts[:, 0].max()) + pad + 1) + 1, wet_x0)
    box = (slice(wet_y0, wet_y1), slice(wet_x0, wet_x1))
    wet_before = canvas.top_v[box] + canvas.und_v[box]
    # Paint on a blade isn't held in lanes like paint in bristles: its
    # colour bands wander a little from side to side along the stroke.
    period = max(0.5 * w_typ, 4.0)
    n_s = int(arc[-1] / period) + 3
    bands = value_noise_2d(8 * n_s, 64, 8.0, rng)
    return _BladeStart(wet_y0, wet_x0, wet_before, bands, period)


def _wandering_bands(run: _Run, F, st, part):
    """Where across the blade (-1..1) the paint a knife lays on each pixel of
    ``part`` (wy0, wx0, ch, cw) of the window of step ``st`` (footprint
    ``F``) comes from: paint on a blade isn't held in lanes, so its colour
    bands wander a little from side to side along the stroke."""
    blade = run.blade
    wy0, wx0, ch, cw = part
    return eng.wandering_bands(
        F,
        wy0,
        wx0,
        ch,
        cw,
        float(run.arc[st.i]),
        blade.band_period,
        blade.bands,
        _KNIFE_BAND_WANDER,
    )


def _knife_taps(run: _Run, st, F):
    """The bristle taps of a knife's step ``st`` (footprint ``F``) over its
    part of the canvas, and (None if that is all of its window) over all of
    its window, past the edge of the canvas too (see :func:`_whole_window`):
    worked out once over the window, and cropped to the canvas."""
    ring = run.brush.ring
    if st.ch == st.hh and st.cw == st.ww:
        u = _wandering_bands(run, F, st, (st.wy0, st.wx0, st.ch, st.cw))
        return eng.bristle_taps(u, st.span, st.nsub, ring, st.rho), None
    u = _wandering_bands(run, F, st, (0, 0, st.hh, st.ww))
    taps_all = eng.bristle_taps(u, st.span, st.nsub, ring, st.rho)
    taps = eng.crop_taps(taps_all, st.ww, st.wy0, st.wx0, st.ch, st.cw)
    return taps, taps_all


def _knife_step(run: _Run, s: _Step) -> None:
    """One step of a knife, in stages that change the canvas in this order:
    the paint it pushes aside (4), the film it drags (5), the paint squeezed
    out beside it, the slab it lays and the paint it presses out at its ends
    (6), the peaks it pulls up as it lifts off, and the layers it folds
    together (7)."""
    st = s.st
    area = max(run.area_unit * st.r * st.r, 1e-9)  # of the blade, px
    if st.ch == 0 or st.cw == 0:
        _knife_step_off(run, s, area)
        return
    # (how readily the paint takes stirring, as it lies before this step)
    f_mix = eng.surface_factors(run.cv, run.aux, st, run.K)[2]
    # Where the blade sticks out past the edge of the canvas, it goes on as
    # over a canvas going on as it is at its edge (see _engine.off_canvas):
    # it rests and rides on that, and lays its slab there from the paint on
    # it, which is gone. (Taking the step over the canvas alone, the paint
    # under the part past the edge stayed on the blade, and went down on the
    # visible part, a third to three quarters too much of it.)
    whole = _whole_window(run, s) if st.ch < st.hh or st.cw < st.ww else None

    # 1) All the paint on a knife is on the blade, ready to go down; several
    #    paints loaded side by side blend slowly as it works. (A knife doesn't
    #    pick wet paint up as it goes: it drags it, step 5.)
    eng.work_load(run.br, run.K, st, np.inf)
    col_vol, _load, col_load, _col_body, body = eng.load_stats(
        run.brush.tip, run.brush.reservoir, st.Wn
    )
    gap, hold = _blade_gap(run, st.p)  # 2)
    if whole is None:
        level, ride = _blade_level(run, s, gap, _film(run, s, area))  # 3)
    else:
        level_w, ride_w = _blade_level(run, whole.s, gap, _film(run, whole.s, area))
        level = np.ascontiguousarray(level_w[s.wsl])
        ride = np.ascontiguousarray(ride_w[s.wsl])
        whole = whole._replace(level=level_w, ride=ride_w)
    ridges = _push_aside(run, s, level)  # 4)
    if run.bt.clumps == 0:
        _drag_film(run, s, level)  # 5)
    eng.squeeze_out(run.br, run.K, st, col_vol, col_load, hold, ridges, run.col_u)
    grid, c_tab, h_tab = s.tables
    eng.lay_ridges_off(
        run.cv, run.pend, run.br, run.K, st, ridges, grid, c_tab, h_tab, _NO_TEX2, True
    )
    _fill_under_blade(run, s, level, ride, area, whole)  # 6)
    # A knife lifted off thick paint pulls it up into little peaks and torn ridges.
    if run.eff["lift"] > 0 and st.i == len(run.pts) - 1:
        _lift_blade(run, s, body)
    _fold_layers(run, s, level, f_mix)  # 7)


def _blade_gap(run: _Run, p: float) -> tuple[float, float]:
    """2) The blade rides above the surface at a gap set by pressure ``p``
    (all its paint is a film under it; what doesn't fit under it rides along
    on it). Returns the gap and how much paint the blade holds (load units)."""
    bt = run.bt
    gap = _KNIFE_GAP * (1.0 - p) ** 1.6 * (0.55 + 0.45 * bt.stiffness) + 0.05
    if bt.clumps > 0:  # the teeth of a comb dig in, down to the threads
        gap *= 0.03
    return gap, gap * run.area_unit / 4.0 + _KNIFE_CARRY


def _film(run: _Run, s: _Step, area: float):
    """How thick the film of paint on the blade is over each pixel, from the
    paint on it as it stands now: where the blade rides (3) and what it
    fills (6) each read it, before and after it has spread and squeezed
    some out."""
    brush, st = run.brush, s.st
    return (
        eng.at_pixels(s.taps, brush.tip[:, IV] + brush.reservoir[:, IV], st.ch, st.cw)
        * COLUMNS
        / area
    )


def _blade_rest(run: _Run, s: _Step):
    """3) Where the blade sits: on the high points of the surface below
    (thread tops for a flexible blade, the highest ridges around for a stiff
    one). (Worked out pixel by pixel by _engine.blade_rest, blade_bins and
    blade_sink.)"""
    canvas, bt, st, relief = run.canvas, run.bt, s.st, run.relief
    rest, old_v, vmax, under_blade, any_under, top_ib, und_ib = eng.blade_rest(
        canvas.top,
        canvas.und,
        canvas.top_id,
        relief.peaks_fine,
        relief.peaks_wide,
        st.cy0,
        st.cx0,
        st.ch,
        st.cw,
        run.stroke_id,
        run.stiff_blade,
        s.F,
        st.wy0,
        st.wx0,
    )
    # Wet paint from earlier strokes holds the blade up too, the heap it pushes
    # included (a knife rides on a wave of paint): it rests on its high points
    # and sinks in the harder it is pressed (less into stiff paint). A light
    # touch lays new paint on top of the old.
    if not (vmax > 0.05 and any_under):
        return rest.astype(np.float64)
    # (summed by numpy as the pixels lie: on the canvas, or gathered from it
    # past its edge)
    if s.sl is not None:
        und_ib = canvas.und[..., IB][s.sl]
    old_body = float((und_ib.sum() + top_ib.sum()) / max(float(old_v.sum()), 1e-9))
    # How far into it the blade sinks, as a share of its thickness: a light
    # touch skims, a firm one cuts in, pressing hard goes nearly down to the
    # canvas, however thick the paint. Stiff paint resists a little.
    depth = min(max(st.p**_KNIFE_SINK * (1.15 - 0.3 * min(old_body, 1.0)), 0.0), 1.0)
    # narrow teeth concentrate the pressure: they cut right through
    if bt.clumps > 0:
        depth = max(depth, 0.9)
    # The blade is rigid along the direction it moves: its height varies only
    # along its length. It rests on the high points under it (not on the pile
    # it pushes ahead), bending to follow them over a span set by its
    # stiffness: what lies under it at each position along it (whole px from
    # its end), averaged over ``span`` px either way.
    span = int(min(max(0.1 * run.w_typ * (0.5 + run.stiff_blade), 2), 10))
    hi, b, cnt, tot_hi, tot_rest = eng.blade_bins(
        relief.floor, st.cy0, st.cx0, old_v, rest, under_blade, s.F, st.wy0, st.wx0
    )
    kernel = np.ones(2 * span + 1)
    return eng.blade_sink(
        rest,
        hi,
        b,
        np.convolve(cnt, kernel, mode="same"),
        np.convolve(tot_hi, kernel, mode="same"),
        np.convolve(tot_rest, kernel, mode="same"),
        depth,
    )


def _nearest(run: _Run, st):
    """The pixels of the canvas the step's part of it (``st``: cy0, cx0, ch,
    cw) lies at; past its edge (see :func:`_whole_window`), the nearest ones
    (to index a plane of the canvas with)."""
    H, W = run.canvas.height, run.canvas.width
    return np.ix_(
        np.clip(np.arange(st.cy0, st.cy0 + st.ch), 0, H - 1),
        np.clip(np.arange(st.cx0, st.cx0 + st.cw), 0, W - 1),
    )


def _blade_level(run: _Run, s: _Step, gap: float, film):
    """3) How high the blade's edge rides over each pixel: on what it rests
    on (:func:`_blade_rest`), at ``gap``, less as the paint under it runs low
    (``film``, see :func:`_film`). Returns the level and how high it rides on
    the paint (see _engine.blade_level)."""
    st = s.st
    # The blade rides on the paint under it: as that runs low, it sinks and the
    # slab thins out (not a comb's teeth). Nicks and grit on its edge leave
    # fine scratches along the stroke; it wavers as it is pushed (chatter) and
    # is never pressed quite evenly along its length (tilt).
    # (skimming, the blade glides: no chatter; its nicks still streak the paint)
    chat = 0.25 * (1.0 - eng.smoothstep(_KNIFE_SKIM, 0.05, st.p))
    return eng.blade_level(
        _blade_rest(run, s),
        film,
        _KNIFE_FILM,
        gap,
        run.bt.clumps > 0,
        1.0 + chat * run.chatter[st.i],
        run.blade_tilt,
        s.F,
        st.wy0,
        st.wx0,
    )


def _push_aside(run: _Run, s: _Step, level):
    """4) Wet paint standing above the blade is bulldozed ahead by the flow
    (5); near the edges of the mark, some of the paint that was lying there
    spills out sideways (lips; the pile the blade carries rides along with
    it). Nicks and flex let streaks of it slip under the blade and stay. A
    comb's teeth push what they cut into ridges between them. Returns the
    paint spilt to each side of the blade, as :func:`_engine.squeeze_out`
    adds to it: (all round, left, right)."""
    canvas, st = run.canvas, s.st
    wsl = s.wsl
    hair, expo_c = s.hair[wsl], s.expo[wsl]
    # all round, left, right
    ridges = (np.zeros(PACKED), np.zeros(PACKED), np.zeros(PACKED))
    flowing = run.bt.clumps == 0
    wet = np.zeros((0, 0), np.float32)  # (a comb: none spills)
    if flowing:  # what was lying there before the stroke
        blade = run.blade
        wet = np.ascontiguousarray(
            blade.wet_before[
                st.cy0 - blade.wet_y0 : st.cy0 - blade.wet_y0 + st.ch,
                st.cx0 - blade.wet_x0 : st.cx0 - blade.wet_x0 + st.cw,
            ]
        )
    # (what stands above the blade, the share of it a pass moves, and near
    # the edges of the mark what spills: see _engine.blade_excess)
    excess, tv, emax = eng.blade_excess(
        canvas.top,
        canvas.und,
        run.relief.floor,
        st.cy0,
        st.cx0,
        level,
        1.0 - np.exp(-4.0 * expo_c * (0.15 + 1.7 * hair**2)),
        s.F,
        st.wy0,
        st.wx0,
        max(st.r * st.lmax, 1e-6),  # the mark's half-width, px
        _KNIFE_SPILL_PX,
        _KNIFE_SPILL,
        wet,
    )
    if emax > 1e-3:
        moved = eng.take(canvas.top, canvas.und, st.cy0, st.cx0, excess, tv)
        if flowing:
            eng.add_sides(ridges, moved, np.ascontiguousarray(s.lat_px[wsl]), 1.0)
        else:
            fresh = (
                s.between
                * _smoothstep(-st.ds - 1.0, 0.0, s.along - s.c)
                * _smoothstep(1.0, -1.0, s.along - s.c - s.h)
            )[wsl]
            eng.spread(
                cv=run.cv,
                pend=run.pend,
                br=run.br,
                sy0=st.cy0,
                sx0=st.cx0,
                w=fresh,
                pk=moved.sum(axis=(0, 1)),
                cover=np.zeros(fresh.shape),
                stroke_id=run.stroke_id,
                by_film=False,
            )
    return ridges


def _drag_film(run: _Run, s: _Step, level) -> None:
    """5) The blade drags the wet paint it touches, as a thin-film flow:
    between the moving blade and the still canvas the paint shears, so paint
    near the blade travels almost with it and paint at the canvas hardly
    moves; what stands above the blade is bulldozed ahead of it. Each wet
    layer moves at its own speed, so their order is kept (white dragged over
    purple stays on top). The blade glides on its own fresh paint."""
    st, wsl = s.st, s.wsl
    hair = s.hair[wsl]
    # (planes passed as arrays of their own: one compiled kernel, whether
    # the step's window is all on the canvas or not)
    eng.flow_under_blade(
        run.cv,
        st.cy0,
        st.cx0,
        run.stroke_id,
        np.ascontiguousarray(run.relief.floor[s.sl]),
        level,
        np.ascontiguousarray(s.aa[wsl]),
        np.ascontiguousarray(s.along[wsl]),
        np.ascontiguousarray(s.c[wsl]),
        np.ascontiguousarray(s.h[wsl]),
        st.ds,
        st.tx,
        st.ty,
        np.clip((0.5 + 0.45 * st.p) * (0.6 + 0.8 * hair), 0.0, 1.0),
        _VISCOUS,
    )


def _fill_under_blade(run: _Run, s: _Step, level, ride, area: float, whole) -> None:
    """6) The blade fills the space under it, up to its ``level``, from the
    film of paint it carries (see :func:`_slab`), and presses some out at
    its ends (:func:`_press_out_ends`). If that makes three wet layers, the
    two most alike merge (white laid on white-over-purple joins the white; a
    new colour stays on top of the others). With ``whole`` (see
    :func:`_whole_window`), the blade also fills under its part past the edge
    of the canvas, from the paint on it, which is gone."""
    if not (run.brush.tip[:, IV].sum() > 0 and run.bt.deposits):
        return
    st, wsl = s.st, s.wsl
    if whole is None:
        d, skim = _slab(run, s, level, area)
        extra = np.empty(0)
    else:
        d_w, skim = _slab(run, whole.s, whole.level, area, whole.surf_add)
        d = np.ascontiguousarray(d_w[wsl])
        d_off = d_w.copy()
        d_off[wsl] = 0.0  # (on the canvas: d)
        extra = eng.column_demand(whole.s.taps, d_off)
        _lay_off(run, st, d_off)
    if run.bt.clumps == 0:
        _press_out_ends(run, s, ride, skim)
        if whole is not None:
            _press_out_past_edge(run, whole, skim)
    eng.lay_off(
        run.cv,
        run.pend,
        run.br,
        s.taps,
        st.cy0,
        st.cx0,
        d,
        np.ascontiguousarray(s.cover[wsl]),
        run.stroke_id,
        True,
        False,
        np.ascontiguousarray(s.expo[wsl]),
        False,
        extra,
    )


class _Whole(NamedTuple):
    """A step of a knife over the whole of its window (see
    :func:`_whole_window`)."""

    s: _Step  # the step, its part of the canvas all of the window
    on: _Step  # the step as it is: its part on the canvas is on.wsl of the window
    surf_add: np.ndarray  # what to add to the surface read off the canvas
    level: np.ndarray | None = None  # how high the blade rides (see _blade_level)
    ride: np.ndarray | None = None


def _whole_window(run: _Run, s: _Step) -> _Whole:
    """The step ``s`` over the whole of its window, past the edge of the
    canvas too, where the canvas is taken to go on as it is at its edge:
    as a step whose part of the canvas is all of it (its pixels read from
    the nearest canvas pixel, see :func:`_nearest`)."""
    st = s.st
    st_w = st._replace(cy0=st.y0, cx0=st.x0, ch=st.hh, cw=st.ww, wy0=0, wx0=0)
    sw = _Step(st_w, s.F, s.tables, s.taps_all)
    sw.sl = None  # (gathered: see _nearest)
    # Past the edge, the surface is the canvas's at its edge, with the slab
    # this stroke has laid out there in place of the one it laid at the edge
    # (a blade fills up to its level: over its own slab it lays no more).
    canvas = run.canvas
    surf_add = eng.off_surface(
        canvas.top,
        canvas.top_id,
        run.stroke_id,
        st.y0,
        st.x0,
        _laid_off(run, st),
        st.wy0,
        st.wx0,
        st.ch,
        st.cw,
    )
    return _Whole(sw, s, surf_add)


# Side (px) of the tiles in which the slab a knife lays past the edge of the
# canvas is kept (see _laid_off).
_OFF_TILE = 32


def _laid_off(run: _Run, st):
    """The slab the stroke has laid past the edge of the canvas, over the
    window of step ``st``. (Kept in tiles of _OFF_TILE px, only where it has
    laid any: what a long stroke keeps grows with the off-canvas stretch the
    blade passed over, not with the stroke's bounding box.)"""
    tiles = run.knife_off
    out = np.zeros((st.hh, st.ww))
    if not tiles:
        return out
    T = _OFF_TILE
    for ty in range(st.y0 // T, (st.y0 + st.hh - 1) // T + 1):
        for tx in range(st.x0 // T, (st.x0 + st.ww - 1) // T + 1):
            tile = tiles.get((ty, tx))
            if tile is None:
                continue
            y0, x0 = max(ty * T, st.y0), max(tx * T, st.x0)
            y1, x1 = min((ty + 1) * T, st.y0 + st.hh), min((tx + 1) * T, st.x0 + st.ww)
            out[y0 - st.y0 : y1 - st.y0, x0 - st.x0 : x1 - st.x0] = tile[
                y0 - ty * T : y1 - ty * T, x0 - tx * T : x1 - tx * T
            ]
    return out


def _lay_off(run: _Run, st, d) -> None:
    """Record the slab ``d`` laid over the window of step ``st`` past the
    edge of the canvas (zero on the canvas; see :func:`_laid_off`)."""
    if run.knife_off is None:
        run.knife_off = {}
    tiles = run.knife_off
    T = _OFF_TILE
    for ty in range(st.y0 // T, (st.y0 + st.hh - 1) // T + 1):
        for tx in range(st.x0 // T, (st.x0 + st.ww - 1) // T + 1):
            y0, x0 = max(ty * T, st.y0), max(tx * T, st.x0)
            y1, x1 = min((ty + 1) * T, st.y0 + st.hh), min((tx + 1) * T, st.x0 + st.ww)
            part = d[y0 - st.y0 : y1 - st.y0, x0 - st.x0 : x1 - st.x0]
            if not part.any():
                continue
            tile = tiles.get((ty, tx))
            if tile is None:
                tile = tiles[ty, tx] = np.zeros((T, T), np.float32)
            tile[y0 - ty * T : y1 - ty * T, x0 - tx * T : x1 - tx * T] += part


def _press_out_past_edge(run: _Run, whole: _Whole, skim: float) -> None:
    """The paint the blade presses out at an end past the edge of the
    canvas (see :func:`_press_out_ends`) is gone."""
    sw, on = whole.s, whole.on.st
    d_end = _end_press(whole.ride, sw, skim, (on.wy0, on.wx0, on.ch, on.cw))
    if d_end.max() > 1e-6:
        eng.take_tips(run.brush.tip, sw.taps, d_end)


def _end_press(ride, s: _Step, skim: float, on=(0, 0, 0, 0)):
    """How much paint the blade presses out near its ends over each pixel of
    the step's part of the canvas (see :func:`_press_out_ends`), none in the
    part ``on`` (wy0, wx0, ch, cw) of it (see _engine.blade_ends)."""
    st = s.st
    # (skimming, the blade spreads no film to press out; it takes a slab
    # of some thickness to stand in a bead)
    # (as much as the slab is thick, over each step's share of a pass:
    # not what this step filled, which wavers from step to step with
    # the blade's chatter and would come out as a bead at every step)
    return eng.blade_ends(
        ride,
        s.F,
        st.wy0,
        st.wx0,
        max(st.r * st.lmax, 1e-6),
        _KNIFE_END_PX,
        _KNIFE_END * (0.8 + st.p) * (1.0 - skim),
        *on,
    )


def _knife_step_off(run: _Run, s: _Step, area: float) -> None:
    """A step of a knife wholly off the canvas: it goes on as over a canvas
    going on as it is at its edge (see :func:`_whole_window`), and the paint
    it lays and squeezes out there is gone."""
    st = s.st
    eng.work_load(run.br, run.K, st, np.inf)
    col_vol, _load, col_load, _col_body, _body = eng.load_stats(
        run.brush.tip, run.brush.reservoir, st.Wn
    )
    gap, hold = _blade_gap(run, st.p)
    ridges = (np.zeros(PACKED), np.zeros(PACKED), np.zeros(PACKED))
    eng.squeeze_out(run.br, run.K, st, col_vol, col_load, hold, ridges, run.col_u)
    if not (run.brush.tip[:, IV].sum() > 0 and run.bt.deposits):
        return
    whole = _whole_window(run, s)
    # (nothing changes the paint on the blade between where it rides and what
    # it fills: the same film)
    film = _film(run, whole.s, area)
    level_w, ride_w = _blade_level(run, whole.s, gap, film)
    whole = whole._replace(level=level_w, ride=ride_w)
    d_w, skim = _slab(run, whole.s, level_w, area, whole.surf_add, film)
    if run.bt.clumps == 0:
        _press_out_past_edge(run, whole, skim)
    extra = eng.column_demand(whole.s.taps, d_w)
    _lay_off(run, st, d_w)
    empty = np.zeros((0, 0))
    eng.lay_off(
        run.cv,
        run.pend,
        run.br,
        s.taps,
        0,
        0,
        empty,
        empty,
        run.stroke_id,
        True,
        False,
        empty,
        False,
        extra,
    )


def _slab(run: _Run, s: _Step, level, area: float, surf_add=None, film=None):
    """How much paint the blade lays under it over each pixel this step (see
    :func:`_fill_under_blade`): with plenty of paint, a smooth slab; running
    low, it only reaches the high points (broken colour). Barely touching,
    the blade isn't spreading a film any more: only the bead of paint on its
    edge reaches down, and it touches the high points of the surface (thread
    tops, ridges of paint) and leaves paint on them, nothing in the hollows:
    broken colour ("let it break"). Returns that, and how much the blade
    skims (0: not at all)."""
    canvas, st = run.canvas, s.st
    p, wsl = st.p, s.wsl
    hair, expo_c = s.hair[wsl], s.expo[wsl]
    if film is None:  # (the paint on the blade as it is now: see _film)
        film = _film(run, s, area)
    # Barely touching, the blade skims: it fills less of the gap under it.
    skim = eng.smoothstep(_KNIFE_SKIM, 0.05, p)
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
    # The gap between the blade and the surface (past the edge of the canvas,
    # plus surf_add: see _whole_window), as far as its film reaches, filled
    # by the share k_fill gives over this step's share of a pass (see
    # _engine.blade_slab).
    d, surf = eng.blade_slab(
        canvas.top,
        canvas.und,
        run.relief.floor,
        st.cy0,
        st.cx0,
        level,
        np.zeros((0, 0)) if surf_add is None else surf_add,
        film,
        1.0 - np.exp(-k_fill * expo_c),
    )
    if skim > 0 and run.bt.clumps == 0:
        bead = np.minimum(film, 1.2)
        touch = np.clip((surf - (level - bead)) / np.maximum(bead, 1e-6), 0.0, 1.0)
        # the bead is uneven and the paint below still follows the threads
        # of the canvas: barely touching, it catches in streaks along the
        # blade's nicks and on the thread tops, the fewer the lighter the touch
        grain = 0.55 * hair + 0.45 * run.relief.tooth[_nearest(run, st)]
        thr = _KNIFE_TOUCH - 2.2 * p
        touch = touch * _smoothstep(thr - 0.1, thr + 0.1, grain)
        # the bead leaves paint wherever it passes, in proportion to how
        # much of a pass this step is (for the same reason)
        pass_bead = n * (1.0 - np.exp(-4.0 / n))
        d += skim * _KNIFE_BEAD * pass_bead * bead * touch**0.7 * expo_c
    return d, skim


def _press_out_ends(run: _Run, s: _Step, ride, skim: float) -> None:
    """Near its two ends the blade presses paint out sideways, more the
    harder it is pressed: it stands in a bead on the edge of the slab and
    beyond each end of the blade (see :func:`_press_out`)."""
    st, wsl = s.st, s.wsl
    d_end = _end_press(ride, s, skim)
    if d_end.max() > 1e-6:
        _press_out(run, st, s.taps, s.lat_px[wsl], s.along[wsl], d_end)


def _lift_blade(run: _Run, s: _Step, body: float) -> None:
    """A knife lifted off thick paint (at the last step) pulls it up into
    little peaks and torn ridges, the stiffer the paint on it (``body``)
    the more (see :func:`_lift_off`)."""
    st, wsl = s.st, s.wsl
    under = (s.aa * np.clip((s.h - np.abs(s.rel)) / 1.5, 0.0, 1.0))[wsl]
    # most along the trailing edge, the last of the blade to leave the paint
    under = under * (0.3 + 0.7 * _smoothstep(0.3, -0.5, (s.rel / s.h)[wsl]))
    _lift_off(
        run.canvas,
        s.sl,
        under,
        run.eff["lift"] * (0.5 + 0.8 * body),
        max(2.5, 0.08 * st.Wd),
        np.degrees(np.arctan2(st.ty, st.tx)),
        run.rng,
        run.stroke_id,
    )


def _fold_layers(run: _Run, s: _Step, level, f_mix) -> None:
    """7) A blade only shears the paint it cuts through: riding within the
    top layer it leaves the one below alone; cutting into it, it folds the
    two into each other in streaks (marbling), not evenly."""
    st, wsl = s.st, s.wsl
    p = st.p
    expo_c = s.expo[wsl]
    m = 1.0 - np.exp(-run.eff["mixing"] * 3.0 * p * p * expo_c * s.streak[wsl] * f_mix)
    # (in streaks along its nicks, as far as it cuts into the layer below:
    # see _engine.blade_folds)
    m = eng.blade_folds(
        run.canvas.und, run.relief.floor, st.cy0, st.cx0, level, m, s.F, st.wy0, st.wx0
    )
    eng.stir(run.cv, st.cy0, st.cx0, m)


# ----------------------------------------------------------------------------
# helpers


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
        np.ascontiguousarray(lat),
        np.ascontiguousarray(along_px),
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
            cv=run.cv,
            pend=run.pend,
            br=run.br,
            sy0=y0,
            sx0=x0,
            w=w,
            pk=pk,
            cover=_smoothstep(0.1, 0.6, w),
            stroke_id=run.stroke_id,
            by_film=False,
        )


def _lift_off(canvas, sl, under, amp, cell, direction, rng, stroke_id):
    """The stroke's own paint under a tool lifted off it is pulled up into
    torn crests and peaks, stretched along the way the tool was moving
    (volume-conserving; other strokes' paint is left as it is)."""
    if amp <= 0 or not under.any():
        return
    top = canvas.top[sl]
    own = canvas.top_id[sl] == stroke_id
    V = np.where(own, top[..., IV], 0.0).astype(np.float64)
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
    # (scaling each pixel's paint: where the stroke's own paint holds more
    # than one colour, that makes some pigment and destroys some, up to a few
    # percent of it; moving paint between pixels instead either mixes the
    # colours together or, between neighbours only, hardly piles it up)
    fac = np.where(own, fac, 1.0)
    top *= fac.astype(np.float32)[..., None]
