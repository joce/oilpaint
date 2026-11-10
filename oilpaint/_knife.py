"""One step of a knife: a rigid blade riding on the paint, laying its load
as a slab, dragging the wet paint under it and pushing it aside, and lifted
off at the end."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np

from . import _engine as eng
from ._engine import COLUMNS, IB, IV, PACKED
from ._geometry import _Step
from ._hair import _NO_TEX2
from .brush import _clip, _smoothstep
from .noise import aniso_noise

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
