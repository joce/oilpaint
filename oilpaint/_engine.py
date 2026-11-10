"""The compiled core of the stroke engine (numba).

Everything here works on plain arrays, one pixel at a time, so a step of a
small brush costs microseconds rather than the overhead of hundreds of small
numpy calls. :mod:`oilpaint.brush` holds the brush and stroke model and calls
these kernels; the physics is described there.

Arrays: the canvas layers ``top``/``und`` are (H, W, PACKED) float32, their
coverage ``top_a``/``und_a`` and the stroke ids ``top_id`` (H, W). Paint on a
brush is (COLUMNS, PACKED) float64 per depth (tip, belly, heel). A step works
on a window of the canvas; per-pixel inputs are window-sized.
"""

import math
from collections import namedtuple

import numpy as np
from numba import njit

from .spectral import N_BANDS

N = N_BANDS
# packed indices of scattering, volume and body (stiffness x volume)
IS, IV, IB = N, N + 1, N + 2
PACKED = N + 3
#: Number of bristle columns across a brush. Each holds its own paint.
COLUMNS = 24

# Two paints this alike (see alike) count as much the same paint.
SAME_PAINT = 0.6
# A new film of paint this thick covers the wet paint under it about two-thirds
# of the way: it buries that much of it (see deposit) ...
COVERING_FILM = 0.15
# ... and once it would bury this much, it buries it all (see the pending paint).
BURIED = 0.9
# Where the body of a brush passes over (a full pass at this much of its
# full-strength profile), even a light touch lays its paint on top.
BODY0, BODY1 = 0.5, 0.8
# Two wet layers this alike look the same mixed or one over the other (a sky
# worked in several strokes: ~0.1; white over that sky: ~0.4 and up).
SAME_LAYER = 0.2
# Width of the footprint's soft (anti-aliased) edge in pixels. Kept narrow:
# a wide soft edge would read as a thin film of paint, which tints differently.
EDGE_PX = 1.5
# Paint a brush holds on to and never releases (in width squares).
RESIDUAL = 0.03
# Paint up in the heel only comes down when the brush is pressed: at
# touchdown (more with more pressure), and whenever it is pressed harder
# during the stroke. It does not trickle out on its own.
HEEL_SQUEEZE = 0.15
# Diffusion between neighbouring columns per brush width of travel, at blending 1 and
# full pressure.
BLEND_RATE = 2.3
# Paint (in load units: brush-width squares) the hair holds inside at medium
# pressure, per unit of capacity, for a brush 20 px wide or more (smaller
# brushes hold more for their size). Paint loaded beyond that sits on the
# outside as a glob: it goes on thick, and pressing squeezes it out at the
# edges of the stroke.
HOLD = 11.0
# Width (px) over which a glob of paint rounds off at the edges of a stroke.
GLOB_EDGE_PX = 2.5
# Extra layer thickness per load unit of glob, for paint of body 1.
IMPASTO = 0.12
# How deep wet paint must reach up the hairs (in units of paint thickness, per
# unit of tip capacity) for half of what a brush lifts from it to go past the
# tips into the body of the brush (see past_tips). With 6, a full hog flat
# dragged through medium wet paint (1.5 thick) ends up with about 3% of that
# paint in its body (under 1% if it all went to the tips): enough to tint its
# next stroke all along. A thin film (0.5) still mostly dirties the tips.
TIP_DEPTH = 6.0
# Share of an overload squeezed out at the edges per width of travel (knife, brush).
SQUEEZE_KNIFE, SQUEEZE_BRUSH = 0.03, 0.04
# Paint thinner than this can't form a continuous film (as canvas.MIN_FILM).
MIN_FILM = 0.15
# Steepest slope (paint thickness per pixel) that fluid (body 0) and very
# stiff (body 1) paint can hold before it flows.
SLOPE_FLUID, SLOPE_STIFF = 0.05, 4.0

# Per-pixel footprint quantities, as planes of the array `footprint` returns.
(
    F_ALONG,
    F_LAT,
    F_U,
    F_HAIR,
    F_AA,
    F_PROFILE,
    F_C,
    F_H,
    F_REL,
    F_OVERLAP,
    F_EXPO,
    F_COVER,
    F_STREAK,
    F_BETWEEN,
    F_EDGE,
) = range(15)
_NF = 15

#: Settings of a stroke that stay the same along it.
StrokeConsts = namedtuple(
    "StrokeConsts",
    [
        "stroke_id",
        "circle",
        "is_dab",
        "ring",
        "rigid",
        "deposits",
        "margin",
        "tooth_w",
        "drag",
        "pickup",
        "streaks",
        "mixing",
        "smear",
        "blending",
        "rough",
        "ridge",
        "stiffness",
        "capacity",
        "dig",
        "softness",
        "slant",
        "clumps",
        "feather",
        "end_k",
        "surface_acts",
        "slickness",
        "absorbency",
        "surface_tooth",
        "arc_end",
        # where the brush's hair pattern (see hair_field) lies in stroke
        # coordinates: distance along the stroke and position across the
        # bristles (px) of its first sample, and its sample spacing (px)
        "hair_s0",
        "hair_x0",
        "hair_ds",
        "hair_dx",
        # the stroke's nominal step along its path (px)
        "spacing",
        # the brush's own size: how wide it paints on average (px)
        "bundle",
    ],
)

#: One step: where the tool is and how it is held, and the window it works on.
#: (x0, y0, hh, ww): the window around the swept segment (may stick out of the
#: canvas); (cy0, cx0, ch, cw): its part on the canvas, at (wy0, wx0) in it.
Step = namedtuple(
    "Step",
    [
        "i",
        "first",
        "last",
        "touchdown",
        "x0",
        "y0",
        "hh",
        "ww",
        "cy0",
        "cx0",
        "ch",
        "cw",
        "wy0",
        "wx0",
        "cx",
        "cy",
        "tx",
        "ty",
        "ds",
        "p",
        "p_now",
        "p_before",
        "r0",
        "r1",
        "r",
        "Wd",
        "Wn",
        "side",
        "uscale",
        "lmax",
        "h_tab_max",
        "arc_i",
        "lip",
        "wet",
        "dry",
        # how the bristles are sampled across a pixel (see bristle_taps), and how far a
        # round is rolled
        "span",
        "nsub",
        "rho",
        # how far pressure splays the bristles (half-width over the bundle's own)
        "splay",
        # lifting off at the end of the stroke: over how many px of it the
        # bristle clumps leave the paint, and how far they pull it up as they go
        # (see footprint)
        "lift_len",
        "lift_amp",
        # how far a dab's load goes beyond what its hair holds (0: none; 1: as
        # much again)
        "overload",
    ],
)


# ----------------------------------------------------------------------------
# small helpers


@njit(cache=True, error_model="numpy")
def clip(x, lo, hi):
    return min(max(x, lo), hi)


@njit(cache=True, error_model="numpy")
def smoothstep(e0, e1, x):
    t = min(max((x - e0) / (e1 - e0), 0.0), 1.0)
    return t * t * (3.0 - 2.0 * t)


@njit(cache=True, error_model="numpy")
def interp(x, x0, x1, y, left, right):
    """np.interp on an evenly spaced grid from x0 to x1."""
    n = y.shape[0]
    if x < x0:
        return left
    if x > x1:
        return right
    t = (x - x0) / (x1 - x0) * (n - 1)
    j = min(int(t), n - 2)
    return y[j] + (y[j + 1] - y[j]) * (t - j)


@njit(cache=True, error_model="numpy")
def deposit_thickness(load):
    """Layer thickness laid down by a brush holding ``load`` width-squares of paint.

    Roughly 1 (a normal, covering layer) for a well-loaded brush, thicker for a
    heavily loaded one (impasto), thinning out as the brush runs dry.
    """
    free = max(load - RESIDUAL, 0.0)
    return free / (free + 0.3) * (1.0 + 0.3 * math.log1p(free / 6.0))


@njit(cache=True, error_model="numpy")
def alike(a, b):
    """How different two packed paints are (0 = the same paint). Each is
    compared in its own precision, so two layers of the same paint on the
    canvas (single precision) come out exactly alike, not almost."""
    floor = np.float32(1e-9)
    va, vb = max(a[IV], floor), max(b[IV], floor)
    tot = 0.0
    for c in range(IV):
        tot += abs(math.log1p(a[c] / va) - math.log1p(b[c] / vb))
    return tot / IV


@njit(cache=True, error_model="numpy")
def harmless(top, und):
    """Burying the top layer changes nothing you could see: the layer below
    is much the same paint (or there is none)."""
    if und[IV] < 1e-4:
        return True
    zero, floor = np.float32(0.0), np.float32(1e-9)
    va, vb = max(max(top[IV], zero), floor), max(max(und[IV], zero), floor)
    tot = 0.0
    for c in range(IV):
        tot += abs(
            math.log1p(max(top[c], zero) / va) - math.log1p(max(und[c], zero) / vb)
        )
        if not tot / IV < SAME_LAYER:  # (the differences only add up)
            return False
    return True


@njit(cache=True, error_model="numpy")
def buries(film):
    """How much of the wet top layer a new film of paint this thick buries:
    a covering film all of it, a faint trace hardly any."""
    return 1.0 - math.exp(-film / COVERING_FILM)


@njit(cache=True, error_model="numpy")
def buried(film, reach):
    """How much of the old top layer goes under: all of it where the body of
    the brush went over it or it laid a covering film, less elsewhere."""
    return max(buries(film), smoothstep(BODY0, BODY1, reach))


# ----------------------------------------------------------------------------
# the paint on the tool


@njit(cache=True, error_model="numpy")
def diffuse(cols, k, ring):
    """Spread paint between neighbouring columns (in place, conserving it).
    On a ring (a round brush) the last column neighbours the first."""
    if k <= 0:
        return
    n_sub = int(math.ceil(k / 0.24))
    kk = k / n_sub
    C, P = cols.shape
    flow = np.empty((C, P))
    for _ in range(n_sub):
        if ring:
            for b in range(C):
                nb, pb = (b + 1) % C, (b - 1) % C
                for c in range(P):
                    flow[b, c] = kk * (cols[nb, c] - 2.0 * cols[b, c] + cols[pb, c])
            cols += flow
        else:
            for b in range(C - 1):
                for c in range(P):
                    # from column b+1 into b
                    flow[b, c] = kk * (cols[b + 1, c] - cols[b, c])
            for b in range(C - 1):
                for c in range(P):
                    cols[b, c] += flow[b, c]
                    cols[b + 1, c] -= flow[b, c]


@njit(cache=True, error_model="numpy")
def seep(heel, belly, rate):
    """Squeeze paint from the heel down into the belly (more when the belly is
    emptier)."""
    for b in range(heel.shape[0]):
        hv = heel[b, IV]
        move_v = rate * max(hv - belly[b, IV], 0.0) * 0.5
        if move_v > 0 and hv > 0:
            f = move_v / hv
            for c in range(PACKED):
                m = heel[b, c] * f
                heel[b, c] -= m
                belly[b, c] += m


@njit(cache=True, error_model="numpy")
def swap_layers(tip, belly, rate):
    """Exchange paint between the tip and the belly of each column (in place).
    ``rate`` is the share of the smaller of the two that changes places; at
    most all of it (a long step of a narrow brush asks for more)."""
    if rate <= 0:
        return
    rate = min(rate, 1.0)
    for b in range(tip.shape[0]):
        tv, bv = tip[b, IV], belly[b, IV]
        e = rate * min(tv, bv)
        if e > 0:
            ft, fb = e / tv, e / bv
            for c in range(PACKED):
                from_tip, from_belly = tip[b, c] * ft, belly[b, c] * fb
                tip[b, c] += from_belly - from_tip
                belly[b, c] += from_tip - from_belly


@njit(cache=True, error_model="numpy")
def refill(tip, belly, capacity):
    """Top up each column's tip from its own belly."""
    for b in range(tip.shape[0]):
        need = capacity / COLUMNS - tip[b, IV]
        have = belly[b, IV]
        if need > 0 and have > 0:
            f = min(need / have, 1.0)
            for c in range(PACKED):
                m = belly[b, c] * f
                tip[b, c] += m
                belly[b, c] -= m


@njit(cache=True, error_model="numpy")
def overflow(tip, belly, capacity):
    """What the tips hold beyond their capacity goes back into the belly
    (this is how the brush gets dirty)."""
    cap_col = capacity / COLUMNS
    for b in range(tip.shape[0]):
        over = tip[b, IV] - cap_col
        if over > 0:
            f = over / max(tip[b, IV], 1e-30)
            for c in range(PACKED):
                m = tip[b, c] * f
                tip[b, c] -= m
                belly[b, c] += m


@njit(cache=True, error_model="numpy")
def work_load(br, K, st, capacity):
    """Refill each column's tip from its belly; squeeze the heel when the tool
    is pressed down or harder; the paints on the tool blend tip with belly and
    side to side as it works."""
    tip, belly, heel = br
    p = st.p
    refill(tip, belly, capacity)
    # pressing the brush down squeezes some paint from the heel towards the tip
    if st.first:
        seep(heel, belly, min(HEEL_SQUEEZE * (0.3 + 2.8 * p * p), 1.0))
    elif st.p_now > st.p_before:  # ... and so does bearing down harder
        seep(heel, belly, min(HEEL_SQUEEZE * 5.0 * (st.p_now - st.p_before), 1.0))
    blend = K.blending
    swap_layers(tip, belly, 0.375 * blend * st.ds / st.Wd)
    swap_layers(belly, heel, 0.1 * blend * st.ds / st.Wd)
    k = BLEND_RATE * blend * (0.2 + p) ** 2 * st.ds / st.Wd
    for layer in (tip, belly, heel):
        for b in range(COLUMNS):
            # (an empty layer, often the heel, has nothing to spread)
            if layer[b, IV] != 0:
                diffuse(layer, k, K.ring)
                break


@njit(cache=True, error_model="numpy")
def load_stats(tip, belly, Wn):
    """How much paint a tool carries, per column, in load units (width
    squares): (col_vol, load, col_load, col_body, body)."""
    col_vol = np.empty(COLUMNS)
    col_load = np.empty(COLUMNS)
    col_body = np.empty(COLUMNS)
    tot_v, tot_b = 0.0, 0.0
    for b in range(COLUMNS):
        v = belly[b, IV] + tip[b, IV]
        cb = belly[b, IB] + tip[b, IB]
        col_vol[b] = v
        col_load[b] = v * COLUMNS / (Wn * Wn)
        col_body[b] = cb / max(v, 1e-30) if v > 0 else 0.0
        tot_v += v
        tot_b += cb
    return col_vol, tot_v / (Wn * Wn), col_load, col_body, tot_b / max(tot_v, 1e-30)


# ----------------------------------------------------------------------------
# the hair of a brush


@njit(cache=True, error_model="numpy")
def lattice(g, a, b):
    """Smooth value noise from the random lattice ``g`` at (a, b), in lattice
    cells (0 <= a, b; clamped at the far edge)."""
    ia = min(int(a), g.shape[0] - 2)
    ib = min(int(b), g.shape[1] - 2)
    fa = min(max(a - ia, 0.0), 1.0)
    fb = min(max(b - ib, 0.0), 1.0)
    fa = fa * fa * (3.0 - 2.0 * fa)
    fb = fb * fb * (3.0 - 2.0 * fb)
    top = g[ia, ib] + (g[ia, ib + 1] - g[ia, ib]) * fb
    bot = g[ia + 1, ib] + (g[ia + 1, ib + 1] - g[ia + 1, ib]) * fb
    return top + (bot - top) * fa


@njit(cache=True, error_model="numpy")
def lattice_cell(n, a):
    """Where ``a`` (in cells) falls on a lattice of ``n`` points: the cell, and
    the smoothed fraction across it (as lattice() works it out)."""
    ia = min(int(a), n - 2)
    f = min(max(a - ia, 0.0), 1.0)
    return ia, f * f * (3.0 - 2.0 * f)


@njit(cache=True, error_model="numpy")
def lattice1(g, a):
    """Smooth 1-D value noise from the random lattice ``g`` at ``a`` cells."""
    ia = min(int(a), g.shape[0] - 2)
    fa = min(max(a - ia, 0.0), 1.0)
    fa = fa * fa * (3.0 - 2.0 * fa)
    return g[ia] + (g[ia + 1] - g[ia]) * fa


@njit(cache=True, error_model="numpy")
def sample_field(f, a, b):
    """Bilinear sample of the 2-D field ``f`` at fractional indices (a, b),
    clamped to its edges."""
    n0, n1 = f.shape
    a = min(max(a, 0.0), n0 - 1.001)
    b = min(max(b, 0.0), n1 - 1.001)
    ia, ib = int(a), int(b)
    fa, fb = a - ia, b - ib
    top = f[ia, ib] + (f[ia, ib + 1] - f[ia, ib]) * fb
    bot = f[ia + 1, ib] + (f[ia + 1, ib + 1] - f[ia + 1, ib]) * fb
    return top + (bot - top) * fa


# Half-width (px) over which a brush's bristle clumps leave the paint together
# as it lifts off (see footprint).
LIFT_CLUMP = 2.0
# How far (px, a Gaussian's sigma) a brush's hair pattern is blurred across
# its lanes (see hair_field).
HAIR_BLUR = 0.6
#: Spread of a brush's hair pattern about its mean of 0.5 (the streaks, the
#: raggedness of the edges and where a dry brush catches are calibrated to it).
HAIR_SPREAD = 0.225


@njit(cache=True, error_model="numpy")
def hair_field(n_s, n_x, ds, dx, cells, grids, lanes, lane_cells, wander):
    """The hair pattern of a brush over one stroke, sampled every ``ds`` px
    along the stroke and ``dx`` px across the bundle: low on the lines of the
    bristle clumps (furrows), high between them (ridges), mean 0.5.

    Across, it is two noise patterns (``lanes``: coarse and fine cells, in
    px), the clumps of two ways the hair can part. Along the stroke the clumps
    drift sideways (by up to ``wander`` px; neighbouring clumps drift
    together, so where they crowd furrows merge and where they part a new one
    opens), the brush now and then regroups from one parting to the other,
    and the clumps bear down more or less (furrows deepen and fade). Each of
    these is smooth noise from a random lattice in ``grids`` with cells
    ``cells`` (along, across) px: drift, finer drift, regrouping, bearing."""
    out = np.empty((n_s, n_x))
    off = 2.0 * wander + 1.0  # (lane coordinates stay positive)
    c_coarse, c_fine = lane_cells[0], lane_cells[1]
    # (where each row and each column falls in each lattice, worked out once:
    # the same numbers lattice() works out at every sample)
    ia = np.empty((4, n_s), np.int64)
    fa = np.empty((4, n_s))
    ib = np.empty((4, n_x), np.int64)
    fb = np.empty((4, n_x))
    for k in range(4):
        n0, n1 = grids[k].shape
        for i in range(n_s):
            ia[k, i], fa[k, i] = lattice_cell(n0, (i * ds) / cells[k, 0])
        for j in range(n_x):
            ib[k, j], fb[k, j] = lattice_cell(n1, (j * dx) / cells[k, 1])
    v4 = np.empty(4)
    tot, tot2 = 0.0, 0.0
    for i in range(n_s):
        for j in range(n_x):
            x = j * dx
            for k in range(4):
                g = grids[k]
                a0, b0, ffa, ffb = ia[k, i], ib[k, j], fa[k, i], fb[k, j]
                top = g[a0, b0] + (g[a0, b0 + 1] - g[a0, b0]) * ffb
                bot = g[a0 + 1, b0] + (g[a0 + 1, b0 + 1] - g[a0 + 1, b0]) * ffb
                v4[k] = top + (bot - top) * ffa
            d = 0.8 * (2.0 * v4[0] - 1.0)
            d += 0.2 * (2.0 * v4[1] - 1.0)
            xx = x + wander * d + off
            pa = 0.62 * lattice1(lanes[0], xx / c_coarse)
            pa += 0.38 * lattice1(lanes[1], xx / c_fine)
            pb = 0.62 * lattice1(lanes[2], xx / c_coarse)
            pb += 0.38 * lattice1(lanes[3], xx / c_fine)
            m = smoothstep(0.4, 0.6, v4[2])
            a = 0.45 + 0.95 * v4[3]
            v = (pa + (pb - pa) * m - 0.5) * a
            out[i, j] = v
            tot += v
            tot2 += v * v
    # Blurred across a little: no lane narrower than a couple of pixels. (Finer
    # ones are lost in the pixels anyway; and a narrow crest running across the
    # pixel grid on a slant would come out as a row of beads, one per pixel it
    # steps over.)
    sig = HAIR_BLUR / dx
    rad = int(math.ceil(2.5 * sig))
    if rad > 0:
        ker = np.empty(2 * rad + 1)
        for k in range(-rad, rad + 1):
            ker[k + rad] = math.exp(-0.5 * (k / sig) ** 2)
        ker /= ker.sum()
        row = np.empty(n_x)
        tot, tot2 = 0.0, 0.0
        for i in range(n_s):
            for j in range(n_x):
                acc = 0.0
                for k in range(-rad, rad + 1):
                    acc += ker[k + rad] * out[i, min(max(j + k, 0), n_x - 1)]
                row[j] = acc
            for j in range(n_x):
                out[i, j] = row[j]
                tot += row[j]
                tot2 += row[j] * row[j]
    n = n_s * n_x
    mean = tot / n
    sd = math.sqrt(max(tot2 / n - mean * mean, 1e-12))
    for i in range(n_s):
        for j in range(n_x):
            out[i, j] = clip(0.5 + (out[i, j] - mean) / sd * HAIR_SPREAD, 0.0, 1.0)
    return out


# ----------------------------------------------------------------------------
# where the tool touches the canvas


@njit(cache=True, error_model="numpy")
def footprint(st, K, tex, tex2, grid, c_tab, h_tab):
    """The footprint of one step on its window: planes F_* of per-pixel
    geometry, exposure and coverage, and whether it exposes any pixel of the
    canvas.

    A pixel's *exposure* is the fraction of a full pass of the footprint over
    it that happens during this step (exact for a convex footprint sweeping a
    straight segment), times the tool's lateral profile: flat-topped, with
    low ridges at the edges when pressed, ragged edges from the hair pattern,
    a long feathered falloff on soft brushes, stippled when dabbed lightly.
    Its *cover* is how much of the pixel the footprint passes over (the
    anti-aliased edge), which is geometry, not paint thickness."""
    F = np.empty((_NF, st.hh, st.ww))
    cx, cy, tx, ty, ds = st.cx, st.cy, st.tx, st.ty, st.ds
    r0, r1, r, p, lmax = st.r0, st.r1, st.r, st.p, st.lmax
    n2 = tex2.shape[0] - 1
    half_aa = 0.5 * EDGE_PX
    soft_pow = 1.0 + 4.0 * K.softness
    span = max(K.feather * 2.0 * r, 1.0)
    hit = False
    for yy in range(st.hh):
        gy = (st.y0 + yy) + 0.5 - cy
        for xx in range(st.ww):
            gx = (st.x0 + xx) + 0.5 - cx
            along = gx * tx + gy * ty  # <= 0 behind the brush centre
            lat_px = -gx * ty + gy * tx
            # the brush size changes smoothly through the segment (no steps when a flat
            # tapers)
            r_px = r1 + (r0 - r1) * clip(-along / ds, 0.0, 1.0) if r0 != r1 else r
            lat = lat_px / r_px  # signed, in brush half-widths
            u = lat * st.uscale  # position across the bristles
            if K.is_dab:
                ix = min(max(int((u + 1.3) / 2.6 * n2), 0), n2)
                iy = min(max(int((along / r + 1.3) / 2.6 * n2), 0), n2)
                hair = tex2[iy, ix]
            else:
                # the bristle clumps (a blade's nicks) where this pixel lies along
                # the stroke and across the bundle (fixed in stroke coordinates:
                # every step sees the same lanes at a pixel)
                hair = sample_field(
                    tex2,
                    (st.arc_i + along - K.hair_s0) / K.hair_ds,
                    (lat_px * st.uscale / st.splay - K.hair_x0) / K.hair_dx,
                )
            lr = abs(lat) / lmax * (1.0 + K.rough * (hair - 0.5))
            # anti-aliasing: how much of each pixel the footprint covers across its edge
            edge_px = (1.0 - lr) * (r_px * lmax)
            aa = smoothstep(-half_aa, half_aa, edge_px)
            falloff = aa
            if K.softness > 0 and not K.is_dab:  # a long, feathery falloff
                falloff *= smoothstep(1.0, 1.0 - K.softness, lr) ** soft_pow
            profile = falloff * (0.92 + K.ridge * p * smoothstep(0.55, 0.95, lr))
            pulled, left_on = 1.0, 1.0
            if st.lift_len > 0:
                # lifting off: over the last stretch of the stroke the bristle
                # clumps leave the paint one after another (a ragged end, not a
                # cut in the shape of the footprint), each pulling the paint up
                # into a crest where it leaves
                s_left = K.arc_end - (st.arc_i + along)
                if s_left < st.lift_len:
                    t = 1.0 - max(s_left, 0.0) / st.lift_len
                    # (a clump a few px wide leaves together: the hair pattern
                    # smoothed across, its spread restored; a clump thins out over
                    # the last part of its way, and a fine single hair doesn't
                    # draw a line of its own)
                    fs = (st.arc_i + along - K.hair_s0) / K.hair_ds
                    fx = (lat_px * st.uscale / st.splay - K.hair_x0) / K.hair_dx
                    dxs = LIFT_CLUMP / K.hair_dx
                    clump = 0.25 * (
                        sample_field(tex2, fs, fx - dxs)
                        + 2.0 * hair
                        + sample_field(tex2, fs, fx + dxs)
                    )
                    clump = 0.5 + 1.6 * (clump - 0.5)
                    g = smoothstep(t - 0.3, t + 0.3, 0.1 + 0.75 * clump)
                    # (thinning to its tip, and pulled up just behind it)
                    profile *= g * math.sqrt(g)
                    left_on = g
                    pulled = 1.0 + st.lift_amp * 6.75 * g * g * (1.0 - g)
            if K.slant > 0:  # bias-cut bristles: the toe (u = -1) touches first
                press = 0.25 + 0.9 * p
                profile *= clip(
                    (press - K.slant * (1.0 + clip(u, -1.0, 1.0)) / 2.0) / 0.3, 0.0, 1.0
                )
            between = 0.0
            # separate clumps of bristles (fan brushes), or the teeth of a comb
            if K.clumps > 0:
                m = 0.5 + 0.5 * math.cos(
                    math.pi * K.clumps * (clip(u, -1.0, 1.0) + 1.0) + 2.0 * (hair - 0.5)
                )
                profile *= smoothstep(0.3, 0.7, m)
                # the gaps between them
                between = falloff * smoothstep(0.6, 0.2, m)
            # chord of the footprint through the pixel along the motion
            if K.circle:
                c = 0.0
                h = max(r_px * math.sqrt(max(1.0 - lr * lr, 0.0)), 0.5)
            else:
                g0, g1 = grid[0], grid[grid.shape[0] - 1]
                c = (
                    interp(lat, g0, g1, c_tab, c_tab[0], c_tab[c_tab.shape[0] - 1])
                    * r_px
                )
                h = max(
                    interp(lat, g0, g1, h_tab, h_tab[0], h_tab[h_tab.shape[0] - 1])
                    * r_px,
                    0.5,
                )
            rel = along - c
            if K.is_dab:
                # pressed straight down, the edge is the same all round: how far
                # out from the middle, not across a stroke
                rr = math.sqrt(
                    lr * lr + clip(rel / h, -1.0, 1.0) ** 2 * max(1.0 - lr * lr, 0.0)
                )
                # the paint builds up a little all round the rim (not in two
                # lips, as along the sides of a stroke)
                profile *= (0.92 + K.ridge * p * smoothstep(0.55, 0.95, rr)) / (
                    0.92 + K.ridge * p * smoothstep(0.55, 0.95, lr)
                )
                if K.softness > 0:  # a soft brush pressed down: feathered all round
                    profile *= smoothstep(1.0, 1.0 - K.softness, rr) ** soft_pow
                # tapped lightly, only the tips of the bristles touch: a stippled,
                # broken mark
                # (a glob of paint on it touches all over even then)
                tip_thr = 0.75 - 1.5 * p - 0.4 * min(st.overload, 1.5)
                if tip_thr > 0:
                    profile *= smoothstep(tip_thr - 0.12, tip_thr + 0.12, hair)
            overlap = max(min(0.0, rel + h) - max(-ds, rel - h), 0.0)
            # where the tool touches down / lifts off, top up to a full pass (solid
            # ends)
            if st.first and K.end_k > 0:
                a0 = rel + ds
                overlap += (
                    K.end_k
                    * smoothstep(-h - half_aa, -h + half_aa, a0)
                    * clip(h - a0, 0.0, 2.0 * h)
                )
            if st.last and K.end_k > 0:
                overlap += (
                    K.end_k
                    * smoothstep(h + half_aa, h - half_aa, rel)
                    * clip(h + rel, 0.0, 2.0 * h)
                )
            swept = overlap / (2.0 * h)  # fraction of a full pass
            if K.is_dab:
                # pressed down once: one full pass at most (where the mark is
                # short, at its two ends, the nudge and the set-down and lift-off
                # would otherwise count it two or three times)
                swept = min(swept, 1.0)
            expo = profile * swept
            if K.feather > 0 and not K.is_dab:
                s_pix = st.arc_i + along  # distance along the stroke
                expo *= smoothstep(-r, -r + span, s_pix) * smoothstep(
                    -r, -r + span, K.arc_end - s_pix
                )
            F[F_ALONG, yy, xx] = along
            F[F_LAT, yy, xx] = lat_px
            F[F_U, yy, xx] = u
            F[F_HAIR, yy, xx] = hair
            F[F_AA, yy, xx] = aa
            F[F_PROFILE, yy, xx] = profile
            F[F_C, yy, xx] = c
            F[F_H, yy, xx] = h
            F[F_REL, yy, xx] = rel
            F[F_OVERLAP, yy, xx] = overlap
            F[F_EXPO, yy, xx] = expo
            # geometric coverage added during this step
            # (where its clumps have left, it covers nothing)
            F[F_COVER, yy, xx] = aa * swept * left_on
            F[F_STREAK, yy, xx] = pulled * max(
                1.0 + K.streaks * 1.6 * (hair - 0.5), 0.0
            )
            F[F_BETWEEN, yy, xx] = between
            F[F_EDGE, yy, xx] = edge_px
            if (
                expo != 0
                and st.wy0 <= yy < st.wy0 + st.ch
                and st.wx0 <= xx < st.wx0 + st.cw
            ):
                hit = True
    return F, hit


@njit(cache=True, error_model="numpy")
def bristle_taps(u_px, span, nsub, ring, rho):
    """Which bristle columns are above each pixel of the (ch, cw) window:
    up to COLUMNS (column, weight) taps per pixel, blending linearly between
    the nearest columns, sampled ``nsub`` times across the pixel."""
    ch, cw = u_px.shape
    npx = ch * cw
    cols = np.empty((npx, COLUMNS), np.int32)
    wts = np.empty((npx, COLUMNS))
    ntap = np.zeros(npx, np.int32)
    acc = np.zeros(COLUMNS)
    for y in range(ch):
        for x in range(cw):
            q = y * cw + x
            acc[:] = 0.0
            for k in range(nsub):
                uc = clip(u_px[y, x] + span * ((k + 0.5) / nsub - 0.5), -1.0, 1.0)
                if ring:
                    # around a round bundle, a point across the mark is touched by the
                    # bristles at the front and the back of the bundle; rolling turns
                    # the whole ring
                    for side in range(2):
                        theta = (
                            -uc * math.pi / 2.0
                            if side == 0
                            else math.pi + uc * math.pi / 2.0
                        )
                        pos = (
                            (theta - rho) / (2.0 * math.pi) * COLUMNS - 0.5
                        ) % COLUMNS
                        c0 = min(int(pos), COLUMNS - 1)
                        fr = pos - c0
                        acc[c0] += 0.5 * (1.0 - fr) / nsub
                        acc[(c0 + 1) % COLUMNS] += 0.5 * fr / nsub
                else:
                    # column b covers an equal slice of the brush, centred at (b + 0.5)
                    # / COLUMNS
                    colf = (uc + 1.0) * 0.5 * COLUMNS - 0.5
                    c0 = int(clip(math.floor(colf), 0.0, COLUMNS - 2.0))
                    fr = clip(colf - c0, 0.0, 1.0)
                    acc[c0] += (1.0 - fr) / nsub
                    acc[c0 + 1] += fr / nsub
            n = 0
            for b in range(COLUMNS):
                if acc[b] != 0.0:
                    cols[q, n] = b
                    wts[q, n] = acc[b]
                    n += 1
            ntap[q] = n
    return cols, wts, ntap


@njit(cache=True, error_model="numpy")
def at_pixels(taps, per_column, ch, cw):
    """A per-column quantity, as seen at each pixel under the brush."""
    cols, wts, ntap = taps
    out = np.empty((ch, cw))
    for y in range(ch):
        for x in range(cw):
            q = y * cw + x
            v = 0.0
            for t in range(ntap[q]):
                v += wts[q, t] * per_column[cols[q, t]]
            out[y, x] = v
    return out


@njit(cache=True, error_model="numpy")
def surface_factors(cv, aux, st, K):
    """How the bare surface changes pickup, deposit and mixing (per pixel) and
    drag: where the brush works directly on the priming (little paint below),
    the surface itself decides how paint grips and slides."""
    top, und = cv[0], cv[1]
    dry_h = aux[3]
    ch, cw = st.ch, st.cw
    f_pick = np.ones((ch, cw))
    f_dep = np.ones((ch, cw))
    f_mix = np.ones((ch, cw))
    if not K.surface_acts:
        return f_pick, f_dep, f_mix, 1.0
    sl, ab, to = K.slickness, K.absorbency, K.surface_tooth
    sum_bare = 0.0
    for y in range(ch):
        for x in range(cw):
            Y, X = st.cy0 + y, st.cx0 + x
            wet = top[Y, X, IV] + und[Y, X, IV]
            dh = dry_h[Y, X]
            bare = math.exp(-(wet + dh) / 0.4)
            # lifting wet paint depends on how it adheres to what is beneath: it wipes
            # cleanly off a slick surface but stays stained into an absorbent one
            on_ground = math.exp(-wet / 1.5) * math.exp(-dh / 0.4)
            f_pick[y, x] = max(1.0 + on_ground * (2.0 * (sl - 0.5) - 0.8 * ab), 0.1)
            f_dep[y, x] = max(
                1.0 + bare * (0.6 * ab + 0.4 * (to - 0.5) - 0.4 * (sl - 0.5)), 0.4
            )
            f_mix[y, x] = 1.0 - 0.6 * ab * bare
            sum_bare += bare
    return f_pick, f_dep, f_mix, 1.0 + (sl - 0.5) * sum_bare / (ch * cw)


@njit(cache=True, error_model="numpy")
def surface_factors_on(cv, aux, st, part, K):
    """:func:`surface_factors` over the part ``part`` of the step's window
    (see :func:`touched`); drag still from the bare surface over all of it."""
    top, und = cv[0], cv[1]
    dry_h = aux[3]
    ch, cw = part.ch, part.cw
    f_pick = np.ones((ch, cw))
    f_dep = np.ones((ch, cw))
    f_mix = np.ones((ch, cw))
    if not K.surface_acts:
        return f_pick, f_dep, f_mix, 1.0
    sl, ab, to = K.slickness, K.absorbency, K.surface_tooth
    sum_bare = 0.0
    for y in range(st.ch):
        for x in range(st.cw):
            Y, X = st.cy0 + y, st.cx0 + x
            wet = top[Y, X, IV] + und[Y, X, IV]
            dh = dry_h[Y, X]
            bare = math.exp(-(wet + dh) / 0.4)
            sum_bare += bare
            py, px = Y - part.cy0, X - part.cx0
            if not (0 <= py < ch and 0 <= px < cw):
                continue
            on_ground = math.exp(-wet / 1.5) * math.exp(-dh / 0.4)
            f_pick[py, px] = max(1.0 + on_ground * (2.0 * (sl - 0.5) - 0.8 * ab), 0.1)
            f_dep[py, px] = max(
                1.0 + bare * (0.6 * ab + 0.4 * (to - 0.5) - 0.4 * (sl - 0.5)), 0.4
            )
            f_mix[py, px] = 1.0 - 0.6 * ab * bare
    return f_pick, f_dep, f_mix, 1.0 + (sl - 0.5) * sum_bare / (st.ch * st.cw)


@njit(cache=True, error_model="numpy")
def touched(st, F):
    """The part of the step's window on the canvas that the tool passes over
    in this step (where its footprint ``F`` covers a pixel and sweeps over
    it), as a Step whose canvas part (cy0, cx0, ch, cw at wy0, wx0) is that
    part. Everywhere else in the window its exposure and coverage are 0: the
    step adds nothing there, moves nothing and lifts nothing."""
    y_lo, y_hi, x_lo, x_hi = st.ch, -1, st.cw, -1
    for y in range(st.ch):
        for x in range(st.cw):
            wy, wx = st.wy0 + y, st.wx0 + x
            if F[F_AA, wy, wx] != 0 and F[F_OVERLAP, wy, wx] != 0:
                y_lo, y_hi = min(y_lo, y), max(y_hi, y)
                x_lo, x_hi = min(x_lo, x), max(x_hi, x)
    if y_hi < 0:  # (not reached: there is no step without exposure)
        y_lo, y_hi, x_lo, x_hi = 0, -1, 0, -1
    return Step(
        st.i,
        st.first,
        st.last,
        st.touchdown,
        st.x0,
        st.y0,
        st.hh,
        st.ww,
        st.cy0 + y_lo,
        st.cx0 + x_lo,
        y_hi - y_lo + 1,
        x_hi - x_lo + 1,
        st.wy0 + y_lo,
        st.wx0 + x_lo,
        st.cx,
        st.cy,
        st.tx,
        st.ty,
        st.ds,
        st.p,
        st.p_now,
        st.p_before,
        st.r0,
        st.r1,
        st.r,
        st.Wd,
        st.Wn,
        st.side,
        st.uscale,
        st.lmax,
        st.h_tab_max,
        st.arc_i,
        st.lip,
        st.wet,
        st.dry,
        st.span,
        st.nsub,
        st.rho,
        st.splay,
        st.lift_len,
        st.lift_amp,
        st.overload,
    )


# ----------------------------------------------------------------------------
# moving paint on the canvas


@njit(cache=True, error_model="numpy")
def deposit(
    cv, pend, sy0, sx0, dep, d, cover, stroke_id, moved, by_film, reach, has_reach
):
    """Lay paint ``dep`` (packed, per pixel; volume ``d``) on top of the canvas
    window at (sy0, sx0), covering ``cover`` more of each pixel.

    Each pixel holds two wet layers. New paint goes on top and the old top
    joins the layer below. Paint that is only being shoved around (``moved``:
    pushed ahead of a knife, squeezed into a ridge) instead merges with
    whichever layer it is most like, so that, say, white pushed over white
    stays a white skin over the purple beneath it.

    With ``by_film``, how much of the old top goes under depends on how well
    the new paint covers it (:func:`buries`): a covering film buries it, a
    faint trace (the feathered edge of a soft brush) only stirs into it.
    Buried, it would mix with the layer below and show through the faint new
    film: white over wet blue would turn blue along every soft edge. Where the
    body of the brush passes over (``reach``: this step's share of a full pass
    at full strength), even a thin film goes on top: a light touch lays paint
    over wet paint. (Where the two old layers are much the same paint it goes
    under anyway.) During a brushstroke this waits until the stroke has passed
    (``pend``, see brush._pending): a pixel first grazed by the edge of a
    looping stroke may later get its body."""
    top, und, top_a, und_a, ids = cv
    paint, pcov, prch, pon, pinfo = pend
    holding = pinfo[0] != 0 and by_film and not moved
    hh, ww = d.shape
    for y in range(hh):
        Y = sy0 + y
        for x in range(ww):
            X = sx0 + x
            dd = d[y, x]
            touched = dd > 1e-7
            old = touched and ids[Y, X] != stroke_id
            laid = False  # nothing more to lay here in this step
            if moved and old and top[Y, X, IV] > 1e-6 and und[Y, X, IV] > 1e-6:
                # three layers in two places: merge the two most alike. The new paint
                # joins the top layer only if it really is much the same paint (white
                # pushed onto white); a different colour stays on top of the others.
                near = alike(top[Y, X], dep[y, x])
                if near < SAME_PAINT and near < alike(und[Y, X], top[Y, X]):
                    for c in range(PACKED):
                        top[Y, X, c] += np.float32(dep[y, x, c])
                    laid, touched, old = True, False, False
            # an earlier top layer worth keeping out of the layer below: hold the new
            # paint back until we know how well the stroke covers it
            hold = False
            known_harmless = False  # (as worked out here: it is the same below)
            if (
                holding
                and old
                and pinfo[1] <= Y < pinfo[1] + pinfo[3]
                and pinfo[2] <= X < pinfo[2] + pinfo[4]
                and top[Y, X, IV] > 1e-4
            ):
                known_harmless = harmless(top[Y, X], und[Y, X])
                hold = not known_harmless
            if hold:
                py, px = Y - pinfo[1], X - pinfo[2]
                for c in range(PACKED):
                    paint[py, px, c] += np.float32(dep[y, x, c])
                pcov[py, px] += cover[y, x]
                if has_reach:
                    prch[py, px] += reach[y, x]
                pon[py, px] = True
                # (where paint is held: rows pinfo[5]..pinfo[6], columns
                # pinfo[7]..pinfo[8])
                pinfo[5], pinfo[6] = min(pinfo[5], py), max(pinfo[6], py + 1)
                pinfo[7], pinfo[8] = min(pinfo[7], px), max(pinfo[8], px + 1)
                # the body of the brush has been over it, or it laid a covering film:
                # bury the old top now
                if buried(paint[py, px, IV], prch[py, px]) >= BURIED:
                    for c in range(PACKED):
                        und[Y, X, c] += top[Y, X, c]
                        top[Y, X, c] = paint[py, px, c]
                        paint[py, px, c] = 0.0
                    und_a[Y, X] = max(und_a[Y, X], top_a[Y, X])
                    top_a[Y, X] = min(pcov[py, px], 1.0)
                    ids[Y, X] = stroke_id
                    pcov[py, px] = 0.0
                    prch[py, px] = 0.0
                    pon[py, px] = False
                laid, touched, old = True, False, False
            if old:
                # paint from an earlier stroke stops being "top" once new paint covers
                # it
                if not by_film:
                    for c in range(PACKED):
                        und[Y, X, c] += top[Y, X, c]
                        top[Y, X, c] = 0.0
                    und_a[Y, X] = max(und_a[Y, X], top_a[Y, X])
                    top_a[Y, X] = 0.0
                else:
                    k = buries(dd)
                    if has_reach:
                        k = max(k, smoothstep(BODY0, BODY1, reach[y, x]))
                    if k < 1.0 and (known_harmless or harmless(top[Y, X], und[Y, X])):
                        k = 1.0
                    k32 = np.float32(k)
                    for c in range(PACKED):
                        going = k32 * top[Y, X, c]
                        und[Y, X, c] += going
                        top[Y, X, c] -= going
                    if k32 > 0.01:
                        und_a[Y, X] = max(und_a[Y, X], top_a[Y, X])
                    top_a[Y, X] *= np.float32(1.0) - k32
                ids[Y, X] = stroke_id
            # the tool covers what it passes over with this stroke's paint, also where
            # it adds nothing new in this step (it already laid paint there)
            if touched or (ids[Y, X] == stroke_id and top[Y, X, IV] > 1e-6):
                top_a[Y, X] = min(
                    top_a[Y, X] + np.float32(cover[y, x]), np.float32(1.0)
                )
            if not laid:
                for c in range(PACKED):
                    top[Y, X, c] += np.float32(dep[y, x, c])


@njit(cache=True, error_model="numpy")
def settle_pending(cv, pend, stroke_id):
    """The stroke has passed: settle the pixels its edge only grazed (see
    brush._pending). The old top keeps its place, the share the faint new
    film would bury goes under, and the new paint stirs into the rest."""
    top, und, top_a, und_a, ids = cv
    paint, pcov, prch, pon, pinfo = pend
    if pinfo[0] == 0:
        return
    for py in range(pinfo[5], pinfo[6]):
        for px in range(pinfo[7], pinfo[8]):
            if not pon[py, px]:
                continue
            Y, X = pinfo[1] + py, pinfo[2] + px
            k = buried(paint[py, px, IV], prch[py, px])
            if k < 1.0 and harmless(top[Y, X], und[Y, X]):
                k = 1.0
            k32 = np.float32(k)
            for c in range(PACKED):
                going = k32 * top[Y, X, c]
                und[Y, X, c] += going
                top[Y, X, c] += paint[py, px, c] - going
            if k32 > 0.01:
                und_a[Y, X] = max(und_a[Y, X], top_a[Y, X])
            top_a[Y, X] = max(
                top_a[Y, X] * (np.float32(1.0) - k32),
                min(pcov[py, px], np.float32(1.0)),
            )
            ids[Y, X] = stroke_id


@njit(cache=True, error_model="numpy")
def share_by_reach(tip, taps, d):
    """A brush pressed straight down splays its bristles over the whole mark:
    each column gives its paint over the part of the mark it reaches. Shares
    the paint in the tips out between the columns in proportion to what is
    asked of each (a round's middle columns reach right across the mark, its
    edge columns only the short ends), each keeping its own paint; the total
    stays the same. Otherwise the ends of the mark, fed by full columns, would
    come out thicker than its middle."""
    cols, wts, ntap = taps
    hh, ww = d.shape
    demand = np.zeros(COLUMNS)
    for y in range(hh):
        for x in range(ww):
            q = y * ww + x
            for t in range(ntap[q]):
                demand[cols[q, t]] += wts[q, t] * d[y, x]
    held, asked = 0.0, 0.0
    for b in range(COLUMNS):
        if tip[b, IV] > 0:
            held += tip[b, IV]
            asked += demand[b]
    if held <= 0 or asked <= 0:
        return
    for b in range(COLUMNS):
        tv = tip[b, IV]
        if tv > 0:
            f = held * demand[b] / asked / tv
            for c in range(PACKED):
                tip[b, c] *= f


# Share of the glob on a brush that a dab presses out (at medium pressure).
DAB_GLOB = 0.18


@njit(cache=True, error_model="numpy")
def press_glob(d, F, st, part, col_vol, col_over, hold, pattern):
    """A loaded brush pressed straight down (a dab) leaves part of the glob
    it carries as a mound under it: highest in the middle, flatter the harder
    it is pressed, lumpy where its tips (``pattern``) touched. Adds it to the
    volume ``d`` the dab asks for (over ``part`` of its footprint ``F``)."""
    glob = 0.0
    for b in range(COLUMNS):
        if col_over[b] > hold:
            glob += col_vol[b] * (col_over[b] - hold) / col_over[b]
    if glob <= 0:
        return
    p = st.p
    vol = DAB_GLOB * (0.6 + 0.8 * p) * glob
    a = 1.6 - 1.2 * p  # how domed
    ch, cw = d.shape
    w = np.zeros((ch, cw))
    tot = 0.0
    n2 = pattern.shape[0] - 1
    for y in range(ch):
        for x in range(cw):
            wy, wx = part.wy0 + y, part.wx0 + x
            aa = F[F_AA, wy, wx]
            if aa <= 0 or F[F_OVERLAP, wy, wx] <= 0:
                continue
            lr = min(abs(F[F_LAT, wy, wx]) / max(st.r * st.lmax, 1e-9), 1.0)
            rel = clip(F[F_REL, wy, wx] / max(F[F_H, wy, wx], 1e-9), -1.0, 1.0)
            rr = min(math.sqrt(lr * lr + rel * rel * (1.0 - lr * lr)), 1.0)
            u = F[F_U, wy, wx]
            ix = min(max(int((u + 1.3) / 2.6 * n2), 0), n2)
            iy = min(max(int((F[F_ALONG, wy, wx] / st.r + 1.3) / 2.6 * n2), 0), n2)
            v = aa * (1.0 - rr * rr) ** a * (0.6 + 0.8 * pattern[iy, ix])
            w[y, x] = v
            tot += v
    if tot <= 0:
        return
    for y in range(ch):
        for x in range(cw):
            d[y, x] += vol * w[y, x] / tot


@njit(cache=True, error_model="numpy")
def feed_glob(tip, belly, taps, d, col_glob):
    """A glob of paint sits on the outside of the hair, not in its tips: where
    a column carries one, what this step asks of it beyond what its tip holds
    comes straight from the glob (the belly). So how thick an overloaded
    brush lays follows its load, and doesn't depend on how many steps the
    stroke is cut into (fed only from its tip, refilled once a step, a column
    could give no more per step than the tip holds: a cap that a stroke cut
    into finer steps, a curve, got past)."""
    cols, wts, ntap = taps
    hh, ww = d.shape
    demand = np.zeros(COLUMNS)
    for y in range(hh):
        for x in range(ww):
            q = y * ww + x
            for t in range(ntap[q]):
                demand[cols[q, t]] += wts[q, t] * d[y, x]
    for b in range(COLUMNS):
        need = demand[b] - tip[b, IV]
        have = belly[b, IV]
        if col_glob[b] > 0 and need > 0 and have > 0:
            f = min(need / have, 1.0)
            for c in range(PACKED):
                m = belly[b, c] * f
                tip[b, c] += m
                belly[b, c] -= m


@njit(cache=True, error_model="numpy")
def lay(
    cv, pend, br, taps, sy0, sx0, d, cover, stroke_id, moved, by_film, reach, has_reach
):
    """Lay volume ``d`` per pixel from the tips of the columns above it; each
    column can only give what it holds."""
    tip = br[0]
    cols, wts, ntap = taps
    hh, ww = d.shape
    demand = np.zeros(COLUMNS)
    for y in range(hh):
        for x in range(ww):
            q = y * ww + x
            for t in range(ntap[q]):
                demand[cols[q, t]] += wts[q, t] * d[y, x]
    give = np.empty(COLUMNS)
    comp = np.empty((COLUMNS, PACKED))  # paint per unit volume
    for b in range(COLUMNS):
        tv = tip[b, IV]
        give[b] = tv / max(demand[b], 1e-30) if demand[b] > tv else 1.0
        for c in range(PACKED):
            comp[b, c] = tip[b, c] / (tv if tv > 0 else 1.0)
    dep = np.zeros((hh, ww, PACKED))
    d_px = np.empty((hh, ww))
    used = np.zeros(COLUMNS)
    total = 0.0
    for y in range(hh):
        for x in range(ww):
            q = y * ww + x
            v = 0.0
            for t in range(ntap[q]):
                b = cols[q, t]
                # volume from column b at this pixel
                part = wts[q, t] * give[b] * d[y, x]
                v += part
                used[b] += part
                for c in range(PACKED):
                    dep[y, x, c] += part * comp[b, c]
            d_px[y, x] = v
            total += v
    if total > 0:
        deposit(
            cv,
            pend,
            sy0,
            sx0,
            dep,
            d_px,
            cover,
            stroke_id,
            moved,
            by_film,
            reach,
            has_reach,
        )
        for b in range(COLUMNS):
            for c in range(PACKED):
                tip[b, c] -= used[b] * comp[b, c]
            tip[b, IV] = max(tip[b, IV], 0.0)


@njit(cache=True, error_model="numpy")
def take_tips(tip, taps, d):
    """Take paint off the tips of the columns above each pixel, ``d`` per
    pixel (each column giving what it holds at most), and return it packed."""
    cols, wts, ntap = taps
    hh, ww = d.shape
    demand = np.zeros(COLUMNS)
    for y in range(hh):
        for x in range(ww):
            q = y * ww + x
            for t in range(ntap[q]):
                demand[cols[q, t]] += wts[q, t] * d[y, x]
    out = np.zeros(PACKED)
    for b in range(COLUMNS):
        tv = tip[b, IV]
        if demand[b] <= 0 or tv <= 0:
            continue
        f = min(demand[b] / tv, 1.0)
        for c in range(PACKED):
            m = tip[b, c] * f
            out[c] += m
            tip[b, c] -= m
    return out


@njit(cache=True, error_model="numpy")
def splat_out(d, lat, along, oy, ox, hh, ww, tx, ty, ds, s0, s_end, half, out):
    """Where paint pressed out at a blade's ends lands: ``d`` per pixel (a
    window whose top-left pixel is ``oy``, ``ox`` px into the (hh, ww) window
    returned), at lateral position ``lat`` and ``along`` the motion (tx, ty),
    moves out to within ``out`` px beyond the end of the blade (``half`` px
    from its middle), smeared along the stroke over two steps each way (a
    triangle; not past the stroke's ends, ``s0 + along`` within 0..s_end).
    Returns the weights on the left and the right (lat < 0, > 0), bilinear."""
    wl = np.zeros((hh, ww))
    wr = np.zeros((hh, ww))
    h, w = d.shape
    for y in range(h):
        for x in range(w):
            v = d[y, x]
            if v <= 1e-9:
                continue
            sgn = 1.0 if lat[y, x] > 0 else -1.0
            tgt = wr if lat[y, x] > 0 else wl
            inside = half - abs(lat[y, x])
            s = s0 + along[y, x]
            for k in range(3):
                fk = 0.3 if k == 0 else (0.45 if k == 1 else 0.25)
                o = inside + out * (k + 0.5) / 3.0
                for j in range(-4, 5):
                    fa = (5.0 - abs(j)) / 25.0
                    a = j / 2.0 * ds
                    if s + a < 0.0 or s + a > s_end:
                        a = 0.0
                    fy = oy + y + sgn * o * tx + a * ty
                    fx = ox + x - sgn * o * ty + a * tx
                    iy, ix = int(math.floor(fy)), int(math.floor(fx))
                    ay, ax = fy - iy, fx - ix
                    m = v * fk * fa
                    for dy in range(2):
                        wy = ay if dy else 1.0 - ay
                        jy = iy + dy
                        if jy < 0 or jy >= hh:
                            continue
                        for dx in range(2):
                            wx = ax if dx else 1.0 - ax
                            jx = ix + dx
                            if 0 <= jx < ww:
                                tgt[jy, jx] += m * wy * wx
    return wl, wr


@njit(cache=True, error_model="numpy")
def spread(cv, pend, br, sy0, sx0, w, pk, cover, stroke_id, by_film):
    """Lay the packed paint ``pk`` over the window at (sy0, sx0) in proportion
    to the weights ``w`` (a squeezed-out ridge covering ``cover`` of each
    pixel). If none of it lands on the canvas, it stays on the tool."""
    if pk[IV] <= 0:
        return
    tot = w.sum()
    if tot <= 1e-9:
        tip = br[0]
        for b in range(COLUMNS):
            for c in range(PACKED):
                tip[b, c] += pk[c] / COLUMNS
        return
    # (only where it lands: elsewhere there is nothing to lay, and it covers
    # nothing)
    hh, ww = w.shape
    y0, y1, x0, x1 = hh, 0, ww, 0
    for y in range(hh):
        for x in range(ww):
            if w[y, x] != 0 or cover[y, x] != 0:
                y0, y1 = min(y0, y), max(y1, y + 1)
                x0, x1 = min(x0, x), max(x1, x + 1)
    if y0 >= y1:
        return
    dep = np.empty((y1 - y0, x1 - x0, PACKED))
    d = np.empty((y1 - y0, x1 - x0))
    for y in range(y0, y1):
        for x in range(x0, x1):
            share = w[y, x] / tot
            d[y - y0, x - x0] = share * pk[IV]
            for c in range(PACKED):
                dep[y - y0, x - x0, c] = share * pk[c]
    deposit(
        cv,
        pend,
        sy0 + y0,
        sx0 + x0,
        dep,
        d,
        cover[y0:y1, x0:x1],
        stroke_id,
        True,
        by_film,
        d,
        False,
    )


@njit(cache=True, error_model="numpy")
def shift(a, sx, sy):
    """Bilinear translation: out[y, x] = a[y - sy, x - sx], zero outside."""
    ix, iy = int(math.floor(sx)), int(math.floor(sy))
    fx, fy = sx - ix, sy - iy
    h, w = a.shape[0], a.shape[1]
    out = np.zeros_like(a)
    for kx in range(2):
        dx, wx = (ix, 1.0 - fx) if kx == 0 else (ix + 1, fx)
        for ky in range(2):
            dy, wy = (iy, 1.0 - fy) if ky == 0 else (iy + 1, fy)
            wgt = np.float32(wx * wy)
            if wx * wy < 1e-6 or abs(dx) >= w or abs(dy) >= h:
                continue
            # (a row at a time: its pixels' channels lie in one run)
            C = a.shape[2]
            x_lo, x_hi = max(dx, 0), min(w + dx, w)
            n = (x_hi - x_lo) * C
            for y in range(max(dy, 0), min(h + dy, h)):
                src = a[y - dy].reshape(-1)[(x_lo - dx) * C :]
                dst = out[y].reshape(-1)[x_lo * C :]
                for i in range(n):
                    dst[i] += wgt * src[i]
    return out


@njit(cache=True, error_model="numpy")
def take(top, und, sy0, sx0, excess, tv):
    """Take volume ``excess`` per pixel off the wet paint of the window at
    (sy0, sx0), from the top layer (of volume ``tv``) first, and return it (packed)."""
    hh, ww = excess.shape
    moved = np.empty((hh, ww, PACKED))
    for y in range(hh):
        for x in range(ww):
            Y, X = sy0 + y, sx0 + x
            t, uv = tv[y, x], und[Y, X, IV]
            from_top = min(excess[y, x], t)
            from_und = clip(excess[y, x] - from_top, 0.0, uv)
            ft = np.float32(from_top / (t if t > 0 else 1.0))
            fu = np.float32(from_und / (uv if uv > 0 else 1.0))
            for c in range(PACKED):
                mt, mu = top[Y, X, c] * ft, und[Y, X, c] * fu
                moved[y, x, c] = mt + mu
                top[Y, X, c] -= mt
                und[Y, X, c] -= mu
    return moved


@njit(cache=True, error_model="numpy")
def place(cv, pend, pk, x0, y0, sx, sy, stroke_id):
    """Put the packed paint ``pk`` (a window whose top-left pixel is (x0, y0))
    back on the canvas, moved by (sx, sy) pixels: paint pushed ahead of the tool."""
    H, W = cv[0].shape[0], cv[0].shape[1]
    m = int(math.ceil(max(abs(sx), abs(sy)))) + 1
    h, w = pk.shape[0], pk.shape[1]
    rx0, ry0 = max(x0 - m, 0), max(y0 - m, 0)
    rx1, ry1 = min(x0 + w + m, W), min(y0 + h + m, H)
    buf = np.zeros((ry1 - ry0, rx1 - rx0, PACKED), np.float32)
    for y in range(h):
        for x in range(w):
            for c in range(PACKED):
                buf[y0 - ry0 + y, x0 - rx0 + x, c] = pk[y, x, c]
    moved = shift(buf, sx, sy)
    d = np.empty((ry1 - ry0, rx1 - rx0))
    dmax = 0.0
    for y in range(d.shape[0]):
        for x in range(d.shape[1]):
            d[y, x] = moved[y, x, IV]
            dmax = max(dmax, d[y, x])
    if dmax <= 0:
        return
    # moved paint doesn't change how much of each pixel is covered
    deposit(
        cv, pend, ry0, rx0, moved, d, np.zeros_like(d), stroke_id, True, True, d, False
    )


@njit(cache=True, error_model="numpy")
def drag(top, und, f, x0, y0, sx, sy, p, wx0, wy0, wh, ww):
    """Move a fraction ``f`` of the wet paint by (sx, sy) pixels. ``f`` is
    defined on the footprint window whose top-left pixel is (x0, y0) (all of
    it, or the part of it where ``f`` isn't 0 of the whole window at (wx0,
    wy0), wh by ww px: a layer moves only if there is wet paint in that). Paint
    is conserved except what is pushed off the canvas. The older layer moves
    less."""
    H, W = top.shape[0], top.shape[1]
    m = int(math.ceil(max(abs(sx), abs(sy)))) + 1
    fh, fw = f.shape
    rx0, ry0 = max(x0 - m, 0), max(y0 - m, 0)
    rx1, ry1 = min(x0 + fw + m, W), min(y0 + fh + m, H)
    fx0, fy0 = max(x0, 0), max(y0, 0)
    fx1, fy1 = min(x0 + fw, W), min(y0 + fh, H)
    if fx0 >= fx1 or fy0 >= fy1:
        return
    # where to look for wet paint: around the whole window
    hx0, hy0 = max(wx0 - m, 0), max(wy0 - m, 0)
    hx1, hy1 = min(wx0 + ww + m, W), min(wy0 + wh + m, H)
    rh, rw = ry1 - ry0, rx1 - rx0
    frac = np.zeros((rh, rw), np.float32)
    for Y in range(fy0, fy1):
        for X in range(fx0, fx1):
            frac[Y - ry0, X - rx0] = f[Y - y0, X - x0]
    for li in range(2):
        layer = top if li == 0 else und
        scale = np.float32(1.0 if li == 0 else p)
        if scale <= 0:
            continue
        has = False
        for Y in range(hy0, hy1):
            for X in range(hx0, hx1):
                if layer[Y, X, IV] != 0:
                    has = True
                    break
            if has:
                break
        if not has:
            continue
        moved = np.zeros((rh, rw, PACKED), np.float32)
        for y in range(rh):
            for x in range(rw):
                fr = frac[y, x] * scale
                if fr != 0:
                    for c in range(PACKED):
                        moved[y, x, c] = fr * layer[ry0 + y, rx0 + x, c]
        shifted = shift(moved, sx, sy)
        n = rw * PACKED
        for y in range(rh):
            # (a row at a time: its pixels' channels lie in one run)
            row = layer[ry0 + y, rx0 : rx0 + rw].reshape(-1)
            s_row, m_row = shifted[y].reshape(-1), moved[y].reshape(-1)
            for i in range(n):
                row[i] += s_row[i] - m_row[i]


@njit(cache=True, error_model="numpy")
def stir(cv, sy0, sx0, m):
    """Homogenise the two wet sublayers of the window at (sy0, sx0) by a
    fraction ``m`` (0 = untouched, 1 = both become the same mixture). Each
    sublayer keeps its volume."""
    top, und, top_a, und_a = cv[0], cv[1], cv[2], cv[3]
    hh, ww = m.shape
    for y in range(hh):
        for x in range(ww):
            Y, X = sy0 + y, sx0 + x
            mm = np.float32(m[y, x])
            tv, uv = top[Y, X, IV], und[Y, X, IV]
            if mm == 0 or not (tv > 0 and uv > 0):
                continue
            if mm > 0.01:
                und_a[Y, X] = max(und_a[Y, X], top_a[Y, X])
            share = tv / max(tv + uv, np.float32(1e-12))
            # top <- (1 - m) * top + m * share * (top + under); under keeps the rest
            for c in range(PACKED):
                total = top[Y, X, c] + und[Y, X, c]
                new_top = (np.float32(1.0) - mm) * top[Y, X, c] + (mm * share) * total
                top[Y, X, c] = new_top
                und[Y, X, c] = total - new_top


@njit(cache=True, error_model="numpy")
def merge_same(cv, sy0, sx0, own, thr):
    """Where both wet layers of the window are much the same paint (not the
    stroke's own), merge them into the lower one."""
    top, und, top_a, und_a = cv[0], cv[1], cv[2], cv[3]
    hh, ww = own.shape
    for y in range(hh):
        for x in range(ww):
            Y, X = sy0 + y, sx0 + x
            if own[y, x] or not (top[Y, X, IV] > 1e-6 and und[Y, X, IV] > 1e-6):
                continue
            if alike(top[Y, X], und[Y, X]) < thr:
                for c in range(PACKED):
                    und[Y, X, c] += top[Y, X, c]
                    top[Y, X, c] = 0.0
                und_a[Y, X] = max(und_a[Y, X], top_a[Y, X])
                top_a[Y, X] = 0.0


# ----------------------------------------------------------------------------
# paint squeezed or pushed out of the footprint: low ridges along its edges


@njit(cache=True, error_model="numpy")
def add_sides(ridges, moved, lat, scale):
    """Paint pushed out to the left and right edges of the footprint."""
    left, right = ridges[1], ridges[2]
    hh, ww = lat.shape
    for y in range(hh):
        for x in range(ww):
            side = right if lat[y, x] > 0 else left
            for c in range(PACKED):
                side[c] += scale * moved[y, x, c]


@njit(cache=True, error_model="numpy")
def squeeze_out(br, K, st, col_vol, col_load, hold, ridges, col_u):
    """An overloaded tool pressed down squeezes paint out at its edges,
    unevenly: the lips along a stroke swell and break. Only the paint under a
    knife's blade gets squeezed, not the whole glob riding on it."""
    tip, belly = br[0], br[1]
    ring, left, right = ridges
    over = np.empty(COLUMNS)
    anything = False
    for b in range(COLUMNS):
        over[b] = max(col_load[b] - hold, 0.0)
        if K.rigid:
            over[b] = min(over[b], hold + 1.0)
        anything = anything or over[b] > 0
    if not (K.deposits and anything):
        return
    p = st.p
    rate = (
        (SQUEEZE_KNIFE if K.rigid else SQUEEZE_BRUSH)
        * (0.2 + p * p)
        * (0.5 + K.stiffness)
        * st.ds
        / st.Wd
        * (0.2 + 1.6 * st.lip)
    )
    if st.lift_len > 0:
        # (lifting off, the clumps press less and less out at the edges)
        rate *= clip((K.arc_end - st.arc_i) / st.lift_len, 0.0, 1.0)
    if st.touchdown:
        rate += 0.03 * (0.3 + p)
    for b in range(COLUMNS):
        tk = min(over[b] * (st.Wn * st.Wn / COLUMNS) * rate, col_vol[b])
        frac = tk / max(col_vol[b], 1e-30) if col_vol[b] > 0 else 0.0
        w_right = smoothstep(-0.3, 0.3, col_u[b] * st.side)
        for c in range(PACKED):
            out = (tip[b, c] + belly[b, c]) * frac
            tip[b, c] -= tip[b, c] * frac
            belly[b, c] -= belly[b, c] * frac
            if st.touchdown:
                ring[c] += out
            else:
                right[c] += out * w_right
                left[c] += out * (1.0 - w_right)


@njit(cache=True, error_model="numpy")
# cross-section of a squeezed-out ridge of width w, just outside the edge
def _tent(e, w):
    return clip(1.0 - abs(e - 0.35 * w) / (0.65 * w + 0.5), 0.0, 1.0)


@njit(cache=True, error_model="numpy")
# cross-section of a ridge a brush pushed out, of width w: steep on the side
# of the stroke, a longer slope outside
def _ridge(e, w):
    peak = 0.25 * w
    if e < peak:
        return clip(1.0 - (peak - e) / (0.3 * w + 0.5), 0.0, 1.0)
    return clip(1.0 - (e - peak) / (0.75 * w + 0.5), 0.0, 1.0)


@njit(cache=True, error_model="numpy")
# how much of each pixel the ridge covers (anti-aliased outer edge)
def _geo(e, w):
    return clip(w - e, 0.0, 1.0) if e > -0.3 * w else 0.0


@njit(cache=True, error_model="numpy")
def lay_ridges(cv, pend, br, K, st, ridges, grid, c_tab, h_tab, hair):
    """Put the paint pushed out of the footprint on the canvas, as low ridges
    just outside its edges: all around it (when the tool is set down) or along
    its left and right sides.

    Where a loaded brush is set down at the start of a stroke, what it
    squeezes out is a bead around where it landed, as wide as it is high:
    behind it and at its sides (the brush then moves on over the front of
    it), lumpy where the bristle clumps pressed it out (``hair``, see
    hair_field). A dab, pressed straight down and lifted, leaves a rim all
    round."""
    ring, left, right = ridges
    if not (ring[IV] > 0 or left[IV] > 0 or right[IV] > 0):
        return
    H, W = cv[0].shape[0], cv[0].shape[1]
    margin = K.margin
    # the ridges lie just outside the footprint: work on a slightly larger window
    m_px = int(math.ceil(margin))
    ey0, ex0 = max(st.y0 - m_px, 0), max(st.x0 - m_px, 0)
    ey1, ex1 = min(st.y0 + st.hh + m_px, H), min(st.x0 + st.ww + m_px, W)
    if ey0 >= ey1 or ex0 >= ex1:
        return
    eh, ew = ey1 - ey0, ex1 - ex0
    e_along = np.empty((eh, ew))
    e_lat = np.empty((eh, ew))
    e_r = np.empty((eh, ew))
    for y in range(eh):
        gy = (ey0 + y) + 0.5 - st.cy
        for x in range(ew):
            gx = (ex0 + x) + 0.5 - st.cx
            a = gx * st.tx + gy * st.ty
            e_along[y, x] = a
            e_lat[y, x] = -gx * st.ty + gy * st.tx
            e_r[y, x] = (
                st.r1 + (st.r0 - st.r1) * clip(-a / st.ds, 0.0, 1.0)
                if st.r0 != st.r1
                else st.r
            )
    lmax, by_film = st.lmax, not K.rigid
    # a soft tool's paint fades out before its outline: the squeezed-out paint
    # lies against where the mark fades, not out at the outline
    inset = 0.6 * K.softness * st.r
    g0, g1 = grid[0], grid[grid.shape[0] - 1]
    nt = c_tab.shape[0]
    w_t = np.empty((eh, ew))
    w_g = np.empty((eh, ew))
    if ring[IV] > 0:
        # a bead where a stroke starts; all around a dab or a knife
        bead = not (K.is_dab or K.rigid)
        # all around the footprint: as wide as its narrow side allows
        w = clip(
            0.24 * st.r * min(lmax, 1.0 if K.circle else st.h_tab_max),
            0.6,
            margin - 1.0,
        )
        set_down, hb = 0.0, 1.0
        if bead:
            # around where the brush landed, at the start of this first step
            set_down = st.ds
            # the length of the back and the sides of the footprint
            if K.circle:
                rim_len = 2.0 * math.pi * st.r * 0.75
            else:
                rim_len = 2.0 * st.r * (lmax + 2.0 * st.h_tab_max)
            # about as high (1 unit of paint ~ 0.6 px) as it is wide
            w = clip(1.4 * math.sqrt(ring[IV] / max(rim_len, 1e-6)), 1.5, margin - 1.0)
            hb = max(0.5 * st.r * (1.0 if K.circle else st.h_tab_max), 1.0)
        for y in range(eh):
            for x in range(ew):
                ea = e_along[y, x] + set_down
                if K.circle:
                    e = math.sqrt(e_lat[y, x] ** 2 + ea**2) - e_r[y, x]
                else:
                    el = e_lat[y, x] / e_r[y, x]
                    e_c = interp(el, g0, g1, c_tab, c_tab[0], c_tab[nt - 1]) * e_r[y, x]
                    e_h = interp(el, g0, g1, h_tab, 0.0, 0.0) * e_r[y, x]
                    # how far outside: rounded past the ends of the mark, not
                    # squared off (which left lips sticking out there)
                    a = abs(e_lat[y, x]) - e_r[y, x] * lmax
                    b = abs(ea - e_c) - e_h
                    e = math.sqrt(max(a, 0.0) ** 2 + max(b, 0.0) ** 2) + min(
                        max(a, b), 0.0
                    )
                if bead:
                    # a mound across the edge, more of it outside, gone at the front
                    t = (e + inset - 0.3 * w) / w
                    back = smoothstep(hb, -hb, ea)
                    lump = 0.35 + 1.3 * sample_field(
                        hair,
                        (st.arc_i + e_along[y, x] - K.hair_s0) / K.hair_ds,
                        (e_lat[y, x] * st.uscale / st.splay - K.hair_x0) / K.hair_dx,
                    )
                    w_t[y, x] = max(1.0 - t * t, 0.0) * (0.03 + 0.97 * back) * lump
                    w_g[y, x] = _geo(e + inset, 1.3 * w)
                else:
                    w_t[y, x] = _tent(e + inset, w)
                    w_g[y, x] = _geo(e + inset, w)
        spread(cv, pend, br, ey0, ex0, w_t, ring, w_g, K.stroke_id, by_film)
    if left[IV] > 0 or right[IV] > 0:
        w = clip(0.24 * st.r * lmax, 0.6, margin - 1.0)  # scales with the mark's width
        # soft-ended, so that the ridges laid at successive steps run together smoothly
        ramp = 0.5 * st.ds + 0.7
        brushed = not K.rigid
        for k in range(2):
            pk = left if k == 0 else right
            if pk[IV] <= 0:
                continue
            if brushed:
                # a brush's ridge is as wide as the paint in it makes it (about as
                # high as it is wide: 1 unit of paint ~ 0.6 px), from the paint
                # pushed out per px of travel
                w = clip(1.2 * math.sqrt(pk[IV] / max(st.ds, 0.5)), 0.8, margin - 1.0)
            for y in range(eh):
                for x in range(ew):
                    lat = e_lat[y, x]
                    if (k == 0 and lat < 0) or (k == 1 and lat > 0):
                        ce = (
                            0.0
                            if K.circle
                            else (c_tab[nt - 1] if lat > 0 else c_tab[0]) * e_r[y, x]
                        )
                        rel_e = e_along[y, x] - ce
                        box = smoothstep(
                            -st.ds - ramp, -st.ds + ramp, rel_e
                        ) * smoothstep(ramp, -ramp, rel_e)
                        e_side = abs(lat) - e_r[y, x] * lmax + inset
                        if brushed:
                            # steep against the stroke, a longer slope outside; it
                            # swells and breaks where the clumps at the edge of the
                            # brush pushed it out, and skips under a light touch
                            hv = sample_field(
                                hair,
                                (st.arc_i + e_along[y, x] - K.hair_s0) / K.hair_ds,
                                (lat * st.uscale / st.splay - K.hair_x0) / K.hair_dx,
                            )
                            lump = (0.3 + 1.4 * hv) * smoothstep(
                                0.55 - st.p, 0.75 - st.p, hv + 0.3
                            )
                            w_t[y, x] = _ridge(e_side, w) * box * lump
                        else:
                            w_t[y, x] = _tent(e_side, w) * box
                        w_g[y, x] = _geo(e_side, w) * box
                    else:
                        w_t[y, x] = 0.0
                        w_g[y, x] = 0.0
            spread(cv, pend, br, ey0, ex0, w_t, pk, w_g, K.stroke_id, by_film)


# ----------------------------------------------------------------------------
# one step of a brush


@njit(cache=True, error_model="numpy")
def past_tips(h, p, K):
    """Share of the paint a brush lifts that goes past the tips into the belly.
    A thin film of wet paint only wets the tips; deeper paint comes further up
    between the hairs, and pressing bends them (soft hair more) so that more of
    their length lies in it. ``h``: the wet paint from earlier strokes there."""
    if h <= 0.0:
        return 0.0
    reach = h * (0.5 + p * (1.6 - K.stiffness))
    return reach / (reach + TIP_DEPTH * K.smear)


# How strongly a brush's bristles lift its own fresh paint over a pass, for the
# same pickup, relative to old wet paint (see own_lift).
OWN_LIFT = 1.5


@njit(cache=True, error_model="numpy")
def own_lift(g, n):
    """Share of the paint a brush lays that stays on the canvas once its own
    bristles have passed over it. ``g``: how strongly they lift wet paint
    over a whole pass; ``n``: how many times the length of its footprint
    along the stroke is of its nominal step (paint laid at the front of the
    footprint is passed over by the rest of it: a long footprint, a round's,
    lifts more of its own fresh paint back than a flat's thin edge of hair).
    The bristles behind lift some of what those in front just laid and carry
    it on; what a pass leaves is (1 - e^-g) / (n (1 - e^(-g/n))) of what it
    laid. The brush doesn't lift its own fresh paint step by step (see
    pick_up: done per step, that took the most from where successive
    footprints overlapped, a ripple at the step period, and made how much
    paint goes on depend on how finely the stroke is cut): it lays that much
    less instead."""
    if g < 1e-6 or n <= 1.0:
        return 1.0
    return (1.0 - math.exp(-g)) / (n * (1.0 - math.exp(-g / n)))


@njit(cache=True, error_model="numpy")
def pick_up(cv, br, taps, F, st, K, f_pick):
    """The tool lifts wet paint. Bristles plough through it: they lift paint
    along the hair lines and leave it between them (brush marks). Thick fresh
    paint shields the older layer below. What is lifted goes into the columns
    that touched it: into their tips, and from deep wet paint partly further up
    (see past_tips)."""
    top, und, ids = cv[0], cv[1], cv[4]
    tip, belly = br[0], br[1]
    cols, wts, ntap = taps
    p = st.p
    gain = K.pickup * 1.2 * (0.25 + 1.5 * min(p, 1.0))
    if gain <= 0:
        return
    lifted = np.empty(PACKED)
    for y in range(st.ch):
        for x in range(st.cw):
            Y, X = st.cy0 + y, st.cx0 + x
            q = y * st.cw + x
            plough = F[F_EXPO, st.wy0 + y, st.wx0 + x] * max(
                2.0 - F[F_STREAK, st.wy0 + y, st.wx0 + x], 0.0
            )
            k = gain * plough * f_pick[y, x]
            if k == 0:
                continue
            own = ids[Y, X] == K.stroke_id
            # (not the paint this stroke has just laid: that is what the tips gave,
            # see own_lift; it shields the older paint below)
            a_top = np.float32(0.0 if own else 1.0 - math.exp(-k))
            a_und = np.float32(
                (1.0 - math.exp(-k * p)) * math.exp(-top[Y, X, IV] / 1.5)
            )
            other = und[Y, X, IV] + (0.0 if own else top[Y, X, IV])
            deep = past_tips(other, p, K)
            for li in range(2):
                layer = top if li == 0 else und
                a = a_top if li == 0 else a_und
                for c in range(PACKED):
                    lifted[c] = a * layer[Y, X, c]
                    layer[Y, X, c] -= lifted[c]
                for t in range(ntap[q]):
                    b, wt = cols[q, t], wts[q, t]
                    for c in range(PACKED):
                        tip[b, c] += wt * (1.0 - deep) * lifted[c]
                        belly[b, c] += wt * deep * lifted[c]


@njit(cache=True, error_model="numpy")
def brush_step(st, K, tex, tex2, tables, cv, aux, pend, br, col_u):
    """One step of a brush along its stroke: where it touches the canvas, and
    what it does there. Returns (how coated the bristles are with wet paint,
    whether the brush has run dry)."""
    F, hit = footprint(st, K, tex, tex2, tables[0], tables[1], tables[2])
    if not hit:
        return st.wet, st.dry
    # most of the work is only where the tool passes over in this step
    part = touched(st, F)
    taps = bristle_taps(
        F[F_U, part.wy0 : part.wy0 + part.ch, part.wx0 : part.wx0 + part.cw],
        st.span,
        st.nsub,
        K.ring,
        st.rho,
    )
    return brush_step_on(st, part, K, F, taps, cv, aux, pend, br, tables, col_u, tex2)


@njit(cache=True, error_model="numpy")
def brush_step_on(st, part, K, F, taps, cv, aux, pend, br, tables, col_u, hair):
    """One step of a brush, over its footprint ``F`` (see brush_step), on the
    ``part`` of its window it passes over (see :func:`touched`; ``taps`` are
    for that part)."""
    top, und, ids = cv[0], cv[1], cv[4]
    floor, tooth_map, hollows = aux[0], aux[1], aux[2]
    tip, belly = br[0], br[1]
    grid, c_tab, h_tab = tables
    p = st.p
    f_pick, f_dep, f_mix, f_drag = surface_factors_on(cv, aux, st, part, K)

    # 0) drag: wet paint under the brush is pushed along with it
    if K.drag > 0:
        f = np.empty((part.ch, part.cw))
        coef = K.drag * f_drag * (0.5 + p)
        # A loaded brush glides on the paint it has just laid; one running dry
        # has nothing to glide on and smears it on ahead (a dry-brush tail).
        on_brush = 0.0
        for b in range(COLUMNS):
            on_brush += tip[b, IV] + belly[b, IV]
        own_drag = smoothstep(0.8, 0.15, on_brush / (st.Wn * st.Wn))
        for y in range(part.ch):
            for x in range(part.cw):
                wy, wx = part.wy0 + y, part.wx0 + x
                # (the share of a pixel's paint that is moved: a strong drag
                # pressed hard on a slick surface would move more than is
                # there, leaving negative paint that grows step by step)
                f[y, x] = min(
                    coef
                    * F[F_PROFILE, wy, wx]
                    * clip(F[F_OVERLAP, wy, wx] / st.ds, 0.0, 1.0),
                    1.0,
                )
                if ids[part.cy0 + y, part.cx0 + x] == K.stroke_id:
                    f[y, x] *= own_drag
        drag(
            top,
            und,
            f,
            part.cx0,
            part.cy0,
            st.tx * st.ds,
            st.ty * st.ds,
            p,
            st.x0,
            st.y0,
            st.hh,
            st.ww,
        )

    # 1) refill the tips, work the paints on the brush, pick up wet paint
    capacity = max(K.smear, 0.05) * 0.8 * st.Wd * st.Wd
    work_load(br, K, st, capacity)
    pick_up(cv, br, taps, F, part, K, f_pick)

    # 2) how much paint the brush carries, per column. Paint beyond what the hair
    #    holds is a glob on the outside: it goes on thick, the more so the stiffer the
    #    paint.
    col_vol, load, col_load, col_body, body = load_stats(tip, belly, st.Wn)
    # What the hair holds is a matter of the size of the bundle, not of how wide
    # the mark is: touching lightly with a pointed brush, or lowering it at a
    # tapered end, doesn't put the paint in it beyond what it holds.
    wg = max(st.Wn, 0.6 * K.bundle)
    k_g = (st.Wn / wg) ** 2
    # small brushes have long hair for their width: they hold far more for their size
    hold = HOLD * K.capacity * (1.3 - 0.6 * p) * max(1.0, 20.0 / wg) ** 1.5
    glob_gain = IMPASTO * (1.2 - 0.5 * p)
    thick = deposit_thickness(load) * (0.5 + p) + glob_gain * body * max(
        k_g * load - hold, 0.0
    )
    col_glob = np.empty(COLUMNS)
    col_thick = np.empty(COLUMNS)
    col_fill = np.empty(COLUMNS)
    col_over = np.empty(COLUMNS)
    for b in range(COLUMNS):
        col_over[b] = k_g * col_load[b]
        col_glob[b] = glob_gain * col_body[b] * max(col_over[b] - hold, 0.0)
        col_thick[b] = deposit_thickness(col_load[b]) * (0.5 + p) + col_glob[b]
        col_fill[b] = clip((col_load[b] - RESIDUAL) / 0.6, 0.0, 1.0)

    # 3) the bristles comb through thick wet paint already there (not what they just
    #    laid), deeper along the hair lines and under pressure: what stands above them
    #    is shifted, partly onto the brush, partly out to the sides (ridges), the rest
    #    pushed ahead (left as a ridge where the brush lifts)
    # ring, left, right
    ridges = (np.zeros(PACKED), np.zeros(PACKED), np.zeros(PACKED))
    # (over all of the window on the canvas)
    ch, cw, cy0, cx0, wy0, wx0 = st.ch, st.cw, st.cy0, st.cx0, st.wy0, st.wx0
    hb = max(1.25 * thick, 3.5 * (1.15 - p) * (1.3 - 0.8 * K.stiffness))
    digging = K.dig > 0
    vmax = 0.0
    for y in range(ch):
        for x in range(cw):
            vmax = max(vmax, top[cy0 + y, cx0 + x, IV] + und[cy0 + y, cx0 + x, IV])
    if digging or vmax > (1.0 - 0.4 * K.streaks) * hb:
        tv = np.empty((ch, cw))
        wet_h = np.empty((ch, cw))
        excess = np.empty((ch, cw))
        emax = 0.0
        depth = K.dig * smoothstep(0.05, 0.6, p)
        for y in range(ch):
            for x in range(cw):
                Y, X = cy0 + y, cx0 + x
                # (heights summed in single precision, like the paint itself)
                t = top[Y, X, IV] if ids[Y, X] != K.stroke_id else np.float32(0.0)
                tv[y, x] = t
                wet_h[y, x] = t + und[Y, X, IV]
                e = F[F_EXPO, wy0 + y, wx0 + x]
                # a point cut into the paint: a groove through a share of all of it
                if digging:
                    ex = (t + und[Y, X, IV]) * depth * (1.0 - math.exp(-6.0 * e))
                else:
                    level = floor[Y, X] + hb * (
                        1.0 + 0.8 * K.streaks * (F[F_HAIR, wy0 + y, wx0 + x] - 0.5)
                    )
                    ex = max((floor[Y, X] + t + und[Y, X, IV]) - level, 0.0)
                    # shifted gradually as it passes
                    ex *= 1.0 - math.exp(-0.8 * K.stiffness**2 * e)
                    # a light tap only touches with the bristle tips: it doesn't plough
                    if K.is_dab:
                        ex *= smoothstep(0.4, 0.8, p)
                excess[y, x] = ex
                emax = max(emax, ex)
        if emax > 1e-3:
            moved = take(top, und, cy0, cx0, excess, tv)
            share_pick = 0.5 * min(K.pickup, 1.0)
            cols, wts, ntap = bristle_taps(
                F[F_U, wy0 : wy0 + ch, wx0 : wx0 + cw], st.span, st.nsub, K.ring, st.rho
            )
            for y in range(ch):
                for x in range(cw):
                    q = y * cw + x
                    deep = past_tips(wet_h[y, x], p, K)
                    for t in range(ntap[q]):
                        b, wt = cols[q, t], wts[q, t] * share_pick
                        for c in range(PACKED):
                            m = wt * moved[y, x, c]
                            tip[b, c] += (1.0 - deep) * m
                            belly[b, c] += deep * m
            moved *= 1.0 - share_pick
            if st.touchdown:
                for y in range(ch):
                    for x in range(cw):
                        for c in range(PACKED):
                            ridges[0][c] += moved[y, x, c]
            else:
                # a point pushes it all aside
                to_side = 1.0 if digging else 0.35 / (1.0 - share_pick)
                add_sides(
                    ridges, moved, F[F_LAT, wy0 : wy0 + ch, wx0 : wx0 + cw], to_side
                )
                if to_side < 1.0:
                    moved *= 1.0 - to_side
                    place(
                        cv,
                        pend,
                        moved,
                        cx0,
                        cy0,
                        st.tx * st.ds,
                        st.ty * st.ds,
                        K.stroke_id,
                    )
    squeeze_out(br, K, st, col_vol, col_over, hold, ridges, col_u)
    lay_ridges(cv, pend, br, K, st, ridges, grid, c_tab, h_tab, hair)

    # 4) deposit from the tips. Each column's load sets how thick its paint goes on
    #    and when it starts to run dry. Bristles that were just in wet paint are
    #    coated and keep laying a continuous smear for a short distance.
    #    (Only where it passes over: elsewhere its contact is 0.)
    ch, cw = part.ch, part.cw
    cy0, cx0, wy0, wx0 = part.cy0, part.cx0, part.wy0, part.wx0
    contact = np.empty((ch, cw))
    wet_contact = np.empty((ch, cw))
    sum_contact, sum_wet = 0.0, 0.0
    for y in range(ch):
        for x in range(cw):
            Y, X = cy0 + y, cx0 + x
            ct = F[F_EXPO, wy0 + y, wx0 + x] * F[F_STREAK, wy0 + y, wx0 + x]
            wc = smoothstep(0.05, 0.35, top[Y, X, IV] + und[Y, X, IV])
            contact[y, x] = ct
            wet_contact[y, x] = wc
            sum_contact += ct
            sum_wet += ct * wc
    norm = max(sum_contact, 1e-12)
    wet = max(sum_wet / norm, st.wet * math.exp(-st.ds / st.Wd))
    fill = at_pixels(taps, col_fill, ch, cw)
    fill_all = max(clip((load - RESIDUAL) / 0.6, 0.0, 1.0), 0.9 * wet)
    # dry brush: only the canvas tooth (and hair lines) catch paint; a toothy surface
    # breaks up a lightly pressed stroke even when the brush is loaded (scumbling)
    rough_t = K.surface_tooth - 0.5
    rough_thr = (3.2 if rough_t > 0 else 0.6) * rough_t * (1.0 - p) ** 1.2
    q = np.empty((ch, cw))
    tooth = np.empty((ch, cw))
    sum_q, sum_all = 0.0, 0.0
    for y in range(ch):
        for x in range(cw):
            fl = clip(max(fill[y, x], 0.9 * wet), 0.0, 1.0)
            thr = 1.5 * (1.0 - fl) ** 0.8 - 0.35 + rough_thr
            to = (
                K.tooth_w * tooth_map[cy0 + y, cx0 + x]
                + (1.0 - K.tooth_w) * F[F_HAIR, wy0 + y, wx0 + x]
            )
            tooth[y, x] = to
            catch = max(smoothstep(thr - 0.08, thr + 0.08, to), wet_contact[y, x])
            q[y, x] = contact[y, x] * catch
            sum_q += q[y, x]
            # (once the brush as a whole is effectively dry, the rest stays in the
            # bristles)
            thr_all = 1.5 * (1.0 - fill_all) ** 0.8 - 0.35 + rough_thr
            sum_all += contact[y, x] * max(
                smoothstep(thr_all - 0.08, thr_all + 0.08, to), wet_contact[y, x]
            )
    # when only the tooth catches, paint concentrates on those peaks
    q /= max(sum_q / norm, 0.2)
    dry = st.dry
    # a brush that has run dry only comes back to life by dragging through wet paint
    # from earlier strokes
    if dry:
        s = 0.0
        for y in range(ch):
            for x in range(cw):
                Y, X = cy0 + y, cx0 + x
                other = und[Y, X, IV] + (
                    top[Y, X, IV] if ids[Y, X] != K.stroke_id else 0.0
                )
                s += contact[y, x] * smoothstep(0.05, 0.35, other)
        dry = s / norm < 0.12
    else:
        dry = sum_all / norm < 0.04
    tip_v = 0.0
    for b in range(COLUMNS):
        tip_v += tip[b, IV]
    if not dry and tip_v > 0 and K.deposits:
        thick_px = at_pixels(taps, col_thick, ch, cw)
        glob_px = at_pixels(taps, col_glob, ch, cw)
        d = np.empty((ch, cw))
        lift_gain = OWN_LIFT * K.pickup * 1.2 * (0.05 + 1.9 * min(p, 1.0))
        for y in range(ch):
            for x in range(cw):
                tp = thick_px[y, x]
                # absorbent / toothy grounds pull paint off the brush
                dd = tp * q[y, x] * f_dep[y, x]
                # what the bristles behind lift again and carry on (see own_lift)
                dd *= own_lift(
                    lift_gain
                    * f_pick[y, x]
                    * F[F_PROFILE, wy0 + y, wx0 + x]
                    * max(2.0 - F[F_STREAK, wy0 + y, wx0 + x], 0.0),
                    # (a dab is one press: nothing passes over it again)
                    (
                        1.0
                        if K.is_dab
                        else 0.4 + 2.0 * F[F_H, wy0 + y, wx0 + x] / K.spacing
                    ),
                )
                # thick paint from a glob is combed by the bristles into deep grooves
                # and ridges
                g = glob_px[y, x] / max(tp, 1e-9)
                dd *= max(
                    1.0
                    + g * 2.0 * K.streaks * (2.0 * F[F_HAIR, wy0 + y, wx0 + x] - 1.0),
                    0.15,
                )
                # and it rounds off over the last px or two at the stroke's edges
                # (rather than standing in a cliff one pixel wide, which on a slant
                # steps from pixel to pixel)
                if g > 0 and not K.is_dab:
                    dd *= (
                        1.0
                        - g
                        + g
                        * smoothstep(-0.5, GLOB_EDGE_PX, F[F_EDGE, wy0 + y, wx0 + x])
                    )
                # paint laid on thick fills the hollows of the weave (and of dried
                # paint) under it
                dd *= max(
                    1.0
                    + smoothstep(0.3, 2.0, tp)
                    * hollows[cy0 + y, cx0 + x]
                    / max(tp, 1e-6),
                    0.0,
                )
                d[y, x] = dd
        if K.is_dab:
            # pressed down, an overloaded brush leaves part of its glob as a mound
            press_glob(d, F, st, part, col_vol, col_over, hold, hair)
            share_by_reach(tip, taps, d)
        feed_glob(tip, belly, taps, d, col_glob)
        lay(
            cv,
            pend,
            br,
            taps,
            cy0,
            cx0,
            d,
            F[F_COVER, wy0 : wy0 + ch, wx0 : wx0 + cw],
            K.stroke_id,
            False,
            True,
            F[F_EXPO, wy0 : wy0 + ch, wx0 : wx0 + cw],
            True,
        )
        # the sides of the brush brush past the canvas edge
        full = 0.0
        for y in range(st.hh):
            for x in range(st.ww):
                full += F[F_EXPO, y, x] * F[F_STREAK, y, x]
        off_canvas = (
            thick
            * own_lift(lift_gain, 0.4 + 2.0 * st.r / K.spacing)
            * (full - sum_contact)
        )
        left = 0.0
        for b in range(COLUMNS):
            left += tip[b, IV]
        if off_canvas > 0 and left > 0:
            tip *= max(1.0 - off_canvas / left, 0.0)
    overflow(tip, belly, capacity)

    # 5) stir: pressure churns the fresh top layer and the wet paint below into each
    # other
    m = np.empty((ch, cw))
    for y in range(ch):
        for x in range(cw):
            m[y, x] = 1.0 - math.exp(
                -K.mixing * 3.0 * p * p * contact[y, x] * f_mix[y, x]
            )
    stir(cv, cy0, cx0, m)
    return wet, dry


# ----------------------------------------------------------------------------
# after the stroke


@njit(cache=True, error_model="numpy")
def level(top, und, dry_h, ry0, ry1, rx0, rx1, iters):
    """Let wet paint settle over the region. Oil paint holds its shape up to a
    point: it only flows where it is steeper than its stiffness (body) can
    hold. Fluid paint levels out almost flat (its brush marks and ridges
    disappear); stiff paint keeps them and only slumps a little at sheer
    edges. Thin, fluid paint stays within the area it covers; thick paint can
    slump beyond it."""
    H, W = ry1 - ry0, rx1 - rx0
    # (heights and slopes in single precision, like the paint: whether paint is
    # steeper than it holds is decided on the same numbers as the paint's own)
    V = np.empty((H, W), np.float32)
    crit = np.empty((H, W), np.float32)
    vmax = 0.0
    for y in range(H):
        for x in range(W):
            v = top[ry0 + y, rx0 + x, IV] + und[ry0 + y, rx0 + x, IV]
            V[y, x] = v
            vmax = max(vmax, v)
            body = np.float32(1.0)
            if v > 1e-6:
                body = (top[ry0 + y, rx0 + x, IB] + und[ry0 + y, rx0 + x, IB]) / max(
                    v, np.float32(1e-6)
                )
            # the steepest slope it holds
            crit[y, x] = np.float32(SLOPE_FLUID) + np.float32(
                SLOPE_STIFF - SLOPE_FLUID
            ) * (body * body * body * body)
    if vmax < 0.3:
        return
    # where is it steeper than it can hold?
    by0, by1, bx0, bx1 = H, -1, W, -1
    for y in range(H):
        for x in range(W):
            if V[y, x] <= 0.05:
                continue
            s = dry_h[ry0 + y, rx0 + x] + V[y, x]
            steep = False
            for dy, dx in ((0, 1), (0, -1), (1, 0), (-1, 0)):
                yy, xx = y + dy, x + dx
                if (
                    0 <= yy < H
                    and 0 <= xx < W
                    and abs(dry_h[ry0 + yy, rx0 + xx] + V[yy, xx] - s) > crit[y, x]
                ):
                    steep = True
            if steep:
                by0, by1 = min(by0, y), max(by1, y)
                bx0, bx1 = min(bx0, x), max(bx1, x)
    if by1 < 0:
        return
    pad = iters + 2
    y0, y1 = max(by0 - pad, 0), min(by1 + pad + 1, H)
    x0, x1 = max(bx0 - pad, 0), min(bx1 + pad + 1, W)
    h, w = y1 - y0, x1 - x0
    # the paint's volume and the dry surface under it, over the box (the
    # volume kept up to date as paint moves)
    Vs = np.empty((h, w), np.float32)
    Dh = np.empty((h, w), np.float32)
    for y in range(h):
        for x in range(w):
            Vs[y, x] = V[y0 + y, x0 + x]
            Dh[y, x] = dry_h[ry0 + y0 + y, rx0 + x0 + x]
    fa = np.zeros((h, w), np.float32)
    fb = np.zeros((h, w), np.float32)
    ta = np.empty(PACKED, np.float32)
    ua = np.empty(PACKED, np.float32)
    # (the same, for every column)
    tcol = np.empty((w, PACKED), np.float32)
    ucol = np.empty((w, PACKED), np.float32)
    saved_col = np.empty(w, np.bool_)
    for _ in range(iters):
        flowed = False
        for axis in range(2):
            # downhill flow beyond what the paint can hold; onto bare canvas only from a
            # sheer edge. Pair (a, b): b is the next pixel down (axis 0) or to the right
            # (axis 1).
            any_flow = False
            for y in range(h - (1 if axis == 0 else 0)):
                for x in range(w - (1 if axis == 1 else 0)):
                    yb, xb = (y + 1, x) if axis == 0 else (y, x + 1)
                    Sa = Dh[y, x] + Vs[y, x]
                    Sb = Dh[yb, xb] + Vs[yb, xb]
                    dS = Sa - Sb
                    zero, quarter = np.float32(0.0), np.float32(0.25)
                    qa = (
                        max(dS - crit[y0 + y, x0 + x], zero)
                        if (Vs[yb, xb] > 0.1 or dS > 2.0)
                        else zero
                    )
                    qb = (
                        max(-dS - crit[y0 + yb, x0 + xb], zero)
                        if (Vs[y, x] > 0.1 or -dS > 2.0)
                        else zero
                    )
                    fa[y, x] = (
                        min(quarter * qa / max(Vs[y, x], np.float32(1e-6)), quarter)
                        if qa > 0
                        else zero
                    )
                    fb[y, x] = (
                        min(quarter * qb / max(Vs[yb, xb], np.float32(1e-6)), quarter)
                        if qb > 0
                        else zero
                    )
                    if qa > 0 or qb > 0:
                        any_flow = True
            if not any_flow:
                continue
            flowed = True
            # move it, all pairs at once (from the paint as it was before this pass):
            # walk each line; a pixel changed by the pair behind it keeps a copy of how
            # it was
            if axis == 0:
                # (the columns side by side, a row at a time: each column is walked
                # down as on its own, and the rows are read in the order they lie)
                for x in range(w):
                    saved_col[x] = False
                for k in range(1, h):
                    yp, y = k - 1, k
                    Ya, Yb = ry0 + y0 + yp, ry0 + y0 + y
                    for x in range(w):
                        f_a, f_b = fa[yp, x], fb[yp, x]
                        if not (f_a > 0 or f_b > 0):
                            saved_col[x] = False
                            continue
                        X = rx0 + x0 + x
                        sv = saved_col[x]
                        for c in range(PACKED):
                            t_a = tcol[x, c] if sv else top[Ya, X, c]
                            u_a = ucol[x, c] if sv else und[Ya, X, c]
                            t_b, u_b = top[Yb, X, c], und[Yb, X, c]
                            tcol[x, c], ucol[x, c] = t_b, u_b
                            mt = t_a * f_a - t_b * f_b
                            mu = u_a * f_a - u_b * f_b
                            top[Ya, X, c] -= mt
                            und[Ya, X, c] -= mu
                            top[Yb, X, c] += mt
                            und[Yb, X, c] += mu
                        saved_col[x] = True
                        # (both pixels changed)
                        Vs[yp, x] = top[Ya, X, IV] + und[Ya, X, IV]
                        Vs[y, x] = top[Yb, X, IV] + und[Yb, X, IV]
                continue
            for line in range(h):
                saved = False  # (ta, ua) hold the previous pixel as it was
                for k in range(1, w):
                    yp, xp = line, k - 1
                    f_a, f_b = fa[yp, xp], fb[yp, xp]
                    if not (f_a > 0 or f_b > 0):
                        saved = False
                        continue
                    y, x = line, k
                    Ya, Xa = ry0 + y0 + yp, rx0 + x0 + xp
                    Yb, Xb = ry0 + y0 + y, rx0 + x0 + x
                    for c in range(PACKED):
                        t_a = ta[c] if saved else top[Ya, Xa, c]
                        u_a = ua[c] if saved else und[Ya, Xa, c]
                        t_b, u_b = top[Yb, Xb, c], und[Yb, Xb, c]
                        # (b as it was: a, for the next pair)
                        ta[c], ua[c] = t_b, u_b
                        mt = t_a * f_a - t_b * f_b
                        mu = u_a * f_a - u_b * f_b
                        top[Ya, Xa, c] -= mt
                        und[Ya, Xa, c] -= mu
                        top[Yb, Xb, c] += mt
                        und[Yb, Xb, c] += mu
                    saved = True
                    # (both pixels changed)
                    Vs[yp, xp] = top[Ya, Xa, IV] + und[Ya, Xa, IV]
                    Vs[y, x] = top[Yb, Xb, IV] + und[Yb, Xb, IV]
        if not flowed:  # it has settled: nothing more will move
            break


# How readily the surface of wet paint rounds off (surface tension against
# its yield), per pass, for tube paint and for stiff paint; and how thick the
# paint must be for it to matter.
SOFTEN_TUBE, SOFTEN_STIFF, SOFTEN_MIN = 0.26, 0.02, 1.2
# Two paints this alike (see alike) flow into each other as they round off.
SOFTEN_SAME = 0.1


@njit(cache=True, error_model="numpy")
def soften(top, und, dry_h, ids, stroke_id, ry0, ry1, rx0, rx1, passes):
    """The fresh paint of a stroke rounds off a little where it is thick: the
    sharpest crests and furrows soften (a diffusion of its surface height,
    volume-conserving), more in tube paint than in paint stiffened with
    impasto medium, not at all in thin paint. Only the stroke's own top layer
    moves, and only between pixels that both hold a thick film of it, of
    much the same paint (the bands of a double load stay as they were laid)."""
    H, W = ry1 - ry0, rx1 - rx0
    if H < 3 or W < 3:
        return
    # (only over the part that holds a thick film of the stroke's paint)
    y_lo, y_hi, x_lo, x_hi = H, -1, W, -1
    for y in range(H):
        for x in range(W):
            Y, X = ry0 + y, rx0 + x
            if ids[Y, X] == stroke_id and top[Y, X, IV] >= SOFTEN_MIN:
                y_lo, y_hi = min(y_lo, y), max(y_hi, y)
                x_lo, x_hi = min(x_lo, x), max(x_hi, x)
    if y_hi < 0:
        return
    oy, ox = ry0 + y_lo, rx0 + x_lo
    H, W = y_hi - y_lo + 1, x_hi - x_lo + 1
    k = np.zeros((H, W), np.float32)
    # what each pixel's paint is (as alike compares paints), once
    sig = np.zeros((H, W, IV), np.float32)
    for y in range(H):
        for x in range(W):
            Y, X = oy + y, ox + x
            v = top[Y, X, IV]
            if ids[Y, X] != stroke_id or v < SOFTEN_MIN:
                continue
            body = top[Y, X, IB] / v
            k[y, x] = SOFTEN_STIFF + (SOFTEN_TUBE - SOFTEN_STIFF) * smoothstep(
                0.82, 0.66, body
            )
            for c in range(IV):
                sig[y, x, c] = math.log1p(top[Y, X, c] / v)
    # which neighbours (right, below) hold much the same paint, once
    same_r = np.zeros((H, W), np.bool_)
    same_d = np.zeros((H, W), np.bool_)
    for y in range(H):
        for x in range(W):
            if k[y, x] == 0:
                continue
            for dy, dx in ((0, 1), (1, 0)):
                yy, xx = y + dy, x + dx
                if not (yy < H and xx < W and k[yy, xx] != 0):
                    continue
                diff = np.float32(0.0)
                for c in range(IV):
                    diff += abs(sig[y, x, c] - sig[yy, xx, c])
                if diff / IV < SOFTEN_SAME:
                    if dy == 0:
                        same_r[y, x] = True
                    else:
                        same_d[y, x] = True
    S = np.empty((H, W), np.float32)
    snap = np.empty((H, W, PACKED), np.float32)
    for _ in range(passes):
        for y in range(H):
            for x in range(W):
                Y, X = oy + y, ox + x
                S[y, x] = dry_h[Y, X] + top[Y, X, IV] + und[Y, X, IV]
                if k[y, x] != 0:
                    for c in range(PACKED):
                        snap[y, x, c] = top[Y, X, c]
        # each pixel gives each lower neighbour a share of its paint (at most a
        # fifth to each, so nothing runs dry)
        for y in range(H):
            for x in range(W):
                if k[y, x] == 0:
                    continue
                Y, X = oy + y, ox + x
                for dy, dx in ((0, 1), (0, -1), (1, 0), (-1, 0)):
                    yy, xx = y + dy, x + dx
                    if not (0 <= yy < H and 0 <= xx < W and k[yy, xx] != 0):
                        continue
                    d = S[y, x] - S[yy, xx]
                    if d <= 0:
                        continue
                    if dx == 1:
                        same = same_r[y, x]
                    elif dx == -1:
                        same = same_r[y, xx]
                    elif dy == 1:
                        same = same_d[y, x]
                    else:
                        same = same_d[yy, x]
                    if not same:
                        continue
                    q = min(
                        min(k[y, x], k[yy, xx]) * d * np.float32(0.5),
                        0.2 * snap[y, x, IV],
                    )
                    f = q / max(snap[y, x, IV], np.float32(1e-6))
                    for c in range(PACKED):
                        m = snap[y, x, c] * f
                        top[Y, X, c] -= m
                        top[oy + yy, ox + xx, c] += m


@njit(cache=True, error_model="numpy")
def clear_negative(top, und, ry0, ry1, rx0, rx1):
    """Single-precision rounding, moving very thick paint about, can leave a
    layer of a pixel holding a hair less than nothing of some pigment, or of
    paint (-1e-3 in a layer hundreds of units thick): none there instead."""
    zero = np.float32(0.0)
    for layer in (top, und):
        for y in range(ry0, ry1):
            row = layer[y]
            for x in range(rx0, rx1):
                if row[x, IV] <= zero:
                    if row[x, IV] < zero:
                        for c in range(PACKED):
                            row[x, c] = zero
                    continue
                for c in range(PACKED):
                    if row[x, c] < zero:
                        row[x, c] = zero


@njit(cache=True, error_model="numpy")
def fix_coverage(top, und, top_a, und_a, ry0, ry1, rx0, rx1):
    """Each pixel records how much of it its wet paint covers (anti-aliased
    edges): a film of thickness volume / coverage over that part. Paint pushed
    into a pixel (dragged, slumped, squeezed out) doesn't update that record,
    so make it consistent over the region. A pixel with paint all around it
    isn't at an edge: it is fully covered. Elsewhere, no pixel holds a film
    much thicker than the mostly-covered pixels around it; where it would,
    the paint covers more of it."""
    H, W = ry1 - ry0, rx1 - rx0
    if H <= 0 or W <= 0:
        return
    # every neighbour (3 x 3, within the region) carries a film of paint
    inside = np.empty((H, W), np.bool_)
    for y in range(H):
        for x in range(W):
            lo = np.inf
            for yy in range(max(y - 1, 0), min(y + 2, H)):
                for xx in range(max(x - 1, 0), min(x + 2, W)):
                    lo = min(
                        lo, top[ry0 + yy, rx0 + xx, IV] + und[ry0 + yy, rx0 + xx, IV]
                    )
            inside[y, x] = lo > MIN_FILM
    reliable = np.empty((H, W), np.float32)
    for li in range(2):
        layer = top if li == 0 else und
        A = top_a if li == 0 else und_a
        for y in range(H):  # film thickness where it is well defined
            for x in range(W):
                a = A[ry0 + y, rx0 + x]
                reliable[y, x] = (
                    layer[ry0 + y, rx0 + x, IV] / max(a, np.float32(1e-6))
                    if a >= 0.5
                    else 0.0
                )
        for y in range(H):
            for x in range(W):
                a = A[ry0 + y, rx0 + x]
                if not a > 0:
                    continue
                v = layer[ry0 + y, rx0 + x, IV]
                if inside[y, x] and v > 0.5 * MIN_FILM:
                    need = 1.0
                else:
                    # the thickest well-defined film within 2 px (within the region)
                    nb = 0.0
                    for yy in range(max(y - 2, 0), min(y + 3, H)):
                        for xx in range(max(x - 2, 0), min(x + 3, W)):
                            nb = max(nb, reliable[yy, xx])
                    need = min(v / (1.25 * max(nb, 1e-6)), 1.0) if nb > 0 else 0.0
                if need > a:
                    A[ry0 + y, rx0 + x] = need


@njit(cache=True, error_model="numpy")
def max_filter(a, r):
    """Maximum over the (2r+1)-square around each pixel, edges clamped, in
    single precision: what ``_filters._max_filter`` gives."""
    h, w = a.shape
    tmp = np.empty((h, w), np.float32)
    for y in range(h):
        for x in range(w):
            m = np.float32(a[max(y - r, 0), x])
            for yy in range(max(y - r, 0) + 1, min(y + r + 1, h)):
                v = np.float32(a[yy, x])
                if v > m:
                    m = v
            tmp[y, x] = m
    out = np.empty((h, w), np.float32)
    for y in range(h):
        for x in range(w):
            m = tmp[y, max(x - r, 0)]
            for xx in range(max(x - r, 0) + 1, min(x + r + 1, w)):
                if tmp[y, xx] > m:
                    m = tmp[y, xx]
            out[y, x] = m
    return out


@njit(cache=True, error_model="numpy")
def box_blur(a, r):
    """Separable box blur with edge clamping (radius r)."""
    h, w = a.shape
    n = 2 * r + 1
    tmp = np.empty((h, w))
    out = np.empty((h, w))
    for y in range(h):
        for x in range(w):
            acc = 0.0
            for k in range(-r, r + 1):
                acc += a[min(max(y + k, 0), h - 1), x]
            tmp[y, x] = acc / n
    for y in range(h):
        for x in range(w):
            acc = 0.0
            for k in range(-r, r + 1):
                acc += tmp[y, min(max(x + k, 0), w - 1)]
            out[y, x] = acc / n
    return out


@njit(cache=True, error_model="numpy")
def box_blur_rows(a, r, lo, hi, out):
    """:func:`box_blur` of ``a`` only in each row y's columns lo[y]..hi[y] (the
    same numbers there), into ``out``. It reads ``a`` within r px of them."""
    h, w = a.shape
    n = 2 * r + 1
    tmp = np.empty(w)
    for y in range(h):
        if lo[y] >= hi[y]:
            continue
        for x in range(max(lo[y] - r, 0), min(hi[y] + r, w)):
            acc = 0.0
            for k in range(-r, r + 1):
                acc += a[min(max(y + k, 0), h - 1), x]
            tmp[x] = acc / n
        for x in range(lo[y], hi[y]):
            acc = 0.0
            for k in range(-r, r + 1):
                acc += tmp[min(max(x + k, 0), w - 1)]
            out[y, x] = acc / n


@njit(cache=True, error_model="numpy")
def grown(lo, hi, r, w):
    """Row spans (lo[y]..hi[y] in each row y) grown by r px all round, within
    0..w."""
    h = lo.shape[0]
    lo2 = np.full(h, w, np.int64)
    hi2 = np.zeros(h, np.int64)
    for y in range(h):
        if lo[y] >= hi[y]:
            continue
        for yy in range(max(y - r, 0), min(y + r + 1, h)):
            lo2[yy] = min(lo2[yy], max(lo[y] - r, 0))
            hi2[yy] = max(hi2[yy], min(hi[y] + r, w))
    return lo2, hi2


@njit(cache=True, error_model="numpy")
def layer_speed(z0, z1, gap, slip):
    """Speed, as a fraction of the blade's, of a wet layer between heights z0
    and z1 above the canvas, with the blade at height ``gap``: below the blade
    the paint shears (Couette flow, speed rising with height, times ``slip``);
    above it, it is pushed along with the blade."""
    thick = max(z1 - z0, 1e-9)
    lo, hi = min(z0, gap), min(z1, gap)
    sheared = max(hi - lo, 0.0)
    plug = max(z1 - max(z0, gap), 0.0)
    return (sheared * slip * 0.5 * (hi + lo) / gap + plug) / thick


@njit(cache=True, error_model="numpy")
def flow_under_blade(
    cv,
    sy0,
    sx0,
    stroke_id,
    floor,
    level,
    lateral,
    along,
    c,
    half,
    ds,
    tx,
    ty,
    slip,
    viscous,
):
    """Move the wet paint a blade touches as it travels ``ds`` px along (tx, ty)
    in the window at (sy0, sx0) (thin-film shear and bulldozing, see
    ``layer_speed``).

    The blade's footprint at each sub-step is where ``lateral`` > 0 and the
    pixel's position ``along`` the motion lies within ``half`` of the chord
    centre ``c`` (relative to the blade's position at the end of the step).
    Conservative upwind fluxes, in sub-steps short enough to be stable."""
    top, und, ids = cv[0], cv[1], cv[4]
    hh, ww = level.shape
    own = np.empty((hh, ww), np.bool_)
    g = np.empty((hh, ww))
    for y in range(hh):
        for x in range(ww):
            own[y, x] = ids[sy0 + y, sx0 + x] == stroke_id
            g[y, x] = max(level[y, x] - floor[y, x], 0.2)
    # under paint, the blade doesn't feel the individual threads of the canvas
    gap = box_blur(g, 2)
    # the paint under the blade must reach it somewhere for anything to happen
    reaches = False
    for y in range(hh):
        for x in range(ww):
            Y, X = sy0 + y, sx0 + x
            reach = und[Y, X, IV] + (0.0 if own[y, x] else top[Y, X, IV])
            if reach > gap[y, x] - 1.0 and lateral[y, x] > 0:
                reaches = True
    if not reaches:
        return
    # two wet layers of the same paint can merge without any visible change:
    # do so, so that paint dragged in from elsewhere lands on top of them
    merge_same(cv, sy0, sx0, own, 0.25)
    ax, ay = abs(tx), abs(ty)
    n_sub = max(int(math.ceil(ds * (ax + ay))), 1)
    step = ds / n_sub
    xdir, ydir = (1 if tx >= 0 else -1), (1 if ty >= 0 else -1)
    inside = np.empty((hh, ww))
    u_src = np.empty((hh, ww))
    t_src = np.empty((hh, ww))
    u = np.empty((hh, ww))
    t = np.empty((hh, ww))
    # Nothing moves where the blade isn't (there the speeds are 0): the work is
    # done only where it is, and where the paint it moves goes, as the columns
    # lo[y]..hi[y] of each row y.
    in_lo = np.empty(hh, np.int64)
    in_hi = np.empty(hh, np.int64)
    c_lo = np.empty(hh, np.int64)
    c_hi = np.empty(hh, np.int64)
    for k in range(1, n_sub + 1):
        # the blade's position along the motion
        pos = -ds + k * step
        any_in = False
        for y in range(hh):
            in_lo[y], in_hi[y] = ww, 0
            for x in range(ww):
                v = lateral[y, x] * clip(
                    half[y, x] - abs(along[y, x] - pos - c[y, x]) + 0.5, 0.0, 1.0
                )
                inside[y, x] = v
                if v != 0:
                    any_in = True
                    in_lo[y], in_hi[y] = min(in_lo[y], x), x + 1
        if not any_in:
            continue
        # the flow is smooth over a pixel or two: speeds from lightly smoothed heights
        lo1, hi1 = grown(in_lo, in_hi, 1, ww)
        for y in range(hh):
            for x in range(lo1[y], hi1[y]):
                u_src[y, x] = und[sy0 + y, sx0 + x, IV]
                t_src[y, x] = 0.0 if own[y, x] else top[sy0 + y, sx0 + x, IV]
        box_blur_rows(u_src, 1, in_lo, in_hi, u)
        box_blur_rows(t_src, 1, in_lo, in_hi, t)
        c_u = np.zeros((hh, ww))
        c_t = np.zeros((hh, ww))
        any_c = False
        for y in range(hh):
            c_lo[y], c_hi[y] = ww, 0
            for x in range(in_lo[y], in_hi[y]):
                # (heights in single precision, like the paint: a film too thin to
                # change the height of the paint below has no thickness, and no speed of
                # its own)
                uu, tt = np.float32(u[y, x]), np.float32(t[y, x])
                # the blade glides on its own fresh paint
                shielded = own[y, x] and top[sy0 + y, sx0 + x, IV] > 0.3
                contact = (
                    0.0
                    if shielded
                    else inside[y, x] * clip(uu + tt - gap[y, x] + 1.0, 0.0, 1.0)
                )
                c_u[y, x] = contact * layer_speed(
                    np.float32(0.0), uu, gap[y, x], slip[y, x]
                )
                c_t[y, x] = contact * layer_speed(uu, uu + tt, gap[y, x], slip[y, x])
                if contact != 0:
                    any_c = True
                    c_lo[y], c_hi[y] = min(c_lo[y], x), x + 1
        if not any_c:
            continue
        # viscosity: neighbouring paint drags each other along, so the speed varies
        # smoothly (without it, the paint tears into foam)
        f_lo, f_hi = grown(c_lo, c_hi, viscous, ww)
        frac_u = np.zeros((hh, ww))
        frac_t = np.zeros((hh, ww))
        box_blur_rows(c_u, viscous, f_lo, f_hi, frac_u)
        box_blur_rows(c_t, viscous, f_lo, f_hi, frac_t)
        # (and the pixels next to those, which they pass paint on to)
        g_lo, g_hi = grown(f_lo, f_hi, 1, ww)
        fx = np.zeros((hh, ww), np.float32)
        fy = np.zeros((hh, ww), np.float32)
        for li in range(2):
            layer = und if li == 0 else top
            frac = frac_u if li == 0 else frac_t
            for y in range(hh):
                for x in range(f_lo[y], f_hi[y]):
                    f = clip(frac[y, x], 0.0, 1.0) * step
                    if li == 1 and own[y, x]:
                        f = 0.0
                    fx[y, x] = f * ax
                    fy[y, x] = f * ay
            # each pixel passes a share of its paint on to its neighbour downstream in x
            # and in y (walking upstream, so the neighbours are still as they were)
            for j in range(hh):
                y = hh - 1 - j if ydir > 0 else j
                lo_y, hi_y = g_lo[y], g_hi[y]
                for i in range(hi_y - lo_y):
                    x = hi_y - 1 - i if xdir > 0 else lo_y + i
                    Y, X = sy0 + y, sx0 + x
                    xs, ys = x - xdir, y - ydir
                    for ch in range(PACKED):
                        v = layer[Y, X, ch]
                        nv = v - (v * fx[y, x] + v * fy[y, x])
                        if 0 <= xs < ww:
                            nv += layer[Y, sx0 + xs, ch] * fx[y, xs]
                        if 0 <= ys < hh:
                            nv += layer[sy0 + ys, X, ch] * fy[ys, x]
                        layer[Y, X, ch] = nv


# --- a stale cache -------------------------------------------------------------
# The kernels are cached on disk (cache=True), with the band count of
# spectral.py (N, PACKED) compiled into them as constants. numba throws a
# cached kernel away only when this file changes, so after a change to
# spectral.WAVELENGTHS alone the cached kernels would read and write past the
# ends of every paint vector. This tiny kernel is cached the same way: if it
# remembers another size, so do the others. (It is the only kernel compiled or
# loaded at import; the others still wait for their first call.)
@njit(cache=True)
def _compiled_packed():
    return PACKED


if _compiled_packed() != PACKED:
    raise RuntimeError(
        f"numba cache is stale: the cached kernels of {__name__} were compiled "
        f"for {_compiled_packed() - (PACKED - N)} spectral bands, spectral.py "
        f"now has {N}. "
        "Delete the cache and import again: the *.nbi and *.nbc files in "
        f"{_compiled_packed.stats.cache_path} (numba keeps them in the "
        "__pycache__ next to this file, or under NUMBA_CACHE_DIR if it is set)."
    )
