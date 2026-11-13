"""The compiled core of the stroke engine (numba).

Everything here works on plain arrays, one pixel at a time, so a step of a
small brush costs microseconds rather than the overhead of hundreds of small
numpy calls. :mod:`oilpaint.brush` holds the brush and stroke model and calls
these kernels; the physics is described there.

Arrays: the canvas layers ``top``/``und`` are (H, W, PACKED) float32, their
coverage ``top_a``/``und_a`` and the stroke ids ``top_id`` (H, W). Paint on a
brush is (COLUMNS, PACKED) float64 per depth (tip, belly, heel). A step works
on a window of the canvas; per-pixel inputs are window-sized.

The kernels take these in named tuples, by the same names throughout: the
canvas's Layers (``cv``), what lies Beneath the wet paint (``aux``), the Load
on a tool (``br``), the Pending paint of a brushstroke (``pend``), a step's
Taps and Ridges, a Step (``st``: its Window, Motion and Hold), and the
stroke's StrokeConsts (``K``).

brush_step is the exception: it takes the stroke's bundles once per stroke
as plain tuples and names them on entry (see brush_kit). numba's dispatcher
resolves a named tuple's type more slowly than a plain one's on every call,
and a brush makes one such call per step. A knife's kernels take the named
tuples as they are: handing them over plain would need a rename at the top
of each of about a dozen entry kernels, and leaving them named costs a few
percent at most on the knife scenes.
"""

import math
from collections import namedtuple

import numpy as np
from numba import njit

from .spectral import _check_bands

# Compiling: the helpers that split a kernel into named steps, shared ones
# included (bury_top, hair_at, radius_at, demand_with, ...), are compiled
# into the kernels that call them: inline="always". The kernels, and the
# building blocks that were there before them (deposit, spread_part,
# sample_field, smoothstep, ...), are compiled on their own. Compiled on its
# own, every njit function costs about 0.3 s of cold compile (the split
# helpers would add some 13 s to it, against 4 s inlined), and a call per
# pixel to one taking the stroke's settings (st, K) can cost run time.

# The band count, written out rather than taken from spectral.N_BANDS: numba
# compiles it into the cached kernels as a constant, so a change of band count
# must be an edit here, which numba sees as a change of this file. The check
# refuses a count that differs from spectral.py's, and a cache that kept its
# stamp all the same (see spectral._check_bands).
N = 24


@njit(cache=True, error_model="numpy")
def _compiled_bands():
    return N


_check_bands(__file__, N, _compiled_bands)

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
# How much paint changes places between the tip and the belly, and between
# the belly and the heel, per brush width of travel at blending 1 (see
# work_load; Brush.work uses the same rates on the palette).
TIP_SWAP, HEEL_SWAP = 0.375, 0.1
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
# Paint a brush pushes aside (squeezed out of its hair, or ploughed out of
# the wet paint under it) gets out to its side ridges freely from within
# EDGE_FREE px of the nearer edge of its footprint; from d px in, only the
# share 1 / (1 + ((d - EDGE_FREE) / EDGE_SQUEEZE) ** 2) of it: under a plate
# the paint has to travel to an edge, and the rate a pressure thins it falls
# as 1 / width^2 (a squeeze film), so under a wide brush the paint in the
# middle stays. What doesn't get out stays under the brush: on it (its
# column stays loaded and lays thicker) or where it was (not ploughed). See
# escapes. A footprint no more than 2 EDGE_FREE px across the stroke is not
# affected (a hog flat 36 px wide pressed at 0.9 spreads to about 38 px),
# and the share is 1 at d = EDGE_FREE: no step where it starts. Measured on
# a white hog flat pressed at 0.9 down 5 units of wet blue, loaded 45 w /
# 36, five strokes, the medians over their rows (ratio of the ridges' width
# at half height to their height, 1 unit ~ 0.6 px; height, units; paint a
# side, units x px; a 36 px flat: 0.93, 10, 69):
#   no limit:      70 px 0.62, 21, 172;  120 px 0.43, 42, 473
#   EDGE_FREE 14:  70 px 1.05, 12, 96;   120 px 1.20, 15, 166 (and 36 px
#                  changes: 0.98, 9.6, 65)
#   EDGE_FREE 20:  70 px 0.84, 15, 123;  120 px 1.02, 18, 208
#   EDGE_FREE 28:  70 px 0.67, 19, 156;  120 px 0.78, 23, 269
# 20 is the least that leaves a 36 px flat as it was. EDGE_SQUEEZE 3: with 6,
# the 120 px flat's ratio was 0.88 (the tail lets more out).
EDGE_FREE, EDGE_SQUEEZE = 20.0, 3.0
# A brush's set-down bead reaches this many times its width past the
# footprint (see _ring_weights: its cross-section falls to nothing there).
BEAD_REACH = 1.3
# Where the stroke's own paint at its edge ends in a slope, its last pixel
# covers at least as much as the slope carried on says (see cover_ramps),
# not as little as the films beside it would: a pixel holding 2 units next
# to 3 and 4.4 covered a third of itself, pale in a solid edge. The paint
# that settles at the foot of a slope after the brush has passed (see level,
# soften) does not change its coverage. On the level edges of heavily loaded
# red 120 px and 70 px flats (seeds 1 to 5, and 1 to 10), the outer pixels
# holding over a unit come out 25 and 22 levels paler than the stroke on
# average with it, 87 and 73 without it (a pale fringe all along the edge,
# even with RIDGE_COVER). By how wide the stroke's footprint reached across
# it (the most of r * lmax over its steps): fully from more than 2 EDGE_FREE
# px, not at all up to 2 RAMP_FROM, smoothly in between (no step where it
# starts).
RAMP_FROM = 14.0
# ... and only on a slope: as far as the film two pixels in is thicker than
# the one next to it (relative to the thicker), from none at RAMP_RISE[0]
# to all at RAMP_RISE[1] (0.3 at the foot of a heavy 120 px flat's edge;
# 0.06 at the edge of a plateau).
RAMP_RISE = (0.1, 0.25)
# Only the films of the stroke's own pixels covered at least this much (and
# not at its edge) say how the slope runs: a partly covered pixel beside
# the foot, along a slanting edge, is that edge's anti-aliasing.
RAMP_SOLID = 0.95
# A foot whose two neighbours along the edge hold paint more unlike than this
# (relative to the more) is where a slanting edge's staircase climbs from
# pixel to pixel: left as it is, its anti-aliasing. Without this check,
# of 240 columns crossing the edge of a heavy black stroke (120 units, five
# canvases) 168 to 210 kept an in-between pixel at 50 to 140 px, against
# 208 to 240 with no slope cover at all; with it, as with none (one column
# in 12 000 differs) (measured before RIDGE_COVER).
RAMP_EVEN = 0.15
# A brush's side ridge (see _ridge) holds paint half a px further out than
# _geo(e, w) covers, so its last pixel took paint that covered nothing:
# right after the steps of a heavy 70 px stroke a foot held 0.82 units over
# 0.17 of itself, a film 4.8 thick beside one of 2.1, and the paint that
# settled into it later made it thicker still: pale feet along a level edge,
# a notch at each step of a slanting one. On a wide brush its cover reaches
# RIDGE_COVER px further, as far as its paint, by how wide the footprint is
# across the stroke (as cover_ramps: smoothly from RAMP_FROM to EDGE_FREE).
# Heavily loaded red 70 px and 120 px flats (seeds 1 to 10, and 1 to 5):
# their level edges' position wavers from column to column (the mean of its
# second difference) by 0.113 and 0.123 px instead of 0.169 and 0.165;
# black 50 px strokes loaded 120 keep an in-between pixel in 236 of 240
# columns across a slanting edge instead of 199. Narrower footprints keep
# their cover: there it took up to 4 of 120 such columns from a heavily
# loaded 10 px stroke and widened what shows of it by up to half a px.
RIDGE_COVER = 0.5
# Paint thinner than this can't form a continuous film (the renderer, see
# canvas.MIN_FILM, counts it by the same measure).
MIN_FILM = 0.15
# A dab's pattern of bristle tips spans this many brush half-widths each way
# from the centre (see dab_cell; _dab.SPAN is the same, written out there).
DAB_SPAN = 1.3
# A layer of a pixel holding no more paint than this holds none: what is
# there is rounding dust (single-precision paint moved about leaves specks of
# 1e-8 either side of zero), far below any film you could see (under 1e-5 of
# MIN_FILM: as a film it would cover 7e-6 of the pixel). It covers nothing,
# passes no coverage on, is not stirred, and is not a layer to merge with.
EMPTY = 1e-6
# A deposit of less than this is a speck landing, not paint laid: it claims
# nothing and buries nothing.
SPECK = 1e-7
# Paint that slumped or was pushed past the edge of the paint the tool laid
# lies there as the thin edge of the films beside it: it covers all of its
# pixel once it is this share as thick as they are, less in proportion (see
# fix_coverage). (Rendered by its volume alone, any of it thicker than
# MIN_FILM came out solid: along a slanted edge, where paint slumps into
# the pixels past each step of the staircase, a tooth at every step.)
EDGE_FILM = 0.45
# Such paint right beside a covered pixel carries on that pixel's edge only
# if that pixel's film is at least this share as thick as the paint pushed
# out: the edge of a comparable film (see fix_coverage).
EDGE_RAMP = 0.5
# ... and only within this many pixels of the covered paint (further out,
# it covers its pixels as its volume makes it, see EDGE_FILM).
EDGE_REACH = 2
# A pixel covered less than this by the tool counts as one it didn't cover,
# for paint pushed into it (see fix_coverage).
PUSHED_A = 0.05
# A film at least this thick (1% of MIN_FILM) says how thick the paint
# around it is (see fix_coverage); thinner ones are traces.
FILM_KNOWN = 1e-2 * MIN_FILM
# Steepest slope (paint thickness per pixel) that fluid (body 0) and very
# stiff (body 1) paint can hold before it flows.
SLOPE_FLUID, SLOPE_STIFF = 0.05, 4.0

# Ploughing (see _plough). A brush combs through the thick wet paint already
# there. Lightly, it rides on the paint it carries; pressed hard, its bristles
# go down through that paint towards a level above the solid surface that has
# nothing to do with its load: GAP (units of paint thickness) times (1.15 -
# pressure) times (1.3 - 0.8 stiffness) (see _gap). They cut the paint above
# it away, deeper along the lines of their clumps the more the brush leaves
# streaks, push more of it out to the sides the harder they are pressed, lay
# their own paint more into their grooves than onto the old paint left standing
# between them, and the old paint folds over the new there: the streaky,
# marbled mark of wet-in-wet work.
GAP = 3.5
# It sets in between these pressures ...
PLOUGH_P0, PLOUGH_P1 = 0.4, 0.6
# ... and between these thicknesses of the earlier wet paint the brush meets
# at a pixel (units of paint thickness): a film a brush can't cut a groove
# into (a wet sky scrubbed on thin) is brushed as it always was ...
PLOUGH_T0, PLOUGH_T1 = 1.5, 2.5
# ... and only in paint stiff enough to hold a groove: from the first body to
# the second (see paint.Paint). The walls of a groove a few units deep between
# clumps a few px apart are about a unit per px steep; paint holds a slope of
# SLOPE_FLUID + (SLOPE_STIFF - SLOPE_FLUID) body^4 (see level), about 0.4 at
# body 0.55 and 1 at 0.7 (tube paint): softer paint, such as paint thinned
# with medium for wet-on-wet work, flows back in and is brushed as it always
# was.
PLOUGH_B0, PLOUGH_B1 = 0.55, 0.7
# How fast hair clears the paint above that level: a pass of hair of
# stiffness s, where its clumps press as hard as on average, leaves
# e^-(PLOUGH_RATE s^3) of it (soft hair bends and slides over thick paint: a
# sable flat, stiffness 0.24, clears about a fifteenth of it, a sable round a
# tenth, a synthetic about half, a hog flat nine tenths). The rest of what a
# brush does ploughing (the side push, where it lays its paint, the folding)
# goes with that share, 1 - e^-(PLOUGH_RATE s^3).
PLOUGH_RATE = 5.0
# How much harder the clumps of the bristles press than the hair between them,
# per unit of the brush's streaks (see BrushType.streaks): where the hair
# pattern (0..1, mean 0.5, see hair_field) is h, they press e^(LANE_K streaks
# (1 - 2h)) times as hard as on average. They lay paint in proportion, and cut
# at a rate that goes with its square root.
LANE_K = 6.0
# Share of the paint the bristles shift that they push out to the sides of
# the stroke; ploughing at full pressure with hair that clears all it cuts,
# they push this much more (the rest of what the brush doesn't take up is
# pushed ahead).
SIDE_SHARE, SIDE_PLOUGH = 0.35, 0.4
# Ploughing, the old paint pushed up between the clumps folds up over the new
# paint they lay there, once the stroke has passed (see fold_pending): over a
# pass it gathers FOLD times the brush's mixing times pressure squared (a
# brush that doesn't mix doesn't fold), in inverse proportion to how hard the
# clumps press (most between them, least along their lines), as far as the
# brush ploughs; for a total f, 1 - e^-f of the thinner of the two wet layers
# trades places with the other. Meanwhile the brush stirs the two layers less
# (see _stir_fraction): in its grooves its paint lies on what is left of the
# old.
FOLD = 1.2

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
        # how far out (px) past the footprint the paint it pushes out lies as
        # a rule (each step's ridges are worked out over its window with this
        # much around it), and how far a brush's side ridges may reach at
        # most: one holding more paint than fits in the first is wider, up to
        # the second (see lay_ridges_off; the stroke reserves that much room)
        "margin",
        "room",
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
        # how much deeper than the weave's own the deepest hollow of the
        # surface is (see Canvas._beyond_weave): a brush whose tips reach
        # that deep reaches everywhere (see _reach)
        "beyond_weave",
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
        # whether it may plough wet paint anywhere: it has bristles (a drop
        # has none) and is pressed above PLOUGH_P0 somewhere along it
        "ploughs",
    ],
)

#: The pixels a step works on: (x0, y0, hh, ww), the window around the swept
#: segment (it may stick out of the canvas); (cy0, cx0, ch, cw), its part on
#: the canvas, at (wy0, wx0) in the window. (The part of it a tool passes
#: over, see touched, is a Window too.)
Window = namedtuple(
    "Window", ["x0", "y0", "hh", "ww", "cy0", "cx0", "ch", "cw", "wy0", "wx0"]
)

#: How the tool moves in a step: step ``i`` of the stroke (the ``first`` or
#: ``last``, or one that sets it down, ``touchdown``), to (cx, cy) along
#: (tx, ty) over ``ds`` px, ending ``arc_i`` px along the stroke, where the
#: lips squeezed out along it swell by ``lip``.
Motion = namedtuple(
    "Motion",
    ["i", "first", "last", "touchdown", "cx", "cy", "tx", "ty", "ds", "arc_i", "lip"],
)

#: How the tool is held in a step.
Hold = namedtuple(
    "Hold",
    [
        # pressure: over the step, at its end and at its start
        "p",
        "p_now",
        "p_before",
        # half-width (px) at the start, at the end and on average; the width,
        # and the bundle's own width there (twice its nominal half-width)
        "r0",
        "r1",
        "r",
        "Wd",
        "Wn",
        # the footprint turned: -1 for a brush turned past 90 degrees (its
        # sides swapped); position across the bristles per half-width across
        # the motion; its lateral half-extent and longest half-chord (in
        # half-widths, see _geometry._chord_table)
        "side",
        "uscale",
        "lmax",
        "h_tab_max",
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

#: One step: the window it works on, how the tool moves and how it is held.
Step = namedtuple("Step", ["win", "motion", "hold"])

#: The wet paint on the canvas (``cv``): the two layers, their coverage and
#: the id of the stroke that laid the top layer (see the module docstring).
Layers = namedtuple("Layers", ["top", "und", "top_a", "und_a", "top_id"])

#: What the wet paint lies on (``aux``): the solid surface (the weave and
#: dried paint, in units of paint thickness), what a nearly dry brush catches
#: on (0..1), how far each point lies below the surface around it, the
#: height of the dried paint, and how far each point of the weave alone lies
#: below the weave around it (the part of ``hollows`` that is the weave's:
#: see _beyond_reach).
Beneath = namedtuple(
    "Beneath", ["floor", "tooth", "hollows", "dry_height", "weave_hollows"]
)

#: The paint on a tool (``br``), (COLUMNS, PACKED) per depth.
Load = namedtuple("Load", ["tip", "belly", "heel"])

#: Paint a brushstroke holds back (``pend``, see stroke._pending): per pixel
#: of its box, the paint, the cover and the reach it brought, and whether
#: any is held; ``where`` (int64), indexed by the PEND_* below; how far the
#: wet paint the brush met there can be ploughed (see _met); and how far the
#: old paint there is to fold up over the new once the stroke has passed (see
#: fold_pending).
Pending = namedtuple(
    "Pending",
    ["paint", "cover", "reach", "held", "where", "met", "folds"],
)
# whether anything is held at all (0: a knife's, which holds nothing); the
# box on the canvas (y0, x0, h, w); the rows and columns of it where any is
# held (row0..row1, col0..col1); and those where any old paint is to fold up
# (see fold_pending); and how far out past the footprint (px) the ridges
# reached where they reached past K.margin (0: nowhere, see lay_ridges_off)
(
    PEND_ON,
    PEND_Y0,
    PEND_X0,
    PEND_H,
    PEND_W,
    PEND_ROW0,
    PEND_ROW1,
    PEND_COL0,
    PEND_COL1,
    PEND_FOLD_ROW0,
    PEND_FOLD_ROW1,
    PEND_FOLD_COL0,
    PEND_FOLD_COL1,
    PEND_REACH,
) = range(14)

#: Which bristle columns are above each pixel (see bristle_taps): for pixel q
#: (row-major), ``ntap[q]`` columns ``cols[q, :]`` with weights ``wts[q, :]``.
Taps = namedtuple("Taps", ["cols", "wts", "ntap"])

#: Paint pushed out of a footprint (packed): all round it, to its left and to
#: its right (see squeeze_out, lay_ridges_off).
Ridges = namedtuple("Ridges", ["ring", "left", "right"])


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
def alike_below(a, b, thr):
    """``alike(a, b) < thr``, without its logarithms where bounds on it
    already tell (see likeness_bounds)."""
    floor = np.float32(1e-9)
    lo, hi = likeness_bounds(a, b, max(a[IV], floor), max(b[IV], floor), False)
    if hi / IV < thr:
        return True
    if lo / IV >= thr:
        return False
    return alike(a, b) < thr


@njit(cache=True, error_model="numpy")
def alike_upto(a, b, thr):
    """``alike(a, b)``, or inf where a bound on it shows it is ``thr`` or more
    (see likeness_bounds)."""
    floor = np.float32(1e-9)
    lo, _ = likeness_bounds(a, b, max(a[IV], floor), max(b[IV], floor), False)
    if lo / IV >= thr:
        return np.inf
    return alike(a, b)


@njit(cache=True, error_model="numpy")
def likeness_bounds(a, b, va, vb, clamp):
    """Bounds (lower, upper) on the sum of |log1p(a[c] / va) - log1p(b[c] /
    vb)| over the bands as :func:`alike` (with ``clamp``, :func:`harmless`:
    each component at least 0) works it out, rounding and all; NaN where
    there are none (a component below 0). The comparisons that decide with
    them decide as the logarithms would, without working out 52 of them.

    With x = a[c] / va and y = b[c] / vb (worked out as there), each term is
    |log((1 + x) / (1 + y))| = 2 atanh(z), z = |x - y| / (2 + x + y) < 1,
    which lies between 2 z and 2 z / (1 - z^2) (the series of atanh). The
    computed term lies within 1e-6 (x + y) of the exact one: in single
    precision, log1p is the C library's log1pf, measured within 1.29 units
    in the last place (ulp) of the exact value over all float32 on this
    build (glibc); the margin allows about 7 ulp on each of the two
    logarithms, the rounding of their difference included (log1p(x) <= x,
    and an ulp of a float32 is at most 1.2e-7 of it). A C library or a vectorised
    log (SVML) with a larger error needs a larger margin."""
    zero = np.float32(0.0)
    lo, hi, lowest = 0.0, 0.0, 0.0
    for c in range(IV):
        if clamp:
            x, y = max(a[c], zero) / va, max(b[c], zero) / vb
        else:
            x, y = a[c] / va, b[c] / vb
        xf, yf = np.float64(x), np.float64(y)
        lowest = min(lowest, xf, yf)
        z = abs(xf - yf) / (2.0 + xf + yf)
        m = 1e-6 * (xf + yf)
        lo += 2.0 * z - m
        hi += 2.0 * z / (1.0 - z * z) + m
    if lowest < 0:
        return np.nan, np.nan
    return lo - 1e-12, hi + 1e-12


@njit(cache=True, error_model="numpy")
def harmless(top, und):
    """Burying the top layer changes nothing you could see: the layer below
    is much the same paint (or there is none)."""
    if und[IV] < 1e-4:
        return True
    zero, floor = np.float32(0.0), np.float32(1e-9)
    va, vb = max(max(top[IV], zero), floor), max(max(und[IV], zero), floor)
    # (decided by bounds where they tell, see likeness_bounds)
    lo, hi = likeness_bounds(top, und, va, vb, True)
    if hi / IV < SAME_LAYER:
        return True
    if lo / IV >= SAME_LAYER:
        return False
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
    tip, belly, heel = br.tip, br.belly, br.heel
    p = st.hold.p
    refill(tip, belly, capacity)
    # pressing the brush down squeezes some paint from the heel towards the tip
    if st.motion.first:
        seep(heel, belly, min(HEEL_SQUEEZE * (0.3 + 2.8 * p * p), 1.0))
    elif st.hold.p_now > st.hold.p_before:  # ... and so does bearing down harder
        seep(
            heel,
            belly,
            min(HEEL_SQUEEZE * 5.0 * (st.hold.p_now - st.hold.p_before), 1.0),
        )
    blend = K.blending
    swap_layers(tip, belly, TIP_SWAP * blend * st.motion.ds / st.hold.Wd)
    swap_layers(belly, heel, HEEL_SWAP * blend * st.motion.ds / st.hold.Wd)
    k = BLEND_RATE * blend * (0.2 + p) ** 2 * st.motion.ds / st.hold.Wd
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
    ia, fa, ib, fb = _lattice_cells(n_s, n_x, ds, dx, cells, grids)
    v4 = np.empty(4)
    for i in range(n_s):
        for j in range(n_x):
            for k in range(4):
                v4[k] = _lattice_at(grids[k], ia[k, i], ib[k, j], fa[k, i], fb[k, j])
            out[i, j] = _hair_value(j * dx, v4, lanes, lane_cells, wander, off)
    _blur_across(out, HAIR_BLUR / dx)
    _spread_hair(out)
    return out


@njit(cache=True, error_model="numpy", inline="always")
def _lattice_cells(n_s, n_x, ds, dx, cells, grids):
    """Where each of ``n_s`` rows (``ds`` px apart) and ``n_x`` columns (``dx``
    px apart) falls in each of the four lattices ``grids`` of cells ``cells``:
    (row cells, row fractions, column cells, column fractions), the same
    numbers lattice() works out at every sample."""
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
    return ia, fa, ib, fb


@njit(cache=True, error_model="numpy", inline="always")
def _lattice_at(g, a0, b0, fa, fb):
    """Smooth value noise from the lattice ``g`` in cell (a0, b0), at the
    smoothed fractions (fa, fb) across it (see lattice_cell)."""
    top = g[a0, b0] + (g[a0, b0 + 1] - g[a0, b0]) * fb
    bot = g[a0 + 1, b0] + (g[a0 + 1, b0 + 1] - g[a0 + 1, b0]) * fb
    return top + (bot - top) * fa


@njit(cache=True, error_model="numpy", inline="always")
def _hair_value(x, v4, lanes, lane_cells, wander, off):
    """The hair pattern (before it is blurred and spread, see hair_field) at
    ``x`` px across the bundle, from the drift, finer drift, regrouping and
    bearing noise there (``v4``)."""
    d = 0.8 * (2.0 * v4[0] - 1.0)
    d += 0.2 * (2.0 * v4[1] - 1.0)
    xx = x + wander * d + off
    c_coarse, c_fine = lane_cells[0], lane_cells[1]
    pa = 0.62 * lattice1(lanes[0], xx / c_coarse)
    pa += 0.38 * lattice1(lanes[1], xx / c_fine)
    pb = 0.62 * lattice1(lanes[2], xx / c_coarse)
    pb += 0.38 * lattice1(lanes[3], xx / c_fine)
    m = smoothstep(0.4, 0.6, v4[2])
    a = 0.45 + 0.95 * v4[3]
    return (pa + (pb - pa) * m - 0.5) * a


@njit(cache=True, error_model="numpy", inline="always")
def _blur_across(out, sig):
    """Blur each row of ``out`` with a Gaussian of ``sig`` samples (clamped at
    the ends): no lane of the hair narrower than a couple of pixels. (Finer
    ones are lost in the pixels anyway; and a narrow crest running across the
    pixel grid on a slant would come out as a row of beads, one per pixel it
    steps over.)"""
    n_s, n_x = out.shape
    rad = int(math.ceil(2.5 * sig))
    if rad <= 0:
        return
    ker = np.empty(2 * rad + 1)
    for k in range(-rad, rad + 1):
        ker[k + rad] = math.exp(-0.5 * (k / sig) ** 2)
    ker /= ker.sum()
    row = np.empty(n_x)
    for i in range(n_s):
        for j in range(n_x):
            acc = 0.0
            for k in range(-rad, rad + 1):
                acc += ker[k + rad] * out[i, min(max(j + k, 0), n_x - 1)]
            row[j] = acc
        for j in range(n_x):
            out[i, j] = row[j]


@njit(cache=True, error_model="numpy", inline="always")
def _spread_hair(out):
    """Rescale ``out`` to mean 0.5 and spread HAIR_SPREAD, clipped to 0..1."""
    n_s, n_x = out.shape
    tot, tot2 = 0.0, 0.0
    for i in range(n_s):
        for j in range(n_x):
            tot += out[i, j]
            tot2 += out[i, j] * out[i, j]
    n = n_s * n_x
    mean = tot / n
    sd = math.sqrt(max(tot2 / n - mean * mean, 1e-12))
    for i in range(n_s):
        for j in range(n_x):
            out[i, j] = clip(0.5 + (out[i, j] - mean) / sd * HAIR_SPREAD, 0.0, 1.0)


# ----------------------------------------------------------------------------
# where the tool touches the canvas


@njit(cache=True, error_model="numpy")
def may_touch(st, K, c_tab):
    """The box (y0, y1, x0, x1) of the step's window outside which the
    footprint (see :func:`footprint`) covers no pixel it sweeps over: none
    further across the motion than its widest (its edge roughened by the
    hair at most as much as a hair pattern, 0 to 1, lets it) and its
    anti-aliasing, and none further along it than its chords reach, at the
    step's start and end, plus the anti-aliasing of the solid ends (a margin
    of 0.01 px for rounding)."""
    win = st.win
    half_aa, m = 0.5 * EDGE_PX, 0.01
    rs = (st.hold.r0, st.hold.r1, st.hold.r)
    rmax = max(rs[0], rs[1], rs[2])
    # across: |lat| (1 + rough (hair - 0.5)) / lmax is at least lr there
    f_lo = 1.0 - 0.5 * abs(K.rough)
    lat_max = (rmax * st.hold.lmax + half_aa + m) / f_lo if f_lo > 0 else np.inf
    # along: the chord's centre and half-length at most as far as at either end
    if K.circle:
        c_lo, c_hi, h_hi = 0.0, 0.0, max(rmax, 0.5)
    else:
        lo, hi = c_tab.min(), c_tab.max()
        c_lo = min(lo * rs[0], lo * rs[1], lo * rs[2])
        c_hi = max(hi * rs[0], hi * rs[1], hi * rs[2])
        h_hi = max(st.hold.h_tab_max * rmax, 0.5)
    a_lo = c_lo - h_hi - half_aa - st.motion.ds - m
    a_hi = c_hi + h_hi + half_aa + m
    y0, y1, x0, x1 = win.hh, 0, win.ww, 0
    for yy in range(win.hh):
        gy = (win.y0 + yy) + 0.5 - st.motion.cy
        for xx in range(win.ww):
            gx = (win.x0 + xx) + 0.5 - st.motion.cx
            along = gx * st.motion.tx + gy * st.motion.ty
            lat_px = -gx * st.motion.ty + gy * st.motion.tx
            if abs(lat_px) < lat_max and a_lo < along < a_hi:
                y0, y1 = min(y0, yy), max(y1, yy + 1)
                x0, x1 = min(x0, xx), max(x1, xx + 1)
    if y0 >= y1:
        return 0, 0, 0, 0
    return y0, y1, x0, x1


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
    anti-aliased edge), which is geometry, not paint thickness.

    A brush's step works it out only over the box of the window where the
    footprint may cover a pixel and sweep over it (see :func:`may_touch`);
    elsewhere all planes are 0, as is a pixel's cover, coverage or sweep
    there (everything the step reads of a pixel it doesn't touch: see
    :func:`touched`). A knife's reads all of its window."""
    win = st.win
    if not K.rigid:
        F = np.zeros((_NF, win.hh, win.ww))
        by0, by1, bx0, bx1 = may_touch(st, K, c_tab)
    else:
        F = np.empty((_NF, win.hh, win.ww))
        by0, by1, bx0, bx1 = 0, win.hh, 0, win.ww
    cx, cy, tx, ty, ds = (
        st.motion.cx,
        st.motion.cy,
        st.motion.tx,
        st.motion.ty,
        st.motion.ds,
    )
    r0, r1, r, p, lmax = st.hold.r0, st.hold.r1, st.hold.r, st.hold.p, st.hold.lmax
    n2 = tex2.shape[0] - 1
    half_aa = 0.5 * EDGE_PX
    soft_pow = 1.0 + 4.0 * K.softness
    span = max(K.feather * 2.0 * r, 1.0)
    hit = False
    for yy in range(by0, by1):
        gy = (win.y0 + yy) + 0.5 - cy
        for xx in range(bx0, bx1):
            gx = (win.x0 + xx) + 0.5 - cx
            along = gx * tx + gy * ty  # <= 0 behind the brush centre
            lat_px = -gx * ty + gy * tx
            r_px = radius_at(along, ds, r0, r1, r)
            lat = lat_px / r_px  # signed, in brush half-widths
            u = lat * st.hold.uscale  # position across the bristles
            if K.is_dab:
                iy, ix = dab_cell(u, along, r, n2)
                hair = tex2[iy, ix]
            else:
                hair = hair_at(tex2, st, K, along, lat_px)
            lr = abs(lat) / lmax * (1.0 + K.rough * (hair - 0.5))
            # anti-aliasing: how much of each pixel the footprint covers across its edge
            edge_px = (1.0 - lr) * (r_px * lmax)
            aa = smoothstep(-half_aa, half_aa, edge_px)
            falloff, profile = _lateral_profile(K, p, lr, aa, soft_pow)
            shape, left_on, pulled = _lifting_off(st, K, tex2, along, lat_px, hair)
            profile *= shape
            profile *= _slant(K, p, u)
            clumped, between = _clumps(K, u, hair, falloff)
            profile *= clumped
            c, h = _chord(K, lat, lr, r_px, grid, c_tab, h_tab)
            rel = along - c
            if K.is_dab:
                profile = _pressed_down(profile, st, K, p, lr, rel / h, hair, soft_pow)
            overlap, swept = _swept(st, K, rel, h, ds, half_aa)
            expo = profile * swept
            if K.feather > 0 and not K.is_dab:
                expo *= _feathered(K, st.motion.arc_i + along, r, span)
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
                and win.wy0 <= yy < win.wy0 + win.ch
                and win.wx0 <= xx < win.wx0 + win.cw
            ):
                hit = True
    return F, hit


@njit(cache=True, error_model="numpy", inline="always")
def radius_at(along, ds, r0, r1, r):
    """The brush's half-width (px) at ``along`` px from its centre along the
    step: it changes smoothly from ``r0`` at the start of the step (``ds`` px
    back) to ``r1`` at its end, so a flat tapers without steps (``r`` when it
    doesn't change)."""
    return r1 + (r0 - r1) * clip(-along / ds, 0.0, 1.0) if r0 != r1 else r


@njit(cache=True, error_model="numpy", inline="always")
def dab_cell(u, along, r, n2):
    """The cell (iy, ix) of a dab's tip pattern (``n2`` + 1 cells a side, over
    DAB_SPAN half-widths each way) at ``u`` across the bristles and ``along``
    px (``r``: the brush's half-width)."""
    ix = min(max(int((u + DAB_SPAN) / (2.0 * DAB_SPAN) * n2), 0), n2)
    iy = min(max(int((along / r + DAB_SPAN) / (2.0 * DAB_SPAN) * n2), 0), n2)
    return iy, ix


@njit(cache=True, error_model="numpy", inline="always")
def hair_coords(st, K, along, lat_px):
    """Where a pixel ``along`` px along the step and ``lat_px`` px across it
    lies in the brush's hair pattern (see hair_field), in its samples: along
    the stroke, across the bundle. (Fixed in stroke coordinates: every step
    sees the same lanes at a pixel.)"""
    return (
        (st.motion.arc_i + along - K.hair_s0) / K.hair_ds,
        (lat_px * st.hold.uscale / st.hold.splay - K.hair_x0) / K.hair_dx,
    )


@njit(cache=True, error_model="numpy", inline="always")
def hair_at(hair, st, K, along, lat_px):
    """The hair pattern ``hair`` (0 to 1: the bristle clumps, a blade's nicks)
    at a pixel ``along`` px along the step and ``lat_px`` px across it."""
    fs, fx = hair_coords(st, K, along, lat_px)
    return sample_field(hair, fs, fx)


@njit(cache=True, error_model="numpy", inline="always")
def _lateral_profile(K, p, lr, aa, soft_pow):
    """The footprint's profile across the stroke at ``lr`` (its distance out
    from the middle, 1 at the edge), anti-aliased ``aa``: (its falloff, a
    soft brush's long feathered one; the profile, flat-topped with low ridges
    at the edges when pressed ``p``)."""
    falloff = aa
    if K.softness > 0 and not K.is_dab:  # a long, feathery falloff
        falloff *= smoothstep(1.0, 1.0 - K.softness, lr) ** soft_pow
    return falloff, falloff * (0.92 + K.ridge * p * smoothstep(0.55, 0.95, lr))


@njit(cache=True, error_model="numpy", inline="always")
def _lifting_off(st, K, tex2, along, lat_px, hair):
    """Lifting off: over the last stretch of the stroke the bristle clumps
    leave the paint one after another (a ragged end, not a cut in the shape of
    the footprint), each pulling the paint up into a crest where it leaves.
    Returns (the factor on the profile, how much of the pixel the brush still
    covers, how far the paint is pulled up: 1 none), all 1 elsewhere."""
    if st.hold.lift_len > 0:
        s_left = K.arc_end - (st.motion.arc_i + along)
        if s_left < st.hold.lift_len:
            t = 1.0 - max(s_left, 0.0) / st.hold.lift_len
            # (a clump a few px wide leaves together: the hair pattern
            # smoothed across, its spread restored; a clump thins out over
            # the last part of its way, and a fine single hair doesn't
            # draw a line of its own)
            fs, fx = hair_coords(st, K, along, lat_px)
            dxs = LIFT_CLUMP / K.hair_dx
            clump = 0.25 * (
                sample_field(tex2, fs, fx - dxs)
                + 2.0 * hair
                + sample_field(tex2, fs, fx + dxs)
            )
            clump = 0.5 + 1.6 * (clump - 0.5)
            g = smoothstep(t - 0.3, t + 0.3, 0.1 + 0.75 * clump)
            # (thinning to its tip, and pulled up just behind it)
            return (
                g * math.sqrt(g),
                g,
                1.0 + st.hold.lift_amp * 6.75 * g * g * (1.0 - g),
            )
    return 1.0, 1.0, 1.0


@njit(cache=True, error_model="numpy", inline="always")
def _slant(K, p, u):
    """Bias-cut bristles: the toe (u = -1) touches first. The factor on the
    profile at ``u`` across the bristles, pressed ``p``; 1 for square-cut
    bristles (``K.slant`` 0)."""
    if K.slant > 0:
        press = 0.25 + 0.9 * p
        return clip(
            (press - K.slant * (1.0 + clip(u, -1.0, 1.0)) / 2.0) / 0.3, 0.0, 1.0
        )
    return 1.0


@njit(cache=True, error_model="numpy", inline="always")
def _clumps(K, u, hair, falloff):
    """Separate clumps of bristles (fan brushes), or the teeth of a comb, at
    ``u`` across the bristles: (the factor on the profile there, how much it
    lies in a gap between them, of ``falloff``); (1, 0) without."""
    if K.clumps > 0:
        m = 0.5 + 0.5 * math.cos(
            math.pi * K.clumps * (clip(u, -1.0, 1.0) + 1.0) + 2.0 * (hair - 0.5)
        )
        return smoothstep(0.3, 0.7, m), falloff * smoothstep(0.6, 0.2, m)
    return 1.0, 0.0


@njit(cache=True, error_model="numpy", inline="always")
def _chord(K, lat, lr, r_px, grid, c_tab, h_tab):
    """The chord of the footprint through a pixel along the motion, at
    ``lat`` half-widths across (``lr``: its distance out, ``r_px``: the
    half-width): (its centre, its half-length, at least 0.5), in px."""
    if K.circle:
        return 0.0, max(r_px * math.sqrt(max(1.0 - lr * lr, 0.0)), 0.5)
    g0, g1 = grid[0], grid[grid.shape[0] - 1]
    c = interp(lat, g0, g1, c_tab, c_tab[0], c_tab[c_tab.shape[0] - 1]) * r_px
    h = max(
        interp(lat, g0, g1, h_tab, h_tab[0], h_tab[h_tab.shape[0] - 1]) * r_px,
        0.5,
    )
    return c, h


@njit(cache=True, error_model="numpy", inline="always")
def _pressed_down(profile, st, K, p, lr, rel_h, hair, soft_pow):
    """A dab's ``profile`` at a pixel ``lr`` out across it and ``rel_h`` along
    (its chord's half-lengths from its middle). Pressed straight down, the
    edge is the same all round: how far out from the middle, not across a
    stroke."""
    rr = math.sqrt(lr * lr + clip(rel_h, -1.0, 1.0) ** 2 * max(1.0 - lr * lr, 0.0))
    # the paint builds up a little all round the rim (not in two lips, as
    # along the sides of a stroke)
    profile *= (0.92 + K.ridge * p * smoothstep(0.55, 0.95, rr)) / (
        0.92 + K.ridge * p * smoothstep(0.55, 0.95, lr)
    )
    if K.softness > 0:  # a soft brush pressed down: feathered all round
        profile *= smoothstep(1.0, 1.0 - K.softness, rr) ** soft_pow
    # tapped lightly, only the tips of the bristles touch: a stippled, broken
    # mark (a glob of paint on it touches all over even then)
    tip_thr = 0.75 - 1.5 * p - 0.4 * min(st.hold.overload, 1.5)
    if tip_thr > 0:
        profile *= smoothstep(tip_thr - 0.12, tip_thr + 0.12, hair)
    return profile


@njit(cache=True, error_model="numpy", inline="always")
def _swept(st, K, rel, h, ds, half_aa):
    """How far (px) the footprint's chord through a pixel (``rel`` px past
    its centre, ``h`` its half-length) passes over it in this step of ``ds``
    px, topped up to a full pass where the tool touches down or lifts off
    (solid ends); and that as a fraction of a full pass."""
    overlap = max(min(0.0, rel + h) - max(-ds, rel - h), 0.0)
    if st.motion.first and K.end_k > 0:
        a0 = rel + ds
        overlap += (
            K.end_k
            * smoothstep(-h - half_aa, -h + half_aa, a0)
            * clip(h - a0, 0.0, 2.0 * h)
        )
    if st.motion.last and K.end_k > 0:
        overlap += (
            K.end_k
            * smoothstep(h + half_aa, h - half_aa, rel)
            * clip(h + rel, 0.0, 2.0 * h)
        )
    swept = overlap / (2.0 * h)  # fraction of a full pass
    if K.is_dab:
        # pressed down once: one full pass at most (where the mark is short,
        # at its two ends, the nudge and the set-down and lift-off would
        # otherwise count it two or three times)
        swept = min(swept, 1.0)
    return overlap, swept


@njit(cache=True, error_model="numpy", inline="always")
def _feathered(K, s_pix, r, span):
    """A feathered stroke fades in and out over ``span`` px from each end: the
    factor at ``s_pix`` px along it (``r``: the brush's half-width)."""
    return smoothstep(-r, -r + span, s_pix) * smoothstep(
        -r, -r + span, K.arc_end - s_pix
    )


@njit(cache=True, error_model="numpy")
def bristle_taps(u_px, span, nsub, ring, rho):
    """Which bristle columns are above each pixel of the (ch, cw) window:
    up to COLUMNS (column, weight) taps per pixel, blending linearly between
    the nearest columns, sampled ``nsub`` times across the pixel (Taps)."""
    ch, cw = u_px.shape
    npx = ch * cw
    # (each sample touches two columns, two on each side of a ring: no pixel
    # has more taps than that)
    most = min(4 * nsub if ring else 2 * nsub, COLUMNS)
    cols = np.empty((npx, most), np.int32)
    wts = np.empty((npx, most))
    ntap = np.zeros(npx, np.int32)
    # (the columns a pixel touches, lo..hi, are all that is set in acc: only
    # those are read, in order, and cleared again)
    acc = np.zeros(COLUMNS)
    for y in range(ch):
        for x in range(cw):
            q = y * cw + x
            lo, hi = COLUMNS, -1
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
                        c1 = (c0 + 1) % COLUMNS
                        fr = pos - c0
                        acc[c0] += 0.5 * (1.0 - fr) / nsub
                        acc[c1] += 0.5 * fr / nsub
                        lo, hi = min(lo, c0, c1), max(hi, c0, c1)
                else:
                    # column b covers an equal slice of the brush, centred at (b + 0.5)
                    # / COLUMNS
                    colf = (uc + 1.0) * 0.5 * COLUMNS - 0.5
                    c0 = int(clip(math.floor(colf), 0.0, COLUMNS - 2.0))
                    fr = clip(colf - c0, 0.0, 1.0)
                    acc[c0] += (1.0 - fr) / nsub
                    acc[c0 + 1] += fr / nsub
                    lo, hi = min(lo, c0), max(hi, c0 + 1)
            n = 0
            for b in range(lo, hi + 1):
                if acc[b] != 0.0:
                    cols[q, n] = b
                    wts[q, n] = acc[b]
                    n += 1
                acc[b] = 0.0
            ntap[q] = n
    return Taps(cols, wts, ntap)


@njit(cache=True, error_model="numpy")
def crop_taps(taps, ww, wy0, wx0, ch, cw):
    """The taps (see :func:`bristle_taps`) of the (ch, cw) part at (wy0, wx0)
    of a window ``ww`` px wide, from the taps of all of it."""
    cols, wts, ntap = taps.cols, taps.wts, taps.ntap
    n = ch * cw
    out_cols = np.empty((n, cols.shape[1]), np.int32)
    out_wts = np.empty((n, cols.shape[1]))
    out_n = np.zeros(n, np.int32)
    for y in range(ch):
        for x in range(cw):
            q, r = (wy0 + y) * ww + wx0 + x, y * cw + x
            out_n[r] = ntap[q]
            for t in range(ntap[q]):
                out_cols[r, t] = cols[q, t]
                out_wts[r, t] = wts[q, t]
    return Taps(out_cols, out_wts, out_n)


@njit(cache=True, error_model="numpy")
def wandering_bands(F, wy0, wx0, ch, cw, s0, period, bands, wander):
    """Where across a knife's blade (-1..1) the paint it lays on each pixel of
    the (ch, cw) part at (wy0, wx0) of the footprint ``F`` comes from: the
    pixel's place across the blade, moved by the colour bands on the blade,
    which wander by up to ``wander`` half-lengths along the stroke (``bands``:
    8 rows to ``period`` px along it, from ``s0`` px at the step's end; 64
    columns across). Bilinear in ``bands``, as _filters._bilinear."""
    out = np.empty((ch, cw))
    n0, n1 = bands.shape
    for y in range(ch):
        for x in range(cw):
            u = F[F_U, wy0 + y, wx0 + x]
            a = min(
                max((s0 + F[F_ALONG, wy0 + y, wx0 + x]) / period * 8.0, 0.0), n0 - 1.001
            )
            b = min(max((u + 1.0) * 0.5 * 63.0, 0.0), 62.999)
            ia, ib = min(int(a), n0 - 2), min(int(b), n1 - 2)
            fa, fb = a - ia, b - ib
            v = (bands[ia, ib] * (1 - fb) + bands[ia, ib + 1] * fb) * (1 - fa) + (
                bands[ia + 1, ib] * (1 - fb) + bands[ia + 1, ib + 1] * fb
            ) * fa
            out[y, x] = u + wander * (2.0 * v - 1.0)
    return out


@njit(cache=True, error_model="numpy")
def at_pixels(taps, per_column, ch, cw):
    """A per-column quantity, as seen at each pixel under the brush."""
    cols, wts, ntap = taps.cols, taps.wts, taps.ntap
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
def surface_factors(cv, aux, win, K):
    """How the bare surface changes pickup, deposit and mixing (per pixel) and
    drag, over the part on the canvas of a step's window ``win``: where the
    brush works directly on the priming (little paint below), the surface
    itself decides how paint grips and slides."""
    top, und = cv.top, cv.und
    dry_h = aux.dry_height
    ch, cw = win.ch, win.cw
    f_pick = np.ones((ch, cw))
    f_dep = np.ones((ch, cw))
    f_mix = np.ones((ch, cw))
    if not K.surface_acts:
        return f_pick, f_dep, f_mix, 1.0
    sl, ab, to = K.slickness, K.absorbency, K.surface_tooth
    sum_bare = 0.0
    for y in range(ch):
        for x in range(cw):
            Y, X = win.cy0 + y, win.cx0 + x
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
    win = st.win
    top, und = cv.top, cv.und
    dry_h = aux.dry_height
    ch, cw = part.ch, part.cw
    f_pick = np.ones((ch, cw))
    f_dep = np.ones((ch, cw))
    f_mix = np.ones((ch, cw))
    if not K.surface_acts:
        return f_pick, f_dep, f_mix, 1.0
    sl, ab, to = K.slickness, K.absorbency, K.surface_tooth
    sum_bare = 0.0
    for y in range(win.ch):
        for x in range(win.cw):
            Y, X = win.cy0 + y, win.cx0 + x
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
    # (off the canvas, the surface goes on as it is at its edge: see off_canvas)
    n_px = max(win.ch * win.cw, 1)
    return f_pick, f_dep, f_mix, 1.0 + (sl - 0.5) * sum_bare / n_px


@njit(cache=True, error_model="numpy")
def surface_at(cv, aux, Y, X, K):
    """:func:`surface_factors` (pickup, deposit) at the one pixel (Y, X)."""
    if not K.surface_acts:
        return 1.0, 1.0
    top, und, dry_h = cv.top, cv.und, aux.dry_height
    sl, ab, to = K.slickness, K.absorbency, K.surface_tooth
    wet = top[Y, X, IV] + und[Y, X, IV]
    dh = dry_h[Y, X]
    bare = math.exp(-(wet + dh) / 0.4)
    on_ground = math.exp(-wet / 1.5) * math.exp(-dh / 0.4)
    f_pick = max(1.0 + on_ground * (2.0 * (sl - 0.5) - 0.8 * ab), 0.1)
    f_dep = max(1.0 + bare * (0.6 * ab + 0.4 * (to - 0.5) - 0.4 * (sl - 0.5)), 0.4)
    return f_pick, f_dep


@njit(cache=True, error_model="numpy")
def _passes_off(st, F, H, W, y, x):
    """Whether window pixel (y, x) lies off the canvas and the tool passes
    over it in this step."""
    Y, X = st.win.y0 + y, st.win.x0 + x
    on = 0 <= Y < H and 0 <= X < W
    return not on and F[F_AA, y, x] != 0 and F[F_OVERLAP, y, x] != 0


@njit(cache=True, error_model="numpy")
def off_canvas(st, F, H, W):
    """The pixels of the step's window that lie off the canvas and that the
    tool passes over (as :func:`touched`): rows of (window y, window x, and
    the nearest canvas pixel's Y, X). There the tool goes on as if the canvas
    went on as it is at its edge: it uses up the paint it would lay there,
    from the bristles that would lay it."""
    win = st.win
    if win.ch == win.hh and win.cw == win.ww:
        return np.empty((0, 4), np.int64)
    n = 0
    for y in range(win.hh):
        for x in range(win.ww):
            if _passes_off(st, F, H, W, y, x):
                n += 1
    out = np.empty((n, 4), np.int64)
    i = 0
    for y in range(win.hh):
        for x in range(win.ww):
            if _passes_off(st, F, H, W, y, x):
                out[i, 0], out[i, 1] = y, x
                out[i, 2] = min(max(win.y0 + y, 0), H - 1)
                out[i, 3] = min(max(win.x0 + x, 0), W - 1)
                i += 1
    return out


@njit(cache=True, error_model="numpy")
def column_demand(taps, d):
    """What the volume ``d`` per pixel asks of each bristle column."""
    cols, wts, ntap = taps.cols, taps.wts, taps.ntap
    hh, ww = d.shape
    demand = np.zeros(COLUMNS)
    for y in range(hh):
        for x in range(ww):
            q = y * ww + x
            for t in range(ntap[q]):
                demand[cols[q, t]] += wts[q, t] * d[y, x]
    return demand


@njit(cache=True, error_model="numpy")
def touched(st, F):
    """The part of the step's window on the canvas that the tool passes over
    in this step (where its footprint ``F`` covers a pixel and sweeps over
    it), as a Window whose canvas part (cy0, cx0, ch, cw at wy0, wx0) is that
    part. Everywhere else in the window its exposure and coverage are 0: the
    step adds nothing there, moves nothing and lifts nothing."""
    win = st.win
    y_lo, y_hi, x_lo, x_hi = win.ch, -1, win.cw, -1
    for y in range(win.ch):
        for x in range(win.cw):
            wy, wx = win.wy0 + y, win.wx0 + x
            if F[F_AA, wy, wx] != 0 and F[F_OVERLAP, wy, wx] != 0:
                y_lo, y_hi = min(y_lo, y), max(y_hi, y)
                x_lo, x_hi = min(x_lo, x), max(x_hi, x)
    if y_hi < 0:  # (not reached: there is no step without exposure)
        y_lo, y_hi, x_lo, x_hi = 0, -1, 0, -1
    return Window(
        x0=win.x0,
        y0=win.y0,
        hh=win.hh,
        ww=win.ww,
        cy0=win.cy0 + y_lo,
        cx0=win.cx0 + x_lo,
        ch=y_hi - y_lo + 1,
        cw=x_hi - x_lo + 1,
        wy0=win.wy0 + y_lo,
        wx0=win.wx0 + x_lo,
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
    (``pend``, see Pending): a pixel first grazed by the edge of a
    looping stroke may later get its body."""
    top, und, top_a, und_a, ids = cv.top, cv.und, cv.top_a, cv.und_a, cv.top_id
    holding = pend.where[PEND_ON] != 0 and by_film and not moved
    hh, ww = d.shape
    for y in range(hh):
        Y = sy0 + y
        for x in range(ww):
            X = sx0 + x
            dd = d[y, x]
            # (a speck of dust landing is no paint laid)
            touched = dd > SPECK
            old = touched and ids[Y, X] != stroke_id
            laid = False  # nothing more to lay here in this step
            if moved and old and _join_top_if_alike(top, und, dep[y, x], Y, X):
                laid, touched, old = True, False, False
            # an earlier top layer worth keeping out of the layer below: hold the new
            # paint back until we know how well the stroke covers it
            known_harmless = False  # (as worked out here: it is the same below)
            if (
                holding
                and old
                and _pending_at(pend.where, Y, X)
                and top[Y, X, IV] > 1e-4
            ):
                known_harmless = harmless(top[Y, X], und[Y, X])
                if not known_harmless:
                    _hold_back(
                        cv, pend, dep, cover, reach, has_reach, y, x, Y, X, stroke_id
                    )
                    laid, touched, old = True, False, False
            if old:
                # paint from an earlier stroke stops being "top" once new paint covers
                # it (an empty top layer has no coverage to pass on)
                if not by_film:
                    bury_top(top, und, top_a, und_a, Y, X)
                else:
                    k = _film_buries(
                        top, und, dd, reach, has_reach, y, x, Y, X, known_harmless
                    )
                    _bury_share(top, und, top_a, und_a, Y, X, np.float32(k))
                ids[Y, X] = stroke_id
            # the tool covers what it passes over with this stroke's paint, also where
            # it adds nothing new in this step (it already laid paint there)
            if touched or (ids[Y, X] == stroke_id and top[Y, X, IV] > EMPTY):
                top_a[Y, X] = min(
                    top_a[Y, X] + np.float32(cover[y, x]), np.float32(1.0)
                )
            if not laid:
                for c in range(PACKED):
                    top[Y, X, c] += np.float32(dep[y, x, c])


@njit(cache=True, error_model="numpy", inline="always")
def _join_top_if_alike(top, und, pk, Y, X):
    """Paint ``pk`` shoved onto pixel (Y, X) that holds two wet layers makes
    three layers in two places: merge the two most alike. The new paint joins
    the top layer only if it really is much the same paint (white pushed onto
    white); a different colour stays on top of the others. Adds it to the top
    layer if so, and returns whether it did."""
    if not (top[Y, X, IV] > EMPTY and und[Y, X, IV] > EMPTY):
        return False
    near = alike_upto(top[Y, X], pk, SAME_PAINT)
    if near < SAME_PAINT and near < alike(und[Y, X], top[Y, X]):
        for c in range(PACKED):
            top[Y, X, c] += np.float32(pk[c])
        return True
    return False


@njit(cache=True, error_model="numpy", inline="always")
def _pending_at(where, Y, X):
    """Whether canvas pixel (Y, X) lies in the box of pending paint (``where``,
    see Pending)."""
    y0, x0 = where[PEND_Y0], where[PEND_X0]
    return y0 <= Y < y0 + where[PEND_H] and x0 <= X < x0 + where[PEND_W]


@njit(cache=True, error_model="numpy", inline="always")
def _hold_back(cv, pend, dep, cover, reach, has_reach, y, x, Y, X, stroke_id):
    """Hold the paint ``dep`` laid at window pixel (y, x), canvas pixel (Y, X),
    back in the pending paint (see deposit), with the ``cover`` and ``reach``
    it brought. Once the body of the brush has been over it, or it laid a
    covering film, it goes on top of the old top, which it buries."""
    top, und, top_a, und_a, ids = cv.top, cv.und, cv.top_a, cv.und_a, cv.top_id
    paint, pcov, prch, pon = pend.paint, pend.cover, pend.reach, pend.held
    where = pend.where
    py, px = Y - where[PEND_Y0], X - where[PEND_X0]
    for c in range(PACKED):
        paint[py, px, c] += np.float32(dep[y, x, c])
    pcov[py, px] += cover[y, x]
    if has_reach:
        prch[py, px] += reach[y, x]
    pon[py, px] = True
    # (where paint is held)
    where[PEND_ROW0] = min(where[PEND_ROW0], py)
    where[PEND_ROW1] = max(where[PEND_ROW1], py + 1)
    where[PEND_COL0] = min(where[PEND_COL0], px)
    where[PEND_COL1] = max(where[PEND_COL1], px + 1)
    if buried(paint[py, px, IV], prch[py, px]) >= BURIED:
        if top[Y, X, IV] > EMPTY:
            und_a[Y, X] = max(und_a[Y, X], top_a[Y, X])
        for c in range(PACKED):
            und[Y, X, c] += top[Y, X, c]
            top[Y, X, c] = paint[py, px, c]
            paint[py, px, c] = 0.0
        top_a[Y, X] = min(pcov[py, px], 1.0)
        ids[Y, X] = stroke_id
        pcov[py, px] = 0.0
        prch[py, px] = 0.0
        pon[py, px] = False


@njit(cache=True, error_model="numpy", inline="always")
def bury_top(top, und, top_a, und_a, Y, X):
    """The whole top layer of pixel (Y, X) goes under: it joins the layer
    below, which then covers as much of the pixel as either did (an empty top
    layer has no coverage to pass on), and the top layer is empty."""
    has_top = top[Y, X, IV] > EMPTY
    for c in range(PACKED):
        und[Y, X, c] += top[Y, X, c]
        top[Y, X, c] = 0.0
    if has_top:
        und_a[Y, X] = max(und_a[Y, X], top_a[Y, X])
    top_a[Y, X] = 0.0


@njit(cache=True, error_model="numpy", inline="always")
def _film_buries(top, und, dd, reach, has_reach, y, x, Y, X, known_harmless):
    """The share of the old top layer at canvas pixel (Y, X) that new paint
    ``dd`` thick buries (see :func:`buries`); at least as much as the body of
    the brush passing over buries (``reach`` at window pixel (y, x), from
    none at BODY0 to all of it at BODY1); all of it where the two old layers
    are much the same paint (``known_harmless``: found so already)."""
    k = buries(dd)
    if has_reach:
        k = max(k, smoothstep(BODY0, BODY1, reach[y, x]))
    if k < 1.0 and (known_harmless or harmless(top[Y, X], und[Y, X])):
        k = 1.0
    return k


@njit(cache=True, error_model="numpy", inline="always")
def _bury_share(top, und, top_a, und_a, Y, X, k32):
    """The share ``k32`` (single precision) of the top layer of pixel (Y, X)
    goes under, and it covers that much less of the pixel; the layer below
    covers as much as either did once more than a speck of it went."""
    has_top = top[Y, X, IV] > EMPTY
    for c in range(PACKED):
        going = k32 * top[Y, X, c]
        und[Y, X, c] += going
        top[Y, X, c] -= going
    if k32 > 0.01 and has_top:
        und_a[Y, X] = max(und_a[Y, X], top_a[Y, X])
    top_a[Y, X] *= np.float32(1.0) - k32


@njit(cache=True, error_model="numpy")
def settle_pending(cv, pend, stroke_id):
    """The stroke has passed: settle the pixels its edge only grazed (see
    Pending). The old top keeps its place, the share the faint new
    film would bury goes under, and the new paint stirs into the rest."""
    top, und, top_a, und_a, ids = cv.top, cv.und, cv.top_a, cv.und_a, cv.top_id
    paint, pcov, prch, pon = pend.paint, pend.cover, pend.reach, pend.held
    where = pend.where
    if where[PEND_ON] == 0:
        return
    for py in range(where[PEND_ROW0], where[PEND_ROW1]):
        for px in range(where[PEND_COL0], where[PEND_COL1]):
            if not pon[py, px]:
                continue
            Y, X = where[PEND_Y0] + py, where[PEND_X0] + px
            k = buried(paint[py, px, IV], prch[py, px])
            if k < 1.0 and harmless(top[Y, X], und[Y, X]):
                k = 1.0
            k32 = np.float32(k)
            # (an empty top layer has no coverage to pass on)
            if k32 > 0.01 and top[Y, X, IV] > EMPTY:
                und_a[Y, X] = max(und_a[Y, X], top_a[Y, X])
            for c in range(PACKED):
                going = k32 * top[Y, X, c]
                und[Y, X, c] += going
                top[Y, X, c] += paint[py, px, c] - going
            top_a[Y, X] = max(
                top_a[Y, X] * (np.float32(1.0) - k32),
                min(pcov[py, px], np.float32(1.0)),
            )
            ids[Y, X] = stroke_id


@njit(cache=True, error_model="numpy")
def share_by_reach(tip, taps, d, extra, ring):
    """A brush pressed straight down splays its bristles over the whole mark:
    each column gives its paint over the part of the mark it reaches. Shares
    the paint in the tips out between the columns in proportion to what is
    asked of each (a round's middle columns reach right across the mark, its
    edge columns only the short ends), each keeping its own paint; the total
    stays the same. Otherwise the ends of the mark, fed by full columns, would
    come out thicker than its middle. ``extra``: more asked of each column
    (off the canvas; empty: nothing). ``ring``: the columns go round a
    round's bundle.

    Paint moves between the columns with what is in it, from each column to
    its neighbours, as far as it has to go: every pigment is kept, and each
    column keeps as much of its own paint as it can. (Scaling each column
    instead made and destroyed pigment wherever the columns held different
    paints.) Where they all hold the same paint, that is the same as scaling
    each column, which is what is done."""
    demand = demand_with(taps, d, extra)
    held, asked = 0.0, 0.0
    for b in range(COLUMNS):
        if tip[b, IV] > 0:
            held += tip[b, IV]
            asked += demand[b]
    if held <= 0 or asked <= 0:
        return
    if same_paint(tip):
        _scale_columns(tip, held, demand, asked)
        return
    flow = _column_flows(tip, held, demand, asked)
    start = _cut_ring(flow) if ring else 0
    _pass_along(tip, flow, start)


@njit(cache=True, error_model="numpy", inline="always")
def demand_with(taps, d, extra):
    """What the volume ``d`` per pixel asks of each bristle column (see
    column_demand), plus ``extra`` volume asked of each (empty: nothing)."""
    demand = column_demand(taps, d)
    if extra.shape[0] > 0:
        for b in range(COLUMNS):
            demand[b] += extra[b]
    return demand


@njit(cache=True, error_model="numpy", inline="always")
def _scale_columns(tip, held, demand, asked):
    """Scale each column of ``tip`` holding paint so that it holds its share
    of the ``held`` volume, in proportion to the volume ``demand`` asked of
    it (``asked`` of all of them)."""
    for b in range(COLUMNS):
        tv = tip[b, IV]
        if tv > 0:
            f = held * demand[b] / asked / tv
            for c in range(PACKED):
                tip[b, c] *= f


@njit(cache=True, error_model="numpy", inline="always")
def _column_flows(tip, held, demand, asked):
    """The volume that flows across the boundary between each column and the
    next (to the right: > 0) so that each ends up with its share of the
    ``held`` volume (see _scale_columns)."""
    flow = np.zeros(COLUMNS)
    acc = 0.0
    for b in range(COLUMNS - 1):
        tv = tip[b, IV]
        if tv > 0:
            acc += tv - held * demand[b] / asked
        flow[b] = acc
    return flow


@njit(cache=True, error_model="numpy", inline="always")
def _cut_ring(flow):
    """On a ring (a round's bundle) the last column is next to the first: the
    same flow all the way round can be taken off, and the least paint moves
    when that is a median of the flows, so that paint near the seam goes the
    short way round. Taking off one of the flows itself leaves its boundary
    with none at all: cut the ring there, and the sweep is the same as along
    a flat's row of columns (each column takes in what comes to it before it
    gives). Takes it off ``flow``, and returns the column after the cut."""
    k = int(np.argsort(flow)[COLUMNS // 2])
    c = flow[k]
    for b in range(COLUMNS):
        flow[b] -= c
    flow[k] = 0.0
    return (k + 1) % COLUMNS


@njit(cache=True, error_model="numpy", inline="always")
def _pass_along(tip, flow, start):
    """Move the paint of the columns of ``tip`` across their boundaries as
    ``flow`` says, from column ``start`` on round the row."""
    nb = COLUMNS - 1
    # moved from the giving side as it is then (having taken in what comes to
    # it from the other side first): to the right, then to the left
    for i in range(nb):
        b = (start + i) % COLUMNS
        if flow[b] > 0:
            _move_paint(tip, b, (b + 1) % COLUMNS, flow[b])
    for i in range(nb - 1, -1, -1):
        b = (start + i) % COLUMNS
        if flow[b] < 0:
            _move_paint(tip, (b + 1) % COLUMNS, b, -flow[b])


@njit(cache=True, error_model="numpy")
def _move_paint(cols, src, dst, vol):
    """Move volume ``vol`` of column ``src``'s paint, with what is in it,
    into column ``dst`` (all it holds at most)."""
    have = cols[src, IV]
    if have <= 0:
        return
    f = min(vol / have, 1.0)
    for c in range(PACKED):
        m = cols[src, c] * f
        cols[src, c] -= m
        cols[dst, c] += m


# Two columns' paints are the same paint (for share_by_reach, and two pixels'
# for pull_up) if what is in a unit of each differs by no more than this,
# relative to the largest component: rounding.
SAME_COLUMN_PAINT = 1e-6


@njit(cache=True, error_model="numpy")
def same_paint(cols):
    """Whether the rows of ``cols`` (n, PACKED: a brush's columns, or pixels)
    holding paint (volume > 0) all hold the same paint (to rounding, see
    SAME_COLUMN_PAINT)."""
    ref = -1
    big = 0.0
    for b in range(cols.shape[0]):
        tv = cols[b, IV]
        if tv > 0:
            if ref < 0:
                ref = b
            for c in range(PACKED):
                big = max(big, abs(cols[b, c] / tv))
    if ref < 0:
        return True
    rv = cols[ref, IV]
    tol = SAME_COLUMN_PAINT * big
    for b in range(cols.shape[0]):
        tv = cols[b, IV]
        if tv > 0 and b != ref:
            for c in range(PACKED):
                if abs(cols[b, c] / tv - cols[ref, c] / rv) > tol:
                    return False
    return True


# Share of the glob on a brush that a dab presses out (at medium pressure).
DAB_GLOB = 0.18


@njit(cache=True, error_model="numpy")
def press_glob(d, F, st, part, col_vol, col_over, hold, pattern, off, d_off):
    """A loaded brush pressed straight down (a dab) leaves part of the glob
    it carries as a mound under it: highest in the middle, flatter the harder
    it is pressed, lumpy where its tips (``pattern``) touched. Adds it to the
    volume ``d`` the dab asks for (over ``part`` of its footprint ``F``), and
    ``d_off`` at the pixels ``off`` the canvas (see :func:`off_canvas`)."""
    glob = 0.0
    for b in range(COLUMNS):
        if col_over[b] > hold:
            glob += col_vol[b] * (col_over[b] - hold) / col_over[b]
    if glob <= 0:
        return
    p = st.hold.p
    vol = DAB_GLOB * (0.6 + 0.8 * p) * glob
    a = 1.6 - 1.2 * p  # how domed
    ch, cw = d.shape
    w = np.zeros((ch, cw))
    tot = 0.0
    for y in range(ch):
        for x in range(cw):
            v = _mound(F, st, part.wy0 + y, part.wx0 + x, a, pattern)
            w[y, x] = v
            tot += v
    no = off.shape[0]
    w_off = np.zeros(no)
    for i in range(no):
        v = _mound(F, st, off[i, 0], off[i, 1], a, pattern)
        w_off[i] = v
        tot += v
    if tot <= 0:
        return
    for y in range(ch):
        for x in range(cw):
            d[y, x] += vol * w[y, x] / tot
    for i in range(no):
        d_off[i] += vol * w_off[i] / tot


@njit(cache=True, error_model="numpy")
def _mound(F, st, wy, wx, a, pattern):
    """How much of a dab's mound (see press_glob) lies on window pixel (wy, wx)."""
    aa = F[F_AA, wy, wx]
    if aa <= 0 or F[F_OVERLAP, wy, wx] <= 0:
        return 0.0
    n2 = pattern.shape[0] - 1
    lr = min(abs(F[F_LAT, wy, wx]) / max(st.hold.r * st.hold.lmax, 1e-9), 1.0)
    rel = clip(F[F_REL, wy, wx] / max(F[F_H, wy, wx], 1e-9), -1.0, 1.0)
    rr = min(math.sqrt(lr * lr + rel * rel * (1.0 - lr * lr)), 1.0)
    iy, ix = dab_cell(F[F_U, wy, wx], F[F_ALONG, wy, wx], st.hold.r, n2)
    return aa * (1.0 - rr * rr) ** a * (0.6 + 0.8 * pattern[iy, ix])


@njit(cache=True, error_model="numpy")
def feed_glob(tip, belly, taps, d, col_glob, extra):
    """A glob of paint sits on the outside of the hair, not in its tips: where
    a column carries one, what this step asks of it beyond what its tip holds
    comes straight from the glob (the belly). So how thick an overloaded
    brush lays follows its load, and doesn't depend on how many steps the
    stroke is cut into (fed only from its tip, refilled once a step, a column
    could give no more per step than the tip holds: a cap that a stroke cut
    into finer steps, a curve, got past). ``extra``: more asked of each
    column (off the canvas; empty: nothing)."""
    demand = demand_with(taps, d, extra)
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
def lay_off(
    cv,
    pend,
    br,
    taps,
    sy0,
    sx0,
    d,
    cover,
    stroke_id,
    moved,
    by_film,
    reach,
    has_reach,
    extra,
):
    """Lay volume ``d`` per pixel from the tips of the columns above it (see
    deposit for ``cover`` to ``has_reach``); each column can only give what
    it holds. The columns are also asked for ``extra`` volume each (empty:
    nothing) that they lay off the canvas: they give their share of it as
    of the rest, and it is gone."""
    tip = br.tip
    give, comp = _give(tip, demand_with(taps, d, extra))
    used = np.zeros(COLUMNS)
    dep, d_px, total = _from_columns(taps, d, give, comp, used)
    gone = 0.0
    if extra.shape[0] > 0:
        for b in range(COLUMNS):
            part = extra[b] * give[b]
            used[b] += part
            gone += part
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
    if total > 0 or gone > 0:
        for b in range(COLUMNS):
            for c in range(PACKED):
                tip[b, c] -= used[b] * comp[b, c]
            tip[b, IV] = max(tip[b, IV], 0.0)


@njit(cache=True, error_model="numpy", inline="always")
def _give(tip, demand):
    """How much of what is asked of each column (``demand``, volume) it gives:
    all of it, or as much as it holds; and what is in a unit volume of its
    paint (packed). Returns (give, comp)."""
    give = np.empty(COLUMNS)
    comp = np.empty((COLUMNS, PACKED))
    for b in range(COLUMNS):
        tv = tip[b, IV]
        give[b] = tv / max(demand[b], 1e-30) if demand[b] > tv else 1.0
        for c in range(PACKED):
            comp[b, c] = tip[b, c] / (tv if tv > 0 else 1.0)
    return give, comp


@njit(cache=True, error_model="numpy", inline="always")
def _from_columns(taps, d, give, comp, used):
    """The paint the columns above each pixel lay there, asked for volume
    ``d`` per pixel and giving their share ``give`` of it, of paint ``comp``
    (see _give); adds what each gives to ``used``. Returns the paint
    (packed, per pixel), its volume per pixel, and the volume in all."""
    cols, wts, ntap = taps.cols, taps.wts, taps.ntap
    hh, ww = d.shape
    dep = np.zeros((hh, ww, PACKED))
    d_px = np.empty((hh, ww))
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
    return dep, d_px, total


@njit(cache=True, error_model="numpy")
def take_tips(tip, taps, d):
    """Take paint off the tips of the columns above each pixel, ``d`` per
    pixel (each column giving what it holds at most), and return it packed."""
    cols, wts, ntap = taps.cols, taps.wts, taps.ntap
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
    spread_part(cv, pend, br, sy0, sx0, w, pk, cover, stroke_id, by_film, 0.0)


@njit(cache=True, error_model="numpy")
def spread_part(cv, pend, br, sy0, sx0, w, pk, cover, stroke_id, by_film, w_off):
    """:func:`spread`, where the weights ``w`` on the canvas are part of a
    ridge whose weights off the canvas add up to ``w_off``: that share of the
    paint lands off the canvas and is gone. (If there is no weight anywhere,
    the paint stays on the tool.)"""
    if pk[IV] <= 0:
        return
    tot = w.sum() + w_off
    if tot <= 1e-9:
        tip = br.tip
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
    # (a ridge of no more paint than a pixel's dust is no ridge: what there is
    # of it goes down, but it covers nothing)
    covers = cover[y0:y1, x0:x1] if pk[IV] > EMPTY else np.zeros((y1 - y0, x1 - x0))
    deposit(
        cv,
        pend,
        sy0 + y0,
        sx0 + x0,
        dep,
        d,
        covers,
        stroke_id,
        True,
        by_film,
        d,
        False,
    )


@njit(cache=True, error_model="numpy")
def shift(a, sx, sy):
    """Bilinear translation: out[y, x] = a[y - sy, x - sx], zero outside."""
    h, w = a.shape[0], a.shape[1]
    out = np.zeros_like(a)
    dxs, dys, wgts = shift_taps(sx, sy, h, w)
    for y in range(h):
        shift_row(a, y, dxs, dys, wgts, out[y].reshape(-1))
    return out


@njit(cache=True, error_model="numpy")
def shift_taps(sx, sy, h, w):
    """The taps of :func:`shift` by (sx, sy) of an (h, w) array: the
    offsets (dx, dy) and weights of the up to four whole-pixel shifts it
    adds up, in the order it adds them."""
    ix, iy = int(math.floor(sx)), int(math.floor(sy))
    fx, fy = sx - ix, sy - iy
    dxs = np.empty(4, np.int64)
    dys = np.empty(4, np.int64)
    wgts = np.empty(4, np.float32)
    n = 0
    for kx in range(2):
        dx, wx = (ix, 1.0 - fx) if kx == 0 else (ix + 1, fx)
        for ky in range(2):
            dy, wy = (iy, 1.0 - fy) if ky == 0 else (iy + 1, fy)
            if wx * wy < 1e-6 or abs(dx) >= w or abs(dy) >= h:
                continue
            dxs[n], dys[n], wgts[n] = dx, dy, np.float32(wx * wy)
            n += 1
    return dxs[:n], dys[:n], wgts[:n]


@njit(cache=True, error_model="numpy")
def shift_row(a, y, dxs, dys, wgts, out_row):
    """Add row ``y`` of the shift of ``a`` with the taps (dxs, dys, wgts) of
    :func:`shift_taps` to ``out_row`` (its pixels' channels in one run)."""
    h, w, C = a.shape
    for t in range(dxs.shape[0]):
        dx, dy = dxs[t], dys[t]
        if not max(dy, 0) <= y < min(h + dy, h):
            continue
        x_lo, x_hi = max(dx, 0), min(w + dx, w)
        src = a[y - dy].reshape(-1)[(x_lo - dx) * C :]
        dst = out_row[x_lo * C :]
        wgt = wgts[t]
        for i in range((x_hi - x_lo) * C):
            dst[i] += wgt * src[i]


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
            # (a layer holding less than nothing, rounding dust, gives nothing:
            # taking from it would make paint)
            t = t if t >= 0 else 0.0
            uv = uv if uv >= 0 else np.float32(0.0)
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
    H, W = cv.top.shape[0], cv.top.shape[1]
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
        if not scale <= 0 and _any_wet(layer, hy0, hy1, hx0, hx1):
            _shift_share(layer, frac, scale, ry0, rx0, sx, sy)


@njit(cache=True, error_model="numpy", inline="always")
def _any_wet(layer, y0, y1, x0, x1):
    """Whether ``layer`` holds any paint (or rounding dust) in rows y0..y1,
    columns x0..x1."""
    for Y in range(y0, y1):
        for X in range(x0, x1):
            if layer[Y, X, IV] != 0:
                return True
    return False


@njit(cache=True, error_model="numpy", inline="always")
def _shift_share(layer, frac, scale, ry0, rx0, sx, sy):
    """Move the share ``frac`` times ``scale`` (single precision, per pixel)
    of the paint in ``layer`` over the block at (ry0, rx0) the size of
    ``frac`` by (sx, sy) pixels (see drag)."""
    rh, rw = frac.shape
    moved = np.zeros((rh, rw, PACKED), np.float32)
    for y in range(rh):
        for x in range(rw):
            fr = frac[y, x] * scale
            if fr != 0:
                for c in range(PACKED):
                    moved[y, x, c] = fr * layer[ry0 + y, rx0 + x, c]
    n = rw * PACKED
    zero = np.float32(0.0)
    # (what lands in each row, shifted a row at a time: see shift)
    dxs, dys, wgts = shift_taps(sx, sy, rh, rw)
    s_row = np.empty(n, np.float32)
    for y in range(rh):
        s_row[:] = 0.0
        shift_row(moved, y, dxs, dys, wgts, s_row)
        # (a row at a time: its pixels' channels lie in one run; rounding
        # doesn't leave less than nothing where all of it moved on)
        row = layer[ry0 + y, rx0 : rx0 + rw].reshape(-1)
        m_row = moved[y].reshape(-1)
        for i in range(n):
            v = row[i] + (s_row[i] - m_row[i])
            row[i] = max(v, zero)


@njit(cache=True, error_model="numpy")
def stir(cv, sy0, sx0, m):
    """Homogenise the two wet sublayers of the window at (sy0, sx0) by a
    fraction ``m`` (0 = untouched, 1 = both become the same mixture). Each
    sublayer keeps its volume."""
    top, und, top_a, und_a = cv.top, cv.und, cv.top_a, cv.und_a
    hh, ww = m.shape
    for y in range(hh):
        for x in range(ww):
            Y, X = sy0 + y, sx0 + x
            mm = np.float32(m[y, x])
            tv, uv = top[Y, X, IV], und[Y, X, IV]
            # (with no paint in one of the two layers there is nothing to stir:
            # specks of dust are not paint)
            if mm == 0 or not (tv > EMPTY and uv > EMPTY):
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
def fold_up(cv, sy0, sx0, f):
    """The wet paint below folds up over the top layer of the window at (sy0,
    sx0) by a fraction ``f`` (0: not at all, 1: as much of it as the thinner
    of the two layers holds changes places): equal volumes of the two layers
    trade places, each keeping its volume, so the top layer takes on the
    paint below in proportion."""
    top, und, top_a, und_a = cv.top, cv.und, cv.top_a, cv.und_a
    hh, ww = f.shape
    for y in range(hh):
        for x in range(ww):
            Y, X = sy0 + y, sx0 + x
            tv, uv = top[Y, X, IV], und[Y, X, IV]
            # (specks of dust are not paint)
            if f[y, x] == 0 or not (tv > EMPTY and uv > EMPTY):
                continue
            v = f[y, x] * min(tv, uv)
            ft, fu = np.float32(v / tv), np.float32(v / uv)
            if f[y, x] > 0.01:
                und_a[Y, X] = max(und_a[Y, X], top_a[Y, X])
            for c in range(PACKED):
                down, up = ft * top[Y, X, c], fu * und[Y, X, c]
                top[Y, X, c] += up - down
                und[Y, X, c] += down - up


@njit(cache=True, error_model="numpy")
def merge_same(cv, sy0, sx0, own, thr):
    """Where both wet layers of the window are much the same paint (not the
    stroke's own), merge them into the lower one."""
    top, und, top_a, und_a = cv.top, cv.und, cv.top_a, cv.und_a
    hh, ww = own.shape
    for y in range(hh):
        for x in range(ww):
            Y, X = sy0 + y, sx0 + x
            if own[y, x] or not (top[Y, X, IV] > EMPTY and und[Y, X, IV] > EMPTY):
                continue
            if alike_below(top[Y, X], und[Y, X], thr):
                # (the top layer holds paint here, so bury_top always passes
                # its coverage on to the layer below)
                bury_top(top, und, top_a, und_a, Y, X)


# ----------------------------------------------------------------------------
# paint squeezed or pushed out of the footprint: low ridges along its edges


@njit(cache=True, error_model="numpy")
def add_sides(ridges, moved, lat, scale):
    """Paint pushed out to the left and right edges of the footprint."""
    left, right = ridges.left, ridges.right
    hh, ww = lat.shape
    for y in range(hh):
        for x in range(ww):
            side = right if lat[y, x] > 0 else left
            for c in range(PACKED):
                side[c] += scale * moved[y, x, c]


@njit(cache=True, error_model="numpy", inline="always")
def escapes(d):
    """The share of the paint a brush pushes aside ``d`` px in from the
    nearer edge of its footprint that gets out to the side (see
    EDGE_FREE)."""
    if d <= EDGE_FREE:
        return 1.0
    t = (d - EDGE_FREE) / EDGE_SQUEEZE
    return 1.0 / (1.0 + t * t)


@njit(cache=True, error_model="numpy", inline="always")
def _wide(K, st):
    """Whether a brush's footprint is wide enough that
    paint under its middle can't all get out (see EDGE_FREE)."""
    return not K.rigid and st.hold.r * st.hold.lmax > EDGE_FREE


@njit(cache=True, error_model="numpy")
def squeeze_out(br, K, st, col_vol, col_load, hold, ridges, col_u):
    """An overloaded tool pressed down squeezes paint out at its edges,
    unevenly: the lips along a stroke swell and break. Only the paint under a
    knife's blade gets squeezed, not the whole glob riding on it; a wide
    brush squeezes out little of what its columns deep under it hold (see
    EDGE_FREE): that stays on them."""
    tip, belly = br.tip, br.belly
    ring, left, right = ridges.ring, ridges.left, ridges.right
    over = np.empty(COLUMNS)
    anything = False
    for b in range(COLUMNS):
        over[b] = max(col_load[b] - hold, 0.0)
        if K.rigid:
            over[b] = min(over[b], hold + 1.0)
        anything = anything or over[b] > 0
    if not (K.deposits and anything):
        return
    p = st.hold.p
    rate = (
        (SQUEEZE_KNIFE if K.rigid else SQUEEZE_BRUSH)
        * (0.2 + p * p)
        * (0.5 + K.stiffness)
        * st.motion.ds
        / st.hold.Wd
        * (0.2 + 1.6 * st.motion.lip)
    )
    if st.hold.lift_len > 0:
        # (lifting off, the clumps press less and less out at the edges)
        rate *= clip((K.arc_end - st.motion.arc_i) / st.hold.lift_len, 0.0, 1.0)
    # As a brush is set down it squeezes much more out (set_down): into the
    # bead all round where it landed. What it squeezes out as any step does
    # goes to its sides along the way it moves in this step, as at the steps
    # after it: its side ridges start where it was set down and join the
    # bead. (They used to start a step late: hard flats pressed at 0.9 and
    # loaded 45 w / 36, down 5 units of wet blue, had their side crest fall
    # just past the bead to 0.18 of its height further on at 36 px, 0.28 at
    # 70 and 0.55 at 120; now 0.49, 0.68 and 0.88; on bare canvas 0.82, 0.66
    # and 0.55, now 1.05, 1.09 and 0.97. Seeds 1 to 5.) A knife's and a
    # dab's all go into the rim round them, as before: a dab doesn't move,
    # and a knife lays no bead (its rim is the slab's edge all round).
    aside = 1.0
    if st.motion.touchdown:
        set_down = 0.03 * (0.3 + p)
        aside = 0.0 if (K.rigid or K.is_dab) else rate / (rate + set_down)
        rate += set_down
    wide = _wide(K, st)
    # (the footprint reaches half px either side of its centre across the
    # stroke; a column at u across the bristles lies |u| r / |uscale| px from
    # it: on a flat turned to the stroke, closer in than u says)
    half = st.hold.r * st.hold.lmax
    col_px = st.hold.r / abs(st.hold.uscale)
    for b in range(COLUMNS):
        rate_b = rate
        if wide:
            # a column deep under a wide brush squeezes out less: the rest
            # stays on it
            rate_b = rate * escapes(half - abs(col_u[b]) * col_px)
        tk = min(over[b] * (st.hold.Wn * st.hold.Wn / COLUMNS) * rate_b, col_vol[b])
        frac = tk / max(col_vol[b], 1e-30) if col_vol[b] > 0 else 0.0
        w_right = smoothstep(-0.3, 0.3, col_u[b] * st.hold.side)
        for c in range(PACKED):
            out = (tip[b, c] + belly[b, c]) * frac
            tip[b, c] -= tip[b, c] * frac
            belly[b, c] -= belly[b, c] * frac
            side = out * aside
            ring[c] += out - side
            right[c] += side * w_right
            left[c] += side * (1.0 - w_right)


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
    just outside its edges (see lay_ridges_off). Where a ridge would lie off
    the canvas, all of it goes on the part on the canvas."""
    lay_ridges_off(cv, pend, br, K, st, ridges, grid, c_tab, h_tab, hair, False)


@njit(cache=True, error_model="numpy")
def _outside(w, oy0, oy1, ox0, ox1):
    """The sum of ``w`` outside its block [oy0, oy1) x [ox0, ox1)."""
    if oy0 == 0 and ox0 == 0 and oy1 == w.shape[0] and ox1 == w.shape[1]:
        return 0.0
    s = 0.0
    for y in range(w.shape[0]):
        for x in range(w.shape[1]):
            if not (oy0 <= y < oy1 and ox0 <= x < ox1):
                s += w[y, x]
    return s


@njit(cache=True, error_model="numpy")
def lay_ridges_off(cv, pend, br, K, st, ridges, grid, c_tab, h_tab, hair, lose_off):
    """Put the paint pushed out of the footprint on the canvas, as low ridges
    just outside its edges: all around it (when the tool is set down) or along
    its left and right sides. With ``lose_off``, the share of a ridge that
    lies off the canvas is laid there, and is gone (as by a brush going on
    past the edge, see off_canvas).

    Where a loaded brush is set down at the start of a stroke, what it
    squeezes out is a bead around where it landed, as wide as it is high:
    behind it and at its sides (the brush then moves on over the front of
    it), lumpy where the bristle clumps pressed it out (``hair``, see
    hair_field). A dab, pressed straight down and lifted, leaves a rim all
    round.

    A brush's side ridges are as wide as the paint in them makes them, up to
    ``K.room``. As a rule they lie within ``K.margin`` px of the footprint; a
    step whose side ridges reach further works them out over a window that
    much wider across the stroke, and first records what the stroke meets
    there (see _meet, _room_taken)."""
    ring, left, right = ridges.ring, ridges.left, ridges.right
    if not (ring[IV] > 0 or left[IV] > 0 or right[IV] > 0):
        return
    H, W = cv.top.shape[0], cv.top.shape[1]
    widths = (_side_width(K, st, left), _side_width(K, st, right))
    reach = _ridge_reach(K, ridges, widths)
    win, box = _ridge_window(st, _ridge_pads(st, K.margin, reach), H, W, lose_off)
    oy0, ox0, oy1, ox1 = box
    if oy0 >= oy1 or ox0 >= ox1:
        return
    if reach > K.margin:
        _room_taken(cv, K, pend, win, reach)
    geo = _window_geometry(st, win)
    e_along, e_lat, e_r = geo
    # a soft tool's paint fades out before its outline: the squeezed-out paint
    # lies against where the mark fades, not out at the outline
    inset = 0.6 * K.softness * st.hold.r
    weights = (np.empty(e_along.shape), np.empty(e_along.shape))
    w_t, w_g = weights
    if ring[IV] > 0:
        _ring_weights(K, st, ring[IV], geo, (grid, c_tab, h_tab), hair, inset, weights)
        _spread_ridge(cv, pend, br, K, ring, weights, win, box, lose_off)
    if left[IV] > 0 or right[IV] > 0:
        # soft-ended, so that the ridges laid at successive steps run together smoothly
        ramp = 0.5 * st.motion.ds + 0.7
        for k in range(2):
            pk = left if k == 0 else right
            if pk[IV] <= 0:
                continue
            w = widths[k]
            for y in range(e_lat.shape[0]):
                for x in range(e_lat.shape[1]):
                    lat = e_lat[y, x]
                    if (k == 0 and lat < 0) or (k == 1 and lat > 0):
                        w_t[y, x], w_g[y, x] = _side_weight(
                            K,
                            st,
                            lat,
                            e_along[y, x],
                            e_r[y, x],
                            w,
                            ramp,
                            inset,
                            c_tab,
                            hair,
                        )
                    else:
                        w_t[y, x] = 0.0
                        w_g[y, x] = 0.0
            _spread_ridge(cv, pend, br, K, pk, weights, win, box, lose_off)


@njit(cache=True, error_model="numpy", inline="always")
def _ridge_reach(K, ridges, widths):
    """How far past the footprint (px) the side ridges of a step reach: its
    margin, or as far as they (``widths``) and a px."""
    reach = K.margin
    if ridges.left[IV] > 0:
        reach = max(reach, widths[0] + 1.0)
    if ridges.right[IV] > 0:
        reach = max(reach, widths[1] + 1.0)
    return reach


@njit(cache=True, error_model="numpy", inline="always")
def _side_width(K, st, pk):
    """How wide (px) the ridge of the paint ``pk`` a step pushes out to one
    side lies. A brush's is as wide as the paint in it makes it (about as
    high as it is wide: 1 unit of paint ~ 0.6 px), from the paint pushed out
    per px of travel, up to the room the stroke has (K.room); a knife's
    scales with the mark's width, within the margin."""
    if K.rigid:
        return clip(0.24 * st.hold.r * st.hold.lmax, 0.6, K.margin - 1.0)
    return clip(1.2 * math.sqrt(pk[IV] / max(st.motion.ds, 0.5)), 0.8, K.room - 1.0)


@njit(cache=True, error_model="numpy", inline="always")
def _ridge_pads(st, margin, reach):
    """How many px the ridge window of a step reaches past the step's
    window, down and across the canvas: ``margin`` all round, and as far as
    side ridges reaching ``reach`` px past the footprint need across the
    stroke."""
    more = max(reach - margin, 0.0)
    pad_y = margin + more * abs(st.motion.tx)
    pad_x = margin + more * abs(st.motion.ty)
    return int(math.ceil(pad_y)), int(math.ceil(pad_x))


@njit(cache=True, error_model="numpy", inline="always")
def _room_taken(cv, K, pend, win, reach):
    """A brush's side ridges reach ``reach`` px past its footprint, further
    than K.margin, over the window ``win`` (ey0, ex0, ey1, ex1): record what
    the stroke meets there before they land on it (see _meet, whose band
    reaches K.margin), and how far out they reached (PEND_REACH: the stroke
    settles as far, see stroke._Run.settle)."""
    where = pend.where
    if where[PEND_ON] == 0:
        return
    where[PEND_REACH] = max(where[PEND_REACH], int(math.ceil(reach)))
    if not _meets(K):
        return
    ey0, ex0, ey1, ex1 = win
    _meet_box(cv, K, pend, ey0, ey1, ex0, ex1)


@njit(cache=True, error_model="numpy", inline="always")
def _ridge_window(st, pads, H, W, lose_off):
    """The window the ridges of a step are worked out on: the step's window
    with ``pads`` (px down, px across the canvas) around it (the ridges lie
    just outside the footprint), the part of it on the canvas, or all of it
    with ``lose_off``. Returns the window (ey0, ex0, ey1, ex1: rows and
    columns in canvas coordinates; with ``lose_off`` it may reach past the
    canvas), and the block of it on the canvas (oy0, ox0, oy1, ox1, in the
    window; empty: none)."""
    win = st.win
    my, mx = pads
    ey0, ex0 = max(win.y0 - my, 0), max(win.x0 - mx, 0)
    ey1, ex1 = min(win.y0 + win.hh + my, H), min(win.x0 + win.ww + mx, W)
    oy0, ox0, oy1, ox1 = 0, 0, ey1 - ey0, ex1 - ex0
    if lose_off:
        # (the whole window, off the canvas too)
        fy0, fx0 = win.y0 - my, win.x0 - mx
        oy0, ox0, oy1, ox1 = ey0 - fy0, ex0 - fx0, ey1 - fy0, ex1 - fx0
        ey0, ex0 = fy0, fx0
        ey1, ex1 = win.y0 + win.hh + my, win.x0 + win.ww + mx
    return (ey0, ex0, ey1, ex1), (oy0, ox0, oy1, ox1)


@njit(cache=True, error_model="numpy", inline="always")
def _window_geometry(st, win):
    """Where each pixel of the window ``win`` (ey0, ex0, ey1, ex1, on the
    canvas) lies relative to the step: (px along it, px across it, from the
    brush centre; the brush's half-width there, see radius_at)."""
    ey0, ex0, ey1, ex1 = win
    eh, ew = ey1 - ey0, ex1 - ex0
    e_along = np.empty((eh, ew))
    e_lat = np.empty((eh, ew))
    e_r = np.empty((eh, ew))
    for y in range(eh):
        gy = (ey0 + y) + 0.5 - st.motion.cy
        for x in range(ew):
            gx = (ex0 + x) + 0.5 - st.motion.cx
            a = gx * st.motion.tx + gy * st.motion.ty
            e_along[y, x] = a
            e_lat[y, x] = -gx * st.motion.ty + gy * st.motion.tx
            e_r[y, x] = radius_at(a, st.motion.ds, st.hold.r0, st.hold.r1, st.hold.r)
    return e_along, e_lat, e_r


@njit(cache=True, error_model="numpy", inline="always")
def _ring_weights(K, st, vol, geo, tables, hair, inset, weights):
    """How the paint squeezed out all around the footprint (``vol`` of it)
    lies over the window (``geo``, see _window_geometry; ``tables``: the
    footprint's (grid, c_tab, h_tab)): into ``weights`` (w_t, w_g) its share
    at each pixel and how much of the pixel it covers. All around a
    dab or a knife, as wide as the footprint's narrow side allows; where a
    stroke starts, a bead about as high as it is wide around where the brush
    landed, behind it and at its sides, lumpy where the bristle clumps
    pressed it out (``inset``: how far inside the outline the mark fades)."""
    e_along, e_lat, e_r = geo
    grid, c_tab, h_tab = tables
    w_t, w_g = weights
    bead = not (K.is_dab or K.rigid)
    w = _ring_width(K, st, vol)
    set_down, hb = 0.0, 1.0
    if bead:
        # around where the brush landed, at the start of this first step
        set_down = st.motion.ds
        hb = max(0.5 * st.hold.r * (1.0 if K.circle else st.hold.h_tab_max), 1.0)
    eh, ew = e_along.shape
    for y in range(eh):
        for x in range(ew):
            ea = e_along[y, x] + set_down
            e = _outside_by(K, st, e_lat[y, x], ea, e_r[y, x], grid, c_tab, h_tab)
            if bead:
                # a mound across the edge, more of it outside, gone at the front
                t = (e + inset - 0.3 * w) / w
                back = smoothstep(hb, -hb, ea)
                lump = 0.35 + 1.3 * hair_at(hair, st, K, e_along[y, x], e_lat[y, x])
                w_t[y, x] = max(1.0 - t * t, 0.0) * (0.03 + 0.97 * back) * lump
                w_g[y, x] = _geo(e + inset, BEAD_REACH * w)
            else:
                w_t[y, x] = _tent(e + inset, w)
                w_g[y, x] = _geo(e + inset, w)


@njit(cache=True, error_model="numpy", inline="always")
def _ring_width(K, st, vol):
    """How wide (px) the paint ``vol`` squeezed out all around the footprint
    lies (see _ring_weights). All around a dab or a knife, as wide as the
    footprint's narrow side allows; where a brush is set down, a bead about
    as high (1 unit of paint ~ 0.6 px) as it is wide (its outer edge
    reaching BEAD_REACH times as far); both within the margin, the window a
    step's ridges are worked out over. (A wide brush's bead is held to a
    narrower w than its side ridges may have, but its cross-section is not
    theirs: a parabola 2 w wide, as wide at half height as a side ridge
    nearly twice as wide (see _ridge). A 120 px flat's bead is as wide at
    half height as its side ridges, 11.4 px against 11.2 down 5 units of wet
    blue and 12.1 against 12.3 on bare canvas, and as high against those
    beside it as a 36 px flat's (0.89 times, 1.20 on bare canvas, against
    0.83 and 1.67: seeds 1 to 5). Held to K.room, it came out twice as wide
    at half height as they are, a broad pale cap.)"""
    lmax = st.hold.lmax
    if K.is_dab or K.rigid:
        return clip(
            0.24 * st.hold.r * min(lmax, 1.0 if K.circle else st.hold.h_tab_max),
            0.6,
            K.margin - 1.0,
        )
    # the length of the back and the sides of the footprint
    if K.circle:
        rim_len = 2.0 * math.pi * st.hold.r * 0.75
    else:
        rim_len = 2.0 * st.hold.r * (lmax + 2.0 * st.hold.h_tab_max)
    return clip(1.4 * math.sqrt(vol / max(rim_len, 1e-6)), 1.5, K.margin - 1.0)


@njit(cache=True, error_model="numpy", inline="always")
def _outside_by(K, st, lat, ea, r, grid, c_tab, h_tab):
    """How far (px) a pixel ``lat`` px across the step and ``ea`` along it
    lies outside the footprint (``r``: its half-width there); < 0 inside.
    Rounded past the ends of the mark, not squared off (which left lips
    sticking out there)."""
    if K.circle:
        return math.sqrt(lat**2 + ea**2) - r
    g0, g1 = grid[0], grid[grid.shape[0] - 1]
    el = lat / r
    e_c = interp(el, g0, g1, c_tab, c_tab[0], c_tab[c_tab.shape[0] - 1]) * r
    e_h = interp(el, g0, g1, h_tab, 0.0, 0.0) * r
    a = abs(lat) - r * st.hold.lmax
    b = abs(ea - e_c) - e_h
    return math.sqrt(max(a, 0.0) ** 2 + max(b, 0.0) ** 2) + min(max(a, b), 0.0)


@njit(cache=True, error_model="numpy", inline="always")
def _side_weight(K, st, lat, along, r, w, ramp, inset, c_tab, hair):
    """A ridge ``w`` px wide along a side of the step's footprint at a pixel
    ``lat`` px across and ``along`` px along the step (``r``: the footprint's
    half-width there), soft-ended over ``ramp`` px: (its share of the
    ridge's paint there, how much of the pixel it covers)."""
    nt = c_tab.shape[0]
    ce = 0.0 if K.circle else (c_tab[nt - 1] if lat > 0 else c_tab[0]) * r
    rel_e = along - ce
    box = smoothstep(-st.motion.ds - ramp, -st.motion.ds + ramp, rel_e) * smoothstep(
        ramp, -ramp, rel_e
    )
    e_side = abs(lat) - r * st.hold.lmax + inset
    if not K.rigid:
        # steep against the stroke, a longer slope outside; it swells and
        # breaks where the clumps at the edge of the brush pushed it out, and
        # skips under a light touch
        hv = hair_at(hair, st, K, along, lat)
        lump = (0.3 + 1.4 * hv) * smoothstep(
            0.55 - st.hold.p, 0.75 - st.hold.p, hv + 0.3
        )
        # (its paint reaches half a px past where _geo(e, w) stops covering;
        # a wide brush's covers as far, see RIDGE_COVER)
        reach = w + RIDGE_COVER * smoothstep(
            RAMP_FROM, EDGE_FREE, st.hold.r * st.hold.lmax
        )
        return _ridge(e_side, w) * box * lump, _geo(e_side, reach) * box
    return _tent(e_side, w) * box, _geo(e_side, w) * box


@njit(cache=True, error_model="numpy", inline="always")
def _spread_ridge(cv, pend, br, K, pk, weights, win, box, lose_off):
    """Lay the ridge paint ``pk`` over the window ``win`` as its ``weights``
    (w_t, w_g: its share, its cover) say (see spread_part), on the canvas
    the ``box`` (oy0, ox0, oy1, ox1) of it; with ``lose_off``, the share of
    it off the canvas is gone. A knife's ridges bury the paint under them
    whole; a brush's by how well they cover it."""
    w_t, w_g = weights
    ey0, ex0, _, _ = win
    oy0, ox0, oy1, ox1 = box
    by_film = not K.rigid
    spread_part(
        cv,
        pend,
        br,
        ey0 + oy0,
        ex0 + ox0,
        w_t[oy0:oy1, ox0:ox1],
        pk,
        w_g[oy0:oy1, ox0:ox1],
        K.stroke_id,
        by_film,
        _outside(w_t, oy0, oy1, ox0, ox1) if lose_off else 0.0,
    )


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
def pick_up(cv, br, taps, F, part, p, K, f_pick):
    """The tool lifts wet paint. Bristles plough through it: they lift paint
    along the hair lines and leave it between them (brush marks). Thick fresh
    paint shields the older layer below. What is lifted goes into the columns
    that touched it: into their tips, and from deep wet paint partly further up
    (see past_tips). Over the ``part`` of the step's window it passes over
    (see :func:`touched`), pressed ``p``."""
    top, und, ids = cv.top, cv.und, cv.top_id
    tip, belly = br.tip, br.belly
    cols, wts, ntap = taps.cols, taps.wts, taps.ntap
    gain = K.pickup * 1.2 * (0.25 + 1.5 * min(p, 1.0))
    if gain <= 0:
        return
    lifted = np.empty(PACKED)
    lifted_und = np.empty(PACKED)
    for y in range(part.ch):
        for x in range(part.cw):
            Y, X = part.cy0 + y, part.cx0 + x
            q = y * part.cw + x
            plough = F[F_EXPO, part.wy0 + y, part.wx0 + x] * max(
                2.0 - F[F_STREAK, part.wy0 + y, part.wx0 + x], 0.0
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
            for c in range(PACKED):
                lifted[c] = a_top * top[Y, X, c]
                top[Y, X, c] -= lifted[c]
                lifted_und[c] = a_und * und[Y, X, c]
                und[Y, X, c] -= lifted_und[c]
            # (into each column what it lifted from the top layer, then from
            # the one below)
            for t in range(ntap[q]):
                b, wt = cols[q, t], wts[q, t]
                for c in range(PACKED):
                    tip[b, c] += wt * (1.0 - deep) * lifted[c]
                    belly[b, c] += wt * deep * lifted[c]
                    tip[b, c] += wt * (1.0 - deep) * lifted_und[c]
                    belly[b, c] += wt * deep * lifted_und[c]


#: How far down into the hollows of the dried surface the tips of a brush
#: reach, in units of paint thickness, below what the weave's own hollows
#: account for (see _reach): REACH_MIN (always: the weave's hollows and a
#: flat film over them are never out of reach), and REACH_P * p^3 *
#: (REACH_SOFT - stiffness) as it is pressed (little at a light touch, soon
#: a lot; soft hair bends down into them more readily than stiff), and
#: REACH_LOAD * load^2 as it carries more paint (``load`` in width-squares:
#: the paint standing on the tips of a loaded brush bridges down into them;
#: a nearly dry one has next to none). Deeper than that, what the tips
#: offer catches less and less (see _beyond_reach) ...
REACH_MIN, REACH_P, REACH_SOFT, REACH_LOAD = 0.01, 6.0, 1.6, 0.03
#: ... over a depth of REACH_W0 + REACH_WK * reach (the further the tips
#: reach, the more gradually they give out), past which it takes REACH_LIFT
#: more of the tooth or hair (``to``, see _catch) to catch than on the
#: surface around: a point just out of reach still catches where the tooth
#: and the hair favour it, one well out of it nowhere.
REACH_W0, REACH_WK, REACH_LIFT = 0.3, 1.0, 1.5


@njit(cache=True, error_model="numpy", inline="always")
def _catch_at(fill, wet, to, wet_contact, rough_thr):
    """How much of the paint a bristle tip with this column's ``fill`` (the
    bristles coated ``wet``) offers a pixel of tooth and hair ``to`` catches
    there, beyond the threshold ``rough_thr`` (see _catch)."""
    fl = clip(max(fill, 0.9 * wet), 0.0, 1.0)
    thr = 1.5 * (1.0 - fl) ** 0.8 - 0.35 + rough_thr
    return max(smoothstep(thr - 0.08, thr + 0.08, to), wet_contact)


@njit(cache=True, error_model="numpy")
def _catch(fill, wet, to, wet_contact, fill_all, rough_thr):
    """How much of the paint a bristle tip offers a pixel catches there (see
    brush_step_on): (with this column's fill, with the brush's as a whole)."""
    catch = _catch_at(fill, wet, to, wet_contact, rough_thr)
    thr_all = 1.5 * (1.0 - fill_all) ** 0.8 - 0.35 + rough_thr
    return catch, max(smoothstep(thr_all - 0.08, thr_all + 0.08, to), wet_contact)


@njit(cache=True, error_model="numpy", inline="always")
def _reach(p, K, load):
    """How far down into the hollows of the dried surface, beyond the weave's
    own, the tips of a brush pressed ``p`` and carrying ``load``
    (width-squares) reach (see REACH_MIN): REACH_MIN at least, never less (a
    negative reach would read as "everywhere": see _contact_catch_wet_dry).
    Wet paint lying in a hollow makes it no shallower: the tips touch it
    there as anywhere (``wet_contact``, see _catch_at)."""
    press = max(REACH_P * p * p * p * (REACH_SOFT - K.stiffness), 0.0)
    return REACH_MIN + press + REACH_LOAD * load * load


@njit(cache=True, error_model="numpy", inline="always")
def _beyond_reach(hollow, weave, reach):
    """How much more it takes to catch at a point ``hollow`` below the dried
    surface around it (see Beneath), where the weave alone lies ``weave``
    below the weave around it, for tips that reach ``reach`` deeper than the
    weave's own hollows (see REACH_W0): 0 within reach, and wherever the
    surface is no deeper than the weave. (Wet paint lying in the hollow
    does not make it shallower: it catches through ``wet_contact``.)"""
    d = hollow - max(weave, 0.0) - reach
    if not d > 0.0:
        return 0.0
    return REACH_LIFT * d / (REACH_W0 + REACH_WK * reach)


@njit(cache=True, error_model="numpy")
def lay_amount(F, wy, wx, K, lift_gain, tp, glob, q, f_dep, f_pick, hollow, comb):
    """The volume a brush lays at window pixel (wy, wx) in a step (see
    brush_step_on): ``tp`` thick from its tips there (``glob`` of it from a
    glob), caught ``q``, on a surface taking ``f_dep`` and lifting ``f_pick``
    with ``hollow`` to fill, combing a glob into ridges ``comb`` as far as
    usual (less where it ploughs: the old paint left standing between its
    clumps takes their place)."""
    # absorbent / toothy grounds pull paint off the brush
    dd = tp * q * f_dep
    # what the bristles behind lift again and carry on (see own_lift)
    dd *= own_lift(
        lift_gain * f_pick * F[F_PROFILE, wy, wx] * max(2.0 - F[F_STREAK, wy, wx], 0.0),
        # (a dab is one press: nothing passes over it again)
        (1.0 if K.is_dab else 0.4 + 2.0 * F[F_H, wy, wx] / K.spacing),
    )
    # thick paint from a glob is combed by the bristles into deep grooves and
    # ridges
    g = glob / max(tp, 1e-9)
    dd *= max(1.0 + g * 2.0 * K.streaks * comb * (2.0 * F[F_HAIR, wy, wx] - 1.0), 0.15)
    # and it rounds off over the last px or two at the stroke's edges (rather
    # than standing in a cliff one pixel wide, which on a slant steps from pixel
    # to pixel)
    if g > 0 and not K.is_dab:
        dd *= 1.0 - g + g * smoothstep(-0.5, GLOB_EDGE_PX, F[F_EDGE, wy, wx])
    # paint laid on thick fills the hollows of the weave (and of dried paint)
    # under it
    dd *= max(1.0 + smoothstep(0.3, 2.0, tp) * hollow / max(tp, 1e-6), 0.0)
    return dd


@njit(cache=True, error_model="numpy")
def _touches_off(win, F):
    """Whether the tool passes over any pixel of the step's window ``win``
    off the canvas (where it uses up paint as over a canvas going on there)."""
    if win.ch == win.hh and win.cw == win.ww:
        return False
    for y in range(win.hh):
        for x in range(win.ww):
            on = win.wy0 <= y < win.wy0 + win.ch and win.wx0 <= x < win.wx0 + win.cw
            if not on and F[F_EXPO, y, x] != 0:
                return True
    return False


@njit(cache=True, error_model="numpy")
def brush_step(st, wet, dry, tables, kit):
    """One step ``st`` of a brush along its stroke, with the footprint's chord
    tables ``tables`` (grid, c_tab, h_tab): where it touches the canvas, and
    what it does there. The brush comes to it this coated with wet paint
    (``wet``), and run dry or not (``dry``); ``kit`` is what stays the same
    along the stroke (see brush_kit). Returns (how coated the bristles are
    with wet paint, whether the brush has run dry) after it."""
    K, tex, tex2, cv, aux, pend, br, col_u = _unpack_kit(kit)
    grid, c_tab, h_tab = tables
    F, hit = footprint(st, K, tex, tex2, grid, c_tab, h_tab)
    if not hit and not _touches_off(st.win, F):
        return wet, dry
    # most of the work is only where the tool passes over in this step
    part = touched(st, F)
    # (the planes it reads, as arrays of their own: one compiled bristle_taps
    # for these and a knife's)
    taps = bristle_taps(
        np.ascontiguousarray(
            F[F_U, part.wy0 : part.wy0 + part.ch, part.wx0 : part.wx0 + part.cw]
        ),
        st.hold.span,
        st.hold.nsub,
        K.ring,
        st.hold.rho,
    )
    return brush_step_on(
        st, part, wet, dry, K, F, taps, cv, aux, pend, br, tables, col_u, tex2
    )


def brush_kit(K, tex, tex2, cv, aux, pend, br, col_u):
    """What :func:`brush_step` takes of a stroke, the same at every step:
    the stroke's settings ``K`` (a StrokeConsts), its textures, the canvas's
    Layers ``cv`` and Beneath ``aux``, the Pending paint ``pend``, the Load
    on the brush ``br``, and where its columns sit across the mark
    ``col_u``. Made once per stroke.

    They go to the kernel as plain tuples, and it names them again on entry
    (see _unpack_kit): numba's dispatcher types a named tuple by its class
    and its field names at every call, a few times slower than a plain one,
    which would cost several percent of the time a small brush takes."""
    named = (
        (K, StrokeConsts),
        (cv, Layers),
        (aux, Beneath),
        (pend, Pending),
        (br, Load),
    )
    for value, cls in named:
        if not isinstance(value, cls):
            raise TypeError(f"expected a {cls.__name__}, got {type(value).__name__}")
    # (in the order _unpack_kit reads them)
    return (tuple(K), tex, tex2, tuple(cv), tuple(aux), tuple(pend), tuple(br), col_u)


@njit(cache=True, error_model="numpy", inline="always")
def _unpack_kit(kit):
    """The bundles of :func:`brush_kit`, named again: (K, tex, tex2, cv,
    aux, pend, br, col_u)."""
    k_t, tex, tex2, cv_t, aux_t, pend_t, br_t, col_u = kit
    return (
        StrokeConsts(*k_t),
        tex,
        tex2,
        Layers(*cv_t),
        Beneath(*aux_t),
        Pending(*pend_t),
        Load(*br_t),
        col_u,
    )


@njit(cache=True, error_model="numpy")
def brush_step_on(
    st, part, wet, dry, K, F, taps, cv, aux, pend, br, tables, col_u, hair
):
    """One step of a brush, over its footprint ``F`` (see brush_step), on the
    ``part`` of its window it passes over (see :func:`touched`; ``taps`` are
    for that part)."""
    win = st.win
    top, und = cv.top, cv.und
    tip, belly = br.tip, br.belly
    grid, c_tab, h_tab = tables
    p = st.hold.p
    # (true and false as run-time values, not constants: one compiled kernel
    # for these calls and a knife's)
    brush, knife = not K.rigid, K.rigid
    f_pick, f_dep, f_mix, f_drag = surface_factors_on(cv, aux, st, part, K)
    # how far the bristles plough through thick wet paint at each pixel of the
    # step's window on the canvas (see _plough), and whether anywhere
    ploughing = _ploughs(cv, F, st, K, pend)

    # 0) drag: wet paint under the brush is pushed along with it
    if K.drag > 0:
        f = _drag_fraction(cv, br, F, st, part, K, f_drag)
        ds = st.motion.ds
        drag(
            top,
            und,
            f,
            part.cx0,
            part.cy0,
            st.motion.tx * ds,
            st.motion.ty * ds,
            p,
            win.x0,
            win.y0,
            win.hh,
            win.ww,
        )

    # 1) refill the tips, work the paints on the brush, pick up wet paint
    capacity = max(K.smear, 0.05) * 0.8 * st.hold.Wd * st.hold.Wd
    work_load(br, K, st, capacity)
    pick_up(cv, br, taps, F, part, p, K, f_pick)

    # 2) how much paint the brush carries, per column. Paint beyond what the hair
    #    holds is a glob on the outside: it goes on thick, the more so the stiffer the
    #    paint.
    col_vol, load, col_over, col_glob, col_thick, col_fill, hold, thick = _column_loads(
        br, st, K
    )

    # 3) the bristles comb through thick wet paint already there (not what they just
    #    laid), deeper along the hair lines and under pressure: what stands above them
    #    is shifted, partly onto the brush, partly out to the sides (ridges), the rest
    #    pushed ahead (left as a ridge where the brush lifts)
    ridges = Ridges(np.zeros(PACKED), np.zeros(PACKED), np.zeros(PACKED))
    _plough(cv, aux, pend, br, F, st, K, thick, ridges, ploughing)
    squeeze_out(br, K, st, col_vol, col_over, hold, ridges, col_u)
    lay_ridges_off(cv, pend, br, K, st, ridges, grid, c_tab, h_tab, hair, brush)

    # 4) deposit from the tips. Each column's load sets how thick its paint goes on
    #    and when it starts to run dry. Bristles that were just in wet paint are
    #    coated and keep laying a continuous smear for a short distance.
    #    (Only where it passes over: elsewhere its contact is 0.)
    contact, off, o_taps, q, o_q, wet, dry = _contact_catch_wet_dry(
        st, part, wet, dry, K, F, taps, cv, aux, col_fill, load
    )
    if not dry and _tip_volume(tip) > 0 and K.deposits:
        lift_gain = OWN_LIFT * K.pickup * 1.2 * (0.05 + 1.9 * min(p, 1.0))
        laying = (col_thick, col_glob, lift_gain)
        d = _lay_on(aux, F, st.win, part, K, taps, laying, q, f_dep, f_pick, ploughing)
        d_off = _lay_beyond(cv, aux, F, st.win, off, K, o_taps, laying, o_q, ploughing)
        if ploughing[1]:
            # ploughing, it lays its paint more into the grooves it cuts
            _into_grooves(d, d_off, st.win, part, off, K, ploughing)
        if K.is_dab:
            # pressed down, an overloaded brush leaves part of its glob as a mound
            press_glob(d, F, st, part, col_vol, col_over, hold, hair, off, d_off)
        # what the bristles off the canvas lay there, per column
        no = off.shape[0]
        extra = column_demand(o_taps, d_off.reshape((1, no))) if no > 0 else d_off
        if K.is_dab:
            share_by_reach(tip, taps, d, extra, K.ring)
        feed_glob(tip, belly, taps, d, col_glob, extra)
        ch, cw, wy0, wx0 = part.ch, part.cw, part.wy0, part.wx0
        lay_off(
            cv,
            pend,
            br,
            taps,
            part.cy0,
            part.cx0,
            d,
            np.ascontiguousarray(F[F_COVER, wy0 : wy0 + ch, wx0 : wx0 + cw]),
            K.stroke_id,
            knife,
            brush,
            np.ascontiguousarray(F[F_EXPO, wy0 : wy0 + ch, wx0 : wx0 + cw]),
            brush,
            extra,
        )
    overflow(tip, belly, capacity)

    # 5) stir: pressure churns the fresh top layer and the wet paint below into each
    # other; ploughing, the old paint between the clumps folds over the new
    if ploughing[1]:
        m = _fold_fractions(K, p, contact, f_mix, st, part, ploughing, pend)
        stir(cv, part.cy0, part.cx0, m)
    else:
        stir(cv, part.cy0, part.cx0, _stir_fraction(K, p, contact, f_mix))
    return wet, dry


@njit(cache=True, error_model="numpy", inline="always")
def _drag_fraction(cv, br, F, st, part, K, f_drag):
    """The share of each pixel's wet paint the brush drags along in a step
    (over the ``part`` of its window it passes over), on a surface that lets
    it drag ``f_drag``. A loaded brush glides on the paint it has just laid;
    one running dry has nothing to glide on and smears it on ahead (a
    dry-brush tail)."""
    ids = cv.top_id
    tip, belly = br.tip, br.belly
    f = np.empty((part.ch, part.cw))
    coef = K.drag * f_drag * (0.5 + st.hold.p)
    on_brush = 0.0
    for b in range(COLUMNS):
        on_brush += tip[b, IV] + belly[b, IV]
    own_drag = smoothstep(0.8, 0.15, on_brush / (st.hold.Wn * st.hold.Wn))
    for y in range(part.ch):
        for x in range(part.cw):
            wy, wx = part.wy0 + y, part.wx0 + x
            # (the share of a pixel's paint that is moved: a strong drag
            # pressed hard on a slick surface would move more than is
            # there, leaving negative paint that grows step by step)
            f[y, x] = min(
                coef
                * F[F_PROFILE, wy, wx]
                * clip(F[F_OVERLAP, wy, wx] / st.motion.ds, 0.0, 1.0),
                1.0,
            )
            if ids[part.cy0 + y, part.cx0 + x] == K.stroke_id:
                f[y, x] *= own_drag
    return f


@njit(cache=True, error_model="numpy", inline="always")
def _column_loads(br, st, K):
    """How much paint a brush carries (see load_stats), and how thick it lays
    it, per column: (col_vol, load, col_over: its load for the size of the
    bundle, col_glob: how thick the glob beyond what its hair holds lays,
    col_thick: how thick it lays in all, col_fill: how full it is, 0 to 1;
    hold: the load its hair holds; thick: how thick the brush as a whole
    lays). What the hair holds is a matter of the size of the bundle, not of
    how wide the mark is: touching lightly with a pointed brush, or lowering
    it at a tapered end, doesn't put the paint in it beyond what it holds."""
    p = st.hold.p
    col_vol, load, col_load, col_body, body = load_stats(br.tip, br.belly, st.hold.Wn)
    wg = max(st.hold.Wn, 0.6 * K.bundle)
    k_g = (st.hold.Wn / wg) ** 2
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
    return col_vol, load, col_over, col_glob, col_thick, col_fill, hold, thick


@njit(cache=True, error_model="numpy", inline="always")
def _ploughs(cv, F, st, K, pend):
    """How far a brush's bristles plough through the earlier strokes' wet
    paint at each pixel of the step's window on the canvas, 0 to 1 (see
    GAP), whether they do anywhere, and how hard their clumps press there
    where they do (see _lane; 1 elsewhere): as hard as they are pressed, as
    far as the paint the stroke met there can be ploughed (see _meet, which
    this brings up to date at every step, however lightly the brush is
    pressed: a stroke may come back harder over paint it first met
    lightly). How much stiff hair then cuts, and soft hair less, is a matter
    of the cut itself (see _bite). (A point cuts instead, a dab or a drop of
    spatter only presses, and a stroke never pressed above PLOUGH_P0 has
    nothing to record: see StrokeConsts.ploughs. Where they plough nowhere,
    the two arrays are placeholders, not to be read.)"""
    win = st.win
    press = smoothstep(PLOUGH_P0, PLOUGH_P1, st.hold.p)
    nowhere = np.zeros((1, 1))
    if not _meets(K) or pend.where[PEND_ON] == 0:
        return nowhere, False, nowhere
    # (what the stroke meets is recorded however lightly it is pressed: it
    # may come back over it harder, after a light pass has laid its paint
    # there)
    _meet(cv, st, K, pend)
    if not press > 0:
        return nowhere, False, nowhere
    met, where = pend.met, pend.where
    by0, bx0 = where[PEND_Y0], where[PEND_X0]
    y0, y1 = max(win.cy0, by0), min(win.cy0 + win.ch, by0 + where[PEND_H])
    x0, x1 = max(win.cx0, bx0), min(win.cx0 + win.cw, bx0 + where[PEND_W])
    if not _any_above(met, y0 - by0, y1 - by0, x0 - bx0, x1 - bx0):
        return nowhere, False, nowhere
    w = np.zeros((win.ch, win.cw))
    lane = np.ones((win.ch, win.cw))
    H, W = cv.top_id.shape
    for Y in range(y0, y1):
        for X in range(x0, x1):
            v = met[Y - by0, X - bx0]
            if v > 0:
                y, x = Y - win.cy0, X - win.cx0
                w[y, x] = press * v
                # (where the brush passes in the step, and at the canvas's
                # edges, which stand for the pixels past them)
                wy, wx = win.wy0 + y, win.wx0 + x
                edge = Y == 0 or X == 0 or Y == H - 1 or X == W - 1
                if F[F_EXPO, wy, wx] > 0 or edge:
                    lane[y, x] = _lane(K, F[F_HAIR, wy, wx])
    return w, True, lane


@njit(cache=True, error_model="numpy", inline="always")
def _any_above(a, y0, y1, x0, x1):
    """Whether any of ``a[y0:y1, x0:x1]`` is above 0."""
    for y in range(y0, y1):
        for x in range(x0, x1):
            if a[y, x] > 0:
                return True
    return False


@njit(cache=True, error_model="numpy", inline="always")
def _meets(K):
    """Whether a stroke records what it meets (see _meet): one with bristles
    that may plough (a point cuts instead, a dab or a drop of spatter only
    presses, see StrokeConsts.ploughs)."""
    return not (K.dig > 0 or K.is_dab or not K.ploughs)


@njit(cache=True, error_model="numpy", inline="always")
def _meet(cv, st, K, pend):
    """Record how far the wet paint the brush is about to meet can be
    ploughed (see _met), at each pixel of the step's window on the canvas and
    all round it as far as the brush moves in the step or lays the paint it
    pushes out to its sides as a rule (``K.margin``; further out, see
    _room_taken), whichever is further: before what it pushes ahead or aside
    lands there. It records a pixel the first
    time it comes there, however lightly pressed (and the pixel holds none
    of the stroke's own paint): what the stroke met there, not what it left
    of it, stirred into it or pushed onto it. (Not yet recorded: below 0.)"""
    win = st.win
    m = int(math.ceil(max(st.motion.ds, K.margin))) + 1
    # (the window, so much bigger)
    _meet_box(
        cv,
        K,
        pend,
        win.cy0 - m,
        win.cy0 + win.ch + m,
        win.cx0 - m,
        win.cx0 + win.cw + m,
    )


@njit(cache=True, error_model="numpy", inline="always")
def _meet_box(cv, K, pend, y0, y1, x0, x1):
    """:func:`_meet` over the rows ``y0``..``y1`` and columns ``x0``..``x1``
    of the canvas, as far as they lie within the box the stroke can reach."""
    ids = cv.top_id
    met, where = pend.met, pend.where
    by0, bx0 = where[PEND_Y0], where[PEND_X0]
    y0, y1 = max(y0, by0), min(y1, by0 + where[PEND_H])
    x0, x1 = max(x0, bx0), min(x1, bx0 + where[PEND_W])
    for Y in range(y0, y1):
        for X in range(x0, x1):
            if met[Y - by0, X - bx0] < 0 and ids[Y, X] != K.stroke_id:
                met[Y - by0, X - bx0] = _met(cv, Y, X)


@njit(cache=True, error_model="numpy", inline="always")
def _met(cv, Y, X):
    """How far the wet paint at pixel (Y, X) (both layers: the top one from
    an earlier stroke) can be ploughed, 0 to 1: as far as it is thick enough
    to cut a groove into and stiff enough to hold one."""
    v = cv.top[Y, X, IV] + cv.und[Y, X, IV]
    if not v > 0:
        return 0.0
    body = (cv.top[Y, X, IB] + cv.und[Y, X, IB]) / v
    thick = smoothstep(PLOUGH_T0, PLOUGH_T1, v)
    return thick * smoothstep(PLOUGH_B0, PLOUGH_B1, body)


@njit(cache=True, error_model="numpy", inline="always")
def _bite(K):
    """The share of the paint above the bristles' level that a pass of a
    brush's hair clears (see PLOUGH_RATE): how far a brush ploughs, from a
    little for soft hair to almost all for stiff."""
    return 1.0 - math.exp(-PLOUGH_RATE * K.stiffness**3)


@njit(cache=True, error_model="numpy", inline="always")
def _lane(K, hair):
    """How hard the bristles press at a pixel where the brush's hair pattern
    is ``hair``, against how hard they press on average (see LANE_K):
    harder along the lines of their clumps, the more the brush streaks."""
    return math.exp(LANE_K * K.streaks * (1.0 - 2.0 * hair))


@njit(cache=True, error_model="numpy", inline="always")
def _plough(cv, aux, pend, br, F, st, K, thick, ridges, ploughing):
    """The bristles of a brush laying paint ``thick`` comb through the thick
    wet paint already there (over all of the step's window on the canvas):
    what stands above them is shifted, partly onto the brush, partly out to
    the sides (into ``ridges``), the rest pushed ahead; under a wide brush,
    what would go aside from deep under it mostly isn't shifted at all (see
    EDGE_FREE, _held_back). A point (``K.dig``)
    cuts a groove through a share of all of it. Where they plough
    (``ploughing``: how far at each pixel of the window on the canvas, and
    whether anywhere; see _ploughs), they go down through the paint they
    carry and cut into the paint below (see GAP)."""
    win = st.win
    top, und = cv.top, cv.und
    ploughs = ploughing[1]
    p = st.hold.p
    gap = _gap(p, K.stiffness)
    hb = max(1.25 * thick, gap)
    digging = K.dig > 0
    vmax = 0.0
    for y in range(win.ch):
        for x in range(win.cw):
            Y, X = win.cy0 + y, win.cx0 + x
            vmax = max(vmax, top[Y, X, IV] + und[Y, X, IV])
    if not (digging or ploughs or vmax > (1.0 - 0.4 * K.streaks) * hb):
        return
    tv, wet_h, excess, emax = _excess(cv, aux, F, st, K, (hb, gap), ploughing)
    if not emax > 1e-3:
        return
    share_pick = 0.5 * min(K.pickup, 1.0)
    wy0, wx0 = win.wy0, win.wx0
    lat = np.ascontiguousarray(F[F_LAT, wy0 : wy0 + win.ch, wx0 : wx0 + win.cw])
    if _wide(K, st) and not digging:
        # paint deep under a wide brush that can't get out to its sides is
        # not shifted at all: it stays where it is (see _held_back)
        s_side = _side_shares(st, K, ploughing, share_pick)
        picks = _held_back(excess, s_side, lat, st.hold.r * st.hold.lmax, share_pick)
        moved = take(top, und, win.cy0, win.cx0, excess, tv)
        _onto_brush_each(br, moved, wet_h, F, st, K, picks)
        _to_sides_or_ahead(cv, pend, st, K, ridges, moved, lat, s_side)
        return
    moved = take(top, und, win.cy0, win.cx0, excess, tv)
    _onto_brush(br, moved, wet_h, F, st, K, share_pick)
    moved *= 1.0 - share_pick
    _aside_and_ahead(cv, pend, st, K, ridges, moved, lat, ploughing, share_pick)


@njit(cache=True, error_model="numpy", inline="always")
def _aside_and_ahead(cv, pend, st, K, ridges, moved, lat, ploughing, share_pick):
    """What the bristles of a brush no wider than 2 EDGE_FREE (or a point)
    shifted (``moved``, per pixel of the step's window on the canvas, less
    what went onto the brush): all of it into the bead as the brush is set
    down; else a share out to its sides, the rest pushed ahead (see
    _plough)."""
    win = st.win
    plough, ploughs = ploughing[0], ploughing[1]
    p = st.hold.p
    digging = K.dig > 0
    if st.motion.touchdown:
        ring = ridges.ring
        for y in range(win.ch):
            for x in range(win.cw):
                for c in range(PACKED):
                    ring[c] += moved[y, x, c]
        return
    # a point pushes it all aside
    to_side = 1.0 if digging else SIDE_SHARE / (1.0 - share_pick)
    if ploughs:
        # ploughing, the harder and the more it cuts, the more it pushes out
        # to the sides
        more = SIDE_PLOUGH * p * _bite(K) / (1.0 - share_pick)
        _to_sides(ridges, moved, lat, to_side, more * plough)
    else:
        add_sides(ridges, moved, lat, to_side)
        if to_side < 1.0:
            moved *= 1.0 - to_side
    if to_side < 1.0:
        ds = st.motion.ds
        place(
            cv,
            pend,
            moved,
            win.cx0,
            win.cy0,
            st.motion.tx * ds,
            st.motion.ty * ds,
            K.stroke_id,
        )


@njit(cache=True, error_model="numpy", inline="always")
def _side_shares(st, K, ploughing, share_pick):
    """The share of the paint the bristles shift at each pixel of the step's
    window on the canvas (what is left of it once the brush has picked up
    its share ``share_pick``) that a brush pushes out to its sides: all of
    it as the brush is set down (into the bead), else SIDE_SHARE, and more
    where it ploughs (see _plough), all of it at most."""
    win = st.win
    s_side = np.ones((win.ch, win.cw))
    if st.motion.touchdown:
        return s_side
    plough, ploughs = ploughing[0], ploughing[1]
    to_side = SIDE_SHARE / (1.0 - share_pick)
    more = SIDE_PLOUGH * st.hold.p * _bite(K) / (1.0 - share_pick)
    for y in range(win.ch):
        for x in range(win.cw):
            m = more * plough[y, x] if ploughs else 0.0
            s_side[y, x] = min(max(to_side + m, 0.0), 1.0)
    return s_side


@njit(cache=True, error_model="numpy", inline="always")
def _held_back(excess, s_side, lat, half, share_pick):
    """What a wide brush shifts at each pixel of its step's window on the
    canvas, deep under its footprint (``half`` px each side of its centre;
    ``lat``: px across it). Of the paint E its bristles would shift there,
    the brush picks up ``share_pick`` as everywhere, and of the rest the
    share ``s_side`` would go aside (see _side_shares), the rest ahead; but
    of what would go aside only the share f that escapes from that deep
    (see escapes) does, and the rest of it is not shifted at all. So, with
    stay = s_side (1 - f), it shifts only E (1 - (1 - share_pick) stay):
    ``excess`` is cut to that; ``s_side`` becomes the share of what is left
    of it once the brush has picked up its own that goes aside, s_side f /
    ((1 - s_side) + s_side f); and the share of it the brush picks up,
    share_pick / (1 - (1 - share_pick) stay), is returned per pixel. The
    brush gets
    share_pick E, the sides s_side f (1 - share_pick) E and ahead goes
    (1 - s_side) (1 - share_pick) E, as without the limit, and s_side (1 -
    f) (1 - share_pick) E stays where it was."""
    hh, ww = lat.shape
    picks = np.full((hh, ww), share_pick)
    for y in range(hh):
        for x in range(ww):
            f = escapes(half - abs(lat[y, x]))
            if f >= 1.0:
                continue
            s = s_side[y, x]
            stay = s * (1.0 - f)
            kept = 1.0 - (1.0 - share_pick) * stay
            excess[y, x] *= kept
            picks[y, x] = share_pick / kept
            # (written so that it is exactly 1 for s = 1: 1 - (1 - f) is not
            # always f in floating point)
            rest = (1.0 - s) + s * f
            s_side[y, x] = s * f / rest if rest > 0.0 else 1.0
    return picks


@njit(cache=True, error_model="numpy", inline="always")
def _onto_brush_each(br, moved, wet_h, F, st, K, picks):
    """:func:`_onto_brush` with the share the brush picks up given per pixel
    (``picks``, see _held_back); ``moved`` keeps the rest."""
    win = st.win
    tip, belly = br.tip, br.belly
    ch, cw, wy0, wx0 = win.ch, win.cw, win.wy0, win.wx0
    u = np.ascontiguousarray(F[F_U, wy0 : wy0 + ch, wx0 : wx0 + cw])
    taps = bristle_taps(u, st.hold.span, st.hold.nsub, K.ring, st.hold.rho)
    cols, wts, ntap = taps.cols, taps.wts, taps.ntap
    for y in range(ch):
        for x in range(cw):
            q = y * cw + x
            deep = past_tips(wet_h[y, x], st.hold.p, K)
            for t in range(ntap[q]):
                b, wt = cols[q, t], wts[q, t] * picks[y, x]
                for c in range(PACKED):
                    m = wt * moved[y, x, c]
                    tip[b, c] += (1.0 - deep) * m
                    belly[b, c] += deep * m
            for c in range(PACKED):
                moved[y, x, c] *= 1.0 - picks[y, x]


@njit(cache=True, error_model="numpy", inline="always")
def _to_sides_or_ahead(cv, pend, st, K, ridges, moved, lat, s_side):
    """The paint the bristles of a wide brush shifted (``moved``, per pixel
    of the step's window on the canvas, less what went onto the brush): all
    of it all round into the bead as the brush is set down; else the share
    ``s_side`` of it out to the sides, the rest pushed ahead (see
    _plough)."""
    win = st.win
    left, right, ring = ridges.left, ridges.right, ridges.ring
    if st.motion.touchdown:
        for y in range(win.ch):
            for x in range(win.cw):
                for c in range(PACKED):
                    ring[c] += moved[y, x, c]
        return
    ahead = False
    for y in range(win.ch):
        for x in range(win.cw):
            sh = s_side[y, x]
            side = right if lat[y, x] > 0 else left
            for c in range(PACKED):
                m = moved[y, x, c]
                side[c] += sh * m
                moved[y, x, c] = (1.0 - sh) * m
            ahead = ahead or sh < 1.0
    if ahead:
        ds = st.motion.ds
        place(
            cv,
            pend,
            moved,
            win.cx0,
            win.cy0,
            st.motion.tx * ds,
            st.motion.ty * ds,
            K.stroke_id,
        )


@njit(cache=True, error_model="numpy", inline="always")
def _gap(p, stiffness):
    """How high (units of paint thickness) the tips of a brush's bristles,
    of ``stiffness``, ride above the solid surface pressed ``p``, whatever
    paint it carries (see GAP): stiff hair pressed hard comes close to it."""
    return GAP * (1.15 - p) * (1.3 - 0.8 * stiffness)


@njit(cache=True, error_model="numpy", inline="always")
def _to_sides(ridges, moved, lat, to_side, more):
    """Push the share ``to_side`` of the paint the bristles shifted
    (``moved``, packed per pixel of the step's window on the canvas) out to
    the left and right edges of the footprint (see add_sides), and the share
    ``more`` more of it at each pixel (all of it at most); keep the rest of
    it in ``moved``."""
    left, right = ridges.left, ridges.right
    hh, ww = lat.shape
    for y in range(hh):
        for x in range(ww):
            side = right if lat[y, x] > 0 else left
            s = min(max(to_side + more[y, x], 0.0), 1.0)
            for c in range(PACKED):
                side[c] += s * moved[y, x, c]
                moved[y, x, c] *= 1.0 - s


@njit(cache=True, error_model="numpy", inline="always")
def _excess(cv, aux, F, st, K, levels, ploughing):
    """How much of the earlier strokes' wet paint at each pixel of the step's
    window on the canvas stands above the bristles and is shifted as they
    pass (``levels``: (how high they ride on the paint they carry, the gap
    they leave pressed through it), see _riding and _ploughing, as far as
    they plough there, ``ploughing``: see _ploughs); or, digging, the share
    of it a point cuts through. Returns (the volume of the top layer there,
    of all the wet paint, the excess, the most of it)."""
    win = st.win
    top, und, ids = cv.top, cv.und, cv.top_id
    floor = aux.floor
    hb, gap = levels
    plough, ploughs, lane = ploughing
    ch, cw, cy0, cx0, wy0, wx0 = (
        win.ch,
        win.cw,
        win.cy0,
        win.cx0,
        win.wy0,
        win.wx0,
    )
    tv = np.empty((ch, cw))
    wet_h = np.empty((ch, cw))
    excess = np.empty((ch, cw))
    emax = 0.0
    depth = K.dig * smoothstep(0.05, 0.6, st.hold.p)
    for y in range(ch):
        for x in range(cw):
            Y, X = cy0 + y, cx0 + x
            # (heights summed in single precision, like the paint itself)
            t = top[Y, X, IV] if ids[Y, X] != K.stroke_id else np.float32(0.0)
            tv[y, x] = t
            wet_h[y, x] = t + und[Y, X, IV]
            e = F[F_EXPO, wy0 + y, wx0 + x]
            hair = F[F_HAIR, wy0 + y, wx0 + x]
            # a point cut into the paint: a groove through a share of all of it
            if K.dig > 0:
                ex = (t + und[Y, X, IV]) * depth * (1.0 - math.exp(-6.0 * e))
            else:
                ex = _riding(K, st, floor[Y, X], t, und[Y, X, IV], hb, hair, e)
            if ploughs and plough[y, x] > 0:
                # (riding lower than the gap, it shaves what stands above that)
                cut = max(_ploughing(K, wet_h[y, x], gap, lane[y, x], e), ex)
                ex += plough[y, x] * (cut - ex)
            excess[y, x] = ex
            emax = max(emax, ex)
    return tv, wet_h, excess, emax


@njit(cache=True, error_model="numpy", inline="always")
def _riding(K, st, floor, t, uv, hb, hair, e):
    """How much of the wet paint (``t`` in the top layer from earlier
    strokes, ``uv`` below it, on a solid surface ``floor`` high) at a pixel
    the bristles shift as they pass in a step (exposure ``e``) riding ``hb``
    above the surface, deeper along the hair lines (``hair``)."""
    level = floor + hb * (1.0 + 0.8 * K.streaks * (hair - 0.5))
    ex = max((floor + t + uv) - level, 0.0)
    # shifted gradually as it passes
    ex *= 1.0 - math.exp(-0.8 * K.stiffness**2 * e)
    # a light tap only touches with the bristle tips: it doesn't plough
    if K.is_dab:
        ex *= smoothstep(0.4, 0.8, st.hold.p)
    return ex


@njit(cache=True, error_model="numpy", inline="always")
def _ploughing(K, wet, gap, lane, e):
    """How much of the wet paint (``wet`` thick) at a pixel the bristles,
    pressed down through it to ``gap`` above the solid surface, cut out of
    it as they pass in a step (exposure ``e``): of what stands above the
    gap, as much as their stiffness lets them clear (see PLOUGH_RATE), more
    along the lines of their clumps (they press ``lane`` times as hard as on
    average there: see _lane and LANE_K)."""
    rate = PLOUGH_RATE * K.stiffness**3 * math.sqrt(lane)
    return max(wet - gap, 0.0) * (1.0 - math.exp(-rate * e))


@njit(cache=True, error_model="numpy", inline="always")
def _onto_brush(br, moved, wet_h, F, st, K, share_pick):
    """The share ``share_pick`` of the paint the bristles shifted (``moved``,
    packed per pixel of the step's window on the canvas) goes onto the
    columns above it: into their tips, and from deep wet paint (``wet_h``)
    partly further up (see past_tips)."""
    win = st.win
    tip, belly = br.tip, br.belly
    ch, cw, wy0, wx0 = win.ch, win.cw, win.wy0, win.wx0
    u = np.ascontiguousarray(F[F_U, wy0 : wy0 + ch, wx0 : wx0 + cw])
    taps = bristle_taps(u, st.hold.span, st.hold.nsub, K.ring, st.hold.rho)
    cols, wts, ntap = taps.cols, taps.wts, taps.ntap
    for y in range(ch):
        for x in range(cw):
            q = y * cw + x
            deep = past_tips(wet_h[y, x], st.hold.p, K)
            for t in range(ntap[q]):
                b, wt = cols[q, t], wts[q, t] * share_pick
                for c in range(PACKED):
                    m = wt * moved[y, x, c]
                    tip[b, c] += (1.0 - deep) * m
                    belly[b, c] += deep * m


@njit(cache=True, error_model="numpy", inline="always")
def _contact_catch_wet_dry(st, part, wet, dry, K, F, taps, cv, aux, col_fill, load):
    """How the brush touches the paint in a step (it lays nothing): its
    contact at each pixel it passes over (see _contact); the pixels off the
    canvas it passes over and the bristle taps there; how much of what its
    tips offer catches at each pixel, on and off the canvas (see _catch),
    given how full each column is (``col_fill``) and the brush's ``load``;
    how coated the bristles are with wet paint after it (``wet`` before);
    and whether the brush has run dry (see _runs_dry; ``dry`` before).
    Returns (contact, off, o_taps, q, o_q, wet, dry)."""
    top = cv.top
    contact, wet_contact, sums = _contact(cv, F, part)
    # Where the brush sticks out past the edge of the canvas, it goes on as if
    # the canvas went on as it is at its edge (see off_canvas): those bristles
    # use up the paint they would lay there, and count in how the brush as a
    # whole is doing, as they would on a larger canvas. (Taking that paint
    # from all the bristles alike emptied the ones on the canvas early.)
    off = off_canvas(st, F, top.shape[0], top.shape[1])
    o_contact, o_wet, o_u, sums = _contact_off(cv, F, off, sums)
    sum_contact, sum_wet = sums
    o_taps = bristle_taps(o_u, st.hold.span, st.hold.nsub, K.ring, st.hold.rho)
    norm = max(sum_contact, 1e-12)
    wet = max(sum_wet / norm, wet * math.exp(-st.motion.ds / st.hold.Wd))
    fill = at_pixels(taps, col_fill, part.ch, part.cw)
    o_fill = at_pixels(o_taps, col_fill, 1, off.shape[0])
    fill_all = max(clip((load - RESIDUAL) / 0.6, 0.0, 1.0), 0.9 * wet)
    # dry brush: only the canvas tooth (and hair lines) catch paint; a toothy surface
    # breaks up a lightly pressed stroke even when the brush is loaded (scumbling)
    rough_t = K.surface_tooth - 0.5
    rough_thr = (3.2 if rough_t > 0 else 0.6) * rough_t * (1.0 - st.hold.p) ** 1.2
    # light pressure and little paint: the tips ride over the hollows of the
    # surface deeper than they reach (see _reach)
    reach = _reach(st.hold.p, K, load)
    # (-1: it reaches everywhere, nothing to look at; _reach is never
    # negative itself)
    reach = reach if K.beyond_weave > reach else -1.0
    brush_state = (wet, fill_all, rough_thr, reach)
    q, sums = _catch_on(aux, F, part, K, fill, wet_contact, contact, brush_state)
    o_q, sums = _catch_off(aux, F, off, K, o_fill, o_wet, o_contact, brush_state, sums)
    sum_q, sum_all = sums
    # when only the tooth catches, paint concentrates on those peaks
    q_norm = max(sum_q / norm, 0.2)
    q /= q_norm
    o_q /= q_norm
    dry = _runs_dry(cv, part, K, off, contact, o_contact, norm, sum_all, dry)
    return contact, off, o_taps, q, o_q, wet, dry


@njit(cache=True, error_model="numpy", inline="always")
def _contact(cv, F, part):
    """How the brush touches each pixel of the ``part`` of its window it
    passes over: its contact (exposure along the streaks), and how wet the
    paint there is, 0 to 1; and (the sum of the contact, of contact times
    wetness)."""
    top, und = cv.top, cv.und
    ch, cw, cy0, cx0, wy0, wx0 = (
        part.ch,
        part.cw,
        part.cy0,
        part.cx0,
        part.wy0,
        part.wx0,
    )
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
    return contact, wet_contact, (sum_contact, sum_wet)


@njit(cache=True, error_model="numpy", inline="always")
def _contact_off(cv, F, off, sums):
    """:func:`_contact` at the pixels ``off`` the canvas (see off_canvas),
    added to its ``sums``; and the brush's position across its bristles there
    (as a row of pixels). (The sums carry on from the canvas's, in the same
    order of additions: not to be summed apart and added.)"""
    top, und = cv.top, cv.und
    sum_contact, sum_wet = sums
    no = off.shape[0]
    o_contact = np.empty(no)
    o_wet = np.empty(no)
    o_u = np.empty((1, no))
    for i in range(no):
        wy, wx, Y, X = off[i, 0], off[i, 1], off[i, 2], off[i, 3]
        ct = F[F_EXPO, wy, wx] * F[F_STREAK, wy, wx]
        wc = smoothstep(0.05, 0.35, top[Y, X, IV] + und[Y, X, IV])
        o_contact[i] = ct
        o_wet[i] = wc
        o_u[0, i] = F[F_U, wy, wx]
        sum_contact += ct
        sum_wet += ct * wc
    return o_contact, o_wet, o_u, (sum_contact, sum_wet)


@njit(cache=True, error_model="numpy", inline="always")
def _catch_on(aux, F, part, K, fill, wet_contact, contact, brush_state):
    """How much of the paint its tips offer each pixel the brush passes over
    catches there (see _catch; ``brush_state``: how coated the bristles are,
    how full the brush is as a whole, the roughness threshold), times its
    contact; and (the sum of that, of the contact times what would catch
    with the brush's fill as a whole). Dry brush: only the canvas tooth (and
    hair lines) catch paint. Where the surface lies deeper than the tips
    reach (``reach``, -1 if they reach everywhere), less catches (see
    _beyond_reach); not counted in the sums, so what the tips keep there
    stays on them."""
    tooth_map, hollows, weave = aux.tooth, aux.hollows, aux.weave_hollows
    wet, fill_all, rough_thr, reach = brush_state
    ch, cw, cy0, cx0, wy0, wx0 = (
        part.ch,
        part.cw,
        part.cy0,
        part.cx0,
        part.wy0,
        part.wx0,
    )
    q = np.empty((ch, cw))
    sum_q, sum_all = 0.0, 0.0
    for y in range(ch):
        for x in range(cw):
            to = (
                K.tooth_w * tooth_map[cy0 + y, cx0 + x]
                + (1.0 - K.tooth_w) * F[F_HAIR, wy0 + y, wx0 + x]
            )
            q[y, x], a = _catch(
                fill[y, x], wet, to, wet_contact[y, x], fill_all, rough_thr
            )
            q[y, x] *= contact[y, x]
            sum_q += q[y, x]
            # (once the brush as a whole is effectively dry, the rest stays in the
            # bristles)
            sum_all += contact[y, x] * a
            # where the tips do not reach, less catches: after the sums, so
            # that what they keep stays on them (see _reach)
            if reach >= 0.0:
                at = (hollows[cy0 + y, cx0 + x], weave[cy0 + y, cx0 + x], reach)
                pt = (fill[y, x], wet, to, wet_contact[y, x], rough_thr)
                q[y, x] = _within_reach(q[y, x], contact[y, x], pt, at)
    return q, (sum_q, sum_all)


@njit(cache=True, error_model="numpy", inline="always")
def _catch_off(aux, F, off, K, o_fill, o_wet, o_contact, brush_state, sums):
    """:func:`_catch_on` at the pixels ``off`` the canvas, added to its
    ``sums``. (The sums carry on from the canvas's, in the same order of
    additions: not to be summed apart and added.)"""
    tooth_map, hollows, weave = aux.tooth, aux.hollows, aux.weave_hollows
    wet, fill_all, rough_thr, reach = brush_state
    sum_q, sum_all = sums
    no = off.shape[0]
    o_q = np.empty(no)
    for i in range(no):
        wy, wx, Y, X = off[i, 0], off[i, 1], off[i, 2], off[i, 3]
        to = K.tooth_w * tooth_map[Y, X] + (1.0 - K.tooth_w) * F[F_HAIR, wy, wx]
        o_q[i], a = _catch(o_fill[0, i], wet, to, o_wet[i], fill_all, rough_thr)
        o_q[i] *= o_contact[i]
        sum_q += o_q[i]
        sum_all += o_contact[i] * a
        if reach >= 0.0:
            at = (hollows[Y, X], weave[Y, X], reach)
            pt = (o_fill[0, i], wet, to, o_wet[i], rough_thr)
            o_q[i] = _within_reach(o_q[i], o_contact[i], pt, at)
    return o_q, (sum_q, sum_all)


@njit(cache=True, error_model="numpy", inline="always")
def _within_reach(q, contact, pt, at):
    """What catches at a point ``at`` = (hollow, weave, reach: see
    _beyond_reach) where ``q`` would catch within reach, of the tips there
    ``pt`` = (fill, wet, to, wet_contact, rough_thr: see _catch_at), in
    ``contact``: ``q`` within reach, less and less beyond."""
    lift = _beyond_reach(at[0], at[1], at[2])
    if lift > 0.0:
        fill, wet, to, wet_contact, rough_thr = pt
        return contact * _catch_at(fill, wet, to, wet_contact, rough_thr + lift)
    return q


@njit(cache=True, error_model="numpy", inline="always")
def _runs_dry(cv, part, K, off, contact, o_contact, norm, sum_all, was_dry):
    """Whether the brush has run dry after this step: once the paint it
    offers effectively catches nowhere (``sum_all`` of its whole ``norm``
    contact). A brush that has run dry only comes back to life by dragging
    through wet paint from earlier strokes."""
    if not was_dry:
        return sum_all / norm < 0.04
    s = 0.0
    for y in range(part.ch):
        for x in range(part.cw):
            other = _earlier_wet(cv, part.cy0 + y, part.cx0 + x, K.stroke_id)
            s += contact[y, x] * smoothstep(0.05, 0.35, other)
    for i in range(off.shape[0]):
        other = _earlier_wet(cv, off[i, 2], off[i, 3], K.stroke_id)
        s += o_contact[i] * smoothstep(0.05, 0.35, other)
    return s / norm < 0.12


@njit(cache=True, error_model="numpy", inline="always")
def _earlier_wet(cv, Y, X, stroke_id):
    """The volume of wet paint at pixel (Y, X) from strokes before
    ``stroke_id``."""
    top, und, ids = cv.top, cv.und, cv.top_id
    return und[Y, X, IV] + (top[Y, X, IV] if ids[Y, X] != stroke_id else 0.0)


@njit(cache=True, error_model="numpy", inline="always")
def _tip_volume(tip):
    """The volume of paint in the tips of all the columns."""
    tip_v = 0.0
    for b in range(COLUMNS):
        tip_v += tip[b, IV]
    return tip_v


@njit(cache=True, error_model="numpy", inline="always")
def _lay_on(aux, F, win, part, K, taps, laying, q, f_dep, f_pick, ploughing):
    """The volume the brush lays at each pixel of the ``part`` of the step's
    window ``win`` it passes over (see lay_amount): the columns above it
    laying ``laying`` = (col_thick, col_glob: that much of it from a glob,
    lift_gain: its own bristles lifting with that, see own_lift), caught
    ``q``, on a surface taking ``f_dep`` and lifting ``f_pick``, ploughing
    ``ploughing`` (see _ploughs: a glob is combed less where it does)."""
    col_thick, col_glob, lift_gain = laying
    plough = ploughing[0]
    bite = _bite(K)
    hollows = aux.hollows
    ch, cw, cy0, cx0, wy0, wx0 = (
        part.ch,
        part.cw,
        part.cy0,
        part.cx0,
        part.wy0,
        part.wx0,
    )
    thick_px = at_pixels(taps, col_thick, ch, cw)
    glob_px = at_pixels(taps, col_glob, ch, cw)
    d = np.empty((ch, cw))
    for y in range(ch):
        for x in range(cw):
            comb = 1.0
            if ploughing[1]:
                comb -= bite * plough[cy0 - win.cy0 + y, cx0 - win.cx0 + x]
            d[y, x] = lay_amount(
                F,
                wy0 + y,
                wx0 + x,
                K,
                lift_gain,
                thick_px[y, x],
                glob_px[y, x],
                q[y, x],
                f_dep[y, x],
                f_pick[y, x],
                hollows[cy0 + y, cx0 + x],
                comb,
            )
    return d


@njit(cache=True, error_model="numpy", inline="always")
def _lay_beyond(cv, aux, F, win, off, K, o_taps, laying, o_q, ploughing):
    """:func:`_lay_on` at each of the pixels ``off`` the canvas the brush
    passes over, as on the canvas at its edge."""
    col_thick, col_glob, lift_gain = laying
    plough = ploughing[0]
    bite = _bite(K)
    hollows = aux.hollows
    no = off.shape[0]
    d_off = np.zeros(no)
    if no == 0:
        return d_off
    o_thick = at_pixels(o_taps, col_thick, 1, no)
    o_glob = at_pixels(o_taps, col_glob, 1, no)
    for i in range(no):
        wy, wx, Y, X = off[i, 0], off[i, 1], off[i, 2], off[i, 3]
        fp, fd = surface_at(cv, aux, Y, X, K)
        d_off[i] = lay_amount(
            F,
            wy,
            wx,
            K,
            lift_gain,
            o_thick[0, i],
            o_glob[0, i],
            o_q[i],
            fd,
            fp,
            hollows[Y, X],
            1.0 - bite * _at(plough, Y - win.cy0, X - win.cx0, 0.0),
        )
    return d_off


@njit(cache=True, error_model="numpy", inline="always")
def _into_grooves(d, d_off, win, part, off, K, ploughing):
    """A brush ploughing (``ploughing``: how far at each pixel of the step's
    window ``win`` on the canvas, and how hard its clumps press there, see
    _ploughs) lays its paint more where they press harder, as far as it
    ploughs and its hair cuts
    (see _bite), and as much in all: it rescales the volume it lays at each
    pixel of the ``part`` of the window it passes over (``d``) and at the
    pixels ``off`` the canvas (``d_off``, as at the canvas's edge, ``off``
    as off_canvas gives them). Where it doesn't plough, it lays as it
    would."""
    w, lane, o_w, o_lane = _lanes_under(win, part, off, K, ploughing)
    laid, pressed = 0.0, 0.0
    for y in range(part.ch):
        for x in range(part.cw):
            laid += d[y, x] * w[y, x]
            pressed += d[y, x] * w[y, x] * lane[y, x]
    for i in range(off.shape[0]):
        laid += d_off[i] * o_w[i]
        pressed += d_off[i] * o_w[i] * o_lane[i]
    if not (laid > 0 and pressed > 0):
        return
    # (in proportion to how hard they press, against the mean over what it
    # lays where it ploughs: what it lays there adds up to the same)
    mean = pressed / laid
    for y in range(part.ch):
        for x in range(part.cw):
            d[y, x] *= 1.0 + w[y, x] * (lane[y, x] / mean - 1.0)
    for i in range(off.shape[0]):
        d_off[i] *= 1.0 + o_w[i] * (o_lane[i] / mean - 1.0)


@njit(cache=True, error_model="numpy", inline="always")
def _at(a, y, x, outside):
    """``a[y, x]``, or ``outside`` past its edges (a pixel off the canvas
    is taken as at the canvas's edge, which lies outside the step's window
    on the canvas when none of the window is on the canvas)."""
    if 0 <= y < a.shape[0] and 0 <= x < a.shape[1]:
        return a[y, x]
    return outside


@njit(cache=True, error_model="numpy", inline="always")
def _lanes_under(win, part, off, K, ploughing):
    """How far a brush ploughing (``ploughing``, over the step's window
    ``win`` on the canvas, see _ploughs) shifts the paint it lays towards
    where its clumps press (see _into_grooves), and how hard they press, at
    each pixel of the ``part`` of the window it passes over and at the
    pixels ``off`` the canvas (as at the canvas's edge)."""
    plough, _, lanes = ploughing
    bite = _bite(K)
    oy, ox = part.cy0 - win.cy0, part.cx0 - win.cx0
    w = np.empty((part.ch, part.cw))
    lane = np.empty((part.ch, part.cw))
    for y in range(part.ch):
        for x in range(part.cw):
            w[y, x] = bite * plough[oy + y, ox + x]
            lane[y, x] = lanes[oy + y, ox + x]
    no = off.shape[0]
    o_w = np.empty(no)
    o_lane = np.empty(no)
    for i in range(no):
        Y, X = off[i, 2] - win.cy0, off[i, 3] - win.cx0
        o_w[i] = bite * _at(plough, Y, X, 0.0)
        o_lane[i] = _at(lanes, Y, X, 1.0)
    return w, lane, o_w, o_lane


@njit(cache=True, error_model="numpy", inline="always")
def _fold_fractions(K, p, contact, f_mix, st, part, ploughing, pend):
    """Where a brush ploughs (``ploughing``: how far at each pixel of the
    step's window on the canvas, and how hard its clumps press there, see
    _ploughs): how far it stirs the two wet layers into each other at each
    pixel of the ``part`` of the window it passes over (see _stir_fraction;
    less as far as it ploughs and its hair cuts, see _bite: in its grooves
    the new paint lies on what is left of the old). And it adds to how far
    the old paint below folds up over the new there once the stroke has
    passed (``pend.folds``, see fold_pending; between its clumps, where they
    press less, the more: see FOLD)."""
    plough, _, lanes = ploughing
    folds, where = pend.folds, pend.where
    bite = _bite(K)
    oy, ox = part.cy0 - st.win.cy0, part.cx0 - st.win.cx0
    ch, cw = contact.shape
    m = np.empty((ch, cw))
    for y in range(ch):
        for x in range(cw):
            w = bite * plough[oy + y, ox + x]
            k = p * p * contact[y, x] * f_mix[y, x]
            m[y, x] = 1.0 - math.exp(
                -K.mixing * 3.0 * p * p * contact[y, x] * f_mix[y, x] * (1.0 - w)
            )
            Y, X = part.cy0 + y, part.cx0 + x
            if w > 0 and _pending_at(where, Y, X):
                py, px = Y - where[PEND_Y0], X - where[PEND_X0]
                fold = FOLD * K.mixing * k * w / lanes[oy + y, ox + x]
                folds[py, px] += np.float32(fold)
                where[PEND_FOLD_ROW0] = min(where[PEND_FOLD_ROW0], py)
                where[PEND_FOLD_ROW1] = max(where[PEND_FOLD_ROW1], py + 1)
                where[PEND_FOLD_COL0] = min(where[PEND_FOLD_COL0], px)
                where[PEND_FOLD_COL1] = max(where[PEND_FOLD_COL1], px + 1)
    return m


@njit(cache=True, error_model="numpy")
def fold_pending(cv, pend):
    """The stroke has passed: where it ploughed, the old paint below folds up
    over the new as far as it gathered to (``pend.folds``: f, folding
    1 - e^-f of it, see fold_up). Done once, on the paint as the stroke left
    it, so that it doesn't matter how finely the stroke was cut."""
    where = pend.where
    r0, r1, c0, c1 = (
        where[PEND_FOLD_ROW0],
        where[PEND_FOLD_ROW1],
        where[PEND_FOLD_COL0],
        where[PEND_FOLD_COL1],
    )
    if where[PEND_ON] == 0 or r0 >= r1 or c0 >= c1:
        return
    # (over the rows and columns where any landed)
    folds = pend.folds
    f = np.zeros((r1 - r0, c1 - c0))
    for py in range(r0, r1):
        for px in range(c0, c1):
            if folds[py, px] > 0:
                f[py - r0, px - c0] = 1.0 - math.exp(-folds[py, px])
    fold_up(cv, where[PEND_Y0] + r0, where[PEND_X0] + c0, f)


@njit(cache=True, error_model="numpy", inline="always")
def _stir_fraction(K, p, contact, f_mix):
    """How far pressure ``p`` churns the fresh top layer and the wet paint
    below into each other at each pixel (see stir), with the brush's
    ``contact`` there on a surface that lets it mix ``f_mix``."""
    ch, cw = contact.shape
    m = np.empty((ch, cw))
    for y in range(ch):
        for x in range(cw):
            m[y, x] = 1.0 - math.exp(
                -K.mixing * 3.0 * p * p * contact[y, x] * f_mix[y, x]
            )
    return m


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
    V, crit, vmax = _heights(top, und, ry0, ry1, rx0, rx1)
    if vmax < 0.3:
        return
    by0, by1, bx0, bx1 = _steep_box(V, crit, dry_h, ry0, rx0)
    if by1 < 0:
        return
    H, W = V.shape
    pad = iters + 2
    y0, y1 = max(by0 - pad, 0), min(by1 + pad + 1, H)
    x0, x1 = max(bx0 - pad, 0), min(bx1 + pad + 1, W)
    _relax(top, und, dry_h, V, crit, ry0, rx0, y0, y1, x0, x1, iters)


@njit(cache=True, error_model="numpy", inline="always")
def _heights(top, und, ry0, ry1, rx0, rx1):
    """The volume of wet paint at each pixel of the region and the steepest
    slope it holds (by its body), and the most volume. (Heights and slopes in
    single precision, like the paint: whether paint is steeper than it holds
    is decided on the same numbers as the paint's own.)"""
    H, W = ry1 - ry0, rx1 - rx0
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
    return V, crit, vmax


@njit(cache=True, error_model="numpy", inline="always")
def _steep_box(V, crit, dry_h, ry0, rx0):
    """The box (y0, y1, x0, x1, inclusive; y1 < 0: none) of the region's
    pixels whose paint (volume ``V``, on the dry surface ``dry_h``) is
    steeper towards a neighbour than it holds (``crit``)."""
    H, W = V.shape
    by0, by1, bx0, bx1 = H, -1, W, -1
    for y in range(H):
        for x in range(W):
            if not V[y, x] <= 0.05 and _steep_at(V, crit, dry_h, ry0, rx0, y, x):
                by0, by1 = min(by0, y), max(by1, y)
                bx0, bx1 = min(bx0, x), max(bx1, x)
    return by0, by1, bx0, bx1


@njit(cache=True, error_model="numpy", inline="always")
def _steep_at(V, crit, dry_h, ry0, rx0, y, x):
    """Whether the paint at pixel (y, x) of the region is steeper towards a
    neighbour than it holds (see _steep_box)."""
    H, W = V.shape
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
    return steep


@njit(cache=True, error_model="numpy", inline="always")
def _relax(top, und, dry_h, V, crit, ry0, rx0, y0, y1, x0, x1, iters):
    """Let the paint flow downhill beyond what it holds (see level), over
    the box y0..y1 by x0..x1 of the region, in up to ``iters`` passes, down
    and then across in each, until it has settled."""
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
    oy, ox = ry0 + y0, rx0 + x0
    for _ in range(iters):
        flowed = False
        for axis in range(2):
            if not _pair_flows(Vs, Dh, crit, y0, x0, axis, fa, fb):
                continue
            flowed = True
            if axis == 0:
                _flow_down(top, und, Vs, fa, fb, oy, ox, tcol, ucol, saved_col)
            else:
                _flow_across(top, und, Vs, fa, fb, oy, ox, ta, ua)
        if not flowed:  # it has settled: nothing more will move
            break


@njit(cache=True, error_model="numpy", inline="always")
def _pair_flows(Vs, Dh, crit, y0, x0, axis, fa, fb):
    """The downhill flow between each pixel a of the box and the next one b,
    down (``axis`` 0) or to the right (1), beyond what the paint can hold;
    onto bare canvas only from a sheer edge. Sets the share of a's paint that
    goes to b (``fa``) and of b's that goes to a (``fb``), and returns
    whether any flows."""
    h, w = Vs.shape
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
    return any_flow


@njit(cache=True, error_model="numpy", inline="always")
def _flow_down(top, und, Vs, fa, fb, oy, ox, tcol, ucol, saved_col):
    """Move the paint between each pixel of the box at canvas pixel (oy, ox)
    and the one below it as ``fa`` and ``fb`` say (see _pair_flows), all
    pairs at once, from the paint as it was before this pass: the columns
    side by side, a row at a time (each column is walked down as on its
    own, and the rows are read in the order they lie); a pixel changed by
    the pair above it keeps a copy of how it was (``tcol``, ``ucol``)."""
    h, w = Vs.shape
    for x in range(w):
        saved_col[x] = False
    for k in range(1, h):
        yp, y = k - 1, k
        Ya, Yb = oy + yp, oy + y
        for x in range(w):
            f_a, f_b = fa[yp, x], fb[yp, x]
            if not (f_a > 0 or f_b > 0):
                saved_col[x] = False
                continue
            X = ox + x
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


@njit(cache=True, error_model="numpy", inline="always")
def _flow_across(top, und, Vs, fa, fb, oy, ox, ta, ua):
    """:func:`_flow_down` between each pixel and the one to its right, a row
    at a time (``ta``, ``ua``: the copy of the pixel before)."""
    h, w = Vs.shape
    for line in range(h):
        saved = False  # (ta, ua) hold the previous pixel as it was
        for k in range(1, w):
            yp, xp = line, k - 1
            f_a, f_b = fa[yp, xp], fb[yp, xp]
            if not (f_a > 0 or f_b > 0):
                saved = False
                continue
            y, x = line, k
            Ya, Xa = oy + yp, ox + xp
            Yb, Xb = oy + y, ox + x
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
    y_lo, y_hi, x_lo, x_hi = _own_thick_box(top, ids, stroke_id, ry0, ry1, rx0, rx1)
    if y_hi < 0:
        return
    oy, ox = ry0 + y_lo, rx0 + x_lo
    H, W = y_hi - y_lo + 1, x_hi - x_lo + 1
    k, sig = _soften_rates(top, ids, stroke_id, oy, ox, H, W)
    same_r, same_d = _same_neighbours(k, sig)
    S = np.empty((H, W), np.float32)
    snap = np.empty((H, W, PACKED), np.float32)
    for _ in range(passes):
        _snapshot(top, und, dry_h, k, oy, ox, S, snap)
        # each pixel gives each lower neighbour a share of its paint (at most a
        # fifth to each, so nothing runs dry)
        for y in range(H):
            for x in range(W):
                if k[y, x] != 0:
                    _round_off(top, k, S, snap, same_r, same_d, oy, ox, y, x)


@njit(cache=True, error_model="numpy", inline="always")
def _own_thick_box(top, ids, stroke_id, ry0, ry1, rx0, rx1):
    """The box (y_lo, y_hi, x_lo, x_hi, inclusive, in the region; y_hi < 0:
    none) of the pixels of the region that hold a thick film (SOFTEN_MIN) of
    the stroke's own paint on top."""
    H, W = ry1 - ry0, rx1 - rx0
    y_lo, y_hi, x_lo, x_hi = H, -1, W, -1
    for y in range(H):
        for x in range(W):
            Y, X = ry0 + y, rx0 + x
            if ids[Y, X] == stroke_id and top[Y, X, IV] >= SOFTEN_MIN:
                y_lo, y_hi = min(y_lo, y), max(y_hi, y)
                x_lo, x_hi = min(x_lo, x), max(x_hi, x)
    return y_lo, y_hi, x_lo, x_hi


@njit(cache=True, error_model="numpy", inline="always")
def _soften_rates(top, ids, stroke_id, oy, ox, H, W):
    """How readily the stroke's thick paint on top at each pixel of the (H,
    W) box at canvas pixel (oy, ox) rounds off, by its body (0 where there is
    none to soften); and what its paint is (as alike compares paints)."""
    k = np.zeros((H, W), np.float32)
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
    return k, sig


@njit(cache=True, error_model="numpy", inline="always")
def _same_neighbours(k, sig):
    """Which pixels with paint to soften (``k`` not 0) have a neighbour to the
    right, and below, that has too and holds much the same paint (``sig``,
    see _soften_rates; SOFTEN_SAME)."""
    H, W = k.shape
    same_r = np.zeros((H, W), np.bool_)
    same_d = np.zeros((H, W), np.bool_)
    for y in range(H):
        for x in range(W):
            if k[y, x] == 0:
                continue
            if x + 1 < W and k[y, x + 1] != 0:
                same_r[y, x] = _sig_diff(sig, y, x, y, x + 1) / IV < SOFTEN_SAME
            if y + 1 < H and k[y + 1, x] != 0:
                same_d[y, x] = _sig_diff(sig, y, x, y + 1, x) / IV < SOFTEN_SAME
    return same_r, same_d


@njit(cache=True, error_model="numpy", inline="always")
def _sig_diff(sig, y, x, yy, xx):
    """How far apart the paints at (y, x) and (yy, xx) are (see
    _soften_rates), summed over the bands (single precision)."""
    diff = np.float32(0.0)
    for c in range(IV):
        diff += abs(sig[y, x, c] - sig[yy, xx, c])
    return diff


@njit(cache=True, error_model="numpy", inline="always")
def _snapshot(top, und, dry_h, k, oy, ox, S, snap):
    """The height of the surface (``S``: the dry surface and the wet paint)
    at each pixel of the box at canvas pixel (oy, ox), and the paint on top
    where it softens (``snap``), as they are at the start of a pass."""
    H, W = k.shape
    for y in range(H):
        for x in range(W):
            Y, X = oy + y, ox + x
            S[y, x] = dry_h[Y, X] + top[Y, X, IV] + und[Y, X, IV]
            if k[y, x] != 0:
                for c in range(PACKED):
                    snap[y, x, c] = top[Y, X, c]


@njit(cache=True, error_model="numpy", inline="always")
def _round_off(top, k, S, snap, same_r, same_d, oy, ox, y, x):
    """Pixel (y, x) of the box at canvas pixel (oy, ox) gives each lower
    neighbour of much the same paint (``same_r``, ``same_d``) a share of its
    paint as it was at the start of the pass (``snap``): by how much lower
    its surface ``S`` is and how readily they round off (``k``), at most a
    fifth."""
    H, W = k.shape
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
                if row[x, IV] < zero:
                    for c in range(PACKED):
                        row[x, c] = zero
                    continue
                for c in range(PACKED):
                    v = row[x, c]
                    row[x, c] = max(v, zero)


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
    inside = _inside_paint(top, und, ry0, ry1, rx0, rx1)
    reliable = np.empty((H, W), np.float32)
    uncovered = np.zeros((H, W), np.bool_)
    for li in range(2):
        uncovered[:] = False
        layer = top if li == 0 else und
        A = top_a if li == 0 else und_a
        _reliable_films(layer, A, reliable, ry0, rx0)
        if _cover_films(layer, A, reliable, inside, uncovered, ry0, rx0):
            _cover_pushed(layer, A, reliable, inside, uncovered, ry0, rx0)


@njit(cache=True, error_model="numpy", inline="always")
def _inside_paint(top, und, ry0, ry1, rx0, rx1):
    """Which pixels of the region have a film of wet paint (MIN_FILM) on
    every neighbour (3 x 3, within the region): the least of them, along each
    row and then down each column."""
    H, W = ry1 - ry0, rx1 - rx0
    vol = np.empty((H, W), np.float32)
    for y in range(H):
        for x in range(W):
            vol[y, x] = top[ry0 + y, rx0 + x, IV] + und[ry0 + y, rx0 + x, IV]
    row_lo = np.empty((H, W), np.float32)
    for y in range(H):
        for x in range(W):
            lo = vol[y, x]
            if x > 0:
                lo = min(lo, vol[y, x - 1])
            if x < W - 1:
                lo = min(lo, vol[y, x + 1])
            row_lo[y, x] = lo
    inside = np.empty((H, W), np.bool_)
    for y in range(H):
        for x in range(W):
            lo = row_lo[y, x]
            if y > 0:
                lo = min(lo, row_lo[y - 1, x])
            if y < H - 1:
                lo = min(lo, row_lo[y + 1, x])
            inside[y, x] = lo > MIN_FILM
    return inside


@njit(cache=True, error_model="numpy", inline="always")
def _reliable_films(layer, A, reliable, ry0, rx0):
    """The film thickness of ``layer`` at each pixel of the region at (ry0,
    rx0) where it is well defined (it covers at least half the pixel, by
    ``A``), into ``reliable`` (0 elsewhere). A layer with no paint (dust at
    most) covers nothing: it would pass that coverage on to the next paint
    to land there."""
    H, W = reliable.shape
    for y in range(H):
        for x in range(W):
            if layer[ry0 + y, rx0 + x, IV] <= EMPTY:
                A[ry0 + y, rx0 + x] = 0.0
            a = A[ry0 + y, rx0 + x]
            reliable[y, x] = (
                layer[ry0 + y, rx0 + x, IV] / max(a, np.float32(1e-6))
                if a >= 0.5
                else 0.0
            )


@njit(cache=True, error_model="numpy", inline="always")
def _cover_films(layer, A, reliable, inside, uncovered, ry0, rx0):
    """Make each pixel's coverage ``A`` of ``layer`` over the region at (ry0,
    rx0) at least what its paint needs (see _film_cover). Marks in
    ``uncovered`` the pixels holding paint the tool didn't cover, and
    returns whether there are any. (Paint that slumped, or was pushed, into
    a pixel the tool didn't cover has no coverage of its own: see EDGE_FILM.
    A pixel the tool's edge only just touched counts as one it didn't cover:
    whether a speck of a deposit reached it is no reason for its paint to
    cover it differently.)"""
    H, W = reliable.shape
    any_uncovered = False
    for y in range(H):
        for x in range(W):
            a = A[ry0 + y, rx0 + x]
            v = layer[ry0 + y, rx0 + x, IV]
            if not a >= PUSHED_A:
                if v > EMPTY:
                    uncovered[y, x] = True
                    any_uncovered = True
                    continue
                if not a > 0:
                    continue
            need = _film_cover(v, inside[y, x], reliable, y, x)
            if need > a:
                A[ry0 + y, rx0 + x] = need
    return any_uncovered


@njit(cache=True, error_model="numpy", inline="always")
def _film_cover(v, inside, reliable, y, x):
    """How much of pixel (y, x) of the region paint of volume ``v`` covers:
    all of it ``inside`` paint, else as much as makes it no thicker than the
    thickest well-defined film (``reliable``) within 2 px. (A film next to
    nothing thick, a trace, says nothing about how much of this pixel the
    paint covers: no more than no film.)"""
    if inside and v > 0.5 * MIN_FILM:
        return 1.0
    H, W = reliable.shape
    nb = 0.0
    for yy in range(max(y - 2, 0), min(y + 3, H)):
        for xx in range(max(x - 2, 0), min(x + 3, W)):
            nb = max(nb, reliable[yy, xx])
    need = min(v / (1.25 * max(nb, 1e-6)), 1.0) if nb > 0 else 0.0
    if nb < FILM_KNOWN:
        need *= smoothstep(EMPTY, FILM_KNOWN, nb)
    return need


@njit(cache=True, error_model="numpy")
def cover_ramps(top, top_a, ids, stroke_id, region, ramps):
    """Where the paint the stroke ``stroke_id`` laid on top ends in a slope
    (a wide brush's ridge, or its heavy paint slumped to its edge), the last
    pixel covers at least as much as the slope says, not as little as the
    films beside it would (see _film_cover: a pixel holding 2 units next to
    3 and 4.4 covered a third of itself, pale in a solid edge), times
    ``ramps`` (0 to 1), over the ``region`` (ry0, ry1, rx0, rx1).

    Only the stroke's own paint on top, at the edge of it (a pixel beside
    one holding less than MIN_FILM of it) and holding at least MIN_FILM:
    other strokes' paint, the lower layer and thinner traces keep the
    coverage fix_coverage gave them. The slope is read only from the
    stroke's own pixels that it covers (RAMP_SOLID) and that are not at the
    edge themselves, so a pixel this covers more never tells the slope:
    running it again changes nothing."""
    ry0, ry1, rx0, rx1 = region
    H, W = ry1 - ry0, rx1 - rx0
    if H <= 0 or W <= 0:
        return
    own = np.zeros((H, W), np.float32)
    for y in range(H):
        for x in range(W):
            if ids[ry0 + y, rx0 + x] == stroke_id:
                own[y, x] = top[ry0 + y, rx0 + x, IV]
    solid = np.zeros((H, W), np.float32)
    for y in range(H):
        for x in range(W):
            a = top_a[ry0 + y, rx0 + x]
            if a >= RAMP_SOLID and own[y, x] > EMPTY and not _at_edge(own, y, x):
                solid[y, x] = own[y, x] / a
    for y in range(H):
        for x in range(W):
            a = top_a[ry0 + y, rx0 + x]
            v = own[y, x]
            if not (v >= MIN_FILM and a < RAMP_SOLID and _at_edge(own, y, x)):
                continue
            need = ramps * _ramp_cover(v, solid, own, y, x)
            if need > a:
                top_a[ry0 + y, rx0 + x] = need


@njit(cache=True, error_model="numpy", inline="always")
def _at_edge(own, y, x):
    """Whether pixel (y, x) of the region has a neighbour (left, right, above
    or below, within the region) holding less than MIN_FILM of the stroke's
    own paint (``own``)."""
    H, W = own.shape
    for dy, dx in ((0, 1), (0, -1), (1, 0), (-1, 0)):
        yy, xx = y + dy, x + dx
        if 0 <= yy < H and 0 <= xx < W and own[yy, xx] < MIN_FILM:
            return True
    return False


@njit(cache=True, error_model="numpy", inline="always")
def _along(own, y, x, dy, dx):
    """Whether both neighbours of pixel (y, x) along the edge (across the
    direction (dy, dx) it rises inward in) hold at least MIN_FILM of the
    stroke's own paint, and much alike (within RAMP_EVEN): a run of a level
    edge, not a step of a slanting edge's staircase (whose partly covered
    pixels are its anti-aliasing)."""
    H, W = own.shape
    for k in (1, -1):
        yy, xx = y + k * dx, x + k * dy
        if not (0 <= yy < H and 0 <= xx < W and own[yy, xx] >= MIN_FILM):
            return False
    # (along a level edge the edge pixels hold much alike; along a slanting
    # one, what they hold climbs from pixel to pixel of each run of its
    # staircase. Their paint, not their coverage: the pass changes that)
    v1, v2 = own[y + dx, x + dy], own[y - dx, x - dy]
    return abs(v1 - v2) <= RAMP_EVEN * max(v1, v2)


@njit(cache=True, error_model="numpy", inline="always")
def _ramp_cover(v, solid, own, y, x):
    """How much of pixel (y, x) of the region paint of volume ``v`` covers at
    the foot of a slope, at least (0: nothing to say): as much as makes it
    as thick as the films of the two pixels inward of it (left, right, above
    or below, both read from ``solid``, see cover_ramps) carried on one pixel
    (at least a quarter of the nearer one), where they rise inward: fully
    where the inner one is at least RAMP_RISE[1] thicker than the nearer
    (relative to the inner), not at all below RAMP_RISE[0] (the plateau of a
    stroke's edge keeps its anti-aliasing), in proportion between; up the
    steepest of the four directions (the way inward: along a slanting edge,
    the pixels beside it on its own row rise too, more gently), along a run
    of the edge (see _along). It depends on nothing it changes: the result
    of cover_ramps doesn't either."""
    H, W = solid.shape
    rise, r1, r2 = 0.0, 0.0, 0.0
    for dy, dx in ((0, 1), (0, -1), (1, 0), (-1, 0)):
        y1, x1, y2, x2 = y + dy, x + dx, y + 2 * dy, x + 2 * dx
        if not (0 <= y2 < H and 0 <= x2 < W):
            continue
        f1, f2 = solid[y1, x1], solid[y2, x2]
        if f1 > 0 and f2 - f1 > rise and _along(own, y, x, dy, dx):
            rise, r1, r2 = f2 - f1, f1, f2
    if not rise > 0:
        return 0.0
    g = smoothstep(RAMP_RISE[0], RAMP_RISE[1], rise / r2)
    film = max(2.0 * r1 - r2, 0.25 * r1)
    return g * min(v / film, 1.0)


@njit(cache=True, error_model="numpy")
def _cover_pushed(layer, A, reliable, inside, uncovered, ry0, rx0):
    """Coverage for paint in pixels the tool didn't cover (slumped or pushed
    there: see fix_coverage). Inside paint, all of the pixel. Right beside an
    edge (left, right, above or below a covered pixel of like paint whose film
    is comparable, see EDGE_RAMP), it carries on the edge's anti-aliased ramp:
    covered no more than the least covered of those pixels, and a pixel
    further out (EDGE_REACH) no more than the least of those it lies beside.
    Elsewhere (a band of it further out, a speck), as much as its volume
    makes of the films of like paint around it (EDGE_FILM). Each pass reads
    only what was settled before it: the result doesn't depend on the order
    the pixels are taken in."""
    H, W = uncovered.shape
    # (only over the rows and columns that hold such paint)
    y_lo, y_hi, x_lo, x_hi = H, 0, W, 0
    for y in range(H):
        for x in range(W):
            if uncovered[y, x]:
                y_lo, y_hi = min(y_lo, y), max(y_hi, y + 1)
                x_lo, x_hi = min(x_lo, x), max(x_hi, x + 1)
    box = (y_lo, y_hi, x_lo, x_hi)
    # inside paint first: fully covered, and covered paint for those beside it
    for y in range(y_lo, y_hi):
        for x in range(x_lo, x_hi):
            thick = layer[ry0 + y, rx0 + x, IV] > 0.5 * MIN_FILM
            if uncovered[y, x] and inside[y, x] and thick:
                A[ry0 + y, rx0 + x] = 1.0
                uncovered[y, x] = False
    cap, when = _edge_caps(layer, A, uncovered, ry0, rx0, box)
    for y in range(y_lo, y_hi):
        for x in range(x_lo, x_hi):
            if not uncovered[y, x]:
                continue
            v = layer[ry0 + y, rx0 + x, IV]
            a = A[ry0 + y, rx0 + x]
            if when[y, x] > 0:
                # (a trace thinner than a film covers no more than that)
                A[ry0 + y, rx0 + x] = max(min(cap[y, x], v / MIN_FILM), a)
            else:
                nb = _edge_film(layer, reliable, ry0, rx0, y, x, H, W)
                n = min(v / (EDGE_FILM * max(nb, 1e-6)), 1.0) if nb > 0 else 0.0
                if nb < FILM_KNOWN:
                    n *= smoothstep(EMPTY, FILM_KNOWN, nb)
                A[ry0 + y, rx0 + x] = max(n, a)


@njit(cache=True, error_model="numpy", inline="always")
def _edge_caps(layer, A, uncovered, ry0, rx0, box):
    """The edge's anti-aliased ramp carried on into the ``uncovered`` pixels
    of ``layer`` (see _cover_pushed), a pixel out at a time (EDGE_REACH
    times): from the covered pixels, then from those given coverage from
    them in the pass before (each pass reads only what was settled before
    it), over the ``box`` (y_lo, y_hi, x_lo, x_hi) of them. Returns the most
    each may cover, and the pass it had that in (0: none)."""
    H, W = uncovered.shape
    y_lo, y_hi, x_lo, x_hi = box
    cap = np.full((H, W), 2.0)
    when = np.zeros((H, W), np.int8)  # the pass a pixel's cap was had in
    for k in range(1, EDGE_REACH + 1):
        for y in range(y_lo, y_hi):
            for x in range(x_lo, x_hi):
                if not uncovered[y, x] or when[y, x] != 0:
                    continue
                c_min = _edge_cap(layer, A, uncovered, cap, when, ry0, rx0, y, x, k)
                if c_min <= 1.0:
                    cap[y, x] = c_min
                    when[y, x] = k
    return cap, when


@njit(cache=True, error_model="numpy", inline="always")
def _edge_cap(layer, A, uncovered, cap, when, ry0, rx0, y, x, k):
    """The least coverage of the pixels left, right, above and below pixel
    (y, x) of the region whose paint is the edge of its paint (covered, or
    given a cap in a pass before ``k``; of like paint and a comparable film,
    see EDGE_RAMP); 2 if there are none."""
    H, W = uncovered.shape
    v = layer[ry0 + y, rx0 + x, IV]
    me = layer[ry0 + y, rx0 + x]
    c_min = 2.0
    for d in range(4):
        yy = y + (1 if d == 0 else -1 if d == 1 else 0)
        xx = x + (1 if d == 2 else -1 if d == 3 else 0)
        if not (0 <= yy < H and 0 <= xx < W):
            continue
        if uncovered[yy, xx]:
            if not (0 < when[yy, xx] < k):
                continue
            c = cap[yy, xx]
        else:
            c = A[ry0 + yy, rx0 + xx]
        if not c > 0:
            continue
        other = layer[ry0 + yy, rx0 + xx]
        # (a faint film beside it is not the edge of this paint)
        if other[IV] / c < EDGE_RAMP * v:
            continue
        if alike_below(me, other, SAME_PAINT):
            c_min = min(c_min, c)
    return c_min


@njit(cache=True, error_model="numpy")
def _edge_film(layer, reliable, ry0, rx0, y, x, H, W):
    """How thick the paint around pixel (y, x) of the region is, for paint in
    it that has no coverage (see fix_coverage): the mean of the well-defined
    films (``reliable``) of much the same paint within 2 px; failing those,
    the volume of the like paint around it (its mean, weighted by volume: a
    speck more or less beside it changes nothing)."""
    me = layer[ry0 + y, rx0 + x]
    tot, n, vtot, vn = 0.0, 0, 0.0, 0.0
    for yy in range(max(y - 2, 0), min(y + 3, H)):
        for xx in range(max(x - 2, 0), min(x + 3, W)):
            if yy == y and xx == x:
                continue
            other = layer[ry0 + yy, rx0 + xx]
            if not other[IV] > EMPTY or not alike_below(me, other, SAME_PAINT):
                continue
            if reliable[yy, xx] > 0:
                tot += reliable[yy, xx]
                n += 1
            vtot += other[IV] * other[IV]
            vn += other[IV]
    if n > 0:
        return tot / n
    return vtot / vn if vn > 0 else 0.0


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


# ----------------------------------------------------------------------------
# a knife's blade (see _knife): its stages pixel by pixel, with the same
# arithmetic, in the same order and precision, as numpy works them out there.
# (numpy's exp and log differ from the C library's in the last bit, so those
# stay in numpy: the kernels take them worked out.) A stage works on the
# (h, w) pixels from (oy, ox) of the canvas, the part of the step's
# footprint ``F`` at (wy0, wx0): its part of the canvas, or all of its window,
# where past the edge of the canvas each pixel reads the nearest one on it.


@njit(cache=True, error_model="numpy")
def np_max(a, b):
    """``np.maximum(a, b)`` (no NaN): b when the two are equal (+0 or -0)."""
    return max(b, a)


@njit(cache=True, error_model="numpy")
def np_min(a, b):
    """``np.minimum(a, b)`` (no NaN): b when the two are equal."""
    return min(b, a)


@njit(cache=True, error_model="numpy")
def nearest(i, n):
    """The nearest of the indices 0..n-1 to ``i``."""
    return min(max(i, 0), n - 1)


@njit(cache=True, error_model="numpy")
def blade_rest(
    top, und, ids, fine, wide, oy, ox, h, w, stroke_id, stiff, F, wy0, wx0, gathered
):
    """Where a blade of stiffness ``stiff`` rests on the solid surface (from
    the thread tops, ``fine``, to the highest ridges around, ``wide``); the
    wet paint of earlier strokes (single precision, as the paint) and the
    most of it; which pixels lie under the blade (mostly covered by its
    chord); whether any do; the IB plane of the top layer where it is an
    earlier stroke's (0 elsewhere); and, if ``gathered`` (a window past the
    edge of the canvas), the IB plane of the layer below (empty otherwise).
    See _knife._blade_rest."""
    H, W = top.shape[0], top.shape[1]
    rest = np.empty((h, w), np.float32)
    old_v = np.empty((h, w), np.float32)
    top_ib = np.empty((h, w), np.float32)
    und_ib = np.empty((h, w) if gathered else (0, 0), np.float32)
    under = np.empty((h, w), np.bool_)
    k = np.float32(stiff)
    zero = np.float32(0.0)
    vmax = -np.inf
    any_under = False
    for y in range(h):
        Y = nearest(oy + y, H)
        for x in range(w):
            X = nearest(ox + x, W)
            rest[y, x] = fine[Y, X] + k * (wide[Y, X] - fine[Y, X])
            earlier = ids[Y, X] != stroke_id
            v = und[Y, X, IV] + (top[Y, X, IV] if earlier else zero)
            old_v[y, x] = v
            vmax = max(vmax, v)
            top_ib[y, x] = top[Y, X, IB] if earlier else zero
            if gathered:
                und_ib[y, x] = und[Y, X, IB]
            wy, wx = wy0 + y, wx0 + x
            ub = (
                F[F_AA, wy, wx]
                * clip(F[F_H, wy, wx] - abs(F[F_REL, wy, wx]) + 0.5, 0.0, 1.0)
                > 0.5
            )
            under[y, x] = ub
            any_under = any_under or ub
    return rest, old_v, vmax, under, any_under, top_ib, und_ib


@njit(cache=True, error_model="numpy")
def blade_bins(floor, oy, ox, old_v, rest, under, F, wy0, wx0):
    """The high points of the paint and surface under a blade (``floor`` and
    ``old_v``, the highest within a pixel), and, by position along the
    blade (whole pixels from its end), how many pixels lie under it and the
    sum of those and of where it rests (``rest``) over them: what
    _knife._along_blade averages along it. Returns (high points, each
    pixel's position, counts, sums of high points, sums of rest)."""
    H, W = floor.shape
    h, w = old_v.shape
    hi_in = np.empty((h, w), np.float32)
    lat_min = np.inf
    for y in range(h):
        for x in range(w):
            hi_in[y, x] = floor[nearest(oy + y, H), nearest(ox + x, W)] + old_v[y, x]
            lat_min = min(lat_min, F[F_LAT, wy0 + y, wx0 + x])
    hi = max_filter(hi_in, 1)
    b = np.empty((h, w), np.int64)
    n = 0
    for y in range(h):
        for x in range(w):
            b[y, x] = int(np.rint(F[F_LAT, wy0 + y, wx0 + x] - lat_min))
            n = max(n, b[y, x] + 1)
    cnt = np.zeros(n)
    tot_hi = np.zeros(n)
    tot_rest = np.zeros(n)
    for y in range(h):
        for x in range(w):
            if under[y, x]:
                cnt[b[y, x]] += 1.0
                tot_hi[b[y, x]] += hi[y, x]
                tot_rest[b[y, x]] += rest[y, x]
    return hi, b, cnt, tot_hi, tot_rest


@njit(cache=True, error_model="numpy")
def blade_sink(rest, hi, b, cnt_s, hi_s, rest_s, depth):
    """Where the blade rests over each pixel (see _knife._blade_rest): on the
    high points under it, averaged along the blade (``hi_s`` over ``cnt_s``
    pixels at each position ``b``), sunk ``depth`` of the way down towards
    the surface below (``rest_s``), and never below that."""
    h, w = rest.shape
    out = np.empty((h, w))
    for y in range(h):
        for x in range(w):
            k = b[y, x]
            if cnt_s[k] > 0:
                high = hi_s[k] / np_max(cnt_s[k], 1e-9)
                base = rest_s[k] / np_max(cnt_s[k], 1e-9)
            else:
                high, base = np.float64(hi[y, x]), np.float64(rest[y, x])
            out[y, x] = np_max(
                np.float64(rest[y, x]), high - depth * np_max(high - base, 0.0)
            )
    return out


@njit(cache=True, error_model="numpy")
def blade_level(rest, film, k_film, gap, comb, chat, tilt, F, wy0, wx0):
    """How high the blade rides over each pixel (see _knife._blade_level):
    ``gap`` above where it rests (``rest``), or no higher than ``k_film`` of
    the ``film`` of paint under it (not a comb's teeth), wavering by ``chat``
    (1 + its chatter), the nicks of its edge (the hair plane of ``F``) and
    its ``tilt``. Returns the level and how high it rides."""
    h, w = rest.shape
    level = np.empty((h, w))
    ride = np.empty((h, w))
    for y in range(h):
        for x in range(w):
            r = gap if comb else np_min(gap, k_film * film[y, x])
            ride[y, x] = r
            hair, u = F[F_HAIR, wy0 + y, wx0 + x], F[F_U, wy0 + y, wx0 + x]
            level[y, x] = rest[y, x] + r * (
                chat + 0.12 * (hair - 0.5) + tilt * clip(u, -1.0, 1.0)
            )
    return level, ride


@njit(cache=True, error_model="numpy")
def blade_slab(top, und, floor, oy, ox, level, surf_add, film, fill):
    """How much paint the blade lays under it over each pixel (see
    _knife._slab): it fills the share ``fill`` (1 - e^(-k exposure), worked
    out in numpy) of the gap between its ``level`` and the surface (the
    solid surface and the wet paint, plus ``surf_add`` where it is not
    empty), where the ``film`` on it reaches. Returns that and the surface."""
    H, W = floor.shape
    h, w = level.shape
    add = surf_add.shape[0] > 0
    d = np.empty((h, w))
    surf = np.empty((h, w))
    for y in range(h):
        Y = nearest(oy + y, H)
        for x in range(w):
            X = nearest(ox + x, W)
            s = np.float64(floor[Y, X] + top[Y, X, IV] + und[Y, X, IV])
            if add:
                s += surf_add[y, x]
            surf[y, x] = s
            gap_px = np_max(level[y, x] - s, 0.0)
            reach = smoothstep(1.3, 0.7, gap_px / np_max(film[y, x], 1e-9))
            d[y, x] = gap_px * reach * fill[y, x]
    return d, surf


@njit(cache=True, error_model="numpy")
def off_surface(top, ids, stroke_id, oy, ox, laid, on_y0, on_x0, on_h, on_w):
    """What a blade over the (h, w) pixels from (oy, ox), past the edge of
    the canvas, adds to the surface it reads off the nearest canvas pixel
    (see _knife._whole_window): the slab ``laid`` out there, in place of
    the stroke's own paint at that pixel. Nothing in the (on_h, on_w) part
    at (on_y0, on_x0), on the canvas."""
    H, W = ids.shape
    h, w = laid.shape
    out = np.empty((h, w))
    for y in range(h):
        Y = nearest(oy + y, H)
        for x in range(w):
            if on_y0 <= y < on_y0 + on_h and on_x0 <= x < on_x0 + on_w:
                out[y, x] = 0.0
                continue
            X = nearest(ox + x, W)
            own = top[Y, X, IV] if ids[Y, X] == stroke_id else np.float32(0.0)
            out[y, x] = laid[y, x] - own
    return out


@njit(cache=True, error_model="numpy")
def blade_ends(ride, F, wy0, wx0, half, end_px, coef, on_y0, on_x0, on_h, on_w):
    """How much paint the blade presses out near its two ends over each pixel
    (see _knife._press_out_ends): ``coef`` of how high it rides, where the
    pixel lies within ``end_px`` of an end (``half``: the half-length of the
    blade, px) and as much as it passes over it; none in the (on_h, on_w)
    part at (on_y0, on_x0)."""
    h, w = ride.shape
    out = np.empty((h, w))
    for y in range(h):
        for x in range(w):
            if on_y0 <= y < on_y0 + on_h and on_x0 <= x < on_x0 + on_w:
                out[y, x] = 0.0
                continue
            r = ride[y, x]
            near = smoothstep(end_px, 0.0, half - abs(F[F_LAT, wy0 + y, wx0 + x]))
            out[y, x] = (
                r * F[F_EXPO, wy0 + y, wx0 + x] * coef * near * smoothstep(0.5, 2.0, r)
            )
    return out


@njit(cache=True, error_model="numpy")
def blade_excess(
    top, und, floor, sy0, sx0, level, fill, F, wy0, wx0, half, spill_px, spill, wet
):
    """The wet paint standing above the blade over each pixel of the (h, w)
    window at (sy0, sx0) (see _knife._push_aside), the share ``fill`` of it
    the blade passes over, and of that, within ``spill_px`` of the edges of
    the mark (``half`` px from its middle), the share ``spill`` of it or of
    the paint lying there before the stroke (``wet``), whichever is less,
    that spills out sideways (``wet`` empty, a comb: all of it stays).
    Returns that, the volume of the top layer, and the most of it."""
    h, w = level.shape
    excess = np.empty((h, w))
    tv = np.empty((h, w))
    emax = -np.inf
    for y in range(h):
        for x in range(w):
            Y, X = sy0 + y, sx0 + x
            t = np.float64(top[Y, X, IV])
            tv[y, x] = t
            e = np_max(floor[Y, X] + t + und[Y, X, IV] - level[y, x], 0.0)
            e *= fill[y, x]
            if wet.shape[0] > 0:
                near = smoothstep(spill_px, 0.0, half - abs(F[F_LAT, wy0 + y, wx0 + x]))
                e = spill * near * np_min(e, np.float64(wet[y, x]))
            excess[y, x] = e
            emax = max(emax, e)
    return excess, tv, emax


@njit(cache=True, error_model="numpy")
def blade_folds(und, floor, sy0, sx0, level, mix, F, wy0, wx0):
    """How far the blade folds the two wet layers of each pixel of the window
    at (sy0, sx0) into each other (see _knife._fold_layers): ``mix`` (1 -
    e^(-...), worked out in numpy), in streaks along its nicks, as far as
    it cuts into the layer below its ``level``."""
    h, w = level.shape
    m = np.empty((h, w))
    half = np.float32(0.5)
    for y in range(h):
        for x in range(w):
            Y, X = sy0 + y, sx0 + x
            into = clip(
                np.float64(floor[Y, X] + und[Y, X, IV] + half) - level[y, x], 0.0, 1.0
            )
            hair = F[F_HAIR, wy0 + y, wx0 + x]
            m[y, x] = clip(mix[y, x] * (0.1 + 1.8 * hair) * into, 0.0, 1.0)
    return m


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
    aa,
    along,
    c,
    chord_half,
    ds,
    tx,
    ty,
    slip,
    viscous,
):
    """Move the wet paint a blade touches as it travels ``ds`` px along (tx, ty)
    in the window at (sy0, sx0) (thin-film shear and bulldozing, see
    ``layer_speed``).

    The blade's footprint at each sub-step is where its anti-aliased
    coverage ``aa`` (the footprint's F_AA) is above 0 and the pixel's
    position ``along`` the motion lies within ``chord_half`` (the chord's
    half-length, F_H) of the chord centre ``c`` (relative to the blade's
    position at the end of the step). Conservative upwind fluxes, in
    sub-steps short enough to be stable.

    The per-pixel arrays (``floor``, ``level``, ``aa``, ``along``,
    ``c``, ``chord_half``, ``slip``) all cover the window: they have one shape."""
    top, und = cv.top, cv.und
    # (the helpers below take that shape from whichever of them they are given)
    hh, ww = level.shape
    own, gap = _blade_gap(cv, level, floor, sy0, sx0, stroke_id)
    # the paint under the blade must reach it somewhere for anything to happen
    if not _paint_reaches(cv, own, gap, aa, sy0, sx0):
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
    in_rows = (np.empty(hh, np.int64), np.empty(hh, np.int64))
    c_rows = (np.empty(hh, np.int64), np.empty(hh, np.int64))
    for k in range(1, n_sub + 1):
        # the blade's position along the motion
        pos = -ds + k * step
        if not _under_blade(aa, chord_half, along, c, pos, inside, in_rows):
            continue
        # the flow is smooth over a pixel or two: speeds from lightly smoothed heights
        _smoothed_heights(cv, own, sy0, sx0, in_rows, (u_src, t_src), (u, t))
        c_u, c_t, any_c = _contact_speeds(
            cv, own, inside, gap, slip, sy0, sx0, in_rows, (u, t), c_rows
        )
        if not any_c:
            continue
        # viscosity: neighbouring paint drags each other along, so the speed varies
        # smoothly (without it, the paint tears into foam)
        c_lo, c_hi = c_rows
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
            _shares_moved(frac, own, li, step, ax, ay, (f_lo, f_hi), fx, fy)
            _upwind(layer, fx, fy, (g_lo, g_hi), sy0, sx0, xdir, ydir)


@njit(cache=True, error_model="numpy", inline="always")
def _blade_gap(cv, level, floor, sy0, sx0, stroke_id):
    """Which pixels of the window at (sy0, sx0) hold the stroke's own paint
    on top, and how high the blade rides over the solid surface there (px,
    at least 0.2), blurred: under paint, the blade doesn't feel the
    individual threads of the canvas."""
    ids = cv.top_id
    hh, ww = level.shape
    own = np.empty((hh, ww), np.bool_)
    g = np.empty((hh, ww))
    for y in range(hh):
        for x in range(ww):
            own[y, x] = ids[sy0 + y, sx0 + x] == stroke_id
            g[y, x] = max(level[y, x] - floor[y, x], 0.2)
    return own, box_blur(g, 2)


@njit(cache=True, error_model="numpy", inline="always")
def _paint_reaches(cv, own, gap, aa, sy0, sx0):
    """Whether the earlier strokes' wet paint reaches up to within a px of
    the blade (``gap`` above the surface) anywhere the blade covers any of
    the pixel (``aa`` > 0) in the window at (sy0, sx0)."""
    top, und = cv.top, cv.und
    hh, ww = gap.shape
    for y in range(hh):
        for x in range(ww):
            Y, X = sy0 + y, sx0 + x
            reach = und[Y, X, IV] + (0.0 if own[y, x] else top[Y, X, IV])
            if reach > gap[y, x] - 1.0 and aa[y, x] > 0:
                return True
    return False


@njit(cache=True, error_model="numpy", inline="always")
def _under_blade(aa, chord_half, along, c, pos, inside, in_rows):
    """How much of each pixel the blade is over at ``pos`` px along the
    motion (see flow_under_blade), into ``inside``; into ``in_rows``, the
    columns of each row where it is over any, in_lo..in_hi; and whether it
    is over any pixel."""
    in_lo, in_hi = in_rows
    hh, ww = aa.shape
    any_in = False
    for y in range(hh):
        in_lo[y], in_hi[y] = ww, 0
        for x in range(ww):
            v = aa[y, x] * clip(
                chord_half[y, x] - abs(along[y, x] - pos - c[y, x]) + 0.5, 0.0, 1.0
            )
            inside[y, x] = v
            if v != 0:
                any_in = True
                in_lo[y], in_hi[y] = min(in_lo[y], x), x + 1
    return any_in


@njit(cache=True, error_model="numpy", inline="always")
def _smoothed_heights(cv, own, sy0, sx0, in_rows, src, heights):
    """The thickness of the layer below (u) and of the earlier strokes'
    paint on top (t) at the pixels the blade is over (``in_rows``: in_lo..in_hi
    of each row), lightly smoothed, into ``heights`` (u, t); from ``src``
    (u_src, t_src), filled in here a pixel further out."""
    top, und = cv.top, cv.und
    in_lo, in_hi = in_rows
    u_src, t_src = src
    u, t = heights
    hh, ww = own.shape
    lo1, hi1 = grown(in_lo, in_hi, 1, ww)
    for y in range(hh):
        for x in range(lo1[y], hi1[y]):
            u_src[y, x] = und[sy0 + y, sx0 + x, IV]
            t_src[y, x] = 0.0 if own[y, x] else top[sy0 + y, sx0 + x, IV]
    box_blur_rows(u_src, 1, in_lo, in_hi, u)
    box_blur_rows(t_src, 1, in_lo, in_hi, t)


@njit(cache=True, error_model="numpy", inline="always")
def _contact_speeds(cv, own, inside, gap, slip, sy0, sx0, in_rows, heights, c_rows):
    """How fast the blade, over ``inside`` of each pixel at ``gap`` above
    the surface, moves the two wet layers (``heights``: (u, t) thick, see
    _smoothed_heights; see layer_speed), as a fraction of its own speed:
    (below, on top, whether it moves any). Sets in ``c_rows`` the columns of
    each row where it does, c_lo..c_hi."""
    top = cv.top
    in_lo, in_hi = in_rows
    u, t = heights
    c_lo, c_hi = c_rows
    hh, ww = own.shape
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
    return c_u, c_t, any_c


@njit(cache=True, error_model="numpy", inline="always")
def _shares_moved(frac, own, li, step, ax, ay, f_rows, fx, fy):
    """The share of each pixel's paint in layer ``li`` (0 below, 1 on top:
    the stroke's own paint on top stays) that moves on to its neighbour
    downstream in x (``fx``) and in y (``fy``) in a sub-step of ``step`` px
    along (ax, ay), at ``frac`` of the blade's speed (``f_rows``: columns
    f_lo..f_hi of each row)."""
    f_lo, f_hi = f_rows
    hh = frac.shape[0]
    for y in range(hh):
        for x in range(f_lo[y], f_hi[y]):
            f = clip(frac[y, x], 0.0, 1.0) * step
            if li == 1 and own[y, x]:
                f = 0.0
            fx[y, x] = f * ax
            fy[y, x] = f * ay


@njit(cache=True, error_model="numpy", inline="always")
def _upwind(layer, fx, fy, g_rows, sy0, sx0, xdir, ydir):
    """Each pixel of the window at (sy0, sx0) (``g_rows``: columns
    g_lo..g_hi of each row) passes the shares ``fx`` and ``fy`` of its
    paint in ``layer`` on to its neighbour downstream in x and in y
    (``xdir``, ``ydir``: +1 or -1), walking upstream, so the neighbours are
    still as they were."""
    g_lo, g_hi = g_rows
    hh, ww = fx.shape
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
                # (rounding doesn't leave less than nothing where all
                # of it moved on)
                layer[Y, X, ch] = nv if nv >= 0 else np.float32(0.0)


# ----------------------------------------------------------------------------
# a tool lifted off: paint pulled up into peaks (see stroke._dab_lift and
# _knife._lift_off)

# Sinkhorn passes of the local plan pull_up moves paint along. What the plan
# can't place goes through a pool shared among all the pixels that rise:
# paint with no rising pixel within reach (a dab's rim), and what the plan
# gives a rising pixel beyond its share where the pixels near it can't be
# balanced. On the marks tried that is 18% to 47% of what a dab moves and 4%
# to 19% of what a knife moves; more passes don't lower a dab's (33% on a
# round hog dab with 4, 20 or 100). The count must stay small: where the
# local problem can't be balanced, the factors a and b grow (and shrink)
# geometrically with every pass (on a palette knife's lift b spans 1e-15 to
# 1e14 after 4 passes, 1e-298 to inf after 100, and the paint comes out
# NaN). With a pool, every pixel ends with the volume asked of it whatever
# the count.
PULL_PASSES = 4
# Factors keep the volume (for the scaling pull_up falls back on) if what the
# pixels that fall give up and what those that rise ask for differ by no
# more than this share of the volume of the pixels that change. The callers'
# factors, normalised in double precision, keep it to 2.4e-16 at worst (178
# lifts of dabs and knives, one to three colours).
PULL_KEEPS = 1e-12


@njit(cache=True, error_model="numpy")
def pull_up(top, fac, sigma):
    """Each pixel of ``top`` (h, w, PACKED: the part of the canvas lifted
    from) is to end with ``fac`` (float64: with single-precision factors a
    pixel's volume can be off by more than their rounding) times its
    volume. The factors should keep the total volume (both callers make
    them so), and must be 1 wherever the paint is to be left alone (paint
    of other strokes). Paint moves, as whole paint (every pigment,
    scattering, volume and body with it), from the pixels that fall to those
    that rise: those that fall give up that share of their paint, those that
    rise keep theirs and take in what comes to them, so every channel is
    kept. Paint within reach of a rising pixel (a square of about 1.7
    ``sigma`` px each way, see _pull_reach) goes to it along a local plan;
    the rest (18% to 47% of what a dab moves, 4% to 19% of what a knife
    moves: see PULL_PASSES) is shared among the rising pixels of the whole
    mark, in proportion to what each still lacks. (Scaling each pixel's
    paint kept the volume but, where the paint holds more than one colour,
    made some pigment and destroyed some: up to a fifth of a band.)

    Where all the pixels that change hold the same paint and the factors
    keep the volume (PULL_KEEPS), it is scaled as before: cheap, and a
    single-colour mark is left as it always was. "The same paint" is
    same_paint's: what is in a unit of each differs by no more than
    SAME_COLUMN_PAINT (1e-6) of its largest component, so a trace of a
    pigment can be made or lost there, by up to that share of the largest
    component times the volume moved; every channel is kept to that and to
    single-precision rounding.

    A pixel with no paint (volume 0) neither gives nor takes, whatever its
    factor. If no pixel would rise, or none fall, nothing changes; if the
    factors don't keep the volume, every channel is still kept, but those
    that rise take in what the others give, not what they asked for."""
    y0, y1, x0, x1 = _changing(fac)
    if y0 >= y1:
        return
    t = top[y0:y1, x0:x1]
    f = fac[y0:y1, x0:x1]
    loss, gain = _falls_and_rises(t, f)
    if not (loss.sum() > 0.0 and gain.sum() > 0.0):
        return
    if _keeps_volume(t, loss, gain) and _uniform(t, loss, gain):
        _scale_paint(t, f, loss, gain)
        return
    k = _pull_reach(sigma, y1 - y0 + x1 - x0)
    a, b, kb = _balance(loss, gain, k, PULL_PASSES)
    got, pool = _pulled(t, f, loss, a, b, kb, k)
    _settle_pulled(t, gain, got, pool)


@njit(cache=True, error_model="numpy", inline="always")
def _keeps_volume(top, loss, gain):
    """Whether what the pixels that fall give up and what those that rise
    ask for are the same, to PULL_KEEPS of the volume of all of them."""
    held = 0.0
    for y in range(loss.shape[0]):
        for x in range(loss.shape[1]):
            if loss[y, x] > 0.0 or gain[y, x] > 0.0:
                held += np.float64(top[y, x, IV])
    return abs(loss.sum() - gain.sum()) <= PULL_KEEPS * held


@njit(cache=True, error_model="numpy", inline="always")
def _changing(fac):
    """The box (y0, y1, x0, x1) round the pixels whose factor isn't 1
    (empty: none)."""
    h, w = fac.shape
    y0, y1, x0, x1 = h, 0, w, 0
    for y in range(h):
        for x in range(w):
            if fac[y, x] != 1.0:
                y0, y1 = min(y0, y), max(y1, y + 1)
                x0, x1 = min(x0, x), max(x1, x + 1)
    return y0, y1, x0, x1


@njit(cache=True, error_model="numpy", inline="always")
def _falls_and_rises(top, fac):
    """The volume each pixel holding paint is to give up (h, w) and to take
    in."""
    h, w = fac.shape
    loss = np.zeros((h, w))
    gain = np.zeros((h, w))
    for y in range(h):
        for x in range(w):
            v = np.float64(top[y, x, IV])
            f = fac[y, x]
            if v > 0.0:
                if f < 1.0:
                    loss[y, x] = v * (1.0 - f)
                elif f > 1.0:
                    gain[y, x] = v * (f - 1.0)
    return loss, gain


@njit(cache=True, error_model="numpy", inline="always")
def _uniform(top, loss, gain):
    """Whether the pixels that give or take all hold the same paint, as
    same_paint decides it for a brush's columns."""
    h, w = loss.shape
    n = 0
    for y in range(h):
        for x in range(w):
            if loss[y, x] > 0.0 or gain[y, x] > 0.0:
                n += 1
    rows = np.empty((n, PACKED))
    i = 0
    for y in range(h):
        for x in range(w):
            if loss[y, x] > 0.0 or gain[y, x] > 0.0:
                for c in range(PACKED):
                    rows[i, c] = top[y, x, c]
                i += 1
    return same_paint(rows)


@njit(cache=True, error_model="numpy", inline="always")
def _scale_paint(top, fac, loss, gain):
    """Each pixel that gives or takes: its paint scaled by its factor (in
    single precision, as numpy scaled it)."""
    for y in range(fac.shape[0]):
        for x in range(fac.shape[1]):
            if loss[y, x] > 0.0 or gain[y, x] > 0.0:
                f = np.float32(fac[y, x])
                for c in range(PACKED):
                    top[y, x, c] = top[y, x, c] * f


@njit(cache=True, error_model="numpy", inline="always")
def _pull_reach(sigma, most):
    """Half-width k (px) of the square window the plan reaches over: one
    whose spread (variance k (k + 1) / 3 along each axis) is a Gaussian's
    of sigma ``sigma``, about 1.7 sigma; 0 (each pixel alone: all of it goes
    through the pool) for none. A sigma past ``most`` counts as ``most``,
    whose window (about 1.7 ``most``) already spans the box."""
    if not sigma > 0.0:
        return 0
    s = min(sigma, float(most))
    # (k (k + 1) / 3 = s^2, to the nearest whole k: about 1.7 s)
    return int(math.floor(math.sqrt(3.0 * s * s + 0.25)))


@njit(cache=True, error_model="numpy", inline="always")
def _window_plan(m, k):
    """For the windows x - k..x + k along a line of ``m`` (cut off at its
    ends), summed within blocks of 2k + 1 (see _window_rows): where each
    block starts and ends, and for each window its first and last place and
    which partial sums make it up (0 the start of the last one's block to
    it, 1 the first one to the end of its block, 2 both)."""
    span = 2 * k + 1
    first = np.empty(m, np.bool_)
    last = np.empty(m, np.bool_)
    lo = np.empty(m, np.int64)
    hi = np.empty(m, np.int64)
    ends = np.empty(m, np.int64)
    for x in range(m):
        first[x] = x % span == 0
        last[x] = x == m - 1 or (x + 1) % span == 0
        lo[x], hi[x] = max(x - k, 0), min(x + k, m - 1)
        if lo[x] // span != hi[x] // span:
            ends[x] = 2
        else:
            ends[x] = 0 if lo[x] % span == 0 else 1
    return first, last, lo, hi, ends


@njit(cache=True, error_model="numpy", inline="always")
def _window_rows(src, out, plan, pre, suf):
    """The sums of ``src`` (h, w, n) over x - k..x + k along each row (cut
    off at its ends) into ``out``, by sums within blocks of 2k + 1: a window
    covers the end of one block and the start of the next, so each sum adds
    two partial sums of what is never less than nothing, and subtracts
    nothing (a running sum would subtract, and could leave a hair less than
    nothing where there is nothing). (``plan``: see _window_plan; ``pre``,
    ``suf``: (w, n), room for the partial sums.)"""
    h, w, n = src.shape
    first, last, lo, hi, ends = plan
    # (adding 0.0 leaves a sum that is never less than nothing as it is)
    for y in range(h):
        for x in range(w):
            for c in range(n):
                before = 0.0 if first[x] else pre[x - 1, c]
                pre[x, c] = before + src[y, x, c]
        for x in range(w - 1, -1, -1):
            for c in range(n):
                after = 0.0 if last[x] else suf[x + 1, c]
                suf[x, c] = after + src[y, x, c]
        for x in range(w):
            a, b = lo[x], hi[x]
            ends_at, starts_at = ends[x] != 0, ends[x] != 1
            for c in range(n):
                end = suf[a, c] if ends_at else 0.0
                out[y, x, c] = end + (pre[b, c] if starts_at else 0.0)


@njit(cache=True, error_model="numpy", inline="always")
def _window_columns(src, out, plan):
    """The same down each column, a row at a time (so that it reads and
    writes along rows), in place: the partial sums from the end of each
    block go into ``out`` and those from its start into ``src`` (which is
    left holding them); each row of ``out`` is then made, from the last up,
    of a row of ``src`` below it and one of ``out`` above it that is still
    as it was."""
    h, w, n = src.shape
    first, last, lo, hi, ends = plan
    for y in range(h - 1, -1, -1):
        for x in range(w):
            for c in range(n):
                after = 0.0 if last[y] else out[y + 1, x, c]
                out[y, x, c] = after + src[y, x, c]
    for y in range(1, h):
        if not first[y]:
            for x in range(w):
                for c in range(n):
                    src[y, x, c] += src[y - 1, x, c]
    for y in range(h - 1, -1, -1):
        a, b = lo[y], hi[y]
        ends_at, starts_at = ends[y] != 0, ends[y] != 1
        for x in range(w):
            for c in range(n):
                end = out[a, x, c] if ends_at else 0.0
                out[y, x, c] = end + (src[b, x, c] if starts_at else 0.0)


@njit(cache=True, error_model="numpy", inline="always")
def _window(src, k, out, room):
    """The sums of ``src`` (h, w, n) over the squares of half-width ``k``
    round each pixel (cut off at the edges), along rows, then down columns,
    into ``out``. The same square round each pair of pixels counts either
    way, so the sum of x times window(y) is the sum of window(x) times y:
    which keeps the paint moved by it. Its cost does not hang on ``k``.
    (``room``: the buffers, see _window_room.)"""
    tmp, line_pre, line_suf = room
    h, w, _ = src.shape
    _window_rows(src, tmp, _window_plan(w, k), line_pre, line_suf)
    _window_columns(tmp, out, _window_plan(h, k))


@njit(cache=True, error_model="numpy", inline="always")
def _window_room(h, w, n):
    """Buffers for _window over (h, w, n)."""
    return np.empty((h, w, n)), np.empty((w, n)), np.empty((w, n))


@njit(cache=True, error_model="numpy", inline="always")
def _share(want, got):
    """want / got where both hold anything, else 0 (h, w, 1)."""
    out = np.zeros(want.shape)
    for y in range(want.shape[0]):
        for x in range(want.shape[1]):
            if want[y, x, 0] > 0.0 and got[y, x, 0] > 0.0:
                out[y, x, 0] = want[y, x, 0] / got[y, x, 0]
    return out


@njit(cache=True, error_model="numpy", inline="always")
def _balance(loss, gain, k, passes):
    """The Sinkhorn factors (a over the pixels that give, b over those that
    take; (h, w, 1)) of the plan a_i K_ij b_j between them (K: 1 within the
    square of half-width ``k``, else 0), and K b: each giver gives exactly
    ``loss`` (where any taker is within reach). Each taker gets about
    ``gain`` where the givers and takers near it can be balanced; where
    they can't, some get much more and some much less (see PULL_PASSES), and
    _settle_pulled evens that out through the pool."""
    h, w = loss.shape
    loss = loss.reshape((h, w, 1))
    gain = gain.reshape((h, w, 1))
    b = np.zeros((h, w, 1))
    for y in range(h):
        for x in range(w):
            if gain[y, x, 0] > 0.0:
                b[y, x, 0] = 1.0
    room = _window_room(h, w, 1)
    kb = np.empty((h, w, 1))
    ka = np.empty((h, w, 1))
    _window(b, k, kb, room)
    a = _share(loss, kb)
    for _ in range(passes):
        _window(a, k, ka, room)
        nb = _share(gain, ka)
        for y in range(h):
            for x in range(w):
                if nb[y, x, 0] > 0.0:
                    b[y, x, 0] = nb[y, x, 0]
        _window(b, k, kb, room)
        a = _share(loss, kb)
    return a, b, kb


@njit(cache=True, error_model="numpy", inline="always")
def _pulled(top, fac, loss, a, b, kb, k):
    """What each pixel that rises takes in along the plan (h, w, PACKED),
    and the pool (PACKED) of what had no taker within reach. Those that
    fall are left with ``fac`` of their paint."""
    h, w = loss.shape
    pool = np.zeros(PACKED)
    src = np.zeros((h, w, PACKED))
    for y in range(h):
        for x in range(w):
            if loss[y, x] > 0.0:
                v = np.float64(top[y, x, IV])
                for c in range(PACKED):
                    # (what is in a unit of this pixel's paint)
                    unit = np.float64(top[y, x, c]) / v
                    if kb[y, x, 0] > 0.0:
                        src[y, x, c] = unit * a[y, x, 0]
                    else:
                        pool[c] += unit * loss[y, x]
    got = np.empty((h, w, PACKED))
    # (along rows into src itself: each row's partial sums are taken before
    # it is written; then down columns, in place, into got)
    line_pre, line_suf = np.empty((w, PACKED)), np.empty((w, PACKED))
    _window_rows(src, src, _window_plan(w, k), line_pre, line_suf)
    _window_columns(src, got, _window_plan(h, k))
    for y in range(h):
        for x in range(w):
            for c in range(PACKED):
                got[y, x, c] *= b[y, x, 0]
            if loss[y, x] > 0.0:
                f = np.float32(fac[y, x])
                for c in range(PACKED):
                    top[y, x, c] = top[y, x, c] * f
    return got, pool


@njit(cache=True, error_model="numpy", inline="always")
def _settle_pulled(top, gain, got, pool):
    """Those that took in more than their gain give the rest back to the
    pool; the pool goes to those short of theirs, in proportion to what they
    lack (or to their gain, if none lacks any); then each pixel that rises
    keeps its paint and adds what it took in."""
    h, w = gain.shape
    short = np.zeros((h, w))
    lacking, gaining = 0.0, 0.0
    for y in range(h):
        for x in range(w):
            if gain[y, x] <= 0.0:
                continue
            gaining += gain[y, x]
            v = got[y, x, IV]
            if v > gain[y, x]:
                back = (v - gain[y, x]) / v
                for c in range(PACKED):
                    p = got[y, x, c] * back
                    pool[c] += p
                    got[y, x, c] -= p
            else:
                short[y, x] = gain[y, x] - v
                lacking += short[y, x]
    for y in range(h):
        for x in range(w):
            if gain[y, x] <= 0.0:
                continue
            part = short[y, x] / lacking if lacking > 0.0 else gain[y, x] / gaining
            for c in range(PACKED):
                take = got[y, x, c] + pool[c] * part
                top[y, x, c] += np.float32(take)
