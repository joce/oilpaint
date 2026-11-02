"""A mountain lake painted wet-on-wet, in the manner of Bob Ross.

An original composition built the way he worked on The Joy of Painting, in
one sitting, everything wet:

* the canvas coated with a thin film of Liquid White, so every colour after
  it blends softly;
* the sky in criss-cross strokes with a two-inch brush; clouds scrubbed in
  with little circles, their puffs bright on top, their bases melting into
  the blue;
* the mountain cut in with the palette knife, its snow laid on with almost
  no pressure so it breaks over the dark paint, and mist tapped in at its base;
* distant trees tapped in with a fan brush, evergreens built from the top
  down with the fan, happy little bushes tapped in dark and then highlighted;
* the lake brushed in, reflections pulled straight down and lightly across,
  banks laid in with the knife and the water's edge "cut" with a knife line.

Every mark is a ``canvas.stroke(...)``; numpy only decides where they go.

    python examples/bob_ross_style.py [output.png]
"""

import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from oilpaint import Brush, Canvas, Paint, brushes, palette as P, surfaces  # noqa: E402

W, H = 1200, 900
rng = np.random.default_rng(1983)

# ===========================================================================
# His palette (the nearest paints in oilpaint.palette, and simple mixes)
# ===========================================================================


def mix(*parts):
    return Paint.mix(*parts)


TITANIUM_WHITE = P.TITANIUM_WHITE
PHTHALO_BLUE = P.PHTHALO_BLUE
# (his phthalo blue mixed with white goes a clean, bright blue; ours greys, so
# brighten it with cerulean)
SKY_BLUE = mix((P.PHTHALO_BLUE, 1), (P.CERULEAN_BLUE, 2))
PRUSSIAN_BLUE = mix((P.PHTHALO_BLUE, 3), (P.IVORY_BLACK, 1))
MIDNIGHT_BLACK = P.IVORY_BLACK
PHTHALO_GREEN = P.PHTHALO_GREEN
SAP_GREEN = P.SAP_GREEN
ALIZARIN = P.ALIZARIN_CRIMSON
VAN_DYKE_BROWN = mix((P.BURNT_UMBER, 3), (P.IVORY_BLACK, 1))
DARK_SIENNA = mix((P.BURNT_SIENNA, 2), (P.BURNT_UMBER, 1))
CADMIUM_YELLOW = P.CADMIUM_YELLOW
YELLOW_OCHRE = P.YELLOW_OCHRE
INDIAN_YELLOW = mix((P.CADMIUM_YELLOW, 2), (P.CADMIUM_ORANGE, 1))
BRIGHT_RED = P.CADMIUM_RED
# Liquid White: white thinned to a creamy fluid that stays wet and blends
LIQUID_WHITE = mix((TITANIUM_WHITE, 1), (P.MEDIUM, 1.2)).with_body(0.15)

# mixes he makes on the palette
MOUNTAIN_DARK = mix((PRUSSIAN_BLUE, 2), (VAN_DYKE_BROWN, 1.5), (ALIZARIN, 1), (MIDNIGHT_BLACK, 0.3), (TITANIUM_WHITE, 1.6))
SNOW_LIT = mix((TITANIUM_WHITE, 20), (DARK_SIENNA, 0.3))
SNOW_SHADE = mix((TITANIUM_WHITE, 6), (PRUSSIAN_BLUE, 1), (MOUNTAIN_DARK, 0.4))
EVERGREEN_DARK = mix((PHTHALO_GREEN, 1), (PRUSSIAN_BLUE, 0.3), (VAN_DYKE_BROWN, 1.2), (MIDNIGHT_BLACK, 0.8),
                     (SAP_GREEN, 0.6))
BUSH_DARK = mix((SAP_GREEN, 2), (PHTHALO_GREEN, 0.5), (VAN_DYKE_BROWN, 1), (MIDNIGHT_BLACK, 0.4))
HI_GREEN = mix((SAP_GREEN, 1), (CADMIUM_YELLOW, 1.2))
HI_YELLOW = mix((CADMIUM_YELLOW, 2), (SAP_GREEN, 0.4))
HI_OCHRE = mix((YELLOW_OCHRE, 2), (INDIAN_YELLOW, 1))
HI_RED = mix((BRIGHT_RED, 1), (INDIAN_YELLOW, 1), (DARK_SIENNA, 0.5))

# ===========================================================================
# His brushes, at the scale of this canvas (about 0.5 mm per pixel)
# ===========================================================================
TWO_INCH = brushes.make("flat", 100, "hog", length=0.75, capacity=1.6, name="two-inch brush")
ONE_INCH = brushes.make("flat", 50, "hog", length=0.7, capacity=1.4, name="one-inch brush")
FAN = brushes.make("fan", 48, "hog", length=0.7, clumps=7, name="fan brush")
BLENDER = brushes.MOP.sized(90).with_(name="soft blender")
MIST_BRUSH = brushes.MOP.sized(120).with_(softness=0.8, name="soft blender, for mist")
LINER = brushes.RIGGER.sized(3).with_(name="liner brush")
ROUND_SMALL = brushes.make("round", 8, "synthetic", length=0.6, spread=0.2, name="small round")
CLOUD_BRUSH = brushes.make("round", 60, "squirrel", length=0.5, softness=0.85, name="cloud brush")
TOP_TAPPER = brushes.make("round", 20, "hog", length=0.5, softness=0.45, name="one-inch brush, corner")
TAPPER = brushes.make("round", 30, "hog", length=0.5, softness=0.25, name="brush corner, tapping")
EDGE_KNIFE = brushes.knife("straight", 40, flex=0.5, aspect=0.04, name="knife edge")
BANK_KNIFE = brushes.knife("straight", 40, flex=0.5, aspect=0.15, name="knife, banks")
KNIFE = brushes.knife("straight", 70, flex=0.5, name="painting knife")
SMALL_KNIFE = brushes.knife("straight", 40, flex=0.5, name="small painting knife")

# pre-primed cotton canvas, as he used: an irregular weave, primed so it drinks no oil
CANVAS = surfaces.COTTON_DUCK.with_(absorbency=0.0, tooth=0.5, relief=1.0, scale=3.5, irregularity=0.35,
                                    ground="#f6f4ee", name="primed cotton canvas")
canvas = Canvas(W, H, surface=CANVAS, seed=1983)
N_STROKES = [0]


def st(pts, paint, width=None, brush=None, **kw):
    N_STROKES[0] += 1
    pts = [tuple(map(float, p)) for p in np.atleast_2d(pts)]
    return canvas.stroke(pts, paint=paint, width=width, brush=brush, **kw)


def loaded(tool, paint, amount, across=None):
    return Brush(tool).load(paint, amount, across=across)


def stage(name):
    print(f"[{time.strftime('%H:%M:%S')}] {name}  (strokes so far: {N_STROKES[0]})", flush=True)


def ridge_fn(ctrl):
    ctrl = np.asarray(ctrl, float)
    return lambda x: np.interp(x, ctrl[:, 0], ctrl[:, 1])


# ===========================================================================
# Composition
# ===========================================================================
HORIZON = 545                 # the far shore
MAIN_RIDGE = [(90, 480), (200, 410), (300, 352), (360, 330), (430, 262), (488, 186), (512, 172), (540, 196),
              (575, 214), (612, 200), (650, 236), (700, 280), (770, 330), (860, 390), (960, 450), (1040, 490)]
BACK_RIDGE = [(-20, 430), (60, 400), (150, 372), (230, 392), (300, 420), (700, 430), (780, 372), (850, 330),
              (905, 302), (960, 330), (1040, 352), (1120, 380), (1220, 400)]
MOUNTAIN_BASE = 500


# ===========================================================================
# 1. Liquid White and the sky
# ===========================================================================
RELIEF, GLOSS = 0.25, 0.6      # how the finished canvas is lit


def thin(paint):
    """Paint thinned to a soft, buttery consistency for wet-on-wet work."""
    return mix((paint, 1), (P.MEDIUM, 0.15)).with_body(0.45)


def liquid_white():
    """A thin, even film over the whole canvas: long horizontal passes with the
    two-inch brush, each loaded with just enough to cover it."""
    stage("liquid white")
    for y in np.arange(30, H + 50, 55):
        b = loaded(TWO_INCH, LIQUID_WHITE, (W + 160) / 100 * 0.6)
        st([(-80, y), (W / 2, y + rng.normal(0, 2)), (W + 80, y)], None, brush=b, pressure=0.15)


def criss_cross(brush, y0, y1, x0=-60, x1=W + 60, size=110, pressure=0.55, reload=None, step=0.45):
    """Little X strokes across a band, as he scrubs in a sky or water,
    going back to the palette for a touch more paint after each one."""
    for y in np.arange(y0, y1, size * step):
        for x in np.arange(x0, x1, size * 0.8):
            jx, jy = rng.normal(0, size * 0.1, 2)
            d = size * 0.5
            st([(x + jx - d, y + jy - d * 0.55), (x + jx + d, y + jy + d * 0.55)], None, brush=brush,
               pressure=pressure)
            st([(x + jx - d, y + jy + d * 0.55), (x + jx + d, y + jy - d * 0.55)], None, brush=brush,
               pressure=pressure)
            if reload is not None:
                reload(brush, y)


def paint_sky():
    stage("sky")
    # phthalo blue scrubbed in with criss-cross strokes, darkest at the top; the
    # brush runs out as it works down, and the Liquid White lightens it
    def sky_blue(y):          # deep at the top, paling to a warm glow at the horizon
        t = float(np.clip((y + 30) / 520, 0, 1))
        return thin(mix((SKY_BLUE, 1.0), (TITANIUM_WHITE, 0.1 + 14 * t ** 1.8),
                        (ALIZARIN, 0.3 * max(0.0, t - 0.65))))

    def reload(br, y):
        br.load(sky_blue(y), 0.9 + 0.9 * max(0.0, 1 - (y + 30) / 250))

    b = loaded(TWO_INCH, sky_blue(-30), 1.2)
    criss_cross(b, -30, 500, size=130, reload=reload)
    # blend it all with a clean, dry brush: long, light horizontal strokes
    for y in np.arange(-10, 520, 50):
        st([(-80, y), (W / 2, y + rng.normal(0, 3)), (W + 80, y)], None, brush=Brush(BLENDER), pressure=0.12)
    # the water, with the same brush: horizontal strokes only, the colour of
    # the sky with a touch of green, light near the far shore, deeper below
    def water_blue(y):
        t = float(np.clip((y - 540) / (H - 540), 0, 1))
        return thin(mix((SKY_BLUE, 1.0), (PHTHALO_GREEN, 0.15 + 0.2 * t), (TITANIUM_WHITE, 6.0 * (1 - t) ** 2 + 0.2)))
    for y in np.arange(530, H + 40, 40):
        b = loaded(TWO_INCH, water_blue(y), (W + 160) / 100 * 0.8)
        st([(-80, y), (W / 2, y + rng.normal(0, 2)), (W + 80, y)], None, brush=b, pressure=0.5)
    for y in np.arange(530, H + 40, 45):
        st([(-80, y), (W / 2, y), (W + 80, y)], None, brush=Brush(BLENDER), pressure=0.12)


def cloud_top(puffs, base):
    """The top edge of a cloud: round puffs (x, y, radius) heaped on a flat base."""
    def top(x):
        x = np.asarray(x, float)
        t = np.full(x.shape, np.inf)
        for px, py, pr in puffs:
            d = pr ** 2 - (x - px) ** 2
            t = np.where(d > 0, np.minimum(t, py - np.sqrt(np.clip(d, 0, None))), t)
        return t
    return top


def scrub_path(a, b, rr, advance, r):
    """Little scrubbing circles of radius ``rr``, travelling from a to b."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    ln = np.linalg.norm(b - a)
    n = max(1, int(ln / advance))
    t = np.linspace(0, 2 * np.pi * n, 7 * n + 1)
    phase = r.uniform(0, 2 * np.pi)
    c = a + (b - a) * (t / t[-1])[:, None]
    return c + np.stack([rr * np.cos(t + phase), rr * 0.8 * np.sin(t + phase)], 1)


def scrub_rows(top, x0, x1, y_from, y_to, size, r, inset=0.48, ragged=0.0, base=None):
    """Scrub the inside of a cloud row after row, from ``y_from`` to ``y_to``,
    back and forth. ``ragged`` wobbles the rows near the base so it isn't ruled."""
    xs = np.arange(x0, x1, 2.0)
    step = size * 0.45
    pts, flip = [], False
    for y in np.arange(y_from, y_to, step if y_to > y_from else -step):
        y += r.normal(0, step * 0.1)
        inside = top(xs) + size * inset <= y
        if not inside.any():
            continue
        edges = np.flatnonzero(np.diff(np.r_[0, inside.astype(int), 0]))
        runs = [(xs[s] + size * inset * 0.3, xs[e - 1] - size * inset * 0.3) for s, e in zip(edges[::2], edges[1::2])]
        if flip:
            runs = [(b, a) for a, b in runs[::-1]]
        wob = ragged * size * np.clip(1 - (base - y) / size, 0, 1) if base is not None else 0.0
        for a, b in runs:
            knots = np.linspace(a, b, max(2, int(abs(b - a) / size) + 1))
            ys = y + r.normal(0, wob + 1e-9, len(knots))
            for k in range(len(knots) - 1):
                pts.extend(scrub_path((knots[k], ys[k]), (knots[k + 1], ys[k + 1]), size * 0.2, size * 0.35, r))
        flip = not flip
    return np.array(pts)


def path_length(pts):
    return float(np.linalg.norm(np.diff(pts, axis=0), axis=1).sum())


def cloud(puffs, base, size=50, seed=0):
    """A cumulus, the way he builds one: puffs of clean white with bright tops,
    heaped on a flat base that melts into the sky.

    Wet-on-wet, white scrubbed into the blue turns blue. So first he lifts
    some of the wet blue where the bright puffs will go, with a clean brush
    (wiped after every few circles). Then the white is scrubbed in with little
    circles in one long go, top down: where the blue was lifted it stays white;
    lower down, pressed a little harder into the blue, it turns a soft blue-grey
    and the base dissolves. Last, each puff's sunlit top is laid in with clean
    white, the ones at the back first, so each puff in front stands out
    against the shadowed underside of the one behind it."""
    r = np.random.default_rng(seed)
    top = cloud_top(puffs, base)
    x0 = min(px - pr for px, py, pr in puffs) - 10
    x1 = max(px + pr for px, py, pr in puffs) + 10
    y_top = min(py - pr for px, py, pr in puffs)
    lift_to = max(py for px, py, pr in puffs) + size * 0.3          # just below the lowest puff's middle
    # lift the blue: three passes, a clean brush every few circles
    for _ in range(3):
        path = scrub_rows(top, x0, x1, y_top, lift_to, size * 0.89, r, inset=0.34)
        for chunk in np.array_split(path, max(1, len(path) // 21)):
            if len(chunk) > 2:
                st(chunk, None, brush=Brush(CLOUD_BRUSH.sized(size * 0.8)), pressure=0.9, pickup=1.0)
    # the body: one long scrub from the top down. (One stroke, so the brush works
    # into its own wet white; a new stroke over it would leave a seam at its soft edge.)
    path = scrub_rows(top, x0, x1, y_top + size * 0.3, base - size * 0.2, size, r, ragged=0.25, base=base)
    below = np.clip((path[:, 1] - lift_to) / (base - lift_to), 0, 1)
    pressure = (0.35 + 0.25 * below ** 1.5) * (1 + 0.1 * np.sin(np.arange(len(path)) * 2 * np.pi / 14))
    total = 0.3 * path_length(path) / size
    b = Brush(CLOUD_BRUSH.sized(size)).load(thin(TITANIUM_WHITE), min(total, 12))
    if total > 12:      # more than the hair holds: the rest is worked up into the heel
        b.load(thin(TITANIUM_WHITE), total - 12, depth="heel")     # (pressing squeezes it out as he goes)
    st(path, None, brush=b, pressure=list(pressure), pickup=0.05)
    # the sunlit tops: each puff filled from its rim inwards with clean white,
    # with a firmer brush, so the tops stay crisp; the puffs at the back first
    w = size * 0.6
    for px, py, pr in sorted(puffs, key=lambda q: q[1]):
        path = []
        for k, R in enumerate(np.arange(pr - w * 0.45, pr * 0.35, -w * 0.35)):
            span = 0.8 * np.sqrt(R / pr)
            angles = np.linspace(np.pi * (1.5 - span / 2), np.pi * (1.5 + span / 2), 7)[::1 if k % 2 == 0 else -1]
            arc = [(px + R * np.cos(a), py + R * np.sin(a)) for a in angles]
            for p, q in zip(arc[:-1], arc[1:]):
                path.extend(scrub_path(p, q, w * 0.15, w * 0.3, r))
        path = np.array(path)
        b = loaded(CLOUD_BRUSH.sized(w).with_(softness=0.1), thin(TITANIUM_WHITE), 1.6 * path_length(path) / w)
        st(path, None, brush=b, pressure=0.3, pickup=0.03)


def wisp(x0, x1, y, width, seed=0):
    """A thin streak of cloud, low in the sky: a little white on a very soft
    brush, pulled across barely touching, lowered in and lifted off at the ends."""
    r = np.random.default_rng(seed)
    path = np.stack([np.linspace(x0, x1, 5), y + r.normal(0, width * 0.03, 5)], 1)
    b = loaded(CLOUD_BRUSH.sized(width).with_(softness=0.95), thin(TITANIUM_WHITE), 0.25 * path_length(path) / width)
    st(path, None, brush=b, pressure=lambda u: 0.15 * np.sin(np.pi * u) ** 0.7, pickup=0.15, taper=0.3)


# puffs (x, y, radius) of each cloud, and its base
BIG_CLOUD = ([(690, 182, 44), (748, 146, 56), (812, 112, 64), (880, 100, 62), (948, 126, 58), (1012, 156, 50),
              (1064, 188, 38)], 228)
LEFT_CLOUD = ([(205, 152, 38), (262, 126, 50), (325, 112, 54), (388, 130, 46), (436, 156, 30)], 186)
SMALL_CLOUD = ([(345, 284, 22), (378, 274, 28), (412, 288, 18)], 302)


def paint_clouds():
    stage("clouds")
    cloud(*BIG_CLOUD, size=50, seed=1)
    cloud(*LEFT_CLOUD, size=42, seed=2)
    cloud(*SMALL_CLOUD, size=28, seed=3)
    # and a few thin streaks low in the sky, just above the far mountains
    wisp(740, 970, 290, 46, seed=4)
    wisp(850, 1040, 262, 34, seed=5)
    wisp(160, 330, 308, 40, seed=6)


# ===========================================================================
# 2. The mountain: knife work
# ===========================================================================
def ridge_segments(ridge_ctrl, step):
    """Points along a ridge line every ``step`` px, with the ridge's slope
    there as a brush angle on the canvas (degrees, positive rising to the right)."""
    ctrl = np.asarray(ridge_ctrl, float)
    d = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(ctrl, axis=0), axis=1))])
    out = []
    for t in np.arange(0.0, d[-1], step):
        i = min(int(np.searchsorted(d, t, side="right")) - 1, len(ctrl) - 2)
        (x0, y0), (x1, y1) = ctrl[i], ctrl[i + 1]
        f = (t - d[i]) / max(d[i + 1] - d[i], 1e-9)
        # (the canvas measures a fixed brush angle the other way round from image y)
        out.append((x0 + f * (x1 - x0), y0 + f * (y1 - y0), -float(np.degrees(np.arctan2(y1 - y0, x1 - x0)))))
    return out


def knife_mountain(ridge_ctrl, base, dark, seed=0):
    """Cut in with the knife: strokes pulled down from the ridge, the blade
    laid along the slope so its edge leaves a crisp silhouette; each stroke
    thins out as the knife runs low towards the base."""
    r = np.random.default_rng(seed)
    ridge = ridge_fn(ridge_ctrl)
    # first the body of the mountain in a pale, hazy version of the colour,
    # brushed across with the two-inch brush: the lower slopes lost in mist
    ctrl = np.asarray(ridge_ctrl, float)
    haze = thin(mix((dark, 1), (TITANIUM_WHITE, 3.5), (SKY_BLUE, 0.4)))
    for y in np.arange(ctrl[:, 1].min() + 40, base + 10, 16):
        inside = ctrl[:, 1] <= y - 12
        if inside.sum() < 2:
            continue
        xa, xb = ctrl[inside, 0].min(), ctrl[inside, 0].max()
        # the brush travels along its length: a band no taller than the ridge allows
        b = loaded(TWO_INCH, haze, max(0.5, (xb - xa) / 100 * 0.5))
        st([(xa + 15, y), ((xa + xb) / 2, y + r.normal(0, 2)), (xb - 15, y)], None, brush=b, angle=90, pressure=0.35,
           taper=0.1)
    segs = ridge_segments(ridge_ctrl, 22)
    lower = mix((dark, 1), (TITANIUM_WHITE, 1.2))        # lighter towards the base, as it mixes with the white
    for j, (x, y, slope) in enumerate(segs):
        if base - y < 8:
            continue
        ln = (base - y) * r.uniform(0.45, 0.75)
        k = Brush(KNIFE).load(dark, 0.5 * ln / KNIFE.width).load(lower, 0.4 * ln / KNIFE.width, depth="heel")
        # near a peak, the blade is held on the side of the slope it is on and
        # never pokes above the ridge line: shorten it to fit under the ridge
        lo, hi = max(0, j - 2), min(len(segs) - 1, j + 2)
        bend = abs(segs[hi][2] - segs[lo][2])
        width = KNIFE.width * (1.0 if bend < 25 else 0.45)
        y0 = y + 0.5 * KNIFE.aspect * width + (3 if bend >= 25 else 0)
        st([(x, y0), (x + r.normal(0, 3), y0 + ln * 0.5), (x + r.normal(0, 5), y0 + ln)], None, brush=k,
           width=width, angle=slope, follow_path=False, pressure=0.65, lift=0.0)
    return ridge


def snow(ridge_ctrl, base, lit, shade, seed=0, amount=0.22, pressure=0.1):
    """Highlights: the knife edge laid along the ridge and pulled down the
    face, barely touching, with very little paint, so it breaks over the dark.
    The lit colour on faces turned to the light (the ridge rising to the
    right: the face looks left), the shadow colour on those turned away."""
    r = np.random.default_rng(seed)
    for x, y, slope in ridge_segments(ridge_ctrl, 9):
        if base - y < 25:
            continue
        rising = slope > 0
        paint = lit if rising else shade
        ln = (base - y) * r.uniform(0.3, 0.7)
        # down the face and out, away from the ridge
        a = np.radians(slope + (-90 if rising else 90) * 0.0)
        fall = np.array([(-0.55 if rising else 0.55), 1.0])
        fall /= np.linalg.norm(fall)
        k = Brush(SMALL_KNIFE).load(paint, amount * r.uniform(0.7, 1.4))
        y0 = y + 6
        st([(x, y0), (x + fall[0] * ln * 0.5, y0 + fall[1] * ln * 0.5), (x + fall[0] * ln, y0 + fall[1] * ln)], None,
           brush=k, angle=slope, follow_path=False, pressure=[pressure, pressure * 0.5, 0.02], lift=0.0)


def paint_mountains():
    stage("mountains")
    knife_mountain(BACK_RIDGE, MOUNTAIN_BASE, mix((MOUNTAIN_DARK, 1), (TITANIUM_WHITE, 1.8), (SKY_BLUE, 0.4)), seed=1)
    snow(BACK_RIDGE, MOUNTAIN_BASE, mix((SNOW_LIT, 3), (SNOW_SHADE, 1)), mix((SNOW_SHADE, 2), (TITANIUM_WHITE, 1)),
         seed=2)
    knife_mountain(MAIN_RIDGE, MOUNTAIN_BASE, MOUNTAIN_DARK, seed=3)
    snow(MAIN_RIDGE, MOUNTAIN_BASE, SNOW_LIT, SNOW_SHADE, seed=4)


# ===========================================================================
# 3. Land, trees and bushes
# ===========================================================================
FAR_SHORE = 548
# the shorelines of the land on each side; the water lies between them
LEFT_SHORE = np.array([(-20, 610), (120, 628), (250, 660), (360, 712), (430, 780), (455, 850), (440, 930)])
RIGHT_SHORE = np.array([(1220, 600), (1090, 622), (960, 662), (870, 720), (815, 800), (805, 870), (830, 930)])
EVERGREENS = [(60, 700, 600, 1.0), (165, 672, 440, 0.85), (250, 690, 300, 0.7), (1150, 690, 640, 1.05),
              (1055, 668, 430, 0.85), (975, 690, 290, 0.7)]


def shore_x(shore, y):
    return float(np.interp(y, shore[:, 1], shore[:, 0]))


def far_trees():
    """The distant tree line: a misty blue-green, tapped in with the corner of
    the fan brush as little vertical strokes, lighter at the bottom where it
    meets the mist over the water."""
    r = np.random.default_rng(11)
    for layer, (col, top) in enumerate(((mix((EVERGREEN_DARK, 1), (TITANIUM_WHITE, 0.7), (SKY_BLUE, 0.5)), 60),
                                        (mix((EVERGREEN_DARK, 1), (TITANIUM_WHITE, 0.2), (SKY_BLUE, 0.15)), 36))):
        for x in np.arange(-10, W + 10, 6 if layer else 8):
            h = r.uniform(0.35, 1.0) * top
            yb = FAR_SHORE + 4 - layer * 2
            # each little tree: a tapering stroke, pointed at the top
            st([(x, yb), (x + r.normal(0, 1), yb - h * 0.5), (x + r.normal(0, 1), yb - h)], col,
               width=[r.uniform(7, 11), 4, 0.8], brush=ROUND_SMALL, amount=16, pressure=0.5, mixing=0.0, pickup=0.0)
        # the base dissolves into mist: a clean brush along it, barely touching
        st([(-60, FAR_SHORE - 2), (W / 2, FAR_SHORE), (W + 60, FAR_SHORE - 2)], None, brush=Brush(BLENDER.sized(30)),
           pressure=0.15)


def evergreen(x, base, h, scale=1.0, seed=0):
    """An evergreen the way he paints them: a trunk line with the knife, then
    the fan brush loaded with dark green, starting at the top with a touch of
    the corner and pressing harder as it works down, the branches growing
    wider, drooping a little; then highlights on one side, barely touching."""
    r = np.random.default_rng(seed)
    st([(x, base), (x + r.normal(0, 2), base - h)], VAN_DYKE_BROWN, width=3 * scale, brush=LINER.sized(4 * scale),
       amount=60, pressure=0.4)
    y = base - h
    k = 0
    while y < base - 0.04 * h:
        t = (y - (base - h)) / h           # 0 at the top
        span = (4 + t * 0.3 * h) * r.uniform(0.75, 1.25)
        for side in (-1, 1):
            if r.random() < 0.1 and t > 0.1:
                continue                   # a gap: light shows through the tree
            for j in range(2):             # a pull from the trunk out and down, and one at the tip
                a0 = j * 0.45 * span
                ln = span * (1.0 if j == 0 else 0.6) * r.uniform(0.8, 1.1)
                droop = np.radians(r.uniform(18, 38))
                p0 = (x + side * a0, y + a0 * 0.3 + r.normal(0, 1.5))
                p1 = (p0[0] + side * ln * np.cos(droop), p0[1] + ln * np.sin(droop))
                st([p0, ((p0[0] + p1[0]) / 2, (p0[1] + p1[1]) / 2 + r.normal(0, 1.5)), p1], EVERGREEN_DARK,
                   width=[(9 + 16 * t) * scale, (4 + 6 * t) * scale], brush=FAN, angle=0, amount=10,
                   pressure=0.45 + 0.35 * t, mixing=0.04, pickup=0.05)
        y += (5 + 7 * t) * scale * r.uniform(0.6, 1.4)
        k += 1
    rows = k
    # highlights: lighter green on the side towards the light, at the ends of
    # the branches, barely touching
    hi = mix((EVERGREEN_DARK, 1), (HI_GREEN, 1.8))
    for k in range(int(rows * 0.7)):
        t = 0.15 + 0.85 * k / (rows * 0.7)
        y = base - h + t * h * 0.93
        span = (3 + t * 0.3 * h) * r.uniform(0.6, 1.0)
        side = -1 if r.random() < 0.8 else 1
        p0 = (x + side * span * r.uniform(0.3, 0.6), y + span * 0.12)
        ln = span * r.uniform(0.3, 0.55)
        p1 = (p0[0] + side * ln * 0.85, p0[1] + ln * 0.45)
        st([p0, p1], hi, width=[(6 + 10 * t) * scale, 3 * scale], brush=FAN, angle=0, amount=3, pressure=0.15,
           mixing=0.04, pickup=0.05)


def reflect(x0, x1, y, depth, colour, seed=0):
    """A reflection: paint along the shore, pulled straight down with the
    two-inch brush, then brushed very lightly across."""
    r = np.random.default_rng(seed)
    for x in np.arange(x0, x1, 40):
        b = Brush(TWO_INCH).load(colour, 1.0 * depth / TWO_INCH.width + 0.5)
        d = depth * r.uniform(0.7, 1.0)
        st([(x, y - 5), (x + r.normal(0, 2), y + d * 0.5), (x + r.normal(0, 2), y + d)], None, brush=b, angle=0,
           pressure=[0.5, 0.35, 0.05], mixing=0.3)
    # (the brush travels along its length, so it only touches a narrow band)
    for yy in np.arange(y + 14, y + depth, 12):
        st([(x0 - 30, yy), ((x0 + x1) / 2, yy + r.normal(0, 1)), (x1 + 30, yy)], None, brush=Brush(TWO_INCH),
           angle=90, pressure=0.08)


def paint_far_land():
    stage("distant trees")
    far_trees()
    # their reflection in the still water
    reflect(-20, W + 20, FAR_SHORE, 40, mix((EVERGREEN_DARK, 1), (TITANIUM_WHITE, 2.2), (SKY_BLUE, 0.5)), seed=12)
    # the water's edge: a light line cut along the far shore with the knife edge
    edge_line([(-20, FAR_SHORE + 2), (W / 2, FAR_SHORE + 3), (W + 20, FAR_SHORE + 2)], seed=13, pressure=0.3)


LAND_DARK = mix((VAN_DYKE_BROWN, 2), (SAP_GREEN, 1), (PHTHALO_GREEN, 0.3), (MIDNIGHT_BLACK, 0.4))


def land_mass(shore, side, seed=0):
    """The dark land on one side, tapped in from the shore outward to the edge
    of the canvas (the bushes will cover most of it)."""
    r = np.random.default_rng(seed)
    for y in np.arange(shore[:, 1].min() - 30, H + 30, 15):
        xs = shore_x(shore, y)
        x_end = -40 if side < 0 else W + 40
        for x in np.arange(xs, x_end, side * 20):
            q = (x + r.normal(0, 6), y + r.normal(0, 6))
            st([q], thin(LAND_DARK), width=r.uniform(32, 44), brush=TAPPER, amount=3, pressure=0.3, mixing=0.03,
               pickup=0.03)


def paint_land():
    stage("land on either side")
    land_mass(LEFT_SHORE, -1, seed=30)
    land_mass(RIGHT_SHORE, 1, seed=31)


def paint_evergreens():
    stage("evergreens")
    for k, (x, base, h, sc) in enumerate(EVERGREENS):
        evergreen(x, base, h, sc, seed=20 + k)


def bush(cx, cy, w, h, highlights, seed=0):
    """A happy little bush: tapped in dark with the corner of a brush, then
    the highlights tapped on top, barely touching, so they break and let the
    dark show through: the top and the side towards the light."""
    r = np.random.default_rng(seed)
    for k in range(int(w * h / 70) + 4):
        a = r.uniform(0, 2 * np.pi)
        rr = np.sqrt(r.uniform(0, 1))
        q = (cx + rr * 0.5 * w * np.cos(a), cy - h / 2 + rr * 0.5 * h * np.sin(a))
        edge = rr > 0.75                   # the outline breaks up into leaves
        st([q], LAND_DARK, width=r.uniform(9, 16), brush=TAPPER, amount=4, pressure=0.25 if edge else 0.45,
           mixing=0.05, pickup=0.05)
    for k in range(int(w * h / 45) + 4):
        a = r.uniform(np.pi * 0.95, np.pi * 2.05)          # the upper half, towards the light
        rr = np.sqrt(r.uniform(0.15, 1))
        q = (cx + rr * 0.5 * w * np.cos(a) - 0.12 * w, cy - h / 2 + rr * 0.5 * h * np.sin(a))
        col = highlights[int(r.integers(len(highlights)))]
        st([q], col, width=r.uniform(6, 12), brush=TAPPER, amount=2, pressure=r.uniform(0.12, 0.25), mixing=0.03,
           pickup=0.02)


def paint_bushes():
    stage("bushes")
    r = np.random.default_rng(40)
    greens = [HI_GREEN, HI_YELLOW, mix((HI_GREEN, 1), (HI_YELLOW, 1))]
    autumn = [HI_OCHRE, HI_YELLOW, HI_RED, HI_GREEN]
    for shore, side in ((LEFT_SHORE, -1), (RIGHT_SHORE, 1)):
        for y in np.arange(shore[:, 1].min() + 20, H + 40, 30):
            xs = shore_x(shore, y)
            for k in range(7):
                w = r.uniform(70, 140)
                x = xs + side * (w * 0.35 + k * w * 0.7) + r.normal(0, 10)
                if x < -60 or x > W + 60:
                    break
                bush(x, y + r.uniform(0, 12), w, w * r.uniform(0.45, 0.7),
                     autumn if r.random() < 0.3 else greens, seed=int(r.integers(1 << 30)))


BANK_DARK = mix((VAN_DYKE_BROWN, 2), (DARK_SIENNA, 1))
BANK_LIGHT = mix((VAN_DYKE_BROWN, 1), (TITANIUM_WHITE, 2.5), (DARK_SIENNA, 0.3))


def edge_line(pts, seed=0, pressure=0.3, amount=2.5):
    """A water line: the edge of the knife loaded with a little Liquid White,
    cut across the water."""
    r = np.random.default_rng(seed)
    k = Brush(EDGE_KNIFE).load(mix((TITANIUM_WHITE, 3), (SKY_BLUE, 0.4)), amount)
    pts = np.asarray(pts, float)
    pts = pts + np.stack([np.zeros(len(pts)), r.normal(0, 0.6, len(pts))], -1)
    st(pts, None, brush=k, angle=90, pressure=pressure, lift=0.0)


def bank(shore, side, seed=0):
    """The banks where land meets water: the knife laid flat with dark brown,
    pulled along the shore; then highlights skimmed on, barely touching; and
    a water line cut along the edge."""
    r = np.random.default_rng(seed)
    ys = np.arange(shore[:, 1].min() + 40, shore[:, 1].max() - 60, 75)
    for y in ys:
        y = y + r.normal(0, 10)
        x = shore_x(shore, y)
        ln = r.uniform(35, 70)
        # a thin strip of brown pulled along the water's edge, the knife held
        # along its length
        pts = [(x + side * ln, y - r.uniform(2, 6)), (x + side * ln * 0.5, y + r.normal(0, 1.5)), (x + 6 * side * -1, y + 2)]
        st(pts, BANK_DARK, brush=BANK_KNIFE, amount=4, angle=90, pressure=0.55)
        st(pts, BANK_LIGHT, brush=BANK_KNIFE, amount=0.6, angle=90, pressure=0.08, lift=0.0)
        edge_line([(x - side * 4, y + 6), (x - side * r.uniform(15, 40), y + 6 + r.normal(0, 0.8))], seed=int(y),
                  pressure=0.18, amount=1.2)


def paint_banks():
    stage("banks and water lines")
    bank(LEFT_SHORE, -1, seed=50)
    bank(RIGHT_SHORE, 1, seed=51)
    r = np.random.default_rng(52)
    for k in range(9):                     # a few more water lines across the open water
        y = r.uniform(FAR_SHORE + 40, H - 60)
        x0 = r.uniform(shore_x(LEFT_SHORE, y) + 30, W / 2)
        ln = r.uniform(30, 90)
        if x0 + ln > shore_x(RIGHT_SHORE, y) - 20:
            continue
        edge_line([(x0, y), (x0 + ln / 2, y + r.normal(0, 0.5)), (x0 + ln, y)], seed=60 + k, pressure=0.15,
                  amount=1.2)


STAGES = ["liquid_white", "paint_sky", "paint_clouds", "paint_mountains", "paint_far_land", "paint_land",
          "paint_evergreens", "paint_banks", "paint_bushes"]


def main(out, upto=None):
    t0 = time.time()
    for name in STAGES[:upto]:
        globals()[name]()
    canvas.save(out, relief=RELIEF, gloss=GLOSS)
    print(f"wrote {out}: {N_STROKES[0]} strokes in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(__file__), "output", "bob_ross_style.png"),
         int(sys.argv[2]) if len(sys.argv) > 2 else None)
