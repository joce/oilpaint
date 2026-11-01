"""A winter village in the manner of Pieter Bruegel the Elder.

An original composition: a high viewpoint from a snowy hillside of bare
trees, woodgatherers hauling a sledge of faggots, a valley with a frozen
river and mill pond full of skaters, a village with its inn, mill and church,
and Alpine crags in a green-grey sky. Painted entirely with the oilpaint API:
every mark is a `canvas.stroke(...)` with paints mixed from
`oilpaint.palette` via `Paint.mix`; numpy only computes where the strokes go.
The picture is built up like a panel, back to front, letting layers dry.

    python examples/bruegel_winter.py [output.png]
"""

import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from oilpaint import Canvas, Paint, palette as P  # noqa: E402

W, H = 1500, 1060
rng = np.random.default_rng(1565)


def pick(seq):
    return seq[int(rng.integers(len(seq)))]


# ===========================================================================
# Palette: lead white, bone black, the earths, a copper green, some vermilion
# ===========================================================================
WHITE, BLACK = P.TITANIUM_WHITE, P.IVORY_BLACK
RAW_UMBER, BURNT_UMBER, OCHRE, SIENNA = P.RAW_UMBER, P.BURNT_UMBER, P.YELLOW_OCHRE, P.BURNT_SIENNA
GREEN, BLUE, RED, MEDIUM = P.VIRIDIAN, P.ULTRAMARINE_BLUE, P.CADMIUM_RED, P.MEDIUM


def mix(*parts):
    return Paint.mix(*parts)


SKY_TOP = mix((WHITE, 1), (GREEN, 2.1), (BLACK, 0.31), (RAW_UMBER, 0.32))
SKY_MID = mix((WHITE, 1), (GREEN, 0.92), (BLACK, 0.11), (RAW_UMBER, 0.28))
SKY_LOW = mix((WHITE, 1), (GREEN, 0.22), (BLACK, 0.054), (OCHRE, 0.058))
SKY_GLOW = mix((WHITE, 1), (GREEN, 0.12), (OCHRE, 0.12), (BLACK, 0.03))
ICE = mix((WHITE, 1), (GREEN, 1.091), (BLACK, 0.125), (RAW_UMBER, 0.076), (OCHRE, 0.05))
ICE_DARK = mix((WHITE, 1), (GREEN, 2.54), (BLACK, 0.323), (RAW_UMBER, 0.043), (OCHRE, 0.101))
ICE_LIGHT = mix((WHITE, 1), (GREEN, 0.444), (BLACK, 0.054), (RAW_UMBER, 0.027), (OCHRE, 0.028))
SNOW = mix((WHITE, 10), (OCHRE, 0.06), (RAW_UMBER, 0.03))
SNOW_WARM = mix((WHITE, 10), (OCHRE, 0.18), (RAW_UMBER, 0.03))
SNOW_FAR = mix((SNOW, 3), (SKY_LOW, 1))
SNOW_SHADOW = mix((WHITE, 1), (BLUE, 0.004), (BLACK, 0.053), (GREEN, 0.108))
SNOW_HALF = mix((SNOW, 1), (SNOW_SHADOW, 1))
SNOW_DEEP = mix((SNOW_SHADOW, 2), (ICE_DARK, 1))
ROCK = mix((WHITE, 1), (BLACK, 0.29), (GREEN, 0.92), (RAW_UMBER, 0.15))
ROCK_DARK = mix((ROCK, 2), (BLACK, 0.25), (RAW_UMBER, 0.3))
ROCK_LIGHT = mix((ROCK, 1), (WHITE, 1.5))
HAZE = mix((ROCK_LIGHT, 1), (SKY_LOW, 1))
MID_TREES = mix((BURNT_UMBER, 1), (BLACK, 0.6), (WHITE, 0.5), (GREEN, 0.3))
TRUNK = mix((BURNT_UMBER, 1), (BLACK, 2.2))
BRAMBLE = mix((SIENNA, 1), (OCHRE, 0.6), (P.CADMIUM_ORANGE, 0.2))
TRUNK_LIGHT = mix((BURNT_UMBER, 1), (BLACK, 0.3), (WHITE, 0.45), (OCHRE, 0.2))
TWIG = mix((BURNT_UMBER, 1), (BLACK, 0.8), (SIENNA, 0.2))
WILLOW = mix((SIENNA, 1), (BURNT_UMBER, 0.8), (OCHRE, 0.3))
REED = mix((OCHRE, 1), (RAW_UMBER, 0.8), (WHITE, 0.3))
WALL = mix((OCHRE, 1), (BURNT_UMBER, 0.96), (WHITE, 0.53), (RAW_UMBER, 0.07))
PLASTER = mix((WHITE, 3), (OCHRE, 0.3), (RAW_UMBER, 0.15))
BRICK = mix((SIENNA, 1), (RED, 0.25), (BURNT_UMBER, 0.35))
BRICK_SHADE = mix((BRICK, 1), (BURNT_UMBER, 0.6), (BLACK, 0.2))
STONE = mix((WHITE, 2), (RAW_UMBER, 0.5), (BLACK, 0.15), (OCHRE, 0.1))
STONE_SHADE = mix((STONE, 2), (BLACK, 0.3), (RAW_UMBER, 0.3))
THATCH = mix((OCHRE, 1), (RAW_UMBER, 0.9), (BURNT_UMBER, 0.3))
SLATE = mix((BLACK, 1), (WHITE, 0.8), (GREEN, 0.3), (BLUE, 0.15))
WOOD = mix((BURNT_UMBER, 1), (OCHRE, 0.4), (WHITE, 0.2))
DARK = mix((BURNT_UMBER, 1), (BLACK, 0.8))
NEAR_BLACK = mix((BLACK, 1), (BURNT_UMBER, 0.3))
WINDOW = mix((BLACK, 1), (BURNT_UMBER, 0.5), (GREEN, 0.1))
C_BROWN = mix((BURNT_UMBER, 1), (BLACK, 0.5), (OCHRE, 0.15))
C_TAN = mix((OCHRE, 1), (RAW_UMBER, 0.7), (WHITE, 0.4))
C_GREY = mix((BLACK, 1), (WHITE, 0.8), (RAW_UMBER, 0.4))
C_BLUE = mix((BLUE, 1), (BLACK, 0.4), (WHITE, 0.9), (GREEN, 0.1))
C_RED = mix((RED, 1), (SIENNA, 0.2), (BURNT_UMBER, 0.08))
C_DULL_RED = mix((RED, 1), (BURNT_UMBER, 0.6))
C_GREEN = mix((GREEN, 1), (RAW_UMBER, 0.8), (WHITE, 0.4), (BLACK, 0.2))
C_WHITE = mix((WHITE, 4), (OCHRE, 0.15), (RAW_UMBER, 0.1))
C_BLACK = mix((BLACK, 1), (RAW_UMBER, 0.2))
FLESH = mix((WHITE, 1), (SIENNA, 0.28), (OCHRE, 0.12), (RED, 0.03))
C_PINK = mix((WHITE, 1), (RED, 0.25), (OCHRE, 0.2), (SIENNA, 0.1))
C_OLIVE = mix((OCHRE, 1), (GREEN, 0.4), (BLACK, 0.3), (WHITE, 0.3))
COATS = [C_BROWN, C_BROWN, C_TAN, C_GREY, C_BLUE, C_RED, C_DULL_RED, C_GREEN, C_BLACK, C_BLACK, DARK, C_PINK,
         C_OLIVE]
SKIRTS = [C_BLACK, C_BLUE, C_DULL_RED, C_BROWN, C_GREY, C_GREEN]
FIRE = mix((P.CADMIUM_ORANGE, 1), (P.CADMIUM_YELLOW, 0.6))
FLAME_CORE = mix((P.CADMIUM_YELLOW, 1), (WHITE, 0.7))
EMBER = mix((RED, 1), (P.CADMIUM_ORANGE, 0.5))
SMOKE = mix((WHITE, 1), (BLACK, 0.14), (RAW_UMBER, 0.12), (MEDIUM, 5))
DOG_BROWN = mix((BURNT_UMBER, 1), (OCHRE, 0.35), (BLACK, 0.3))
HORSE = mix((BURNT_UMBER, 1), (SIENNA, 0.4), (BLACK, 0.3))
EARTH = mix((RAW_UMBER, 1), (BURNT_UMBER, 0.5), (WHITE, 0.25))

# ===========================================================================
# Low-level helpers. Everything ends up as canvas.stroke(...)
# ===========================================================================
canvas = Canvas(W, H, ground="#efe8d8", texture=0.25, weave=2.5, seed=1565)
N_STROKES = [0]


def length(pts):
    pts = np.asarray(pts, float)
    return float(np.linalg.norm(np.diff(pts, axis=0), axis=1).sum()) if len(pts) > 1 else 0.0


def st(pts, paint, width, amount=None, pressure=0.35, **kw):
    """One stroke; by default the brush holds enough paint for the whole path."""
    pts = [tuple(map(float, p)) for p in pts]
    w = float(np.mean(width)) if np.ndim(width) else float(width)
    if amount is None:
        amount = 2.5 + 1.25 * length(pts) / max(w, 1.0)
    N_STROKES[0] += 1
    return canvas.stroke(pts, paint=paint, amount=amount, width=width, pressure=pressure, **kw)


def dab(x, y, paint, size, **kw):
    st([(x, y), (x + 0.01, y)], paint, size, **kw)


def scan_segments(poly, y):
    poly = np.asarray(poly, float)
    xs = []
    for (x0, y0), (x1, y1) in zip(poly, np.roll(poly, -1, axis=0)):
        if (y0 <= y < y1) or (y1 <= y < y0):
            xs.append(x0 + (y - y0) * (x1 - x0) / (y1 - y0))
    xs.sort()
    return list(zip(xs[0::2], xs[1::2]))


def inside(poly, x, y):
    return any(a <= x <= b for a, b in scan_segments(poly, y))


def fill(poly, paint_at, width, spacing, angle=0.0, seg_len=(120, 220), wobble=2.0, overrun=0.3,
         pressure=0.4, **kw):
    """Cover a polygon with roughly parallel strokes at `angle` (radians).
    `paint_at` is a Paint or a function (x, y) -> Paint."""
    get = paint_at if callable(paint_at) else (lambda x, y: paint_at)
    c, s = np.cos(angle), np.sin(angle)
    rot = np.array([[c, s], [-s, c]])
    local = np.asarray(poly, float) @ rot.T
    lo, hi = local[:, 1].min(), local[:, 1].max()
    ys = [0.5 * (lo + hi)] if hi - lo < spacing else np.arange(lo + spacing / 2, hi, spacing)
    for yl in ys:
        for xa, xb in scan_segments(local, yl):
            xa -= overrun * width
            xb += overrun * width
            if xb - xa < 0.5:
                continue
            x = xa - rng.uniform(0, seg_len[0] * 0.5)
            while x < xb:
                ln = rng.uniform(*seg_len)
                x0, x1 = max(x, xa), min(x + ln, xb)
                if x1 - x0 > 0.3:
                    n = max(2, int((x1 - x0) / 40) + 2)
                    xx = np.linspace(x0, x1, n)
                    yy = yl + rng.normal(0, wobble, n)
                    world = np.stack([xx, yy], -1) @ rot
                    mid = world[len(world) // 2]
                    st(world, get(*mid), width * rng.uniform(0.9, 1.1), pressure=pressure, **kw)
                x += ln * rng.uniform(0.55, 0.8)


def blob(cx, cy, rx, ry, n=48, rough=0.08, seed=None, smooth=2):
    r = np.random.default_rng(seed)
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    k = 1 + rough * (r.random(n) - 0.5) * 2
    for _ in range(smooth):
        k = (k + np.roll(k, 1) + np.roll(k, -1)) / 3
    return np.stack([cx + rx * k * np.cos(t), cy + ry * k * np.sin(t)], -1)


def ribbon(center, widths):
    """Polygon of a band of varying width around a centre line, plus its two edges."""
    c = np.asarray(center, float)
    w = np.asarray(widths, float)
    tan = np.gradient(c, axis=0)
    tan /= np.linalg.norm(tan, axis=1, keepdims=True)
    nrm = np.stack([-tan[:, 1], tan[:, 0]], -1)
    left = c + nrm * w[:, None] / 2
    right = c - nrm * w[:, None] / 2
    return np.vstack([left, right[::-1]]), left, right


def resample(pts, n):
    pts = np.asarray(pts, float)
    d = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))])
    t = np.linspace(0, d[-1], n)
    return np.stack([np.interp(t, d, pts[:, 0]), np.interp(t, d, pts[:, 1])], -1)


def lerp_paint(a, b, t):
    t = float(np.clip(t, 0, 1))
    if t <= 0.001:
        return a
    if t >= 0.999:
        return b
    return mix((a, 1 - t), (b, t))


def ramp(keys, t):
    t = float(np.clip(t, 0, 1)) * (len(keys) - 1)
    i = min(int(t), len(keys) - 2)
    return lerp_paint(keys[i], keys[i + 1], t - i)


def shade_of(p, k=0.5):
    return mix((p, 1), (DARK, k))


def light_of(p, k=0.5):
    return mix((p, 1), (WHITE, k))


HAZE_TARGET = mix((SKY_LOW, 2), (SNOW_FAR, 1))


def atmos(p, depth):
    """Fade a paint toward the haze of the distance (depth 0 = near, 1 = horizon)."""
    return lerp_paint(p, HAZE_TARGET, 0.75 * float(np.clip(depth, 0, 1)) ** 1.2)


def stage(name):
    print(f"[{time.strftime('%H:%M:%S')}] {name}  (strokes so far: {N_STROKES[0]})", flush=True)


# ===========================================================================
# Geometry of the scene
# ===========================================================================
HORIZON = 372


def depth_at(y):
    """0 at the bottom of the picture, 1 at the far edge of the valley."""
    return float(np.clip((H - y) / (H - HORIZON), 0, 1))


def scale_at(y):
    """Height in px of a standing figure on the valley floor at y."""
    return 6 + 92 * (1 - depth_at(y)) ** 1.6


CRAG_CTRL = [(-30, 300), (20, 212), (55, 240), (105, 150), (138, 188), (178, 170), (214, 232), (262, 136),
             (300, 166), (338, 150), (360, 160), (398, 236), (446, 214), (490, 282), (548, 296), (610, 340),
             (680, 378)]
FAR_RANGE_CTRL = [(-30, 330), (60, 298), (150, 312), (260, 280), (370, 300), (470, 266), (560, 300),
                  (660, 312), (760, 330), (860, 352), (940, 372)]
RIGHT_HILLS_CTRL = [(820, 372), (920, 356), (1010, 360), (1110, 340), (1210, 346), (1300, 332), (1400, 340),
                    (1530, 330)]

RIVER_FAR = [(640, 381), (700, 390), (672, 402), (590, 414), (540, 432), (572, 458), (640, 486), (690, 522)]
RIVER_FAR_W = [3, 4, 6, 8, 11, 14, 18, 26]
POND = blob(700, 578, 345, 60, n=90, rough=0.22, seed=7, smooth=4)
RIVER_NEAR_C = resample([(420, 600), (372, 628), (330, 672), (300, 712), (238, 760), (190, 820), (150, 862),
                         (95, 920), (40, 990), (-40, 1050), (-120, 1110)], 60)
RIVER_NEAR_W = np.interp(np.linspace(0, 1, 60), np.linspace(0, 1, 8), [70, 95, 125, 165, 215, 270, 330, 380]) * \
    (1 + 0.12 * np.sin(np.linspace(0, 1, 60) * 23) + 0.06 * np.sin(np.linspace(0, 1, 60) * 51 + 1.0))
RIVER_NEAR, RIVER_NEAR_L, RIVER_NEAR_R = ribbon(RIVER_NEAR_C, RIVER_NEAR_W)
SMALL_POND = blob(1120, 438, 92, 9, n=40, rough=0.25, seed=8, smooth=3)

HILL_EDGE = np.array([(540, H + 20), (650, 985), (760, 902), (870, 820), (985, 742), (1100, 668), (1220, 600),
                      (1350, 540), (1530, 482)])
HILL = np.vstack([HILL_EDGE, [(W + 20, H + 20)]])


def hill_edge_y(x):
    return np.interp(x, HILL_EDGE[:, 0], HILL_EDGE[:, 1])


def fractal_ridge(ctrl, step=4.0, rough=0.35, seed=0):
    r = np.random.default_rng(seed)
    pts = np.asarray(ctrl, float)
    while np.max(np.diff(pts[:, 0])) > step:
        new = [pts[0]]
        for a, b in zip(pts[:-1], pts[1:]):
            mid = (a + b) / 2
            mid[1] += r.normal(0, rough * abs(b[0] - a[0]) * 0.5)
            new += [mid, b]
        pts = np.array(new)
    return pts


FOOTHILLS_CTRL = [(-30, 322), (40, 304), (110, 314), (190, 298), (270, 312), (350, 300), (430, 314),
                  (520, 322), (610, 334), (700, 348), (790, 362), (880, 374)]
CRAGS = fractal_ridge(CRAG_CTRL, rough=0.3, seed=11)
FOOTHILLS = fractal_ridge(FOOTHILLS_CTRL, rough=0.04, seed=14)
FAR_RANGE = fractal_ridge(FAR_RANGE_CTRL, rough=0.22, seed=12)
RIGHT_HILLS = fractal_ridge(RIGHT_HILLS_CTRL, rough=0.06, seed=13)


def ridge_y(ridge, x):
    return np.interp(x, ridge[:, 0], ridge[:, 1])


# ===========================================================================
# Sky and distance
# ===========================================================================
def paint_sky():
    stage("sky")
    sky = np.array([(-20, -20), (W + 20, -20), (W + 20, HORIZON + 30), (-20, HORIZON + 30)])
    fill(sky, lambda x, y: ramp([SKY_TOP, SKY_MID, SKY_LOW, SKY_GLOW], (y + 10) / (HORIZON + 30)),
         width=34, spacing=14, wobble=1.5, pressure=0.45, seg_len=(160, 300))
    for y in np.arange(8, HORIZON + 20, 15):    # blend with a clean brush
        st([(-20, y), (W / 2, y + rng.normal(0, 2)), (W + 20, y)], None, 36, amount=0, pressure=0.3, pickup=0.2)
    canvas.dry()


def hatch_range(ridge, base, lit_paint, shade_paint, ledge_paint, density=1.0, max_len=1.0, seed=0):
    """Engraving-like fall lines below a ridge: lit where the slope faces the
    light (the ridge rising to the right), shaded where it faces away."""
    r = np.random.default_rng(seed)
    xs, ys = ridge[:, 0], ridge[:, 1]
    slope = np.convolve(np.gradient(ys, xs), np.ones(5) / 5, mode="same")
    for i in range(0, len(ridge), max(1, int(2 / density))):
        x, y = ridge[i]
        h = base - y
        if h < 4:
            continue
        lit = slope[i] < 0.05
        for _ in range(1 + int(r.random() < 0.5)):
            ln = min(h * r.uniform(0.15, 0.55) * max_len, 90)
            d = -1 if lit else 1
            ang = np.pi / 2 + d * r.uniform(0.15, 0.55)
            x0, y0 = x + r.normal(0, 1), y + r.uniform(1, 4)
            pts = [(x0, y0), (x0 + 0.5 * ln * np.cos(ang) + r.normal(0, 1.5), y0 + 0.5 * ln * np.sin(ang)),
                   (x0 + ln * np.cos(ang), min(y0 + ln * np.sin(ang), base))]
            st(pts, lit_paint if lit else shade_paint, [r.uniform(1.5, 3.2), 0.6], pressure=0.25)
    for _ in range(int(len(ridge) * 0.25 * density)):
        x = r.uniform(xs.min() + 5, xs.max() - 5)
        top = ridge_y(ridge, x)
        y = r.uniform(top + 4, max(top + 5, top + 0.5 * (base - top)))
        ln = r.uniform(4, 16)
        st([(x, y), (x + ln, y + r.uniform(-2, 3))], ledge_paint, r.uniform(1.0, 2.0), pressure=0.2)


def crag_planes(ridge, base, lit, shade, step=5, seed=0):
    """Broad strokes falling from the ridge: lit planes where the slope faces
    the light (left), shaded planes where it faces away."""
    r = np.random.default_rng(seed)
    xs, ys = ridge[:, 0], ridge[:, 1]
    slope = np.convolve(np.gradient(ys, xs), np.ones(7) / 7, mode="same")
    for i in range(0, len(ridge), max(1, int(step / 4))):
        x, y = ridge[i]
        h = base - y
        if h < 6:
            continue
        is_lit = slope[i] < 0.0
        ang = np.pi / 2 + (-1 if is_lit else 1) * r.uniform(0.2, 0.45)
        ln = h * r.uniform(0.5, 0.95)
        pts = [(x, y + 1.5), (x + 0.5 * ln * np.cos(ang) + r.normal(0, 1), y + 0.5 * ln * np.sin(ang)),
               (x + ln * np.cos(ang), y + ln * np.sin(ang))]
        k = float(np.clip(1 - (y - 120) / 260, 0, 1))         # higher = crisper
        st(pts, lerp_paint(lit if is_lit else shade, HAZE, 0.55 - 0.35 * k), [r.uniform(5, 8), r.uniform(3, 5)],
           pressure=0.3, mixing=0.4, taper=(0, 0.4))


def paint_mountains():
    stage("mountains")
    far = np.vstack([FAR_RANGE, [(FAR_RANGE[-1, 0], HORIZON + 12), (FAR_RANGE[0, 0], HORIZON + 12)]])
    fill(far, mix((SKY_LOW, 3), (ROCK_LIGHT, 1)), width=8, spacing=4, wobble=0.5, seg_len=(60, 140))
    right = np.vstack([RIGHT_HILLS, [(RIGHT_HILLS[-1, 0], HORIZON + 12), (RIGHT_HILLS[0, 0], HORIZON + 12)]])
    fill(right, mix((SKY_LOW, 4), (ROCK_LIGHT, 1), (OCHRE, 0.05)), width=8, spacing=4, wobble=0.5)
    body = np.vstack([CRAGS, [(CRAGS[-1, 0], HORIZON + 22), (CRAGS[0, 0], HORIZON + 22)]])
    fill(body, HAZE, width=8, spacing=4, angle=0.25, wobble=0.4, seg_len=(50, 120))
    canvas.dry()
    # broad planes of light snow and shaded rock, then a few crisp accents
    crag_planes(FAR_RANGE, HORIZON, mix((WHITE, 2), (SKY_LOW, 3)), mix((ROCK_LIGHT, 1), (SKY_LOW, 3)), seed=20)
    crag_planes(CRAGS, 330, mix((WHITE, 8), (SKY_LOW, 1)), mix((ROCK_DARK, 2), (ROCK, 1)), seed=21)
    canvas.dry()
    hatch_range(CRAGS, 330, mix((WHITE, 5), (SKY_LOW, 1)), mix((ROCK_DARK, 1), (ROCK, 1)),
                mix((WHITE, 4), (SKY_LOW, 1)), density=0.45, max_len=0.6, seed=22)
    ys = CRAGS[:, 1]
    for i in range(2, len(CRAGS) - 2, 4):     # dark clefts on the shaded flanks
        if ys[i] < 280 and ys[i + 2] > ys[i] + 1.5 and rng.random() < 0.5:
            x, y = CRAGS[i]
            ln = (330 - y) * rng.uniform(0.2, 0.45)
            st([(x + 1, y + 2), (x + 3 + rng.normal(0, 2), y + ln * 0.5), (x + 5 + rng.normal(0, 3), y + ln)],
               ROCK_DARK, [2.0, 1.2, 0.5], pressure=0.25)
    st(CRAGS[::2], mix((ROCK_DARK, 1), (ROCK, 2)), 1.0, pressure=0.2)
    # a castle on the highest crag, a ruined tower on a lower one
    cx = 338
    cy = ridge_y(CRAGS, cx) + 2
    for dx, hgt, w in ((-10, 12, 4), (-4, 18, 5), (4, 24, 4), (11, 14, 4)):
        st([(cx + dx, cy + 6), (cx + dx, cy - hgt)], STONE_SHADE, w, pressure=0.25)
        st([(cx + dx - 1, cy - hgt + 1), (cx + dx - 1, cy - hgt - 2)], WHITE, 1.4, pressure=0.2)
    st([(cx - 12, cy - 8), (cx + 13, cy - 8)], STONE_SHADE, 2.5)
    rx = 175
    ry_ = ridge_y(CRAGS, rx) + 4
    st([(rx, ry_ + 6), (rx, ry_ - 12)], STONE_SHADE, 4)
    st([(rx + 1, ry_ - 12), (rx + 3, ry_ - 9)], ROCK_DARK, 2)
    canvas.dry()

    # rolling foothills in front of the crags, with woods and a hamlet
    hills = np.vstack([FOOTHILLS, [(FOOTHILLS[-1, 0], HORIZON + 14), (FOOTHILLS[0, 0], HORIZON + 14)]])
    fill(hills, lambda x, y: lerp_paint(mix((SNOW_FAR, 2), (SKY_LOW, 1)), SNOW_FAR, (y - 300) / 70), width=10,
         spacing=5, wobble=0.6, seg_len=(60, 160))
    canvas.dry()
    xs, ys = FOOTHILLS[:, 0], FOOTHILLS[:, 1]
    slope = np.gradient(ys, xs)
    for i in range(0, len(FOOTHILLS), 2):     # shade on the slopes that face away from the light
        if slope[i] > 0.08:
            x, y = FOOTHILLS[i]
            st([(x, y + 2), (x + 6, y + 14)], mix((SNOW_SHADOW, 1), (SKY_LOW, 1)), 5, pressure=0.25, mixing=0.5)
    wood = mix((MID_TREES, 1), (SKY_LOW, 2))
    for _ in range(260):                      # woods: clusters of tiny trees on the slopes
        cx_ = rng.uniform(-20, 860)
        top = ridge_y(FOOTHILLS, cx_)
        if rng.random() < 0.55:
            y = top + rng.uniform(1, 10)
        else:
            y = rng.uniform(top + 5, HORIZON + 8)
        h = rng.uniform(4, 9) * (0.7 + 0.5 * (y - 300) / 80)
        st([(cx_, y), (cx_ + rng.normal(0, 0.5), y - h)], wood, max(0.8, h * 0.12), pressure=0.2)
        st([(cx_ - h * 0.3, y - h * 0.5), (cx_, y - h), (cx_ + h * 0.3, y - h * 0.55)], wood, max(0.7, h * 0.07),
           pressure=0.2)
    for x in (210, 236, 252, 560, 590):       # a hamlet on the hills
        y = ridge_y(FOOTHILLS, x) + 8
        building(x, y, 9, 6, 6, 12, dirx=1, chimney=False, windows=0, door=False, far=0.9)
    church(222, ridge_y(FOOTHILLS, 222) + 10, 0.22, far=0.9)
    canvas.dry()


# ===========================================================================
# Terrain and ice
# ===========================================================================
ROADS = [
    # trodden paths wandering from the inn round the pond, and along the left bank
    [(560, 848), (538, 806), (530, 762), (548, 720), (590, 690), (650, 672), (720, 668), (800, 672),
     (880, 668), (950, 652), (1010, 628)],
    [(720, 668), (760, 640), (790, 600), (812, 560), (830, 530)],
    [(140, 700), (170, 662), (205, 625), (232, 590), (270, 560), (320, 534)],
]


def paint_terrain():
    stage("valley snow")
    valley = np.array([(-20, HORIZON - 4), (W + 20, HORIZON - 4), (W + 20, H + 20), (-20, H + 20)])
    fill(valley, lambda x, y: ramp([SNOW_FAR, SNOW, SNOW_WARM], 1 - depth_at(y)), width=24, spacing=11,
         wobble=1.0, pressure=0.45)
    canvas.dry()
    # low mounds and banks: a cool shadow on the side away from the light
    for _ in range(26):
        x, y = rng.uniform(600, 1450), rng.uniform(HORIZON + 30, 760)
        sc = 0.3 + 0.7 * (1 - depth_at(y))
        ln = rng.uniform(60, 160) * sc
        st([(x, y), (x + ln * 0.5, y - 4 * sc), (x + ln, y + 2 * sc)], mix((SNOW, 2), (SNOW_SHADOW, 1)),
           rng.uniform(4, 9) * sc, pressure=0.3, mixing=0.6, taper=(0.1, 0.6))
    # trodden paths: a trampled band with cart ruts and footprints
    trodden = mix((SNOW, 4), (SNOW_SHADOW, 1), (EARTH, 0.2))
    rut = mix((EARTH, 1), (SNOW_SHADOW, 3))
    for road in ROADS:
        pts = resample(road, max(6, int(length(road) / 12)))
        pts = pts + rng.normal(0, 1.2, pts.shape)
        tan = np.gradient(pts, axis=0)
        tan /= np.linalg.norm(tan, axis=1, keepdims=True) + 1e-9
        nrm = np.stack([-tan[:, 1], tan[:, 0]], -1)
        ws = np.array([3 + 20 * (1 - depth_at(y)) for _, y in pts])
        st(pts, trodden, list(ws * 0.9), pressure=0.25, mixing=0.6)
        for side in (-0.22, 0.22):
            r_ = pts + nrm * (ws[:, None] * side) + rng.normal(0, 0.4, pts.shape)
            st(r_, rut, list(np.maximum(0.8, ws * 0.06)), pressure=0.2, amount=4 + length(r_) / 2)
        for k in range(int(length(road) / 18)):
            i = int(rng.integers(len(pts)))
            q = pts[i] + nrm[i] * ws[i] * rng.uniform(-0.45, 0.45)
            dab(q[0], q[1], rut, max(0.8, ws[i] * 0.07))
    canvas.dry()


def broken_line(pts, paint, width, gap=0.3, seg=(10, 40), **kw):
    """A line drawn as a series of short strokes with gaps, like a shoreline."""
    pts = resample(pts, max(2, int(length(pts))))
    i = 0
    while i < len(pts) - 2:
        n = int(rng.uniform(*seg))
        j = min(len(pts) - 1, i + n)
        w = width * rng.uniform(0.5, 1.2) if np.isscalar(width) else width
        st(pts[i:j + 1:max(1, (j - i) // 4)], paint, w, taper=0.3, **kw)
        i = j + int(n * gap * rng.uniform(0.3, 1.5))


def paint_ice():
    stage("ice")
    far_poly, far_l, _ = ribbon(RIVER_FAR, RIVER_FAR_W)
    st(RIVER_FAR, ICE, RIVER_FAR_W, pressure=0.3)
    st(far_l + [0, 1], ICE_DARK, [0.8, 1, 1.2, 1.4, 1.6, 2, 2.2, 2.6], pressure=0.2)
    for pond in (POND, SMALL_POND):
        cy = pond[:, 1].mean()
        ry = np.ptp(pond[:, 1]) / 2
        w = max(3, min(12, ry * 0.4))
        fill(pond, lambda x, y: ramp([ICE_DARK, ICE, ICE_LIGHT, ICE], (y - (cy - ry)) / (2 * ry)), width=w,
             spacing=w * 0.5, wobble=0.3, seg_len=(60, 170), overrun=-0.45)
    flow = -0.76
    fill(RIVER_NEAR, lambda x, y: ramp([ICE, ICE_LIGHT, ICE, mix((ICE, 1), (ICE_DARK, 1))],
                                       0.5 + 0.5 * np.sin(x / 70 - y / 110)),
         width=12, spacing=5, angle=flow, wobble=0.6, seg_len=(60, 180), overrun=-0.45)
    # trace the outlines so the shores are smooth rather than stair-stepped
    for pond, w in ((POND, 5), (SMALL_POND, 2)):
        ring = np.vstack([pond, pond[:1]])
        st(ring, ICE, w, pressure=0.3, mixing=0.3)
    for edge in (RIVER_NEAR_L, RIVER_NEAR_R):
        st(edge, ICE, 7, pressure=0.3, mixing=0.3)
    canvas.dry()
    # shores: broken shadow under the far banks, snow drifting over the near banks
    for pond in (POND, SMALL_POND):
        cy = pond[:, 1].mean()
        ry = np.ptp(pond[:, 1]) / 2
        top = pond[pond[:, 1] < cy]
        top = top[np.argsort(top[:, 0])]
        broken_line(top + [0, max(1.0, ry * 0.07)], mix((ICE_DARK, 2), (ICE, 1)), max(1.2, ry * 0.11), gap=0.4,
                    pressure=0.25, mixing=0.4)
        bot = pond[pond[:, 1] >= cy]
        bot = bot[np.argsort(bot[:, 0])]
        broken_line(bot + [0, -max(0.5, ry * 0.03)], SNOW, max(1.2, ry * 0.07), gap=1.2, seg=(20, 70),
                    pressure=0.25)
    broken_line(RIVER_NEAR_L[2:] + [2, 2], mix((ICE_DARK, 2), (ICE, 1)), 7, gap=0.35, seg=(15, 60),
                pressure=0.25, mixing=0.4)
    broken_line(RIVER_NEAR_R[:-3] + [-2, -2], SNOW, 6, gap=1.0, seg=(25, 80), pressure=0.25)
    # streaks along the flow: pale reflections, skate marks, thin drifts of snow
    ca, sa = np.cos(flow), np.sin(flow)
    for poly, n, sc, ang in ((POND, 60, 0.6, 0.0), (RIVER_NEAR, 120, 1.0, flow)):
        c, s_ = np.cos(ang), np.sin(ang)
        placed = 0
        while placed < n:
            x = rng.uniform(poly[:, 0].min(), poly[:, 0].max())
            y = rng.uniform(poly[:, 1].min(), min(poly[:, 1].max(), H))
            if not inside(poly, x, y):
                continue
            placed += 1
            s = sc * (0.3 + 0.7 * (1 - depth_at(y)))
            kind = rng.random()
            ln = rng.uniform(30, 150) * s
            bend = rng.normal(0, 6 * s)
            a = ang + rng.normal(0, 0.18)
            pts = [(x, y), (x + ln / 2 * np.cos(a) - bend * np.sin(a), y + ln / 2 * np.sin(a) + bend * np.cos(a)),
                   (x + ln * np.cos(a), y + ln * np.sin(a))]
            if not inside(poly, *pts[-1]):
                continue
            if kind < 0.4:      # skate marks
                st(pts, mix((ICE_DARK, 1), (ICE, 1)) if rng.random() < 0.5 else ICE_LIGHT, max(0.6, 0.8 * s),
                   pressure=0.15, amount=3, taper=0.3)
            elif kind < 0.75:   # pale reflections
                st(pts, ICE_LIGHT, max(1.2, 5 * s), pressure=0.2, mixing=0.5, taper=0.45)
            else:               # thin drifts of snow blown across the ice
                st(pts, SNOW, [max(0.8, 1.5 * s), max(1.2, 4 * s), max(0.8, 1.5 * s)], pressure=0.25, taper=0.3)
    # a hole cut in the ice for water, with broken slabs around it
    hole = blob(140, 962, 30, 9, n=24, rough=0.3, seed=3, smooth=1)
    fill(hole, mix((ICE_DARK, 2), (BLACK, 0.15), (GREEN, 0.3)), width=3, spacing=1.5, overrun=-0.3, seg_len=(20, 60))
    upper = hole[hole[:, 1] < 962]
    st(upper[np.argsort(upper[:, 0])] + [0, -1.5], ICE_LIGHT, 3)
    for k in range(5):
        a = rng.uniform(0, 2 * np.pi)
        px, py = 140 + 42 * np.cos(a), 962 + 14 * np.sin(a)
        slab = blob(px, py, rng.uniform(5, 9), rng.uniform(2, 4), n=8, rough=0.4)
        fill(slab, ICE_LIGHT, width=2, spacing=1, overrun=-0.3, seg_len=(20, 40), wobble=0)
        st([(px - 6, py + 2), (px + 6, py + 2)], ICE_DARK, 1.2)
    canvas.dry()


# ===========================================================================
# Buildings
# ===========================================================================
BUILDINGS = []


def gable_outline(o, w, h, roof, stepped):
    """Front (gable-end) wall outline, bottom-left at o."""
    if not stepped:
        return [o, o + [w, 0], o + [w, -h], o + [w / 2, -h - roof], o + [0, -h]]
    n = 3
    left = [np.array([0.0, -h])]
    for k in range(n):
        y_top = -h - roof * (k + 1) / (n + 1)
        x_out = (w / 2) * k / (n + 0.6)
        x_in = (w / 2) * (k + 1) / (n + 0.6)
        left += [np.array([x_out, y_top]), np.array([x_in, y_top])]
    left += [np.array([left[-1][0], -h - roof])]
    right = [np.array([w - p[0], p[1]]) for p in left]
    return [o, o + [w, 0]] + [o + p for p in right] + [o + p for p in left[::-1]]


def building(x, y, w, h, roof, depth, dirx=1, wall=WALL, kind="cottage", windows=2, door=True,
             chimney=True, timber=False, stepped=False, smoke=False, snow=True, roof_paint=THATCH, far=0.0):
    """Base pass of a gabled building: (x, y) is the bottom-left of the gable
    wall, w its width, h the wall height, `roof` the gable height; the side
    wall recedes by `depth` toward `dirx`. Details come in `building_details`."""
    d = np.array([dirx * depth, -0.30 * depth])
    o = np.array([x, y], float)
    front = gable_outline(o, w, h, roof, stepped)
    sx = w if dirx > 0 else 0
    side = [o + [sx, 0], o + [sx, 0] + d, o + [sx, -h] + d, o + [sx, -h]]
    ridge0 = o + [w / 2, -h - roof]
    eave0 = o + [sx, -h]
    roof_poly = [ridge0, ridge0 + d, eave0 + d, eave0]
    lit_side = dirx < 0                           # light comes from the left
    front_paint = atmos(wall, far)
    side_paint = atmos(light_of(wall, 0.25) if lit_side else shade_of(wall, 0.45), far)
    sw = max(1.2, min(w, depth) / 5)
    # roof first, so the gable wall overlaps it cleanly
    rp = atmos(SNOW if snow else roof_paint, far)
    fill(roof_poly, rp, width=max(1.2, depth / 8), spacing=max(0.7, depth / 18), angle=np.arctan2(d[1], d[0]),
         seg_len=(500, 600), wobble=0.1, overrun=-0.2, pressure=0.3)
    def varied(p):
        tones = [p, p, shade_of(p, 0.06), light_of(p, 0.08), mix((p, 1), (OCHRE, 0.06))]
        return lambda *_: tones[int(rng.integers(len(tones)))]

    fill(side, varied(side_paint), width=sw, spacing=max(0.7, sw / 2), angle=np.pi / 2, seg_len=(500, 600),
         wobble=0.1, overrun=-0.3, pressure=0.3)
    fill(front, varied(front_paint), width=max(1.2, w / 7), spacing=max(0.7, w / 14), angle=np.pi / 2,
         seg_len=(500, 600), wobble=0.1, overrun=-0.3, pressure=0.3)
    BUILDINGS.append(dict(kind=kind, o=o, w=w, h=h, roof=roof, d=d, dirx=dirx, wall=wall, windows=windows,
                          door=door, chimney=chimney, timber=timber, stepped=stepped, smoke=smoke, snow=snow,
                          front=front, roof_poly=roof_poly, far=far, roof_paint=roof_paint))


def smoke(x, y, scale=1.0):
    """A plume of thin smoke drifting left on a light wind."""
    pts = [(x, y)]
    cx, cy = x, y
    for k in range(6):
        cx -= (6 + 9 * k) * scale + rng.normal(0, 2)
        cy -= (12 - k) * scale + rng.normal(0, 1)
        pts.append((cx, cy))
    widths = [1.5 * scale + 2.2 * scale * k for k in range(len(pts))]
    st(pts, SMOKE, widths, amount=3 + 20 * scale, pressure=0.15, mixing=0.0, pickup=0.0, drag=0.0, taper=(0, 0.3))


def building_details(b):
    o, w, h, roof, d, far = b["o"], b["w"], b["h"], b["roof"], b["d"], b["far"]
    s = max(w, 10) / 60.0
    dk = atmos(DARK, far)
    win = atmos(WINDOW, far)
    lw = max(0.8, 1.6 * s)
    ridge0, ridge1, eave1, eave0 = [np.asarray(p) for p in b["roof_poly"]]
    if b["snow"]:
        # model the snowy roof: a cool glaze under the ridge and at its far end
        cool = mix((SNOW_SHADOW, 1), (MEDIUM, 1.5))
        st([ridge0 + (eave0 - ridge0) * 0.2, ridge1 + (eave1 - ridge1) * 0.2], atmos(cool, far),
           max(1.0, roof * 0.3), pressure=0.2)
        st([ridge1, eave1], atmos(mix((SNOW_SHADOW, 2), (DARK, 0.2)), far), lw * 0.7, pressure=0.2)
        # snow along the ridge, overhanging the eaves, a dark shadow line beneath
        st([eave0 + [0, lw * 1.3], eave1 + [0, lw * 1.3]], dk, lw * 1.1, pressure=0.2)
        sag = [eave0 + t * (eave1 - eave0) + [0, 0.7 * lw * np.sin(np.pi * t)] for t in np.linspace(0, 1, 5)]
        st(sag, atmos(SNOW, far), lw * 1.6, pressure=0.25)
        st([ridge0, ridge1], atmos(WHITE, far), lw * 1.3, pressure=0.2)
        under = BRICK_SHADE if b["kind"] == "brick" else b["roof_paint"]
        st([eave0 + [0, lw * 2.3], eave1 + [0, lw * 2.3]], atmos(under, far), lw * 0.8, pressure=0.2)
        # a few dark patches where snow has slid off the thatch
        if s > 0.4:
            for _ in range(2):
                t1, t2 = rng.uniform(0.2, 0.8), rng.uniform(0.3, 0.9)
                p = ridge0 + t1 * (eave0 - ridge0) + t2 * d
                st([p, p + d * 0.12], atmos(mix((b["roof_paint"], 1), (SNOW, 1)), far), lw * 0.9, pressure=0.2)
    else:
        for t in np.linspace(0.15, 0.85, 4):
            a, c = ridge0 + t * (eave0 - ridge0), ridge1 + t * (eave1 - ridge1)
            st([a, c], atmos(shade_of(b["roof_paint"], 0.3), far), lw * 0.6, pressure=0.2)
    fr = [np.asarray(p) for p in b["front"]]
    if not b["stepped"]:
        gl, gt, gr = o + [0, -h], o + [w / 2, -h - roof], o + [w, -h]
        st([gl + [-lw * 0.3, 0], gt], dk, lw * 0.9, pressure=0.2)
        st([gt, gr + [lw * 0.3, 0]], dk, lw * 0.9, pressure=0.2)
        if b["snow"]:
            # the snow layer riding the gable edges, thick and rounded at the eaves
            thick = max(1.0, w * 0.075)
            for a, c in ((gl, gt), (gt, gr)):
                st([a + [0, -thick * 0.6], c + [0, -thick * 0.6]], atmos(WHITE, far), thick, pressure=0.25)
            dab(gl[0] - thick * 0.2, gl[1] - thick * 0.3, atmos(WHITE, far), thick * 1.2)
            dab(gr[0] + thick * 0.2, gr[1] - thick * 0.3, atmos(WHITE, far), thick * 1.2)
            if far < 0.5 and w > 25:      # icicles under the snow lip of the eave
                for t in np.linspace(0.05, 0.95, int(abs(d[0]) / 5)):
                    p = eave0 + t * (eave1 - eave0) + [0, lw * 1.8]
                    st([p, p + [0.3, rng.uniform(2, 9) * s]], atmos(ICE_LIGHT, far), [max(0.9, 1.6 * s), 0.5],
                       pressure=0.2)
    else:
        for p0, p1 in zip(fr[2:-1], fr[3:]):
            if abs(p1[1] - p0[1]) < 1e-6 and p0[1] < o[1] - h + 0.1:
                st([p0 + [0, -lw * 0.6], p1 + [0, -lw * 0.6]], atmos(WHITE, far), lw * 1.3, pressure=0.2)
    if b["timber"]:
        tc = atmos(mix((BURNT_UMBER, 1), (BLACK, 0.6)), far)
        tw_ = lw * 0.7
        n_posts = max(2, int(w / 14))
        for k in range(n_posts + 1):
            px = o[0] + w * k / n_posts
            st([(px, o[1] - lw), (px, o[1] - h)], tc, tw_, pressure=0.2)
        for fy in (0.04, 0.52, 0.98):
            st([(o[0], o[1] - h * fy), (o[0] + w, o[1] - h * fy)], tc, tw_, pressure=0.2)
        for k in (0, n_posts - 1):          # braces in the end bays
            x0_ = o[0] + w * k / n_posts
            x1_ = o[0] + w * (k + 1) / n_posts
            if k == 0:
                st([(x0_, o[1] - h * 0.52), (x1_, o[1] - h * 0.98)], tc, tw_ * 0.8, pressure=0.2)
            else:
                st([(x1_, o[1] - h * 0.52), (x0_, o[1] - h * 0.98)], tc, tw_ * 0.8, pressure=0.2)
        sx = w if b["dirx"] > 0 else 0
        n_side = max(2, int(abs(d[0]) / 14))
        for k in range(n_side + 1):
            p = o + [sx, 0] + d * (k / n_side)
            st([p + [0, -lw], p + [0, -h]], tc, tw_ * 0.9, pressure=0.2)
        for fy in (0.52, 0.98):
            st([o + [sx, -h * fy], o + [sx, -h * fy] + d], tc, tw_ * 0.9, pressure=0.2)
    nwin = b["windows"]
    for k in range(nwin):
        fx = (k + 1) / (nwin + 1)
        if b["door"] and abs(fx - 0.5) < 0.12 and nwin > 1:
            continue
        wx = o[0] + w * fx
        ww = max(0.9, w * 0.1)
        st([(wx, o[1] - h * 0.70), (wx, o[1] - h * 0.46)], win, ww, pressure=0.2)
        if w > 40:          # mullion, transom and a shutter
            st([(wx, o[1] - h * 0.70), (wx, o[1] - h * 0.46)], atmos(WOOD, far), max(0.7, ww * 0.15), pressure=0.2)
            st([(wx - ww / 2, o[1] - h * 0.6), (wx + ww / 2, o[1] - h * 0.6)], atmos(WOOD, far),
               max(0.7, ww * 0.15), pressure=0.2)
            st([(wx + ww * 0.6, o[1] - h * 0.70), (wx + ww * 0.6, o[1] - h * 0.47)],
               atmos(shade_of(WOOD, 0.3), far), max(0.8, ww * 0.45), pressure=0.2)
        st([(wx - w * 0.065, o[1] - h * 0.44), (wx + w * 0.065, o[1] - h * 0.44)], atmos(SNOW, far),
           max(0.7, lw * 0.7), pressure=0.2)
    if w > 14 and not b["stepped"]:
        wx = o[0] + w * 0.5
        st([(wx, o[1] - h - roof * 0.32), (wx, o[1] - h - roof * 0.08)], win, max(0.8, w * 0.05), pressure=0.2)
    sx = w if b["dirx"] > 0 else 0
    ns = max(1, int(abs(d[0]) / max(8.0, 16 * s)))
    for k in range(ns):
        t = (k + 0.5) / ns
        p = o + [sx, 0] + d * t
        if b["door"] and k == ns // 2:
            st([p + [0, -lw * 0.5], p + [0, -h * 0.58]], atmos(NEAR_BLACK, far), max(1.0, abs(d[0]) / ns * 0.42),
               pressure=0.2)
        else:
            st([p + [0, -h * 0.66], p + [0, -h * 0.45]], win, max(0.8, abs(d[0]) / ns * 0.28), pressure=0.2)
    # snow banked against the foot of the walls
    sx = w if b["dirx"] > 0 else 0
    bank = atmos(SNOW, far)
    st([o + [-2 * lw, lw], o + [w * 0.5, lw * 0.4], o + [w + lw, lw]], bank, max(1.0, h * 0.09), pressure=0.25)
    st([o + [sx, lw], o + [sx, lw] + d], bank, max(1.0, h * 0.08), pressure=0.25)
    if b["chimney"]:
        base = ridge0 + 0.35 * (eave0 - ridge0) + 0.65 * d
        cw = max(1.5, 5 * s)
        top = base + [0, -roof * 0.55]
        st([base + [0, 2], top], atmos(BRICK, far), cw, pressure=0.25)
        st([top + [-cw * 0.6, 0], top + [cw * 0.6, 0]], atmos(WHITE, far), max(1, cw * 0.5), pressure=0.2)
        if b["smoke"]:
            smoke(top[0], top[1] - 2, scale=min(0.6, max(0.25, s)))


def church(x, y, s, dirx=1, far=0.0):
    """Nave with a west tower and spire."""
    building(x, y, 30 * s, 26 * s, 20 * s, 70 * s, dirx=dirx, wall=STONE, windows=1, door=False,
             chimney=False, far=far)
    tw, th = 16 * s, 62 * s
    tx = x - tw * 0.4 if dirx > 0 else x + 30 * s - tw * 0.6
    tower = [(tx, y), (tx + tw, y), (tx + tw, y - th), (tx, y - th)]
    fill(tower, atmos(STONE, far), width=max(1.2, tw / 4), spacing=max(0.7, tw / 8), angle=np.pi / 2,
         seg_len=(500, 600), wobble=0.1, overrun=-0.3)
    side = [(tx + tw, y), (tx + tw + 7 * s, y - 2.5 * s), (tx + tw + 7 * s, y - th - 2.5 * s), (tx + tw, y - th)]
    fill(side, atmos(STONE_SHADE, far), width=max(1.0, 3 * s), spacing=max(0.6, 1.5 * s), angle=np.pi / 2,
         seg_len=(500, 600), wobble=0.1, overrun=-0.3)
    BUILDINGS.append(dict(kind="tower", tx=tx, tw=tw, th=th, y=y, s=s, far=far))


def tower_details(b):
    tx, tw, th, y, s, far = b["tx"], b["tw"], b["th"], b["y"], b["s"], b["far"]
    dk = atmos(DARK, far)
    spire = [(tx - 1 * s, y - th), (tx + tw + 8 * s, y - th - 2 * s), (tx + tw * 0.55, y - th - 48 * s)]
    fill(spire, atmos(SLATE, far), width=max(1, 3 * s), spacing=max(0.6, 1.5 * s), angle=np.pi / 2,
         seg_len=(500, 600), wobble=0.1, overrun=-0.3)
    st([(tx + tw * 0.55, y - th - 48 * s), (tx - 1 * s, y - th)], atmos(WHITE, far), max(0.8, 1.6 * s))
    st([(tx + tw * 0.55, y - th - 48 * s), (tx + tw * 0.55, y - th - 57 * s)], dk, max(0.7, 1.2 * s))
    st([(tx + tw * 0.4, y - th - 53 * s), (tx + tw * 0.7, y - th - 53 * s)], dk, max(0.7, 1 * s))
    for hy in (0.25, 0.55, 0.82):
        st([(tx + tw / 2, y - th * hy), (tx + tw / 2, y - th * hy - 8 * s)], atmos(WINDOW, far), max(0.8, 3.2 * s))
    st([(tx, y - th - 1), (tx + tw, y - th - 1)], atmos(WHITE, far), max(0.8, 1.5 * s))


def mill(x, y, s):
    """A watermill: a stone mill house with a big frozen wheel on its left."""
    building(x, y, 60 * s, 40 * s, 30 * s, 70 * s, dirx=1, wall=STONE, windows=2, smoke=True)
    BUILDINGS.append(dict(kind="wheel", x=x - 6 * s, y=y - 22 * s, r=30 * s, s=s))


def wheel_details(b):
    x, y, r, s = b["x"], b["y"], b["r"], b["s"]
    for a in np.linspace(0, np.pi, 6, endpoint=False):
        st([(x - r * np.cos(a), y - r * np.sin(a)), (x + r * np.cos(a), y + r * np.sin(a))], DARK, max(1, 2.2 * s))
    st([(x + r * np.cos(a), y + r * 0.95 * np.sin(a)) for a in np.linspace(0, 2 * np.pi, 30)], DARK,
       max(1.2, 4 * s), pressure=0.3)
    st([(x + r * 0.8 * np.cos(a), y + r * 0.76 * np.sin(a)) for a in np.linspace(0, 2 * np.pi, 26)], WOOD,
       max(0.8, 2 * s), pressure=0.25)
    for a in np.linspace(0, 2 * np.pi, 16, endpoint=False):     # paddles
        st([(x + r * np.cos(a), y + r * 0.95 * np.sin(a)),
            (x + (r + 5 * s) * np.cos(a), y + (r + 5 * s) * 0.95 * np.sin(a))], DARK, max(1, 3 * s))
    st([(x + r * np.cos(a), y + r * 0.95 * np.sin(a) - 2 * s) for a in np.linspace(np.pi * 1.1, np.pi * 1.9, 8)],
       WHITE, max(1, 2.5 * s))
    for k in range(9):     # icicles hanging from the rim
        ix = x - r * 0.8 + k * r * 0.2
        iy = y + r * 0.95 * np.sqrt(max(0, 1 - ((ix - x) / r) ** 2))
        st([(ix, iy), (ix + 0.5, iy + rng.uniform(6, 16) * s)], ICE_LIGHT, [max(1, 2.4 * s), 0.6], pressure=0.2)
    dab(x, y, DARK, 6 * s)


def post_mill(x, y, s, far=0.6):
    """A distant post windmill on its trestle."""
    dk = atmos(DARK, far)
    st([(x, y), (x, y - 14 * s)], dk, 1.5 * s)
    st([(x - 7 * s, y), (x, y - 8 * s), (x + 7 * s, y)], dk, 1.0 * s)
    body = [(x - 6 * s, y - 12 * s), (x + 6 * s, y - 12 * s), (x + 6 * s, y - 28 * s), (x - 6 * s, y - 30 * s)]
    fill(body, atmos(WOOD, far), width=2 * s, spacing=1 * s, angle=np.pi / 2, seg_len=(400, 500), wobble=0,
         overrun=-0.3)
    st([(x - 6 * s, y - 31 * s), (x + 6 * s, y - 29 * s)], atmos(WHITE, far), 1.5 * s)
    hub = (x - 7 * s, y - 24 * s)
    for a in (0.5, 0.5 + np.pi / 2, 0.5 + np.pi, 0.5 + 1.5 * np.pi):
        tip = (hub[0] + 26 * s * np.cos(a), hub[1] + 26 * s * np.sin(a))
        st([hub, tip], dk, 1.2 * s)
        n = (np.cos(a + np.pi / 2), np.sin(a + np.pi / 2))
        st([(hub[0] + 8 * s * np.cos(a) + 3 * s * n[0], hub[1] + 8 * s * np.sin(a) + 3 * s * n[1]),
            (tip[0] + 3 * s * n[0], tip[1] + 3 * s * n[1])], atmos(C_GREY, far), 0.8 * s)


# ===========================================================================
# Vegetation, fences, haystacks
# ===========================================================================
def far_tree(x, y, h, paint):
    """A small bare tree seen from afar: a thin trunk under a lacy crown."""
    lace = mix((paint, 1), (MEDIUM, 1.2))
    top = (x + rng.normal(0, h * 0.04), y - h * 0.55)
    st([(x, y), top], paint, max(0.9, h * 0.055), pressure=0.25)
    for _ in range(int(5 + h / 2.5)):
        sy = y - h * rng.uniform(0.35, 0.6)
        a = -np.pi / 2 + rng.normal(0, 0.55)
        ln = h * rng.uniform(0.25, 0.5)
        mx, my = x + ln * 0.5 * np.cos(a), sy + ln * 0.5 * np.sin(a)
        st([(x, sy), (mx, my), (mx + ln * 0.5 * np.cos(a - 0.2 * np.sign(np.cos(a))), my + ln * 0.5 * np.sin(a))],
           lace, max(0.6, h * 0.02), pressure=0.2)


def tree_row(p0, p1, h, spacing=None, var=0.35):
    (xa, ya), (xb, yb) = p0, p1
    spacing = spacing or h * 0.8
    n = max(2, int(np.hypot(xb - xa, yb - ya) / spacing))
    for t in np.linspace(0, 1, n):
        x = xa + (xb - xa) * t + rng.normal(0, spacing * 0.2)
        y = ya + (yb - ya) * t + rng.normal(0, 1.5)
        far_tree(x, y, h * rng.uniform(1 - var, 1 + var), atmos(MID_TREES, depth_at(y) ** 0.7))


def tree(x, y, height, base_w, lean=0.0, seed=0, paint=TRUNK, depth=5, snow_on=True, spread=1.0, lace_thin=0.8,
         fan=(3, 7)):
    """A tall bare tree: a slender trunk with limbs reaching up and out,
    dividing into fine twigs; snow lies on the upper side of the limbs.
    Returns (snow lines to paint once the tree is dry, perches for birds)."""
    r = np.random.default_rng(seed)
    perches, snow_lines = [], []
    twig_col = mix((paint, 1), (TWIG, 1))
    lace = mix((twig_col, 1), (MEDIUM, lace_thin))

    def grow(x0, y0, ang, ln, w0, level):
        n = 5
        pts = [(x0, y0)]
        a = ang
        for k in range(1, n):
            a += r.normal(0, 0.10) + 0.10 * (-np.pi / 2 - a) * (level / depth)     # tips turn upward
            px, py = pts[-1]
            pts.append((px + ln / (n - 1) * np.cos(a), py + ln / (n - 1) * np.sin(a)))
        w1 = max(0.6, w0 * 0.45)
        st(pts, paint if level < 3 else twig_col, [w0, (w0 + w1) / 2, w1], pressure=0.3)
        dirs = np.diff(np.asarray(pts), axis=0)
        horiz = abs(np.mean(dirs[:, 0])) / (np.mean(np.linalg.norm(dirs, axis=1)) + 1e-9)
        if snow_on and w0 > 2.6 and horiz > 0.5 and r.random() < 0.45:
            snow_lines.append(([(px, py - w0 * 0.45) for px, py in pts[: 3 + int(r.random() * 2)]], w0 * 0.5))
        if 2 <= level <= 3 and horiz > 0.4 and r.random() < 0.15:
            perches.append(pts[2])
        if level >= depth or ln < 5:
            ex, ey = pts[-1]
            for _ in range(int(r.integers(*fan))):     # a fan of fine twigs: the lace of the crown
                ta = a + r.normal(0, 0.55)
                tl = ln * r.uniform(0.4, 1.0)
                mx, my = ex + tl * 0.5 * np.cos(ta), ey + tl * 0.5 * np.sin(ta)
                tb = ta + r.normal(0, 0.25)
                st([(ex, ey), (mx, my), (mx + tl * 0.5 * np.cos(tb), my + tl * 0.5 * np.sin(tb))], lace,
                   [0.9, 0.6], pressure=0.2)
            return
        for _ in range(int(r.integers(2, 4))):
            t = r.uniform(0.25, 0.95)
            i = min(int(t * (n - 1)), n - 2)
            f = t * (n - 1) - i
            bx = pts[i][0] + (pts[i + 1][0] - pts[i][0]) * f
            by = pts[i][1] + (pts[i + 1][1] - pts[i][1]) * f
            child = a + (1 if r.random() < 0.5 else -1) * r.uniform(0.3, 0.8) * spread
            child = -np.pi / 2 + (child + np.pi / 2) * 0.88
            grow(bx, by, child, ln * r.uniform(0.4, 0.68), max(0.6, w0 * 0.55 * (1 - t * 0.35)), level + 1)

    n = 10
    trunk = [(x, y)]
    a0 = -np.pi / 2 + lean
    for k in range(1, n):
        px, py = trunk[-1]
        a = a0 + r.normal(0, 0.035) + 0.015 * np.sin(k)
        trunk.append((px + height / (n - 1) * np.cos(a), py + height / (n - 1) * np.sin(a)))
    widths = [base_w * (1.15 if k == 0 else (1 - k / n) ** 0.8) + 0.8 for k in range(n)]
    st(trunk, paint, widths, pressure=0.35)
    # bark: a lit edge on the left, dark furrows, snow banked at the root
    st([(px - widths[k] * 0.32, py) for k, (px, py) in enumerate(trunk[:6])], TRUNK_LIGHT,
       [max(1.0, w * 0.22) for w in widths[:6]], pressure=0.2)
    for _ in range(int(base_w * 1.5)):
        k = int(r.integers(0, 5))
        px, py = trunk[k]
        off = r.uniform(-0.25, 0.3) * widths[k]
        st([(px + off, py - r.uniform(0, 6)), (px + off + r.normal(0, 0.5), py - height / n * r.uniform(0.5, 1.4))],
           NEAR_BLACK, max(0.7, widths[k] * 0.08), pressure=0.2)
    if snow_on:
        st([(x - base_w * 1.6, y + 1), (x, y - 3), (x + base_w * 1.2, y + 2)], SNOW, [3, 5, 3], pressure=0.3)
    n_limbs = int(5 + height / 90)
    for k in range(n_limbs):
        t = 0.42 + 0.56 * (k / n_limbs) + r.uniform(-0.03, 0.03)
        i = min(int(t * (n - 1)), n - 2)
        f = t * (n - 1) - i
        bx = trunk[i][0] + (trunk[i + 1][0] - trunk[i][0]) * f
        by = trunk[i][1] + (trunk[i + 1][1] - trunk[i][1]) * f
        side = 1 if k % 2 else -1
        ang = -np.pi / 2 + side * r.uniform(0.45, 1.15) * spread + lean
        ln = height * r.uniform(0.16, 0.34) * (1.25 - t * 0.7)
        grow(bx, by, ang, ln, max(1.0, widths[i] * r.uniform(0.35, 0.55)), 1)
    grow(trunk[-1][0], trunk[-1][1], -np.pi / 2 + lean + r.normal(0, 0.1), height * 0.15, widths[-1], 2)
    return snow_lines, perches


def willow(x, y, h, seed=0, far=0.0):
    """A pollarded willow: a gnarled stump crowned with a fountain of fine rods."""
    r = np.random.default_rng(seed)
    tw = h * 0.085
    top = np.array([x + r.normal(0, h * 0.05), y - h * 0.4])
    trunk = atmos(TRUNK, far)
    st([(x, y), (x + r.normal(0, tw * 0.5), y - h * 0.2), top], trunk, [tw * 1.3, tw * 0.9, tw * 1.15],
       pressure=0.35)
    st([(x - tw * 0.35, y), (top[0] - tw * 0.4, top[1] + tw)], atmos(TRUNK_LIGHT, far), max(0.8, tw * 0.2),
       pressure=0.2)
    dab(top[0], top[1] + tw * 0.2, trunk, tw * 1.35)
    rods = [atmos(WILLOW, far), atmos(BRAMBLE, far), atmos(TWIG, far)]
    for _ in range(int(35 + h * 0.9)):
        a = -np.pi / 2 + float(np.clip(r.normal(0, 0.7), -1.45, 1.45))
        ln = h * r.uniform(0.35, 0.9)
        sx, sy = top + [r.normal(0, tw * 0.5), r.normal(0, tw * 0.25)]
        bend = 0.35 * np.sign(np.cos(a)) * r.uniform(0, 1)        # rods arch outward
        mx, my = sx + ln * 0.5 * np.cos(a), sy + ln * 0.5 * np.sin(a)
        st([(sx, sy), (mx, my), (mx + ln * 0.5 * np.cos(a + bend), my + ln * 0.5 * np.sin(a + bend))],
           mix((rods[int(r.integers(3))], 1), (MEDIUM, 1.0)), [1.0, 0.6], pressure=0.2)
    st([top + [-tw * 0.7, -tw * 0.4], top + [tw * 0.7, -tw * 0.6]], atmos(SNOW, far), max(1, tw * 0.55),
       pressure=0.2)


def bush(x, y, h, far=0.0, col=None):
    col = col or TWIG
    for _ in range(int(8 + h / 2)):
        a = -np.pi / 2 + rng.normal(0, 0.7)
        ln = h * rng.uniform(0.4, 1.0)
        st([(x + rng.normal(0, h * 0.2), y), (x + ln * np.cos(a), y + ln * np.sin(a))], atmos(col, far),
           [max(0.8, h * 0.05), 0.5], pressure=0.2)


def reeds(x, y, h, n=6, far=0.0):
    for _ in range(n):
        a = -np.pi / 2 + rng.normal(0, 0.25)
        ln = h * rng.uniform(0.5, 1.0)
        bx = x + rng.normal(0, h * 0.3)
        st([(bx, y), (bx + ln * np.cos(a), y + ln * np.sin(a))], atmos(REED, far), [max(0.7, h * 0.05), 0.4],
           pressure=0.2)


def fence(pts, h, far=0.0, wattle=True):
    pts = resample(pts, max(2, int(length(pts) / max(4, h * 0.6))))
    dk = atmos(DARK, far)
    for px, py in pts:
        st([(px, py + 1), (px + rng.normal(0, 0.4), py - h)], dk, max(0.8, h * 0.09), pressure=0.2)
    if wattle:
        for fy in (0.3, 0.55, 0.8):
            st([(px, py - h * fy) for px, py in pts], atmos(WOOD, far), max(0.8, h * 0.12), pressure=0.2)
    else:
        for fy in (0.45, 0.85):
            st([(px, py - h * fy) for px, py in pts], dk, max(0.7, h * 0.07), pressure=0.2)
    st([(px, py - h - 0.8) for px, py in pts], atmos(WHITE, far), max(0.8, h * 0.1), pressure=0.2)


def haystack(x, y, s, far=0.0):
    body = blob(x, y - 16 * s, 16 * s, 17 * s, n=24, rough=0.08)
    body = body[body[:, 1] <= y]
    body = np.vstack([body, [(x + 16 * s, y), (x - 16 * s, y)]])
    fill(body, atmos(THATCH, far), width=max(1.2, 4 * s), spacing=max(0.7, 2 * s), angle=np.pi / 2,
         seg_len=(200, 300), wobble=0.2, overrun=-0.3)
    cap = blob(x - 2 * s, y - 24 * s, 15 * s, 10 * s, n=20, rough=0.1)
    fill(cap, atmos(SNOW, far), width=max(1.2, 3 * s), spacing=max(0.7, 1.5 * s), wobble=0.2, overrun=-0.3,
         seg_len=(100, 200))
    st([(x + 10 * s, y - 2), (x + 14 * s, y - 20 * s)], atmos(shade_of(THATCH, 0.6), far), max(1, 4 * s))
    st([(x, y - 33 * s), (x, y - 44 * s)], atmos(DARK, far), max(0.8, 1.5 * s))


def woodpile(x, y, s):
    for row in range(4):
        for k in range(5 - row):
            cx = x + (k + row * 0.5) * 7 * s
            cy = y - row * 6 * s
            dab(cx, cy, WOOD, 6 * s)
            dab(cx - 1 * s, cy - 1 * s, light_of(WOOD, 0.4), 2.2 * s)
    st([(x - 3 * s, y - 25 * s), (x + 33 * s, y - 25 * s)], SNOW, 4 * s)


# ===========================================================================
# People, animals, birds
# ===========================================================================
def person(x, y, h, coat, legs=None, facing=1, pose="walk", hat="hood", hat_col=None, skirt=False,
           apron=False, lean=0.0, props=(), shadow=True, arms=None):
    """A Bruegel peasant, built from rounded overlapping strokes and lit from the
    left: stocky body, heavy tunic or skirt, hood, cap, hat or white headcloth.
    (x, y) is the point between the feet, h the standing height in px."""
    if h < 14:
        return tiny_person(x, y, h, coat, facing=facing, pose=pose, skirt=skirt)
    u = h / 100.0
    f = facing if facing != 0 else 1
    big = h > 45
    if legs is None:
        legs = C_BLACK if rng.random() < 0.45 else pick([C_GREY, C_RED, C_BROWN, C_WHITE, C_BLUE])
    if hat_col is None:
        hat_col = coat if hat == "hood" else pick([C_BLACK, C_RED, DARK, C_GREY])
    dk_coat, lt_coat = shade_of(coat, 0.6), light_of(coat, 0.3)
    if shadow:
        st([(x - 8 * u, y + 1.5 * u), (x + 40 * u, y + 7 * u)], SNOW_SHADOW, [10 * u, 3 * u], pressure=0.2,
           mixing=0.3)
    hip = np.array([x + f * lean * 16 * u, y - 47 * u])
    sho = np.array([x + f * lean * 44 * u, y - 81 * u])
    head = sho + [f * (3 + lean * 9) * u, -12 * u]
    # ---- legs and shoes
    if pose == "walk":
        feet = [np.array([x + f * 11 * u, y]), np.array([x - f * 10 * u, y - 1 * u])]
    elif pose == "skate":
        feet = [np.array([x + f * 8 * u, y]), np.array([x - f * 24 * u, y - 8 * u])]
    elif pose == "bend":
        feet = [np.array([x + f * 7 * u, y]), np.array([x - f * 11 * u, y])]
    else:
        feet = [np.array([x - 5 * u, y]), np.array([x + 5 * u, y])]
    if not skirt or pose == "skate":
        for k, ft in enumerate(feet):
            top = hip + [(4 if k == 0 else -4) * u * f, 2 * u]
            knee = (top + ft) / 2 + [f * 4 * u, -2 * u]
            leg_col = legs if k == 0 else shade_of(legs, 0.25)
            st([top, knee, ft + [0, -3 * u]], leg_col, [10 * u, 7.5 * u, 6 * u], pressure=0.3)
            if big:
                st([top + [-f * 3 * u, 2 * u], knee + [-f * 3 * u, 0], ft + [-f * 2 * u, -4 * u]], shade_of(legs, 0.5),
                   max(0.8, 2 * u), pressure=0.2)
            st([ft + [-f * 3 * u, -2.5 * u], ft + [f * 6 * u, -1.5 * u]], C_BLACK, max(1.0, 5.5 * u), pressure=0.25)
            if pose == "skate":
                st([ft + [-f * 9 * u, 1 * u], ft + [f * 10 * u, 0.5 * u], ft + [f * 12 * u, -2 * u]], DARK,
                   max(0.7, 1.6 * u), pressure=0.2)
    # ---- body: rounded, from overlapping tapered strokes
    if skirt:
        hem_y, hem_w = y - 3 * u, 48 * u
    else:
        hem_y, hem_w = hip[1] + 13 * u, 36 * u
    hem_c = np.array([hip[0] + f * lean * 5 * u, hem_y])
    for off in (-1, 0, 1):
        top = sho + [off * 8 * u, 3 * u]
        mid = hip + [off * 11 * u, 0]
        bot = hem_c + [off * hem_w * 0.33, -2 * u]
        col = coat if off != 1 else shade_of(coat, 0.15)
        st([top, mid, bot], col, [14 * u, 15 * u, 16 * u if not skirt else 18 * u], pressure=0.3)
    st([sho + [-13 * u, 5 * u], sho + [0, 1 * u], sho + [13 * u, 5 * u]], coat, [9 * u, 11 * u, 9 * u], pressure=0.3)
    # light on the left, shade on the right, folds in the cloth
    st([sho + [-10 * u, 6 * u], hip + [-11 * u, 0], hem_c + [-hem_w / 2 + 3 * u, -3 * u]], lt_coat,
       max(0.8, 3.5 * u), pressure=0.2)
    st([sho + [11 * u, 6 * u], hip + [11 * u, 0], hem_c + [hem_w / 2 - 3 * u, -3 * u]], dk_coat,
       max(1.0, 5 * u), pressure=0.25)
    if big:
        for k in (-0.15, 0.2):
            st([hip + [k * 20 * u, 4 * u], hem_c + [k * hem_w * 0.8, -3 * u]], shade_of(coat, 0.35),
               max(0.8, 1.6 * u), pressure=0.2)
    st([hem_c + [-hem_w / 2 + 2 * u, -2 * u], hem_c + [hem_w / 2 - 2 * u, -2 * u]],
       C_RED if (skirt and rng.random() < 0.45) else shade_of(coat, 0.45), max(1, 3.5 * u), pressure=0.2)
    st([hip + [-12 * u, -3 * u], hip + [12 * u, -3 * u]], C_BLACK, max(0.8, 2.8 * u), pressure=0.2)
    if not skirt and big and rng.random() < 0.6:       # a pouch and a knife at the belt
        dab(hip[0] - f * 9 * u, hip[1] + 2 * u, pick([C_BLACK, C_TAN, DARK]), max(1, 7 * u))
        st([hip + [-f * 3 * u, 0], hip + [-f * 1 * u, 12 * u]], C_GREY, max(0.7, 1.5 * u))
    if apron:
        ax = hip[0] + f * 4 * u
        st([(ax, hip[1] - 3 * u), (ax + f * 2 * u, (hip[1] + hem_y) / 2), (ax + f * 3 * u, hem_y - 4 * u)],
           C_WHITE, [12 * u, 15 * u, 17 * u], pressure=0.25)
        st([(ax + 5 * u, hip[1]), (ax + 6 * u, hem_y - 5 * u)], light_of(SNOW_SHADOW, 0.2), max(0.8, 1.5 * u))
    # ---- arms
    if arms is None:
        if pose == "walk":
            arms = [sho + [f * 12 * u, 36 * u], sho + [-f * 9 * u, 35 * u]]
        elif pose == "skate":
            arms = ([sho + [-f * 6 * u, 32 * u], sho + [-f * 2 * u, 33 * u]] if rng.random() < 0.5 else
                    [sho + [f * 32 * u, 6 * u], sho + [-f * 30 * u, 2 * u]])
        elif pose == "carry":
            arms = [sho + [f * 6 * u, -4 * u], sho + [-f * 4 * u, -6 * u]]
        elif pose == "bend":
            arms = [sho + [f * 22 * u, 32 * u], sho + [f * 15 * u, 36 * u]]
        else:
            arms = [sho + [f * 8 * u, 34 * u], sho + [-f * 9 * u, 34 * u]]
    arms = [np.asarray(a, float) for a in arms]
    for k, hand in enumerate(arms):
        side = f if k == 0 else -f
        s0 = sho + [side * 12 * u, 5 * u]
        elbow = (s0 + hand) / 2 + [side * 4 * u, 3 * u]
        col = coat if k == 0 else shade_of(coat, 0.3)
        st([s0, elbow, hand], col, [8.5 * u, 7 * u, 5.5 * u], pressure=0.25)
        if big:
            st([elbow + [side * 2.5 * u, 0], hand + [side * 1.5 * u, -1 * u]], shade_of(col, 0.4), max(0.8, 1.5 * u))
        dab(hand[0], hand[1], FLESH, max(1.0, 5.5 * u))
    # ---- head and headgear
    dab(head[0], head[1], FLESH, max(1.2, 16 * u))
    if f != 0 and big:
        dab(head[0] + f * 6 * u, head[1] + 1 * u, mix((FLESH, 1), (C_DULL_RED, 0.3)), max(1, 4 * u))      # nose
        dab(head[0] + f * 2 * u, head[1] + 3 * u, mix((FLESH, 1), (C_DULL_RED, 0.25)), max(1, 5 * u))     # cheek
    if hat == "round":
        dab(head[0] - f * 1 * u, head[1] - 4 * u, hat_col, max(1, 13 * u))
        st([head + [-11 * u, -3 * u], head + [11 * u, -4 * u]], hat_col, max(0.8, 3.5 * u), pressure=0.25)
    elif hat == "hood":      # a hood over the head with a short cape on the shoulders
        dab(head[0] - f * 2 * u, head[1] - 2 * u, hat_col, max(1.2, 17 * u))
        st([head + [-f * 6 * u, 4 * u], sho + [-f * 12 * u, 7 * u], sho + [f * 9 * u, 6 * u]], hat_col,
           [11 * u, 12 * u, 9 * u], pressure=0.25)
        dab(head[0] + f * 3.5 * u, head[1] + 1.5 * u, FLESH, max(1, 8 * u))
        if big:
            st([head + [f * 6 * u, -6 * u], head + [f * 7 * u, 6 * u]], shade_of(hat_col, 0.4), max(0.8, 1.8 * u))
    elif hat == "cap":
        dab(head[0] - f * 2 * u, head[1] - 1 * u, pick([DARK, C_BROWN, C_TAN]), max(1, 12 * u))      # hair
        dab(head[0] + f * 3 * u, head[1] + 1.5 * u, FLESH, max(1, 9 * u))
        st([head + [-7 * u, -6 * u], head + [7 * u, -7 * u]], hat_col, max(1, 7 * u), pressure=0.25)
    elif hat == "kerchief":   # a crisp white headcloth: a squarish fold over the head, down to the nape
        dab(head[0] - f * 1.5 * u, head[1] - 3 * u, C_WHITE, max(1.2, 18 * u))
        st([head + [-f * 6 * u, -5 * u], head + [-f * 7 * u, 5 * u]], C_WHITE, max(1.2, 12 * u), pressure=0.25)
        st([head + [-f * 9 * u, -4 * u], head + [-f * 10 * u, 5 * u]], mix((C_WHITE, 2), (SNOW_SHADOW, 2)),
           max(0.8, 2.2 * u), pressure=0.2)
        if big:
            st([head + [-f * 1 * u, -11 * u], head + [-f * 5 * u, -2 * u]], mix((C_WHITE, 2), (SNOW_SHADOW, 2)),
               max(0.8, 1.6 * u), pressure=0.2)
        dab(head[0] + f * 4 * u, head[1] + 2 * u, FLESH, max(1, 8 * u))
    elif hat == "fur":
        dab(head[0] + f * 3 * u, head[1] + 1.5 * u, FLESH, max(1, 9 * u))
        st([head + [-9 * u, -6 * u], head + [9 * u, -7 * u]], DARK, max(1, 11 * u), pressure=0.25)
    # ---- things carried
    for prop in props:
        if prop == "bundle":      # a faggot of sticks slung on the back
            bx, by = sho + [-f * 16 * u, 14 * u]
            for k in range(12):
                o = (k - 6) * 1.6 * u
                e0, e1 = rng.uniform(-5, 4) * u, rng.uniform(-4, 7) * u      # ragged, fanning ends
                st([(bx - f * (16 * u + e0) + o * 1.4, by - 22 * u - e0),
                    (bx + f * (10 * u + e1) + o * 0.7, by + 22 * u + e1)],
                   pick([WOOD, TWIG, light_of(WOOD, 0.3)]), max(0.8, 2.0 * u), pressure=0.2)
            st([(bx - 9 * u, by - 4 * u), (bx + 9 * u, by + 2 * u)], C_BLACK, max(0.8, 1.8 * u))
            st([(bx - f * 16 * u - 8 * u, by - 24 * u), (bx - f * 16 * u + 8 * u, by - 22 * u)], SNOW, max(0.8, 2 * u))
        elif prop == "axe":
            a0 = sho + [f * 6 * u, -4 * u]
            a1 = sho + [-f * 32 * u, 24 * u]
            st([a0, a1], WOOD, max(0.8, 2.6 * u))
            st([a1 + [-f * 2 * u, -6 * u], a1 + [f * 3 * u, 5 * u]], C_GREY, max(1, 6 * u))
        elif prop == "stick":
            hand = arms[0]
            st([hand + [f * 4 * u, -14 * u], (hand[0] + f * 12 * u, y + 1)], WOOD, max(0.7, 2.2 * u))
        elif prop == "basket":
            hand = arms[1]
            st([hand + [-9 * u, 5 * u], hand + [9 * u, 5 * u]], C_TAN, max(1.2, 10 * u))
            st([hand + [-8 * u, 0], hand + [0, -7 * u], hand + [8 * u, 0]], WOOD, max(0.7, 1.6 * u))
        elif prop == "yoke":
            st([sho + [-28 * u, 0], sho + [28 * u, -1 * u]], WOOD, max(0.8, 3.2 * u))
            for side in (-1, 1):
                px = sho[0] + side * 27 * u
                st([(px, sho[1]), (px, sho[1] + 22 * u)], DARK, max(0.6, 1 * u))
                st([(px - 7 * u, sho[1] + 29 * u), (px + 7 * u, sho[1] + 29 * u)], WOOD, max(1.2, 11 * u))
        elif prop == "kolf":
            hand = arms[0]
            end = (hand[0] + f * 18 * u, y)
            st([hand, end], WOOD, max(0.7, 2 * u))
            st([end, (end[0] + f * 7 * u, end[1] - 1 * u)], WOOD, max(0.8, 2.8 * u))
            dab(end[0] + f * 17 * u, y - 1 * u, C_WHITE, max(1, 4 * u))
        elif prop == "sack":
            st([sho + [-f * 10 * u, 0], sho + [-f * 16 * u, 26 * u]], C_TAN, [18 * u, 22 * u])
            st([sho + [-f * 4 * u, 4 * u], sho + [-f * 10 * u, 28 * u]], shade_of(C_TAN, 0.3), max(1, 4 * u))
        elif prop == "shovel":
            hand = arms[0]
            st([hand + [-f * 14 * u, -14 * u], hand + [f * 16 * u, 16 * u]], WOOD, max(0.8, 2.2 * u))
            st([hand + [f * 14 * u, 14 * u], hand + [f * 24 * u, 22 * u]], WOOD, max(1, 8 * u))


def tiny_person(x, y, h, coat, facing=1, pose="walk", skirt=False):
    """A distant figure in four strokes."""
    u = h / 10.0
    lean = 0.25 * facing if pose == "skate" else 0.0
    if not skirt:
        st([(x - 0.6 * u * facing, y), (x, y - 4.2 * u)], C_BLACK, max(0.7, 0.9 * u), pressure=0.25)
        st([(x + 0.8 * u * facing, y), (x, y - 4.2 * u)], C_BLACK, max(0.7, 0.9 * u), pressure=0.25)
    st([(x, y - (0.5 if skirt else 3.8) * u), (x + lean * 4 * u, y - 7.6 * u)], coat,
       [max(1.0, (2.8 if skirt else 2.0) * u), max(0.9, 1.6 * u)], pressure=0.3)
    dab(x + lean * 5 * u, y - 8.8 * u, pick([DARK, C_BLACK, C_WHITE, FLESH]), max(0.9, 1.4 * u))
    if pose == "skate" and rng.random() < 0.5:
        st([(x - 2.2 * u, y - 6.5 * u), (x + 2.2 * u, y - 6.8 * u)], coat, max(0.6, 0.6 * u), pressure=0.2)
    st([(x + 0.5 * u, y + 0.3 * u), (x + 4 * u, y + 0.8 * u)], ICE_DARK, max(0.6, 0.8 * u), pressure=0.15)


def fallen_skater(x, y, h, coat):
    u = h / 100.0
    st([(x - 30 * u, y - 4 * u), (x + 10 * u, y - 10 * u)], coat, [18 * u, 20 * u])
    dab(x - 36 * u, y - 4 * u, FLESH, 12 * u)
    st([(x - 40 * u, y - 8 * u), (x - 34 * u, y - 12 * u)], C_BLACK, 9 * u)
    st([(x + 8 * u, y - 10 * u), (x + 24 * u, y - 34 * u)], C_BLACK, [7 * u, 5 * u])
    st([(x + 6 * u, y - 6 * u), (x + 30 * u, y - 18 * u)], C_GREY, [7 * u, 5 * u])
    st([(x + 20 * u, y - 36 * u), (x + 30 * u, y - 34 * u)], DARK, 2 * u)
    st([(x - 22 * u, y - 14 * u), (x - 14 * u, y - 34 * u)], coat, [6 * u, 5 * u])
    dab(x - 14 * u, y - 35 * u, FLESH, 4 * u)
    st([(x - 40 * u, y + 2 * u), (x + 34 * u, y + 2 * u)], ICE_DARK, 4 * u, pressure=0.2)


def chair_sled(x, y, h, facing=1):
    """A child on a chair-sled pushed by an adult."""
    u = h / 100.0
    f = facing
    cx = x + f * 30 * u
    st([(cx - 16 * u, y), (cx + 18 * u, y), (cx + 22 * u, y - 4 * u)], DARK, max(0.8, 2.2 * u))
    st([(cx - 12 * u, y - 4 * u), (cx - 12 * u, y - 26 * u)], WOOD, max(0.8, 2.5 * u))
    st([(cx - 12 * u, y - 12 * u), (cx + 12 * u, y - 12 * u)], WOOD, max(1, 5 * u))
    person(cx, y - 12 * u, h * 0.42, pick([C_RED, C_BLUE, C_TAN]), facing=f, pose="stand", hat="cap", shadow=False)
    person(x, y, h, pick(COATS), facing=f, pose="walk", lean=0.25,
           arms=[(cx - 14 * u, y - 30 * u), (cx - 16 * u, y - 28 * u)])


def dog(x, y, h, facing=1, paint=None):
    """A lean hunting dog; h is roughly its length in px."""
    paint = paint or pick([DOG_BROWN, DARK, C_WHITE, DOG_BROWN, C_TAN])
    u = h / 100.0
    f = facing
    dkp = shade_of(paint, 0.45)
    rear, chest = np.array([x - f * 30 * u, y - 50 * u]), np.array([x + f * 22 * u, y - 54 * u])
    st([(x - f * 10 * u, y + 1), (x + f * 40 * u, y + 3)], SNOW_SHADOW, max(1, 6 * u), pressure=0.2)
    # far legs, then the body, then near legs
    for leg, dx in ((dkp, -26), (dkp, 18)):
        st([(x + f * dx * u, y - 46 * u), (x + f * (dx + 3) * u, y - 22 * u), (x + f * (dx + 1) * u, y)], leg,
           [6 * u, 4 * u, 3.5 * u])
    st([rear, (x - f * 4 * u, y - 46 * u), chest], paint, [20 * u, 18 * u, 24 * u], pressure=0.3)
    st([(x - f * 14 * u, y - 40 * u), (x + f * 16 * u, y - 38 * u)], dkp, max(1, 6 * u))       # belly shadow
    st([rear + [0, -8 * u], chest + [0, -10 * u]], light_of(paint, 0.4), max(0.8, 3 * u))       # lit back
    hind = [rear + [f * 2 * u, 6 * u], (x - f * 38 * u, y - 24 * u), (x - f * 30 * u, y - 10 * u), (x - f * 32 * u, y)]
    st(hind, paint, [9 * u, 6 * u, 4 * u, 3.5 * u])
    st([chest + [-f * 2 * u, 8 * u], (x + f * 24 * u, y - 24 * u), (x + f * 26 * u, y)], paint, [8 * u, 5 * u, 4 * u])
    # neck, head with a long muzzle, ear, curled tail
    neck_top = chest + [f * 12 * u, -20 * u]
    st([chest + [0, -4 * u], neck_top], paint, [14 * u, 10 * u])
    dab(neck_top[0], neck_top[1], paint, 12 * u)
    st([neck_top + [f * 2 * u, 0], neck_top + [f * 18 * u, 6 * u]], paint, [9 * u, 4 * u])
    dab(neck_top[0] + f * 18 * u, neck_top[1] + 6 * u, C_BLACK, max(1, 3 * u))
    st([neck_top + [-f * 2 * u, -3 * u], neck_top + [-f * 8 * u, 6 * u]], dkp, [6 * u, 3 * u])
    st([rear + [-f * 2 * u, -4 * u], rear + [-f * 14 * u, -20 * u], rear + [-f * 6 * u, -32 * u]], paint,
       [5 * u, 3.5 * u, 1.5 * u])


def horse_sleigh(x, y, h, facing=-1):
    """A horse drawing a sleigh with its driver."""
    u = h / 100.0
    f = facing
    hx = x + f * 60 * u
    st([(hx - f * 28 * u, y - 52 * u), (hx + f * 16 * u, y - 56 * u)], HORSE, [30 * u, 28 * u], pressure=0.3)
    st([(hx + f * 14 * u, y - 60 * u), (hx + f * 30 * u, y - 86 * u)], HORSE, [14 * u, 9 * u])
    st([(hx + f * 29 * u, y - 87 * u), (hx + f * 40 * u, y - 76 * u)], HORSE, [9 * u, 6 * u])
    st([(hx + f * 22 * u, y - 82 * u), (hx + f * 16 * u, y - 64 * u)], C_BLACK, 4 * u)
    for dx, lift in ((-24, 0), (-17, 6), (8, 0), (15, -4)):
        st([(hx + f * dx * u, y - 44 * u), (hx + f * (dx + 3) * u, y - 22 * u),
            (hx + f * (dx + 1 + lift * 0.2) * u, y - lift * u)], shade_of(HORSE, 0.2), [7 * u, 5 * u, 4 * u])
    st([(hx - f * 30 * u, y - 54 * u), (hx - f * 38 * u, y - 30 * u)], C_BLACK, [6 * u, 3 * u])
    st([(hx - f * 26 * u, y - 70 * u), (hx + f * 6 * u, y - 72 * u)], light_of(HORSE, 0.4), 3 * u)
    st([(hx + f * 10 * u, y - 62 * u), (x + f * 4 * u, y - 30 * u)], DARK, max(0.8, 2 * u))
    st([(hx - f * 10 * u, y - 46 * u), (x + f * 4 * u, y - 26 * u)], DARK, max(0.8, 2 * u))
    body = [(x - f * 40 * u, y - 10 * u), (x + f * 6 * u, y - 10 * u), (x + f * 4 * u, y - 34 * u),
            (x - f * 34 * u, y - 38 * u)]
    fill(body, C_DULL_RED, width=6 * u, spacing=3 * u, seg_len=(300, 400), wobble=0.1, overrun=-0.3)
    st([(x - f * 44 * u, y - 2 * u), (x + f * 6 * u, y - 2 * u), (x + f * 14 * u, y - 14 * u)], DARK, max(1, 3 * u))
    person(x - f * 16 * u, y - 26 * u, h * 0.7, pick([C_BLACK, C_GREY, C_BROWN]), facing=f, pose="stand",
           hat="fur", shadow=False, arms=[(x + f * 8 * u, y - 70 * u), (x + f * 4 * u, y - 66 * u)])
    st([(x + f * 8 * u, y - 70 * u), (hx + f * 20 * u, y - 70 * u)], DARK, max(0.6, 1 * u))
    st([(x - f * 44 * u, y + 2 * u), (hx + f * 30 * u, y + 3 * u)], ICE_DARK, max(1, 6 * u), pressure=0.2)


def crow_perched(x, y, s, paint=None):
    paint = paint or NEAR_BLACK
    st([(x - 2 * s, y + 1 * s), (x + 6 * s, y - 5 * s)], paint, [9 * s, 7 * s])
    st([(x - 3 * s, y + 2 * s), (x - 10 * s, y + 9 * s)], paint, [4 * s, 1.0])
    dab(x + 7 * s, y - 7 * s, paint, 5.5 * s)
    st([(x + 9 * s, y - 7 * s), (x + 13 * s, y - 6 * s)], paint, [2 * s, 0.6])
    st([(x, y + 3 * s), (x + 1 * s, y + 6 * s)], paint, 0.8)


def bird_flying(x, y, s, paint=None, magpie=False, phase=None):
    paint = paint or NEAR_BLACK
    up = rng.uniform(-0.4, 0.8) if phase is None else phase
    lw = [(x - 30 * s, y - 14 * s * up - 2 * s), (x - 15 * s, y - 10 * s * up - 3 * s), (x - 2 * s, y)]
    rw = [(x + 2 * s, y), (x + 15 * s, y - 12 * s * up - 3 * s), (x + 30 * s, y - 16 * s * up - 1 * s)]
    st(lw, paint, [max(0.8, 2.5 * s), max(1, 5 * s), max(1, 6 * s)])
    st(rw, paint, [max(1, 6 * s), max(1, 5 * s), max(0.8, 2.5 * s)])
    st([(x - 6 * s, y + 1 * s), (x + 9 * s, y)], paint, max(1, 6.5 * s))
    dab(x + 11 * s, y - 1 * s, paint, max(1, 4.5 * s))
    st([(x - 6 * s, y + 1 * s), (x - (24 if magpie else 16) * s, y + 4 * s)], paint, [max(1, 4 * s), max(0.7, 2 * s)])
    if magpie:
        st([(x - 18 * s, y - 9 * s * up - 2 * s), (x - 8 * s, y - 5 * s * up - 1 * s)], WHITE, max(1, 2.5 * s))
        st([(x + 8 * s, y - 6 * s * up - 2 * s), (x + 18 * s, y - 10 * s * up - 2 * s)], WHITE, max(1, 2.5 * s))
        st([(x - 2 * s, y + 2 * s), (x + 5 * s, y + 2 * s)], WHITE, max(1, 2.5 * s))


def fire(x, y, s):
    """A wood fire in the snow: logs, many small tongues of flame, glow and smoke."""
    st([(x - 30 * s, y + 4 * s), (x + 30 * s, y + 6 * s)], mix((FIRE, 1), (MEDIUM, 4)), 10 * s, pressure=0.2)
    for k in range(5):
        st([(x - 18 * s + 9 * k * s, y + 4 * s), (x - 8 * s + 5 * k * s, y - 3 * s)], DARK, 3.5 * s)
    for k in range(18):
        a = -np.pi / 2 + rng.normal(0, 0.3)
        ln = rng.uniform(6, 24) * s
        bx = x + rng.normal(0, 6 * s)
        pts = [(bx, y), (bx + ln * 0.5 * np.cos(a) + rng.normal(0, 1.5 * s), y + ln * 0.5 * np.sin(a)),
               (bx + ln * np.cos(a) + rng.normal(0, 2 * s), y + ln * np.sin(a))]
        st(pts, pick([EMBER, FIRE, FIRE, FLAME_CORE]), [3.2 * s, 2.0 * s, 0.8], pressure=0.3, mixing=0.1)
    for _ in range(8):
        dab(x + rng.normal(0, 8 * s), y - rng.uniform(26, 44) * s, FLAME_CORE, max(1, 1.2 * s))
    smoke(x, y - 30 * s, scale=0.7 * s)


def barrel(x, y, s, snow=True):
    body = blob(x, y - 12 * s, 9 * s, 12 * s, n=20, rough=0.02)
    fill(body, WOOD, width=max(1.2, 3 * s), spacing=max(0.7, 1.5 * s), angle=np.pi / 2, seg_len=(200, 300),
         wobble=0, overrun=-0.3)
    st([(x + 5 * s, y - 2 * s), (x + 6 * s, y - 22 * s)], shade_of(WOOD, 0.5), max(1, 3 * s))
    for hy in (0.25, 0.75):
        st([(x - 9 * s, y - 24 * s * hy), (x, y - 24 * s * hy + 1.5 * s), (x + 9 * s, y - 24 * s * hy)], DARK,
           max(0.8, 1.4 * s))
    if snow:
        st([(x - 8 * s, y - 24 * s), (x + 8 * s, y - 24 * s)], SNOW, max(1, 4 * s))


def cart(x, y, s, facing=1):
    """A two-wheeled farm cart with big spoked wheels, loaded with barrels."""
    f = facing
    st([(x - f * 60 * s, y - 34 * s), (x + f * 40 * s, y - 40 * s)], WOOD, max(1, 8 * s))        # bed
    st([(x - f * 60 * s, y - 44 * s), (x + f * 40 * s, y - 50 * s)], shade_of(WOOD, 0.3), max(1, 3 * s))
    for k in range(6):                                                                      # rails
        px = x - f * 56 * s + f * k * 18 * s
        st([(px, y - 36 * s), (px, y - 52 * s)], shade_of(WOOD, 0.3), max(0.8, 2 * s))
    st([(x + f * 38 * s, y - 38 * s), (x + f * 100 * s, y - 4 * s)], WOOD, max(0.8, 3 * s))  # shafts
    st([(x + f * 38 * s, y - 42 * s), (x + f * 104 * s, y - 10 * s)], shade_of(WOOD, 0.2), max(0.8, 3 * s))
    for bx in (-40, -18, 6):
        barrel(x + f * bx * s, y - 44 * s, s * 0.95)
    st([(x - f * 62 * s, y - 70 * s), (x + f * 20 * s, y - 72 * s)], SNOW, max(1, 4 * s))
    for k, (wx, r) in enumerate(((x - f * 6 * s, 26 * s), (x - f * 14 * s, 24 * s))):
        col = DARK if k == 0 else shade_of(DARK, 0.4)
        if k == 1:
            continue
        for a in np.linspace(0, np.pi, 6, endpoint=False):
            st([(wx - r * np.cos(a), y - r - r * np.sin(a)), (wx + r * np.cos(a), y - r + r * np.sin(a))], col,
               max(0.8, 2 * s))
        ring = [(wx + r * np.cos(a), y - r + r * np.sin(a)) for a in np.linspace(0, 2 * np.pi, 28)]
        st(ring, col, max(1.2, 4 * s), pressure=0.3)
        dab(wx, y - r, col, max(1.5, 7 * s))
        st([(wx + r * np.cos(a), y - r + r * np.sin(a) - 1.5 * s) for a in np.linspace(np.pi * 1.15, np.pi * 1.85, 8)],
           SNOW, max(1, 2.5 * s))
    st([(x - f * 70 * s, y + 2 * s), (x + f * 60 * s, y + 3 * s)], SNOW_SHADOW, max(1, 8 * s), pressure=0.2)


def chicken(x, y, s):
    col = pick([C_WHITE, C_TAN, DOG_BROWN, C_BLACK])
    st([(x - 6 * s, y - 6 * s), (x + 5 * s, y - 7 * s)], col, max(1, 8 * s))
    st([(x - 6 * s, y - 7 * s), (x - 11 * s, y - 13 * s)], shade_of(col, 0.3), [max(1, 4 * s), 0.8])
    dab(x + 7 * s, y - 11 * s, col, max(1, 4.5 * s))
    dab(x + 7.5 * s, y - 14 * s, C_RED, max(0.8, 2 * s))
    for dx in (-1, 2):
        st([(x + dx * s, y - 3 * s), (x + dx * s, y)], OCHRE, max(0.6, 0.8 * s))


def sledge(x, y, s, ang=-0.33):
    """A low sledge loaded with bundles of faggots. (x, y) is its front end,
    the load trails behind it along direction `ang`; runners curl up in front."""
    c, sn = np.cos(ang), np.sin(ang)

    def p(t, o=0.0):        # t along the sledge, o up from the runners
        return (x + c * t * s, y + sn * t * s - o * s)
    st([(x - 4 * s, y - 10 * s), p(0), p(84)], DARK, max(1, 3 * s))
    st([p(2, 6), p(82, 6)], WOOD, max(1, 2.5 * s))
    for b in range(3):                     # three bundles of sticks with ragged, fanning ends
        for k in range(12):
            o = 8 + b * 7 + rng.uniform(0, 6)
            t0, t1 = rng.uniform(-6, 8), rng.uniform(72, 92)
            st([p(t0, o + rng.uniform(-3, 3)), p((t0 + t1) / 2, o), p(t1, o + rng.uniform(-4, 4))],
               pick([WOOD, TWIG, light_of(WOOD, 0.3)]), max(0.8, 1.5 * s), pressure=0.2)
    for t in (14, 42, 68):                 # ropes
        st([p(t, 4), p(t + 2, 31)], C_BLACK, max(0.8, 1.4 * s))
    st([p(6, 32), p(78, 32)], SNOW, max(1, 3 * s))
    st([p(-6, -2), p(90, -3)], SNOW_SHADOW, max(1, 8 * s), pressure=0.2)


def well_sweep(x, y, s):
    """A Flemish well with a long sweep pole."""
    st([(x - 12 * s, y), (x + 12 * s, y)], STONE_SHADE, 10 * s)
    st([(x - 12 * s, y - 6 * s), (x + 12 * s, y - 6 * s)], SNOW, 4 * s)
    st([(x + 22 * s, y + 2 * s), (x + 24 * s, y - 50 * s)], WOOD, 3 * s)            # the post
    st([(x - 30 * s, y - 70 * s), (x + 70 * s, y - 30 * s)], WOOD, 2.2 * s)          # the sweep
    st([(x - 28 * s, y - 70 * s), (x - 2 * s, y - 8 * s)], DARK, 1 * s)              # the rope
    st([(x + 60 * s, y - 34 * s), (x + 72 * s, y - 28 * s)], DARK, 5 * s)            # counterweight
    st([(x - 6 * s, y - 10 * s), (x + 2 * s, y - 10 * s)], WOOD, 5 * s)              # bucket


# ===========================================================================
# Composition
# ===========================================================================
def paint_far_village():
    stage("distant villages, hedgerows and trees")
    rows = [((20, 392), (520, 405), 9), ((700, 386), (1200, 380), 8), ((1240, 395), (1500, 388), 9),
            ((40, 455), (330, 470), 16), ((880, 455), (1180, 448), 15), ((1180, 470), (1480, 455), 18),
            ((1000, 520), (1260, 545), 24), ((-10, 540), (150, 560), 22)]
    for p0, p1, h in rows:
        fence([p0, ((p0[0] + p1[0]) / 2, (p0[1] + p1[1]) / 2 + 2), p1], h * 0.18, far=depth_at(p0[1]) ** 0.7,
              wattle=False)
        tree_row(p0, p1, h)
    for x, y in [(70, 392), (118, 395), (160, 393), (420, 402), (468, 404), (512, 400), (740, 384),
                 (790, 386), (835, 383), (1000, 380), (1050, 382), (1100, 381), (1320, 392), (1380, 393)]:
        sc = rng.uniform(0.3, 0.42)
        building(x, y + 6, 22 * sc, 13 * sc, 11 * sc, 30 * sc, dirx=pick([-1, 1]), chimney=False,
                 windows=1, door=False, far=0.85)
    church(300, 404, 0.32, far=0.85)
    church(930, 383, 0.28, far=0.9)
    post_mill(1240, 372, 0.6, far=0.8)
    for x in (250, 610, 880, 1150, 1420):
        haystack(x + rng.normal(0, 20), 425 + rng.normal(0, 8), 0.35, far=0.75)


def bank_points(edge, outward, t0, t1, n, offset):
    """Points along a river bank edge, pushed `offset` px away from the water."""
    pts = resample(edge, 200)
    out = []
    for t in np.linspace(t0, t1, n):
        i = int(t * 199)
        p = pts[i]
        out.append(p + outward(p) * offset)
    return out


def paint_near_village():
    stage("village around the pond")
    # houses on the far bank of the pond (bottoms kept on the bank, above the ice)
    far_bank = [(345, 512, 0.62, 1, WALL, False), (398, 508, 0.55, -1, PLASTER, True),
                (450, 506, 0.7, 1, WALL, False), (525, 500, 0.58, 1, BRICK, False),
                (980, 506, 0.66, -1, PLASTER, True), (1045, 512, 0.7, 1, WALL, False),
                (1110, 520, 0.6, 1, BRICK, False), (1185, 528, 0.66, -1, WALL, True)]
    for x, y, s, dirx, wall, timber in far_bank:
        building(x, y, 46 * s, 32 * s, 30 * s, 64 * s, dirx=dirx, wall=wall, timber=timber,
                 kind="brick" if wall is BRICK else "cottage", stepped=(wall is BRICK),
                 smoke=rng.random() < 0.5, windows=2, far=0.35)
    church(820, 506, 0.95, dirx=1, far=0.3)
    # the left bank, nearer: farms
    for x, y, s, dirx, wall, timber in [(20, 585, 0.95, 1, WALL, True), (130, 600, 0.8, 1, PLASTER, False),
                                        (215, 575, 0.62, -1, WALL, True), (20, 690, 1.1, 1, BRICK, False)]:
        building(x, y, 46 * s, 34 * s, 32 * s, 70 * s, dirx=dirx, wall=wall, timber=timber,
                 kind="brick" if wall is BRICK else "cottage", stepped=(wall is BRICK), smoke=True, far=0.2)
    building(170, 520, 70, 30, 36, 90, dirx=-1, wall=WOOD, windows=0, chimney=False, far=0.3)   # barn
    mill(470, 662, 1.1)
    # the inn on the near bank, below the hill, with an outhouse
    building(430, 850, 120, 78, 70, 120, dirx=1, wall=WALL, timber=True, windows=3, smoke=True)
    # a farmstead in the middle field, its foot hidden by the lip of the hill
    building(985, 640, 58, 36, 42, 70, dirx=1, wall=WALL, timber=True, smoke=True, far=0.15)
    building(1110, 612, 80, 30, 46, 90, dirx=-1, wall=WOOD, windows=0, chimney=False, far=0.15)
    canvas.dry()

    stage("building details")
    for b in BUILDINGS:
        {"tower": tower_details, "wheel": wheel_details}.get(b["kind"], building_details)(b)
    # the inn sign hangs askew from a pole; a ladder leans on the side wall
    st([(432, 792), (388, 784)], DARK, 4)
    st([(394, 786), (392, 806)], DARK, 1.4)
    st([(386, 784), (382, 802)], DARK, 1.4)
    sign = [(374, 800), (402, 808), (400, 834), (372, 826)]
    fill(sign, C_WHITE, width=5, spacing=2.5, angle=0.25, seg_len=(200, 300), wobble=0.1, overrun=-0.3)
    st([(380, 822), (384, 812), (392, 810), (396, 816)], C_DULL_RED, 2.6)          # a red crown on the sign
    st([(381, 812), (386, 806), (390, 812), (394, 805), (397, 812)], C_DULL_RED, 1.6)
    st([(372, 800), (402, 808)], DARK, 1.6)
    st([(372, 826), (400, 834)], DARK, 1.2)
    for lx in (598, 611):
        st([(lx, 840), (lx + 14, 748)], WOOD, 3)
    for k in range(10):
        t = (k + 0.5) / 10
        st([(598 + 14 * t, 840 - 92 * t), (611 + 14 * t, 840 - 92 * t)], WOOD, 1.8)
    # fences, willows, bushes, haystacks and reeds around the village
    fence([(0, 712), (90, 722), (190, 715)], 12, far=0.15)
    fence([(560, 872), (640, 882), (720, 874)], 16)
    fence([(1000, 600), (1120, 590), (1200, 600)], 10, far=0.25)
    center = np.asarray(RIVER_NEAR_C, float)

    def away(p):
        i = np.argmin(np.linalg.norm(center - p, axis=1))
        v = p - center[i]
        return v / (np.linalg.norm(v) + 1e-9)

    for k, p in enumerate(bank_points(RIVER_NEAR_L, away, 0.1, 0.55, 4, 16)):
        willow(p[0] + rng.normal(0, 8), p[1], (45 + 60 * (p[1] - 600) / 300) * rng.uniform(0.8, 1.15),
               seed=100 + k, far=depth_at(p[1]) * 0.4)
    for k, p in enumerate(bank_points(RIVER_NEAR_R, away, 0.15, 0.4, 2, 20)):
        willow(p[0], p[1], 50 + 60 * (p[1] - 600) / 300, seed=200 + k, far=depth_at(p[1]) * 0.4)
    for x, y, h in [(600, 612, 40), (900, 606, 36), (1050, 596, 34)]:
        willow(x, y, h, seed=int(x * 7 + y), far=0.35)
    for x, y, h in [(1230, 520, 12), (1270, 528, 14), (900, 540, 12), (600, 535, 10), (300, 620, 14)]:
        bush(x, y, h, far=depth_at(y) * 0.5)
    for x, y in [(700, 470), (740, 474), (1240, 470)]:
        haystack(x, y, 0.7, far=0.45)
    haystack(250, 560, 1.0, far=0.25)
    woodpile(660, 866, 1.2)
    for x, y, h, sd in [(600, 505, 120, 31), (660, 508, 100, 32), (930, 500, 130, 33), (1220, 545, 150, 34),
                        (1300, 520, 120, 35), (100, 560, 140, 36), (300, 530, 110, 37), (770, 500, 110, 38)]:
        tree(x, y, h, base_w=max(2, h / 40), seed=sd, paint=atmos(MID_TREES, 0.3), depth=4, snow_on=False,
             lace_thin=2.5, fan=(1, 3))
    for x, y in [(1000, 592), (760, 612), (720, 610)]:
        reeds(x, y, 10, n=6, far=0.3)
    edge = resample(RIVER_NEAR_L, 100)
    for k in range(40):
        t = rng.uniform(0.05, 0.9)
        p = edge[int(t * 99)]
        reeds(p[0] - 4, p[1] - 2, 8 + 18 * t, n=4)
    canvas.dry()


def paint_hill():
    stage("foreground hill")
    fill(HILL, lambda x, y: lerp_paint(SNOW_WARM, SNOW_SHADOW, max(0.0, 1 - (y - hill_edge_y(x)) / 70) * 0.75),
         width=24, spacing=11, angle=-0.55, wobble=1.0, pressure=0.45)
    st(HILL_EDGE, SNOW_DEEP, 16, pressure=0.4, mixing=0.6)
    st(HILL_EDGE + [0, 10], SNOW_HALF, 10, pressure=0.35, mixing=0.6)
    for _ in range(22):     # drifts and hollows on the slope
        x = rng.uniform(600, W)
        y = rng.uniform(hill_edge_y(x) + 30, H)
        ln = rng.uniform(60, 200)
        st([(x, y), (x + ln * 0.5, y - ln * 0.18), (x + ln, y - ln * 0.4)], mix((SNOW, 3), (SNOW_SHADOW, 1)),
           rng.uniform(10, 24), pressure=0.3, mixing=0.7, taper=0.45)
    canvas.dry()
    for _ in range(120):    # dry grasses, brambles and stones poking through along the lip
        x = rng.uniform(560, W)
        y = hill_edge_y(x) + rng.uniform(-3, 22)
        h = rng.uniform(5, 22)
        if rng.random() < 0.7:
            st([(x, y), (x + rng.normal(0, 4), y - h)], lerp_paint(REED, DARK, rng.random()), [1.8, 0.5], pressure=0.2)
        else:
            dab(x, y, lerp_paint(EARTH, DARK, rng.random()), rng.uniform(3, 7))
            st([(x - 3, y - 2), (x + 3, y - 3)], SNOW, 2)
    # orange-brown brambles and dry stems pushing through the snow in the foreground
    for cx_, cy_, n in [(1420, 1030, 26), (1300, 1045, 18), (760, 1050, 14), (1480, 900, 16)]:
        for _ in range(n):
            a = -np.pi / 2 + rng.normal(0, 0.6)
            ln = rng.uniform(25, 110)
            x0 = cx_ + rng.normal(0, 25)
            pts = [(x0, cy_ + 10), (x0 + ln * 0.5 * np.cos(a), cy_ + ln * 0.5 * np.sin(a)),
                   (x0 + ln * np.cos(a + rng.normal(0, 0.3)), cy_ + ln * np.sin(a))]
            st(pts, lerp_paint(BRAMBLE, TWIG, rng.random() * 0.6), [2.0, 1.2, 0.6], pressure=0.2)
            if rng.random() < 0.5:
                st([pts[1], (pts[1][0] + rng.normal(0, 12), pts[1][1] - rng.uniform(5, 20))], BRAMBLE, [1.2, 0.6])
            if rng.random() < 0.3:
                dab(pts[1][0], pts[1][1] - 2, SNOW, 3)
    # sledge runners and footprints coming down the slope
    hill_path = [(1230, 872), (1130, 892), (1020, 900), (920, 905), (840, 915), (770, 930)]
    st(hill_path, mix((SNOW, 3), (SNOW_SHADOW, 1), (EARTH, 0.15)), [14, 16, 18, 18, 18, 18], pressure=0.25)
    track = [(1520, 620), (1440, 690), (1360, 760), (1290, 820), (1230, 862)]
    st(track, SNOW_SHADOW, 2.2, pressure=0.2)
    st([(x + 10, y + 6) for x, y in track], SNOW_SHADOW, 2.2, pressure=0.2)
    for k in range(40):
        t = k / 40
        x = 1500 - 260 * t + 12 * np.sin(k * 1.3)
        y = 640 + 200 * t + 6 * np.cos(k * 0.7)
        st([(x, y), (x + 4, y + 1)], SNOW_SHADOW, 3, pressure=0.2)
    # hare tracks: sets of four prints bounding across the foreground
    for k in range(14):
        x, y = 1460 - 26 * k, 1010 - 6 * k + 4 * np.sin(k)
        for dx, dy in ((0, 0), (3, 4), (12, 0), (12, 5)):
            dab(x + dx, y + dy, SNOW_SHADOW, 2.6)


TREES = []


def paint_trees():
    stage("trees on the hill")
    specs = [(1010, 795, 820, 13, -0.02, 41), (1215, 676, 600, 10, 0.03, 42), (1425, 556, 430, 8, -0.03, 43),
             (880, 880, 480, 6, 0.05, 44)]
    snow_lines, perches = [], []
    for x, y, h, bw, lean, sd in specs:
        sl, pc = tree(x, y, h, base_w=bw, lean=lean, seed=sd, depth=5 if h > 450 else 4, lace_thin=2.0, fan=(2, 5))
        snow_lines += sl
        perches += pc
        TREES.append((x, y, h))
    st([(1320, 638), (1323, 590)], TRUNK, [12, 9])       # a broken stump
    st([(1316, 592), (1332, 582)], TRUNK_LIGHT, 3)
    st([(1313, 592), (1332, 585)], SNOW, 3)
    for _ in range(14):
        x = rng.uniform(600, 1450)
        bush(x, hill_edge_y(x) + rng.uniform(6, 16), rng.uniform(14, 30))
    canvas.dry()
    for pts, w in snow_lines:
        st(pts, SNOW, max(0.8, w), pressure=0.2)
    order = rng.permutation(len(perches))
    for i in order[:9]:
        px, py = perches[i]
        crow_perched(px, py - 3, rng.uniform(0.9, 1.3))


def paint_ice_people():
    stage("skaters on the pond")
    cy = POND[:, 1].mean()
    ry = np.ptp(POND[:, 1]) / 2
    placed = 0
    while placed < 70:
        y = rng.uniform(cy - ry * 0.75, cy + ry * 0.85)
        segs = scan_segments(POND, y)
        if not segs:
            continue
        xa, xb = segs[int(rng.integers(len(segs)))]
        if xb - xa < 40:
            continue
        x = rng.uniform(xa + 15, xb - 15)
        h = scale_at(y) * rng.uniform(0.85, 1.1)
        coat = pick(COATS)
        r = rng.random()
        if r < 0.75:
            person(x, y, h, coat, facing=pick([-1, 1]), pose="skate", lean=rng.uniform(0.1, 0.35),
                   hat=pick(["hood", "cap", "kerchief", "round"]), skirt=rng.random() < 0.2)
        elif r < 0.85:
            person(x, y, h, coat, facing=pick([-1, 1]), pose="bend", lean=0.4, props=("kolf",), hat="cap")
        else:
            person(x, y, h, coat, facing=pick([-1, 1]), pose="stand", hat=pick(["hood", "kerchief"]),
                   skirt=rng.random() < 0.5)
        placed += 1
    for _ in range(14):
        x, y = rng.uniform(1050, 1190), rng.uniform(433, 443)
        tiny_person(x, y, scale_at(y), pick(COATS), facing=pick([-1, 1]), pose="skate")
    for p in resample(RIVER_FAR, 12)[2:]:
        tiny_person(p[0] + rng.normal(0, 3), p[1] + 1, scale_at(p[1]), pick(COATS), pose="skate")

    stage("the near river: skaters, a sleigh, games")
    cast = []
    edge_r = resample(RIVER_NEAR_R, 300)
    edge_l = resample(RIVER_NEAR_L, 300)
    tries = 0
    while len(cast) < 26 and tries < 8000:
        tries += 1
        x, y = rng.uniform(-40, 470), rng.uniform(650, 1050)
        if not all(inside(RIVER_NEAR, x + dx, y) for dx in (-25, 0, 25)):
            continue
        if np.min(np.hypot(edge_r[:, 0] - x, edge_r[:, 1] - y)) < 50:
            continue
        if np.min(np.hypot(edge_l[:, 0] - x, edge_l[:, 1] - y)) < 30:
            continue
        if np.hypot(x - 140, (y - 962) * 2) < 90 or np.hypot(x - 300, (y - 712) * 2) < 80:
            continue
        h = scale_at(y)
        if any(np.hypot(x - cx, (y - cy) * 2.2) < 0.6 * max(h, scale_at(cy)) for cx, cy in cast):
            continue
        cast.append((x, y))
    cast.sort(key=lambda p: p[1])
    special = {3: "fallen", 7: "chair", 11: "kolf", 15: "couple", 19: "yoke", 22: "top", 25: "kolf"}
    for k, (x, y) in enumerate(cast):
        h = scale_at(y) * rng.uniform(0.9, 1.05)
        coat = pick(COATS)
        f = pick([-1, 1])
        kind = special.get(k, "skate")
        if kind == "fallen":
            fallen_skater(x, y, h, coat)
        elif kind == "chair":
            chair_sled(x, y, h, facing=f)
        elif kind == "kolf":
            person(x, y, h, coat, facing=f, pose="bend", lean=0.45, props=("kolf",), hat="cap", hat_col=C_RED)
        elif kind == "couple":
            u = h / 100
            person(x, y, h, C_RED, facing=1, pose="skate", lean=0.2, hat="kerchief", skirt=True,
                   arms=[(x + 20 * u, y - 50 * u), (x - 4 * u, y - 46 * u)])
            person(x + 26 * u, y + 2, h * 1.03, C_BLACK, facing=1, pose="skate", lean=0.22, hat="cap")
        elif kind == "yoke":
            person(x, y, h, C_BLUE, facing=f, pose="walk", hat="kerchief", skirt=True, apron=True, props=("yoke",))
        elif kind == "top":
            ch = h * 0.6
            u = ch / 100
            person(x, y, ch, pick([C_RED, C_TAN]), facing=f, pose="bend", lean=0.3, hat="cap",
                   arms=[(x + f * 22 * u, y - 60 * u), (x + f * 10 * u, y - 40 * u)])
            st([(x + f * 22 * u, y - 60 * u), (x + f * 40 * u, y - 4)], DARK, 1)
            dab(x + f * 38 * u, y - 3, WOOD, max(2, 6 * u))
        else:
            person(x, y, h, coat, facing=f, pose="skate", lean=rng.uniform(0.1, 0.4),
                   hat=pick(["hood", "cap", "kerchief", "hood", "round"]), skirt=rng.random() < 0.2)
    horse_sleigh(305, 712, 40, facing=-1)
    # women drawing water at the hole in the ice, a dog watching
    person(112, 960, scale_at(960), C_BLUE, facing=1, pose="bend", lean=0.5, hat="kerchief", skirt=True, apron=True)
    person(182, 954, scale_at(954), C_DULL_RED, facing=-1, pose="stand", hat="kerchief", skirt=True, props=("yoke",))
    dog(212, 972, scale_at(972) * 0.55, facing=-1)


def paint_land_people():
    stage("woodgatherers on the hill")
    gx, gy = 1100, 942
    s = 118                                             # the hill is close to us: big figures
    sledge(gx + 26, gy - 4, 1.5)
    st([(gx + 22, gy - 16), (gx - 4, gy - 70)], DARK, 2)                # the hauling rope
    person(gx, gy, s, C_BROWN, facing=-1, pose="walk", lean=0.35, hat="hood", hat_col=C_GREY,
           arms=[(gx - 8, gy - 72), (gx + 4, gy - 70)], legs=C_GREY)
    person(gx + 230, gy - 92, s * 0.9, C_BLUE, facing=-1, pose="walk", lean=0.2, hat="kerchief", skirt=True,
           props=("bundle",))
    person(gx + 320, gy - 140, s * 0.82, C_DULL_RED, facing=-1, pose="walk", lean=0.15, hat="round",
           hat_col=C_BLACK, props=("axe",), legs=C_BLACK)
    person(gx + 380, gy - 170, s * 0.5, C_TAN, facing=-1, pose="walk", hat="cap", hat_col=C_RED)
    dog(gx - 120, gy + 22, s * 0.62, facing=-1)
    dog(gx + 140, gy - 34, s * 0.56, facing=-1, paint=C_WHITE)
    dog(gx + 270, gy - 112, s * 0.5, facing=-1, paint=DOG_BROWN)
    for k in range(22):                                 # their footprints behind them
        t = k / 22
        fx_ = gx + 40 + 360 * t + rng.normal(0, 6)
        fy_ = gy + 4 - 175 * t + rng.normal(0, 3)
        dab(fx_, fy_, SNOW_SHADOW, 4)

    stage("people and animals in the fields")
    well_sweep(250, 690, 0.9)
    haystack(820, 720, 1.2, far=0.1)
    for x, y in [(770, 700), (790, 712), (930, 740), (950, 735), (700, 760)]:
        crow_perched(x, y, 0.8)          # crows pecking at the snow
    # a woman with a basket on her head, a man with a ladder, children with a sled
    x, y = 880, 690
    h = scale_at(y)
    person(x, y, h, C_GREEN, facing=1, pose="walk", hat="kerchief", skirt=True, apron=True,
           arms=[(x + 6 * h / 100, y - 112 * h / 100), (x - 4 * h / 100, y - 110 * h / 100)])
    st([(x - 12 * h / 100, y - 112 * h / 100), (x + 16 * h / 100, y - 114 * h / 100)], C_TAN, 10 * h / 100)
    x, y = 960, 712
    h = scale_at(y)
    person(x, y, h, C_GREY, facing=-1, pose="carry", hat="round", hat_col=C_BLACK)
    st([(x - 40 * h / 100, y - 70 * h / 100), (x + 60 * h / 100, y - 92 * h / 100)], WOOD, 3)
    st([(x - 40 * h / 100, y - 76 * h / 100), (x + 60 * h / 100, y - 98 * h / 100)], WOOD, 3)
    for k in range(6):
        t = (k + 0.5) / 6
        a = (x - 40 * h / 100 + 100 * h / 100 * t, y - 70 * h / 100 - 22 * h / 100 * t)
        st([a, (a[0], a[1] - 6 * h / 100)], WOOD, 1.8)
    x, y = 720, 760
    h = scale_at(y) * 0.6
    st([(x + 18, y), (x + 50, y - 3)], DARK, 3)
    st([(x + 22, y - 6), (x + 46, y - 9)], WOOD, 5)
    dab(x + 36, y - 16, C_RED, 8)
    person(x, y, h, C_TAN, facing=-1, pose="walk", lean=0.3, hat="cap", hat_col=C_RED)
    person(x - 22, y + 4, h * 0.95, C_BLUE, facing=-1, pose="walk", lean=0.3, hat="kerchief", skirt=True)
    st([(x + 4, y - 30), (x + 20, y - 3)], DARK, 1)

    stage("the inn yard")
    fire(690, 925, 1.4)
    person(650, 932, scale_at(932), C_GREY, facing=1, pose="stand", hat="hood", hat_col=C_BLACK)
    person(735, 938, scale_at(938), C_RED, facing=-1, pose="bend", lean=0.3, hat="cap", hat_col=C_BLACK,
           props=("stick",))
    person(700, 962, scale_at(962) * 0.6, C_TAN, facing=1, pose="stand", hat="cap")
    person(610, 905, scale_at(905), C_BLUE, facing=1, pose="walk", hat="kerchief", skirt=True, apron=True,
           props=("basket",))
    person(545, 898, scale_at(898), C_BROWN, facing=1, pose="carry", hat="hood", props=("sack",))
    person(632, 735, 38, C_BLACK, facing=-1, pose="bend", lean=0.3, hat="cap", hat_col=C_RED, shadow=False,
           props=("shovel",))                     # clearing snow off the inn roof
    for x, y in [(860, 598), (900, 602), (1150, 560), (420, 545), (640, 512), (1000, 548), (180, 470), (240, 460)]:
        person(x, y, scale_at(y), pick(COATS), facing=pick([-1, 1]), pose="walk",
               hat=pick(["hood", "kerchief", "cap"]), skirt=rng.random() < 0.3,
               props=("bundle",) if rng.random() < 0.3 else ())
    for x, y in [(80, 602), (110, 607), (190, 640), (205, 645)]:
        person(x, y, scale_at(y), pick(COATS), facing=pick([-1, 1]), pose="stand",
               hat=pick(["hood", "kerchief"]), skirt=rng.random() < 0.5)
    cart(330, 1012, 0.95, facing=1)
    for x, y in [(430, 1030), (452, 1040), (470, 1022), (400, 1046)]:
        chicken(x, y, 1.1)
    person(270, 1030, scale_at(1030), C_BROWN, facing=1, pose="carry", hat="round", hat_col=C_BLACK, props=("sack",))
    for x, y, f in [(560, 978, 1), (600, 992, -1)]:      # children throwing snowballs
        h = scale_at(y) * 0.62
        u = h / 100
        person(x, y, h, pick([C_RED, C_TAN, C_BLUE]), facing=f, pose="walk", hat="cap",
               arms=[(x + f * 16 * u, y - 100 * u), (x - f * 8 * u, y - 40 * u)])
        dab(x + f * 18 * u, y - 104 * u, WHITE, 4)


def paint_birds():
    stage("birds in the air")
    bird_flying(610, 230, 0.75, magpie=True, phase=0.5)
    for _ in range(9):
        bird_flying(rng.uniform(700, 1150), rng.uniform(70, 260), rng.uniform(0.35, 0.6))
    for _ in range(18):       # a far flock
        bird_flying(280 + rng.normal(0, 60), 120 + rng.normal(0, 25), rng.uniform(0.12, 0.2),
                    paint=mix((NEAR_BLACK, 1), (SKY_MID, 1)))


STAGES = ["paint_sky", "paint_mountains", "paint_terrain", "paint_ice", "paint_far_village", "paint_near_village",
          "paint_hill", "paint_trees", "paint_ice_people", "paint_land_people", "paint_birds"]


def main(out):
    t0 = time.time()
    for name in STAGES:
        globals()[name]()
    stage("varnish")
    canvas.save(out, relief=0.55, gloss=0.5)
    print(f"wrote {out}: {N_STROKES[0]} strokes in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(__file__), "output", "bruegel_winter.png"))
