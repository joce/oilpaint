"""A reverse glass painting in the manner of Wassily Kandinsky's Murnau years.

An original composition after the glass pictures Kandinsky painted in
1909-1911, when he and Gabriele Münter took up the Bavarian folk craft of
painting behind glass: a hill crowned with onion-domed towers, a rider
galloping up it, a sun with a dark cloud, a winding road, a river with waves
and a little rowing boat, all drawn in swinging black lines, with patches of
colour floating free of them.

It is painted on the back of the glass and seen through the front, so it goes
on in reverse: the black lines first, then the small details, then the
colours, and the sky last of all, each layer left to dry before the next.
Every mark is a `canvas.stroke(...)`; numpy only works out where they go.

    python examples/kandinsky_glass.py [output.png]
"""

import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from oilpaint import Canvas, Paint, brushes, surfaces
from oilpaint import palette as P

W, H = 1000, 760
rng = np.random.default_rng(1911)
YY, XX = np.mgrid[0:H, 0:W]

# ===========================================================================
# Paints: ink for the lines, bright tube colours, all a little thinned so
# they flow out on the glass
# ===========================================================================
FLUID = 0.35
INK = P.IVORY_BLACK.with_body(0.25)
WHITE = P.TITANIUM_WHITE.with_body(FLUID)
RED, VERMILION = P.CADMIUM_RED.with_body(FLUID), P.VERMILION.with_body(FLUID)
ORANGE, YELLOW = P.CADMIUM_ORANGE.with_body(FLUID), P.CADMIUM_YELLOW.with_body(FLUID)
LEMON, OCHRE = P.LEMON_YELLOW.with_body(FLUID), P.YELLOW_OCHRE.with_body(FLUID)
BLUE, COBALT = P.ULTRAMARINE_BLUE.with_body(FLUID), P.COBALT_BLUE.with_body(FLUID)
VIRIDIAN, CRIMSON = P.VIRIDIAN.with_body(FLUID), P.ALIZARIN_CRIMSON.with_body(FLUID)
SIENNA, VIOLET = P.BURNT_SIENNA.with_body(FLUID), P.DIOXAZINE_PURPLE.with_body(FLUID)


def mix(*parts):
    return Paint.mix(*parts)


PINK = mix((WHITE, 3), (CRIMSON, 1))
ROSE = mix((WHITE, 2), (VERMILION, 1), (CRIMSON, 0.4))
LILAC = mix((WHITE, 4), (VIOLET, 0.6), (BLUE, 0.3))
PALE_BLUE = mix((WHITE, 3), (COBALT, 1))
SKY_BLUE = mix((WHITE, 2), (BLUE, 1))
TURQUOISE = mix((WHITE, 1.5), (VIRIDIAN, 1), (COBALT, 0.5))
GREEN = mix((VIRIDIAN, 1), (LEMON, 1.2))
DEEP_GREEN = mix((VIRIDIAN, 2), (BLUE, 0.5))
GOLD = mix((YELLOW, 2), (OCHRE, 1))
CREAM = mix((WHITE, 4), (YELLOW, 1))
SNOW = mix((WHITE, 6), (BLUE, 0.15), (VIOLET, 0.1))
STORM = mix((BLUE, 2), (INK, 0.5), (VIOLET, 0.6))


# ===========================================================================
# Drawing helpers
# ===========================================================================
def run_length(pts):
    pts = np.asarray(pts, float)
    return float(np.hypot(*np.diff(pts, axis=0).T).sum())


def by_hand(pts, wobble=1.6, step=14.0):
    """The path resampled every ``step`` px and drawn by hand: a slow sway
    across it."""
    pts = np.asarray(pts, float)
    seg = np.hypot(*np.diff(pts, axis=0).T)
    s = np.r_[0, np.cumsum(seg)]
    if s[-1] < 2 * step:
        return pts
    t = np.linspace(0, s[-1], int(s[-1] / step) + 2)
    x, y = np.interp(t, s, pts[:, 0]), np.interp(t, s, pts[:, 1])
    dx, dy = np.gradient(x), np.gradient(y)
    norm = np.hypot(dx, dy) + 1e-9
    sway = wobble * np.sin(t / rng.uniform(25, 45) + rng.uniform(0, 6))
    sway += rng.normal(0, 0.35 * wobble, t.size)
    return np.stack([x - sway * dy / norm, y + sway * dx / norm], 1)


def ink(c, pts, w, swell=0.6, steady=False):
    """A black brush line, swelling and thinning like Kandinsky's, with enough
    ink on the brush to reach its end."""
    pts = [tuple(p) for p in by_hand(pts, wobble=0.18 * w)]
    n = len(pts)
    u = np.linspace(0.0, 1.0, n)
    if steady:  # a closed shape: no tapered ends
        widths = w * (0.85 + 0.3 * rng.random(n))
    else:
        widths = w * (1 - swell + swell * np.sin(np.pi * u) ** 0.6)
        widths *= 0.75 + 0.5 * rng.random(n)
    c.stroke(
        pts,
        INK,
        amount=1.6 * run_length(pts) / w + 3,
        brush=brushes.ROUND_SABLE.sized(w),
        width=list(np.maximum(widths, 1.5)),
        pressure=0.7,
    )


def dab(c, x, y, paint, size, amount=3.0):
    a = rng.uniform(0, 2 * np.pi)
    d = 0.25 * size
    c.stroke(
        [
            (x - d * np.cos(a), y - d * np.sin(a)),
            (x + d * np.cos(a), y + d * np.sin(a)),
        ],
        paint,
        amount=amount,
        brush=brushes.ROUND.sized(size),
        pressure=0.6,
    )


def ellipse(cx, cy, rx, ry, a0=0.0, a1=2 * np.pi, n=24, wobble=0.0):
    t = np.linspace(a0, a1, n)
    k = 1 + wobble * np.sin(3 * t + rng.uniform(0, 6)) if wobble else 1
    return np.stack([cx + rx * k * np.cos(t), cy + ry * k * np.sin(t)], 1)


def closed(pts):
    pts = np.asarray(pts, float)
    return np.vstack([pts, pts[:2]])  # back past the start: the join is smooth


def scallops(cx, cy, rx, ry, bumps, n=8):
    """A cloud: a loop of round bumps."""
    pts = []
    for k in range(bumps):
        a0, a1 = 2 * np.pi * k / bumps, 2 * np.pi * (k + 1) / bumps
        for t in np.linspace(0, 1, n, endpoint=False):
            a = a0 + (a1 - a0) * t
            bulge = 1 + 0.18 * np.sin(np.pi * t)
            pts.append((cx + rx * bulge * np.cos(a), cy + ry * bulge * np.sin(a)))
    return np.asarray(pts)


def inside(poly):
    """Pixels inside a polygon (even-odd rule)."""
    poly = np.asarray(poly, float)
    x0, y0 = poly[:, 0], poly[:, 1]
    x1, y1 = np.roll(x0, -1), np.roll(y0, -1)
    m = np.zeros((H, W), bool)
    for a, b, cc, d in zip(x0, y0, x1, y1, strict=True):
        if b == d:
            continue
        lo, hi = int(max(min(b, d), 0)), int(min(max(b, d) + 1, H))
        if lo >= hi:
            continue
        ys, xs = YY[lo:hi], XX[lo:hi]
        hit = ((b > ys) != (d > ys)) & (xs < (cc - a) * (ys - b) / (d - b) + a)
        m[lo:hi] ^= hit
    return m


def erode(m, r):
    """The mask shrunk by ``r`` px."""
    out = m.copy()
    for _ in range(int(r)):
        inner = out.copy()
        inner[1:] &= out[:-1]
        inner[:-1] &= out[1:]
        inner[:, 1:] &= out[:, :-1]
        inner[:, :-1] &= out[:, 1:]
        out = inner
    return out


def mottle(c, region, paints, size, density=1.0, angle=None, amount=3.5, pressure=0.5,
           length=(0.5, 1.4), tool=None, margin=0.45):  # fmt: skip
    """Colour a region with short, overlapping dabs in random directions, from
    a few paints, kept just inside its edges: uneven, as paint goes on glass."""
    is_mask = isinstance(region, np.ndarray) and region.dtype == bool
    m = region if is_mask else inside(region)
    core = erode(m, margin * size)
    if core.sum() < 10:
        core = m
    ys, xs = np.nonzero(core)
    if xs.size == 0:
        return
    n = max(1, int(density * xs.size / (0.45 * size * size)))
    brush = (tool or brushes.FILBERT_HOG).sized(size)
    paints = paints if isinstance(paints, (list, tuple)) else [paints]
    for k in rng.integers(0, xs.size, n):
        x, y = xs[k] + rng.uniform(-0.5, 0.5), ys[k] + rng.uniform(-0.5, 0.5)
        a = rng.uniform(0, np.pi) if angle is None else angle + rng.normal(0, 0.35)
        half = 0.5 * size * rng.uniform(*length)
        c.stroke(
            [
                (x - half * np.cos(a), y - half * np.sin(a)),
                (x + half * np.cos(a), y + half * np.sin(a)),
            ],
            paints[int(rng.integers(len(paints)))],
            amount=amount,
            brush=brush,
            pressure=pressure * rng.uniform(0.8, 1.2),
        )


def fill(c, region, paint, size, angle=0.0, amount=1.0, pressure=0.5, tool=None):
    """Brush a region (a mask or a polygon) with parallel strokes, a little past
    its edges: on the back of the glass the lines in front hide them."""
    is_mask = isinstance(region, np.ndarray) and region.dtype == bool
    full = region if is_mask else inside(region)
    m = erode(full, 0.4 * size)
    if m.sum() < 10:  # a narrow shape: brush it all
        m = full
    if not m.any():
        return
    tool = tool or brushes.FILBERT_HOG
    brush = tool.sized(size)
    ca, sa = np.cos(angle), np.sin(angle)
    ys, xs = np.nonzero(m)
    along, across = xs * ca + ys * sa, -xs * sa + ys * ca
    t = np.arange(along.min() - 2, along.max() + 3, 2.0)
    step = 0.6 * size
    for d in np.arange(across.min() + 0.3 * step, across.max() + step, step):
        d_ = d + rng.uniform(-0.15, 0.15) * step
        px, py = t * ca - d_ * sa, t * sa + d_ * ca
        ok = (px >= 0) & (px < W) & (py >= 0) & (py < H)
        hit = np.zeros_like(ok)
        hit[ok] = m[py[ok].astype(int), px[ok].astype(int)]
        edges = np.flatnonzero(np.diff(np.r_[0, hit.astype(int), 0]))
        for s, e in zip(edges[::2], edges[1::2], strict=True):
            if e - s < 2:
                continue
            ext = 0.05 * size
            t0, t1 = t[s] - ext, t[e - 1] + ext
            n = max(2, int((t1 - t0) / 60) + 2)
            tt = np.linspace(t0, t1, n)
            wob = rng.normal(0, 0.06 * size, n)
            pts = list(
                zip(tt * ca - (d_ + wob) * sa, tt * sa + (d_ + wob) * ca, strict=True)
            )
            c.stroke(
                pts,
                paint,
                amount=amount * (1.2 * (t1 - t0) / size + 3),
                brush=brush,
                pressure=pressure,
            )


# ===========================================================================
# The composition
# ===========================================================================
HILL = np.array(
    [(-30, 470), (60, 432), (180, 388), (300, 348), (420, 320), (520, 310),
     (620, 314), (720, 336), (830, 382), (930, 440), (1030, 500)],
    float,
)  # fmt: skip


def hill_y(x):
    return float(np.interp(x, HILL[:, 0], HILL[:, 1]))


# blue mountains on the horizon, behind the rider, down to the hill
RIDGE = [
    (-30, 262),
    (40, 214),
    (110, 246),
    (190, 192),
    (270, 238),
    (340, 222),
    (420, 280),
    (476, 304),
]
MOUNTAINS = RIDGE + [(x, y + 6) for x, y in HILL[HILL[:, 0] < 476][::-1]]
ROAD_L = [(250, 790), (300, 680), (385, 585), (470, 480), (528, 392), (556, 318)]
ROAD_R = [(345, 790), (380, 690), (448, 600), (525, 500), (578, 405), (592, 320)]
RIVER = [(1030, 548), (935, 566), (862, 604), (806, 660), (770, 720), (752, 790)]
SUN = (848, 150, 66)

# buildings on the hilltop: (left, width, top, roof, wall, roof colour)
TOWN = [
    (448, 44, 266, "gable", PALE_BLUE, RED),
    (494, 34, 176, "spire", ORANGE, CRIMSON),
    (536, 30, 196, "onion", ROSE, GOLD),
    (572, 54, 252, "gable", COBALT, VERMILION),
    (630, 28, 206, "onion", ORANGE, TURQUOISE),
    (664, 40, 262, "gable", PINK, BLUE),
    (708, 24, 250, "onion", GOLD, GREEN),
]


def roof_points(x, w, top, kind):
    if kind == "spire":
        return np.array([(x - 3, top), (x + w / 2, top - 2.4 * w), (x + w + 3, top)])
    if kind == "gable":
        return np.array([(x - 6, top), (x + w / 2, top - 0.55 * w), (x + w + 6, top)])
    cx = x + w / 2  # an onion dome
    return np.array(
        [(x, top), (x - 0.18 * w, top - 0.45 * w), (x + 0.05 * w, top - 0.95 * w),
         (cx, top - 1.4 * w), (x + 0.95 * w, top - 0.95 * w),
         (x + 1.18 * w, top - 0.45 * w), (x + w, top)]
    )  # fmt: skip


# the rider, galloping up the slope (drawn about an origin, then placed)
def _place(pts, ox=228, oy=300, k=1.15):
    return [(ox + k * x, oy + k * y) for x, y in pts]


_BACK = [(-70, -10), (-40, -20), (-5, -22), (30, -28)]
_NECK = [(30, -28), (48, -50), (62, -68), (72, -78)]
_HEAD = [(72, -78), (88, -70), (104, -54), (112, -44), (104, -38), (88, -44), (74, -50)]
_THROAT = [(74, -50), (60, -30), (50, -6)]
_BELLY = [(50, -6), (35, 6), (0, 10), (-35, 8), (-62, 2)]
_RUMP = [(-62, 2), (-80, 12), (-84, -2), (-70, -10)]
HORSE_BODY = _place(
    _BACK + _NECK[1:] + _HEAD[1:] + _THROAT[1:] + _BELLY[1:] + _RUMP[1:-1]
)
HORSE_LINES = [
    _place(_BACK + _NECK[1:]),
    _place(_HEAD),
    _place(_THROAT + _BELLY[1:]),
    _place(_RUMP),
    _place([(46, 0), (68, 14), (92, 12), (100, 20)]),  # front legs, reaching out
    _place([(36, 6), (52, 28), (76, 38), (84, 46)]),
    _place([(-70, 6), (-96, 18), (-118, 14), (-126, 20)]),  # hind legs, kicking back
    _place([(-60, 8), (-78, 32), (-100, 42), (-104, 50)]),
    _place([(-78, -8), (-100, -22), (-124, -16), (-146, -26), (-160, -18)]),  # tail
]
MANE = [_place([(x, y), (x - 14, y - 2)]) for x, y in _NECK[:-1] + [(68, -74)]]
EYE = _place([(90, -62)])[0]
RIDER_BODY = _place([(-14, -24), (-4, -66), (10, -70), (14, -58), (8, -24)])
RIDER_LEG = _place([(-6, -26), (4, -4), (-4, 14)])
ARM = _place([(10, -60), (34, -58), (46, -66)])
LANCE = _place([(-40, -30), (60, -92), (170, -160)])
CAPE = _place([(0, -64), (-30, -76), (-64, -64), (-52, -48), (-18, -48)])
HEAD = (*_place([(6, -82)])[0], 12)

TREE = (104, 486, 48)  # crown centre and radius; the trunk goes down to the hill
CLOUDS = [(170, 118, 108, 42, 7), (420, 70, 80, 28, 5)]
STORM_CLOUD = ellipse(650, 104, 112, 44, n=26, wobble=0.12)
MOUND = [
    (-30, 612),
    (60, 584),
    (150, 592),
    (218, 640),
    (252, 712),
    (258, 790),
    (-30, 790),
]
BOAT = (900, 690)


def wave(x, y, s, flip=1):
    """A breaking wave: a long swell that rolls over into a curl."""
    a = np.linspace(0, 2.4 * np.pi, 18)
    r = s * (1 - 0.7 * a / a.max())
    run = 1.6 * s * a / a.max()
    return np.stack(
        [x + flip * (run + r * np.sin(a)), y - r * (1 - np.cos(a)) * 0.7], 1
    )


def below(curve, dy, x0, x1):
    """A line running along ``curve``, ``dy`` lower, from x0 to x1."""
    xs = np.linspace(x0, x1, 9)
    return [
        (x, np.interp(x, curve[:, 0], curve[:, 1]) + dy * (1 + 0.15 * np.sin(x / 60)))
        for x in xs
    ]


# ===========================================================================
# The stages, in the order they go on the back of the glass
# ===========================================================================
def lines(c):
    """1. The black lines, all of them, first: they will be in front."""
    ink(c, HILL, 12)
    ink(c, RIDGE, 9)
    ink(c, below(HILL, 150, 640, 990), 7)
    ink(c, MOUND[:6], 10)
    ink(c, ROAD_L, 9)
    ink(c, ROAD_R, 9)
    ink(c, RIVER, 10)
    cx, cy, r = SUN
    ink(c, closed(ellipse(cx, cy, r, r, n=28)), 10, steady=True)
    ink(c, closed(ellipse(cx, cy, 0.6 * r, 0.6 * r, n=20)), 6, steady=True)
    for x, w, top, kind, _wall, _roof in TOWN:
        base = hill_y(x + w / 2) + 6
        ink(c, [(x, base), (x, top)], 6, swell=0.3)
        ink(c, [(x + w, top), (x + w, base)], 6, swell=0.3)
        roof = roof_points(x, w, top, kind)
        ink(c, roof, 6, swell=0.3)
        ink(c, [(x, top), (x + w, top)], 4, swell=0.2)
        if kind != "gable":
            tip = roof[len(roof) // 2]
            ink(c, [tip, (tip[0] + 1, tip[1] - 22)], 4, swell=0.2)
    for pts in HORSE_LINES:
        ink(c, pts, 7)
    for pts in MANE:
        ink(c, pts, 4)
    ink(c, closed(RIDER_BODY), 6, steady=True)
    ink(c, RIDER_LEG, 6)
    ink(c, ARM, 5)
    ink(c, closed(CAPE), 6, steady=True)
    ink(c, LANCE, 5, swell=0.3)
    hx, hy, hr = HEAD
    ink(c, closed(ellipse(hx, hy, hr, hr, n=12)), 5, steady=True)
    tx, ty, tr = TREE
    ink(c, closed(scallops(tx, ty, tr, tr * 1.05, 6)), 8, steady=True)
    ink(c, [(tx - 4, ty + tr), (tx - 2, ty + tr + 40), (tx - 6, hill_y(tx) + 26)], 10)
    for cx_, cy_, rx, ry, bumps in CLOUDS:
        ink(c, closed(scallops(cx_, cy_, rx, ry, bumps)), 8, steady=True)
    ink(c, closed(STORM_CLOUD), 9, steady=True)
    # waves on the river, and the boat
    for x, y, s_, flip in ((790, 748, 40, 1), (930, 628, 34, 1), (960, 760, 30, -1)):
        ink(c, wave(x, y, s_, flip), 8)
    bx, by = BOAT
    ink(
        c,
        [
            (bx - 70, by - 10),
            (bx - 24, by + 18),
            (bx + 36, by + 15),
            (bx + 74, by - 15),
        ],
        8,
    )
    ink(c, [(bx - 70, by - 10), (bx + 74, by - 15)], 5, swell=0.2)
    for k in range(3):
        x = bx - 36 + 36 * k
        ink(c, [(x, by - 12), (x + 3, by - 44)], 7, swell=0.3)  # rowers
        dab(c, x + 3, by - 50, INK, 13, 3)
        ink(c, [(x - 4, by - 28), (x + 26, by + 30)], 5, swell=0.3)  # oars
    # furrows on the mound in front
    mound = np.asarray(MOUND[:6], float)
    for dy, x0, x1 in ((48, -10, 200), (98, -10, 228), (150, 20, 246)):
        ink(c, below(mound, dy, x0, x1), 7)


def details(c):
    """2. The small things in front of the colours: windows, eyes, glints."""
    for x, w, top, kind, _wall, _roof in TOWN:
        base = hill_y(x + w / 2)
        for y in np.arange(top + 16, base - 8, 26):
            dab(c, x + w / 2, y, YELLOW if rng.random() < 0.6 else INK, 7)
        if kind == "onion":
            dab(c, x + 0.3 * w, top - 0.7 * w, WHITE, 5, 2)
        if kind == "spire":
            tip = (x + w / 2, top - 2.4 * w)
            c.stroke(
                [tip, (tip[0] + 18, tip[1] + 6)],
                RED,
                amount=4,
                brush=brushes.ROUND.sized(6),
            )
    dab(c, *EYE, INK, 5, 2)
    dab(c, HEAD[0], HEAD[1] - 12, RED, 14, 4)  # the rider's cap
    for _ in range(50):  # foam on the water
        x, y = rng.uniform(760, 1000), rng.uniform(580, 760)
        if (
            y
            > np.interp(x, [p[0] for p in RIVER[::-1]], [p[1] for p in RIVER[::-1]])
            + 12
        ):
            dab(c, x, y, WHITE, rng.uniform(3, 6), 2)


def patches(c):
    """3. Colour floating free of the lines, as Kandinsky let it."""
    for x, y, rx, ry, paints in (
        (420, 452, 60, 36, [PINK, ROSE]),
        (660, 474, 74, 42, [YELLOW, LEMON]),
        (360, 622, 46, 32, [BLUE, COBALT]),
        (566, 690, 62, 34, [ROSE, PINK]),
        (940, 336, 44, 28, [PINK]),
        (84, 360, 52, 28, [YELLOW, GOLD]),
        (318, 118, 42, 24, [PINK, ROSE]),
        (180, 610, 40, 30, [VERMILION]),
    ):
        mottle(c, ellipse(x, y, rx, ry, n=20, wobble=0.15), paints, 20, density=1.6,
               tool=brushes.ROUND, margin=0.2)  # fmt: skip


def colours(c):
    """4. The colours of things, loosely, behind the lines: uneven dabs first,
    then, once they are dry, a solid coat behind them to close the gaps."""
    cx, cy, r = SUN
    small = [
        (inside(ellipse(cx, cy, 0.6 * r, 0.6 * r)), [YELLOW, LEMON], YELLOW, 18),
        (
            inside(ellipse(cx, cy, r, r))
            & ~inside(ellipse(cx, cy, 0.62 * r, 0.62 * r)),
            [VERMILION, ORANGE, RED],
            VERMILION,
            16,
        ),
        (inside(HORSE_BODY), [CREAM, WHITE, LEMON], CREAM, 16),
        (inside(RIDER_BODY), [BLUE, COBALT], BLUE, 10),
        (inside(CAPE), [RED, VERMILION], RED, 12),
        (inside(ellipse(*HEAD, HEAD[2])), [PINK], PINK, 10),
        (inside(scallops(*TREE, TREE[2] * 1.05, 6)), [GREEN, DEEP_GREEN], GREEN, 18),
    ]
    for cx_, cy_, rx, ry, bumps in CLOUDS:
        small.append(
            (
                inside(scallops(cx_, cy_, rx, ry, bumps)),
                [WHITE, SNOW, PALE_BLUE],
                WHITE,
                22,
            )
        )
    bx, by = BOAT
    hull = [
        (bx - 72, by - 11),
        (bx - 24, by + 19),
        (bx + 36, by + 16),
        (bx + 76, by - 16),
    ]
    small.append((inside(hull), [CRIMSON, PINK], CRIMSON, 10))
    for x, w, top, kind, _wall, roof in TOWN:
        small.append((inside(roof_points(x, w, top + 2, kind)), [roof], roof, 10))
    for region, paints, _base, size in small:
        mottle(c, region, paints, size, density=1.6, margin=0.2)
    c.dry()
    for x, w, top, _kind, wall, _roof in TOWN:
        base = hill_y(x + w / 2) + 10
        wall_shape = [(x - 2, base), (x - 2, top), (x + w + 2, top), (x + w + 2, base)]
        fill(c, wall_shape, wall, 14, angle=np.pi / 2)
    for region, _paints, base, size in small:
        fill(c, region, base, 0.8 * size, angle=0.4, amount=1.2)
    c.dry()  # the small things dry before the big ones go on behind them
    # the big areas, the same way
    road = inside(list(ROAD_L) + list(ROAD_R[::-1]))
    mound = inside(MOUND)
    river = inside(list(RIVER) + [(1030, 790)])
    hill = inside(list(HILL) + [(1030, 790), (-30, 790)]) & ~road & ~mound & ~river
    for region, paints, base, size, angle in (
        (inside(STORM_CLOUD), [STORM, BLUE, VIOLET, COBALT, STORM], STORM, 26, 0.1),
        (
            inside(MOUNTAINS),
            [COBALT, SKY_BLUE, BLUE, PALE_BLUE, VIOLET],
            COBALT,
            24,
            -0.3,
        ),
        (road, [OCHRE, GOLD, ORANGE], OCHRE, 22, -1.0),
        (mound, [GREEN, DEEP_GREEN, LEMON, GREEN], GREEN, 26, -0.4),
        (river, [TURQUOISE, BLUE, DEEP_GREEN, SKY_BLUE, COBALT], BLUE, 26, -0.6),
        (hill, [SNOW, WHITE, SNOW, LILAC, SNOW, PALE_BLUE], SNOW, 42, 0.3),
    ):
        mottle(c, region, paints, size, density=1.3, angle=angle)
        c.dry()
        fill(c, region, base, size + 8, angle=angle, amount=1.2)


def sky(c):
    """5. The sky, last of all, over everything: it only shows where nothing
    else has gone on before it."""
    cx, cy, _ = SUN
    size = 44
    brush = brushes.FILBERT_HOG.sized(size)
    pts = [
        (x, y)
        for y in np.arange(-10, H + 20, 0.42 * size)
        for x in np.arange(-20, W + 20, 0.5 * size)
    ]
    for k in rng.permutation(len(pts)):
        x, y = pts[k][0] + rng.uniform(-10, 10), pts[k][1] + rng.uniform(-10, 10)
        d = np.hypot(x - cx, y - cy)
        if d < 120:
            choice = [LEMON, CREAM, YELLOW, ORANGE]
        elif d < 220:
            choice = [CREAM, WHITE, LEMON, PINK]
        elif y < 190 and x < 560:
            choice = [PALE_BLUE, SKY_BLUE, LILAC, WHITE, PALE_BLUE]
        else:
            choice = [WHITE, LILAC, PALE_BLUE, LILAC, PINK, WHITE, SNOW]
        a = rng.normal(0.15, 0.5)
        half = 0.5 * size * rng.uniform(1.0, 2.0)
        c.stroke(
            [
                (x - half * np.cos(a), y - half * np.sin(a)),
                (x + half * np.cos(a), y + half * np.sin(a)),
            ],
            choice[int(rng.integers(len(choice)))],
            amount=4,
            brush=brush,
            pressure=rng.uniform(0.4, 0.7),
        )


class Glass(Canvas):
    """A canvas that counts its strokes."""

    strokes = 0

    def stroke(self, *args, **kwargs):
        self.strokes += 1
        return super().stroke(*args, **kwargs)


def paint_it(path):
    t0 = time.time()
    # Kandinsky backed his glass pictures with black card to deepen the colours
    c = Glass(
        W, H, surface=surfaces.GLASS.with_(ground="#141414"), reverse=True, seed=1911
    )
    for stage in (lines, details, patches, colours, sky):
        print(
            f"[{time.strftime('%H:%M:%S')}] {stage.__name__}",
            f"(strokes so far: {c.strokes})",
            flush=True,
        )
        stage(c)
        c.dry()  # each layer dries before the next goes on behind it
    c.save(path)
    print(f"wrote {path}: {c.strokes} strokes in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    paint_it(
        sys.argv[1]
        if len(sys.argv) > 1
        else os.path.join(os.path.dirname(__file__), "output", "kandinsky_glass.png")
    )
