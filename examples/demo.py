"""Generate a set of demo images showing what the API does.

python examples/demo.py [output_dir]
"""

import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from oilpaint import (
    Bezier,
    Brush,
    Canvas,
    Paint,
)
from oilpaint import (
    brushes as B,
)
from oilpaint import (
    palette as P,
)
from oilpaint import spectral as sp
from oilpaint import (
    surfaces as S,
)
from oilpaint.png import write_png

OUT = (
    sys.argv[1]
    if len(sys.argv) > 1
    else os.path.join(os.path.dirname(__file__), "output")
)
os.makedirs(OUT, exist_ok=True)


def save(canvas, name, **kw):
    path = os.path.join(OUT, name)
    canvas.save(path, **kw)
    print("wrote", path)


# ---------------------------------------------------------------------------
# 1. Mixing chart: Kubelka-Munk pigment mixing vs. naive RGB averaging
# ---------------------------------------------------------------------------
def mixing_chart():
    pairs = [
        (P.ULTRAMARINE_BLUE, P.CADMIUM_YELLOW),
        (P.PHTHALO_BLUE, P.LEMON_YELLOW),
        (P.CADMIUM_RED, P.TITANIUM_WHITE),
        (P.CADMIUM_RED, P.ULTRAMARINE_BLUE),
        (P.QUINACRIDONE_MAGENTA, P.PHTHALO_GREEN),
        (P.IVORY_BLACK, P.TITANIUM_WHITE),
        (P.BURNT_SIENNA, P.ULTRAMARINE_BLUE),
        (P.CADMIUM_YELLOW, P.CADMIUM_RED),
    ]
    steps, cell, gap = 11, 44, 10
    h = len(pairs) * (2 * cell + gap) + gap
    w = steps * cell + 2 * gap
    img = np.ones((h, w, 3)) * 0.96
    for row, (a, b) in enumerate(pairs):
        y = gap + row * (2 * cell + gap)
        for i in range(steps):
            t = i / (steps - 1)
            km = (
                Paint.mix((a, 1 - t), (b, t)).color
                if 0 < t < 1
                else (a.color if t == 0 else b.color)
            )
            lin = (1 - t) * sp.srgb_to_linear(a.color) + t * sp.srgb_to_linear(b.color)
            x = gap + i * cell
            img[y : y + cell, x : x + cell] = km
            img[y + cell : y + 2 * cell - 6, x : x + cell] = sp.linear_to_srgb(lin)
    path = os.path.join(OUT, "mixing_chart.png")
    write_png(path, np.round(img * 255).astype(np.uint8))
    print("wrote", path, "(top row of each pair: paint mixing, bottom: RGB averaging)")


# ---------------------------------------------------------------------------
# 2. Stroke behaviours: amount, pressure, wet-on-wet, dirty brush
# ---------------------------------------------------------------------------
def strokes():
    c = Canvas(900, 520, seed=2)
    # Paint amount: from a starved brush to a heavy load.
    for i, amount in enumerate([1.5, 4, 8, 20]):
        y = 50 + 55 * i
        c.stroke(
            [(40, y), (240, y - 15), (430, y + 5)],
            paint=P.CADMIUM_RED,
            amount=amount,
            width=30,
            seed=i,
        )
    # Wet-on-wet: yellow pulled through the wet blue.
    c.stroke(
        [(480, 70), (650, 40), (860, 90)],
        paint=P.ULTRAMARINE_BLUE,
        amount=12,
        width=44,
        seed=10,
    )
    c.stroke(
        [(480, 150), (650, 120), (860, 170)],
        paint=P.ULTRAMARINE_BLUE,
        amount=12,
        width=44,
        seed=11,
    )
    c.stroke(
        [(560, 20), (590, 140), (560, 260)],
        paint=P.CADMIUM_YELLOW,
        amount=10,
        width=34,
        pressure=0.3,
        seed=12,
    )
    c.stroke(
        [(760, 20), (790, 140), (760, 260)],
        paint=P.CADMIUM_YELLOW,
        amount=10,
        width=34,
        pressure=0.9,
        seed=13,
    )
    # Varying width and pressure along a Bezier curve.
    c.stroke(
        Bezier(
            [
                (60, 300),
                (200, 220),
                (300, 480),
                (440, 380),
                (520, 320),
                (620, 300),
                (700, 420),
            ]
        ),
        paint=P.VIRIDIAN,
        amount=14,
        width=[12, 40, 20],
        pressure=[0.2, 0.8, 0.4],
        seed=20,
    )
    # A white stroke dragged down through the wet reds: the brush gets dirty and
    # keeps tinting the next stroke, painted without reloading or cleaning.
    brush = c.stroke(
        [(140, 15), (150, 140), (140, 255)],
        paint=P.TITANIUM_WHITE,
        amount=26,
        width=30,
        seed=30,
    )
    c.stroke([(560, 330), (700, 300), (870, 360)], brush=brush, width=30, seed=31)
    c.stroke([(560, 420), (700, 390), (870, 450)], brush=brush, width=30, seed=32)
    save(c, "strokes.png")


# ---------------------------------------------------------------------------
# 3. Blending two wet colors with a clean brush
# ---------------------------------------------------------------------------
def blending():
    c = Canvas(700, 300, seed=3)
    for x in range(60, 340, 28):
        c.stroke(
            [(x, 20), (x + 4, 150), (x, 280)],
            paint=P.ULTRAMARINE_BLUE,
            amount=10,
            width=40,
        )
    for x in range(360, 650, 28):
        c.stroke(
            [(x, 20), (x - 4, 150), (x, 280)],
            paint=P.CADMIUM_YELLOW,
            amount=10,
            width=40,
        )
    # Blend the seam with a clean brush: short strokes back and forth across it,
    # the way a painter would. No paint is loaded, the brush only moves wet paint.
    rng = np.random.default_rng(0)
    for rnd in range(3):
        for k, y in enumerate(range(25 + 4 * rnd, 285, 9)):
            x0 = 350 + rng.normal(0, 25)
            half = 70 - 15 * rnd
            xs = (
                (x0 - half, x0 + half) if (k + rnd) % 2 == 0 else (x0 + half, x0 - half)
            )
            c.stroke(
                [(xs[0], y), (xs[1], y + rng.normal(0, 3))], width=26, pressure=0.55
            )
    # A last pass of long, light strokes to smooth the transition.
    for y in range(20, 290, 14):
        c.stroke(
            [(230, y), (470, y + rng.normal(0, 2))],
            width=30,
            pressure=0.25,
            pickup=0.25,
        )
    save(c, "blending.png")


# ---------------------------------------------------------------------------
# 4. Glazing: transparent paint over a dry underpainting
# ---------------------------------------------------------------------------
def glazing():
    c = Canvas(720, 330, seed=4)
    # Underpainting: three horizontal value bands of umber + white, left to dry.
    values = [
        Paint.mix((P.BURNT_UMBER, 1), (P.TITANIUM_WHITE, w)) for w in (0.2, 1.5, 8)
    ]
    for band, paint in enumerate(values):
        for y in range(25 + 100 * band, 100 + 100 * band, 22):
            c.stroke(
                [(15, y), (360, y - 4), (705, y)], paint=paint, amount=30, width=30
            )
    c.dry()
    # Over it, vertical strokes: two transparent glazes (pigment thinned with
    # clear medium), then two opaque paints that hide the underpainting.
    columns = [
        Paint.mix((P.ALIZARIN_CRIMSON, 1), (P.MEDIUM, 2)),
        Paint.mix((P.PHTHALO_BLUE, 1), (P.MEDIUM, 5)),
        P.CADMIUM_RED,
        P.CERULEAN_BLUE,
    ]
    for i, paint in enumerate(columns):
        for j in range(4):
            x = 40 + 170 * i + 30 * j
            c.stroke(
                [(x, 10), (x + 3, 165), (x, 320)],
                paint=paint,
                amount=14,
                width=34,
                pressure=0.35,
            )
    save(c, "glazing.png")


# ---------------------------------------------------------------------------
# 5. The brush rack: every preset doing the same five marks
# ---------------------------------------------------------------------------
def brush_rack():
    rows = [
        ("ROUND", P.ULTRAMARINE_BLUE),
        ("ROUND_SABLE", P.ULTRAMARINE_BLUE),
        ("LINER", P.IVORY_BLACK),
        ("RIGGER", P.BURNT_UMBER),
        ("MOP", P.CERULEAN_BLUE),
        ("DEER_FOOT", P.SAP_GREEN),
        ("FLAT_HOG", P.CADMIUM_RED),
        ("BRIGHT", P.CADMIUM_RED),
        ("ANGLED", P.CADMIUM_ORANGE),
        ("DAGGER", P.ALIZARIN_CRIMSON),
        ("FILBERT_HOG", P.VIRIDIAN),
        ("FILBERT_SABLE", P.VIRIDIAN),
        ("FAN_HOG", P.SAP_GREEN),
        ("FAN_SOFT", P.YELLOW_OCHRE),
        ("PALETTE_KNIFE", P.TITANIUM_WHITE),
        ("BLENDER", P.COBALT_BLUE),
    ]
    c = Canvas(1400, 70 * len(rows) + 20, seed=1)
    for k, (name, paint) in enumerate(rows):
        b = getattr(B, name)
        y = 45 + 70 * k
        # small brushes need more loads to go as far
        load = 12 * max(1.0, 20.0 / b.width)
        # straight
        c.stroke([(30, y), (330, y)], paint, amount=load, brush=b, seed=k)
        # pressure S
        c.stroke(
            [(380, y + 15), (480, y - 18), (580, y + 15), (680, y - 10)],
            paint,
            amount=load,
            brush=b,
            pressure=[0.1, 0.9, 0.2, 0.6],
            seed=k + 100,
        )
        # dab
        c.stroke([(740, y)], paint, amount=min(load, 3), brush=b, seed=k + 200)
        # on edge
        c.stroke(
            [(800, y + 18), (900, y - 18)],
            paint,
            amount=load,
            brush=b,
            angle=90,
            seed=k + 300,
        )
        # twisted
        c.stroke(
            [(960, y), (1360, y)],
            paint,
            amount=load * 1.3,
            brush=b,
            angle=lambda u: 90 * u,
            seed=k + 400,
        )
    save(c, "brush_rack.png")
    print("  rows:", ", ".join(r for r, _ in rows))


# ---------------------------------------------------------------------------
# 6. What the brushes are for
# ---------------------------------------------------------------------------
def brush_uses():
    rng = np.random.default_rng(3)
    c = Canvas(1200, 560, seed=2)
    # calligraphy: a flat held at a fixed angle on the canvas -> thick and thin
    loop = [
        (60 + 50 * np.cos(a) + 18 * a, 120 + 60 * np.sin(a))
        for a in np.linspace(0, 4 * np.pi, 30)
    ]
    c.stroke(
        loop, P.IVORY_BLACK, amount=60, brush=B.FLAT_SABLE, angle=40, follow_path=False
    )
    # filbert leaves: press, then lift
    for k in range(7):
        a = -0.6 + k * 0.2
        x0, y0 = 380 + 20 * k, 210
        c.stroke(
            [(x0, y0), (x0 + 60 * np.sin(a), y0 - 110 * np.cos(a))],
            P.SAP_GREEN,
            amount=6,
            brush=B.FILBERT_SABLE,
            pressure=[0.2, 0.9, 0.1],
            seed=k,
        )
    # hog fan flicked upward: grass
    for k in range(14):
        x = 600 + k * 20 + rng.normal(0, 4)
        c.stroke(
            [(x, 220), (x + rng.normal(0, 12), 110)],
            Paint.mix((P.SAP_GREEN, 1), (P.YELLOW_OCHRE, k % 3 * 0.3)),
            amount=3,
            brush=B.FAN_HOG.sized(30),
            pressure=[0.6, 0.1],
            seed=k,
        )
    # deer foot stippling with a dry brush: foliage
    for k in range(70):
        x, y = 1000 + rng.normal(0, 55), 140 + rng.normal(0, 40)
        c.stroke(
            [(x, y)],
            Paint.mix((P.VIRIDIAN, 1), (P.YELLOW_OCHRE, rng.uniform(0, 1))),
            amount=0.4,
            brush=B.DEER_FOOT,
            pressure=rng.uniform(0.4, 0.8),
            seed=k,
        )
    # mop blending two wet bands: zigzag across the seam, then long light strokes
    c.stroke(
        [(40, 330), (520, 330)], P.CADMIUM_YELLOW, amount=30, brush=B.FLAT_HOG.sized(60)
    )
    c.stroke(
        [(40, 390), (520, 390)], P.CADMIUM_RED, amount=30, brush=B.FLAT_HOG.sized(60)
    )
    mop = B.MOP.sized(40)
    zig = [(50 + 9 * j, 335 if j % 2 == 0 else 385) for j in range(52)]
    c.stroke(zig, None, brush=mop, pressure=0.35)
    c.stroke([(x + 4, y) for x, y in zig[::-1]], None, brush=mop, pressure=0.3)
    for y in (345, 360, 375):
        c.stroke([(40, y), (520, y)], None, brush=mop, pressure=0.25)
    # palette knife: slabs of thick paint
    for k, col in enumerate([P.CERULEAN_BLUE, P.TITANIUM_WHITE, P.CADMIUM_ORANGE]):
        c.stroke(
            [(600 + 30 * k, 300 + 30 * k), (760 + 30 * k, 290 + 30 * k)],
            col,
            amount=25,
            brush=B.PALETTE_KNIFE,
            pressure=0.6,
            angle=15,
            seed=k,
        )
    # rigger: long thin branches
    for k in range(6):
        c.stroke(
            [(900, 520), (920 + 30 * k, 420 - 10 * k), (900 + 60 * k, 300 + 8 * k)],
            P.BURNT_UMBER,
            amount=200,
            brush=B.RIGGER,
            pressure=[0.6, 0.4, 0.1],
            seed=k,
        )
    # angled flat cutting a crisp corner; dagger strokes tapering at both ends
    c.stroke(
        [(40, 470), (240, 470), (240, 540)],
        P.ULTRAMARINE_BLUE,
        amount=20,
        brush=B.ANGLED,
    )
    for k in range(4):
        c.stroke(
            [(320 + 60 * k, 540), (360 + 60 * k, 450)],
            P.ALIZARIN_CRIMSON,
            amount=8,
            brush=B.DAGGER,
            pressure=[0.2, 0.8, 0.1],
            seed=k,
        )
    save(c, "brush_uses.png")


# ---------------------------------------------------------------------------
# 7. Surfaces: the same marks on every support
# ---------------------------------------------------------------------------
def surface_sampler():
    names = [
        "CANVAS",
        "LINEN_FINE",
        "LINEN_COARSE",
        "COTTON_DUCK",
        "CANVAS_BOARD",
        "OIL_PAPER",
        "GESSO_PANEL",
        "COPPER",
        "GLASS",
        "toned",
    ]
    dark = Paint.mix((P.BURNT_UMBER, 1), (P.IVORY_BLACK, 0.5))
    tiles = []
    for k, name in enumerate(names):
        surf = (
            S.LINEN_FINE.with_(toning="#c49a6c", name="Fine linen, toned")
            if name == "toned"
            else getattr(S, name)
        )
        c = Canvas(340, 340, surface=surf, seed=k)
        # a loaded hog flat at normal pressure
        c.stroke(
            [(20, 50), (320, 44)], P.CADMIUM_RED, amount=14, brush=B.FLAT_HOG, seed=1
        )
        # the same brush, lightly dragged: tooth breaks it up (scumbling)
        c.stroke(
            [(20, 100), (320, 94)],
            P.ULTRAMARINE_BLUE,
            amount=8,
            brush=B.FLAT_HOG.sized(30),
            pressure=0.2,
            seed=2,
        )
        # a dark: sinks in (goes matte and lighter) on absorbent grounds
        c.stroke(
            [(20, 150), (320, 146)],
            dark,
            amount=30,
            brush=B.FLAT_SABLE.sized(34),
            seed=3,
        )
        # a wet patch wiped back with a rag: clean on slick surfaces, stained on
        # absorbent ones
        for y in (205, 230, 255):
            c.stroke(
                [(20, y), (320, y)],
                P.YELLOW_OCHRE,
                amount=14,
                brush=B.FLAT_HOG.sized(28),
                seed=4 + y,
            )
        c.stroke([(60, 190), (110, 270)], brush=B.RAG.sized(30), pressure=0.6)
        c.stroke([(200, 190), (250, 270)], brush=B.RAG.sized(30), pressure=0.6)
        c.stroke([(210, 270), (260, 190)], brush=B.RAG.sized(30), pressure=0.6)
        # a palette-knife slab of white
        c.stroke(
            [(40, 305), (300, 300)],
            P.TITANIUM_WHITE,
            amount=25,
            brush=B.PALETTE_KNIFE,
            seed=5,
        )
        tiles.append(np.pad(c.render(), ((6, 6), (6, 6), (0, 0)), constant_values=1.0))
        print("  ", c.surface.name)
    grid = np.vstack([np.hstack(tiles[r * 5 : r * 5 + 5]) for r in range(2)])
    path = os.path.join(OUT, "surfaces.png")
    write_png(path, np.round(grid * 255).astype(np.uint8))
    print("wrote", path)


# ---------------------------------------------------------------------------
# 8. Loading a brush with several paints
# ---------------------------------------------------------------------------
def multi_loading():
    c = Canvas(1300, 760, seed=5)
    W, R, Y, GR = P.TITANIUM_WHITE, P.CADMIUM_RED, P.CADMIUM_YELLOW, P.SAP_GREEN
    # double-loaded soft flat: a ribbon with a light edge
    br = Brush(B.FLAT_SABLE.sized(34)).load(R, 10).load(W, 4, across="left")
    c.stroke([(40, 90), (140, 40), (240, 120), (340, 60)], brush=br, seed=1)
    # the same load, twisted half a turn along the stroke: the colours change sides (a
    # folded ribbon)
    br = Brush(B.FLAT_SABLE.sized(34)).load(R, 10).load(W, 4, across="left")
    c.stroke(
        [(400, 80), (520, 60), (640, 100)], brush=br, angle=lambda u: 180 * u, seed=2
    )
    # one-stroke petals: a double-loaded flat pulled out from the centre
    for k in range(7):
        a = k * 2 * np.pi / 7
        cx, cy = 820, 90
        br = (
            Brush(B.FLAT_SABLE.sized(30))
            .load(P.QUINACRIDONE_MAGENTA, 3)
            .load(W, 2, across="right")
        )
        pts = [
            (cx + 8 * np.cos(a), cy + 8 * np.sin(a)),
            (cx + 40 * np.cos(a + 0.25), cy + 40 * np.sin(a + 0.25)),
            (cx + 62 * np.cos(a), cy + 62 * np.sin(a)),
        ]
        c.stroke(pts, brush=br, pressure=[0.6, 0.7, 0.2], seed=10 + k)
    # tip loading: a green round touched into yellow; each leaf starts yellow and turns
    # green
    for k in range(6):
        br = Brush(B.ROUND_SABLE.sized(16)).load(GR, 5).load(Y, 0.6, depth="tip")
        x0 = 960 + 50 * k
        c.stroke(
            [(x0, 140), (x0 + 20, 90), (x0 + 10, 40)],
            brush=br,
            pressure=[0.8, 0.6, 0.15],
            seed=20 + k,
        )
    # triple-loaded wide flat: a sky gradient in a single stroke
    br = (
        Brush(B.FLAT_HOG.sized(70))
        .load(P.COBALT_BLUE, 6, across=(0.0, 0.34))
        .load(W, 6, across=(0.33, 0.67))
        .load(P.CADMIUM_ORANGE, 6, across=(0.66, 1.0))
        .work(0.5)
    )
    c.stroke([(40, 230), (360, 220), (640, 240)], brush=br, seed=30)
    # paints picked up together, partly mixed: mixing 0.15, 0.45, 1.0
    for k, mix in enumerate((0.15, 0.45, 1.0)):
        br = Brush(B.FILBERT_HOG.sized(40)).load_mix(
            [(R, 1), (Y, 1), (W, 1)], 10, mixing=mix, seed=40 + k
        )
        y = 320 + 70 * k
        c.stroke([(40, y), (330, y - 15), (620, y + 5)], brush=br, seed=40 + k)
    # how long two paints stay apart on the brush: blending 0.05, 0.4, 1.0
    for k, blend in enumerate((0.05, 0.4, 1.0)):
        br = (
            Brush(B.FLAT_HOG.sized(36).with_(blending=blend))
            .load(R, 14, across="left")
            .load(Y, 14, across="right")
        )
        y = 300 + 70 * k
        c.stroke([(700, y), (1270, y)], brush=br, pressure=0.7, seed=50 + k)
    # stroking the brush on the palette first: 0, 2 and 6 passes
    for k, passes in enumerate((0, 2, 6)):
        br = (
            Brush(B.FLAT_SABLE.sized(36))
            .load(P.VIRIDIAN, 8, across="left")
            .load(Y, 8, across="right")
            .work(passes)
        )
        y = 540 + 60 * k
        c.stroke([(40, y), (330, y)], brush=br, seed=60 + k)
    # a double-loaded hog fan flicked upward: light and dark blades in every flick
    rng = np.random.default_rng(7)
    for k in range(16):
        br = (
            Brush(B.FAN_HOG.sized(34))
            .load(GR, 2)
            .load(P.LEMON_YELLOW, 1.2, across="left")
        )
        x = 420 + 26 * k + rng.normal(0, 5)
        c.stroke(
            [(x, 740), (x + rng.normal(0, 14), 560)],
            brush=br,
            pressure=[0.6, 0.1],
            seed=70 + k,
        )
    # a double-loaded palette knife: a two-tone slab
    br = (
        Brush(B.PALETTE_KNIFE.sized(60))
        .load(P.ULTRAMARINE_BLUE, 12)
        .load(W, 10, across="right")
    )
    c.stroke([(900, 560), (1250, 545)], brush=br, pressure=0.6, seed=80)
    # a dirty brush: only its lower side crosses wet red, so only that side carries red
    # on
    c.stroke(
        [(990, 690), (1010, 690)],
        R,
        amount=20,
        brush=B.FLAT_HOG.sized(30),
        angle=90,
        seed=90,
    )
    c.stroke(
        [(900, 672), (1250, 672)], W, amount=14, brush=B.FLAT_HOG.sized(36), seed=91
    )
    save(c, "multi_loading.png")


# ---------------------------------------------------------------------------
# 9. Rolling a round, and how deep the brush goes into the paint
# ---------------------------------------------------------------------------
def manipulations():
    c = Canvas(1300, 520, seed=6)
    R, Y, BL, W = P.CADMIUM_RED, P.CADMIUM_YELLOW, P.ULTRAMARINE_BLUE, P.TITANIUM_WHITE
    # a round with red on one side and yellow on the other: not rolled, rolled half a
    # turn, three turns
    for k, roll in enumerate((0, lambda u: 180 * u, lambda u: 1080 * u)):
        br = (
            Brush(B.ROUND.sized(34).with_(blending=0.1))
            .load(R, 16, across="left")
            .load(Y, 16, across="right")
        )
        y = 50 + 65 * k
        c.stroke([(30, y), (600, y)], brush=br, roll=roll, seed=k)
    # a hog round with three paints around the bundle, rolled back and forth along a
    # tapering branch
    br = Brush(B.ROUND_HOG.sized(26))
    br.load(P.BURNT_UMBER, 10, across=(0.0, 0.34)).load(
        P.YELLOW_OCHRE, 10, across=(0.33, 0.67)
    )
    br.load(P.IVORY_BLACK, 6, across=(0.66, 1.0))
    c.stroke(
        [(660, 180), (800, 120), (960, 100), (1120, 40), (1270, 30)],
        brush=br,
        roll=lambda u: 200 * np.sin(4 * np.pi * u),
        width=[30, 22, 16, 10, 6],
        seed=10,
    )
    # the same amount of paint, dipped to the tip, half way, all the way, and worked
    # into the heel
    for k, depth in enumerate((0.1, 0.5, 1.0, "heel")):
        br = Brush(B.FLAT_HOG.sized(30)).load(BL, 8, depth=depth)
        y = 280 + 60 * k
        c.stroke([(30, y), (600, y)], brush=br, seed=20 + k)
    # a brush dipped to the ferrule runs out early; pressing hard on the next stroke
    # squeezes out the heel
    br = Brush(B.FLAT_HOG.sized(30)).load(P.BURNT_SIENNA, 8, depth=1.0)
    c.stroke([(660, 280), (1270, 280)], brush=br, pressure=0.4, seed=40)
    c.stroke([(660, 340), (1270, 340)], brush=br, pressure=0.95, seed=41)
    # a round with umber in the heel, blue in the belly and white on the tip: the
    # colours come out in turn
    br = (
        Brush(B.ROUND.sized(30))
        .load(P.BURNT_UMBER, 6, depth="heel")
        .load(BL, 6)
        .load(W, 1.5, depth="tip")
    )
    c.stroke([(660, 410), (1270, 410)], brush=br, seed=30)
    c.stroke([(660, 470), (1270, 470)], brush=br, pressure=0.95, seed=31)
    save(c, "manipulations.png")


# ---------------------------------------------------------------------------
# 10. Impasto: thick paint, ridges and the body of the paint
# ---------------------------------------------------------------------------
RAKING = (-0.6, -0.8, 0.7)  # a lower light that shows the relief


def impasto():
    c = Canvas(1300, 720, seed=12)
    R, ORANGE, Y, W, BL = (
        P.CADMIUM_RED,
        P.CADMIUM_ORANGE,
        P.CADMIUM_YELLOW,
        P.TITANIUM_WHITE,
        P.ULTRAMARINE_BLUE,
    )
    hog = B.FLAT_HOG.sized(36)
    # the same hog flat loaded with more and more paint: colour, then relief
    for k, amount in enumerate((6, 15, 30, 60)):
        c.stroke(
            [(40 + 315 * k, 60), (300 + 315 * k, 52)],
            paint=R,
            amount=amount,
            brush=hog,
            seed=k,
        )
    # an overloaded brush pressed harder and harder squeezes paint out into ridges at
    # its edges
    for k, pressure in enumerate((0.2, 0.55, 0.9)):
        c.stroke(
            [(40 + 420 * k, 160), (390 + 420 * k, 150)],
            paint=BL,
            amount=45,
            brush=hog,
            pressure=pressure,
            seed=10 + k,
        )
    # the body of the paint: thinned with medium it levels out; impasto medium holds
    # every bristle mark
    for k, paint in enumerate(
        (
            Paint.mix((ORANGE, 1), (P.MEDIUM, 3)),
            ORANGE,
            Paint.mix((ORANGE, 2), (P.IMPASTO_MEDIUM, 1)),
        )
    ):
        c.stroke(
            [(40 + 420 * k, 260), (390 + 420 * k, 255)],
            paint=paint,
            amount=40,
            brush=hog,
            seed=20 + k,
        )
    # a big two-colour swoosh with a heavily loaded filbert, and a white one dragged
    # through it while wet
    c.stroke(
        [(60, 560), (200, 420), (420, 380), (640, 430)],
        paint=[(R, 90), (ORANGE, 50, "left")],
        brush=B.FILBERT_HOG.sized(70),
        pressure=[0.6, 0.45, 0.35, 0.2],
        seed=30,
    )
    c.stroke(
        [(120, 600), (300, 480), (520, 470)],
        paint=W,
        amount=50,
        brush=B.FILBERT_HOG.sized(36),
        pressure=[0.7, 0.4, 0.2],
        seed=31,
    )
    # thick dabs with a round, side by side and stacked, like a heavy highlight
    rng = np.random.default_rng(3)
    for k in range(14):
        x, y = 760 + rng.normal(0, 60), 520 + rng.normal(0, 45)
        c.stroke(
            [(x, y), (x + rng.normal(15, 6), y + rng.normal(-8, 5))],
            paint=[(Y, 40), (W, 20, "left")],
            brush=B.ROUND_HOG.sized(30),
            pressure=0.4,
            seed=40 + k,
        )
    # an overloaded rigger squiggle and a rolled two-colour round
    c.stroke(
        [(1000, 620), (1060, 420), (1130, 600), (1200, 410), (1260, 560)],
        paint=P.VIRIDIAN,
        amount=40,
        brush=B.ROUND.sized(20),
        pressure=0.5,
        seed=50,
    )
    save(c, "impasto.png", light=RAKING)


# ---------------------------------------------------------------------------
# 11. Painting knives
# ---------------------------------------------------------------------------
def knives():
    c = Canvas(1300, 1000, seed=21)
    W, Y, ORANGE, R, M = (
        P.TITANIUM_WHITE,
        P.CADMIUM_YELLOW,
        P.CADMIUM_ORANGE,
        P.CADMIUM_RED,
        P.QUINACRIDONE_MAGENTA,
    )
    CE, UB, G, UM = P.CERULEAN_BLUE, P.ULTRAMARINE_BLUE, P.SAP_GREEN, P.BURNT_UMBER
    sky = Paint.mix((CE, 1), (W, 2))
    # pressure: a thick slab, a firm one, one scraped thin, one scraped right into the
    # weave
    for k, pressure in enumerate((0.15, 0.4, 0.7, 0.95)):
        c.stroke(
            [(40 + 315 * k, 70), (280 + 315 * k, 65)],
            paint=sky,
            amount=70,
            brush=B.PALETTE_KNIFE.sized(56),
            pressure=pressure,
            seed=k,
        )
    # the knives and their marks
    y = 210
    c.stroke(
        [(40, y), (200, y - 10)],
        paint=[(ORANGE, 40), (Y, 25, "left")],
        brush=B.KNIFE_TROWEL,
        pressure=0.35,
        seed=10,
    )
    # petals: the flat of the knife pressed down at the outer edge and pulled in towards
    # the centre, easing off (small trowel, then pear)
    for k, (cx, knife, paint, wd) in enumerate(
        (
            (300, B.KNIFE_SMALL, [(M, 12), (W, 8, "left")], 30),
            (450, B.KNIFE_PEAR, [(Y, 14), (ORANGE, 8, "right")], 36),
        )
    ):
        for j in range(4):
            a = np.radians(200 + 47 * j)
            c.stroke(
                [
                    (cx + 52 * np.cos(a), y + 25 + 52 * np.sin(a)),
                    (cx + 6 * np.cos(a), y + 25 + 6 * np.sin(a)),
                ],
                paint=paint,
                brush=knife,
                angle=10,
                width=[wd, 0.35 * wd],
                pressure=[0.3, 0.55],
                seed=11 + 4 * k + j,
            )
    # on edge
    c.stroke(
        [(560, y + 30), (760, y - 30)],
        paint=UM,
        amount=25,
        brush=B.KNIFE_STRAIGHT,
        angle=90,
        seed=20,
    )
    c.stroke(
        [(800, y - 25), (960, y - 25), (980, y + 30)],
        paint=R,
        amount=40,
        brush=B.KNIFE_ANGLED,
        pressure=0.4,
        seed=21,
    )
    for k in range(4):  # spoon: round dabs
        c.stroke(
            [(1020 + 34 * k, y + 15 * (-1) ** k)],
            paint=W,
            amount=30,
            brush=B.KNIFE_SPOON,
            pressure=0.3,
            seed=22 + k,
        )
    # comb through wet paint, and a scraper taking a wet area back to the canvas
    y = 350
    c.stroke(
        [(40, y), (560, y)], paint=G, amount=16, brush=B.FLAT_HOG.sized(80), seed=30
    )
    c.stroke([(60, y + 5), (540, y - 5)], brush=B.KNIFE_COMB, pressure=0.6, seed=31)
    c.stroke(
        [(640, y), (1260, y)], paint=UB, amount=16, brush=B.FLAT_HOG.sized(90), seed=32
    )
    c.stroke(
        [(860, y - 60), (1000, y + 60)],
        brush=B.SCRAPER,
        angle=-30,
        pressure=0.95,
        seed=33,
    )
    # knife on wet paint: a clean skin on top; over dry ridges: broken colour; lifting
    # off: torn peaks
    y = 480
    c.stroke(
        [(40, y), (360, y)], paint=UB, amount=16, brush=B.FLAT_HOG.sized(70), seed=40
    )
    c.stroke(
        [(70, y), (330, y + 5)],
        paint=W,
        amount=45,
        brush=B.PALETTE_KNIFE.sized(44),
        pressure=0.3,
        seed=41,
    )
    for k in range(7):
        c.stroke(
            [(440 + 20 * k, y - 50), (460 + 20 * k, y + 50)],
            paint=Y,
            amount=40,
            brush=B.FLAT_HOG.sized(16),
            seed=42 + k,
        )
    c.dry()
    c.stroke(
        [(420, y + 5), (640, y - 5)],
        paint=M,
        amount=18,
        brush=B.PALETTE_KNIFE.sized(60),
        pressure=0.5,
        seed=50,
    )
    for k, lift in enumerate((0.0, 1.0)):
        c.stroke(
            [(760 + 260 * k, y), (960 + 260 * k, y)],
            paint=[(R, 45), (ORANGE, 25, "left")],
            brush=B.PALETTE_KNIFE.sized(56),
            pressure=0.3,
            lift=lift,
            seed=51 + k,
        )
    knife_study(c, 30, 570, 1240, 410, np.random.default_rng(5))
    save(c, "knives.png", light=RAKING)


def knife_drag():
    """White knifed over wet purple; then a clean knife dragged across the white's
    edge, into the purple."""
    purple = Paint.mix(
        (P.DIOXAZINE_PURPLE, 2), (P.QUINACRIDONE_MAGENTA, 1), (P.TITANIUM_WHITE, 1)
    )
    c = Canvas(900, 560, seed=4)
    for y in range(40, 540, 30):  # a thick purple field
        c.stroke(
            [(20, y), (880, y + 3)],
            paint=purple,
            amount=200,
            brush=B.KNIFE_STRAIGHT.sized(60),
            pressure=0.35,
            seed=y,
        )
    # thick white over the left half, wet into wet
    for y in range(70, 510, 30):
        c.stroke(
            [(60, y), (430, y - 2)],
            paint=P.TITANIUM_WHITE,
            amount=90,
            brush=B.KNIFE_STRAIGHT.sized(56),
            pressure=0.4,
            seed=1000 + y,
        )
    # a clean knife dragged from the white into the purple
    for k, pressure in enumerate((0.3, 0.6, 0.9)):
        y = 130 + 150 * k
        c.stroke(
            [(180, y), (480, y + 6), (760, y)],
            brush=B.PALETTE_KNIFE.sized(64),
            pressure=pressure,
            seed=7 + k,
        )
    save(c, "knife_drag.png", light=RAKING)


def knife_study(c, x0, y0, w, h, rng):
    """A small field of flowers under a summer sky, all knife work."""
    W, Y, ORANGE, R = (
        P.TITANIUM_WHITE,
        P.CADMIUM_YELLOW,
        P.CADMIUM_ORANGE,
        P.CADMIUM_RED,
    )

    def seed():
        return int(rng.integers(1 << 30))

    horizon = y0 + 0.45 * h
    # sky, first layer: long overlapping pulls, firm pressure, deep blue above to pale
    # at the horizon
    rows = np.arange(y0 + 10, horizon - 5, 16)
    for y in rows:
        t = (y - y0) / (horizon - y0)
        x = x0 - rng.uniform(20, 80)
        while x < x0 + w:
            L = rng.uniform(120, 240)
            tint = Paint.mix(
                (P.CERULEAN_BLUE, 1),
                (P.ULTRAMARINE_BLUE, 0.4 * (1 - t)),
                (W, 0.6 + 4.0 * t + rng.uniform(0, 0.6)),
            )
            yy = y + rng.normal(0, 3)
            c.stroke(
                [
                    (x, yy),
                    (x + L / 2, yy + rng.normal(0, 3)),
                    (x + L, yy + rng.normal(0, 4)),
                ],
                paint=tint,
                amount=22,
                brush=B.KNIFE_STRAIGHT.sized(rng.uniform(40, 56)),
                pressure=[0.6, 0.5, 0.7],
                seed=seed(),
            )
            x += L * rng.uniform(0.6, 0.9)
    # clouds: thick white turned and twisted under the knife, light pressure, lifted off
    for _k in range(int(w / 80)):
        cx, cy = x0 + rng.uniform(0, w), y0 + rng.uniform(0.15, 0.65) * (horizon - y0)
        for _ in range(5):
            x, y = cx + rng.normal(0, 28), cy + rng.normal(0, 7)
            L = rng.uniform(25, 55)
            turn = rng.normal(0, 40)
            c.stroke(
                [
                    (x, y),
                    (x + L * 0.5, y - rng.uniform(2, 8)),
                    (x + L, y + rng.normal(0, 3)),
                ],
                paint=[
                    (Paint.mix((W, 8), (ORANGE, 0.2)), 26),
                    (Paint.mix((W, 3), (P.CERULEAN_BLUE, 1)), 10, "right"),
                ],
                brush=B.PALETTE_KNIFE.sized(rng.uniform(28, 40)),
                angle=lambda u, t=turn: t * u,
                pressure=[0.2, 0.35, 0.5],
                lift=0.9,
                seed=seed(),
            )
    # distant hills
    for _k in range(int(w / 25)):
        x = x0 + rng.uniform(-60, w)
        c.stroke(
            [
                (x, horizon - rng.uniform(0, 10)),
                (x + rng.uniform(80, 160), horizon + rng.uniform(0, 10)),
            ],
            paint=Paint.mix(
                (P.ULTRAMARINE_BLUE, 1), (W, 2.5 + rng.uniform(0, 1)), (P.SAP_GREEN, 1)
            ),
            amount=25,
            brush=B.KNIFE_STRAIGHT.sized(44),
            angle=50,
            pressure=0.6,
            seed=seed(),
        )
    # the field: a first layer pulled flat along the horizon, then curved overlapping
    # pulls, pale and thin far away, rich and thick in front
    for y in np.arange(horizon - 4, horizon + 60, 14):
        x = x0 - rng.uniform(20, 80)
        while x < x0 + w:
            L = rng.uniform(100, 200)
            green = Paint.mix((P.SAP_GREEN, 2), (Y, rng.uniform(0.5, 1.5)), (W, 1.0))
            c.stroke(
                [(x, y), (x + L, y + rng.normal(0, 3))],
                paint=green,
                amount=22,
                brush=B.KNIFE_STRAIGHT.sized(44),
                pressure=0.6,
                seed=seed(),
            )
            x += L * rng.uniform(0.6, 0.9)
    for _k in range(int(w / 6)):
        near = rng.uniform(0, 1) ** 0.8
        y = horizon + (y0 + h - horizon) * near
        x = x0 + rng.uniform(-60, w)
        green = Paint.mix(
            (P.SAP_GREEN, 2),
            (Y, rng.uniform(0.2, 1.8)),
            (P.VIRIDIAN, rng.uniform(0, 0.8)),
            (W, 1.2 * (1 - near) ** 2),
            (P.BURNT_UMBER, 0.3 * near * rng.uniform(0, 1)),
        )
        L = rng.uniform(50, 120) * (0.5 + near)
        bend = rng.normal(0, 6)
        c.stroke(
            [(x, y), (x + L / 2, y + bend), (x + L, y + rng.normal(0, 3))],
            paint=[(green, 28), (Y, 5, "left")],
            brush=B.KNIFE_TROWEL.sized(26 + 30 * near),
            angle=rng.normal(25, 20),
            pressure=[0.65 - 0.3 * near, 0.5, 0.75],
            seed=seed(),
        )
    # grass and stems: the loaded edge of a straight knife touched to the canvas
    edge = B.KNIFE_STRAIGHT.with_(aspect=0.035, name="knife edge")
    for _k in range(int(w / 3)):
        near = rng.uniform(0, 1) ** 0.6
        x, y = x0 + rng.uniform(0, w), horizon + 20 + (y0 + h - horizon - 20) * near
        green = Paint.mix(
            (P.SAP_GREEN, 1), (Y, rng.uniform(0.3, 2.0)), (W, 0.4 * rng.uniform(0, 1))
        )
        c.stroke(
            [(x, y)],
            paint=green,
            amount=120,
            brush=edge.sized(10 + 40 * near),
            angle=90 + rng.normal(0, 14),
            follow_path=False,
            pressure=0.5,
            seed=seed(),
        )
    # flowers: poppies (small trowel) and daisies (pear), each petal the flat of the
    # knife pressed down at its outer edge and pulled in towards the centre, easing off
    for _k in range(int(w / 60)):
        near = rng.uniform(0, 1) ** 0.7
        cx, cy = (
            x0 + rng.uniform(30, w - 30),
            horizon + 40 + (y0 + h - horizon - 60) * near,
        )
        size = 0.45 + 0.75 * near
        poppy = rng.uniform() < 0.7
        n = 5 if poppy else 9
        for j in range(n):
            a = j * 2 * np.pi / n + rng.uniform(0, 1)
            r0, r1 = (24 if poppy else 30) * size, 5 * size
            wd = (26 if poppy else 13) * size
            c.stroke(
                [
                    (cx + r0 * np.cos(a), cy + 0.7 * r0 * np.sin(a)),
                    (cx + r1 * np.cos(a), cy + 0.7 * r1 * np.sin(a)),
                ],
                paint=(
                    [(R, 10), (ORANGE, 4, "left")]
                    if poppy
                    else [(W, 10), (Y, 3, "right")]
                ),
                brush=(B.KNIFE_SMALL if poppy else B.KNIFE_PEAR),
                width=[wd, 0.35 * wd],
                angle=10,
                pressure=[0.3, 0.5],
                seed=seed(),
            )
        c.stroke(
            [(cx, cy)],
            paint=P.IVORY_BLACK if poppy else Y,
            amount=40,
            brush=B.KNIFE_SPOON.sized(11 * size),
            pressure=0.3,
            seed=seed(),
        )


# ---------------------------------------------------------------------------
# 12. A dirty brush: paint picked up from the canvas, carried into the next stroke
# ---------------------------------------------------------------------------
def dirty_brush():
    c = Canvas(720, 400, seed=7)
    # (layers of wet blue, scraped back to a thin film, pressure of the drag)
    rows = ((1, True, 0.6), (1, False, 0.6), (2, False, 0.6), (1, False, 0.95))
    for k, (layers, scraped, pressure) in enumerate(rows):
        y = 50 + 100 * k
        # wet blue: thin (scraped back), medium, thick (two layers)
        for _ in range(layers):
            for yy in range(y - 28, y + 30, 8):
                c.stroke(
                    [(20, yy), (330, yy)],
                    P.ULTRAMARINE_BLUE,
                    amount=8,
                    brush=B.FLAT_HOG,
                    pressure=0.1,
                )
        if scraped:
            for yy in range(y - 30, y + 36, 14):
                c.stroke([(10, yy), (340, yy)], brush=B.SCRAPER, pressure=0.7)
        # a full yellow brush dragged through it, then lifted and used again on
        # clean canvas without new paint
        br = Brush(B.FLAT_HOG).load(P.CADMIUM_YELLOW, 30)
        c.stroke([(60, y), (220, y)], brush=br, pressure=pressure)
        c.stroke([(380, y), (690, y)], brush=br, pressure=0.6)
    save(c, "dirty_brush.png")


# ---------------------------------------------------------------------------
# 13. Reverse glass: painted on the back, seen through the front
# ---------------------------------------------------------------------------
PETAL = Paint.mix((P.CADMIUM_RED, 3), (P.CADMIUM_ORANGE, 1))
LEAF = Paint.mix((P.SAP_GREEN, 2), (P.CADMIUM_YELLOW, 1))
GLASS_SKY = Paint.mix((P.ULTRAMARINE_BLUE, 2), (P.TITANIUM_WHITE, 1))
# (angle, height up the stem) of the two leaves
LEAVES = ((-0.55, 150), (0.6, 125))


def _ellipse(cx, cy, rx, ry, angle, n=14):
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    ca, sa = np.cos(angle), np.sin(angle)
    pts = [
        (
            cx + ca * rx * np.cos(u) - sa * ry * np.sin(u),
            cy + sa * rx * np.cos(u) + ca * ry * np.sin(u),
        )
        for u in t
    ]
    return [*pts, *pts[:2]]  # closed, with the join smooth


def _leaf(cx, cy, angle, up, n=12):
    """Points around a leaf from the stem at height ``up``, slanting ``angle``."""
    side = 1.0 if angle >= 0 else -1.0
    dx, dy = side * np.cos(angle), -np.sin(abs(angle))  # along the leaf
    nx, ny = -dy, dx  # across it
    u = np.linspace(0.0, 1.0, n)
    half = 16 * np.sin(np.pi * u)
    x0, y0 = cx + 7 + 75 * u * dx, cy + up + 75 * u * dy
    pts = list(zip(x0 + half * nx, y0 + half * ny, strict=True)) + list(
        zip((x0 - half * nx)[::-1], (y0 - half * ny)[::-1], strict=True)
    )
    return [*pts, *pts[:2]]


def _flower_step(c, step, cx=180, cy=150):
    """One stage of a folk-art flower on glass."""

    def line(pts, w=5):  # a black line, with enough paint to reach its end
        run = float(np.sum(np.hypot(*np.diff(np.asarray(pts, float), axis=0).T)))
        c.stroke(
            pts,
            P.IVORY_BLACK,
            amount=1.5 * run / w + 2,
            brush=B.ROUND_SABLE.sized(w),
            pressure=0.7,
        )

    petals = [
        (cx + 58 * np.cos(a), cy + 58 * np.sin(a), a)
        for a in np.arange(6) * np.pi / 3 + 0.3
    ]
    stem = [(cx + 20, cy + 96), (cx + 12, cy + 160), (cx + 6, cy + 240)]
    if step == "outlines":
        for x, y, a in petals:
            line(_ellipse(x, y, 40, 21, a))
        line(_ellipse(cx, cy, 23, 23, 0))
        line(stem, 6)
        for angle, up in LEAVES:
            line(_leaf(cx, cy, angle, up))
    elif step == "details":  # a white gleam on each petal, black seeds
        for x, y, a in petals:
            ca, sa = np.cos(a), np.sin(a)
            gleam = [(x + d * ca - 6 * sa, y + d * sa + 6 * ca) for d in (-14, 10)]
            c.stroke(gleam, P.TITANIUM_WHITE, amount=3, brush=B.ROUND.sized(7))
        for a in np.arange(7) * 2 * np.pi / 7:
            x, y = cx + 11 * np.cos(a), cy + 11 * np.sin(a)
            c.stroke(
                [(x, y), (x + 1, y + 1)],
                P.IVORY_BLACK,
                amount=2,
                brush=B.ROUND.sized(5),
            )
    elif step == "colours":
        for x, y, a in petals:
            ca, sa = np.cos(a), np.sin(a)
            c.stroke(
                [(x + d * ca, y + d * sa) for d in (-38, -20, 0, 20, 38)],
                PETAL,
                amount=10,
                brush=B.FILBERT_HOG.sized(24),
                width=[8, 36, 42, 36, 8],
                pressure=0.55,
            )
        c.stroke(
            _ellipse(cx, cy, 8, 8, 0, 8),
            P.CADMIUM_YELLOW,
            amount=12,
            brush=B.ROUND.sized(30),
        )
        c.stroke(stem, LEAF, amount=12, brush=B.ROUND.sized(12))
        for angle, up in LEAVES:
            side = 1.0 if angle >= 0 else -1.0
            c.stroke(
                [
                    (
                        cx + 7 + side * 75 * u * np.cos(angle),
                        cy + up - 75 * u * np.sin(abs(angle)),
                    )
                    for u in (0.0, 0.3, 0.6, 1.0)
                ],
                LEAF,
                amount=10,
                brush=B.FILBERT_HOG.sized(30),
                width=[6, 30, 30, 6],
            )
    else:  # the background, all over
        for y in range(-6, c.height + 20, 20):
            c.stroke(
                [(-20, y), (c.width + 20, y + 4)],
                GLASS_SKY,
                amount=16,
                brush=B.FLAT_HOG.sized(36),
                pressure=0.6,
            )


def reverse_glass():
    """The same flower painted on the front of the glass in the usual order, on
    the back in reverse order (seen through the glass), and on the back in the
    usual order."""
    usual = ["background", "colours", "details", "outlines"]
    panels = []
    for order, reverse in ((usual, False), (usual[::-1], True), (usual, True)):
        c = Canvas(360, 400, surface=S.GLASS, reverse=reverse, seed=11)
        for step in order:
            _flower_step(c, step)
            c.dry()  # each stage dries before the next goes on
        panels.append(np.pad(c.render(), ((0, 0), (6, 6), (0, 0)), constant_values=1.0))
    path = os.path.join(OUT, "reverse_glass.png")
    write_png(path, np.round(np.hstack(panels) * 255).astype(np.uint8))
    print("wrote", path)


if __name__ == "__main__":
    for fn in (
        mixing_chart,
        strokes,
        blending,
        glazing,
        brush_rack,
        brush_uses,
        surface_sampler,
        multi_loading,
        manipulations,
        impasto,
        knives,
        knife_drag,
        dirty_brush,
        reverse_glass,
    ):
        t0 = time.time()
        fn()
        print(f"  {fn.__name__}: {time.time() - t0:.1f}s")
