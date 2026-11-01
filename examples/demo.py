"""Generate a set of demo images showing what the API does.

    python examples/demo.py [output_dir]
"""

import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from oilpaint import Bezier, Brush, Canvas, Paint, brushes as B, palette as P, surfaces as S  # noqa: E402
from oilpaint import spectral as sp  # noqa: E402
from oilpaint.png import write_png  # noqa: E402

OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(__file__), "output")
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
            km = Paint.mix((a, 1 - t), (b, t)).color if 0 < t < 1 else (a.color if t == 0 else b.color)
            lin = (1 - t) * sp.srgb_to_linear(a.color) + t * sp.srgb_to_linear(b.color)
            x = gap + i * cell
            img[y:y + cell, x:x + cell] = km
            img[y + cell:y + 2 * cell - 6, x:x + cell] = sp.linear_to_srgb(lin)
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
        c.stroke([(40, y), (240, y - 15), (430, y + 5)], paint=P.CADMIUM_RED, amount=amount, width=30, seed=i)
    # Wet-on-wet: yellow pulled through the wet blue.
    c.stroke([(480, 70), (650, 40), (860, 90)], paint=P.ULTRAMARINE_BLUE, amount=12, width=44, seed=10)
    c.stroke([(480, 150), (650, 120), (860, 170)], paint=P.ULTRAMARINE_BLUE, amount=12, width=44, seed=11)
    c.stroke([(560, 20), (590, 140), (560, 260)], paint=P.CADMIUM_YELLOW, amount=10, width=34, pressure=0.3, seed=12)
    c.stroke([(760, 20), (790, 140), (760, 260)], paint=P.CADMIUM_YELLOW, amount=10, width=34, pressure=0.9, seed=13)
    # Varying width and pressure along a Bezier curve.
    c.stroke(Bezier([(60, 300), (200, 220), (300, 480), (440, 380), (520, 320), (620, 300), (700, 420)]),
             paint=P.VIRIDIAN, amount=14, width=[12, 40, 20], pressure=[0.2, 0.8, 0.4], seed=20)
    # A white stroke dragged down through the wet reds: the brush gets dirty and
    # keeps tinting the next stroke, painted without reloading or cleaning.
    brush = c.stroke([(140, 15), (150, 140), (140, 255)], paint=P.TITANIUM_WHITE, amount=26, width=30, seed=30)
    c.stroke([(560, 330), (700, 300), (870, 360)], brush=brush, width=30, seed=31)
    c.stroke([(560, 420), (700, 390), (870, 450)], brush=brush, width=30, seed=32)
    save(c, "strokes.png")


# ---------------------------------------------------------------------------
# 3. Blending two wet colors with a clean brush
# ---------------------------------------------------------------------------
def blending():
    c = Canvas(700, 300, seed=3)
    for x in range(60, 340, 28):
        c.stroke([(x, 20), (x + 4, 150), (x, 280)], paint=P.ULTRAMARINE_BLUE, amount=10, width=40)
    for x in range(360, 650, 28):
        c.stroke([(x, 20), (x - 4, 150), (x, 280)], paint=P.CADMIUM_YELLOW, amount=10, width=40)
    # Blend the seam with a clean brush: short strokes back and forth across it,
    # the way a painter would. No paint is loaded, the brush only moves wet paint.
    rng = np.random.default_rng(0)
    for rnd in range(3):
        for k, y in enumerate(range(25 + 4 * rnd, 285, 9)):
            x0 = 350 + rng.normal(0, 25)
            half = 70 - 15 * rnd
            xs = (x0 - half, x0 + half) if (k + rnd) % 2 == 0 else (x0 + half, x0 - half)
            c.stroke([(xs[0], y), (xs[1], y + rng.normal(0, 3))], width=26, pressure=0.55)
    # A last pass of long, light strokes to smooth the transition.
    for y in range(20, 290, 14):
        c.stroke([(230, y), (470, y + rng.normal(0, 2))], width=30, pressure=0.25, pickup=0.25)
    save(c, "blending.png")


# ---------------------------------------------------------------------------
# 4. Glazing: transparent paint over a dry underpainting
# ---------------------------------------------------------------------------
def glazing():
    c = Canvas(720, 330, seed=4)
    # Underpainting: three horizontal value bands of umber + white, left to dry.
    values = [Paint.mix((P.BURNT_UMBER, 1), (P.TITANIUM_WHITE, w)) for w in (0.2, 1.5, 8)]
    for band, paint in enumerate(values):
        for y in range(25 + 100 * band, 100 + 100 * band, 22):
            c.stroke([(15, y), (360, y - 4), (705, y)], paint=paint, amount=30, width=30)
    c.dry()
    # Over it, vertical strokes: two transparent glazes (pigment thinned with
    # clear medium), then two opaque paints that hide the underpainting.
    columns = [Paint.mix((P.ALIZARIN_CRIMSON, 1), (P.MEDIUM, 2)), Paint.mix((P.PHTHALO_BLUE, 1), (P.MEDIUM, 5)),
               P.CADMIUM_RED, P.CERULEAN_BLUE]
    for i, paint in enumerate(columns):
        for j in range(4):
            x = 40 + 170 * i + 30 * j
            c.stroke([(x, 10), (x + 3, 165), (x, 320)], paint=paint, amount=14, width=34, pressure=0.35)
    save(c, "glazing.png")


# ---------------------------------------------------------------------------
# 5. A small landscape
# ---------------------------------------------------------------------------
def landscape():
    rng = np.random.default_rng(7)
    W, H, horizon = 960, 640, 380
    c = Canvas(W, H, seed=5)
    white, ultra, cerulean = P.TITANIUM_WHITE, P.ULTRAMARINE_BLUE, P.CERULEAN_BLUE
    yellow, orange, red, alizarin = P.CADMIUM_YELLOW, P.CADMIUM_ORANGE, P.CADMIUM_RED, P.ALIZARIN_CRIMSON
    umber, sap, ochre = P.BURNT_UMBER, P.SAP_GREEN, P.YELLOW_OCHRE

    def ramp(keys, t):
        """Palette-knife mix between recipes at key positions 0..1."""
        pos = np.linspace(0, 1, len(keys))
        i = min(int(np.searchsorted(pos, t, side="right")) - 1, len(keys) - 2)
        f = float(np.clip((t - pos[i]) / (pos[i + 1] - pos[i]), 0, 1))
        return Paint.mix((keys[i], 1 - f), (keys[i + 1], f))

    def band(y0, y1, keys, width, amount, step, length=(140, 260), wobble=4.0):
        for y in np.arange(y0, y1, step):
            paint = ramp(keys, (y - y0) / max(y1 - y0, 1))
            x = -60 + rng.uniform(0, 60)
            while x < W:
                ln = rng.uniform(*length)
                j = rng.normal(0, wobble, 3)
                c.stroke([(x, y + j[0]), (x + ln / 2, y + j[1] - 2), (x + ln, y + j[2])], paint=paint,
                         amount=amount * rng.uniform(0.8, 1.2), width=width * rng.uniform(0.85, 1.15), pressure=0.45)
                x += ln * rng.uniform(0.55, 0.8)

    # 1. Sky, wet into wet: deep blue at the top, a lavender-grey transition (blue
    #    into yellow would turn green, just like real paint), warm glow at the horizon.
    sky = [Paint.mix((ultra, 1), (white, 2.0)),
           Paint.mix((cerulean, 1), (white, 3.0), (alizarin, 0.06)),
           Paint.mix((white, 5), (alizarin, 0.12), (ultra, 0.06), (yellow, 0.1)),
           Paint.mix((white, 4), (yellow, 0.25), (orange, 0.12), (alizarin, 0.05)),
           Paint.mix((white, 3), (yellow, 0.45), (orange, 0.45))]
    band(0, horizon + 15, sky, width=34, amount=7, step=15)
    for y in np.arange(10, horizon, 18):   # soften with a clean brush
        c.stroke([(-20, y), (W / 2, y + rng.normal(0, 3)), (W + 20, y)], width=38, pressure=0.3, pickup=0.25)
    c.dry()

    # 2. Sun: one thick impasto dab, then distant hills over the dry sky.
    sun = (640, 318)
    c.stroke([sun], paint=Paint.mix((white, 12), (yellow, 1)), amount=40, width=52, pressure=0.6)
    far = Paint.mix((ultra, 1), (alizarin, 0.35), (white, 4.5), (umber, 0.1))
    near = Paint.mix((ultra, 1), (sap, 1.2), (white, 1.6), (umber, 0.3))
    for paint, base, amp, off in ((far, 60, 30, 0.0), (near, 22, 22, 2.0)):
        def ridge(x):
            return horizon - base - amp * np.sin(x / 150 + off) - 10 * np.sin(x / 43 + off)
        # Strokes follow the hill's contour, from the ridge line down to the horizon.
        for d in np.arange(8, base + amp + 30, 12):
            x = -60 + rng.uniform(0, 40)
            while x < W + 20:
                xs = np.linspace(x, x + 160, 6)
                ys = np.minimum(ridge(xs) + d, horizon + 6)
                c.stroke(list(zip(xs, ys)), paint=paint, amount=5, width=22, pressure=0.5)
                x += rng.uniform(100, 140)

    # 3. Water reflects the sky upside down: warm near the horizon, deep blue in front.
    water = [Paint.mix((white, 3), (yellow, 0.35), (orange, 0.35)),
             Paint.mix((white, 4), (alizarin, 0.12), (ultra, 0.15), (yellow, 0.05)),
             Paint.mix((cerulean, 1), (white, 2.2), (alizarin, 0.05)),
             Paint.mix((ultra, 1), (cerulean, 0.6), (white, 1.0)),
             Paint.mix((ultra, 1), (P.PHTHALO_BLUE, 0.15), (white, 0.5))]
    band(horizon + 4, H + 12, water, width=18, amount=5, step=9, length=(70, 170), wobble=1.5)
    # Glints of light laid on top with a loaded brush and a light touch.
    glint = Paint.mix((white, 14), (yellow, 1))
    for y in np.arange(horizon + 12, H - 30, 8):
        spread = 12 + (y - horizon) * 0.4
        for _ in range(2):
            x = sun[0] + rng.normal(0, spread)
            ln = rng.uniform(12, 40) * (1 + (y - horizon) / 300)
            c.stroke([(x - ln / 2, y), (x + ln / 2, y + rng.normal(0, 1))], paint=glint, amount=12,
                     width=rng.uniform(4, 8), pressure=0.2, mixing=0.0, pickup=0.1)

    # 4. Foreground bank, then grasses and poppies painted wet into wet.
    for k in range(12):
        dy = 12 * k
        c.stroke([(-30, 500 + dy), (160, 520 + dy), (330, 580 + dy * 0.7), (420, 660)],
                 paint=Paint.mix((sap, 1), (umber, 0.4 + 0.08 * k), (ochre, 0.3)), amount=9, width=30, pressure=0.6)
    for _ in range(110):
        x0 = rng.uniform(-20, 400)
        y0 = 525 + 0.3 * x0 + rng.uniform(0, 110)
        h = rng.uniform(35, 120)
        tip = (x0 + rng.normal(10, 12), y0 - h)
        mid = ((x0 + tip[0]) / 2 + rng.normal(0, 5), (y0 + tip[1]) / 2)
        col = Paint.mix((sap, 1), (yellow, rng.uniform(0.0, 1.2)), (ochre, rng.uniform(0, 0.6)), (white, rng.uniform(0, 0.4)))
        c.stroke([(x0, y0), mid, tip], paint=col, amount=rng.uniform(3, 6), width=[6, 4, 1.5], pressure=[0.7, 0.4, 0.1])
    poppy = Paint.mix((red, 1), (orange, 0.15))
    for _ in range(14):
        x, y = rng.uniform(20, 340), rng.uniform(470, 600) + 0.15 * rng.uniform(0, 300)
        c.stroke([(x, y), (x + rng.uniform(3, 7), y + rng.uniform(-2, 2))], paint=poppy, amount=45,
                 width=rng.uniform(9, 15), pressure=0.6, mixing=0.1)

    save(c, "landscape.png")


# ---------------------------------------------------------------------------
# 6. The brush rack: every preset doing the same five marks
# ---------------------------------------------------------------------------
def brush_rack():
    rows = [("ROUND", P.ULTRAMARINE_BLUE), ("ROUND_SABLE", P.ULTRAMARINE_BLUE), ("LINER", P.IVORY_BLACK),
            ("RIGGER", P.BURNT_UMBER), ("MOP", P.CERULEAN_BLUE), ("DEER_FOOT", P.SAP_GREEN),
            ("FLAT_HOG", P.CADMIUM_RED), ("BRIGHT", P.CADMIUM_RED), ("ANGLED", P.CADMIUM_ORANGE),
            ("DAGGER", P.ALIZARIN_CRIMSON), ("FILBERT_HOG", P.VIRIDIAN), ("FILBERT_SABLE", P.VIRIDIAN),
            ("FAN_HOG", P.SAP_GREEN), ("FAN_SOFT", P.YELLOW_OCHRE), ("PALETTE_KNIFE", P.TITANIUM_WHITE),
            ("BLENDER", P.COBALT_BLUE)]
    c = Canvas(1400, 70 * len(rows) + 20, seed=1)
    for k, (name, paint) in enumerate(rows):
        b = getattr(B, name)
        y = 45 + 70 * k
        load = 12 * max(1.0, 20.0 / b.width)        # small brushes need more loads to go as far
        c.stroke([(30, y), (330, y)], paint, amount=load, brush=b, seed=k)                    # straight
        c.stroke([(380, y + 15), (480, y - 18), (580, y + 15), (680, y - 10)], paint, amount=load, brush=b,
                 pressure=[0.1, 0.9, 0.2, 0.6], seed=k + 100)                                  # pressure S
        c.stroke([(740, y)], paint, amount=min(load, 3), brush=b, seed=k + 200)               # dab
        c.stroke([(800, y + 18), (900, y - 18)], paint, amount=load, brush=b, angle=90, seed=k + 300)   # on edge
        c.stroke([(960, y), (1360, y)], paint, amount=load * 1.3, brush=b, angle=lambda u: 90 * u,
                 seed=k + 400)                                                                # twisted
    save(c, "brush_rack.png")
    print("  rows:", ", ".join(r for r, _ in rows))


# ---------------------------------------------------------------------------
# 7. What the brushes are for
# ---------------------------------------------------------------------------
def brush_uses():
    rng = np.random.default_rng(3)
    c = Canvas(1200, 560, seed=2)
    # calligraphy: a flat held at a fixed angle on the canvas -> thick and thin
    loop = [(60 + 50 * np.cos(a) + 18 * a, 120 + 60 * np.sin(a)) for a in np.linspace(0, 4 * np.pi, 30)]
    c.stroke(loop, P.IVORY_BLACK, amount=60, brush=B.FLAT_SABLE, angle=40, follow_path=False)
    # filbert leaves: press, then lift
    for k in range(7):
        a = -0.6 + k * 0.2
        x0, y0 = 380 + 20 * k, 210
        c.stroke([(x0, y0), (x0 + 60 * np.sin(a), y0 - 110 * np.cos(a))], P.SAP_GREEN, amount=6,
                 brush=B.FILBERT_SABLE, pressure=[0.2, 0.9, 0.1], seed=k)
    # hog fan flicked upward: grass
    for k in range(14):
        x = 600 + k * 20 + rng.normal(0, 4)
        c.stroke([(x, 220), (x + rng.normal(0, 12), 110)], Paint.mix((P.SAP_GREEN, 1), (P.YELLOW_OCHRE, k % 3 * 0.3)),
                 amount=3, brush=B.FAN_HOG.sized(30), pressure=[0.6, 0.1], seed=k)
    # deer foot stippling with a dry brush: foliage
    for k in range(70):
        x, y = 1000 + rng.normal(0, 55), 140 + rng.normal(0, 40)
        c.stroke([(x, y)], Paint.mix((P.VIRIDIAN, 1), (P.YELLOW_OCHRE, rng.uniform(0, 1))), amount=0.4,
                 brush=B.DEER_FOOT, pressure=rng.uniform(0.4, 0.8), seed=k)
    # mop blending two wet bands: zigzag across the seam, then long light strokes
    c.stroke([(40, 330), (520, 330)], P.CADMIUM_YELLOW, amount=30, brush=B.FLAT_HOG.sized(60))
    c.stroke([(40, 390), (520, 390)], P.CADMIUM_RED, amount=30, brush=B.FLAT_HOG.sized(60))
    mop = B.MOP.sized(40)
    zig = [(50 + 9 * j, 335 if j % 2 == 0 else 385) for j in range(52)]
    c.stroke(zig, None, brush=mop, pressure=0.35)
    c.stroke([(x + 4, y) for x, y in zig[::-1]], None, brush=mop, pressure=0.3)
    for y in (345, 360, 375):
        c.stroke([(40, y), (520, y)], None, brush=mop, pressure=0.25)
    # palette knife: slabs of thick paint
    for k, col in enumerate([P.CERULEAN_BLUE, P.TITANIUM_WHITE, P.CADMIUM_ORANGE]):
        c.stroke([(600 + 30 * k, 300 + 30 * k), (760 + 30 * k, 290 + 30 * k)], col, amount=25,
                 brush=B.PALETTE_KNIFE, pressure=0.6, angle=15, seed=k)
    # rigger: long thin branches
    for k in range(6):
        c.stroke([(900, 520), (920 + 30 * k, 420 - 10 * k), (900 + 60 * k, 300 + 8 * k)], P.BURNT_UMBER,
                 amount=200, brush=B.RIGGER, pressure=[0.6, 0.4, 0.1], seed=k)
    # angled flat cutting a crisp corner; dagger strokes tapering at both ends
    c.stroke([(40, 470), (240, 470), (240, 540)], P.ULTRAMARINE_BLUE, amount=20, brush=B.ANGLED)
    for k in range(4):
        c.stroke([(320 + 60 * k, 540), (360 + 60 * k, 450)], P.ALIZARIN_CRIMSON, amount=8, brush=B.DAGGER,
                 pressure=[0.2, 0.8, 0.1], seed=k)
    save(c, "brush_uses.png")


# ---------------------------------------------------------------------------
# 8. Surfaces: the same marks on every support
# ---------------------------------------------------------------------------
def surface_sampler():
    names = ["CANVAS", "LINEN_FINE", "LINEN_COARSE", "COTTON_DUCK", "CANVAS_BOARD", "OIL_PAPER",
             "GESSO_PANEL", "COPPER", "toned"]
    dark = Paint.mix((P.BURNT_UMBER, 1), (P.IVORY_BLACK, 0.5))
    tiles = []
    for k, name in enumerate(names):
        surf = S.LINEN_FINE.with_(toning="#c49a6c", name="Fine linen, toned") if name == "toned" else getattr(S, name)
        c = Canvas(340, 340, surface=surf, seed=k)
        # a loaded hog flat at normal pressure
        c.stroke([(20, 50), (320, 44)], P.CADMIUM_RED, amount=14, brush=B.FLAT_HOG, seed=1)
        # the same brush, lightly dragged: tooth breaks it up (scumbling)
        c.stroke([(20, 100), (320, 94)], P.ULTRAMARINE_BLUE, amount=8, brush=B.FLAT_HOG.sized(30), pressure=0.2,
                 seed=2)
        # a dark: sinks in (goes matte and lighter) on absorbent grounds
        c.stroke([(20, 150), (320, 146)], dark, amount=30, brush=B.FLAT_SABLE.sized(34), seed=3)
        # a wet patch wiped back with a rag: clean on slick surfaces, stained on absorbent ones
        for y in (205, 230, 255):
            c.stroke([(20, y), (320, y)], P.YELLOW_OCHRE, amount=14, brush=B.FLAT_HOG.sized(28), seed=4 + y)
        c.stroke([(60, 190), (110, 270)], brush=B.RAG.sized(30), pressure=0.6)
        c.stroke([(200, 190), (250, 270)], brush=B.RAG.sized(30), pressure=0.6)
        c.stroke([(210, 270), (260, 190)], brush=B.RAG.sized(30), pressure=0.6)
        # a palette-knife slab of white
        c.stroke([(40, 305), (300, 300)], P.TITANIUM_WHITE, amount=25, brush=B.PALETTE_KNIFE, seed=5)
        tiles.append(np.pad(c.render(), ((6, 6), (6, 6), (0, 0)), constant_values=1.0))
        print("  ", c.surface.name)
    grid = np.vstack([np.hstack(tiles[r * 3:r * 3 + 3]) for r in range(3)])
    path = os.path.join(OUT, "surfaces.png")
    write_png(path, np.round(grid * 255).astype(np.uint8))
    print("wrote", path)


# ---------------------------------------------------------------------------
# 9. Loading a brush with several paints
# ---------------------------------------------------------------------------
def multi_loading():
    c = Canvas(1300, 760, seed=5)
    W, R, Y, GR = P.TITANIUM_WHITE, P.CADMIUM_RED, P.CADMIUM_YELLOW, P.SAP_GREEN
    # double-loaded soft flat: a ribbon with a light edge
    br = Brush(B.FLAT_SABLE.sized(34)).load(R, 10).load(W, 4, across="left")
    c.stroke([(40, 90), (140, 40), (240, 120), (340, 60)], brush=br, seed=1)
    # the same load, twisted half a turn along the stroke: the colours change sides (a folded ribbon)
    br = Brush(B.FLAT_SABLE.sized(34)).load(R, 10).load(W, 4, across="left")
    c.stroke([(400, 80), (520, 60), (640, 100)], brush=br, angle=lambda u: 180 * u, seed=2)
    # one-stroke petals: a double-loaded flat pulled out from the centre
    for k in range(7):
        a = k * 2 * np.pi / 7
        cx, cy = 820, 90
        br = Brush(B.FLAT_SABLE.sized(30)).load(P.QUINACRIDONE_MAGENTA, 3).load(W, 2, across="right")
        pts = [(cx + 8 * np.cos(a), cy + 8 * np.sin(a)), (cx + 40 * np.cos(a + 0.25), cy + 40 * np.sin(a + 0.25)),
               (cx + 62 * np.cos(a), cy + 62 * np.sin(a))]
        c.stroke(pts, brush=br, pressure=[0.6, 0.7, 0.2], seed=10 + k)
    # tip loading: a green round touched into yellow; each leaf starts yellow and turns green
    for k in range(6):
        br = Brush(B.ROUND_SABLE.sized(16)).load(GR, 5).load(Y, 0.6, depth="tip")
        x0 = 960 + 50 * k
        c.stroke([(x0, 140), (x0 + 20, 90), (x0 + 10, 40)], brush=br, pressure=[0.8, 0.6, 0.15], seed=20 + k)
    # triple-loaded wide flat: a sky gradient in a single stroke
    br = (Brush(B.FLAT_HOG.sized(70)).load(P.COBALT_BLUE, 6, across=(0.0, 0.34)).load(W, 6, across=(0.33, 0.67))
          .load(P.CADMIUM_ORANGE, 6, across=(0.66, 1.0)).work(0.5))
    c.stroke([(40, 230), (360, 220), (640, 240)], brush=br, seed=30)
    # paints picked up together, partly mixed: mixing 0.15, 0.45, 1.0
    for k, mix in enumerate((0.15, 0.45, 1.0)):
        br = Brush(B.FILBERT_HOG.sized(40)).load_mix([(R, 1), (Y, 1), (W, 1)], 10, mixing=mix, seed=40 + k)
        y = 320 + 70 * k
        c.stroke([(40, y), (330, y - 15), (620, y + 5)], brush=br, seed=40 + k)
    # how long two paints stay apart on the brush: blending 0.05, 0.4, 1.0
    for k, blend in enumerate((0.05, 0.4, 1.0)):
        br = Brush(B.FLAT_HOG.sized(36).with_(blending=blend)).load(R, 14, across="left").load(Y, 14, across="right")
        y = 300 + 70 * k
        c.stroke([(700, y), (1270, y)], brush=br, pressure=0.7, seed=50 + k)
    # stroking the brush on the palette first: 0, 2 and 6 passes
    for k, passes in enumerate((0, 2, 6)):
        br = Brush(B.FLAT_SABLE.sized(36)).load(P.VIRIDIAN, 8, across="left").load(Y, 8, across="right").work(passes)
        y = 540 + 60 * k
        c.stroke([(40, y), (330, y)], brush=br, seed=60 + k)
    # a double-loaded hog fan flicked upward: light and dark blades in every flick
    rng = np.random.default_rng(7)
    for k in range(16):
        br = Brush(B.FAN_HOG.sized(34)).load(GR, 2).load(P.LEMON_YELLOW, 1.2, across="left")
        x = 420 + 26 * k + rng.normal(0, 5)
        c.stroke([(x, 740), (x + rng.normal(0, 14), 560)], brush=br, pressure=[0.6, 0.1], seed=70 + k)
    # a double-loaded palette knife: a two-tone slab
    br = Brush(B.PALETTE_KNIFE.sized(60)).load(P.ULTRAMARINE_BLUE, 12).load(W, 10, across="right")
    c.stroke([(900, 560), (1250, 545)], brush=br, pressure=0.6, seed=80)
    # a dirty brush: only its lower side crosses wet red, so only that side carries red on
    c.stroke([(990, 690), (1010, 690)], R, amount=20, brush=B.FLAT_HOG.sized(30), angle=90, seed=90)
    c.stroke([(900, 672), (1250, 672)], W, amount=14, brush=B.FLAT_HOG.sized(36), seed=91)
    save(c, "multi_loading.png")


if __name__ == "__main__":
    for fn in (mixing_chart, strokes, blending, glazing, landscape, brush_rack, brush_uses, surface_sampler,
               multi_loading):
        t0 = time.time()
        fn()
        print(f"  {fn.__name__}: {time.time() - t0:.1f}s")
