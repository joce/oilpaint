"""A study after Rembrandt's Self-Portrait of 1659 (National Gallery of Art,
Washington; public domain), painted the way he worked:

* a canvas with a warm brown ground;
* the dead colouring: the whole picture laid in thinly in browns, light to
  dark, and left to dry;
* the background, the coat and the beret brushed in loosely over it in thin,
  fluid paint, softened with a big clean mop while wet, and left to dry;
* the head worked up wet-into-wet from his palette (lead white, ochres,
  vermilion, red lake, umbers, bone black, a little smalt for the cool half
  tones), the lights piled up thick with stiff lead white;
* the grey hair in long thin strokes following the swirl of the curls, its
  dark curls scratched into the wet paint with the handle of the brush and
  its light ones drawn on with a fine liner;
* thin transparent glazes deepening the shadows once it was dry; then the
  last touches: the glints of the gold chain, the lights in the eyes.

The reproduction in ``examples/reference`` is used the way a copyist uses the
original: to see where things are and how light or dark, warm or cool they
are; a close-up of the head shows the curls of the hair, which are too fine
to survive in the whole picture at that size. Each stroke's paint is mixed from the old palette to match
(:func:`oilpaint.palette.match`), and each stroke follows the forms
(:class:`oilpaint.Reference`).

    python examples/rembrandt_self_portrait.py [output.png]

It takes about 20 minutes; ``REMBRANDT_SCALE=0.5`` paints a half-size version in
about half the time.
"""

import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from oilpaint import Brush, Canvas, Paint, Reference, brushes, palette as P, spectral as sp, surfaces  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
S = float(os.environ.get("REMBRANDT_SCALE", "1"))      # < 1: a quicker, smaller preview
W, H = round(1000 * S), round(1296 * S)
F = W / 960                     # the outlines below are drawn on a 960-pixel-wide reproduction
ref = Reference(os.path.join(HERE, "reference", "rembrandt_self_portrait_1659.jpg"), W, H)
# a closer look at the head, for the finest things in it (the curls of the hair)
HEAD_BOX = (300, 100, 820, 620)               # where it is, on the 960-pixel-wide reproduction
HEAD_FILE = os.path.join(HERE, "reference", "rembrandt_self_portrait_1659_head.jpg")
rng = np.random.default_rng(1659)
GROUND = surfaces.LINEN_FINE.with_(toning="#6b4f37", toning_strength=1.0, name="linen, warm brown ground")
canvas = Canvas(W, H, surface=GROUND, seed=1659)
N_STROKES = [0]
# How the finished canvas is lit: enough relief to catch the broken lights and
# the brushwork, not so much that the thin darks look embossed. (A smaller
# canvas needs a little more: its marks are narrower.)
RELIEF, GLOSS = 0.3 + 0.3 / S, 0.08


def stage(name):
    print(f"[{time.strftime('%H:%M:%S')}] {name}  (strokes so far: {N_STROKES[0]})", flush=True)


# ===========================================================================
# Where things are (outlines on the 960-wide reproduction)
# ===========================================================================
FACE = [(425, 272), (460, 258), (520, 252), (580, 255), (625, 265), (645, 290), (652, 330), (655, 370), (650, 410),
        (640, 440), (615, 462), (590, 480), (565, 505), (545, 525), (515, 535), (485, 525), (462, 500), (445, 470),
        (430, 440), (420, 400), (418, 350), (420, 300)]
HAIR_RIGHT = [(640, 262), (690, 250), (740, 262), (780, 290), (802, 330), (802, 390), (782, 430), (742, 448),
              (700, 442), (662, 430), (654, 380), (650, 320)]
HAIR_LEFT = [(338, 300), (380, 292), (422, 300), (420, 380), (430, 440), (442, 472), (402, 472), (370, 442),
             (345, 400)]
BERET = [(330, 250), (350, 190), (400, 145), (470, 120), (560, 115), (650, 130), (730, 175), (775, 240), (770, 285),
         (700, 262), (640, 240), (560, 228), (480, 232), (420, 255), (360, 300), (335, 320)]
COLLAR = [(545, 528), (600, 478), (650, 448), (700, 424), (722, 440), (692, 470), (640, 502), (600, 545), (570, 585),
          (540, 605), (520, 585)]
HANDS = [(70, 990), (170, 960), (280, 975), (330, 1040), (300, 1120), (230, 1160), (110, 1160), (60, 1100)]
FUR = [(140, 780), (250, 740), (330, 800), (390, 900), (400, 990), (330, 1030), (260, 975), (170, 960), (120, 880)]
FIGURE = [(430, 470), (560, 515), (690, 440), (790, 520), (870, 620), (930, 740), (960, 850), (960, 1244), (0, 1244),
          (0, 990), (60, 930), (130, 790), (240, 730), (340, 700), (400, 610)]          # the body in its coat
# the light caught in each eye: up and to the left of the pupil (the light comes
# from the left); dull in the far eye, clearer in the near one
CATCHLIGHTS = [(457, 346), (559, 345)]
EYES = [(430, 330, 600, 372)]                          # boxes (x0, y0, x1, y1) where the features are
NOSE_MOUTH = [(475, 375, 550, 452), (465, 445, 585, 495)]
CHAIN = (470, 205, 610, 240)                           # the box round the gold chain on the hat band


def mask(*polys, grow=0.0):
    """Boolean canvas mask of polygons (960-scale), grown by ``grow`` (960-scale) px."""
    from PIL import Image, ImageDraw, ImageFilter
    im = Image.new("L", (W, H), 0)
    d = ImageDraw.Draw(im)
    for poly in polys:
        d.polygon([(x * F, y * F) for x, y in poly], fill=255)
    if grow > 0:
        im = im.filter(ImageFilter.MaxFilter(int(2 * grow * F) // 2 * 2 + 1))
    return np.asarray(im) > 127


M_FACE = mask(FACE)
M_HAIR = mask(HAIR_RIGHT, HAIR_LEFT)
M_HAIR_ONLY = M_HAIR & ~M_FACE
M_HEAD = mask(FACE, HAIR_RIGHT, HAIR_LEFT, COLLAR, grow=10)
M_HANDS = mask(HANDS, FUR, grow=8)
M_HAND_FLESH = mask(HANDS)
M_FIGURE = mask(FIGURE)
EVERYWHERE = np.ones((H, W), bool)

# ===========================================================================
# His palette, and piles mixed from it
# ===========================================================================
FLESH_PAINTS = [P.LEAD_WHITE, P.YELLOW_OCHRE, P.RED_OCHRE, P.VERMILION, P.RED_LAKE, P.RAW_UMBER, P.BONE_BLACK,
                P.SMALT]
EARTH_PAINTS = [P.LEAD_WHITE, P.YELLOW_OCHRE, P.RED_OCHRE, P.RAW_UMBER, P.BURNT_UMBER, P.CASSEL_EARTH, P.BONE_BLACK,
                P.SMALT]
GREY_PAINTS = [P.LEAD_WHITE, P.BONE_BLACK, P.RAW_UMBER, P.BURNT_UMBER, P.RED_OCHRE, P.YELLOW_OCHRE, P.SMALT]
GLAZE_PAINTS = [P.CASSEL_EARTH, P.BONE_BLACK, P.RED_LAKE, P.RAW_UMBER]


class Piles:
    """Paint mixed on the palette from a few pigments: one pile for each
    colour asked for (to within ``step`` delta E), mixed the first time it is
    needed and reused after that. ``medium`` thins it (parts of oil per part
    of paint); ``body`` makes it stiffer or more fluid."""

    def __init__(self, paints, over, thickness=1.0, step=3.0, medium=0.0, body=None):
        self.paints, self.over, self.thickness, self.step = paints, over, thickness, step
        self.medium, self.body, self.piles = medium, body, {}

    def __call__(self, rgb):
        lab = sp.srgb_to_lab(np.clip(rgb, 0, 1))
        key = tuple(np.round(lab / self.step).astype(int))
        pile = self.piles.get(key)
        if pile is None:
            pile = P.match(rgb, self.paints, thickness=self.thickness, over=self.over).paint
            if self.medium > 0:
                pile = Paint.mix((pile, 1.0), (P.MEDIUM, self.medium))
            if self.body is not None:
                pile = pile.with_body(self.body)
            self.piles[key] = pile
        return pile


# ===========================================================================
# Laying in strokes where the picture still differs from the reference
# ===========================================================================
TOE = 10.0     # L* below which a photograph's darks are deeper than paint can go


def seen(rgb, toe=TOE):
    """The subject's colour as paint can show it. The deepest darks of a
    photograph are blacker than any paint (the darkest mix on this palette is
    about L* 10), so, like a copyist, lift them gently, keeping them apart,
    and leave the rest alone: L* -> sqrt(L*^2 + toe^2). (In the head a smaller
    toe: its few deep accents, the pupils and nostrils, go as dark as paint can.)"""
    lin = sp.srgb_to_linear(np.clip(rgb, 0.0, 1.0))
    Y = np.maximum(lin @ sp.LUMINANCE, 1e-9)
    d = 6.0 / 29.0
    f = np.where(Y > d ** 3, np.cbrt(Y), Y / (3 * d * d) + 4.0 / 29.0)
    L2 = np.sqrt((116.0 * f - 16.0) ** 2 + np.asarray(toe) ** 2)
    f2 = (L2 + 16.0) / 116.0
    Y2 = np.where(f2 > d, f2 ** 3, 3 * d * d * (f2 - 4.0 / 29.0))
    return sp.linear_to_srgb(lin * (Y2 / Y)[..., None])


_TOE_MAP = []


def toe_map():
    if not _TOE_MAP:
        from oilpaint.reference import _blur
        head = _blur(mask(FACE, COLLAR).astype(np.float32), 8 * S)
        _TOE_MAP.append(TOE - 5.0 * head)
    return _TOE_MAP[0]


def target_map(scale, color_fn=None):
    """The reference (or ``color_fn`` of it) softened to ``scale``, as an image."""
    ys, xs = np.mgrid[0:H, 0:W]
    rgb = seen(ref.color(xs, ys, scale), toe_map()) if color_fn is None else color_fn(xs, ys, scale)
    return np.asarray(rgb, np.float64)


def box_mean(a, r):
    """Mean over a (2r+1)-pixel square around each pixel."""
    from oilpaint.reference import _blur
    return _blur(a.astype(np.float32), max(r / 2.0, 0.5))


def paint_layer(size, where, tool, piles, threshold=6.0, length=3.0, pressure=0.5, load=1.0, scale=None,
                tolerance=0.1, bend=0.6, color_fn=None, spacing=0.75, wander=0.0, pick=None, straight=None,
                correct=0.6):
    """One pass with one brush size, the way a painter refines a picture:
    look where the canvas still differs from the subject by more than
    ``threshold`` (delta E) at the scale of this brush, and put a stroke there
    that follows the forms, in the colour of the subject there. ``piles``
    gives the paint for a colour (or ``pick(x, y)`` chooses the piles per
    stroke). Big brushes make straight strokes (``straight``): broad and quick.
    Like a painter judging by eye, each stroke asks for a colour pushed past
    the subject by ``correct`` times what the canvas still lacks there (thin
    paint takes on the colour beneath it). Returns the number of strokes."""
    straight = size >= 20 if straight is None else straight
    size = max(size * S, 2.0)
    scale = (scale * S if scale is not None else max(size / 3.0, 1.0))
    target = target_map(scale, color_fn)
    now = box_mean(canvas.render(lighting=False), size / 2.0).astype(np.float64)
    err = np.linalg.norm(sp.srgb_to_lab(now) - sp.srgb_to_lab(target), axis=-1)
    err = box_mean(err, size / 2.0)
    lacking = target - now
    lacking *= np.minimum(1.0, 0.15 / np.maximum(np.linalg.norm(lacking, axis=-1, keepdims=True), 1e-9))
    step = size * spacing
    gy, gx = np.mgrid[step / 2:H:step, step / 2:W:step]
    px = (gx + rng.uniform(-0.45, 0.45, gx.shape) * step).ravel()
    py = (gy + rng.uniform(-0.45, 0.45, gy.shape) * step).ravel()
    px, py = np.clip(px, 0, W - 1), np.clip(py, 0, H - 1)
    ix, iy = px.astype(int), py.astype(int)
    keep = where[iy, ix] & (err[iy, ix] > threshold)
    order = rng.permutation(np.flatnonzero(keep))
    for k in order:
        x, y = px[k], py[k]
        path = ref.trace(x, y, length * size, scale=max(scale, 1.5), tolerance=tolerance, bend=bend)
        if len(path) < 2 or np.linalg.norm(path[-1] - path[0]) < 0.6 * size:
            a, _ = ref.direction(x, y, max(scale, 1.5))
            a = float(a) + rng.normal(0, 0.3)
            d = np.array([np.cos(a), np.sin(a)]) * size * 0.45
            path = np.array([[x, y] - d, [x, y] + d])
        if straight:
            path = path[[0, -1]]
        if wander > 0:         # a loose hand: the stroke doesn't quite follow the form
            a = rng.normal(0, wander)
            c, s = np.cos(a), np.sin(a)
            path = (path - [x, y]) @ np.array([[c, s], [-s, c]]) + [x, y]
        yi, xi = int(np.clip(y, 0, H - 1)), int(np.clip(x, 0, W - 1))
        rgb = np.clip(target[yi, xi] + correct * lacking[yi, xi], 0.0, 1.0)
        paint = (pick(x, y) if pick is not None else piles)(rgb)
        run = float(np.linalg.norm(np.diff(path, axis=0), axis=1).sum())
        brush = Brush(tool.sized(size)).load(paint, load * (run / size + 1.0))
        canvas.stroke([tuple(p) for p in path], brush=brush, pressure=pressure)
        N_STROKES[0] += 1
    return len(order)


# ===========================================================================
# 1. The dead colouring
# ===========================================================================
DEAD_DARK, DEAD_MID, DEAD_LIGHT = (np.array(sp.parse_color(c)) for c in ("#1c130c", "#6a4e36", "#cbb38e"))


def dead_tone(xs, ys, scale):
    """The picture as a brown monochrome: the lightness of the subject, in umber."""
    v = np.clip((np.asarray(ref.value(xs, ys, scale)) - 0.05) / 0.72, 0.0, 1.0)[..., None]
    lo = DEAD_DARK + (DEAD_MID - DEAD_DARK) * np.clip(v * 2, 0, 1)
    return lo + (DEAD_LIGHT - DEAD_MID) * np.clip(v * 2 - 1, 0, 1)


def dead_colouring():
    stage("dead colouring")
    piles = Piles([P.LEAD_WHITE, P.RAW_UMBER, P.BURNT_UMBER, P.BONE_BLACK, P.YELLOW_OCHRE], over="#6b4f37",
                  thickness=0.8, step=4.0, medium=0.4, body=0.3)
    edges(piles, dead_tone)
    for size, where, thr in ((70, EVERYWHERE, 0.0), (40, EVERYWHERE, 8.0), (22, M_HEAD | M_HANDS, 8.0)):
        n = paint_layer(size, where, brushes.FLAT_HOG, piles, threshold=thr, length=3.0, pressure=0.6, load=0.55,
                        color_fn=dead_tone, tolerance=0.12, wander=0.25)
        print(f"    {size}px: {n} strokes", flush=True)
    canvas.dry()


def edges(piles, color_fn):
    """Long strokes along the edges of the canvas, so the lay-in reaches them."""
    size = 60 * S
    for (x0, y0, x1, y1) in ((0, 0, W, 0), (0, H, W, H), (0, 0, 0, H), (W, 0, W, H)):
        n = int(max(abs(x1 - x0), abs(y1 - y0)) / (2.5 * size)) + 1
        for k in range(n):
            a, b = k / n, (k + 1) / n
            p0 = (x0 + (x1 - x0) * a, y0 + (y1 - y0) * a)
            p1 = (x0 + (x1 - x0) * b, y0 + (y1 - y0) * b)
            mx, my = min(max((p0[0] + p1[0]) / 2, 1), W - 2), min(max((p0[1] + p1[1]) / 2, 1), H - 2)
            paint = piles(color_fn(np.array([mx]), np.array([my]), 6.0 * S)[0])
            brush = Brush(brushes.FLAT_HOG.sized(size)).load(paint, 0.7 * (2.5 + 1.0))
            canvas.stroke([p0, p1], brush=brush, pressure=0.6)
            N_STROKES[0] += 1


# ===========================================================================
# 2. Background, coat and beret
# ===========================================================================
def lay_in():
    stage("background, coat and beret")
    piles = Piles(EARTH_PAINTS, over="#5e4430", thickness=0.9, step=3.0, medium=0.25, body=0.3)
    for size, where, thr, tool in ((64, EVERYWHERE, 4.0, brushes.FLAT_HOG), (40, EVERYWHERE, 5.0, brushes.FLAT_HOG),
                                   (24, EVERYWHERE, 6.0, brushes.FILBERT_HOG),
                                   (14, M_HEAD | M_HANDS | M_FIGURE, 5.0, brushes.FILBERT_HOG)):
        n = paint_layer(size, where, tool, piles, threshold=thr, length=3.0, pressure=0.55,
                        load=0.65 if size > 30 else 0.8, tolerance=0.1, wander=0.35 if size > 30 else 0.1)
        print(f"    {size}px: {n} strokes", flush=True)
    soften(~(mask(BERET, FIGURE, grow=8) | M_HEAD | M_HANDS))           # the background only
    canvas.dry()


def soften(where):
    """Go over the wet background with a big, soft, clean brush, barely
    touching, so the strokes melt into one another (the brush is wiped now
    and then)."""
    size = 70 * S
    n = 0
    for sweep in range(2):
        step = size * 1.4
        for y in np.arange(step / 2, H, step):
            for x in np.arange(step / 2, W, step * 2.2):
                x0, y0 = x + rng.uniform(-0.4, 0.4) * step, y + rng.uniform(-0.4, 0.4) * step
                if not where[int(np.clip(y0, 0, H - 1)), int(np.clip(x0, 0, W - 1))]:
                    continue
                a = (0.35 if sweep == 0 else -0.6) + rng.normal(0, 0.25)
                d = np.array([np.cos(a), np.sin(a)]) * size * 1.6
                pts = [(x0 - d[0], y0 - d[1]), (x0 + d[0] * 0.1, y0 + d[1] * 0.1 + rng.normal(0, 3 * S)),
                       (x0 + d[0], y0 + d[1])]
                canvas.stroke(pts, brush=Brush(brushes.MOP.sized(size)), pressure=[0.0, 0.14, 0.0])
                n += 1
    N_STROKES[0] += n
    print(f"    softened with {n} passes of a mop", flush=True)


# ===========================================================================
# 3. The head, wet into wet
# ===========================================================================
FLESH = Piles(FLESH_PAINTS, over="#8a6a4c", thickness=1.4, step=2.5)
HAND_FLESH = Piles(FLESH_PAINTS, over="#3a2a1e", thickness=1.0, step=2.5, body=0.35)    # thin and soft, in shadow
EARTHS = Piles(EARTH_PAINTS, over="#3a2a1e", thickness=1.2, step=2.5, body=0.45)
GREYS = Piles(GREY_PAINTS, over="#3a2a1e", thickness=2.5, step=2.5, body=0.4)        # soft, fluid hair


def piles_at(x, y):
    xi, yi = int(np.clip(x, 0, W - 1)), int(np.clip(y, 0, H - 1))
    if M_FACE[yi, xi]:
        return FLESH
    if M_HAND_FLESH[yi, xi]:
        return HAND_FLESH
    if M_HAIR[yi, xi]:
        return GREYS
    return EARTHS


def head():
    """Big to small, each brush only where the picture still needs it. The
    strokes follow the forms: across the brow, down the nose, round the eyes."""
    stage("the head")
    for size, thr, tool in ((16, 5.0, brushes.FILBERT_HOG), (10, 4.5, brushes.FILBERT_HOG),
                            (7, 4.5, brushes.ROUND_HOG), (5, 4.5, brushes.ROUND_HOG),
                            (4, 5.0, brushes.ROUND_SABLE), (3, 5.0, brushes.ROUND_SABLE)):
        where = (M_HEAD | M_HANDS if size >= 7 else M_HEAD) & ~M_HAIR_ONLY      # the hair has its own stage
        n = paint_layer(size, where, tool, None, pick=piles_at, threshold=thr, length=2.5 if size > 4 else 2.0,
                        pressure=0.5, load=1.1, tolerance=0.08, bend=0.7)
        print(f"    {size}px: {n} strokes", flush=True)


def impasto():
    """The lights, piled up with stiff lead white: short, loaded strokes laid
    on with a light touch so they stand up from the surface and break."""
    stage("impasto lights")
    lights = Piles(FLESH_PAINTS, over="#a07858", thickness=2.5, step=2.5, body=0.9)
    hot = M_FACE & (np.asarray(ref.value(*np.mgrid[0:H, 0:W][::-1], 2.0 * S)) > 0.6)
    for size in (10, 7, 5):
        n = paint_layer(size, hot, brushes.FILBERT_HOG, lights, threshold=0.0, length=1.8, pressure=0.22,
                        load=2.6, scale=1.5, tolerance=0.06, spacing=0.9, bend=0.8)
        print(f"    {size}px: {n} strokes", flush=True)


def hair():
    """The grey hair: a soft mass brushed in along the swirl of the curls with
    soft brushes and fluid paint, then the curls themselves, where the
    subject has them: the dark ones scratched into the wet paint with the
    handle of the brush, the light ones drawn on with a fine liner."""
    stage("hair and the brush handle")
    # the tone of it first, then long thin strokes that follow the swirl of the
    # curls everywhere, each in the colour of the hair just there, so the
    # light and dark strands come out of the brushwork itself
    n = paint_layer(8, M_HAIR_ONLY, brushes.FILBERT_SABLE, GREYS, threshold=4.0, length=3.0, pressure=0.4, load=1.0,
                    tolerance=0.1, bend=0.9, straight=False)
    print(f"    8px: {n} strokes", flush=True)
    for size in (5, 3):         # (their colours from the close-up: the silver of single strands)
        n = paint_layer(size, M_HAIR_ONLY, brushes.ROUND_SABLE, GREYS, threshold=0.0, length=6.0, pressure=0.35,
                        load=1.0, scale=1.0, tolerance=0.16, bend=1.0, spacing=1.1, straight=False, correct=0.3,
                        color_fn=close_up_color)
        print(f"    strands {size}px: {n} strokes", flush=True)
    light, dark = curl_lines()
    curls = Piles([P.LEAD_WHITE, P.YELLOW_OCHRE, P.VERMILION, P.RAW_UMBER, P.BONE_BLACK, P.SMALT],
                  over="#4a4038", thickness=1.0, step=3.0, body=0.8)
    for path, rgb in dark:
        canvas.stroke([tuple(q) for q in path], brush=brushes.BRUSH_HANDLE.sized(max(1.7 * S, 1.2)), pressure=0.6)
    for path, rgb in light:
        run = float(np.linalg.norm(np.diff(path, axis=0), axis=1).sum())
        w = max(1.2 * S, 1.0)
        # a silvery line: lighter than the photograph shows it (a thin line is
        # softened by the camera), and a little less yellow
        silver = rgb + 0.35 * (1.0 - rgb)
        silver = 0.75 * silver + 0.25 * silver.mean()
        brush = Brush(brushes.LINER.sized(w)).load(curls(silver), 1.5 * run / w + 2.0)
        canvas.stroke([tuple(q) for q in path], brush=brush, pressure=0.35)
    N_STROKES[0] += len(light) + len(dark)
    print(f"    curls: {len(dark)} scratched, {len(light)} drawn", flush=True)


_CLOSE_UP = []


def close_up():
    """The close-up of the head, as a Reference, and its pixels per reproduction pixel."""
    if not _CLOSE_UP:
        from PIL import Image
        img = Image.open(HEAD_FILE).convert("RGB")
        _CLOSE_UP.append((Reference(img, img.width, img.height), img.width / (HEAD_BOX[2] - HEAD_BOX[0])))
    return _CLOSE_UP[0]


def close_up_color(xs, ys, scale):
    """The subject's colour at canvas points, seen in the close-up of the head."""
    href, k = close_up()
    xs, ys = np.asarray(xs, np.float64), np.asarray(ys, np.float64)
    rgb = href.color((xs / F - HEAD_BOX[0]) * k, (ys / F - HEAD_BOX[1]) * k, scale * k / F)
    return seen(rgb, toe_map()[np.clip(ys.astype(int), 0, H - 1), np.clip(xs.astype(int), 0, W - 1)])


def curl_lines(threshold=0.016, min_len=7):
    """The curls of the hair, read off the close-up of the head: thin lines
    lighter or darker than the hair around them, followed into stroke paths
    (canvas coordinates), each with its colour. Returns (light, dark)."""
    from PIL import Image, ImageDraw, ImageFilter
    href, k = close_up()                          # k: close-up pixels per reproduction pixel
    n = href.width
    m = Image.new("L", (n, n), 0)
    d = ImageDraw.Draw(m)
    for poly in (HAIR_RIGHT, HAIR_LEFT):
        d.polygon([((x - HEAD_BOX[0]) * k, (y - HEAD_BOX[1]) * k) for x, y in poly], fill=255)
    hair = np.asarray(m.filter(ImageFilter.MaxFilter(int(20 * k) // 2 * 2 + 1))) > 127
    ys, xs = np.mgrid[0:n, 0:n]
    diff = np.asarray(href.value(xs, ys, 0.8)) - np.asarray(href.value(xs, ys, 4.0))

    def to_canvas(path):
        pts = np.stack([(path[:, 1] / k + HEAD_BOX[0]) * F, (path[:, 0] / k + HEAD_BOX[1]) * F], axis=1)
        run = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))]
        idx = np.unique(np.r_[np.searchsorted(run, np.arange(0.0, run[-1], 2.0 * S)), len(pts) - 1])
        return pts[idx]                                # a point every 2 px or so

    out = []
    for sign in (1.0, -1.0):
        found = []
        for path in _chains(hair & (sign * diff > threshold), sign * diff, min_len):
            rgb = np.asarray(href.color(path[:, 1], path[:, 0], 0.7), np.float64).mean(axis=0)
            found.append((to_canvas(path), rgb))
        out.append(found)
    return out[0], out[1]


def _chains(line, strength, min_len, reach=3.0, eat=2.0):
    """Follow the pixels of thin lines (a boolean mask) into paths, strongest
    first: from each start, step to the strongest free line pixel nearby that
    keeps going the same way, both ways."""
    H_, W_ = line.shape
    free = line.copy()
    ys, xs = np.nonzero(line)
    r = int(np.ceil(reach))
    oy, ox = np.mgrid[-r:r + 1, -r:r + 1]
    ring = (np.hypot(oy, ox) <= reach) & (np.hypot(oy, ox) >= 1.0)
    oy, ox = oy[ring], ox[ring]
    e = int(np.ceil(eat))
    ey, ex = np.mgrid[-e:e + 1, -e:e + 1]
    disc = np.hypot(ey, ex) <= eat
    ey, ex = ey[disc], ex[disc]

    def take(y, x):
        free[np.clip(y + ey, 0, H_ - 1), np.clip(x + ex, 0, W_ - 1)] = False

    paths = []
    for i in np.argsort(-strength[ys, xs]):
        y, x = ys[i], xs[i]
        if not free[y, x]:
            continue
        take(y, x)
        halves = []
        for first in (True, False):
            pts, cy, cx, d = [], y, x, None
            if not first and halves[0]:              # set off the other way
                d = -np.array([halves[0][0][0] - y, halves[0][0][1] - x], float)
                d /= max(np.hypot(*d), 1e-9)
            while True:
                ny, nx = cy + oy, cx + ox
                ok = (ny >= 0) & (ny < H_) & (nx >= 0) & (nx < W_)
                ny, nx, dy, dx = ny[ok], nx[ok], oy[ok], ox[ok]
                cand = free[ny, nx]
                if not cand.any():
                    break
                ny, nx, dy, dx = ny[cand], nx[cand], dy[cand], dx[cand]
                score = strength[ny, nx].astype(np.float64)
                if d is not None:
                    dot = (dy * d[0] + dx * d[1]) / np.hypot(dy, dx)
                    score = np.where(dot > 0.2, score * (0.5 + dot), -1.0)
                j = int(np.argmax(score))
                if score[j] < 0:
                    break
                d = np.array([dy[j], dx[j]], float) / np.hypot(dy[j], dx[j])
                cy, cx = ny[j], nx[j]
                take(cy, cx)
                pts.append((cy, cx))
            halves.append(pts)
        path = halves[1][::-1] + [(y, x)] + halves[0]
        if len(path) >= min_len:
            paths.append(np.array(path, float))
    return paths


def features():
    """Eyes, nostrils, the line of the mouth: the smallest brushes, short
    strokes, only where the picture still differs from the subject."""
    stage("features")
    zones = np.zeros((H, W), bool)
    for x0, y0, x1, y1 in EYES + NOSE_MOUTH:
        zones[int(y0 * F):int(y1 * F), int(x0 * F):int(x1 * F)] = True
    for size, thr in ((3, 4.0), (2, 4.0)):
        n = paint_layer(size, zones, brushes.ROUND_SABLE, None, pick=piles_at, threshold=thr, length=1.6,
                        pressure=0.45, load=1.2, scale=1.0, tolerance=0.06, bend=0.8, correct=0.8)
        print(f"    {size}px: {n} strokes", flush=True)


# ===========================================================================
# 4. Glazes and last touches
# ===========================================================================
def glazes():
    """Once dry: thin transparent browns over the shadows that came out too
    light, worked in with soft brushes."""
    stage("glazes")
    canvas.dry()
    glaze = Piles(GLAZE_PAINTS, over="#5e4430", thickness=0.4, step=4.0, medium=4.0)
    # Compare with the subject at fine detail: against a blurred one, the light
    # edge of a form next to a dark (the brow under the hat) would look too light.
    cur = sp.srgb_to_lab(canvas.render(lighting=False))
    tgt = sp.srgb_to_lab(target_map(1.5 * S))
    too_light = box_mean((cur[..., 0] - tgt[..., 0]) > 4.0, 3 * S) > 0.5
    dark = np.asarray(ref.value(*np.mgrid[0:H, 0:W][::-1], 1.5 * S)) < 0.42
    head = mask(FACE, HAIR_RIGHT, HAIR_LEFT, COLLAR, HANDS, FUR, grow=30)
    # the big darks of the coat and the background with big soft brushes, the
    # shadows of the head and hands with small ones
    for size, region in ((48, ~head), (24, ~head), (16, head), (8, head)):
        shadows = too_light & dark & region
        # only start where the whole brush stays in the shadow
        inside = box_mean(shadows.astype(np.float32), 0.6 * size * S) > 0.9
        n = paint_layer(size, inside, brushes.FILBERT_SABLE, glaze, threshold=4.0, length=2.5, pressure=0.3,
                        load=0.5, tolerance=0.08, straight=False)
        print(f"    {size}px: {n} strokes", flush=True)


def last_touches():
    stage("last touches")
    # the glints of the gold chain: little touches of stiff yellow where the
    # subject has them (yellow points lighter than what is around them)
    gold = Piles([P.LEAD_TIN_YELLOW, P.YELLOW_OCHRE, P.RAW_UMBER, P.LEAD_WHITE], over="#2a2018", thickness=1.5,
                 step=4.0, body=0.9)
    x0, y0, x1, y1 = (int(v * F) for v in CHAIN)
    ys, xs = np.mgrid[y0:y1, x0:x1]
    lab = sp.srgb_to_lab(np.asarray(ref.color(xs, ys, 0), np.float64))
    around = sp.srgb_to_lab(np.asarray(ref.color(xs, ys, 4 * S), np.float64))[..., 0]
    glints = (lab[..., 0] - around > 6) & (lab[..., 2] > 18) & (lab[..., 0] > 25)
    taken = set()
    for y, x in zip(ys[glints], xs[glints]):
        cell = (int(x / (3 * S)), int(y / (3 * S)))         # about one touch every 3 px
        if cell in taken:
            continue
        taken.add(cell)
        rgb = np.asarray(ref.color(x, y, 0.8 * S), np.float64)
        q = (x + rng.normal(0, 0.5 * S), y + rng.normal(0, 0.5 * S))
        canvas.stroke([q, (q[0] + rng.uniform(1, 3) * S, q[1] + rng.normal(0, 0.6) * S)], paint=gold(rgb),
                      amount=1.5, width=max(rng.uniform(1.6, 2.6) * S, 1.0), pressure=0.3, brush=brushes.ROUND_SABLE)
        N_STROKES[0] += 1
    for (x, y) in CATCHLIGHTS:                    # the light caught in each eye: a touch of greyed white
        oy, ox = np.mgrid[-2:3, -2:3]
        v = np.asarray(ref.value(x * F + ox, y * F + oy, 0))
        i = np.unravel_index(np.argmax(v), v.shape)
        rgb = np.asarray(ref.color(x * F + ox[i], y * F + oy[i], 0), np.float64)
        rgb = rgb + 0.25 * (1.0 - rgb)            # a little brighter than in the photograph: it is a point of light
        glint = P.match(rgb, [P.LEAD_WHITE, P.RAW_UMBER, P.BONE_BLACK, P.SMALT], over="#2a2018").paint
        canvas.stroke([(x * F, y * F)], paint=glint.with_body(0.9), amount=1.0, width=max(1.6 * S, 1.0), pressure=0.25,
                      brush=brushes.ROUND_SABLE)
        N_STROKES[0] += 1


STAGES = ["dead_colouring", "lay_in", "head", "impasto", "hair", "features", "glazes", "last_touches"]


def main(out):
    t0 = time.time()
    for name in STAGES:
        globals()[name]()
    canvas.to_pil(relief=RELIEF, gloss=GLOSS).save(out)
    print(f"wrote {out}: {N_STROKES[0]} strokes in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "output", "rembrandt_self_portrait.png"))
