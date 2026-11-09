"""Time the stroke engine on a fixed mix of marks, and fingerprint what it paints.

    python tools/bench.py                 # the timing table
    python tools/bench.py -k knife -n 9   # some scenes, best of 9
    python tools/bench.py --save DIR      # also keep each scene's canvas state
    python tools/bench.py --compare DIR   # ... and compare with a kept one

Every scene is painted with fixed seeds, so the same code always paints the
same canvas: the ``state`` column is a hash of everything the scene left on
the canvas (and on its brushes). An optimisation that leaves the paint alone
leaves those hashes alone; ``--compare`` says how far any array moved.

Each scene is set up (a canvas, perhaps some wet paint on it) outside the
clock and then timed several times over, each time from a fresh copy; the
table gives the best and the median wall time, the best CPU time, and the
best time per mark. The compiled engine is warmed up first.
"""

from __future__ import annotations

import argparse
import copy
import gc
import hashlib
import os
import statistics
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from oilpaint import Brush, Canvas, Paint, brushes
from oilpaint import palette as P

W, H = 800, 600

#: The arrays that make up what is on a canvas (see ``Canvas.__getstate__``).
STATE = (
    "top",
    "und",
    "top_a",
    "und_a",
    "top_id",
    "dry_height",
    "ground",
    "_tooth",
    "_floor",
    "_hollows",
    "_peaks_fine",
    "_peaks_wide",
    "glass_r",
    "glass_rb",
    "glass_t2",
)


def canvas_state(canvas, tools=()) -> dict[str, np.ndarray]:
    """Everything painting leaves behind: the canvas's arrays, and the paint
    on the brushes used."""
    state = {n: getattr(canvas, n) for n in STATE if hasattr(canvas, n)}
    for i, brush in enumerate(tools):
        state[f"brush{i}"] = np.stack([brush.tip, brush.reservoir, brush.heel])
    return state


def digest(state: dict[str, np.ndarray]) -> str:
    h = hashlib.sha256()
    for name in sorted(state):
        a = np.ascontiguousarray(state[name])
        h.update(f"{name}:{a.dtype}:{a.shape}".encode())
        h.update(a.tobytes())
    return h.hexdigest()


def compare(state: dict[str, np.ndarray], path: str) -> list[str]:
    """How ``state`` differs from the one kept at ``path`` (no lines: not at all)."""
    lines = []
    with np.load(path) as kept:
        for name in sorted(set(state) | set(kept.files)):
            if name not in state or name not in kept.files:
                lines.append(f"{name}: only in one of the two")
                continue
            a, b = state[name], kept[name]
            if a.shape != b.shape or a.dtype != b.dtype:
                lines.append(f"{name}: {a.dtype}{a.shape} was {b.dtype}{b.shape}")
            elif not np.array_equal(a, b, equal_nan=True):
                a64, b64 = a.astype(np.float64), b.astype(np.float64)
                diff = np.abs(a64 - b64)
                rel = diff / np.maximum(np.abs(b64), 1e-12)
                lines.append(
                    f"{name}: {int((diff > 0).sum())} of {diff.size} values differ, "
                    f"max abs {diff.max():.3g}, max rel {rel[diff > 0].max():.3g}"
                )
    return lines


# ----------------------------------------------------------------------------
# the scenes


@dataclass
class Scene:
    name: str
    what: str
    setup: Callable[[], Canvas]
    run: Callable[[Canvas], tuple[int, tuple]]  # -> (marks made, brushes used)
    reps: int = 3


SCENES: list[Scene] = []


def scene(name, what, setup=None, reps=3):
    def register(run):
        SCENES.append(Scene(name, what, setup or blank, run, reps))
        return run

    return register


def blank() -> Canvas:
    return Canvas(W, H, seed=7)


def wet() -> Canvas:
    """A canvas covered with wet paint: bands of colour, laid with a big flat."""
    c = blank()
    colours = (P.ULTRAMARINE_BLUE, P.YELLOW_OCHRE, P.ALIZARIN_CRIMSON, P.VIRIDIAN)
    for i, y in enumerate(range(40, H, 75)):
        c.stroke(
            [(20, y), (W / 2, y + 12), (W - 20, y)],
            colours[i % len(colours)],
            amount=14,
            brush=brushes.FLAT_HOG.sized(80),
            pressure=0.6,
        )
    return c


def impasto() -> Canvas:
    """Thick wet paint for knives to work in."""
    c = blank()
    for i, y in enumerate(range(60, H, 110)):
        knife = Brush(brushes.KNIFE_TROWEL).load(
            (P.TITANIUM_WHITE, P.BURNT_SIENNA, P.ULTRAMARINE_BLUE)[i % 3], 6
        )
        c.stroke([(60, y), (W - 60, y + 20)], brush=knife, pressure=0.35)
    return c


def painted() -> Canvas:
    """Wet paint all over, for ``dry()``."""
    c = wet()
    rng = np.random.default_rng(11)
    for _ in range(40):
        x, y = rng.uniform(40, W - 40), rng.uniform(40, H - 40)
        c.stroke(
            [(x, y), (x + rng.uniform(-80, 80), y + rng.uniform(-60, 60))],
            P.TITANIUM_WHITE,
            amount=5,
            brush=brushes.FILBERT_HOG,
        )
    return c


def _short_strokes(rng, n, length):
    """``n`` paths of about ``length`` px, anywhere on the canvas, any direction."""
    for _ in range(n):
        x, y = rng.uniform(30, W - 30), rng.uniform(30, H - 30)
        a = rng.uniform(0, 2 * np.pi)
        d = 0.5 * length * np.array([np.cos(a), np.sin(a)])
        bend = rng.normal(0, 0.15 * length, 2)
        yield [(x - d[0], y - d[1]), (x + bend[0], y + bend[1]), (x + d[0], y + d[1])]


@scene("round-small", "200 short strokes of a soft 8 px round")
def round_small(c):
    rng = np.random.default_rng(1)
    tool = brushes.ROUND_SABLE.sized(8)
    paints = (P.BURNT_UMBER, P.YELLOW_OCHRE, P.TITANIUM_WHITE, P.CADMIUM_RED)
    for i, path in enumerate(_short_strokes(rng, 200, 28)):
        c.stroke(path, paints[i % 4], amount=4, brush=tool, pressure=0.5)
    return 200, ()


@scene("liner", "150 thin lines with a 3 px liner over wet paint", setup=wet)
def liner(c):
    rng = np.random.default_rng(2)
    for path in _short_strokes(rng, 150, 60):
        brush = Brush(brushes.LINER).load(P.IVORY_BLACK, 22)
        c.stroke(path, brush=brush, pressure=0.35)
    return 150, ()


@scene("flat-wide", "8 long strokes of a 60 px hog flat")
def flat_wide(c):
    rng = np.random.default_rng(3)
    tool = brushes.FLAT_HOG.sized(60)
    for i in range(8):
        y = 40 + i * 70
        c.stroke(
            [(30, y), (W / 2, y + rng.uniform(-25, 25)), (W - 30, y)],
            (P.COBALT_BLUE, P.TITANIUM_WHITE)[i % 2],
            amount=16,
            brush=tool,
            pressure=0.55,
        )
    return 8, ()


@scene("filbert-wet", "12 long filbert strokes across wet paint", setup=wet)
def filbert_wet(c):
    rng = np.random.default_rng(4)
    tool = brushes.FILBERT_HOG.sized(24)
    for i in range(12):
        x = 40 + i * 62
        c.stroke(
            [(x, 30), (x + rng.uniform(-40, 40), H / 2), (x + 10, H - 30)],
            P.TITANIUM_WHITE,
            amount=22,
            brush=tool,
            pressure=(0.3, 0.7)[i % 2],
        )
    return 12, ()


@scene("blend", "8 strokes of a clean blender and a mop over wet paint", setup=wet)
def blend(c):
    rng = np.random.default_rng(5)
    for i in range(8):
        y = rng.uniform(40, H - 40)
        tool = (brushes.BLENDER, brushes.MOP)[i % 2]
        c.stroke(
            [(60, y), (W / 2, y + rng.uniform(-60, 60)), (W - 60, y)],
            brush=tool,
            pressure=[0.0, 0.3, 0.0],
        )
    return 8, ()


@scene("fan", "12 strokes of a hog fan, loaded and dry", setup=wet)
def fan(c):
    rng = np.random.default_rng(6)
    for i, path in enumerate(_short_strokes(rng, 12, 160)):
        c.stroke(
            path,
            P.SAP_GREEN if i % 3 else None,
            amount=5,
            brush=brushes.FAN_HOG,
            pressure=0.4,
        )
    return 12, ()


@scene("knife", "12 palette-knife strokes: laid on, dragged, scraped", setup=impasto)
def knife(c):
    rng = np.random.default_rng(8)
    tools = (
        brushes.PALETTE_KNIFE,
        brushes.KNIFE_TROWEL,
        brushes.KNIFE_PEAR,
        brushes.KNIFE_STRAIGHT,
        brushes.KNIFE_COMB,
        brushes.KNIFE_SMALL,
    )
    paints = (P.TITANIUM_WHITE, P.CADMIUM_YELLOW, P.PHTHALO_BLUE)
    for i, path in enumerate(_short_strokes(rng, 12, 170)):
        tool = tools[i % len(tools)]
        brush = Brush(tool)
        if i % 4 != 3:  # (every fourth: a clean blade, dragging what is there)
            brush.load(paints[i % 3], 3.0)
        c.stroke(path[::2], brush=brush, pressure=(0.25, 0.5, 0.85)[i % 3])
    return 12, ()


@scene("stipple", "400 taps of a stippler, in four loads")
def stipple(c):
    rng = np.random.default_rng(9)
    tools = []
    for k in range(4):
        pts = np.column_stack(
            [rng.uniform(30, W - 30, 100), rng.uniform(30, H - 30, 100)]
        )
        tools.append(
            c.stipple(
                pts,
                (P.SAP_GREEN, P.CADMIUM_YELLOW, P.VIRIDIAN, P.TITANIUM_WHITE)[k],
                amount=30,
                width=(26, 14)[k % 2],
                pressure=rng.uniform(0.15, 0.6, 100),
            )
        )
    return 400, tuple(tools)


@scene("spatter", "three flicks of spatter (about 400 drops)", setup=wet)
def spatter(c):
    n = c.spatter((150, 80), P.TITANIUM_WHITE, amount=1.2, direction=70, seed=1)
    n += c.spatter(
        [(400, 60), (650, 120)], P.CADMIUM_ORANGE, amount=1.2, direction=90,
        reach=200, size=2.0, seed=2,
    )  # fmt: skip
    n += c.spatter((300, 300), P.IVORY_BLACK, amount=0.8, spread=60, size=5, seed=3)
    return n, ()


@scene("dry", "dry() on a painted canvas, twice", setup=painted)
def dry(c):
    c.dry()
    c.stroke([(100, 100), (700, 500)], P.COBALT_BLUE, amount=20, width=30)
    c.dry()
    return 2, ()


@scene("painting", "a small picture, start to finish (about 450 marks)")
def painting(c):
    """Painted the way the examples paint: a toned ground scrubbed in and left
    to dry; the picture blocked in with short strokes of a flat, each from a
    freshly loaded brush; softened with a mop; worked up wet-into-wet with
    filberts and small rounds; lines drawn with a liner; lights laid on with a
    knife; grass flicked in with a fan; foliage stippled; a flick of spatter;
    and, once dry, a glaze."""
    rng = np.random.default_rng(1659)
    marks = 0
    earth = (P.BURNT_UMBER, P.RAW_UMBER, P.YELLOW_OCHRE, P.BURNT_SIENNA)
    sky = (P.COBALT_BLUE, P.ULTRAMARINE_BLUE, P.TITANIUM_WHITE)
    mixes = [
        Paint.mix((a, 2), (b, 1), (P.TITANIUM_WHITE, w))
        for a in earth
        for b in sky
        for w in (0.3, 2.0)
    ]

    def fresh(tool, size, path, paint, load=0.8):
        run = float(np.linalg.norm(np.diff(np.asarray(path), axis=0), axis=1).sum())
        return Brush(tool.sized(size)).load(paint, load * (run / size + 1.0))

    # a toned ground, criss-cross with a two-inch brush
    two_inch = brushes.FLAT_HOG.sized(90)
    for i in range(8):
        y = 40 + i * 75
        slope = 60 if i % 2 else -60
        c.stroke(
            [(-20, y - slope), (W + 20, y + slope)],
            P.RAW_UMBER.with_body(0.3),
            amount=12,
            brush=two_inch,
            pressure=0.5,
        )
        marks += 1
    c.dry()
    # blocked in
    for path in _short_strokes(rng, 90, 70):
        paint = mixes[int(rng.integers(len(mixes)))]
        c.stroke(
            path[::2], brush=fresh(brushes.FLAT_HOG, 28, path[::2], paint), pressure=0.6
        )
        marks += 1
    # softened
    for path in _short_strokes(rng, 5, 260):
        c.stroke(path, brush=Brush(brushes.MOP.sized(60)), pressure=[0.0, 0.14, 0.0])
        marks += 1
    # worked up, big to small
    for size, n, tool in (
        (18, 60, brushes.FILBERT_HOG),
        (10, 80, brushes.FILBERT_SABLE),
        (6, 80, brushes.ROUND_SABLE),
    ):
        for path in _short_strokes(rng, n, 2.5 * size):
            paint = mixes[int(rng.integers(len(mixes)))]
            c.stroke(path, brush=fresh(tool, size, path, paint), pressure=0.5)
            marks += 1
    # drawn lines
    for path in _short_strokes(rng, 40, 50):
        c.stroke(path, brush=fresh(brushes.LINER, 3, path, P.IVORY_BLACK, 1.5))
        marks += 1
    # lights with the knife
    for path in _short_strokes(rng, 8, 90):
        blade = Brush(brushes.PALETTE_KNIFE).load(P.TITANIUM_WHITE, 2.5)
        c.stroke(path[::2], brush=blade, pressure=float(rng.uniform(0.2, 0.6)))
        marks += 1
    # grass
    for path in _short_strokes(rng, 6, 110):
        c.stroke(path, P.SAP_GREEN, amount=4, brush=brushes.FAN_HOG, pressure=0.35)
        marks += 1
    # foliage, sparks
    pts = np.column_stack([rng.uniform(40, W - 40, 60), rng.uniform(40, H / 2, 60)])
    tool = c.stipple(pts, P.VIRIDIAN, amount=20, width=22, pressure=0.35)
    marks += 60
    marks += c.spatter((W / 2, 40), P.CADMIUM_YELLOW, amount=0.4, direction=90, seed=5)
    c.dry()
    # a glaze over the dry paint
    glaze = Paint("#5a3418", opacity=0.25)
    for i in range(5):
        y = 70 + i * 110
        c.stroke(
            [(30, y), (W - 30, y + 15)],
            glaze,
            amount=14,
            brush=brushes.BLENDER.sized(80),
            pressure=0.4,
        )
        marks += 1
    return marks, (tool,)


# ----------------------------------------------------------------------------


def warm_up():
    """Compile (or load from the cache) every kernel the scenes use."""
    c = Canvas(160, 120, seed=1)
    c.stroke([(20, 20), (140, 100)], P.COBALT_BLUE, width=20)
    c.stroke([(20, 100), (140, 20)], P.TITANIUM_WHITE, brush=brushes.FAN_HOG)
    c.stroke([(20, 60), (140, 60)], brush=brushes.MOP.sized(30))
    for tool in (brushes.PALETTE_KNIFE, brushes.KNIFE_COMB):
        c.stroke([(30, 30), (130, 90)], P.CADMIUM_RED, brush=tool, pressure=0.3)
    c.stroke([(30, 90), (130, 30)], brush=brushes.BRUSH_HANDLE, pressure=0.8)
    c.stipple([(40, 40), (80, 80)], P.SAP_GREEN)
    c.spatter((20, 20), P.IVORY_BLACK, amount=0.1, seed=1)
    c.dry()


def main(argv=None):
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    ap.add_argument("-k", metavar="TEXT", help="only scenes whose name contains this")
    ap.add_argument("-n", type=int, help="time each scene this many times")
    ap.add_argument("--save", metavar="DIR", help="keep each scene's state (npz)")
    ap.add_argument("--compare", metavar="DIR", help="compare with a kept state")
    args = ap.parse_args(argv)

    t0 = time.perf_counter()
    warm_up()
    print(
        f"warm-up (compiling, or loading the cache): {time.perf_counter() - t0:.1f} s"
    )
    print(
        f"{'scene':<12} {'marks':>5} {'best ms':>9} {'median':>9} {'cpu ms':>9} "
        f"{'ms/mark':>8}  state"
    )
    differences = []
    for sc in SCENES:
        if args.k and args.k not in sc.name:
            continue
        start = sc.setup()
        wall, cpu, marks, state = [], [], 0, {}
        for _ in range(args.n or sc.reps):
            c = copy.deepcopy(start)
            gc.collect()
            w0, c0 = time.perf_counter(), time.process_time()
            marks, tools = sc.run(c)
            wall.append(time.perf_counter() - w0)
            cpu.append(time.process_time() - c0)
            again = canvas_state(c, tools)
            if state and digest(again) != digest(state):
                differences.append(f"{sc.name}: not the same from one run to the next")
            state = again
        best = min(wall)
        print(
            f"{sc.name:<12} {marks:>5} {1e3 * best:>9.1f} "
            f"{1e3 * statistics.median(wall):>9.1f} {1e3 * min(cpu):>9.1f} "
            f"{1e3 * best / max(marks, 1):>8.2f}  {digest(state)[:12]}"
        )
        if args.save:
            os.makedirs(args.save, exist_ok=True)
            np.savez_compressed(  # pyright: ignore[reportCallIssue]
                os.path.join(args.save, sc.name + ".npz"),
                **state,  # pyright: ignore[reportArgumentType]
            )
        if args.compare:
            path = os.path.join(args.compare, sc.name + ".npz")
            differences += [f"{sc.name}: {line}" for line in compare(state, path)]
    if args.compare:
        print("same paint as", args.compare if not differences else "... no:")
    for line in differences:
        print("  " + line)
    return 1 if differences else 0


if __name__ == "__main__":
    sys.exit(main())
