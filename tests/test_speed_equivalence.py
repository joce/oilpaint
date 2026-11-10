"""The speed work keeps the arithmetic: compiled ports and shortcuts give the
same numbers, bit for bit, as the code they replaced (kept here as the
reference), on random and edge-case inputs."""

import math

import numpy as np
from numba import njit

from oilpaint import _engine as eng
from oilpaint._engine import IV, PACKED, SAME_LAYER, SAME_PAINT


@njit(cache=False, error_model="numpy")
def _harmless_ref(top, und):
    """harmless() as it was: all its logarithms."""
    if und[IV] < 1e-4:
        return True
    zero, floor = np.float32(0.0), np.float32(1e-9)
    va, vb = max(max(top[IV], zero), floor), max(max(und[IV], zero), floor)
    tot = 0.0
    for c in range(IV):
        tot += abs(
            math.log1p(max(top[c], zero) / va) - math.log1p(max(und[c], zero) / vb)
        )
        if not tot / IV < SAME_LAYER:
            return False
    return True


def _paints(rng, n, dtype):
    """Pairs of packed paints: unrelated, nearly the same (to rounding and
    beyond), some components negative or zero, empty."""
    a = rng.uniform(0.0, 3.0, (n, PACKED)) * rng.uniform(0.01, 10.0, (n, 1))
    b = a * (1.0 + rng.normal(0.0, 1.0, (n, 1)) * 10.0 ** rng.uniform(-9, 0, (n, 1)))
    b[: n // 4] = rng.uniform(0.0, 3.0, (n // 4, PACKED))
    a[n // 8 : n // 4, rng.integers(0, PACKED)] = -1e-8
    b[n // 4 : n // 3, :] = 0.0
    a[n // 3 : n // 2 : 7, IV] = 1e-12
    return a.astype(np.float32), b.astype(dtype)


def test_likeness_bounds_decide_as_the_logarithms_do():
    rng = np.random.default_rng(0)
    for dtype in (np.float32, np.float64):
        a, b = _paints(rng, 40_000, dtype)
        for i in range(len(a)):
            full = eng.alike(a[i], b[i])
            for thr in (SAME_PAINT, 0.25, full, np.nextafter(full, 1.0)):
                assert eng.alike_below(a[i], b[i], thr) == (full < thr)
                up = eng.alike_upto(a[i], b[i], thr)
                same = up == full or (math.isnan(up) and math.isnan(full))
                assert same or (up == np.inf and not full < thr)
            if dtype == np.float32:
                assert eng.harmless(a[i], b[i]) == _harmless_ref(a[i], b[i])
    # and they do decide: most ordinary pairs (not tied to a threshold)
    a, b = _paints(rng, 4_000, np.float32)
    floor = np.float32(1e-9)
    decided = 0
    for i in range(len(a)):
        va, vb = max(a[i, IV], floor), max(b[i, IV], floor)
        lo, hi = eng.likeness_bounds(a[i], b[i], va, vb, False)
        decided += hi / IV < SAME_PAINT or lo / IV >= SAME_PAINT
    assert decided > 0.6 * len(a)


def test_harmless_near_its_threshold():
    """Paints whose likeness lies right at SAME_LAYER, either side."""
    rng = np.random.default_rng(1)
    hits = 0
    for _ in range(20_000):
        a = rng.uniform(0.1, 2.0, PACKED).astype(np.float32)
        b = a.copy()
        k = rng.integers(0, IV)
        # one band moved so that the mean difference is about SAME_LAYER
        b[k] = np.float32(np.expm1(np.log1p(a[k] / a[IV]) + SAME_LAYER * IV) * a[IV])
        b[k] *= np.float32(1.0 + rng.normal(0.0, 1e-6))
        got, want = eng.harmless(a, b), _harmless_ref(a, b)
        assert got == want
        hits += want
    assert 0 < hits < 20_000


def _bilinear_ref(grid, y, x):
    """_filters._bilinear, as the knife's colour bands used it."""
    H, W = grid.shape[:2]
    y0, x0 = np.minimum(y.astype(int), H - 2), np.minimum(x.astype(int), W - 2)
    fy, fx = y - y0, x - x0
    return (grid[y0, x0] * (1 - fx) + grid[y0, x0 + 1] * fx) * (1 - fy) + (
        grid[y0 + 1, x0] * (1 - fx) + grid[y0 + 1, x0 + 1] * fx
    ) * fy


def test_wandering_bands_and_cropped_taps():
    rng = np.random.default_rng(2)
    F = rng.normal(0.0, 30.0, (15, 23, 31))
    F[eng.F_U] = rng.uniform(-1.3, 1.3, (23, 31))
    bands = rng.random((40, 64))
    s0, period = 57.25, 9.5
    along_s = np.clip((s0 + F[eng.F_ALONG]) / period * 8.0, 0.0, 40 - 1.001)
    u_band = np.clip((F[eng.F_U] + 1.0) * 0.5 * 63.0, 0.0, 62.999)
    want = F[eng.F_U] + 0.12 * (2.0 * _bilinear_ref(bands, along_s, u_band) - 1.0)
    got = eng.wandering_bands(F, 0, 0, 23, 31, s0, period, bands, 0.12)
    assert got.tobytes() == want.tobytes()
    part = eng.wandering_bands(F, 3, 5, 10, 12, s0, period, bands, 0.12)
    assert part.tobytes() == np.ascontiguousarray(want[3:13, 5:17]).tobytes()
    for ring, nsub in ((False, 1), (False, 5), (True, 2), (True, 8)):
        whole = eng.bristle_taps(want, 0.3, nsub, ring, 0.7)
        crop = eng.crop_taps(whole, 31, 3, 5, 10, 12)
        sub = eng.bristle_taps(
            np.ascontiguousarray(want[3:13, 5:17]), 0.3, nsub, ring, 0.7
        )
        assert (crop[2] == sub[2]).all()
        for q in range(10 * 12):
            n = sub[2][q]
            assert (crop[0][q, :n] == sub[0][q, :n]).all()
            assert crop[1][q, :n].tobytes() == sub[1][q, :n].tobytes()


def _canvas(rng, H=40, W=50):
    """Wet paint with specks, negative dust and signed zeros in it."""
    top = rng.uniform(0.0, 3.0, (H, W, PACKED)).astype(np.float32)
    und = rng.uniform(0.0, 3.0, (H, W, PACKED)).astype(np.float32)
    for a in (top, und):
        a[rng.random((H, W)) < 0.2] = 0.0
        a[rng.random((H, W, PACKED)) < 0.05] = -0.0
        a[rng.random((H, W, PACKED)) < 0.02] *= -1e-6
    floor = rng.uniform(0.0, 1.5, (H, W)).astype(np.float32)
    return top, und, floor


def test_blade_slab_ends_excess_and_folds_match_numpy():
    rng = np.random.default_rng(3)
    top, und, floor = _canvas(rng)
    H, W = floor.shape
    F = rng.normal(0.0, 5.0, (15, 30, 34))
    F[eng.F_EXPO] = np.where(rng.random((30, 34)) < 0.3, 0.0, rng.random((30, 34)))
    F[eng.F_HAIR] = rng.random((30, 34))
    level = rng.uniform(-1.0, 8.0, (30, 34))
    film = rng.uniform(0.0, 3.0, (30, 34))
    film[::4] = 0.0
    fill = 1.0 - np.exp(-4.0 * F[eng.F_EXPO])
    # past the top left edge of the canvas: rows -6.., columns -4..
    ys, xs = np.clip(np.arange(-6, 24), 0, H - 1), np.clip(np.arange(-4, 30), 0, W - 1)
    sl = np.ix_(ys, xs)
    surf_add = rng.normal(0.0, 1.0, (30, 34))
    for add in (None, surf_add):
        surf = floor[sl] + top[..., IV][sl] + und[..., IV][sl]
        if add is not None:
            surf = surf + add
        gap_px = np.clip(level - surf, 0.0, None)
        t = np.clip((gap_px / np.maximum(film, 1e-9) - 1.3) / (0.7 - 1.3), 0.0, 1.0)
        want = gap_px * (t * t * (3.0 - 2.0 * t)) * fill
        got, s = eng.blade_slab(
            top, und, floor, -6, -4, level, np.zeros((0, 0)) if add is None else add,
            film, fill,
        )  # fmt: skip
        assert got.tobytes() == want.tobytes()
        assert s.tobytes() == np.asarray(surf, np.float64).tobytes()
    ride = rng.uniform(0.0, 3.0, (30, 34))

    def ss(e0, e1, x):
        t = np.clip((x - e0) / (e1 - e0), 0.0, 1.0)
        return t * t * (3.0 - 2.0 * t)

    near = ss(5.0, 0.0, 7.5 - np.abs(F[eng.F_LAT]))
    want = ride * F[eng.F_EXPO] * 1.7 * near * ss(0.5, 2.0, ride)
    want[4:14, 6:16] = 0.0
    got = eng.blade_ends(ride, F, 0, 0, 7.5, 5.0, 1.7, 4, 6, 10, 10)
    assert got.tobytes() == want.tobytes()
    # on the canvas, from (5, 7)
    lv = np.ascontiguousarray(level[:20, :25])
    tv = top[5:25, 7:32, IV].astype(np.float64)
    excess = np.clip(floor[5:25, 7:32] + tv + und[5:25, 7:32, IV] - lv, 0.0, None)
    excess *= fill[:20, :25]
    wet = np.ascontiguousarray(top[5:25, 7:32, IV] + und[5:25, 7:32, IV])
    near = ss(8.0, 0.0, 7.5 - np.abs(F[eng.F_LAT][:20, :25]))
    for spill in (False, True):
        want = 0.3 * near * np.minimum(excess, wet) if spill else excess
        got, got_tv, emax = eng.blade_excess(
            top, und, floor, 5, 7, lv, np.ascontiguousarray(fill[:20, :25]), F, 0, 0,
            7.5, 8.0, 0.3, wet if spill else np.zeros((0, 0), np.float32),
        )  # fmt: skip
        assert got.tobytes() == want.tobytes() and got_tv.tobytes() == tv.tobytes()
        assert emax == want.max()
    mix = 1.0 - np.exp(-0.7 * F[eng.F_EXPO][:20, :25])
    into = np.clip(floor[5:25, 7:32] + und[5:25, 7:32, IV] + 0.5 - lv, 0.0, 1.0)
    want = np.clip(mix * (0.1 + 1.8 * F[eng.F_HAIR][:20, :25]) * into, 0.0, 1.0)
    got = eng.blade_folds(und, floor, 5, 7, lv, np.ascontiguousarray(mix), F, 0, 0)
    assert got.tobytes() == want.tobytes()


def test_shift_matches_the_four_tap_shift():
    rng = np.random.default_rng(4)
    a = rng.normal(0.0, 1.0, (17, 23, PACKED)).astype(np.float32)
    a[rng.random(a.shape) < 0.1] = -0.0
    for sx, sy in ((0.3, -0.7), (-2.5, 1.0), (4.0, 0.0), (0.0, 1e-8), (30.0, 2.2)):
        ix, iy = math.floor(sx), math.floor(sy)
        fx, fy = sx - ix, sy - iy
        want = np.zeros_like(a)
        for dx, wx in ((ix, 1.0 - fx), (ix + 1, fx)):
            for dy, wy in ((iy, 1.0 - fy), (iy + 1, fy)):
                if wx * wy < 1e-6 or abs(dx) >= 23 or abs(dy) >= 17:
                    continue
                w = np.float32(wx * wy)
                for y in range(max(dy, 0), min(17 + dy, 17)):
                    lo, hi = max(dx, 0), min(23 + dx, 23)
                    want[y, lo:hi] += w * a[y - dy, lo - dx : hi - dx]
        assert eng.shift(a, sx, sy).tobytes() == want.tobytes()


def test_a_brush_footprint_is_the_same_where_it_is_read():
    """A brush's footprint is worked out only where it can touch (see
    _engine.may_touch): where it touches, and every plane over the part it
    touches, are as over all of the window."""
    from oilpaint import Canvas, Stroke, brushes
    from oilpaint.stroke import _Run

    rng = np.random.default_rng(5)
    for k, bt in enumerate(
        (brushes.FLAT_HOG, brushes.ROUND_SABLE, brushes.FAN_HOG, brushes.FILBERT_HOG,
         brushes.DAGGER, brushes.MOP, brushes.LINER)
    ):  # fmt: skip
        c = Canvas(120, 90, seed=k)
        for _ in range(4):
            pts = [tuple(p) for p in rng.uniform(-20, 130, (3, 2))]
            dab = rng.random() < 0.25
            stroke = Stroke(
                path=pts[:1] if dab else pts, paint="#884422", amount=6.0,
                width=float(rng.uniform(4, 60)), pressure=float(rng.uniform(0.1, 1)),
                brush=bt, angle=float(rng.uniform(0, 180)),
            )  # fmt: skip
            run = _Run.begin(c, stroke)
            for st, tables in run.steps():
                Fb, hb = eng.footprint(st, run.K, run.tex, run.tex2, *tables)
                K = run.K._replace(rigid=True)
                Ff, hf = eng.footprint(st, K, run.tex, run.tex2, *tables)
                assert hb == hf
                # (exposure and cover only where it touches)
                for p in (eng.F_EXPO, eng.F_COVER):
                    assert Fb[p].tobytes() == Ff[p].tobytes()
                touched = (Ff[eng.F_AA] != 0) & (Ff[eng.F_OVERLAP] != 0)
                assert not (touched & (Fb[eng.F_AA] == 0)).any()
                part = eng.touched(st, Fb)
                assert part == eng.touched(st, Ff)
                box = (
                    slice(part.wy0, part.wy0 + part.ch),
                    slice(part.wx0, part.wx0 + part.cw),
                )
                assert (
                    Fb[:, box[0], box[1]].tobytes() == Ff[:, box[0], box[1]].tobytes()
                )


# --- a knife's blade, as it was worked out before the stages were compiled
# (verbatim from the base, ac672aa: _knife._along_blade, _blade_rest and
# _blade_level; ``s.ucoord`` read as the F_U plane, the film passed in)


def _along_blade_ref(lat, under, values, span):
    b = np.round(lat - lat.min()).astype(int)
    n = int(b.max()) + 1
    cnt = np.bincount(b[under], minlength=n).astype(np.float64)
    tot = np.bincount(b[under], values[under], minlength=n)
    kernel = np.ones(2 * span + 1)
    cnt_s = np.convolve(cnt, kernel, mode="same")
    tot_s = np.convolve(tot, kernel, mode="same")
    prof = np.where(cnt_s > 0, tot_s / np.maximum(cnt_s, 1e-9), 0.0)
    return np.where(cnt_s[b] > 0, prof[b], values)


def _blade_rest_ref(run, s, at=None):
    """(``at``: the canvas pixels to read, if not the step's ``sl``: past the
    edge, the gathered nearest ones.)"""
    from oilpaint._engine import IB
    from oilpaint._knife import _KNIFE_SINK

    canvas, bt, st = run.canvas, run.bt, s.st
    p, sl, wsl = st.p, s.sl if at is None else at, s.wsl
    top, und, floor = canvas.top[sl], canvas.und[sl], run.relief.floor[sl]
    fine, wide = run.relief.peaks_fine[sl], run.relief.peaks_wide[sl]
    rest = fine + run.stiff_blade * (wide - fine)
    earlier = canvas.top_id[sl] != run.stroke_id
    old_v = und[..., IV] + np.where(earlier, top[..., IV], 0.0)
    if old_v.max() > 0.05:
        old_body = float(
            (und[..., IB].sum() + np.where(earlier, top[..., IB], 0.0).sum())
            / max(float(old_v.sum()), 1e-9)
        )
        depth = float(
            np.clip(p**_KNIFE_SINK * (1.15 - 0.3 * min(old_body, 1.0)), 0.0, 1.0)
        )
        if bt.clumps > 0:
            depth = max(depth, 0.9)
        under_blade = (s.aa * np.clip(s.h - np.abs(s.rel) + 0.5, 0.0, 1.0))[wsl] > 0.5
        if under_blade.any():
            span = int(np.clip(0.1 * run.w_typ * (0.5 + run.stiff_blade), 2, 10))
            hi = eng.max_filter(floor + old_v, 1)
            high = _along_blade_ref(s.lat_px[wsl], under_blade, hi, span)
            base = _along_blade_ref(s.lat_px[wsl], under_blade, rest, span)
            rest = np.maximum(rest, high - depth * np.maximum(high - base, 0.0))
    return rest


def _blade_level_ref(run, s, gap, film, at=None):
    from oilpaint._filters import _smoothstep
    from oilpaint._knife import _KNIFE_FILM, _KNIFE_SKIM

    st, wsl = s.st, s.wsl
    rest = _blade_rest_ref(run, s, at)
    hair = s.hair[wsl]
    ride = np.minimum(gap, _KNIFE_FILM * film) if run.bt.clumps == 0 else gap
    chat = 0.25 * (1.0 - float(_smoothstep(_KNIFE_SKIM, 0.05, st.p)))
    level = rest + ride * (
        1.0
        + chat * run.chatter[st.i]
        + 0.12 * (hair - 0.5)
        + run.blade_tilt * np.clip(s.F[eng.F_U][wsl], -1.0, 1.0)
    )
    return level, ride


def test_a_knife_rests_and_rides_as_it_did(monkeypatch):
    """Where a knife's blade rests and how high it rides (blade_rest,
    blade_bins, blade_sink, blade_level), on its part of the canvas and over
    all of its window past the edge of the canvas (gathered from the nearest
    pixels), and what it adds to the surface out there (off_surface), as the
    numpy code worked them out: every blade shape, a comb, lifted off, on
    wet paint of earlier strokes; and a blade wider than 128 px on the
    canvas, whose paint below is summed as numpy summed it (over a strided
    view of the canvas, see _knife._blade_rest)."""
    import oilpaint.stroke as stm
    from oilpaint import Brush, Canvas, brushes
    from oilpaint import _knife as kn
    from oilpaint import palette as P

    checked = {"on": 0, "whole": 0, "wide": 0}
    original = kn._knife_step

    def same(got, want):
        want = np.broadcast_to(np.asarray(want, np.float64), got.shape)
        return got.tobytes() == np.ascontiguousarray(want).tobytes()

    def check(run, s):
        st = s.st
        area = max(run.area_unit * st.r * st.r, 1e-9)
        gap, _ = kn._blade_gap(run, st.p)
        if st.ch and st.cw:
            film = kn._film(run, s, area)
            level, ride = kn._blade_level(run, s, gap, film)
            want_l, want_r = _blade_level_ref(run, s, gap, film)
            assert same(level, want_l) and same(ride, want_r)
            checked["on"] += 1
            checked["wide"] += st.cw > 128
        if st.ch < st.hh or st.cw < st.ww:
            whole = kn._whole_window(run, s)
            sw = whole.s
            H, W = run.canvas.height, run.canvas.width
            ys = np.clip(np.arange(st.y0, st.y0 + st.hh), 0, H - 1)
            xs = np.clip(np.arange(st.x0, st.x0 + st.ww), 0, W - 1)
            at = np.ix_(ys, xs)
            film = kn._film(run, sw, area)
            level, ride = kn._blade_level(run, sw, gap, film)
            want_l, want_r = _blade_level_ref(run, sw, gap, film, at)
            assert same(level, want_l) and same(ride, want_r)
            canvas = run.canvas
            own = np.where(
                canvas.top_id[at] == run.stroke_id, canvas.top[at][..., IV], 0.0
            )
            on = np.zeros((st.hh, st.ww), bool)
            on[s.wsl] = True
            want = np.where(on, 0.0, kn._laid_off(run, st) - own)
            assert whole.surf_add.tobytes() == want.tobytes()
            checked["whole"] += 1
        original(run, s)

    monkeypatch.setattr(stm, "_knife_step", check)
    rng = np.random.default_rng(7)
    c = Canvas(220, 160, seed=4)
    tools = (
        brushes.PALETTE_KNIFE, brushes.KNIFE_PEAR, brushes.KNIFE_STRAIGHT,
        brushes.KNIFE_ANGLED, brushes.KNIFE_SPOON, brushes.KNIFE_COMB,
        brushes.KNIFE_TROWEL,
    )  # fmt: skip
    for k in range(14):
        brush = Brush(tools[k % len(tools)])
        if k % 3 != 2:
            brush.load((P.TITANIUM_WHITE, P.CADMIUM_RED)[k % 2], 4.0)
        pts = [tuple(p) for p in rng.uniform(-30, 250, (2, 2))]
        c.stroke(pts, brush=brush, pressure=float(rng.uniform(0.15, 0.9)),
                 lift=(0.0, 0.8)[k % 2], seed=k)  # fmt: skip
    wide = Canvas(400, 200, seed=5)
    wide.stroke([(20, 60), (380, 70)], P.COBALT_BLUE, amount=10, width=60)
    for y in (90, 110):
        blade = Brush(brushes.SCRAPER.sized(260)).load(P.TITANIUM_WHITE, 3.0)
        wide.stroke([(200, y - 40), (205, y + 30)], brush=blade, pressure=0.6)
    assert checked["on"] > 100 and checked["whole"] > 50 and checked["wide"] > 3


def test_a_strided_sum_is_not_that_of_its_copy():
    """Why _knife._blade_rest sums the paint below over the canvas itself:
    numpy sums a strided float32 view row by row and a contiguous copy of it
    as one run, and the two round differently."""
    rng = np.random.default_rng(8)
    differ = 0
    for _ in range(50):
        und = rng.uniform(0.0, 3.0, (40, 400, PACKED)).astype(np.float32)
        view = und[..., 1][3:33, 20:300]
        differ += view.sum() != np.ascontiguousarray(view).sum()
    assert differ > 0
