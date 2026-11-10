"""The relief thick paint takes: furrows, the start and end of a stroke, how
thick a load goes on."""

import numpy as np
import pytest

from oilpaint import Canvas, brushes
from oilpaint import brush as bm
from oilpaint import palette as P

X0, X1, YC = 60, 410, 60


def _stroke(amount=60, bt=None, seed=1, texture=None, paint=None, **kw):
    c = Canvas(470, 120, seed=12, texture=texture)
    brush = c.stroke(
        [(X0, YC), (X1, YC)],
        paint or P.CADMIUM_RED,
        amount=amount,
        brush=bt or brushes.FLAT_HOG.sized(36),
        seed=seed,
        **kw,
    )
    return c, brush


def _furrows(h, y0=YC - 12, y1=YC + 13, x0=X0 + 60, x1=X1 - 60):
    """Cross-profiles of a horizontal stroke with the smooth shape (11 px)
    taken out: the furrows."""
    reg = h[y0:y1, x0:x1].astype(np.float64)
    k = 11
    pad = np.pad(reg, ((k // 2, k // 2), (0, 0)), mode="edge")
    smooth = np.apply_along_axis(
        lambda v: np.convolve(v, np.ones(k) / k, mode="valid"), 0, pad
    )
    return reg - smooth


def _along_corr(det, lag):
    a, b = det[:, :-lag], det[:, lag:]
    num = (a * b).sum(0)
    den = np.sqrt((a * a).sum(0) * (b * b).sum(0)) + 1e-12
    return float((num / den).mean())


def _crests_per_px(field_rows):
    """Crests per px across a set of profiles (1 px apart), lightly smoothed;
    a crest stands at least 0.6 x the rms above the lowest point within 4 px."""
    count, total = 0, 0
    for p in field_rows:
        p = np.convolve(p - p.mean(), [0.25, 0.5, 0.25], mode="valid")
        rms = np.sqrt((p**2).mean())
        for i in range(1, len(p) - 1):
            if p[i] > p[i - 1] and p[i] >= p[i + 1]:
                lo = min(p[max(i - 4, 0) : i].min(), p[i + 1 : i + 5].min())
                count += p[i] - lo > 0.6 * rms
        total += len(p)
    return count / total


def test_furrows_wander_and_fade_along_a_stroke_but_last_a_width():
    corr40, corr120 = [], []
    for seed in (1, 2, 3):
        c, _ = _stroke(seed=seed)
        det = _furrows(c.thickness)
        corr40.append(_along_corr(det, 40))
        corr120.append(_along_corr(det, 120))
    # not one cross-section pulled along the stroke (that was ~0.95) ...
    assert np.mean(corr120) < 0.6
    assert np.mean(corr120) < np.mean(corr40) - 0.15
    # ... but a furrow lasts about a brush width
    assert np.mean(corr40) > 0.35


def _furrow_spacing(w, seed):
    """Spacing (px) of the furrows across the middle of a painted stroke:
    twice the lag of the first minimum of the across-stroke autocorrelation
    of its cross-profiles (each less its straight-line trend)."""
    c = Canvas(470, 160, seed=12)
    c.stroke(
        [(X0, 80), (X1, 80)],
        P.CADMIUM_RED,
        amount=45,
        brush=brushes.FLAT_HOG.sized(w),
        seed=seed,
    )
    half = int(round(0.3 * w))
    reg = c.thickness.astype(np.float64)[80 - half : 80 + half + 1, X0 + 60 : X1 - 60]
    n = reg.shape[0]
    t = np.arange(n) - (n - 1) / 2
    A = np.stack([np.ones(n), t], 1)
    det = reg - A @ np.linalg.lstsq(A, reg, rcond=None)[0]
    R = [
        (det[: n - lag] * det[lag:]).sum()
        / np.sqrt((det[: n - lag] ** 2).sum() * (det[lag:] ** 2).sum())
        for lag in range(n // 2 + 1)
    ]
    for lag in range(1, len(R) - 1):
        if R[lag] < R[lag - 1] and R[lag] <= R[lag + 1]:
            return 2.0 * lag
    return float(n)


def test_painted_furrows_are_about_the_same_size_on_any_width_of_brush():
    narrow = np.mean([_furrow_spacing(18, s) for s in (1, 2, 3)])
    wide = np.mean([_furrow_spacing(72, s) for s in (1, 2, 3)])
    # (a brush four times as wide combed furrows about four times as wide; the
    # measure itself reads a little wider on a wider stroke)
    assert wide < 3.5 * narrow  # (it was 4.3-4.9)


def test_the_hair_pattern_is_the_same_size_on_any_width_of_brush():
    per_px = []
    for w in (18, 72):
        rows = []
        for seed in range(4):
            rng = np.random.default_rng(seed)
            f, (_, x0, _, dx) = bm._hair_field(
                brushes.FLAT_HOG.sized(w), w, w / 2, -20.0, 300.0, rng
            )
            xs = (np.arange(-0.3 * w, 0.3 * w + 1) - x0) / dx
            rows.extend(np.interp(xs, np.arange(f.shape[1]), r) for r in f[::5])
        per_px.append(_crests_per_px(rows))
    assert abs(per_px[0] / per_px[1] - 1.0) < 0.25
    # coarse hog hair: a crest every 4 to 10 px
    assert 1 / 10 < per_px[1] < 1 / 4


def test_no_ripple_at_the_step_period():
    # the brush neither lifts nor pushes its own fresh paint step by step
    c, _ = _stroke(amount=60, texture=0.0)
    stroke = bm.Stroke(path=[(X0, YC), (X1, YC)], brush=brushes.FLAT_HOG.sized(36))
    run = bm._Run(Canvas(470, 120, seed=12), stroke)
    ds = float(np.mean(np.linalg.norm(np.diff(run.pts, axis=0), axis=1)))
    h = c.thickness[YC - 10 : YC + 11, X0 + 60 : X1 - 60].astype(np.float64)
    band = h.mean(0)
    det = band - np.convolve(np.pad(band, 15, mode="edge"), np.ones(31) / 31, "valid")
    phase = ((np.arange(X0 + 60, X1 - 60) + 0.5 - X0) % ds) / ds
    bins = np.minimum((phase * 8).astype(int), 7)
    ripple = np.array([det[bins == k].mean() for k in range(8)]).std()
    assert ripple < 0.006 * h.mean()  # (it was 0.024)


def test_thin_strokes_reach_as_far_as_their_load_says():
    # amount ~ widths of solid stroke, whatever the brush
    for bt in (brushes.ROUND.sized(16), brushes.FLAT_SABLE.sized(24)):
        c = Canvas(500, 80, seed=4)
        w = bt.width
        c.stroke([(20, 40), (480, 40)], P.CADMIUM_RED, amount=8, brush=bt, seed=9)
        cols = np.nonzero(c.top_a.sum(0) > 0.5 * w)[0]
        reach = (cols.max() - cols.min()) / w
        assert 7 < reach < 13


def test_a_skimming_knife_lays_no_sawtooth_along_its_pointed_end():
    # a diamond blade pulled sideways: its point is shorter along the stroke
    # than a step, yet the slab beside it is as smooth as in the middle
    sky = P.TITANIUM_WHITE
    c = Canvas(400, 110, seed=21, texture=0.0)
    c.stroke(
        [(70, 55), (330, 55)],
        sky,
        amount=70,
        brush=brushes.PALETTE_KNIFE.sized(56),
        pressure=0.15,
        seed=2,
    )
    for rows in (slice(29, 41), slice(69, 81)):  # near the heel, near the point
        band = c.thickness[rows, 120:300].astype(np.float64).mean(0)
        det = band - np.convolve(
            np.pad(band, 12, mode="edge"), np.ones(25) / 25, "valid"
        )
        assert det.std() < 0.1  # (it was 0.6 near the point)


def _body(c, w=36):
    half = int(0.35 * w)
    return float(c.thickness[YC - half : YC + half + 1, X0 + 60 : X1 - 60].mean())


def test_an_overloaded_brush_lays_as_thick_as_its_load():
    t60 = _body(_stroke(amount=60)[0])
    t120 = _body(_stroke(amount=120)[0])
    assert t120 > 1.5 * t60  # (it was 1.2-1.45: capped by what the tips hold)


def _deposit_rate(path):
    """Paint laid per px of travel per px of width, from 60 to 120 px along."""
    c = Canvas(520, 260, seed=12)
    stroke = bm.Stroke(
        path=path,
        paint=P.CADMIUM_RED,
        amount=120,
        brush=brushes.FLAT_HOG.sized(36),
        seed=1,
    )
    run = bm._Run(c, stroke)
    run.pend = bm._pending(*run.bounds())
    arc, i, v0, a0 = 0.0, 1, 0.0, 0.0
    while arc < 120:
        if arc >= 60 and not a0:
            v0, a0 = float(c.thickness.sum()), arc
        got = run.step(i)
        assert got is not None
        bm._brush_step(run, *got)
        arc += got[0].ds
        i += 1
    return (float(c.thickness.sum()) - v0) / ((arc - a0) * 36)


def test_how_thick_a_load_goes_on_does_not_depend_on_the_step_size():
    # a curve is cut into much finer steps than a straight stroke
    straight = _deposit_rate([(40, 130), (480, 130)])
    curve = _deposit_rate([(40, 130), (150, 80), (260, 130), (370, 180), (480, 130)])
    assert 0.9 < curve / straight < 1.1  # (it was 1.42)


def test_a_loaded_brush_set_down_leaves_one_bead_behind_where_it_landed():
    for seed in (1, 2):
        c, _ = _stroke(amount=60, seed=seed)
        prof = c.thickness[YC - 10 : YC + 11, X0 - 20 : X0 + 46].astype(float).mean(0)
        body = float(c.thickness[YC - 10 : YC + 11, X0 + 60 : X0 + 120].mean())
        peaks = []
        for i in range(1, len(prof) - 1):
            if prof[i] > max(prof[i - 1], 1.3 * body) and prof[i] >= prof[i + 1]:
                lo = max(prof[max(i - 6, 0) : i].min(), prof[i + 1 : i + 7].min())
                if prof[i] - lo > 0.25 * body:
                    peaks.append(i - 20)
        # one bead (not a wall around the first footprint, echoed step after step)
        assert len(peaks) == 1
        # at the back of where the brush landed: its footprint reaches 4 px back
        assert -7 <= peaks[0] <= -1
        assert 1.3 * body < prof[peaks[0] + 20] < 3.0 * body
        # a bead, not a 1 px wall
        assert (prof > 0.5 * (body + prof.max())).sum() >= 3


def test_a_dab_keeps_a_rim_all_round():
    yy, xx = np.mgrid[0:120, 0:120]
    rr = np.hypot(yy + 0.5 - 60, xx + 0.5 - 60)
    ang = np.arctan2(yy + 0.5 - 60, xx + 0.5 - 60)
    ring = (rr > 12) & (rr < 17)
    even = []
    for seed in (1, 2, 3, 4):
        c = Canvas(120, 120, seed=12)
        c.stroke(
            [(60, 60)],
            P.CADMIUM_YELLOW,
            amount=40,
            brush=brushes.ROUND_HOG.sized(30),
            seed=seed,
        )
        h = c.thickness.astype(np.float64)
        sides = [
            h[ring & (abs(np.angle(np.exp(1j * (ang - a)))) < 0.8)].mean()
            for a in (0.0, 0.5 * np.pi, np.pi, -0.5 * np.pi)
        ]
        even.append(min(sides) / max(sides))
    # paint round its edge on every side (no bead left behind a dab), if not
    # evenly: it is pulled up into a peak off centre as the brush lifts
    assert np.mean(even) > 0.35


def _lane_ends(h, w=36):
    """Where each row of a horizontal stroke ends (px past the path's end)."""
    half = int(0.4 * w)
    ends = []
    for r in range(YC - half, YC + half + 1):
        body = h[r, X1 - 120 : X1 - 60].mean()
        x = X1 + 25
        while x > X0 and h[r, x] < 0.2 * body:
            x -= 1
        ends.append(x - X1)
    return np.array(ends)


def test_a_stiff_brush_lifts_off_clump_by_clump_and_a_soft_one_cleanly():
    hog = np.mean([_lane_ends(_stroke(seed=s)[0].thickness).std() for s in (1, 2)])
    sable = np.mean(
        [
            _lane_ends(
                _stroke(seed=s, bt=brushes.FLAT_SABLE.sized(36))[0].thickness
            ).std()
            for s in (1, 2)
        ]
    )
    clean = _lane_ends(_stroke(lift=0.0)[0].thickness).std()
    assert hog > 0.05 * 36  # a ragged end (it was a straight cut)
    assert sable < 0.5 and clean < 0.5


def test_lifting_off_pulls_thick_paint_up_into_a_lip():
    def lip(**kw):
        out = []
        for seed in (1, 2, 3):
            h = _stroke(amount=60, seed=seed, **kw)[0].thickness
            body = h[YC - 14 : YC + 15, X1 - 120 : X1 - 60].mean()
            last = h[YC - 14 : YC + 15, X1 - 12 : X1 + 8]
            out.append(np.percentile(last, 98) / body)
        return np.mean(out)

    lifted = lip()
    assert lifted > 1.3
    assert lifted > lip(lift=0.0) + 0.1


def test_lifting_off_leaves_a_short_dab_whole():
    # a short stroke of a loaded round: its clumps leave together (no claws)
    solid, total = 0, 0
    for seed in (20, 21, 22, 23):
        c = Canvas(120, 100, seed=3)
        c.stroke(
            [(45, 50), (60, 44)],
            P.CADMIUM_YELLOW,
            amount=40,
            brush=brushes.ROUND_HOG.sized(30),
            pressure=0.4,
            seed=seed,
        )
        covered = c.top_a > 0.5
        rows = np.nonzero(covered.any(1))[0]
        # each row of the mark is one solid run (no gaps between fingers)
        for r in rows[3:-3]:
            xs = np.nonzero(covered[r])[0]
            solid += xs.max() - xs.min() + 1 == len(xs)
            total += 1
    assert solid >= 0.95 * total


def test_a_loaded_dab_is_a_mound_not_a_coin():
    c = Canvas(120, 120, seed=12)
    brush = c.stroke(
        [(60, 60)], P.CADMIUM_YELLOW, amount=40, brush=brushes.ROUND_HOG.sized(30)
    )
    h = c.thickness.astype(np.float64)
    yy, xx = np.mgrid[0:120, 0:120]
    rr = np.hypot(yy + 0.5 - 60, xx + 0.5 - 60)
    rim = max(h[(rr >= k) & (rr < k + 1)].mean() for k in range(22))
    assert h[rr < 3].mean() > 0.8 * rim  # (it was ~0.4: a thin centre, a rim)
    loaded = 40 * 0.9 * 30**2
    assert h.sum() > 0.15 * loaded  # (it was 4%)
    assert h.sum() + brush.volume == pytest.approx(loaded, rel=1e-6)


def test_a_thin_dab_is_unchanged_by_the_mound():
    # below what the hair holds there is no glob to press out
    c = Canvas(120, 120, seed=12)
    c.stroke([(60, 60)], P.CADMIUM_YELLOW, amount=4, brush=brushes.ROUND_HOG.sized(30))
    h = c.thickness
    assert h.max() < 3.0


def test_a_knife_presses_paint_into_ridges_at_its_blade_ends():
    sky = P.TITANIUM_WHITE
    for p in (0.4, 0.7):
        c = Canvas(400, 120, seed=21)
        c.stroke(
            [(70, 60), (330, 60)],
            sky,
            amount=70,
            brush=brushes.PALETTE_KNIFE.sized(56),
            pressure=p,
            seed=2,
        )
        prof = c.thickness[:, 120:280].astype(np.float64).mean(1)
        plateau = prof[46:75].mean()
        ends = max(prof[30:46].max(), prof[75:91].max())
        assert ends > 1.15 * plateau  # (it was flat: 1.05)


def test_a_brush_s_edge_ridge_is_as_wide_as_the_paint_in_it():
    # paint pushed out at the side of a step of a 36 px hog flat, laid as a
    # ridge: twice the paint, a ridge wider (and taller), not just taller
    from oilpaint import _engine as eng

    def ridge(volume):
        c = Canvas(470, 120, seed=12)
        stroke = bm.Stroke(
            path=[(X0, YC), (X1, YC)],
            paint=P.CADMIUM_RED,
            amount=60,
            brush=brushes.FLAT_HOG.sized(36),
            seed=1,
        )
        run = bm._Run(c, stroke)
        got = run.step(20)
        assert got is not None
        st, tables = got
        side = np.zeros(eng.PACKED)
        side[eng.IV] = volume
        side[eng.IS] = volume
        ridges = (np.zeros(eng.PACKED), side, np.zeros(eng.PACKED))
        eng.lay_ridges(run.cv, run.pend, run.br, run.K, st, ridges, *tables, run.tex2)
        prof = c.thickness.astype(np.float64)[:, int(st.cx) - 6 : int(st.cx)].mean(1)
        return int((prof > 0.2 * prof.max()).sum()), float(prof.max())

    w1, h1 = ridge(20.0)
    w2, h2 = ridge(160.0)
    assert w2 >= w1 + 2
    assert h2 > 2.0 * h1


def test_tube_paint_settles_softer_than_paint_stiffened_with_impasto_medium():
    from oilpaint import Paint

    def rms(paint):
        out = []
        for seed in (1, 2):
            c, _ = _stroke(amount=40, seed=seed, paint=paint)
            out.append(np.sqrt((_furrows(c.thickness) ** 2).mean()))
        return np.mean(out)

    tube = P.CADMIUM_ORANGE
    stiff = Paint.mix((tube, 2), (P.IMPASTO_MEDIUM, 1))
    assert rms(tube) < 0.9 * rms(stiff)  # (they were within 4%)


def _lift_length(w, seed):
    """Over how many px the clumps of a hog flat w px wide leave the paint."""
    c = Canvas(560, 200, seed=12)
    stroke = bm.Stroke(
        path=[(60, 100), (460, 100)],
        paint=P.CADMIUM_RED,
        amount=60,
        brush=brushes.FLAT_HOG.sized(w),
        seed=seed,
    )
    run = bm._Run(c, stroke)
    run.pend = bm._pending(*run.bounds())
    last = 0.0
    for i in range(1, len(run.pts)):
        got = run.step(i)
        assert got is not None
        last = got[0].lift_len or last
        bm._brush_step(run, *got)
    return last / run.lift_var


def test_lifting_off_takes_about_as_long_on_a_wide_brush():
    # the bristles are not much longer on a wide brush: the ragged end isn't
    # twice as long on a brush twice as wide
    assert _lift_length(80, 1) < 1.4 * _lift_length(36, 1)


def test_each_stroke_ends_its_own_way():
    ends = [_lane_ends(_stroke(seed=s)[0].thickness).std() for s in (1, 2, 3, 4)]
    assert max(ends) > 1.3 * min(ends)


def test_no_lane_of_hair_is_narrower_than_a_couple_of_pixels():
    # (narrower ones are lost in the pixels; across the pixel grid on a slant
    # they would step from pixel to pixel, a row of beads along each crest)
    for seed in range(3):
        f, (_, _, _, dx) = bm._hair_field(
            brushes.FLAT_HOG.sized(36),
            36,
            18.0,
            -20.0,
            300.0,
            np.random.default_rng(seed),
        )
        spec = (np.abs(np.fft.rfft(f - f.mean(1, keepdims=True), axis=1)) ** 2).mean(0)
        freq = np.fft.rfftfreq(f.shape[1], d=dx)  # cycles per px
        assert spec[freq > 1 / 2.5].sum() < 0.03 * spec[1:].sum()


def _skim(pressure, load=0.22):
    """Paint a small knife skimming dry ridges lays (mean of three strokes)."""
    from oilpaint import Brush

    laid = []
    for seed in (1, 2, 3):
        c = Canvas(400, 200, seed=7)
        for y in range(40, 170, 12):
            c.stroke(
                [(20, y), (380, y)],
                P.IVORY_BLACK,
                amount=30,
                brush=brushes.FLAT_HOG.sized(24),
                seed=seed,
            )
        c.dry()
        before = float(c.thickness.astype(np.float64).sum())
        knife = Brush(brushes.knife("straight", 40, flex=0.5)).load(
            P.TITANIUM_WHITE, load
        )
        c.stroke(
            [(60, 60), (120, 100), (180, 140)],
            brush=knife,
            pressure=pressure,
            seed=seed,
            angle=-40,
            follow_path=False,
            lift=0.0,
        )
        laid.append(float(c.thickness.astype(np.float64).sum()) - before)
    return float(np.mean(laid))


def test_a_skimming_knife_lays_what_it_did():
    # barely touching, the blade's bead leaves a little paint on the high
    # points: on the same dried ridges within about 10% of what the engine laid
    # before the sawtooth fix; these ridges (painted by this engine) stand a
    # little higher than they did, so it lays some 10-17% more here than the
    # old engine did on its own (167, 194 and 231 units). It was 50% more.
    assert 155 < _skim([0.1, 0.05, 0.02]) < 225
    assert 185 < _skim(0.1) < 255
    assert 215 < _skim(0.3) < 285


def _volume(c):
    return float(np.asarray(c.top_v + c.und_v, np.float64).sum())


def test_rounding_off_pressing_out_and_pulling_up_keep_the_paint():
    from oilpaint import Brush

    kw: dict
    for bt, path, kw in (
        # a thick tube stroke that rounds off as it settles (soften)
        (brushes.FLAT_HOG.sized(36), [(60, 60), (410, 60)], {}),
        # a knife pressing paint out at its blade's ends
        (brushes.PALETTE_KNIFE.sized(56), [(70, 60), (330, 60)], {"pressure": 0.6}),
        # a loaded dab pulled up into a peak
        (brushes.ROUND_HOG.sized(30), [(200, 60)], {}),
    ):
        c = Canvas(470, 120, seed=3)
        brush = Brush(bt).load(P.CADMIUM_ORANGE, 60, width=bt.width)
        loaded = brush.volume
        c.stroke(path, brush=brush, seed=5, pressure=kw.get("pressure", 0.5))
        assert _volume(c) + brush.volume == pytest.approx(loaded, rel=1e-5)
        assert (c.top_v >= -1e-4).all() and (c.und_v >= -1e-4).all()


def test_a_dry_brush_tail_keeps_its_length():
    # the green stroke of the strokes demo sheet: a round dragged on until it
    # runs dry and breaks up (its tail ended at x = 424 before the relief work)
    from oilpaint.spline import Bezier

    c = Canvas(900, 520, seed=2)
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
    cols = np.nonzero((c.top_a[270:430] > 0.3).sum(0) > 3)[0]
    assert 405 < cols.max() < 460


def test_how_each_kind_of_tool_lifts_off_by_default():
    names = ("FLAT_HOG", "ROUND_HOG", "BRIGHT", "FLAT_SABLE", "ROUND_SABLE", "MOP")
    lift = {name: bm._physics(getattr(brushes, name))["lift"] for name in names}
    lift["knife"] = bm._physics(brushes.PALETTE_KNIFE)["lift"]
    assert min(lift["FLAT_HOG"], lift["ROUND_HOG"], lift["BRIGHT"]) > 0.9
    assert lift["FLAT_SABLE"] == lift["ROUND_SABLE"] == lift["MOP"] == 0.0
    assert lift["knife"] == 0.3


def test_an_overloaded_liner_carries_its_line_far():
    # (a liner loaded with 200 widths of paint drew a line 375 px long; the glob
    # now feeds it directly and it goes a little further)
    reach = []
    for seed in (1, 2, 3):
        c = Canvas(700, 60, seed=seed)
        c.stroke(
            [(20, 30), (680, 30)],
            P.IVORY_BLACK,
            amount=200,
            brush=brushes.LINER,
            seed=seed,
        )
        cols = np.nonzero(c.top_a.sum(0) > 1.5)[0]
        reach.append(cols.max() - cols.min())
    assert 375 < np.mean(reach) < 440


def test_the_random_hair_and_lift_off_come_from_the_stroke_s_seed():
    def run(seed):
        stroke = bm.Stroke(
            path=[(X0, YC), (X1, YC)],
            paint=P.CADMIUM_RED,
            amount=60,
            brush=brushes.FLAT_HOG.sized(36),
            seed=seed,
        )
        return bm._Run(Canvas(470, 120, seed=12), stroke)

    a, b, other = run(5), run(5), run(6)
    assert np.array_equal(a.tex2, b.tex2) and a.lift_var == b.lift_var
    assert not np.array_equal(a.tex2, other.tex2)
    c1, _ = _stroke(seed=5)
    c2, _ = _stroke(seed=5)
    assert np.array_equal(c1.top, c2.top) and np.array_equal(c1.und, c2.und)


def test_an_explicit_lift_acts_on_soft_brushes_too():
    # soft hair slides off clean by default; told to lift, it pulls up its end
    # on a stroke as on a dab
    def ends(lift, path):
        c = Canvas(300, 200, seed=4)
        c.stroke(
            path,
            P.CADMIUM_RED,
            amount=40,
            brush=brushes.MOP.sized(60),
            seed=2,
            lift=lift,
        )
        return c.thickness.astype(np.float64)

    for path in ([(40, 100), (260, 100)], [(150, 100)]):
        assert np.abs(ends(1.0, path) - ends(0.0, path)).sum() > 1.0
        assert np.array_equal(ends(None, path), ends(0.0, path))


def test_a_very_long_stroke_keeps_its_hair_pattern_fine_enough():
    # a 50,000 px stroke: rows of the hair pattern at most a quarter of its
    # shortest drift cell apart (12 px), not stretched to keep their number down
    f, (_, _, ds, _) = bm._hair_field(
        brushes.ROUND.sized(10), 10, 5.0, -10.0, 50000.0, np.random.default_rng(1)
    )
    assert ds <= 3.0 and f.shape[0] * ds >= 50000
