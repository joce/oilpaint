"""Compiled helpers that stand in for slower code: they must give the same numbers."""

import numpy as np


def test_compiled_max_filter_is_the_max_filter():
    from oilpaint import _engine as eng
    from oilpaint._filters import _max_filter

    rng = np.random.default_rng(3)
    for k in range(40):
        h, w = (int(v) for v in rng.integers(1, 40, 2))
        a = rng.normal(size=(h, w)).astype((np.float32, np.float64)[k % 2])
        r = int(rng.integers(0, 10))
        want, got = _max_filter(a, r), eng.max_filter(a, r)
        assert got.dtype == want.dtype == np.float32
        assert np.array_equal(got, want)


def test_box_blur_of_some_rows_is_the_box_blur_there():
    from oilpaint import _engine as eng

    rng = np.random.default_rng(4)
    for r in (1, 3):
        a = rng.random((30, 41))
        lo = rng.integers(0, 41, 30)
        hi = np.minimum(lo + rng.integers(0, 20, 30), 41)
        out = np.full(a.shape, -1.0)
        eng.box_blur_rows(a, r, lo, hi, out)
        full = eng.box_blur(a, r)
        for y in range(30):
            assert np.array_equal(out[y, lo[y] : hi[y]], full[y, lo[y] : hi[y]])
            assert (out[y, : lo[y]] == -1.0).all() and (out[y, hi[y] :] == -1.0).all()


def test_the_dab_pattern_spans_what_the_engine_reads():
    from oilpaint import _dab
    from oilpaint import _engine as eng

    assert _dab.SPAN == eng.DAB_SPAN


def test_the_renderer_reads_paint_where_the_engine_packs_it():
    from oilpaint import _engine as eng
    from oilpaint import _render

    assert (_render.N, _render.IS, _render.IV) == (eng.N, eng.IS, eng.IV)


def test_dab_patterns_are_laid_out_and_turned_as_before():
    from oilpaint import _dab

    def reference(n, spacing, rng):  # the pattern, in numpy, cone by cone
        X, Y = _dab._coords(n)
        rows = np.arange(-1.2, 1.2 + spacing, 0.866 * spacing)
        pts = [
            (x, y)
            for j, y in enumerate(rows)
            for x in np.arange(
                -1.2 + (0.5 * spacing if j % 2 else 0.0), 1.2 + spacing, spacing
            )
        ]
        pts = np.asarray(pts) + rng.uniform(-0.35, 0.35, (len(pts), 2)) * spacing
        heights = rng.uniform(0.25, 1.0, len(pts)) ** 0.6
        radii = 0.62 * spacing * rng.uniform(0.75, 1.25, len(pts))
        out = np.zeros((n, n))
        cell = 2 * _dab.SPAN / n
        for (px, py), h, rad in zip(pts, heights, radii, strict=True):
            x0 = max(int((px - rad + _dab.SPAN) / cell), 0)
            x1 = min(int((px + rad + _dab.SPAN) / cell) + 2, n)
            y0 = max(int((py - rad + _dab.SPAN) / cell), 0)
            y1 = min(int((py + rad + _dab.SPAN) / cell) + 2, n)
            if x0 >= x1 or y0 >= y1:
                continue
            d = np.hypot(X[y0:y1, x0:x1] - px, Y[y0:y1, x0:x1] - py)
            cone = h * np.clip(1.0 - d / rad, 0.0, 1.0) ** 0.6
            np.maximum(out[y0:y1, x0:x1], cone, out=out[y0:y1, x0:x1])
        return out

    for n, spacing in ((96, 0.3), (130, 0.12), (200, 0.7)):
        want = reference(n, spacing, np.random.default_rng(n))
        got = _dab._bristle_tips(n, spacing, np.random.default_rng(n))
        assert np.array_equal(got, want)
    base = _dab._layout(7, 10.0, 0, 0.5)
    n = base.shape[0]
    X, Y = _dab._coords(n)
    ca, sa = np.cos(1.1), np.sin(1.1)
    ix = ((ca * X + sa * Y + _dab.SPAN) / (2 * _dab.SPAN) * n).astype(int)
    iy = ((-sa * X + ca * Y + _dab.SPAN) / (2 * _dab.SPAN) * n).astype(int)
    ok = (ix >= 0) & (ix < n) & (iy >= 0) & (iy < n)
    want = np.zeros((n, n))
    want[ok] = base[iy[ok], ix[ok]]
    assert np.array_equal(_dab._turned(base, _dab._ticks(n), ca, sa), want)
