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
