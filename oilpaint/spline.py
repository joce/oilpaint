"""Stroke paths: smooth curves resampled by arc length."""

from __future__ import annotations

from typing import Callable, Sequence, Union

import numpy as np

Profile = Union[float, Sequence[float], Callable[[np.ndarray], np.ndarray]]


class Path:
    """Base class. Subclasses implement :meth:`_dense` returning densely sampled
    points plus, for each point, a fractional knot index (0 at the first
    control point, 1 at the second, ...)."""

    n_knots: int = 0

    def _dense(self) -> tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError

    def sample(self, spacing: float):
        """Resample at (roughly) equal arc-length ``spacing`` in pixels.

        Returns ``(points (M,2), tangents (M,2), u (M,), knot (M,))`` where
        ``u`` runs 0..1 along the arc length and ``knot`` is the fractional
        control-point index, used to interpolate per-control-point values.
        """
        pts, knot = self._dense()
        seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
        s = np.concatenate([[0.0], np.cumsum(seg)])
        length = s[-1]
        if length < 1e-9:
            p = pts[:1]
            return p, np.array([[1.0, 0.0]]), np.zeros(1), knot[:1]
        n = max(2, int(np.ceil(length / max(spacing, 1e-3))) + 1)
        si = np.linspace(0.0, length, n)
        out = np.stack([np.interp(si, s, pts[:, 0]), np.interp(si, s, pts[:, 1])], -1)
        k = np.interp(si, s, knot)
        tan = np.gradient(out, axis=0)
        tan /= np.maximum(np.linalg.norm(tan, axis=1, keepdims=True), 1e-12)
        return out, tan, si / length, k

    @property
    def length(self) -> float:
        pts, _ = self._dense()
        return float(np.linalg.norm(np.diff(pts, axis=0), axis=1).sum())


class CatmullRom(Path):
    """Smooth curve passing through every control point.

    ``alpha=0.5`` (centripetal) avoids loops and cusps on uneven spacing.
    """

    def __init__(self, points, alpha: float = 0.5, closed: bool = False):
        self.points = np.asarray(points, dtype=np.float64).reshape(-1, 2)
        self.alpha = alpha
        self.closed = closed
        self.n_knots = len(self.points) + (1 if closed else 0)

    def _dense(self):
        P = self.points
        if len(P) == 1:
            return P.copy(), np.zeros(1)
        if self.closed:
            P = np.vstack([P[-1], P, P[0], P[1]])
        else:  # reflect end points to get natural end tangents
            P = np.vstack([2 * P[0] - P[1], P, 2 * P[-1] - P[-2]])
        # all segments at once: segment i runs from P[i + 1] to P[i + 2]
        p0, p1, p2, p3 = P[:-3], P[1:-2], P[2:-1], P[3:]
        m = np.maximum(8, (np.linalg.norm(p2 - p1, axis=1) / 2).astype(int) + 1)    # samples per segment
        seg = np.repeat(np.arange(len(m)), m)
        t = (np.arange(m.sum()) - np.repeat(np.cumsum(m) - m, m)) * (1.0 / m[seg])    # 0..1 within each
        pts = _catmull_segment(p0[seg], p1[seg], p2[seg], p3[seg], t, self.alpha)
        return np.vstack([pts, P[-2][None]]), np.concatenate([seg + t, [float(len(P) - 3)]])


def _catmull_segment(p0, p1, p2, p3, t, alpha):
    """Points at parameter t (0..1) on the Catmull-Rom segments from p1 to p2
    (one row of p0..p3 per value of t)."""
    def tj(ti, a, b):
        return ti + np.maximum(np.linalg.norm(b - a, axis=1), 1e-6) ** alpha

    t0 = 0.0
    t1 = tj(t0, p0, p1)
    t2 = tj(t1, p1, p2)
    t3 = tj(t2, p2, p3)
    tt = t1 + (t2 - t1) * t
    t1, t2, t3, tt = t1[:, None], t2[:, None], t3[:, None], tt[:, None]
    a1 = (t1 - tt) / (t1 - t0) * p0 + (tt - t0) / (t1 - t0) * p1
    a2 = (t2 - tt) / (t2 - t1) * p1 + (tt - t1) / (t2 - t1) * p2
    a3 = (t3 - tt) / (t3 - t2) * p2 + (tt - t2) / (t3 - t2) * p3
    b1 = (t2 - tt) / (t2 - t0) * a1 + (tt - t0) / (t2 - t0) * a2
    b2 = (t3 - tt) / (t3 - t1) * a2 + (tt - t1) / (t3 - t1) * a3
    return (t2 - tt) / (t2 - t1) * b1 + (tt - t1) / (t2 - t1) * b2


class Bezier(Path):
    """Piecewise cubic Bezier: ``[anchor, handle, handle, anchor, handle, handle, anchor, ...]``
    (3n + 1 points). Knots are the anchors."""

    def __init__(self, points):
        P = np.asarray(points, dtype=np.float64).reshape(-1, 2)
        if (len(P) - 1) % 3 != 0 or len(P) < 4:
            raise ValueError("Bezier needs 3n+1 points: anchor, handle, handle, anchor, ...")
        self.points = P
        self.n_knots = (len(P) - 1) // 3 + 1

    def _dense(self):
        pts, knots = [], []
        P = self.points
        for i in range(self.n_knots - 1):
            p0, p1, p2, p3 = P[3 * i:3 * i + 4]
            approx = np.linalg.norm(p1 - p0) + np.linalg.norm(p2 - p1) + np.linalg.norm(p3 - p2)
            t = np.linspace(0.0, 1.0, max(8, int(approx / 2) + 1), endpoint=False)[:, None]
            mt = 1 - t
            pts.append(mt ** 3 * p0 + 3 * mt * mt * t * p1 + 3 * mt * t * t * p2 + t ** 3 * p3)
            knots.append(i + t[:, 0])
        pts.append(P[-1][None])
        knots.append(np.array([float(self.n_knots - 1)]))
        return np.vstack(pts), np.concatenate(knots)


class Polyline(Path):
    """Straight segments between points."""

    def __init__(self, points):
        self.points = np.asarray(points, dtype=np.float64).reshape(-1, 2)
        self.n_knots = len(self.points)

    def _dense(self):
        return self.points.copy(), np.arange(len(self.points), dtype=np.float64)


def as_path(path) -> Path:
    if isinstance(path, Path):
        return path
    return CatmullRom(path)


def evaluate_profile(profile: Profile, u: np.ndarray, knot: np.ndarray, n_knots: int) -> np.ndarray:
    """Evaluate a per-stroke value (width, pressure, ...) along the samples.

    * scalar -> constant
    * sequence with one value per control point -> interpolated between knots
    * any other sequence -> spread evenly along the stroke length
    * callable f(u) with u in 0..1 along the stroke
    """
    if callable(profile):
        return np.asarray(profile(u), dtype=np.float64) * np.ones_like(u)
    arr = np.asarray(profile, dtype=np.float64)
    if arr.ndim == 0:
        return np.full_like(u, float(arr))
    if arr.size == 1:
        return np.full_like(u, float(arr[0]))
    if arr.size == n_knots:
        return np.interp(knot, np.arange(n_knots), arr)
    return np.interp(u, np.linspace(0, 1, arr.size), arr)
