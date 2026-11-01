"""Paints: absorption + scattering spectra per unit volume."""

from __future__ import annotations

from typing import Iterable, Sequence, Tuple, Union

import numpy as np

from . import spectral as sp

#: Scattering per unit thickness of a fully opaque white paint. A unit-thick
#: layer of it over black reflects ~92%: a normal stroke covers well but a thin
#: scumble lets the underlayer show through.
SCATTER_MAX = 12.0


def _scatter_factor(luminance: float) -> float:
    # Light pigments (titanium white, cadmium yellow) scatter strongly, dark
    # ones (phthalo, black) mostly absorb. Without this, white is far too weak
    # in tints compared with real paint.
    return 0.1 + 0.9 * luminance


ColorLike = Union[str, Sequence[float], "Paint"]

#: Body (stiffness) of paint straight from the tube.
DEFAULT_BODY = 0.7


def _check_body(body) -> float:
    body = float(body)
    if not 0.0 <= body <= 1.0:
        raise ValueError("body must be between 0 (fluid) and 1 (very stiff)")
    return body


class Paint:
    """A paint, described by how it looks and how it behaves.

    Parameters
    ----------
    color:
        How the paint looks brushed out at normal thickness on white canvas.
        ``"#rrggbb"``, a float triple in 0..1, or an int triple in 0..255.
    opacity:
        0..1. How strongly it hides what is underneath. 1 = opaque body
        color (cadmiums, titanium white), ~0.3 = glazing color (alizarin,
        phthalo), 0 = pure dye.
    strength:
        Tinting strength. 2 means half as much of this paint gives the same
        shift in a mixture (phthalo blue is a famously strong tinter).
    name:
        Optional label.
    body:
        0..1: how stiff the paint is. Tube oil paint (~0.7) holds the marks
        of the brush or knife: ridges, peaks, bristle grooves. Paint thinned
        with medium (low body) flows and levels out smooth; impasto medium or
        cold wax (body 1) holds sharp peaks. Stiff paint also stacks up
        thicker when a brush is overloaded.

    Internally a paint is a Kubelka-Munk pigment: an absorption spectrum
    ``K`` and a scattering coefficient ``S`` per unit volume. Mixing paints
    adds those, weighted by volume.
    """

    __slots__ = ("K", "S", "name", "_color", "body")

    def __init__(self, color: ColorLike, opacity: float = 1.0, strength: float = 1.0,
                 name: str | None = None, body: float | None = None):
        if isinstance(color, Paint):
            self.K, self.S = color.K.copy(), color.S
            self.name = name or color.name
            self._color = color._color
            self.body = color.body if body is None else _check_body(body)
            return
        self.body = DEFAULT_BODY if body is None else _check_body(body)
        srgb = sp.parse_color(color)
        target = sp.linear_rgb_to_reflectance(sp.srgb_to_linear(srgb))
        lum = float(sp.LUMINANCE @ sp.srgb_to_linear(srgb))
        opacity = float(np.clip(opacity, 0.0, 1.0))
        S = SCATTER_MAX * _scatter_factor(lum) * opacity ** 2
        K = sp.solve_absorption(target, S)
        self.K = K * strength
        self.S = S * strength
        self.name = name
        self._color = srgb

    # ------------------------------------------------------------------
    @classmethod
    def from_optics(cls, K, S, name: str | None = None, body: float = None) -> "Paint":
        """Build a paint directly from absorption spectrum K (per band, sampled
        at :data:`oilpaint.spectral.WAVELENGTHS`) and scattering S (scalar),
        both per unit volume. Handy for measured pigment data."""
        p = cls.__new__(cls)
        p.K = np.asarray(K, dtype=np.float64).copy()
        p.S = float(S)
        p.name = name
        p._color = None
        p.body = DEFAULT_BODY if body is None else _check_body(body)
        return p

    def with_body(self, body: float) -> "Paint":
        """The same paint, stiffer or more fluid."""
        return Paint(self, body=body)

    @classmethod
    def mix(cls, *parts: Union["Paint", Tuple[ColorLike, float]], name: str | None = None) -> "Paint":
        """Physically mix paints on the palette.

        ``Paint.mix((ultramarine, 1), (titanium_white, 3))`` mixes one part
        ultramarine with three parts white, by volume. Bare paints count as one
        part each.
        """
        K = np.zeros(sp.N_BANDS)
        S = 0.0
        body = 0.0
        total = 0.0
        for part in parts:
            paint, vol = (part, 1.0) if isinstance(part, Paint) else part
            paint = as_paint(paint)
            K += vol * paint.K
            S += vol * paint.S
            body += vol * paint.body
            total += vol
        if total <= 0:
            raise ValueError("mix needs a positive total volume")
        return cls.from_optics(K / total, S / total, name=name, body=body / total)

    # ------------------------------------------------------------------
    def reflectance(self, thickness: float = 1.0, over: ColorLike = "#ffffff") -> np.ndarray:
        """Reflectance spectrum of a layer of this paint over a ground color
        (any color form, or a reflectance spectrum)."""
        is_spectrum = isinstance(over, np.ndarray) and over.shape[-1] == sp.N_BANDS
        ground = over if is_spectrum else sp.srgb_to_reflectance(over)
        return sp.km_layer(self.K * thickness, self.S * thickness, ground)

    def appearance(self, thickness: float = 1.0, over: ColorLike = "#ffffff") -> np.ndarray:
        """sRGB color (0..1) of a layer of this paint over a ground color."""
        return sp.reflectance_to_srgb(self.reflectance(thickness, over))

    @property
    def color(self) -> np.ndarray:
        """sRGB color at unit thickness on white."""
        if self._color is None:
            self._color = self.appearance()
        return self._color

    @property
    def masstone(self) -> np.ndarray:
        """sRGB color of the paint straight from the tube, infinitely thick."""
        return sp.reflectance_to_srgb(sp.km_opaque(self.K, self.S))

    @property
    def hex(self) -> str:
        return sp.to_hex(self.color)

    def __repr__(self) -> str:
        label = f"{self.name!r}, " if self.name else ""
        return f"Paint({label}{self.hex}, S={self.S:.2f})"


def as_paint(p: ColorLike) -> Paint:
    return p if isinstance(p, Paint) else _cached_paint(p)


_CACHE: dict = {}


def _cached_paint(color) -> Paint:
    key = sp.to_hex(sp.parse_color(color))
    paint = _CACHE.get(key)
    if paint is None:
        paint = _CACHE[key] = Paint(color)
    return paint


def mix_colors(colors: Iterable[ColorLike], weights: Iterable[float] | None = None) -> str:
    """Quick helper: hex color of a palette mix of plain colors / paints."""
    colors = list(colors)
    weights = [1.0] * len(colors) if weights is None else list(weights)
    return Paint.mix(*[(as_paint(c), w) for c, w in zip(colors, weights)]).hex
