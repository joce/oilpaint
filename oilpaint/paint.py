"""Paints: absorption + scattering spectra per unit volume."""

from __future__ import annotations

import functools
import math
from collections.abc import Iterable, Sequence

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


#: Body (stiffness) of paint straight from the tube.
DEFAULT_BODY = 0.7


def _check_body(body) -> float:
    body = float(body)
    if not 0.0 <= body <= 1.0:
        raise ValueError("body must be between 0 (fluid) and 1 (very stiff)")
    return body


def _check_strength(strength) -> float:
    strength = float(strength)
    if not (math.isfinite(strength) and strength > 0.0):
        raise ValueError(
            f"strength must be a positive number (1 = normal), got {strength!r}"
        )
    return strength


def _check_opacity(opacity) -> float:
    opacity = float(opacity)
    if not math.isfinite(opacity):
        raise ValueError(f"opacity must be a number in 0..1, got {opacity!r}")
    # values outside 0..1 have always been clipped rather than refused
    return min(max(opacity, 0.0), 1.0)


class Paint:
    """A paint, described by how it looks and how it behaves.

    Parameters
    ----------
    color:
        How the paint looks brushed out at normal thickness on white canvas.
        ``"#rrggbb"``, a float triple in 0..1, or an int triple in 0..255 (a
        triple with a component of 2 or more is on the 0..255 scale; one
        whose largest is between 1 and 2 is refused: see
        :func:`oilpaint.spectral.parse_color`).
    opacity:
        0..1. How strongly it hides what is underneath. 1 = opaque body
        color (cadmiums, titanium white), ~0.3 = glazing color (alizarin,
        phthalo), 0 = pure dye. Default 1.
    strength:
        Tinting strength, > 0. 2 means half as much of this paint gives the
        same shift in a mixture (phthalo blue is a famously strong tinter).
        Default 1.
    name:
        Optional label.
    body:
        0..1: how stiff the paint is. Tube oil paint (~0.7) holds the marks
        of the brush or knife: ridges, peaks, bristle grooves. Paint thinned
        with medium (low body) flows and levels out smooth; impasto medium or
        cold wax (body 1) holds sharp peaks. Stiff paint also stacks up
        thicker when a brush is overloaded.

    ``color`` may also be an existing paint: ``Paint(paint, ...)`` is a copy,
    with a new ``name`` or ``body`` if given. ``strength`` then multiplies
    that paint's own (``Paint(PHTHALO_BLUE, strength=0.5)`` is a phthalo half
    as strong), and ``opacity`` re-derives it from its colour with the new
    opacity, keeping its strength. A paint mixed on the palette or built from
    optics has no colour to re-derive from, so it refuses ``opacity``: mix
    it with :data:`~oilpaint.palette.MEDIUM` to make it more transparent.

    Internally a paint is a Kubelka-Munk pigment: an absorption spectrum
    ``K`` and a scattering coefficient ``S`` per unit volume. Mixing paints
    adds those, weighted by volume.
    """

    __slots__ = ("K", "S", "_color", "_strength", "body", "name")

    def __setstate__(self, state) -> None:
        # Paints pickled before ``_strength`` existed lack it: they are read as
        # paints with no colour to derive them again from.
        self._strength = None
        attributes, slots = state if isinstance(state, tuple) else (state, None)
        for mapping in (attributes, slots):
            for key, value in (mapping or {}).items():
                setattr(self, key, value)

    def __init__(
        self,
        color: ColorLike,
        opacity: float | None = None,
        strength: float | None = None,
        name: str | None = None,
        body: float | None = None,
    ):
        if isinstance(color, Paint):
            self._derive(color, opacity, strength, name, body)
            return
        self.body = DEFAULT_BODY if body is None else _check_body(body)
        self.name = name
        self._from_color(
            color,
            1.0 if opacity is None else _check_opacity(opacity),
            1.0 if strength is None else _check_strength(strength),
        )

    def _from_color(self, color: ColorLike, opacity: float, strength: float) -> None:
        srgb = sp.parse_color(color)
        srgb.flags.writeable = False
        target = sp.linear_rgb_to_reflectance(sp.srgb_to_linear(srgb))
        lum = float(sp.LUMINANCE @ sp.srgb_to_linear(srgb))
        S = SCATTER_MAX * _scatter_factor(lum) * opacity**2
        K = sp.solve_absorption(target, S)
        self.K = K * strength
        self.S = S * strength
        self._color = srgb
        # kept so that Paint(this, opacity=...) can derive it again
        self._strength: float | None = strength

    def _derive(self, paint: Paint, opacity, strength, name, body) -> None:
        """``Paint(paint, ...)``: a copy of ``paint``, changed as asked."""
        self.name = name or paint.name
        self.body = paint.body if body is None else _check_body(body)
        factor = 1.0 if strength is None else _check_strength(strength)
        if opacity is not None:
            if paint._strength is None or paint._color is None:
                raise ValueError(
                    "opacity can only be changed on a paint made from a colour; "
                    f"{paint!r} was mixed, built from optics or unpickled from "
                    "an older version (mix it with palette.MEDIUM to make it "
                    "more transparent)"
                )
            self._from_color(
                paint._color, _check_opacity(opacity), paint._strength * factor
            )
            return
        self.K, self.S = paint.K.copy(), paint.S
        self._color, self._strength = paint._color, paint._strength
        if strength is not None:
            # as in Paint(color, strength=...): absorption and scattering
            # scale together
            self.K, self.S = self.K * factor, self.S * factor
            if self._strength is not None:
                # made from a colour: that colour stays its colour, as for
                # Paint(color, strength=...)
                self._strength *= factor
            else:
                # its colour is how it looks, which the strength changes:
                # worked out again when asked for
                self._color = None

    # ------------------------------------------------------------------
    @classmethod
    def from_optics(
        cls, K, S, name: str | None = None, body: float | None = None
    ) -> Paint:
        """Build a paint directly from absorption spectrum K (per band, sampled
        at :data:`oilpaint.spectral.WAVELENGTHS`) and scattering S (scalar),
        both per unit volume. Handy for measured pigment data."""
        p = cls.__new__(cls)
        p.K = np.asarray(K, dtype=np.float64).copy()
        p.S = float(S)
        p.name = name
        p._color = None
        p._strength = None
        p.body = DEFAULT_BODY if body is None else _check_body(body)
        return p

    def with_body(self, body: float) -> Paint:
        """The same paint, stiffer or more fluid."""
        return Paint(self, body=body)

    @classmethod
    def mix(
        cls,
        *parts: ColorLike | tuple[ColorLike, float],
        name: str | None = None,
    ) -> Paint:
        """Physically mix paints on the palette.

        ``Paint.mix((ultramarine, 1), (titanium_white, 3))`` mixes one part
        ultramarine with three parts white, by volume. Bare paints (or
        colours) count as one part each. Parts must not be negative.
        """
        K = np.zeros(sp.N_BANDS)
        S = 0.0
        body = 0.0
        total = 0.0
        for part in parts:
            paint, vol = _split_part(part)
            paint = as_paint(paint)
            K += vol * paint.K
            S += vol * paint.S
            body += vol * paint.body
            total += vol
        if total <= 0:
            raise ValueError("mix needs a positive total volume")
        return cls.from_optics(K / total, S / total, name=name, body=body / total)

    # ------------------------------------------------------------------
    def reflectance(
        self, thickness: float = 1.0, over: ColorLike = "#ffffff"
    ) -> np.ndarray:
        """Reflectance spectrum of a layer of this paint over a ground color
        (any color form, or a reflectance spectrum)."""
        is_spectrum = isinstance(over, np.ndarray) and over.shape[-1] == sp.N_BANDS
        ground = over if is_spectrum else sp.srgb_to_reflectance(over)
        return sp.km_layer(self.K * thickness, self.S * thickness, ground)

    def appearance(
        self, thickness: float = 1.0, over: ColorLike = "#ffffff"
    ) -> np.ndarray:
        """sRGB color (0..1) of a layer of this paint over a ground color."""
        return sp.reflectance_to_srgb(self.reflectance(thickness, over))

    @property
    def color(self) -> np.ndarray:
        """sRGB color at unit thickness on white."""
        if self._color is None:
            self._color = self.appearance()
            self._color.flags.writeable = False
        # read-only: the paint's optics were derived from it
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


# (Defined after Paint so that it holds the class itself, not the name
# "Paint": a name would be looked up in whichever module's hints use the
# alias, and fail there unless that module imported Paint too.)
ColorLike = str | Sequence[float] | np.ndarray | Paint


def _split_part(part) -> tuple[ColorLike, float]:
    """A part of a :meth:`Paint.mix`: ``(paint or colour, parts)``, or a bare
    paint or colour for one part."""
    is_pair = (
        isinstance(part, (tuple, list))
        and len(part) == 2
        and not isinstance(part[0], (int, float, np.number))
    )
    if not is_pair:
        return part, 1.0
    paint, vol = part
    try:
        vol = float(vol)
    except (TypeError, ValueError):
        raise TypeError(
            f"mix parts are (paint, parts) with parts a number, got {part!r}"
        ) from None
    if not (math.isfinite(vol) and vol >= 0.0):
        raise ValueError(
            f"mix parts must be a non-negative number of parts, got {vol!r}"
        )
    return paint, vol


def as_paint(p: ColorLike) -> Paint:
    """``p`` if it is a paint, else a new opaque paint of exactly that colour.

    The last 1024 colours asked for are remembered, so asking again for one
    (a hex string a stroke is given, say) costs microseconds. A colour not
    seen before means fitting a spectrum, about a millisecond: when painting
    many slightly different colours (sampled from a reference, say), make a
    :class:`Paint` per step of colour you care about and reuse it, as
    ``examples/_studies.py`` does.
    """
    if isinstance(p, Paint):
        return p
    # The paint is built once per colour (it means fitting a spectrum, about
    # a millisecond, and a stroke given a colour string asks for it every
    # time), then copied, so that changing the paint handed out cannot change
    # the next one.
    key = p if isinstance(p, str) else tuple(sp.parse_color(p).tolist())
    return Paint(_cached_paint(key))


@functools.lru_cache(maxsize=1024)
def _cached_paint(key: str | tuple[float, ...]) -> Paint:
    return Paint(key)


def mix_colors(
    colors: Iterable[ColorLike], weights: Iterable[float] | None = None
) -> str:
    """Quick helper: hex color of a palette mix of plain colors / paints."""
    colors = list(colors)
    weights = [1.0] * len(colors) if weights is None else list(weights)
    return Paint.mix(
        *[(as_paint(c), w) for c, w in zip(colors, weights, strict=True)]
    ).hex
