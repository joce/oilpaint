"""Spectral color science used by the paint model.

Everything here works on *reflectance spectra*: arrays whose last axis holds
reflectance (0..1) sampled at :data:`WAVELENGTHS`.

Pipeline
--------
sRGB color  ->  smooth reflectance spectrum      (Burns' LHTSS basis spectra)
paint layer ->  Kubelka-Munk absorption/scatter  (two-flux model, finite thickness)
spectrum    ->  CIE XYZ (D65)  ->  linear sRGB  ->  sRGB

Mixing paints means adding their absorption (K) and scattering (S) spectra,
weighted by volume. That is what makes blue + yellow give green instead of the
grey you get by averaging RGB values.
"""

from __future__ import annotations

import re

import numpy as np

# --------------------------------------------------------------------------
# Wavelength sampling
# --------------------------------------------------------------------------
#: Sample wavelengths in nm. 24 uniform bands over the visible range keep the
#: mixing error below ~0.3 dE (mean) compared with a full 10 nm spectrum,
#: while keeping the per-pixel memory of the canvas reasonable.
WAVELENGTHS = np.linspace(400.0, 700.0, 24)
N_BANDS = WAVELENGTHS.size

# CIE 1931 2-degree standard observer, 380..780 nm, 10 nm steps (x, y, z).
_CIE_WL = np.arange(380.0, 781.0, 10.0)
_CIE_1931 = np.array([
    [0.001368, 0.000039, 0.006450], [0.004243, 0.000120, 0.020050],
    [0.014310, 0.000396, 0.067850], [0.043510, 0.001210, 0.207400],
    [0.134380, 0.004000, 0.645600], [0.283900, 0.011600, 1.385600],
    [0.348280, 0.023000, 1.747060], [0.336200, 0.038000, 1.772110],
    [0.290800, 0.060000, 1.669200], [0.195360, 0.090980, 1.287640],
    [0.095640, 0.139020, 0.812950], [0.032010, 0.208020, 0.465180],
    [0.004900, 0.323000, 0.272000], [0.009300, 0.503000, 0.158200],
    [0.063270, 0.710000, 0.078250], [0.165500, 0.862000, 0.042160],
    [0.290400, 0.954000, 0.020300], [0.433450, 0.994950, 0.008750],
    [0.594500, 0.995000, 0.003900], [0.762100, 0.952000, 0.002100],
    [0.916300, 0.870000, 0.001650], [1.026300, 0.757000, 0.001100],
    [1.062200, 0.631000, 0.000800], [1.002600, 0.503000, 0.000340],
    [0.854450, 0.381000, 0.000190], [0.642400, 0.265000, 0.000050],
    [0.447900, 0.175000, 0.000020], [0.283500, 0.107000, 0.000000],
    [0.164900, 0.061000, 0.000000], [0.087400, 0.032000, 0.000000],
    [0.046770, 0.017000, 0.000000], [0.022700, 0.008210, 0.000000],
    [0.011359, 0.004102, 0.000000], [0.005790, 0.002091, 0.000000],
    [0.002899, 0.001047, 0.000000], [0.001440, 0.000520, 0.000000],
    [0.000690, 0.000249, 0.000000], [0.000332, 0.000120, 0.000000],
    [0.000166, 0.000060, 0.000000], [0.000083, 0.000030, 0.000000],
    [0.000042, 0.000015, 0.000000],
])

# CIE standard illuminant D65, 380..780 nm, 10 nm steps.
_D65 = np.array([
    49.98, 54.65, 82.75, 91.49, 93.43, 86.68, 104.86, 117.01, 117.81, 114.86,
    115.92, 108.81, 109.35, 107.80, 104.79, 107.69, 104.41, 104.05, 100.00,
    96.33, 95.79, 88.69, 90.01, 89.60, 87.70, 83.29, 83.70, 80.03, 80.21,
    82.28, 78.28, 69.72, 71.61, 74.35, 61.60, 69.89, 75.09, 63.59, 46.42,
    66.81, 63.38,
])

_XYZ_TO_LINEAR_SRGB = np.array([
    [3.2404542, -1.5371385, -0.4985314],
    [-0.9692660, 1.8760108, 0.0415560],
    [0.0556434, -0.2040259, 1.0572252],
])


def _build_spectrum_to_rgb(wavelengths: np.ndarray) -> np.ndarray:
    """3xN matrix mapping a sampled reflectance spectrum to linear sRGB.

    The spectrum is treated as piecewise linear between samples and integrated
    against the CIE observer under D65 on a 1 nm grid. Rows are normalised so
    that a perfect white reflector (R = 1) maps exactly to (1, 1, 1).
    """
    fine = np.arange(380.0, 781.0, 1.0)
    cmf = np.stack([np.interp(fine, _CIE_WL, _CIE_1931[:, i]) for i in range(3)], -1)
    weights = cmf * np.interp(fine, _CIE_WL, _D65)[:, None]
    hats = np.empty((wavelengths.size, fine.size))
    for i in range(wavelengths.size):
        unit = np.zeros(wavelengths.size)
        unit[i] = 1.0
        hats[i] = np.interp(fine, wavelengths, unit)
    xyz = (hats @ weights).T
    rgb = _XYZ_TO_LINEAR_SRGB @ (xyz / xyz[1].sum())
    return rgb / rgb.sum(axis=1, keepdims=True)


#: (3, N_BANDS) matrix: linear sRGB = SPECTRUM_TO_RGB @ reflectance
SPECTRUM_TO_RGB = _build_spectrum_to_rgb(WAVELENGTHS)

#: Rec.709 luminance weights for linear sRGB.
LUMINANCE = np.array([0.2126, 0.7152, 0.0722])

# --------------------------------------------------------------------------
# sRGB helpers
# --------------------------------------------------------------------------


def srgb_to_linear(c) -> np.ndarray:
    c = np.asarray(c, dtype=np.float64)
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def linear_to_srgb(c) -> np.ndarray:
    c = np.clip(np.asarray(c, dtype=np.float64), 0.0, 1.0)
    return np.where(c <= 0.0031308, 12.92 * c, 1.055 * c ** (1.0 / 2.4) - 0.055)


_HEX = re.compile(r"^#?([0-9a-fA-F]{6})$")


def parse_color(color) -> np.ndarray:
    """Return an sRGB triple in 0..1.

    Accepts ``"#rrggbb"`` strings, float triples in 0..1, or int triples in
    0..255 (any component > 1 means the 0..255 scale).
    """
    if isinstance(color, str):
        m = _HEX.match(color.strip())
        if not m:
            raise ValueError(f"cannot parse color {color!r}; use '#rrggbb'")
        h = m.group(1)
        return np.array([int(h[i:i + 2], 16) for i in (0, 2, 4)], dtype=np.float64) / 255.0
    c = np.asarray(color, dtype=np.float64).reshape(-1)
    if c.size != 3:
        raise ValueError(f"expected 3 color components, got {c.size}")
    if c.max() > 1.0:
        c = c / 255.0
    return np.clip(c, 0.0, 1.0)


def to_hex(srgb) -> str:
    c = np.clip(np.round(np.asarray(srgb) * 255), 0, 255).astype(int)
    return "#{:02x}{:02x}{:02x}".format(*c)


_RGB_TO_XYZ = np.array([[0.4124564, 0.3575761, 0.1804375],
                        [0.2126729, 0.7151522, 0.0721750],
                        [0.0193339, 0.1191920, 0.9503041]])
_D65 = np.array([0.95047, 1.0, 1.08883])


def srgb_to_lab(srgb) -> np.ndarray:
    """sRGB (..., 3) in 0..1 -> CIELAB (D65): L* 0..100, where a difference
    of ~2.3 is just noticeable."""
    xyz = srgb_to_linear(srgb) @ _RGB_TO_XYZ.T / _D65
    f = np.where(xyz > (6 / 29) ** 3, np.cbrt(np.maximum(xyz, 0.0)), xyz / (3 * (6 / 29) ** 2) + 4 / 29)
    return np.stack([116 * f[..., 1] - 16, 500 * (f[..., 0] - f[..., 1]), 200 * (f[..., 1] - f[..., 2])], axis=-1)


# --------------------------------------------------------------------------
# sRGB -> reflectance (Scott Burns' least hyperbolic tangent slope squared)
# --------------------------------------------------------------------------


def _lhtss(target_linear_rgb: np.ndarray, T: np.ndarray, iters: int = 200) -> np.ndarray:
    """Smoothest reflectance in (0, 1) whose color is ``target_linear_rgb``.

    S. A. Burns, "Generating reflectance curves from sRGB triplets" (2015/2019).
    Newton iteration on the Lagrangian of: minimise sum (z[i+1]-z[i])^2 with
    reflectance = (tanh(z) + 1) / 2 and T @ reflectance = target.
    """
    n = T.shape[1]
    D = 4 * np.eye(n) - 2 * np.eye(n, k=1) - 2 * np.eye(n, k=-1)
    D[0, 0] = D[-1, -1] = 2
    z = np.zeros(n)
    lam = np.zeros(3)
    for _ in range(iters):
        th = np.tanh(z)
        sech2 = 1.0 - th * th
        d1 = sech2 / 2.0
        d2 = -sech2 * th
        F = np.concatenate([D @ z + d1 * (T.T @ lam), T @ ((th + 1.0) / 2.0) - target_linear_rgb])
        if np.abs(F).max() < 1e-12:
            break
        J = np.zeros((n + 3, n + 3))
        J[:n, :n] = D + np.diag(d2 * (T.T @ lam))
        J[:n, n:] = d1[:, None] * T.T
        J[n:, :n] = T * d1[None, :]
        delta = np.linalg.solve(J, -F)
        z += delta[:n]
        lam += delta[n:]
    return (np.tanh(z) + 1.0) / 2.0


def _build_basis() -> np.ndarray:
    # white, cyan, magenta, yellow, red, green, blue
    corners = [(1, 1, 1), (0, 1, 1), (1, 0, 1), (1, 1, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1)]
    eps = 1e-4
    return np.stack([_lhtss(np.clip(np.array(c, float), eps, 1 - eps), SPECTRUM_TO_RGB) for c in corners])


#: (7, N_BANDS) reflectance spectra of white, cyan, magenta, yellow, red, green, blue.
_BASIS = _build_basis()

#: Darkest reflectance any color is mapped to. Real "black" paint reflects a
#: few percent; this keeps Kubelka-Munk absorption finite.
MIN_REFLECTANCE = 1e-3


def linear_rgb_to_reflectance(lrgb) -> np.ndarray:
    """Vectorised linear sRGB (..., 3) -> reflectance spectrum (..., N_BANDS).

    The color is split into white + one secondary + one primary component and
    the matching precomputed smooth spectra are summed. Because the color
    transform is linear, the result reproduces the input color exactly.
    """
    lrgb = np.clip(np.asarray(lrgb, dtype=np.float64), 0.0, 1.0)
    r, g, b = lrgb[..., 0], lrgb[..., 1], lrgb[..., 2]
    w = np.minimum(np.minimum(r, g), b)
    r, g, b = r - w, g - w, b - w
    c = np.minimum(g, b)
    m = np.minimum(r, b)
    y = np.minimum(r, g)
    red = np.minimum(np.maximum(0, r - b), np.maximum(0, r - g))
    green = np.minimum(np.maximum(0, g - b), np.maximum(0, g - r))
    blue = np.minimum(np.maximum(0, b - g), np.maximum(0, b - r))
    weights = np.stack([w, c, m, y, red, green, blue], axis=-1)
    return np.clip(weights @ _BASIS, MIN_REFLECTANCE, 1.0)


def reflectance_to_linear_rgb(R) -> np.ndarray:
    """Reflectance spectrum (..., N_BANDS) -> linear sRGB (..., 3)."""
    return np.asarray(R) @ SPECTRUM_TO_RGB.T


def srgb_to_reflectance(color) -> np.ndarray:
    return linear_rgb_to_reflectance(srgb_to_linear(parse_color(color)))


def reflectance_to_srgb(R) -> np.ndarray:
    return linear_to_srgb(reflectance_to_linear_rgb(R))


# --------------------------------------------------------------------------
# Kubelka-Munk
# --------------------------------------------------------------------------


def km_layer(kappa, sigma, ground):
    """Reflectance of a paint layer lying on a ground of reflectance ``ground``.

    Two-flux Kubelka-Munk solution for a layer of finite thickness. Inputs are
    *optical thicknesses*: ``kappa = K * thickness`` (absorption, per band) and
    ``sigma = S * thickness`` (scattering). Written in a form that stays finite
    for empty layers (returns the ground) and for pure dyes (sigma = 0 gives
    Beer-Lambert: ground * exp(-2 kappa)).
    """
    # Clamp tiny negatives left by float rounding in the paint bookkeeping.
    kappa = np.maximum(np.asarray(kappa), 0.0)
    sigma = np.maximum(np.asarray(sigma), 0.0)
    if sigma.ndim == kappa.ndim - 1:
        sigma = sigma[..., None]
    B = np.sqrt(kappa * kappa + 2.0 * kappa * sigma)
    small = B < 1e-4
    tau = np.where(small, 1.0 - B * B / 3.0, np.tanh(B) / np.where(small, 1.0, B))
    num = sigma * tau - ground * ((sigma + kappa) * tau - 1.0)
    den = (sigma + kappa - ground * sigma) * tau + 1.0
    return num / den


def km_opaque(kappa, sigma):
    """Reflectance of an infinitely thick layer (the paint's masstone)."""
    kappa = np.asarray(kappa)
    sigma = np.asarray(sigma)
    if sigma.ndim == kappa.ndim - 1:
        sigma = sigma[..., None]
    ks = kappa / np.maximum(sigma, 1e-12)
    return 1.0 + ks - np.sqrt(ks * ks + 2.0 * ks)


def solve_absorption(target_R, sigma) -> np.ndarray:
    """Absorption ``kappa`` (per band) such that a unit-thickness layer with
    scattering ``sigma`` looks like ``target_R`` when painted over white.

    Over a perfect white ground the KM layer formula reduces to
    R = (1 - kappa*tau) / (1 + kappa*tau), which is monotonic in kappa, so a
    vectorised bisection in log-space is robust.
    """
    target_R = np.clip(np.asarray(target_R, dtype=np.float64), MIN_REFLECTANCE, 1.0)
    sigma = float(sigma)
    lo = np.full(target_R.shape, -14.0)
    hi = np.full(target_R.shape, 12.0)
    for _ in range(70):
        mid = 0.5 * (lo + hi)
        R = km_layer(np.exp(mid), sigma, 1.0)
        too_dark = R < target_R
        hi = np.where(too_dark, mid, hi)
        lo = np.where(too_dark, lo, mid)
    kappa = np.exp(0.5 * (lo + hi))
    return np.where(target_R >= 0.99999, 0.0, kappa)
