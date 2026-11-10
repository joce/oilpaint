"""Spectral colour and Kubelka-Munk layers."""

import numpy as np

from oilpaint import spectral as sp


def test_rgb_reflectance_roundtrip():
    rng = np.random.default_rng(0)
    cols = rng.uniform(0.15, 1.0, (200, 3))
    back = sp.reflectance_to_srgb(sp.linear_rgb_to_reflectance(sp.srgb_to_linear(cols)))
    assert np.abs(back - cols).max() < 1e-3  # < 0.3/255


def test_reflectance_is_physical():
    rng = np.random.default_rng(1)
    R = sp.linear_rgb_to_reflectance(rng.random((500, 3)))
    assert R.min() >= sp.MIN_REFLECTANCE and R.max() <= 1.0


def test_km_limits():
    ground = np.full(sp.N_BANDS, 0.6)
    K = np.linspace(0.1, 3.0, sp.N_BANDS)
    # no paint -> ground
    assert np.allclose(sp.km_layer(K * 0, 0.0, ground), ground)
    # pure dye -> Beer-Lambert, light passes the layer twice
    assert np.allclose(sp.km_layer(K, 0.0, ground), ground * np.exp(-2 * K), rtol=1e-6)
    # very thick -> masstone, independent of the ground
    assert np.allclose(
        sp.km_layer(K * 200, 400.0, ground), sp.km_opaque(K, 2.0), atol=1e-4
    )
    assert np.allclose(
        sp.km_layer(K * 200, 400.0, 0.0), sp.km_opaque(K, 2.0), atol=1e-4
    )


def test_solve_absorption_reproduces_target():
    target = sp.srgb_to_reflectance("#3a7bd5")
    for S in (0.0, 0.5, 10.0):
        K = sp.solve_absorption(target, S)
        assert np.allclose(sp.km_layer(K, S, 1.0), target, atol=1e-6)


def test_km_film_is_the_layer_seen_on_its_own():
    rng = np.random.default_rng(0)
    k = rng.uniform(0, 20, (50, sp.N_BANDS)) * rng.uniform(0, 1, (50, 1)) ** 3
    s = rng.uniform(0, 10, 50)
    g = rng.uniform(0, 1, (50, sp.N_BANDS))
    R, T = sp.km_film(k, s)
    assert np.allclose(sp.km_layer(k, s, g), R + T**2 * g / (1 - R * g))
