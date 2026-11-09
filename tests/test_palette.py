import numpy as np
import pytest

from oilpaint import palette as P
from oilpaint import spectral as sp

FLESH = [
    P.LEAD_WHITE,
    P.YELLOW_OCHRE,
    P.RED_OCHRE,
    P.VERMILION,
    P.RED_LAKE,
    P.RAW_UMBER,
    P.BONE_BLACK,
    P.SMALT,
]
GLAZE = [P.CASSEL_EARTH, P.BONE_BLACK, P.RED_LAKE, P.RAW_UMBER]
GLINT = [P.LEAD_WHITE, P.RAW_UMBER, P.BONE_BLACK, P.SMALT]

# What match found for these when it tried one block of mixes at a time: the
# search (its order, tie-breaking and refinement) must not change. Exact to
# the last bit on one machine; another BLAS may round a little differently.
PINNED = [
    (
        ("#c89070", P.OLD_MASTER),
        {"over": "#5a4532"},
        [
            (P.LEAD_WHITE, 1.0),
            (P.LEAD_TIN_YELLOW, 1.6061073975130138),
            (P.RED_LAKE, 3.9292592592592586),
        ],
        0.595026589806057,
    ),
    (
        ((0.72, 0.52, 0.40), FLESH),
        {"thickness": 1.4, "over": "#8a6a4c"},
        [
            (P.LEAD_WHITE, 1.6350230835272452),
            (P.RED_OCHRE, 1.0),
            (P.SMALT, 2.5954373627834384),
        ],
        0.49711035297474526,
    ),
    (
        ("#2b1d14", GLAZE),
        {"thickness": 0.4, "over": sp.srgb_to_reflectance("#5e4430")},
        [
            (P.CASSEL_EARTH, 5.548197851539336),
            (P.BONE_BLACK, 1.0),
            (P.RED_LAKE, 1.1444444444444444),
        ],
        1.5090905871891236,
    ),
    (
        ("#3f7a3a", [P.ULTRAMARINE_BLUE, P.CADMIUM_YELLOW]),
        {"max_paints": 2},
        [(P.ULTRAMARINE_BLUE, 3.883495145631067), (P.CADMIUM_YELLOW, 1.0)],
        6.245474341801551,
    ),
    (
        ("#00ff00", P.OLD_MASTER),  # out of reach of an old master
        {"max_paints": 2},
        [(P.LEAD_TIN_YELLOW, 1.427005620251887), (P.SMALT, 1.0)],
        84.85180759598835,
    ),
    (
        ("#9a8f88", GLINT),
        {"max_paints": 4, "over": "#2a2018"},
        [
            (P.LEAD_WHITE, 88.88888888888889),
            (P.RAW_UMBER, 1.0),
            (P.BONE_BLACK, 13.999122776609708),
            (P.SMALT, 50.39684199579494),
        ],
        0.2524517132037494,
    ),
    (
        ("#808080", FLESH),
        {"max_paints": 1, "thickness": 2.5},
        [(P.RAW_UMBER, 1.0)],
        26.28204963265421,
    ),
]


@pytest.mark.parametrize("args, kwargs, parts, error", PINNED)
def test_match_finds_the_same_mixes_as_before(args, kwargs, parts, error):
    m = P.match(*args, **kwargs)
    assert [p for p, _ in m.parts] == [p for p, _ in parts]
    assert [v for _, v in m.parts] == pytest.approx([v for _, v in parts], rel=1e-9)
    assert m.error == pytest.approx(error, rel=1e-9)
    assert np.array_equal(m.paint.K, P.Paint.mix(*m.parts).K)


TWIN_WHITE = P.Paint(P.LEAD_WHITE, name="Twin White")  # the same K and S
# K overflows in band 0 in some mixes (but not all) of these: NaN errors
_K_HUGE, _K_BIG = np.ones(sp.N_BANDS), np.ones(sp.N_BANDS)
_K_HUGE[0], _K_BIG[0] = 0.9 * np.finfo(float).max, 1e291
HUGE = P.Paint.from_optics(_K_HUGE, 1.0, name="huge")
BIG = P.Paint.from_optics(_K_BIG, 1.0, name="big")

# Cases decided by the order of the search and its tie-breaking: the first
# of equal candidates wins (paints, then the main paint, then proportions),
# NaN candidates spoil their block, and the refinement takes the first nudge
# that helps by more than 1e-6, then goes on with the next part (more
# of those below, on a made-up landscape).
ORDER = [
    # twin paints tie exactly: the first one
    (
        ("#ffffff", [P.LEAD_WHITE, P.LEAD_WHITE, P.BONE_BLACK]),
        {"max_paints": 2, "over": "#000000"},
        [(P.LEAD_WHITE, 1.0)],
        10.802489678249188,
    ),
    (
        ("#ffffff", [P.BONE_BLACK, P.LEAD_WHITE, TWIN_WHITE]),
        {"max_paints": 2, "over": "#000000"},
        [(P.LEAD_WHITE, 1.0)],
        10.802489678249188,
    ),
    # refinement: the first nudge that helps, not the best one
    (
        ("#704432", [P.VERMILION, P.YELLOW_OCHRE, P.CASSEL_EARTH, P.BONE_BLACK]),
        {"over": "#5a4532"},
        [
            (P.VERMILION, 1.4380928831189965),
            (P.YELLOW_OCHRE, 1.0),
            (P.BONE_BLACK, 1.08),
        ],
        2.7831218860488534,
    ),
    # a block with a NaN error is passed over, even where its others are good
    (
        ("#40a0c0", [P.LEAD_WHITE, HUGE, BIG]),
        {"thickness": 2.5},
        [(P.LEAD_WHITE, 1.0), (BIG, 57.60000000000001)],
        30.189377580988776,
    ),
]


@pytest.mark.filterwarnings("ignore::RuntimeWarning")  # the NaN case overflows
@pytest.mark.parametrize("args, kwargs, parts, error", ORDER)
def test_match_keeps_its_search_order(args, kwargs, parts, error):
    m = P.match(*args, **kwargs)
    assert [p for p, _ in m.parts] == [p for p, _ in parts]
    assert [v for _, v in m.parts] == pytest.approx([v for _, v in parts], rel=1e-9)
    assert m.error == pytest.approx(error, rel=1e-9)


def test_match_takes_the_first_of_mixes_that_tie():
    # with no paint at all on the ground every mix looks the same; which mix
    # of one paint and two looks best then depends on the last bit
    m = P.match(
        "#acf03b",
        [P.YELLOW_OCHRE, P.DIOXAZINE_PURPLE],
        max_paints=2,
        thickness=0.0,
        over=sp.srgb_to_reflectance("#5a4532"),
    )
    assert m.parts in (
        [(P.YELLOW_OCHRE, 1.0)],  # the first paint alone
        [(P.YELLOW_OCHRE, 64.0), (P.DIOXAZINE_PURPLE, 1.0)],  # first mix of two
    )


def _match_on_landscape(monkeypatch, n, level, target):
    """match n paints whose mixes all look grey, of a level given by
    ``level(K)``, to grey ``target``. Paint i absorbs only in band i, so a
    mix's K[..., i] is exactly its share of paint i."""
    paints = [P.Paint.from_optics(k, 0.0) for k in np.eye(sp.N_BANDS)[:n]]

    def grey(K, S, ground):
        return np.repeat(level(K)[..., None], sp.N_BANDS, axis=-1)

    monkeypatch.setattr(sp, "km_layer", grey)
    monkeypatch.setattr(sp, "_km_layer_batch", grey)
    m = P.match(sp.reflectance_to_srgb(np.full(sp.N_BANDS, target)), paints)
    assert [p for p, _ in m.parts] == paints
    return [v for _, v in m.parts]


# Proportions of the grid of the coarse search, and the mix 1 : G : H.
_G, _H = np.geomspace(1 / 64, 1.0, 10)[[3, 5]]
_SHARE = _G / (1 + _G)


def _near(share):
    """Mixes of 1 : G (shares of the second paint) that a nudge of the first
    by 1.5 makes, but no mix of the grid."""
    return (share > 0.66 * _SHARE) & (share < 1.5 * _SHARE) & (share != _SHARE)


def test_match_takes_the_first_of_two_nudges_that_tie(monkeypatch):
    def level(K):
        share = K[..., 1]
        return np.where(share == _SHARE, 0.7, np.where(_near(share), 0.8, 0.5))

    # 1 : G is the best of the grid; nudging the first paint up or down helps
    # as much: up (the first nudge) wins
    parts = _match_on_landscape(monkeypatch, 2, level, 0.8)
    assert parts == pytest.approx([1.5 / _G, 1.0])


def test_match_ignores_nudges_that_help_by_under_1e_6(monkeypatch):
    def level(K):
        share = K[..., 1]
        return np.where(share == _SHARE, 0.8 - 1e-9, np.where(_near(share), 0.8, 0.5))

    assert _match_on_landscape(monkeypatch, 2, level, 0.8) == pytest.approx(
        [1 / _G, 1.0]
    )


def test_match_goes_on_from_the_next_part_after_a_nudge(monkeypatch):
    # mixes 1 : b : c, by their proportions (b, c) and how good they look
    spots = {
        (_G, _H): 0.7,  # the best of the grid
        (1.5 * _G, _H): 0.75,  # its second part up: better
        (1.5 * _G, _H / 1.5): 0.85,  # then the third down: better still
        (2.25 * _G, 1.5 * _H): 0.8,  # (or then the first down: better too)
    }

    def level(K):
        with np.errstate(divide="ignore", invalid="ignore"):
            b, c = K[..., 1] / K[..., 0], K[..., 2] / K[..., 0]
        out = np.full(b.shape, 0.5)
        for (b0, c0), v in spots.items():
            out[
                np.isclose(b, b0, rtol=1e-9, atol=0)
                & np.isclose(c, c0, rtol=1e-9, atol=0)
            ] = v
        return out

    # after the second part, the third: not the first again
    parts = _match_on_landscape(monkeypatch, 3, level, 0.9)
    assert parts == pytest.approx([1 / (1.5 * _G), 1.0, _H / (2.25 * _G)])


def test_match_checks_its_paints():
    with pytest.raises(ValueError):
        P.match("#808080", [])


def test_km_layer_batch_is_km_layer_to_the_bit():
    rng = np.random.default_rng(3)
    kappa = rng.uniform(0.0, 3.0, (5, 7, sp.N_BANDS)) ** 3
    sigma = rng.uniform(0.0, 12.0, (5, 7))
    kappa[0] = 0.0  # empty layers: the series for tanh(B) / B near 0
    kappa[1, :, :5] = 1e-9
    sigma[2] = 0.0  # pure dyes
    kappa[3, 0, 0] = -1e-12  # rounding below zero is clamped
    sigma[3, 1] = -1e-12
    ground = sp.srgb_to_reflectance("#5e4430")
    assert np.array_equal(
        sp._km_layer_batch(kappa, sigma, ground), sp.km_layer(kappa, sigma, ground)
    )
