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

# Cases decided by the order of the search: NaN candidates spoil their
# block, and the refinement takes the first nudge that helps by more than
# 1e-6, then goes on with the next part. Every decision on the way is made by
# a margin of 1e-5 or more (errors of ~3 and ~30), far above rounding. Ties
# (the first of equal candidates wins) are pinned on a made-up landscape
# below, where they are exact on any machine.
ORDER = [
    # refinement: the first nudge that helps (by 0.025), not the best (0.066)
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


@pytest.mark.parametrize(
    "paints",
    [
        [P.LEAD_WHITE, P.LEAD_WHITE, P.BONE_BLACK],
        [P.BONE_BLACK, P.LEAD_WHITE, TWIN_WHITE],
    ],
)
def test_match_of_twin_paints_is_made_of_them_alone(paints):
    # A mix of two identical paints, (w1 K + w2 K) / (w1 + w2), is K only
    # to within an ulp or so, and which way it rounds depends on the CPU: the
    # white alone or a mix of the twins may come first, so either is right.
    # (Which of equal candidates wins is pinned on the landscape below.)
    m = P.match("#ffffff", paints, max_paints=2, over="#000000")
    assert m.parts
    assert all(p is P.LEAD_WHITE or p is TWIN_WHITE for p, _ in m.parts)
    one = P.match("#ffffff", [P.LEAD_WHITE], max_paints=1, over="#000000")
    assert one.parts == [(P.LEAD_WHITE, 1.0)]
    assert m.error == pytest.approx(one.error, rel=1e-9)
    assert m.error == pytest.approx(10.802489678249188, rel=1e-9)
    np.testing.assert_allclose(m.paint.K, P.LEAD_WHITE.K, rtol=1e-12)
    np.testing.assert_allclose(m.paint.S, P.LEAD_WHITE.S, rtol=1e-12)


def test_match_on_bare_ground_finds_the_ground():
    # with no paint at all on the ground every mix looks the same: the ground
    # (which of them comes first then depends on the last bit; the first of
    # equal candidates is pinned on the landscape below)
    paints = [P.YELLOW_OCHRE, P.DIOXAZINE_PURPLE]
    ground = sp.srgb_to_reflectance("#5a4532")
    m = P.match("#acf03b", paints, max_paints=2, thickness=0.0, over=ground)
    assert 1 <= len(m.parts) <= 2
    assert all(any(p is q for q in paints) for p, _ in m.parts)
    bare = sp.srgb_to_lab(sp.reflectance_to_srgb(ground))
    target = sp.srgb_to_lab(sp.parse_color("#acf03b"))
    assert m.error == pytest.approx(np.linalg.norm(bare - target), rel=1e-9)


def _landscape(monkeypatch, bands, level, target, **kwargs):
    """match paints whose mixes all look grey, of a level given by
    ``level(K)``, to grey ``target``; returns the match and the paints.

    Paint j absorbs only in band ``bands[j]`` (paints on the same band are
    twins under different names), so a mix's K[..., i] is exactly its share
    of the paints on band i: one product per paint, each by 1 or 0, divided
    by the total, the same bits on any machine for mixes of up to two
    paints. A grey of level v has colour (v, v, v) and error
    sqrt(3) |v - target|, by elementwise arithmetic alone: equal levels tie
    exactly, whatever the BLAS."""
    paints = [
        P.Paint.from_optics(np.eye(sp.N_BANDS)[b], 0.0, name=f"band {b} #{j}")
        for j, b in enumerate(bands)
    ]

    def grey(K, S, ground):
        return np.repeat(level(K)[..., None], sp.N_BANDS, axis=-1)

    def rgb(R):
        return np.asarray(R)[..., :3]

    def lab(c):
        return np.asarray(c, dtype=np.float64)

    monkeypatch.setattr(sp, "km_layer", grey)
    monkeypatch.setattr(sp, "_km_layer_batch", grey)
    monkeypatch.setattr(sp, "reflectance_to_srgb", rgb)
    monkeypatch.setattr(sp, "srgb_to_lab", lab)
    return P.match(np.full(3, target), paints, **kwargs), paints


def _match_on_landscape(monkeypatch, n, level, target):
    """:func:`_landscape` with paint i on band i, n of them, which the match
    must all use: returns their parts."""
    m, paints = _landscape(monkeypatch, range(n), level, target)
    assert [p for p, _ in m.parts] == paints
    return [v for _, v in m.parts]


# Proportions of the grid of the coarse search, and the mix 1 : G : H.
_G, _H = np.geomspace(1 / 64, 1.0, 10)[[3, 5]]
_SHARE = _G / (1 + _G)


def _near(share):
    """Mixes of 1 : G (shares of the second paint) that a nudge of the first
    by 1.5 makes, but no mix of the grid."""
    return (share > 0.66 * _SHARE) & (share < 1.5 * _SHARE) & (share != _SHARE)


# The first of equal candidates wins: fewer paints, then the paints in the
# order given, then the main paint in that order, then smaller proportions
# of the others. The best mixes here tie exactly, and no nudge of them helps.


@pytest.mark.parametrize("bands, first", [([0, 0, 1], 0), ([1, 0, 0], 1)])
def test_match_takes_the_first_of_twin_paints(monkeypatch, bands, first):
    # the twins on band 0, alone or mixed together (K exactly the same),
    # look best; the paint on band 1 spoils any mix
    def level(K):
        return np.where(K[..., 0] == 1.0, 0.8, 0.5)

    m, paints = _landscape(monkeypatch, bands, level, 0.9, max_paints=2)
    # the first twin alone: not the second, nor a mix of the two
    assert m.parts == [(paints[first], 1.0)]
    assert m.error == pytest.approx(np.sqrt(3) * (0.9 - 0.8), rel=1e-12)


def test_match_takes_one_paint_over_an_equal_mix_of_two(monkeypatch):
    def level(K):
        share = K[..., 1]  # of the second paint
        return np.where((share == 1.0) | (share == _SHARE), 0.8, 0.5)

    m, paints = _landscape(monkeypatch, [0, 1], level, 0.9, max_paints=2)
    assert m.parts == [(paints[1], 1.0)]  # not 1 : G of the two


def test_match_takes_the_first_of_equal_mixes_of_a_main_paint(monkeypatch):
    def level(K):
        share = K[..., 1]
        return np.where((share == _SHARE) | (share == _H / (1 + _H)), 0.8, 0.5)

    m, paints = _landscape(monkeypatch, [0, 1], level, 0.9, max_paints=2)
    assert m.parts == [(paints[0], 1 / _G), (paints[1], 1.0)]  # 1 : G, not 1 : H


def test_match_takes_the_first_main_paint_of_equal_mixes(monkeypatch):
    def level(K):
        share = K[..., 1]
        return np.where((share == _H / (1 + _H)) | (share == 1 / (1 + _G)), 0.8, 0.5)

    m, paints = _landscape(monkeypatch, [0, 1], level, 0.9, max_paints=2)
    # the first paint as the main one, 1 : H; not the second, G : 1
    assert m.parts == [(paints[0], 1 / _H), (paints[1], 1.0)]


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


def test_palette_match_finds_a_close_mix_of_the_given_paints():
    m = P.match("#c89070", P.OLD_MASTER, over="#5a4532")
    assert m.error < 2.0
    assert all(any(p is q for q in P.OLD_MASTER) for p, _ in m.parts)
    assert len(m.parts) <= 3
    # the mix it returns really looks like that over that ground
    got = sp.srgb_to_lab(m.paint.appearance(1.0, over="#5a4532"))
    assert np.linalg.norm(got - sp.srgb_to_lab(sp.parse_color("#c89070"))) < 2.0
