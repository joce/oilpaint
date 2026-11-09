"""A starter palette of common oil colors.

Colors approximate each paint brushed out on white; opacity and tinting
strength follow the usual behavior of the real pigments (e.g. phthalo is
transparent and very strong, titanium white is opaque and strong).
"""

import functools
import itertools
import math
from typing import NamedTuple

import numpy as np

from . import spectral as sp
from .paint import ColorLike, Paint, as_paint
from .spectral import N_BANDS

TITANIUM_WHITE = Paint("#f6f5f0", opacity=1.0, strength=1.3, name="Titanium White")
ZINC_WHITE = Paint("#f3f3ef", opacity=0.55, strength=0.6, name="Zinc White")
IVORY_BLACK = Paint("#1b1a19", opacity=0.9, strength=1.0, name="Ivory Black")
CADMIUM_YELLOW = Paint("#f7c600", opacity=1.0, name="Cadmium Yellow")
LEMON_YELLOW = Paint("#f2e13a", opacity=0.8, name="Lemon Yellow")
YELLOW_OCHRE = Paint("#c8922f", opacity=0.9, name="Yellow Ochre")
CADMIUM_ORANGE = Paint("#ef7d10", opacity=1.0, name="Cadmium Orange")
CADMIUM_RED = Paint("#d22d1e", opacity=1.0, name="Cadmium Red")
ALIZARIN_CRIMSON = Paint("#b0203a", opacity=0.3, strength=1.6, name="Alizarin Crimson")
QUINACRIDONE_MAGENTA = Paint(
    "#b8246e", opacity=0.3, strength=1.6, name="Quinacridone Magenta"
)
BURNT_SIENNA = Paint("#93432a", opacity=0.6, name="Burnt Sienna")
BURNT_UMBER = Paint("#4f3526", opacity=0.9, name="Burnt Umber")
RAW_UMBER = Paint("#6b5638", opacity=0.9, name="Raw Umber")
ULTRAMARINE_BLUE = Paint("#23399b", opacity=0.45, strength=1.2, name="Ultramarine Blue")
COBALT_BLUE = Paint("#2f5fb3", opacity=0.6, name="Cobalt Blue")
CERULEAN_BLUE = Paint("#2f86c4", opacity=0.9, name="Cerulean Blue")
PHTHALO_BLUE = Paint("#0f3f8c", opacity=0.25, strength=2.5, name="Phthalo Blue")
PHTHALO_GREEN = Paint("#00705a", opacity=0.25, strength=2.5, name="Phthalo Green")
VIRIDIAN = Paint("#1f7a62", opacity=0.4, name="Viridian")
SAP_GREEN = Paint("#4f6b25", opacity=0.4, name="Sap Green")
DIOXAZINE_PURPLE = Paint("#4a2066", opacity=0.3, strength=2.0, name="Dioxazine Purple")

# --- an old master's palette (17th-century Dutch) -----------------------------
#: Lead white: warmer, more translucent and much stiffer than titanium white:
#: it holds the thick, broken lights of a Rembrandt or a Hals.
LEAD_WHITE = Paint("#f0eadc", opacity=0.8, strength=0.8, body=0.85, name="Lead White")
#: Lead-tin yellow: a cool, opaque yellow for the brightest lights.
LEAD_TIN_YELLOW = Paint("#e9d064", opacity=0.95, name="Lead-Tin Yellow")
#: Red ochre: an earthy, opaque brick red.
RED_OCHRE = Paint("#9e4b2e", opacity=0.9, name="Red Ochre")
#: Vermilion: a strong, opaque orange-red.
VERMILION = Paint("#d0381f", opacity=0.95, strength=1.1, name="Vermilion")
#: Red lake (madder or kermes): a transparent crimson for glazes and lips.
RED_LAKE = Paint("#a32642", opacity=0.25, strength=1.3, name="Red Lake")
#: Cassel earth (Van Dyck brown): a deep, warm, fairly transparent brown, for
#: rich darks and for glazing shadows.
CASSEL_EARTH = Paint("#3d2412", opacity=0.3, body=0.5, name="Cassel Earth")
#: Bone black: a warm, slightly transparent black.
BONE_BLACK = Paint("#211b18", opacity=0.7, name="Bone Black")
#: Smalt: ground blue glass, a weak greyish blue (cool half-tones in flesh).
SMALT = Paint("#5c6f9b", opacity=0.35, strength=0.5, name="Smalt")

#: The palette of a Dutch painter around 1650-1660.
OLD_MASTER = [
    LEAD_WHITE,
    LEAD_TIN_YELLOW,
    YELLOW_OCHRE,
    RED_OCHRE,
    VERMILION,
    RED_LAKE,
    RAW_UMBER,
    BURNT_UMBER,
    CASSEL_EARTH,
    BONE_BLACK,
    SMALT,
]

#: Clear oil medium: no pigment. Mix it in to thin a paint into a glaze, e.g.
#: ``Paint.mix((ALIZARIN_CRIMSON, 1), (MEDIUM, 3))``.
MEDIUM = Paint.from_optics(np.zeros(N_BANDS), 0.0, name="Glazing Medium", body=0.05)
#: Clear impasto medium (a stiff gel or cold wax): extends paint and makes it
#: stiffer, so it holds sharp ridges and peaks. ``Paint.mix((CADMIUM_RED, 2),
#: (IMPASTO_MEDIUM, 1))`` is a slightly more transparent, much stiffer red.
IMPASTO_MEDIUM = Paint.from_optics(
    np.zeros(N_BANDS), 0.4, name="Impasto Medium", body=1.0
)

ALL = [v for v in list(globals().values()) if isinstance(v, Paint)]


class Match(NamedTuple):
    """A mix found by :func:`match`: the mixed ``paint``, the recipe ``parts``
    (``[(paint, parts), ...]``, ready for :meth:`Paint.mix`) and how far its
    colour is from the target (CIELAB delta E; under ~2 is hard to tell apart)."""

    paint: Paint
    parts: list
    error: float


#: Proportions :func:`match` tries, from a pinch to equal parts, on a log scale.
_STEPS = np.geomspace(1 / 64, 1.0, 10)
#: About how many mixes :func:`match` evaluates in one batch.
_BATCH_ROWS = 2048


def _blocks(n_paints: int, n: int):
    """The mixes of ``n`` of ``n_paints`` paints that :func:`match` tries
    first, in blocks: one for each combination of paints with each of them in
    turn as the main one (one part), going through every proportion of the
    others. Returns the main paint of each block (blocks,), the others
    (blocks, n - 1), and the proportions of the others (rows, n - 1)."""
    grid = np.array(list(itertools.product(_STEPS, repeat=n - 1)))
    combos = itertools.combinations(range(n_paints), n)
    lead_others = [(c[i], c[:i] + c[i + 1 :]) for c in combos for i in range(n)]
    leads = np.array([lead for lead, _ in lead_others], dtype=np.intp)
    others = np.array([o for _, o in lead_others], dtype=np.intp)
    return leads, others, grid


#: :func:`_blocks`, remembered for small palettes and mixes (under 1 MB each).
_small_blocks = functools.lru_cache(maxsize=32)(_blocks)


def _volumes(n_paints, leads, others, grid) -> np.ndarray:
    """The volume parts of the mixes of some blocks of :func:`_blocks`, as
    (blocks, rows, n_paints)."""
    W = np.zeros((len(leads), len(grid), n_paints))
    blocks = np.arange(len(leads))
    # W[blocks, :, leads] is (blocks, rows): the main paint's column of each
    W[blocks, :, leads] = 1.0
    for k in range(others.shape[1]):
        W[blocks, :, others[:, k]] = grid[:, k]
    return W


def match(
    color: ColorLike,
    paints,
    max_paints: int = 3,
    thickness: float = 1.0,
    over: ColorLike = "#ffffff",
) -> Match:
    """Mix the given paints to match a colour, as a painter would on the palette.

    Tries every combination of up to ``max_paints`` of ``paints`` in many
    proportions, using the same pigment physics as the canvas, then refines
    the best. The colour compared is a layer ``thickness`` thick over
    ``over`` (thick paint is ``thickness`` ~2 or more; a thin scumble is
    ~0.3 over the colour beneath). Returns a :class:`Match`::

        m = palette.match("#c89070", palette.OLD_MASTER, over="#5a4532")
        m.parts   # [(Lead White, 1.0), (Lead-Tin Yellow, 1.6), (Red Lake, 3.9)]
        canvas.stroke(path, paint=m.paint)
    """
    whole = isinstance(max_paints, (int, np.integer)) or (
        isinstance(max_paints, (float, np.floating)) and float(max_paints).is_integer()
    )
    if not whole:
        raise TypeError(f"max_paints must be a whole number, got {max_paints!r}")
    if max_paints < 1:
        raise ValueError(f"max_paints must be 1 or more, got {max_paints!r}")
    max_paints = int(max_paints)
    paints = [as_paint(p) for p in paints]
    if not paints:
        raise ValueError("match needs at least one paint")
    target = sp.srgb_to_lab(sp.parse_color(color))
    is_spectrum = isinstance(over, np.ndarray) and over.shape[-1] == N_BANDS
    ground = over if is_spectrum else sp.srgb_to_reflectance(over)
    Ks = np.array([p.K for p in paints])
    Ss = np.array([p.S for p in paints])

    def errors(W):
        """Delta E of the mixes with volume parts W (..., m, n_paints): one
        result per row. Stacked ``@`` makes one BLAS call per (m, n_paints)
        block and the rest is elementwise, so each block gets the same bits
        as it would on its own (_km_layer_batch: as sp.km_layer)."""
        tot = W.sum(axis=-1, keepdims=True)
        K = (W @ Ks) / tot * thickness
        S = (W @ Ss) / tot[..., 0] * thickness
        R = sp._km_layer_batch(K, S, ground)
        lab = sp.srgb_to_lab(sp.reflectance_to_srgb(R))
        return np.linalg.norm(lab - target, axis=-1)

    # Try the blocks of _blocks in order, many at a time, keeping the first
    # mix with the smallest error: within a block the first minimum, and a
    # block only if its minimum is strictly below the best so far (one with
    # a NaN error is passed over: np.argmin points at the NaN).
    best_w, best_e = None, np.inf
    for n in range(1, min(max_paints, len(paints)) + 1):
        size = math.comb(len(paints), n) * n + 10 ** (n - 1) * (n - 1)
        small = size <= 100_000  # numbers in the arrays of _blocks
        leads, others, grid = (_small_blocks if small else _blocks)(len(paints), n)
        step = max(1, _BATCH_ROWS // len(grid))
        for s in range(0, len(leads), step):
            W = _volumes(len(paints), leads[s : s + step], others[s : s + step], grid)
            e = errors(W)
            i = np.argmin(e, axis=1)
            e_min = e[np.arange(len(e)), i]
            b = int(np.argmin(np.where(np.isnan(e_min), np.inf, e_min)))
            if e_min[b] < best_e:
                best_w, best_e = W[b, i[b]], float(e_min[b])
    # refine: nudge each part up and down, shrinking the step. A pass takes
    # the parts in turn: evaluate all its remaining nudges (two per part) from
    # the current mix, accept the first that helps (the smaller of its two,
    # the first if they tie) and continue from the part after it.
    if best_w is None:  # every mix was NaN: a NaN thickness or ground
        raise ValueError(
            "match found no mix to compare: thickness and over must be finite"
        )
    w = best_w
    for factor in (1.5, 1.2, 1.08, 1.03):
        improved = True
        while improved:
            improved = False
            todo = np.flatnonzero(w > 0)
            while len(todo):
                trials = np.tile(w, (len(todo), 2, 1))
                rows = np.arange(len(todo))
                trials[rows, 0, todo] *= factor
                trials[rows, 1, todo] /= factor
                e = errors(trials)
                k = np.argmin(e, axis=1)
                e_min = e[rows, k]
                better = np.flatnonzero(e_min < best_e - 1e-6)
                if not len(better):
                    break
                j = better[0]
                w, best_e, improved = trials[j, k[j]], float(e_min[j]), True
                todo = todo[j + 1 :]
    w = w / w[w > 0].min()
    parts = [(paints[j], float(w[j])) for j in np.flatnonzero(w > 0)]
    return Match(Paint.mix(*parts), parts, best_e)
