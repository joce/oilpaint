"""A starter palette of common oil colors.

Colors approximate each paint brushed out on white; opacity and tinting
strength follow the usual behavior of the real pigments (e.g. phthalo is
transparent and very strong, titanium white is opaque and strong).
"""

import itertools
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
        """Delta E of the mixes with volume parts W (m, n_paints)."""
        tot = W.sum(axis=1, keepdims=True)
        K = (W @ Ks) / tot * thickness
        S = (W @ Ss) / tot[:, 0] * thickness
        lab = sp.srgb_to_lab(sp.reflectance_to_srgb(sp.km_layer(K, S, ground)))
        return np.linalg.norm(lab - target, axis=-1)

    # proportions from a pinch to equal parts, on a log scale
    steps = np.geomspace(1 / 64, 1.0, 10)
    best_w, best_e = None, np.inf
    for n in range(1, min(max_paints, len(paints)) + 1):
        grid = np.array(list(itertools.product(steps, repeat=n - 1)))
        for combo in itertools.combinations(range(len(paints)), n):
            # each paint in turn as the main one
            for lead in range(n):
                W = np.zeros((max(len(grid), 1), len(paints)))
                W[:, combo[lead]] = 1.0
                others = [c for c in combo if c != combo[lead]]
                if others:
                    W[:, others] = grid
                e = errors(W)
                i = int(np.argmin(e))
                if e[i] < best_e:
                    best_w, best_e = W[i].copy(), float(e[i])
    # refine: nudge each part up and down, shrinking the step
    if best_w is None:  # every mix was NaN: a NaN thickness or ground
        raise ValueError(
            "match found no mix to compare: thickness and over must be finite"
        )
    w = best_w
    for factor in (1.5, 1.2, 1.08, 1.03):
        improved = True
        while improved:
            improved = False
            for j in np.flatnonzero(w > 0):
                trial = np.repeat(w[None, :], 2, axis=0)
                trial[0, j] *= factor
                trial[1, j] /= factor
                e = errors(trial)
                k = int(np.argmin(e))
                if e[k] < best_e - 1e-6:
                    w, best_e, improved = trial[k], float(e[k]), True
    w = w / w[w > 0].min()
    parts = [(paints[j], float(w[j])) for j in np.flatnonzero(w > 0)]
    return Match(Paint.mix(*parts), parts, best_e)
