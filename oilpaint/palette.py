"""A starter palette of common oil colors.

Colors approximate each paint brushed out on white; opacity and tinting
strength follow the usual behavior of the real pigments (e.g. phthalo is
transparent and very strong, titanium white is opaque and strong).
"""

import numpy as np

from .paint import Paint
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
QUINACRIDONE_MAGENTA = Paint("#b8246e", opacity=0.3, strength=1.6, name="Quinacridone Magenta")
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

#: Clear oil medium: no pigment. Mix it in to thin a paint into a glaze, e.g.
#: ``Paint.mix((ALIZARIN_CRIMSON, 1), (MEDIUM, 3))``.
MEDIUM = Paint.from_optics(np.zeros(N_BANDS), 0.0, name="Glazing Medium")

ALL = [v for v in list(globals().values()) if isinstance(v, Paint)]
