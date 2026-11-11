"""Ready-made painting surfaces.

::

    from oilpaint import Canvas, surfaces
    canvas = Canvas(1200, 800, surface=surfaces.LINEN_FINE)
    toned = surfaces.COTTON_DUCK.with_(toning="#a9805a")        # raw-sienna imprimatura
    behind = Canvas(800, 600, surface=surfaces.GLASS, reverse=True)  # reverse glass

Scales are in pixels and assume roughly 1 px ≈ 0.3 mm (a 1200 px canvas is
about 40 cm wide); use ``.with_(scale=...)`` for other sizes.
"""

from __future__ import annotations

from .surface import Surface

#: The original generic canvas (the default).
CANVAS = Surface(name="Canvas")

#: Fine, oil-primed linen: irregular threads with slubs, a little tooth, not absorbent.
#: The classic support for detailed oil painting.
LINEN_FINE = Surface(
    texture="linen",
    scale=3.0,
    irregularity=0.55,
    relief=0.9,
    tooth=0.5,
    absorbency=0.1,
    slickness=0.55,
    ground="#ece6d6",
    name="Fine linen",
)

#: Coarse linen: big, uneven threads that break up a dragged stroke.
LINEN_COARSE = Surface(
    texture="linen",
    scale=6.5,
    irregularity=0.7,
    relief=1.6,
    tooth=0.85,
    absorbency=0.15,
    slickness=0.45,
    ground="#e9e2d0",
    name="Coarse linen",
)

#: Cotton duck with acrylic gesso: an even, mechanical weave; a bright, slightly
#: absorbent priming.
COTTON_DUCK = Surface(
    texture="cotton",
    scale=5.0,
    irregularity=0.1,
    relief=1.3,
    tooth=0.7,
    absorbency=0.35,
    slickness=0.4,
    ground="#f4f2ec",
    name="Cotton duck",
)

#: Oak panel with a chalk-glue gesso, as used by the Flemish masters (Bruegel, van
#: Eyck): smooth, very absorbent, the priming brushmarks and a hint of wood grain show
#: through thin paint.
GESSO_PANEL = Surface(
    texture="wood",
    scale=18.0,
    angle=0.0,
    relief=0.3,
    tooth=0.15,
    absorbency=0.75,
    slickness=0.3,
    ground="#f1ebdc",
    priming_marks=0.6,
    priming_angle=0.0,
    name="Gesso panel",
)

#: Canvas board: fine cotton glued to card, acrylic primed; a light, even texture.
CANVAS_BOARD = Surface(
    texture="cotton",
    scale=3.5,
    irregularity=0.2,
    relief=0.7,
    tooth=0.55,
    absorbency=0.45,
    slickness=0.4,
    ground="#f2f0ea",
    name="Canvas board",
)

#: Oil painting paper: felted fibres with a pronounced tooth; fairly absorbent.
OIL_PAPER = Surface(
    texture="paper",
    scale=3.0,
    relief=0.6,
    tooth=0.75,
    absorbency=0.55,
    slickness=0.3,
    ground="#f0ebe0",
    name="Oil paper",
)

#: Copper plate: glass-smooth and very slick, no absorbency; the warm metal glows
#: through thin paint.
COPPER = Surface(
    texture="smooth",
    relief=0.0,
    tooth=0.0,
    absorbency=0.0,
    slickness=0.95,
    ground="#c4936a",
    name="Copper",
)

#: Glass: perfectly smooth, the slickest surface of all (paint glides on it and
#: wipes off clean) and clear: ``ground`` is what shows through it, here a white
#: backing card. Paint it on the front like copper, or on the back as a reverse
#: glass painting, seen through the glass:
#: ``Canvas(..., surface=surfaces.GLASS, reverse=True)``.
GLASS = Surface(
    texture="smooth",
    relief=0.0,
    tooth=0.0,
    absorbency=0.0,
    slickness=1.0,
    ground="#f3f2ee",
    transparent=True,
    name="Glass",
)

#: Every preset above, by name.
ALL = {
    "CANVAS": CANVAS,
    "LINEN_FINE": LINEN_FINE,
    "LINEN_COARSE": LINEN_COARSE,
    "COTTON_DUCK": COTTON_DUCK,
    "GESSO_PANEL": GESSO_PANEL,
    "CANVAS_BOARD": CANVAS_BOARD,
    "OIL_PAPER": OIL_PAPER,
    "COPPER": COPPER,
    "GLASS": GLASS,
}


def catalog() -> str:
    """A one-line summary of every preset."""
    rows = []
    for key, s in ALL.items():
        rows.append(
            f"{key:13s} {s.texture:7s} scale {s.scale:4.1f}  relief {s.relief:.2f}  "
            f"tooth {s.tooth:.2f}  absorbency {s.absorbency:.2f}  "
            f"slickness {s.slickness:.2f}  {s.name}"
        )
    return "\n".join(rows)
