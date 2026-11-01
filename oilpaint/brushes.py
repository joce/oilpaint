"""A rack of brushes, and a factory to make your own.

Presets cover the usual oil-painting tools::

    from oilpaint import brushes
    canvas.stroke(path, paint, brush=brushes.FLAT_HOG)              # 24 px hog flat
    canvas.stroke(path, paint, brush=brushes.FILBERT_SABLE.sized(40))

Make your own from a shape, a size and a hair, then tweak anything::

    soft_flat = brushes.make("flat", 30, hair="sable", length=0.8)
    grass_fan = brushes.make("fan", 40, hair="hog", clumps=12)
    custom = brushes.FLAT_HOG.with_(aspect=0.35, slant=0.3, streaks=0.8)

Shapes: ``round``, ``flat``, ``filbert``, ``fan`` (see :class:`BrushType`).
Hairs: ``hog`` (stiff bristle with split "flag" tips that hold and spread
thick paint), ``mongoose``, ``synthetic``, ``sable`` (soft, smooth, fine),
``squirrel`` (very soft, holds a lot: mops and blenders).
"""

from __future__ import annotations

from .brush import BrushType

#: Character of common hair types. Stiff hair pushes paint and leaves strong
#: brush marks; soft hair lays it smoothly and blends.
HAIRS = {
    "hog": dict(stiffness=0.85, bristles=0.3, capacity=1.15),
    "mongoose": dict(stiffness=0.6, bristles=0.6, capacity=1.0),
    "synthetic": dict(stiffness=0.55, bristles=0.75, capacity=0.85),
    "sable": dict(stiffness=0.3, bristles=0.85, capacity=1.0),
    "squirrel": dict(stiffness=0.1, bristles=0.95, capacity=1.5),
}

_SHAPES = {
    "round": dict(aspect=1.0, spread=0.5),
    "flat": dict(aspect=0.22, spread=0.05),
    "filbert": dict(aspect=0.4, spread=0.45),
    "fan": dict(aspect=0.06, spread=0.1, clumps=9),
}


def make(shape: str = "round", width: float = 20.0, hair: str = "synthetic", length: float = 0.5,
         name: str | None = None, **overrides) -> BrushType:
    """Build a brush from a shape, a size in pixels and a hair type.

    ``length`` is the bristle length, 0 (short, like a bright: stiffer,
    controlled, smaller tip capacity) to 1 (long, like a rigger or long flat:
    springier, larger tip capacity). Any :class:`BrushType` field can be
    overridden.
    """
    if hair not in HAIRS:
        raise ValueError(f"unknown hair {hair!r}; use one of {sorted(HAIRS)}")
    if shape not in _SHAPES:
        raise ValueError(f"unknown shape {shape!r}; use one of {sorted(_SHAPES)}")
    h = HAIRS[hair]
    stiffness = min(1.0, max(0.0, h["stiffness"] + 0.3 * (0.5 - length)))
    props = dict(_SHAPES[shape])
    # soft hairs splay out more under pressure than stiff ones
    props["spread"] = props["spread"] * (1.3 - 0.6 * stiffness)
    props.update(shape=shape, width=float(width), stiffness=stiffness, bristles=h["bristles"],
                 capacity=h["capacity"] * (0.6 + 0.8 * length),
                 name=name or f"{shape} {hair} {width:g}px")
    props.update(overrides)
    return BrushType(**props)


# --- rounds ------------------------------------------------------------------
#: All-purpose round: a full belly and a point, thin lines with a light touch.
ROUND = make("round", 14, "synthetic", name="Round")
#: Hog round: scrubby, textured, for blocking in.
ROUND_HOG = make("round", 18, "hog", name="Round (hog)")
#: Sable round with a fine point: smooth strokes, very responsive to pressure.
ROUND_SABLE = make("round", 10, "sable", length=0.6, spread=0.75, name="Round (sable)")
#: Liner: ultra-fine point for outlines and detail.
LINER = make("round", 3, "synthetic", length=0.7, spread=0.5, name="Liner")
#: Rigger: long, thin hair for long fine lines (rigging, branches). Load it generously.
RIGGER = make("round", 2.5, "sable", length=1.0, capacity=2.2, spread=0.35, name="Rigger")
#: Mop: big, very soft and absorbent, for soft blending and covering large areas.
MOP = make("round", 70, "squirrel", length=0.8, softness=0.35, streaks=0.1, mixing=0.6, pickup=0.5,
           name="Mop")
#: Deer foot: a round cut on a slant, for stippling foliage and textures.
DEER_FOOT = make("round", 18, "hog", length=0.3, slant=0.35, aspect=0.8, name="Deer foot")

# --- flats -------------------------------------------------------------------
#: Long hog flat: the workhorse for covering large areas; turn it on its edge for lines.
FLAT_HOG = make("flat", 24, "hog", length=0.7, name="Flat (hog)")
#: Soft flat: smooth, even strokes and gentle blending.
FLAT_SABLE = make("flat", 20, "sable", length=0.7, name="Flat (sable)")
#: Bright: a short flat, controlled strokes, pushes stiff paint.
BRIGHT = make("flat", 16, "hog", length=0.15, aspect=0.28, name="Bright")
#: Angled flat: the slanted edge reaches into corners and paints curves.
ANGLED = make("flat", 20, "synthetic", length=0.6, slant=0.55, name="Angled flat")
#: Dagger: an angled brush with a long sharp toe, for fine, tapering strokes.
DAGGER = make("flat", 18, "sable", length=0.9, slant=0.9, aspect=0.16, spread=0.3, name="Dagger")
#: Wide hake-style blender: soft and wide for glazes and smooth gradients.
BLENDER = make("flat", 50, "squirrel", length=0.7, softness=0.2, streaks=0.15, mixing=0.55,
               name="Blender")

# --- filberts ----------------------------------------------------------------
#: Hog filbert: broad strokes or, on its edge, detail; soft-cornered marks.
FILBERT_HOG = make("filbert", 20, "hog", length=0.6, name="Filbert (hog)")
#: Soft filbert: holds plenty of paint, blends smoothly; petals and leaves.
FILBERT_SABLE = make("filbert", 16, "sable", length=0.6, name="Filbert (sable)")

# --- fans --------------------------------------------------------------------
#: Hog fan: separate clumps of stiff bristle, for grass, hair, foliage textures.
FAN_HOG = make("fan", 40, "hog", length=0.7, clumps=10, name="Fan (hog)")
#: Soft fan: gentle blending and feathering.
FAN_SOFT = make("fan", 40, "sable", length=0.7, clumps=16, streaks=0.2, name="Fan (soft)")

# --- knives ------------------------------------------------------------------
#: Palette knife: a thin steel blade. Lays flat, smooth, thick slabs of paint
#: with sharp edges and ridges; no brush marks.
PALETTE_KNIFE = BrushType(shape="flat", width=36, aspect=0.04, stiffness=1.0, bristles=1.0, capacity=0.6,
                          streaks=0.0, rough=0.0, drag=0.6, pickup=0.25, mixing=0.15, blending=0.12,
                          name="Palette knife")

# --- other tools -------------------------------------------------------------
#: A rag: lifts wet paint without putting any back, to wipe out lights or
#: correct. It wipes clean on slick surfaces and leaves a stain on absorbent ones.
RAG = BrushType(shape="round", width=40, aspect=0.8, stiffness=0.2, bristles=1.0, capacity=4.0, streaks=0.15,
                pickup=1.2, drag=0.05, mixing=0.0, softness=0.3, deposits=False, name="Rag")

ALL = {name: value for name, value in list(globals().items()) if isinstance(value, BrushType)}


def catalog() -> str:
    """A one-line summary of every preset."""
    rows = []
    for key, b in ALL.items():
        rows.append(f"{key:14s} {b.shape:8s} {b.width:5.1f}px  stiffness {b.stiffness:.2f}  "
                    f"capacity {b.capacity:.2f}  {b.name}")
    return "\n".join(rows)
