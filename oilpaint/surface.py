"""The surface being painted on: its texture, priming and how it takes paint."""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from . import noise as nz
from . import spectral as sp
from .paint import ColorLike, Paint

TEXTURES = ("canvas", "linen", "cotton", "twill", "paper", "wood", "smooth")


@dataclass(frozen=True)
class Surface:
    """A support with its priming. Presets live in :mod:`oilpaint.surfaces`;
    tweak them with :meth:`with_` or build your own::

        from oilpaint import Canvas, Surface, surfaces
        canvas = Canvas(1200, 800, surface=surfaces.GESSO_PANEL)
        toned = surfaces.LINEN_FINE.with_(toning="#b98a5a", relief=1.4)
        mine = Surface(texture="twill", scale=5, absorbency=0.3, ground="#f2eee4")

    Parameters
    ----------
    texture:
        ``"linen"`` (irregular threads with slubs), ``"cotton"`` (even,
        machine-woven), ``"twill"`` (diagonal weave), ``"paper"``, ``"wood"``
        (grain of a panel under thin priming), ``"smooth"`` (metal, glass), or
        ``"canvas"`` (the original generic weave).
    scale:
        Size of the texture in pixels: thread spacing, paper tooth, or the
        spacing of wood growth rings.
    irregularity:
        For woven textures, 0 = perfectly even threads, 1 = very irregular.
    angle:
        Direction of the wood grain in degrees (0 = horizontal).
    relief:
        How deep the texture is: how strongly it shows under the light and
        through thin paint (0 = flat).
    tooth:
        0..1: how much the surface grips a brush. Rough tooth breaks up dry or
        lightly pressed strokes into a broken scumble and pulls paint off the
        brush; smooth surfaces let strokes glide unbroken.
    absorbency:
        0..1: how much the priming drinks oil from thin paint. An absorbent
        chalk ground grabs the first layer (it drags, blends less, and the
        brush empties faster) and leaves it matte ("sunk in": darks look
        lighter and duller). Oil-primed canvas or copper: 0.
    slickness:
        0..1: how freely paint slides and lifts off the bare surface. On a
        slick panel or copper, paint glides and can be wiped back; rough
        canvas grabs it. 0.5 is neutral.
    ground:
        Color of the priming.
    toning:
        Optional color of a transparent toned layer (imprimatura) brushed over
        the priming, e.g. a raw umber or red ochre wash. ``toning_strength``
        scales it (1 = that color over white).
    priming_marks:
        0..1: how visible the brushmarks of the priming are, in the direction
        ``priming_angle`` (degrees).
    """

    texture: str = "canvas"
    scale: float = 4.0
    irregularity: float = 0.3
    angle: float = 0.0
    relief: float = 1.0
    tooth: float = 0.5
    absorbency: float = 0.0
    slickness: float = 0.5
    ground: ColorLike = "#efe9dc"
    toning: ColorLike | None = None
    toning_strength: float = 1.0
    priming_marks: float = 0.0
    priming_angle: float = 0.0
    name: str = "canvas"

    def __post_init__(self):
        if self.texture not in TEXTURES:
            raise ValueError(f"unknown texture {self.texture!r}; use one of {TEXTURES}")
        if self.scale <= 0:
            raise ValueError("scale must be positive")
        for field in ("tooth", "absorbency", "slickness", "priming_marks"):
            if not 0.0 <= getattr(self, field) <= 1.0:
                raise ValueError(f"{field} must be between 0 and 1")

    def with_(self, **changes) -> Surface:
        """A copy with some properties changed."""
        return replace(self, **changes)

    @property
    def affects_paint(self) -> bool:
        """Whether this surface changes how strokes behave (vs. the neutral default)."""
        return self.absorbency > 0 or self.slickness != 0.5 or self.tooth != 0.5

    # ------------------------------------------------------------------
    def height_map(self, h: int, w: int, rng: np.random.Generator) -> np.ndarray:
        """Surface relief in 0..1."""
        if self.texture == "canvas":
            base = nz.canvas_weave(h, w, self.scale, rng)
        elif self.texture in ("linen", "cotton", "twill"):
            base = nz.woven(
                h, w, self.scale, rng, self.irregularity, twill=self.texture == "twill"
            )
        elif self.texture == "paper":
            base = nz.paper(h, w, self.scale, rng)
        elif self.texture == "wood":
            base = nz.wood(h, w, self.scale, self.angle, rng)
        else:
            base = np.zeros((h, w))
        if self.priming_marks > 0:
            marks = nz.brushmarks(h, w, self.priming_angle, rng)
            base = base + 1.5 * self.priming_marks * marks
            base = (base - base.min()) / max(np.ptp(base), 1e-9)
        return base

    def ground_reflectance(
        self, h: int, w: int, rng: np.random.Generator, image=None
    ) -> np.ndarray:
        """Reflectance spectrum (h, w, N_BANDS) of the primed (and toned) surface."""
        N = sp.N_BANDS
        out = np.empty((h, w, N), np.float32)
        if image is not None:
            img = image.astype(np.float64) / (255.0 if image.dtype == np.uint8 else 1.0)
            for y in range(h):
                out[y] = sp.linear_rgb_to_reflectance(sp.srgb_to_linear(img[y, :, :3]))
        else:
            out[:] = sp.srgb_to_reflectance(self.ground).astype(np.float32)
        if self.priming_marks > 0:
            # thinner priming in the troughs of the brushmarks reads a touch darker
            marks = nz.brushmarks(h, w, self.priming_angle, rng)
            out *= (1.0 - 0.05 * self.priming_marks * (1.0 - marks))[..., None].astype(
                np.float32
            )
        if self.toning is not None and self.toning_strength > 0:
            wash = Paint(self.toning, opacity=0.25)
            # brushed on loosely: broad, soft variations rather than fine streaks
            sweep = nz.aniso_noise(
                h, w, 90, 28, self.priming_angle + 30, rng, octaves=2
            )
            t = (self.toning_strength * (0.85 + 0.3 * sweep))[..., None]
            out[:] = sp.km_layer(wash.K * t, wash.S * t[..., 0], out).astype(np.float32)
        return out


DEFAULT_SURFACE = Surface()
