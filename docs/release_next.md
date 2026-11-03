# Unreleased

## Overview

The first release of oilpaint, a small oil-painting simulator for Python: you
describe brush strokes (a path, a paint, how much of it, a brush, a pressure)
and it simulates how the paint is laid down, picked up, dragged and mixed on
the canvas, with colours that mix like real pigments.

## Highlights

- **Paint that mixes like paint.** Spectral Kubelka-Munk optics: blue and yellow
  make green, glazes stay transparent, white tints and black shades. A palette of
  modern and old-master pigments, and `palette.match` to mix a colour from it.
- **Brushes that hold paint.** Bristle columns at three depths, so a brush can be
  double loaded, dipped to the tip or to the ferrule, rolled, or left dirty from
  the last stroke. Rounds, flats, filberts, fans, mops, riggers and more.
- **Wet paint and impasto.** Wet-into-wet mixing, pickup, smearing and drag;
  thick paint that keeps its ridges and bristle marks, lit with gloss and shadows.
- **Knives.** Palette and painting knives that spread paint as a thin film,
  scrape it back, or leave broken colour on the high points.
- **Painting from a reference.** `Reference` reads colour, value and the
  direction of the forms from an image at any scale.
- **Fast enough for real paintings.** The stroke engine is compiled with numba;
  the example paintings (Bruegel, Bob Ross, Rembrandt) take a few minutes each.

Python 3.11 to 3.15. Install with `pip install oilpaint`.
