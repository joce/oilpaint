# Changelog

All notable changes to oilpaint are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed

- Brushes stay dirty longer after working through thick wet paint. What a
  brush lifts no longer all goes into its tips: deeper wet paint comes
  further up the hairs, more so under pressure and with soft hair, and part
  of it goes into the body of the brush, where it lasts into the next
  strokes. A thin film still only dirties the tips.

## [0.1.0] - 2026-11-03

First release.

### Added

- Strokes along smooth paths (Catmull-Rom, Bezier, polyline), with width,
  pressure, angle and roll that can change along the stroke.
- Paint that mixes like pigment: spectral Kubelka-Munk optics derived from
  RGB colours, opacity, tinting strength and body (stiffness); a palette of
  modern and old-master pigments, and `palette.match` to mix a colour from it.
- Brushes that carry paint in bristle columns at three depths (tip, belly,
  heel): rounds, flats, filberts, fans, angled and dagger brushes, mops,
  riggers, the brush handle (sgraffito) and a rag; double loading, partly
  mixed loads, dirty brushes and rolling a round.
- Wet paint: two wet layers per pixel, wet-into-wet mixing, pickup, smear and
  drag; drying and glazing.
- Impasto: thick paint holds ridges and bristle marks, overloaded brushes
  squeeze paint out at their edges, and the relief is lit, with gloss, cast
  shadows and dark hollows.
- Palette and painting knives: blade shapes, a thin-film flow of the paint
  under the blade, broken colour when skimming, torn peaks when lifting off.
- Surfaces: canvases, linen, panels and papers, with tooth, absorbency,
  slickness and a toned ground.
- Painting from a reference image (`Reference`): colour, value, direction of
  the forms and detail at any scale.
- A compiled (numba) stroke engine.
- Example paintings after Bruegel, Bob Ross and Rembrandt, and demos.
- Python 3.11 to 3.15.

[Unreleased]: https://github.com/joce/oilpaint/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/joce/oilpaint/releases/tag/v0.1.0
