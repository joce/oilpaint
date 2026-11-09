# Changelog

All notable changes to oilpaint are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Stippling: a dab tapped lightly now prints the separate tips of a brush's
  longest bristles, dots that fill in as the pressure rises; `BrushType.tips`
  sets how many. A new `STIPPLER` brush, and `canvas.stipple(points, ...)`
  taps one brush at many points without reloading it.
- Spatter: `canvas.spatter(...)` flicks drops of paint off a brush, round where
  they land head-on, teardrops further off.
- A demo of both (`dabs_and_spatter.png`).
- Two studies after the masters, each painted mostly with one of the new
  tools: Seurat's *Gray Weather, Grande Jatte* (stippled dots) and
  Whistler's *The Falling Rocket* (spatter). Each is painted up coarse to
  fine from a public-domain photograph in `examples/reference`, every pass
  going only where the canvas still differs from it (`examples/_studies.py`).

- Glass (`surfaces.GLASS`): a clear, perfectly smooth and slick support, with
  whatever is behind it showing through.
- Reverse glass painting: `Canvas(..., surface=surfaces.GLASS, reverse=True)`
  paints on the back of the glass and shows it through the front, so the
  first paint laid is in front. A new `reverse_glass` demo shows a flower
  painted on the front, on the back in reverse order, and on the back in the
  wrong order.
- A fourth example painting, `examples/kandinsky_glass.py`: a reverse glass
  painting in the manner of Kandinsky's Murnau years, painted on the back of
  the glass in reverse order.
- `spectral.km_film`: the reflectance and transmittance of a paint film on
  its own.
- A dirty-brush demo (`examples/demo.py`, shown in the README): a full brush
  dragged through thin, medium and thick wet paint, then used again on clean
  canvas.

### Fixed

- Dabs press down evenly: a round or oval dab was up to twice as thick at
  the two ends of its outline as in its middle, and its rim built up in two
  lips instead of all round.
- Paint squeezed out of an overloaded soft tool (a small mop, say) pressed
  hard now lies against the edge of its mark, instead of floating around it
  in a ring.

### Changed

- Brushes stay dirty longer after working through thick wet paint. What a
  brush lifts no longer all goes into its tips: deeper wet paint comes
  further up the hairs, more so under pressure and with soft hair, and part
  of it goes into the body of the brush, where it lasts into the next
  strokes. A thin film still only dirties the tips.
- New lighting of the relief, closer to a painting photographed under a
  studio lamp: crisper bristle marks and edges on thick paint; sharp oil
  highlights on the crests and lips of wet impasto (duller once dry, none on
  thin lean paint); cast shadows from a lamp of some size, sharp at their
  foot and fading further away; less light from the room in grooves,
  hollows and at the foot of thick paint; highlights that roll off to white
  instead of clipping. Flat paint still renders in its own colour.
  `surface_height()` keeps more of the fine relief, so its values change (by
  up to about 0.8 px on impasto), and a very high `relief` can now shade
  deep grooves down to black.
- A lit `render()` checks its arguments before doing any work: `light` must be
  three finite numbers with z > 0 (a lamp in front of the picture), and
  `relief` and `gloss` finite and not negative; otherwise it raises
  `ValueError`. Unlit and reverse renders do not use them and do not check
  them.
- Rendering is faster: compositing, colour conversion and lighting run as
  compiled kernels, about 1.8 to 2.8 times faster lit and 3.5 to 8 times
  unlit (measured on 500 x 650 to 1300 x 1000 canvases), and renders in
  separate threads run in parallel.

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
