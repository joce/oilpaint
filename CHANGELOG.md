# Changelog

All notable changes to oilpaint are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed

- Impasto has the structure of real oil paint. Bristle furrows come from a
  hair pattern over the whole stroke, with clumps a physical size: they
  drift, merge, split and fade along the stroke (a furrow lasts a width or
  two) and are the same size on any width of brush; stroke edges are ragged
  rather than ruled.
- An overloaded brush lays as thick as its load (its glob feeds the deposit
  directly; `IMPASTO` 0.2 -> 0.12), the same on a curve as on a straight
  stroke. Where it is set down it leaves one lumpy bead behind and beside
  where it landed, instead of a thin wall around its first footprint.
- A stiff brush lifts off clump by clump at the end of a stroke, pulling
  thick paint up into crests; soft brushes slide off clean. `lift` now
  applies to brushes too, by default from the stiffness (it was 0).
- Brushes no longer lift or push their own fresh paint step by step: no
  more ripple across the stroke at every step and echoes of the first
  footprint ("nested rectangles"). Thin strokes reach as far as before.
- A loaded dab is a mound of paint pulled up into a peak where the brush
  leaves it, not a thin coin with a rim.
- Knife slabs have ridges along the edges where the blade's ends press
  paint out; blade nicks come and go along the stroke; lifting off pulls
  paint up most along the trailing edge.
- Thick tube paint rounds off a little as it settles, so it looks softer
  than paint stiffened with impasto medium; mid-stiff paint slumps a little
  sooner at steep edges.
- A knife skimming with its blade's point or rounded end no longer leaves a
  sawtooth along that side of the slab; a skimming knife lays about what it
  did before on the same ridges (on ridges painted by this version, which
  stand a little higher, some 10-17% more).
- Ordinary strokes (below what the hair holds) lay the same paint as
  before and reach about as far, but spread it a little differently: by
  brush type, a few percent more or less area at correspondingly thinner or
  thicker paint (up to about 5% either way, depending on the strokes
  measured); the total paint laid is unchanged. Their ends are shaped by
  the lift-off, and their streaks wander.
- Overloaded thin lines (a liner or rigger loaded far beyond what it holds)
  carry further than before, about 7%: the glob feeds the line directly.
- A clean brush blending wet paint only moves it: only paint a stroke lays
  rounds off as it settles.
- `Stroke` and the stroke simulation now live in `oilpaint.stroke` (still
  importable from `oilpaint` and `oilpaint.brush`). A `Stroke` pickled by
  this version cannot be loaded by earlier versions.
- `tools/bench.py --compare` compares what the scenes paint byte for byte
  (a change in the sign of a zero counts, and is reported as such), and
  there are more scenes: dabs of several paints, knives lifted off, tapered
  and curving strokes, reverse glass and strokes running off the canvas.

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

- Glints along slanting crests, furrows and lips of thick paint are lines,
  as on level strokes, instead of strings of beads at the pixel steps.
  Domes and broad slopes keep their highlights exactly, painted round dabs
  to within a few percent.
- Dabs press down evenly: a round or oval dab was up to twice as thick at
  the two ends of its outline as in its middle, and its rim built up in two
  lips instead of all round.
- Paint squeezed out of an overloaded soft tool (a small mop, say) pressed
  hard now lies against the edge of its mark, instead of floating around it
  in a ring.
- A brush dragged hard over wet paint no longer moves more of a pixel's paint
  than is there: `drag` near 1 at high pressure (or a slick surface such as
  glass or copper under a stiff brush pressed fully down) left negative paint
  that grew into huge numbers and NaN in the render. Such strokes now paint
  differently: the share of paint dragged is capped at all of it.
- A stroke's own `mixing`, `pickup`, `drag`, `streaks`, `smear`, `blending`
  and `lift` are clipped to the same ranges as a `BrushType`'s.
- Strokes, brushes and canvases check their arguments where they come in and
  raise `ValueError` or `TypeError` naming the argument, before anything is
  painted: numbers that are not finite (a NaN pressure used to spread NaN
  over the canvas), a `brush` that is not a `BrushType` or `Brush` (it used
  to paint silently with the default round), malformed loads, a negative
  width, an out-of-range `taper` (outside 0..1; a `taper=1.5` used to shrink
  the whole stroke), seeds `np.random.default_rng` would refuse, canvas
  sizes that are not whole numbers, `color_at` off the canvas (it used to
  count from the end).
- `paint=[0.8, 0.1, 0.1]` (a colour given as a list) and loads given as a
  tuple work; a colour with other than three numbers is refused.
- `Brush.load_mix` without a `seed` paints the same each time (on the same
  machine and numpy build): its streaks follow from what is loaded.
- `copy.copy(canvas)` is an independent copy (it used to share the paint, so
  drying the copy dried the original); `Canvas.copy()` does the same.
- `canvas.spatter(at, None)` flicks nothing and returns 0; it used to dab
  clean drops over the wet paint and count them.
- Footprint tables no longer depend on what was painted before (two brushes
  whose aspects round alike to 3 decimals got whichever table was built
  first).
- A numba cache left from another number of spectral bands is refused with
  a clear `RuntimeError` instead of corrupting memory.
- A stroke that runs partly or wholly off the edge of the canvas paints
  what is on the canvas as the same stroke would on a canvas large enough
  to hold it: one running along an edge, one starting, ending or wandering
  off it, a dab cut by an edge, a stroke across a canvas a few pixels wide;
  brushes and knives. The paint the bristles past the edge would have laid
  was taken from all of them alike, so those on the canvas ran dry early (a
  hog flat with 30% of its width past the edge stopped 30 px short over a
  200 px stroke, a round half off 30 to 60 px short); steps wholly off the
  canvas used a rough estimate (a flat started off the canvas arrived with
  up to 1.4 times its paint, a round from far off with a fifth of it, and a
  hog crossing a tiny canvas kept three times what it should); and a knife
  kept the paint under its part past the edge and laid it on the visible
  part (up to 1.8 times the paint, 136 px too long). Now the tool goes on
  past the edge as over a canvas going on as it is at the edge: the
  bristles past it lay their paint there (with the glob that feeds them),
  a blade fills under its part past it (over the slab it has laid there),
  and the share of a squeezed-out ridge that lies past the edge goes there.
  What is on the canvas ends within a few percent of the larger canvas's
  (the dry-out at the very end of a stroke, which hangs on the paint the
  stroke itself laid past the edge, can differ by a few pixels). Strokes
  inside the canvas paint exactly as before. Strokes that run off the
  canvas now cost about what the same stroke costs on a canvas large enough
  to hold it, since the part off the canvas is simulated rather than
  estimated: the bench's `edges` scene takes 2.4 times as long (mostly knife
  steps wholly off the canvas), `knife-lift` a quarter longer, `painting`
  about 6% longer.
- A dab of a brush carrying more than one paint (loaded in bands, marbled
  with `load_mix`, or with paint picked up from the canvas) keeps every
  pigment when its paint is shared out between the bristles. That scaled
  each bristle's paint, which made and destroyed pigment (0.3% to 40% of a
  pigment over a dab or a stipple); paint now moves from bristle to
  bristle with what is in it, round a round's bundle the short way. Where
  every bristle holds the same paint (to a millionth) it is scaled as
  before: a single dab of one paint on a clean canvas is unchanged bit for
  bit, as are most stipples of one paint (where paint picked back up from
  the canvas differs in its last bits, a few differ at that level). The
  peak a loaded dab or knife pulls up as it lifts off is still made by
  scaling its paint pixel by pixel, which keeps volume but not each
  pigment where the mound holds several colours (up to a few percent on a
  three-colour dab); a knife now pulls up only its own paint.
- How much of a pixel paint covers no longer hangs on specks of rounding
  dust (1e-8 of paint either side of zero). A pixel's layer holding no more
  than 1e-6 of paint is empty: it covers nothing, hands no coverage on to
  the layer below when paint lands on it, is not stirred, and a ridge of no
  more paint than that covers nothing where it lands; a speck's neighbour
  thinner than a trace no longer tells `fix_coverage` that a pixel is fully
  covered. Dragging, the knife's flow and lifting paint off no longer leave
  or take less than nothing. Edges painted over such dust could come out a
  dotted line of fully and two-thirds covered pixels whose paint was the
  same. (This is about coverage: other decisions in the engine, such as
  which of three layers merge, still turn on thresholds a speck can tip, so
  a speck can still change a stroke at a pixel or two.)
- The slanted edge of thick paint is a smooth anti-aliased ramp, not a
  dotted line. Paint that slumped or was pushed into the pixels just past
  the edge the brush laid had no coverage of its own, so the renderer drew
  any of it thicker than a film as solid: along a slant, where it gathers
  in the pixel past each step of the staircase, a tooth at every step (566
  such dots along the impasto sheet's swoosh, about 415 now; a 30 degree
  edge of a loaded hog flat goes from 191 to 109, where a perfectly
  anti-aliased edge scores about 110 to 150). Within two pixels of the
  edge of a comparable film of the same paint, such paint now carries on
  that edge's ramp (it is covered no more than the covered paint beside
  it; a band of it along a level edge as fully as that); further out, or
  as a stray speck, it covers its pixel in proportion to its thickness
  beside the films of the same paint around it. The result doesn't depend
  on which way the canvas is scanned. Level edges and thin strokes keep
  their profile; no paint moves.

### Changed

- `Brush.load(paint, amount, across=None, depth="belly", *, width=None)`:
  `width` is now keyword-only, so the third argument is `across`, as the
  README always said. `load(RED, 6, 30)` raises a `TypeError` that says to
  write `load(RED, 6, width=30)`.
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
