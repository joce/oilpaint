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
- `Canvas.copy()`: an independent copy of a canvas (its wet and dry paint,
  its surface and where its random numbers have got to), which paints on
  exactly as the original would. `copy.copy(canvas)` does the same.
- `tools/bench.py`, a benchmark of the stroke engine: sixteen scenes (small
  rounds, liners, wide flats, wet blending, fans, knives laid on and lifted
  off, dabs of several paints, tapered curves, reverse glass, strokes
  running off the canvas, stippling, spatter, drying and a small painting),
  each timed and fingerprinted with a hash of everything it leaves on the
  canvas and its brushes. `--save DIR` keeps each scene's state, and
  `--compare DIR` compares with a kept one byte for byte (a change in the
  sign of a zero counts, and is reported as such).

### Changed

- Where rounding used to decide (see Fixed), the outcome is now the one
  exact arithmetic gives, so pictures with such marks differ in their
  texture from before, as a different throw of the dice: strokes whose
  length is a whole number of steps, curves with a segment a whole even
  number of pixels long, strokes along a half-pixel or 45-degree axis,
  multi-paint dabs at the default pressure, brushes turned by 90 degrees,
  knives at whole-number positions and pressure 0.3, and footprints at some
  angles. All example images are regenerated. In particular:
  - a brush's sides are swapped when it is turned from the stroke's cross
    direction by more than 90 and at most 270 degrees; exactly edge-on, it
    keeps its sides at 90 and swaps them at 270. With `follow_path=True`,
    `angle=90`, 270, 450 and 630 draw as before; `-90` now draws as 270 and
    `-270` as 90 (they drew as 90 and as 270). With `follow_path=False`, a
    brush whose long side lies along the stroke has its sides swapped where
    they differed: `angle=0` on a level stroke drawn left to right, and
    `angle=90`, `-90` and `-270` on an upright one drawn downwards (90 and
    450 now agree, as do -90, 270 and 630);
  - a pixel centred on the stroke's axis sends half its paint to each side
    ridge (it sent all of it to one; the paint a narrow knife pressed out
    there went to neither side, and stayed on the blade);
  - a dab tapped a little more lightly than the default pressure fades its
    bristle tips in smoothly: with no glob of paint on the brush (a glob
    moves the band lower), the paint between them goes from all of it at
    0.5 to none below 0.42 (it dropped to half at once just below 0.5).
    Outside that band, dabs are as they were;
  - a dab's pattern of tips is kept by its width to the nearest pixel, a
    width of exactly half a pixel more rounding up (it rounded to even);
  - a knife going back over its own slab spreads it over more of each pixel
    whether or not it lays more paint there.

- The Bob Ross example paints its clouds the way he did. Titanium white is
  scrubbed straight onto the wet sky with little circles, along the top of
  each billow first and then down into the body; a clean, dry brush blends
  the base into the blue, fluffs it, and is drawn once across. A cloud is
  several such masses, the farthest first, so the bright top of one stands
  against the blended base of the one behind. (They were one long stroke of
  white each, over sky lifted beforehand with a clean brush, on a ruled
  base, and never blended: flat white from top to bottom.) Measured on the
  big cloud, where 1 is clean white and 0 the sky beside it: the white now
  falls from 0.85 near the top to 0.58 near the base instead of 0.95 to
  0.88; the base melts into the sky over 36 px (from 20% to 80%) instead of
  7, the top edge over 3 px instead of 2. The script only; the engine is
  unchanged.
- A wide brush pressed hard through thick paint no longer throws the paint
  under its middle out to its sides. Under a plate the paint has to travel
  to an edge to get out, and the further in the slower: now only paint
  within 20 px of the nearer edge of the brush's footprint gets out
  freely, and from d px in only the share 1 / (1 + ((d - 20) / 3)^2) of
  it. The rest stays under the brush: what it was to squeeze out of its
  hair stays on it, and goes on as a thicker stroke; wet paint it was to
  plough aside stays where it was, in its own layer. A white 120 px hog
  flat pressed at 0.9 down 5 units of wet blue, loaded 45 w / 36, now
  leaves side ridges 17 to 20 units high instead of 40 to 47 (five
  strokes, the median over each one's rows), still about 11 px wide at
  half height: about as high in px (1 unit ~ 0.6 px) as they are wide, as
  a 36 px flat's are, where they stood more than twice as high as wide.
  Their steepest outward fall, the median over the rows, is 2.1 units a px
  instead of 5.0 (3.0 to 3.9 at the steepest row instead of 7.1 to 9.7);
  the middle of the stroke stays 10.0 units thick instead of 7.7, furrowed
  by the hair; the bead where it was set down stands 20 units high instead
  of 54. On bare canvas its ridges stand 13 to 15 units high instead of 29
  to 33. So a wide, heavily loaded stroke no longer has dark rails along
  its sides and a black bar where it was set down. A clean or nearly dry
  wide brush dragged through wet paint takes up its share of what it moves
  as before, step by step, and leaves more of the wet paint under its
  middle (a clean 120 px hog flat drawn hard down 5 units of wet blue
  leaves 2.3 units of it there in the lower layer instead of 1.0, and,
  meeting it again, picks up a third more over the stroke). Wide dabs and
  stipples squeeze a little less out round their rims. A brush whose
  footprint is no more than 40 px across the stroke squeezes and ploughs as
  before: a flat up to 37 to 39 px, a round or filbert up to 30 to 52 px
  depending on its hair and how hard it is pressed (a sable round pressed
  hard, up to 26 px).
- The edges of a wide brush's stroke are anti-aliased where they were
  notched. The last pixel of a side ridge held paint it didn't cover: a
  film several times as thick as the paint beside it, which came out pale,
  so that wherever the edge of a heavy stroke stepped out by a pixel (its
  width wobbles: a "level" edge is a gentle curve) there was a pale notch,
  and the edge of a slanting stroke was a sawtooth. Now a side ridge covers
  as far as its paint reaches. On the edges of heavily loaded strokes of 70
  and 120 px flats (30 strokes each), the edge's position wavers from
  column to column by 0.119 and 0.106 px on average instead of 0.179 and
  0.132; across the slanting edge of a heavily loaded black 50 px stroke,
  237 of 240 columns keep an in-between pixel instead of 204 (ten strokes).
  The edge is softer, not paler along its length: the darkness summed
  across it is the same as before (2.03 and 2.02 rows' worth), but its
  last pixels are graded where they were either solid or notched, so many
  more of them come out somewhat paler than the stroke. Of the outer pixels
  holding over a unit, 23% and 27% come out more than 4 levels paler
  instead of 10% and 9%, 820 and 1707 more than 64 levels instead of 355
  and 633, 108 and 156 more than 128 levels instead of 93 and 100. It goes
  by how wide the footprint is across the stroke, as the slope cover below:
  not at all up to 28 px (there a side ridge's cover is as before), fully
  from 40 px.
- A brush's side ridges start where it is set down and join the bead
  there. They used to start a step late: everything the brush squeezed out
  and ploughed aside as it was set down went into the bead, so there was a
  notch at each corner. Now what any step would squeeze out and plough
  aside goes to its sides along the way it moves in that first step, and
  only what setting it down adds goes into the bead. On bare canvas this
  changes only a brush loaded beyond what its hair holds as it is set
  down. A load of 12 units at the default pressure (0.5) is painted
  exactly as before at any width from 10 to 120 px (hog flat, round and
  filbert), and so is a hard 10 px flat loaded 45 w / 36; pressed at 0.9,
  12 units changes rounds and filberts from 24 px; 20 units changes
  rounds from 16 px, and flats and filberts from 24 px (from 16 px pressed
  at 0.9); 60 units changes a 10 px brush too. Down wet paint it changes
  every stroke, whatever its load, at any size.
  There a small brush's stroke now starts flush, without a bead: the wet
  paint heaped where it was set down was what a step would have pushed
  aside. Heavily loaded hard flats down 5 units of wet blue (five strokes):
  the bead stands 0.02 units instead of 1.19 at 10 px (the paint added
  where it was set down, the mean across the middle of the mark, -2.35
  instead of -1.42: the brush takes up some of the blue there), 0.88
  instead of 2.03 at 16 px; the end still reads as an end, with a light
  top edge and its corners filled. Hard flats pressed at 0.9, loaded 45 w /
  36 (30 strokes each): just past the bead their side ridges stood 0.79,
  0.55 and 0.56 of their height further on at 36, 70 and 120 px on bare
  canvas and stand 1.05, 1.02 and 0.99 now; down 5 units of wet blue 0.17,
  0.34 and 0.46, now 0.84, 0.99 and 1.02. The bead comes out lower, most
  down wet paint at small sizes (the mean of five strokes, the range over
  up to six sets of five): down the wet blue 0.02 units instead of 1.19 at
  10 px, 0.88 instead of 2.03 at 16, 2.7 to 2.8 instead of 5.0 to 5.2 at
  24 px, 5.0 to 5.3 instead of 8.2 to 9.0 at 36, 5.6 to 5.9 instead
  of 9.0 at 40, 10.5 to 11.0 instead of 13.6 to 14.2 at 70 and 18.6 to 20.0
  instead of 21.2 to 23.3 at 120; on bare canvas 3.9 to 4.0 instead of 4.8
  at 24 px, 6.5 to 7.0 instead of 7.9 to 8.1 at 40, 11.9 to 12.7 instead
  of 13.3 to 13.9 at 70 and 19.3 to 21.7 instead of 20.6 to 23.2 at 120. A
  120 px flat's bead stands 0.79 to 1.17 times as high as the side ridges
  beside it down the bed (0.91 to 1.36 before) and 1.12 to 1.72 times on
  bare canvas (1.23 to 1.84). Knives, dabs and stipples are painted
  exactly as before.
- Where a wide brush's own paint at the edge of its stroke ends in a slope,
  the last pixel of it (if it holds at least `MIN_FILM`) covers as much of
  itself as the slope carried on says, instead of as little as the thicker
  paint beside it would: the paint that settles there once the brush has
  passed no longer leaves a pale band along the edge. On the level edges
  of heavy strokes of 120 px and 70 px flats on bare canvas, the outer
  pixels holding over a unit come out 22 and 21 levels paler than the
  stroke on average (30 strokes), against 86 and 72 without this (ten
  strokes) and 8.5 and 10.3 before the side ridges' cover reached as far
  as their paint (see above: the edges are graded now). Other strokes'
  paint, the lower layer and
  thinner traces keep their coverage, and so does a foot whose neighbours
  along the edge hold paint more than 15% unlike: the steps of a slanting
  edge's staircase keep their anti-aliasing (a heavily loaded wide stroke's
  slanting edge keeps as many in-between pixels as it would without this).
  Doing it again changes nothing. It goes by how wide the stroke's
  footprint reached across it: not at all up to 28 px (a flat up to 26 or
  27 px, a round or filbert up to 18 to 36 px depending on its hair and the
  pressure), fully from 40 px, smoothly in between.
- Scumbling over dried impasto: a lightly pressed brush carrying little
  paint now leaves broken paint on the crests of the dried relief, with the
  hollows showing through, and carries what it does not lay further along;
  it used to lay a solid patch over crests and hollows alike until it ran
  dry. Pressed harder or loaded more, it reaches down into the hollows
  again: strokes pressed as hard as the default or harder cover dried paint
  practically as before, and nothing changes on bare canvas or on dried
  paint no rougher than the weave.
- A brush pressed hard through thick wet paint ploughs it: wet-in-wet
  marbling. Its bristles go down through the paint they carry (they used to
  ride on it) to a level set by pressure and stiffness, cut the old paint
  away, deeper along the lines of their clumps, push much of it up into
  ridges beside the stroke, and lay their own paint more into their grooves;
  once the stroke has passed, the old paint left between the grooves folds up
  over the new in proportion, so the stroke is streaked with tints of both
  paints. How much a brush ploughs grows smoothly with the stiffness of its
  hair, from a trace for sable to about half for a synthetic and most for
  hog, and its lanes follow its own hair pattern (more of them on a wider
  brush). A white hog flat pressed at 0.9 through about 5 units of wet blue
  now leaves about 45% of the blue in its path (it used to leave nearly nine
  tenths under an even, bluish white) and heaps it into ridges about twice as
  high as the bed. It sets in smoothly from a pressure of 0.4 to 0.6, and
  pixel by pixel from about 1.5 to 2.5 units of wet paint (as the stroke met
  it) and from a `body` of about 0.55 to 0.7 (tube paint: softer paint flows
  back into a groove), so a light touch, a thin film or paint thinned for
  wet-on-wet work is brushed exactly as before. Dabs, the brush handle and
  knives don't plough, nor do drops of spatter (a new `BrushType.ploughs`,
  False for them). How far the old paint folds over the new goes with the
  brush's `mixing`: a brush with none ploughs but doesn't marble. A
  `stiffness` outside 0 to 1 is taken as the nearest end of that range.
- Thick paint has the relief of real oil paint. The furrows a brush combs
  into it come from a pattern of hair clumps over the whole stroke, each
  clump a physical size (coarser for hog than for sable, the same on any
  width of brush): they drift, merge, split and fade along the stroke, a
  furrow lasting a width or two, and the edges of the stroke are ragged
  rather than ruled.
- An overloaded brush lays as thick as its load, the same on a curve as on
  a straight stroke: the glob of paint beyond what the hair holds feeds the
  deposit directly, so twice the load lays about twice as thick (the
  engine's `IMPASTO` goes from 0.2 to 0.12). Where it is set down it leaves
  one lumpy bead behind and beside where it landed, instead of a thin wall
  around its first footprint.
- A stiff brush lifts off clump by clump at the end of a stroke, a ragged
  end, each clump pulling thick paint up into a crest; soft brushes slide
  off clean. `lift` now applies to brushes as well as knives: by default it
  follows the stiffness (about 1 for hog, 0 for sable and other soft hair;
  it used to be 0 for every brush), and given explicitly it acts on any
  brush that lays paint, strokes and dabs alike.
- A loaded dab is a lumpy mound pulled up into a peak or two where the brush
  leaves it, not a thin coin with a rim.
- A brush no longer lifts or pushes the fresh paint it has just laid itself:
  no more ripple across the stroke at every step, and no echoes of its first
  footprint ("nested rectangles").
- Ordinary strokes (loaded below what the hair holds) lay the same total
  paint as before over about the same area, at correspondingly thinner or
  thicker paint (within about 5% for most brushes, 7% for a hog fan). They
  reach about as far, though the ragged lift-off end of a stiff brush can
  stop up to about a tenth sooner, and their streaks wander. Overloaded thin
  lines (a liner or rigger loaded far beyond what it holds) carry about 7%
  further, since the glob feeds the line directly.
- Paint settles more like oil paint: thick tube paint rounds off a little at
  its crests, so it looks softer than paint stiffened with impasto medium,
  and mid-stiff paint slumps a little sooner at steep edges. Only the paint
  a stroke lays rounds off: a clean brush blending wet paint only moves it.
- Knives: the slab has a ridge along each edge where the blade's ends press
  paint out, more the harder it is pressed; nicks in the blade score nearly
  straight lines that come and go along the stroke; lifting off pulls the
  knife's own paint up, most along the trailing edge. A knife skimming with
  its blade's point or rounded end no longer leaves a sawtooth along that
  side of the slab. A knife skimming over dry ridges lays somewhat more
  paint than before (some 7 to 17% in the cases measured).
- New lighting of the relief, closer to a painting photographed under a
  studio lamp: crisper bristle marks and edges on thick paint; sharp oil
  highlights on the crests and lips of wet impasto (duller once dry, none on
  thin lean paint), gathered along narrow crests so that a glint on a
  slanting crest, furrow or lip is a line, as on a level one, rather than a
  string of beads at the pixel steps (domes and round dabs keep their own);
  cast shadows from a lamp of some size, sharp at their foot and softer
  further away; less light from the room in grooves, hollows and at the
  foot of thick paint; highlights that roll off to white instead of
  clipping. Flat paint still renders in its own colour. `surface_height()`
  keeps more of the fine relief, so its values change (by up to about 0.7
  px on impasto), and a very high `relief` can now shade deep grooves down
  to black.
- Rendering is faster: compositing, colour conversion and lighting run as
  compiled kernels, about 2 to 2.5 times faster lit and 8 to 11 times
  unlit (measured on 500 × 650 and 1300 × 1000 canvases). The kernels
  release the GIL, so renders in separate threads run in parallel.
- Strokes are faster: each step works only on the part of its window the
  tool passes over, and paint is levelled, dragged and compared with less
  work per pixel. On the benchmark's scenes, against the previous version:
  wide and loaded brush strokes, fans, blending and tapered curves are about
  1.5 to 2 times faster, knife strokes (laid on or lifted off) about 1.6 to
  1.8 times, and a small painting of some 500 marks about 2 times; small
  strokes, liners, stippling and reverse glass about 1.1 to 1.4 times. Dabs,
  spatter and `dry()` take about as long as before. Strokes running off the
  canvas now cost what the same stroke costs on a canvas large enough to
  hold it, since the part past the edge is simulated (see Fixed); they take
  about as long as before, or a little less. All sixteen scenes together
  run about 1.6 times faster.
- `palette.match` is about 3 times faster, with the same results: it tries
  its candidate mixes in batches, in the same order as before.
- Strokes, brushes, canvases, paints, paths, `palette.match` and `render()`
  check their arguments where they come in, and raise `ValueError` or
  `TypeError` naming the argument, before anything is painted or changed.
  Among what used to go wrong quietly: a number that is not finite (a NaN
  pressure spread NaN over the canvas), a `brush` that is not a `BrushType`
  or `Brush` (it painted with the default round), malformed loads, a
  negative width, a `taper` outside 0..1 (`taper=1.5` shrank the whole
  stroke), seeds `np.random.default_rng` would refuse, canvas sizes that
  are not whole numbers, `color_at` off the canvas (it counted from the
  end), an empty path or one with points that are not finite, width and
  pressure functions that give values that are not finite, a `max_paints`
  that is not a whole number from 1 up, a paint `strength` of 0 or less and
  negative parts in `Paint.mix`. A lit `render()` needs `light` to be three
  finite numbers with z > 0 (a lamp in front of the picture), and `relief`
  and `gloss` finite and not negative; unlit and reverse renders do not use
  them and do not check them.
- A colour of three floats whose largest is between 1 and 2 (a colour on
  the 0..1 scale pushed a little too far) is refused, instead of being read
  as near-black on the 0..255 scale. A triple with a component of 2 or more
  is on the 0..255 scale, floats included, and numpy `uint8` values always
  are. A ground image (`Canvas(ground=...)`) of whole numbers is on the
  0..255 scale whatever its integer type, not only as `uint8`, and so is a
  `Reference` image of whole numbers whatever its values (one whose values
  were all 0 or 1 used to be read on the 0..1 scale); a float `Reference`
  image is still on the 0..1 scale unless a value is above 1.
- `Brush.load(paint, amount, across=None, depth="belly", *, width=None)`:
  `width` is now keyword-only, so the third argument is `across`, as the
  README always said. `load(RED, 6, 30)` raises a `TypeError` that says to
  write `load(RED, 6, width=30)`.
- `Paint.mix` takes bare paints or colours as one part each, alongside
  `(paint, parts)` pairs. `Paint.color` is read-only.
- A `Reference` keeps only the 16 softened maps it used most recently
  (`Reference.max_maps`), each the size of the canvas, instead of all of
  them; one asked for again after that is made again.
- `Stroke` and the stroke simulation now live in `oilpaint.stroke` (still
  importable from `oilpaint` and `oilpaint.brush`). A `Stroke` pickled by
  this version cannot be loaded by earlier versions; paints, references and
  canvases pickled by earlier versions still load.
- Brushes stay dirty longer after working through thick wet paint. What a
  brush lifts no longer all goes into its tips: deeper wet paint comes
  further up the hairs, more so under pressure and with soft hair, and part
  of it goes into the body of the brush, where it lasts into the next
  strokes. A thin film still only dirties the tips.
- The tables shared by every colour computation (`spectral.WAVELENGTHS`,
  `SPECTRUM_TO_RGB`, `LUMINANCE`, ...) are read-only, and so are
  `Canvas.top_k`, `top_s`, `top_v` and the same of `und`: views of the wet
  paint, which can be written through but not replaced.
  `from oilpaint.brush import *` gives the names `oilpaint.brush.__all__`
  lists: no longer the standard-library and typing names the module imports
  (`math`, `hashlib`, `dataclass`, `Any`, ...), `annotations`, or
  `value_noise_1d` (import it from `oilpaint.noise`). The painted result is
  the same to the last bit.

### Fixed

- A last-bit difference in a stroke's inputs or in the math library
  (another machine) no longer switches a discrete outcome at the places
  found, where a value fell exactly on the line it was decided by in exact
  arithmetic and a hair either side of it after rounding:
  - how many steps a path is cut into, and how many random numbers its
    textures draw, when a length is a whole number of steps (80 px in steps
    of 8), and how finely a Catmull-Rom or Bezier path is traced when a
    segment is a whole even number of pixels long;
  - whether a dab of two paints at the default pressure 0.5 printed the
    gaps between its bristle tips, and which cell of a dab's pattern of
    tips a pixel centred on its axis (at half-pixel positions) reads;
  - which side of a brush turned by exactly 90 degrees was its left;
  - whether a footprint's table reached its outermost point (about one in
    ten, at some angles);
  - which ridge takes the paint of a pixel centred on the stroke's axis (at
    half-pixel positions, and at 45 degrees between whole-number points;
    paint a narrow knife pressed out there went to neither);
  - for a knife at whole-number positions, which pixels lie half under its
    edge and whether paint it presses out lies past the end of its stroke;
  - whether a knife at pressure 0.3 spreads its slab, as its bead laid 0 or
    1e-31 of paint;
  - whether paint a knife shoves onto its own kind joins the film over it;
  - how a tool lifted off moves its paint (as one paint or pigment by
    pigment, and among which pixels), as untouched pixels came out at a
    factor of 1 or a hair from it, or a speck of dust at 0 or 1e-30.

  Other such places may remain.

- How wide the ridges a brush pushes out at its sides may be now grows with
  the brush at every width; it stopped growing at 7 px, so that a wide brush
  pressed hard through thick wet paint pushed it up into a narrow wall that
  read as a dark line along the lit stroke. A white 120 px hog flat pressed
  at 0.9 down 5 units of wet blue now leaves ridges 12 px wide at half
  height, from 9 (the set-up of `tests/test_ridges.py`, three seeds; 11
  px in the set-up of the entry above, which measures each row and takes
  the median; see that entry too for how much paint a wide brush pushes
  out at all). A brush whose limit stays under 7 px (a hog flat up to
  about 55 px, a round up to 33 to 43 px depending on its hair), and a wider
  one whose ridges stay that narrow, paints exactly as before.
- A stroke that runs partly or wholly off the canvas paints what is on the
  canvas as the same stroke would on a canvas large enough to hold it: one
  running along an edge, one starting, ending or wandering off it, a dab
  cut by an edge, a stroke across a canvas a few pixels wide; brushes and
  knives. The tool goes on past the edge as over a canvas going on as it is
  at the edge: the bristles past it lay their paint there (with the glob
  that feeds them), a blade fills under its part past it, and the share of
  a squeezed-out ridge that lies past the edge goes there. The paint the
  bristles past the edge would have laid used to be taken from all of them
  alike, so the part on the canvas ran dry early (a round half off the
  canvas stopped some 30 px short); steps wholly off the canvas used a
  rough estimate, so a stroke coming in from off the canvas arrived with
  too much paint or too little; and a knife laid the paint under its part
  past the edge on the part that showed. What is on the canvas now comes
  within a few percent of the larger canvas's (the dry-out at the very end
  of a stroke, which hangs on the paint the stroke itself laid past the
  edge, can differ by a few pixels). Strokes inside the canvas paint
  exactly as before.
- A dab of a brush carrying more than one paint (loaded in bands, marbled
  with `load_mix`, or with paint picked up from the canvas) keeps every
  pigment when its paint is shared out between the bristles. That used to
  scale each bristle's paint, which made and destroyed pigment (up to 40%
  of a pigment over a dab or a stipple); paint now moves from bristle to
  bristle with what is in it, round a round's bundle the short way. Where
  every bristle holds the same paint (to a millionth) it is scaled as
  before: a single dab of one paint on a clean canvas is unchanged bit for
  bit, as are most stipples of one paint.
- The peaks a loaded dab, or a knife, pulls up as it lifts off keep every
  pigment. They were made by scaling the mark's paint pixel by pixel, which
  kept the volume but, where the paint held several colours, made and
  destroyed pigment (up to a fifth of a band on a three-colour dab, up to
  0.3% on a three-colour knife); paint now moves into the peaks, mostly
  from the paint near them. A mark of one paint on a clean canvas is
  unchanged bit for bit.
- A brush dragged hard over wet paint no longer moves more of a pixel's paint
  than is there: `drag` near 1 at high pressure (or a slick surface such as
  glass or copper under a stiff brush pressed fully down) left negative paint
  that grew into huge numbers and NaN in the render. Such strokes now paint
  differently: the share of paint dragged is capped at all of it.
- The slanted edge of thick paint is a smooth anti-aliased ramp, not a
  dotted line. Paint that slumped or was pushed into the pixels just past
  the edge the brush laid had no coverage of its own, so the renderer drew
  any of it thicker than a film as solid: along a slant, where it gathers
  in the pixel past each step of the staircase, a tooth at every step.
  Within two pixels of the edge of a comparable film of the same paint,
  such paint now carries on that edge's ramp (it is covered no more than
  the covered paint beside it; a band of it along a level edge as fully as
  that); further out, or as a stray speck, it covers its pixel in
  proportion to its thickness beside the films of the same paint around
  it. The result doesn't depend on which way the canvas is scanned. Level
  edges and thin strokes keep their profile; no paint moves.
- How much of a pixel paint covers no longer hangs on specks of rounding
  dust (1e-8 of paint either side of zero). A pixel's layer holding no more
  than 1e-6 of paint is empty: it covers nothing, hands no coverage on to
  the layer below when paint lands on it, is not stirred, and a ridge of no
  more paint than that covers nothing where it lands; a speck's neighbour
  thinner than a trace no longer counts a pixel as fully covered. Dragging,
  the knife's flow and lifting paint off no longer leave or take less than
  nothing. Edges painted over such dust could come out a dotted line of
  fully and two-thirds covered pixels whose paint was the same. (This is
  about coverage: other decisions in the engine, such as which of three
  layers merge, still turn on thresholds a speck can tip, so a speck can
  still change a stroke at a pixel or two.)
- Dabs press down evenly: a round or oval dab was up to twice as thick at
  the two ends of its outline as in its middle, and its rim built up in two
  lips instead of all round.
- Paint squeezed out of an overloaded soft tool (a small mop, say) pressed
  hard now lies against the edge of its mark, instead of floating around it
  in a ring.
- A stroke's own `mixing`, `pickup`, `drag`, `streaks`, `smear`, `blending`
  and `lift` are clipped to the same ranges as a `BrushType`'s (`mixing`,
  `pickup`, `drag` and `streaks` to 0..2, `blending` and `lift` to 0..1,
  `smear` to 0 or more).
- `paint=[0.8, 0.1, 0.1]` (a colour given as a list) and loads given as a
  tuple work; a colour with other than three numbers is refused.
- `Paint(paint, opacity=..., strength=...)` applies them instead of ignoring
  them: `strength` multiplies the paint's own, and `opacity` makes it again
  from its colour. A paint mixed on the palette or built from optics has no
  colour to make it from, and refuses `opacity` (mix it with
  `palette.MEDIUM` instead).
- `Brush.load_mix` without a `seed` paints the same each time (on the same
  machine and numpy build): its streaks follow from what is loaded.
- `copy.copy(canvas)` is an independent copy: it used to share the paint,
  so drying the copy dried the original.
- `canvas.spatter(at, None)` flicks nothing and returns 0; it used to dab
  clean drops over the wet paint and count them.
- Footprint tables no longer depend on what was painted before (two brushes
  whose aspects round alike to 3 decimals got whichever table was built
  first).
- A band count changed in `spectral.py` but not in the engine and the
  renderer, or a numba cache compiled for another band count, is refused at
  import with a clear `RuntimeError`; such a cache used to corrupt memory.
- A canvas one pixel wide or high renders.
- `render(light=None)` lights the picture with the default lamp, as its type
  hint allows; a lit render used to raise `ValueError` for it.
- `typing.get_type_hints` works on the public API: it failed on `Canvas`'s
  methods on Python 3.14 and later.

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
