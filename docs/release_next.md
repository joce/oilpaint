# Unreleased

## Overview

This release makes thick paint look and behave like oil paint, and lights it the way a painting is photographed under a studio lamp. Brushes comb wandering furrows, set down a bead of paint and lift off clump by clump; dabs leave mounds, knives press ridges out at the ends of the blade. It adds stippling, spatter and reverse glass painting, with three new example paintings. Strokes that run off the canvas now paint as they would on a larger canvas, bad arguments raise clear errors instead of painting something odd, and strokes, renders and `palette.match` are faster.

## Highlights

- **Impasto with the structure of oil paint.** Bristle furrows a physical size that drift, merge and fade along the stroke; a lumpy bead where an overloaded brush lands; stiff brushes lifting off clump by clump into crests (`lift` now works on brushes); dabs that leave a mound pulled up into a peak; ridges at the ends of a knife blade. Twice the load lays about twice as thick, on a curve as on a straight stroke.
- **New lighting.** Soft cast shadows from a lamp of some size, less light in grooves and hollows, and sharp oil highlights on the crests of wet paint that run as lines along slanting crests and roll off to white. Flat paint keeps its colour. Rendering is about twice as fast lit and about ten times unlit, and renders in separate threads run in parallel.
- **Stippling and spatter.** `canvas.stipple` taps a brush at many points, printing the tips of its longest bristles when tapped lightly; `canvas.spatter` flicks drops that land round or stretch into teardrops. Studies after Seurat and Whistler show them off.
- **Reverse glass painting.** `surfaces.GLASS` and `Canvas(..., reverse=True)` paint on the back of the glass and show it through the front, as in a new study in the manner of Kandinsky's Murnau years.
- **Strokes off the edge.** A stroke that runs partly or wholly off the canvas paints what shows as the same stroke would on a canvas large enough to hold it, for brushes and knives alike.
- **Clear errors.** Strokes, brushes, canvases, paints, paths, `palette.match` and `render()` check their arguments and name the one that is wrong, before anything is painted. `Brush.load`'s `width` is now keyword-only.
- **Faster.** Wide and loaded strokes, blending and knife strokes are about 1.5 to 2 times as fast, a mixed painting about twice, and `palette.match` about three times. Small strokes are a little faster; dabs, spatter and strokes running off the canvas take about as long as before.
