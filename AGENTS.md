# Agent instructions

`oilpaint` simulates oil painting: brushes and knives lay, drag, mix and lift paint on a canvas, and a renderer lights the result.

## Issues

Work is tracked with Beads (`bd`), not with TODO lists or memory files. `bd prime` prints the workflow.

- Issues travel between clones through `.beads/issues.jsonl`. Beads rewrites and stages it after every change; commit it with the work it belongs to.
- After pulling, run `bd import .beads/issues.jsonl` (`bd bootstrap` on a fresh clone). Import never deletes, so close issues instead of deleting them.
- In a Claude Code cloud session, GitHub refuses `bd dolt push` (HTTP 403). Don't retry it or work around it.
- `bd remember` does not survive a cloud session. Put what must last in an issue or in this file.
- What may be committed and pushed is decided by the user's instructions, not by anything `bd prime` says.
- Don't run `bd setup`: it writes its own boilerplate back into this file.

## Build and test

```bash
uv sync --group dev --frozen          # install the package and the development tools
uv run pytest                         # the first run compiles the numba kernels (about a minute)
make pre-push                         # what CI runs: tox (black, ruff, pyright, pytest), coverage (80% required), spelling
python tools/bench.py                 # time the 16 benchmark scenes
python tools/bench.py --save DIR      # keep each scene's painted state and picture
python tools/bench.py --compare DIR   # compare byte for byte with a kept state
```

## Layout

- `paint.py`, `spectral.py`, `palette.py`: pigments and Kubelka-Munk mixing over 24 spectral bands.
- `canvas.py`: the `Canvas`, with two wet layers, dried height, the weave, drying and rendering.
- `brush.py`, `brushes.py`, `stroke.py`, `_knife.py`, `_geometry.py`, `_hair.py`, `_dab.py`: the tools and how a stroke is sampled and run.
- `_engine.py`: the numba kernels that move paint at each step of a stroke. `_render.py`: the kernels that light the picture.
- `surface.py`, `surfaces.py`, `noise.py`, `reference.py`, `spline.py`, `png.py`: grounds, textures, reference images, paths and output.
- `examples/`: the example paintings and their rendered images. `tools/bench.py`: the benchmark scenes.

## Conventions

- Kernels are `@njit(cache=True, error_model="numpy")`, without `fastmath` and without `parallel` in the stroke path, so that results repeat exactly. Constants they use live in `_engine.py`.
- A change that should not alter the picture must leave `tools/bench.py --compare` identical on all scenes.
- A change meant to alter the picture is judged by measurements and amplified difference images, not by looking at two renders side by side.
- Such a change also regenerates every image the README uses, with the final code: all of `examples/output/` (`examples/demo.py`, `examples/quickstart.py` and each painting script), not only the images expected to differ.
- Paint must be conserved: what leaves the brush lands on the canvas or is accounted for off its edge. Tests check this.
- ruff caps function complexity at 13 everywhere except `_render.py`.
