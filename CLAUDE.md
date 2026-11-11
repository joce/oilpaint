# Project Instructions for AI Agents

This file provides instructions and context for AI coding agents working on this project.

<!-- BEGIN BEADS INTEGRATION v:1 profile:minimal hash:1105d646 -->
## Beads Issue Tracker

This project uses **bd (beads)** for issue tracking. Run `bd prime` to see full workflow context and commands.

### Quick Reference

```bash
bd ready              # Find available work
bd show <id>          # View issue details
bd update <id> --claim  # Claim work
bd close <id>         # Complete work
```

### Rules

- Use `bd` for ALL task tracking — do NOT use TodoWrite, TaskCreate, or markdown TODO lists
- Run `bd prime` for detailed command reference and session close protocol
- Use `bd remember` for persistent knowledge — do NOT use MEMORY.md files

**Architecture in one line:** issues live in a local Dolt DB; sync uses `refs/dolt/data` on your git remote; `.beads/issues.jsonl` is a passive export. See https://github.com/gastownhall/beads/blob/main/docs/core-concepts/sync-concepts.md for details and anti-patterns.

## Agent Context Profiles

The managed Beads block is task-tracking guidance, not permission to override repository, user, or orchestrator instructions.

- **Conservative (default)**: Use `bd` for task tracking. Do not run git commits, git pushes, or Dolt remote sync unless explicitly asked. At handoff, report changed files, validation, and suggested next commands.
- **Minimal**: Keep tool instruction files as pointers to `bd prime`; use the same conservative git policy unless active instructions say otherwise.
- **Team-maintainer**: Only when the repository explicitly opts in, agents may close beads, run quality gates, commit, and push as part of session close. A current "do not commit" or "do not push" instruction still wins.

## Session Completion

This protocol applies when ending a Beads implementation workflow. It is subordinate to explicit user, repository, and orchestrator instructions.

1. **File issues for remaining work** - Create beads for anything that needs follow-up
2. **Run quality gates** (if code changed) - Tests, linters, builds
3. **Update issue status** - Close finished work, update in-progress items
4. **Handle git/sync by active profile**:
   ```bash
   # Conservative/minimal/default: report status and proposed commands; wait for approval.
   git status

   # Team-maintainer opt-in only, unless current instructions forbid it:
   git pull --rebase
   git push
   git status
   ```
5. **Hand off** - Summarize changes, validation, issue status, and any blocked sync/commit/push step

**Critical rules:**
- Explicit user or orchestrator instructions override this Beads block.
- Do not commit or push without clear authority from the active profile or the current user request.
- If a required sync or push is blocked, stop and report the exact command and error.
<!-- END BEADS INTEGRATION -->

## Issue sync in this repository

- Issues live in a local Dolt database under `.beads/embeddeddolt/`, which git ignores.
- `.beads/issues.jsonl` is how issues travel between clones here. Beads rewrites it after every change and stages it (`export.auto` and `export.git-add` in `.beads/config.yaml`). Commit it with the work it belongs to.
- Claude Code cloud sessions may push only their own branch, so GitHub refuses `bd dolt push` to `refs/dolt/data` there (HTTP 403). Do not retry it and do not work around it; the export file is the supported route.
- On a fresh clone, `bd bootstrap` builds the database from `.beads/issues.jsonl`. In a clone that already has a database, run `bd import .beads/issues.jsonl` after pulling. Import adds and updates but never deletes, so close issues instead of deleting them.
- `bd remember` memories are not in the export and are lost with a cloud session. Put anything that must last in an issue or in this file.
- In a cloud session `bd prime` reports "No git remote configured". Beads does not trust a repository under `/home` that belongs to another user's home directory (the session runs as root in `/home/user`). Issue commands and the export work as usual; what may be committed and pushed is decided by the user's instructions, not by that message.

## Build & Test

```bash
uv sync --group dev --frozen    # install the package and the development tools
uv run pytest                   # tests; the first run compiles the numba kernels (about 35 s)
make tox                        # black, ruff, pyright and pytest on every supported Python
make coverage                   # coverage run (at least 80% is required)
make spell                      # cspell over the whole repository
python tools/bench.py           # time the 16 benchmark scenes
python tools/bench.py --save DIR      # keep each scene's painted state
python tools/bench.py --compare DIR   # compare byte for byte with a kept state
```

`make pre-push` runs what CI runs: tox, coverage and the spelling check.

## Architecture Overview

`oilpaint` simulates oil painting: brushes and knives lay, drag, mix and lift paint on a canvas, and a renderer lights the result.

- `paint.py`, `spectral.py`, `palette.py`: pigments and Kubelka-Munk mixing over 24 spectral bands; `palette.match` finds a mix for a target colour.
- `canvas.py`: the `Canvas`, with two wet layers, coverage, dried height, the weave, drying and rendering.
- `brush.py`, `brushes.py`, `stroke.py`, `_knife.py`, `_geometry.py`, `_hair.py`, `_dab.py`: tools, how a stroke is sampled and run, footprints and the hair field.
- `_engine.py`: the numba kernels that move paint for each step of a stroke. `_render.py`: the numba kernels that light and composite the picture.
- `surface.py`, `surfaces.py`, `noise.py`, `reference.py`, `spline.py`, `png.py`: grounds, textures, reference images, paths and output.
- `examples/` holds the example paintings and their rendered images; `tools/bench.py` holds the benchmark scenes.

## Conventions & Patterns

- Kernels are `@njit(cache=True, error_model="numpy")`, without `fastmath` and without `parallel` in the stroke path, so that results repeat exactly.
- A change that should not alter the picture must leave `tools/bench.py --compare` identical on all scenes. A change that is meant to alter it is judged by measurements and amplified difference images, not by looking at two renders side by side.
- Paint must be conserved: what leaves the brush lands on the canvas or is accounted for off its edge. Tests check this.
- ruff caps function complexity at 13 everywhere except `_render.py`.
- Constants that the kernels use live in `_engine.py`.
