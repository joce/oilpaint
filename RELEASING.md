# Releasing oilpaint

oilpaint publishes to [PyPI](https://pypi.org/project/oilpaint/) from GitHub
Actions (`.github/workflows/publish.yml`) using **Trusted Publishing** (OIDC):
no API tokens. The workflow runs when a **GitHub Release is published**; a bare
`git push` of a tag does *not* trigger it.

## One-time setup

1. On PyPI, add a *pending* trusted publisher (Account → Publishing → Add a new
   pending publisher, GitHub tab): project `oilpaint`, owner `joce`, repository
   `oilpaint`, workflow `publish.yml`, environment `pypi`. The first release
   then creates the project.
2. On GitHub, create the `pypi` environment (Settings → Environments → New
   environment) and add yourself as a required reviewer, so that every upload
   waits for your approval.

## Cutting a release

The version is derived from the git tag by `hatch-vcs`: there is **no manual
version bump**. `oilpaint/__init__.py` reads it from `oilpaint/_version.py`,
generated at build time. Tagging `vX.Y.Z` makes the build `X.Y.Z`; commits
after a tag get a `X.Y.(Z+1).devN` version automatically.

1. Move the `## [Unreleased]` notes in `CHANGELOG.md` under a new
   `## [X.Y.Z] - YYYY-MM-DD` heading and update the links at the bottom.
   Commit and push to `main`. Wait for CI to go green.
2. Create the release. This creates the `vX.Y.Z` tag *and* triggers publishing:

   ```bash
   gh release create vX.Y.Z --title "vX.Y.Z" --notes-file docs/release_next.md
   ```

   Write real notes (overview and highlights), not a one-liner; keep a copy as
   `docs/release_vX.Y.Z.md`.
3. The `publish` job pauses on the `pypi` environment. **Approve the deployment**:
   the run page shows *"Review pending deployments"* → tick `pypi` → *Approve and
   deploy*. After approval it uploads to PyPI.
4. Verify: <https://pypi.org/project/oilpaint/> shows the new version.

## Verify locally before releasing

```bash
uv build
uvx twine check dist/*
uv run tox -e wheel
```

The built filenames carry the version `hatch-vcs` derived from git
(`oilpaint-X.Y.Z...`). A clean checkout *at* the tag yields exactly `X.Y.Z`; a
dirty tree or post-tag commit yields a `.devN`/`+g<sha>` suffix. The `wheel` env
installs the built wheel on its own and checks that it imports and paints.
