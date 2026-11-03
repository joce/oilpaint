"""The package version: from the git tag (hatch-vcs), written to _version.py
when the package is built or installed (see pyproject.toml,
[tool.hatch.version])."""

try:
    from ._version import __version__
except ImportError:  # a source checkout that was never built or installed
    __version__ = "0.0.0+unknown"

__all__ = ["__version__"]
