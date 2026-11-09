.PHONY: help tox spell spell-changed coverage pre-push

help:
	@echo "Available commands:"
	@echo "  make tox           - Run tox (installs uv + tox group if needed)"
	@echo "  make spell         - Run cspell on all files (installs npm packages if needed)"
	@echo "  make spell-changed - Run cspell on files changed on this branch or not committed"
	@echo "  make coverage      - Run tox coverage env with HTML report"
	@echo "  make pre-push      - Run tox, coverage, and spell-changed before pushing"

tox:
	@command -v uv >/dev/null 2>&1 || { echo >&2 "uv not found. Install from: https://docs.astral.sh/uv/getting-started/installation/"; exit 1; }
	@uv run tox --version >/dev/null 2>&1 || { echo "Installing tox group..."; uv sync --group tox --frozen; }
	uv run tox

node_modules/.bin/cspell:
	@command -v npm >/dev/null 2>&1 || { echo >&2 "npm not found. Install Node.js from: https://nodejs.org/"; exit 1; }
	npm install

spell: node_modules/.bin/cspell
	npm run spell

# Files changed on this branch (since it left the remote's default branch;
# failing that, its upstream or a local main), committed or not, plus
# untracked ones, so that the check still sees the work once it is committed.
# (core.quotePath=false: git would otherwise quote non-ASCII file names.)
spell-changed: node_modules/.bin/cspell
	@echo "Checking spelling in changed files..."
	@git="git -c core.quotePath=false"; \
	default=$$(git symbolic-ref -q --short refs/remotes/origin/HEAD); \
	base=; \
	for ref in "$$default" origin/main origin/master '@{upstream}' main master; do \
		[ -n "$$ref" ] && base=$$(git merge-base "$$ref" HEAD 2>/dev/null) && break; \
		base=; \
	done; \
	if [ -z "$$base" ]; then \
		echo "(no default branch, upstream or main to compare with: checking uncommitted files only)"; \
	else \
		case "$$ref" in origin/*) ;; *) echo "(no remote default branch: comparing with $$ref)";; esac; \
	fi; \
	files=$$( { \
		$$git diff --name-only --diff-filter=ACMR "$${base:-HEAD}"; \
		$$git ls-files --others --exclude-standard; \
	} | sort -u | while IFS= read -r f; do [ -f "$$f" ] && echo "$$f"; done ); \
	if [ -z "$$files" ]; then \
		echo "No changed files to check."; \
	else \
		echo "Files to check:"; \
		echo "$$files" | sed 's/^/  /'; \
		echo "$$files" | npx --no-install cspell --no-must-find-files --file-list stdin && echo "Spelling check passed!"; \
	fi

coverage:
	@command -v uv >/dev/null 2>&1 || { echo >&2 "uv not found. Install from: https://docs.astral.sh/uv/getting-started/installation/"; exit 1; }
	@uv run tox --version >/dev/null 2>&1 || { echo "Installing tox group..."; uv sync --group tox --frozen; }
	uv run tox -e coverage -- --cov-report=html

pre-push: tox coverage spell-changed
	@echo "All pre-push checks passed!"
