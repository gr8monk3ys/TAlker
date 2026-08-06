# AGENTS.md

Instructions for AI coding agents (and humans) working in this repository.

## Package manager

This project uses **Poetry**, not `uv` (despite what any external tooling
metadata may say). All dependency and environment commands go through
`poetry`.

## Setup

```bash
poetry install --with dev
```

As of this writing this is the first time the dependency graph has actually
resolved cleanly — there was no `poetry.lock` committed, and the raw pins in
`pyproject.toml` conflicted with each other (see "Known-good pins" below).
If `poetry install --with dev` ever starts failing again with a resolver
error, it is almost always a stale/conflicting version pin in
`[tool.poetry.dependencies]` — bump the offending package rather than
loosening the security pins at the bottom of that section.

## Verify command (the one that actually works)

```bash
poetry run pytest
```

equivalently:

```bash
make test
```

This is the command to run after making a change, and it is green on a
clean `poetry install --with dev`. Test config lives in
`[tool.pytest.ini_options]` in `pyproject.toml` (`testpaths = ["tests"]`,
coverage enabled by default).

## Known-good pins

Python is constrained to `>=3.10,<3.13` (not `<4.0`). The ML dependency
stack (`unstructured`, `chromadb`, `sentence-transformers`, `fastembed`)
does not yet support Python 3.13+, so declaring a wider range makes the
Poetry resolver try to satisfy versions of `unstructured`/`fastembed` for
3.13 that don't exist, and resolution fails outright.

The LangChain stack is on the **1.x line** (`langchain`, `langchain-core`,
plus `langchain-classic` for the legacy `chains`/`memory`/`retrievers`
APIs that `src/dashboard/llm.py` uses). The 0.3.x series has known CVEs
with no patched releases, so don't downgrade below the floors declared in
`pyproject.toml`. Direct SDK pins (`openai`, `anthropic`, `cohere`,
`google-generativeai`, `tiktoken`) were removed deliberately — nothing
imports them directly, and the `langchain-*` provider packages resolve
their own compatible SDK versions.

PDF handling uses `pypdf`, the maintained successor of `PyPDF2`. The old
`PyPDF2` package is unmaintained and its last release carries an unfixed
CVE — don't reintroduce it.

## Known gaps (not covered by `make test`, tracked separately)

`make check` (`black --check`, `ruff check`, `mypy`) does **not** currently
pass across the full `src/` tree — most files predate any enforcement of
this and were never formatted/linted against the checked-in config. This is
pre-existing and out of scope for incremental feature/bugfix PRs; please
don't bundle a repo-wide reformat into an unrelated change. If you touch a
file, it's good practice to leave the parts you touched black/ruff-clean
(as this PR does for `tests/conftest.py`, `tests/test_llm.py`, and
`tests/test_providers.py`), but a full-repo formatting pass should be its
own PR so it doesn't drown out real diffs.

`mypy src/dashboard/llm.py src/dashboard/evaluation.py` also has
pre-existing type errors (mostly `Optional`/`None`-typed attributes that
are later assigned real objects). Same caveat applies.
