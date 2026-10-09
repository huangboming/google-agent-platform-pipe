# AGENTS.md

## Project

- This is an Open WebUI Pipe for Google Agent Platform.
- Preserve the low-dependency, single-file deployment experience.
- Do not add Gemini Developer API, ADC, Google SDK, or ADK support unless explicitly requested.

## Source of truth

- `src/google_agent_platform_pipe/` is the editable source.
- `dist/google_agent_platform.py` is generated and committed; never edit it directly.
- Generate it with `uv run python scripts/build.py`.
- Verify it with `uv run python scripts/build.py --check`.
- Project version and Open WebUI Function metadata come from `pyproject.toml`.

## Architecture

- Keep deterministic transformations in module-level pure functions.
- Keep HTTP, streaming, tool execution, and agent-loop orchestration in the imperative shell.
- Avoid compatibility layers unless an actual released contract requires one.

## Tests

- Test observable `pipes()` and `pipe()` behavior rather than private implementation details.
- Every behavior contract must run against both the source package and generated distribution.
- Tests must not require live Google credentials or external API calls.

## Validation

Run before completing changes:

```bash
uv sync --locked
uv run ruff check .
uv run ruff format --check .
uv run pytest
uv run python scripts/build.py --check
```

## Releases

- Record user-visible changes in `CHANGELOG.md`.
- A release tag must equal `v{project.version}`.
- Pushing the tag publishes the matching CHANGELOG section and generated Function.
