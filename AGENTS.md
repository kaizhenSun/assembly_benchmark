# Repository Guidelines

## Project Structure & Module Organization

This repository is an Isaac Lab extension for assembly-task experiments. Core Python code lives in
`source/assembly_benchmark/assembly_benchmark/`: `assembly/` defines `AssemblySpec` scene data,
`tasks/direct/assembly_benchmark/` contains the environment and agent configs, `robots/`,
`controllers/`, and `sensors/` hold reusable runtime components, and `assets/` stores robot and
furniture USD/URDF/mesh assets. Entry-point scripts are under `scripts/`, with RL backend wrappers
in `scripts/rsl_rl`, `scripts/rl_games`, `scripts/sb3`, and `scripts/skrl`. Tests live in `tests/`.
Wiki source is maintained in `docs/wiki/`.

## Build, Test, and Development Commands

- `python -m pip install -e source/assembly_benchmark`: install the extension in editable mode.
- `<PATH_TO_ISAACLAB>/isaaclab.sh -p -m pip install -e source/assembly_benchmark`: install via the
  Isaac Lab launcher when Isaac Lab is not in the active environment.
- `python scripts/list_envs.py`: list registered task IDs.
- `python scripts/zero_agent.py --task=Assembly-Benchmark-Direct-v0 --num_envs 1 --device cuda:0 --headless --enable_cameras`: run a smoke test.
- `python -m pytest tests`: run unit tests.
- `pre-commit run --all-files`: run Ruff, formatting, codespell, license, and file hygiene hooks.

## Coding Style & Naming Conventions

Target Python 3.10+ and keep lines at or below 120 characters. Ruff handles linting, import sorting,
and formatting; use `pre-commit install` before regular development. Follow existing naming:
modules and scene keys use `snake_case` such as `one_leg`; explicit task IDs use
`Assembly-Benchmark-<Name>-Direct-v0`; test functions use `test_<behavior>`.

## Testing Guidelines

Use pytest. Keep fast, dependency-light checks in `tests/`; gate Isaac Sim-only tests with
`pytest.mark.skipif(importlib.util.find_spec("carb") is None, ...)` as existing tests do. When adding
an assembly, cover registry behavior, part names, relation targets, and validation errors. Run
`python -m pytest tests` before opening a PR.

## Commit & Pull Request Guidelines

Recent commits use short, imperative or sentence-case summaries, often with a period, for example
`Refactor the project framework.` Keep commits focused and mention generated asset changes when
USD/URDF outputs are updated. Pull requests should include a concise description, affected task IDs
or scripts, test results, and screenshots or logs for visual Isaac Lab changes.

## Assets & Configuration Tips

Regenerate USD assets with `python scripts/tools/generate_assembly_usd_assets.py --assembly <name> --overwrite`.
Large assets belong in Git LFS; the pre-commit hook rejects added files over 2 MB.
