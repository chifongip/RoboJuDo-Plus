# Repository Guidelines

## Project Structure & Module Organization

`robojudo/` contains the main Python package. Core runtime modules are split by role: `controller/`, `environment/`, `policy/`, and `pipeline/`; shared helpers live in `utils/`, `tools/`, and `config/`. Robot-specific configuration is under `robojudo/config/g1/` and `robojudo/config/h1/`.

Executable entry points are in `scripts/`, for example `scripts/run_pipeline.py` and `scripts/run_tracker_pipeline.py`. Tests are in `tests/`. Robot XMLs, meshes, and model checkpoints live in `assets/`. Documentation is in `docs/`. Optional vendored or patched dependencies are kept in `third_party/`; the Unitree C++/pybind package is under `packages/unitree_cpp/`.

## Build, Test, and Development Commands

- `pip install -e .`: install RoboJuDo in editable mode with dependencies from `requirements.txt`.
- `pip install -e ".[dev]"`: install development tools, including Ruff and pre-commit.
- `python submodule_install.py`: install enabled optional modules from `submodule_cfg.yaml`.
- `python submodule_install.py unitree_cpp`: install a specific optional module.
- `python scripts/run_pipeline.py`: run the default G1 simulation pipeline.
- `python scripts/run_pipeline.py -c g1_beyondmimic`: run a named configuration.
- `python -m unittest discover -s tests`: run the current test suite.

## Coding Style & Naming Conventions

Use Python 3.10+ and keep code formatted with Ruff. The configured line length is 120, and lint rules include pycodestyle, Pyflakes, import sorting, bugbear, and pyupgrade. Prefer snake_case for modules, functions, variables, and config names. Class names should use PascalCase. Keep new module names aligned with patterns such as `g1_*_cfg.py`, `*_policy.py`, `*_ctrl.py`, and `*_env.py`.

## Testing Guidelines

Tests currently use `unittest`; place new tests under `tests/` with names like `test_<feature>.py`. Existing tests validate that registered configs, controllers, environments, policies, and pipelines import cleanly. When adding a registry entry, cover imports without requiring real robot hardware.

## Commit & Pull Request Guidelines

Recent commits use short, imperative, lower-case summaries, often with a scope prefix such as `docs:`. Examples include `docs: clarify protomotion docstring in g1_cfg` and `restore default-pose hold on startup in run_pipeline.py`.

Pull requests should describe the behavior change, list tested commands, link related issues, and include screenshots or logs for UI, visualization, or robot/simulation behavior changes. Note any required assets, checkpoints, SDKs, or `submodule_cfg.yaml` updates.

## Security & Configuration Tips

Do not commit machine-specific credentials, robot network settings, or private model artifacts. Treat real-robot configs with care; document safety assumptions and default to simulation commands when giving examples.
