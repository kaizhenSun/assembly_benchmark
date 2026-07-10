# Installation

## Requirements

- Isaac Lab installed and usable from the command line.
- Python 3.10 or newer in an Isaac Lab, conda, or uv environment.
- A local checkout of this repository.

See the [Isaac Lab installation guide](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html)
for Isaac Sim and Isaac Lab setup.

## Install the Extension

From the repository root:

```bash
python -m pip install -e source/assembly_benchmark
```

If Isaac Lab is not available in the active Python environment, use its launcher:

```bash
<PATH_TO_ISAACLAB>/isaaclab.sh -p -m pip install -e source/assembly_benchmark
```

Use the first command when the active environment can already import Isaac Lab. Use the launcher command when Isaac Lab
owns the Python runtime or when direct imports fail because Isaac Sim modules are missing.

## Verify the Install

```bash
python scripts/list_envs.py
```

The output should include `Assembly-Benchmark-Direct-v0` and the explicit assembly task ids.

After task registration is visible, run the smoke test in [[Running Tasks|Running-Tasks]] to verify that Isaac Sim can
construct the scene, load the R1 Pro and furniture USD assets, and initialize the camera.

## Environment Notes

- Use `--device cuda:0` for the first CUDA device.
- Run GUI tools without `--headless`; smoke tests and training can run headless.
- Set `PYTHONPATH=source/assembly_benchmark` only when importing local source without an editable install.
- See [[Running Tasks|Running-Tasks]] for the first smoke test and required camera flags.

## Common Setup Problems

- `ModuleNotFoundError: assembly_benchmark`: repeat the editable install from the repository root or set `PYTHONPATH`
  for a one-off source-tree command.
- Missing `carb`, `pxr`, or other Isaac Sim modules: activate the Isaac Lab environment or launch the command through
  `isaaclab.sh -p`.
- Camera startup errors: include `--enable_cameras` for generic environment runners.
