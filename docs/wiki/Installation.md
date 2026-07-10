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

## Verify the Install

```bash
python scripts/list_envs.py
```

The output should include `Assembly-Benchmark-Direct-v0` and the explicit assembly task ids.

## Environment Notes

- Use `--device cuda:0` for the first CUDA device.
- Set `PYTHONPATH=source/assembly_benchmark` only when importing local source without an editable install.
- See [[Running Tasks|Running-Tasks]] for the first smoke test and required camera flags.
