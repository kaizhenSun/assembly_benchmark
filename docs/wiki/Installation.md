# Installation

## Requirements

- Isaac Lab installed and usable from the command line.
- A Python environment from Isaac Lab, conda, or uv.
- Access to the repository checkout that contains `source/assembly_benchmark`.

Isaac Lab installation guide:

https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html

## Editable Install

Install this extension in editable mode from the repository root:

```bash
python -m pip install -e source/assembly_benchmark
```

If Isaac Lab is not installed in the active Python environment, use the Isaac Lab launcher:

```bash
<PATH_TO_ISAACLAB>/isaaclab.sh -p -m pip install -e source/assembly_benchmark
```

## Verify the Install

List the registered Gym task ids:

```bash
python scripts/list_envs.py
```

Run a zero-action smoke test:

```bash
python scripts/zero_agent.py \
  --task=Assembly-Benchmark-Direct-v0 \
  --num_envs 1 --device cuda:0 --headless --enable_cameras
```

The assembly scene includes an RGB work camera. Generic Isaac Lab runners must be launched with `--enable_cameras`.
Dedicated tool scripts may set this internally when they own camera setup.

## Environment Notes

- Use `--device cuda:0` when running on the first CUDA device.
- Use `--headless` for smoke tests that do not need a viewer.
- If importing local source modules outside an installed environment, set `PYTHONPATH=source/assembly_benchmark`.
- Generate USD assets before launching a newly ported assembly scene if its generated assets are not already available.

Example asset generation command:

```bash
python scripts/tools/generate_assembly_usd_assets.py --assembly <name> --overwrite
```
