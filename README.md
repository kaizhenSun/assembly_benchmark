# Assembly Benchmark

Assembly Benchmark is an Isaac Lab extension for assembly-task experiments. It provides one generic R1 Pro assembly
environment, while individual assembly scenes are registered through `AssemblySpec`. The current default scene is
`one_leg`.

For architecture details, see [DESIGN.md](source/assembly_benchmark/docs/DESIGN.md).

## Overview

Registered tasks:

```text
Assembly-Benchmark-Direct-v0
Assembly-Benchmark-OneLeg-Direct-v0
```

`Assembly-Benchmark-Direct-v0` is the default generic entry point. It currently points to `one_leg`.
`Assembly-Benchmark-OneLeg-Direct-v0` is the explicit task id for the `one_leg` scene.

## Requirements

- Isaac Lab installed and usable from the command line.
- A Python environment from Isaac Lab, conda, or uv.

Isaac Lab installation guide:
https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html

## Installation

Install this extension in editable mode:

```bash
python -m pip install -e source/assembly_benchmark
```

If Isaac Lab is not installed in the active Python environment, use the Isaac Lab launcher:

```bash
<PATH_TO_ISAACLAB>/isaaclab.sh -p -m pip install -e source/assembly_benchmark
```

## Task Entry Points

List registered environments:

```bash
python scripts/list_envs.py
```

Run the default assembly task:

```bash
python scripts/zero_agent.py \
  --task=Assembly-Benchmark-Direct-v0 \
  --num_envs 1 --device cuda:0 --headless --enable_cameras
```

Run the explicit `one_leg` scene:

```bash
python scripts/zero_agent.py \
  --task=Assembly-Benchmark-OneLeg-Direct-v0 \
  --num_envs 1 --device cuda:0 --headless --enable_cameras
```

The assembly scene includes an RGB work camera. Generic Isaac Lab runners must be launched with `--enable_cameras`.
Tool scripts may set this automatically when they own the camera setup.

## Common Commands

Zero-action smoke test:

```bash
python scripts/zero_agent.py \
  --task=Assembly-Benchmark-Direct-v0 \
  --num_envs 1 --device cuda:0 --headless --enable_cameras
```

Random-action smoke test:

```bash
python scripts/random_agent.py \
  --task=Assembly-Benchmark-Direct-v0 \
  --num_envs 1 --device cuda:0 --headless --enable_cameras
```

Preview the assembled `one_leg` target pose:

```bash
python scripts/tools/preview_one_leg_assembled_pose.py --num_envs 1 --device cuda:0
```

Run the `one_leg` scripted assembly demo:

```bash
python scripts/tools/run_r1_pro_one_leg_scripted_assembly.py \
  --num_envs 1 --device cuda:0 --enable_cameras
```

Run R1 Pro keyboard teleoperation:

```bash
python scripts/tools/run_r1_pro_keyboard_teleop.py \
  --num_envs 1 --device cuda:0 --enable_cameras
```

Regenerate assembly USD assets:

```bash
python scripts/tools/generate_assembly_usd_assets.py --assembly one_leg --overwrite
```

Train with RSL-RL:

```bash
python scripts/rsl_rl/train.py \
  --task=Assembly-Benchmark-Direct-v0 \
  --num_envs 1 --device cuda:0 --headless --enable_cameras
```

Play an RSL-RL policy:

```bash
python scripts/rsl_rl/play.py \
  --task=Assembly-Benchmark-Direct-v0 \
  --num_envs 1 --device cuda:0 --headless --enable_cameras
```

Other RL backends are available under `scripts/rl_games`, `scripts/sb3`, and `scripts/skrl`.

## one_leg Scene

`one_leg` is the current default assembly scene. It contains R1 Pro, LabTable, a base tag, obstacles, one square table
top, and four square table legs.

The current success condition uses the primary relation `square_table_top -> square_table_leg4`. The task succeeds when
leg4 matches any valid table-corner target pose relative to the tabletop.

Assembly parts default to `observe=False`. The default policy observation contains robot joint state, end-effector
poses, and assembly target poses, but not per-part root poses. Set `observe=True` on a part spec only when that part pose
should be exposed to the policy.

The Python API now uses generic `assembly` naming. The on-disk path `assets/furniture/...` is kept as a historical asset
location to avoid cascading USD reference migrations.

## Adding Assembly Scenes

New scenes should follow the existing spec-driven pattern used by `one_leg`:

1. Add a spec module that returns an `AssemblySpec`.
2. Define a unique `scene_key`, asset paths, initial pose, body type, mass, and reset/observe policy for each part.
3. Define at least one `AssemblyRelationSpec` with a parent, child, and child target poses in the parent frame.
4. Register the scene with `assembly_benchmark.assembly.register_assembly(name, factory)`.
5. Generate USD assets:

```bash
python scripts/tools/generate_assembly_usd_assets.py --assembly <name> --overwrite
```

Registration creates an explicit task id:

```text
Assembly-Benchmark-<Name>-Direct-v0
```

## Project Layout

```text
source/assembly_benchmark/
  assembly_benchmark/
    assembly/                          # AssemblySpec, part specs, registry, and Isaac cfg helpers
    assets/robots/r1_pro/              # R1 Pro URDF, meshes, config, and generated USD
    controllers/                       # R1 Pro joint and Differential IK controllers
    robots/                            # Isaac Lab ArticulationCfg definitions
    tasks/direct/assembly_benchmark/   # generic assembly environment, cfg generation, and task registration
  config/extension.toml                # Isaac Lab extension metadata
  docs/DESIGN.md                       # architecture-level design document

scripts/
  list_envs.py                         # list registered tasks
  zero_agent.py                        # zero-action smoke test
  random_agent.py                      # random-action smoke test
  rsl_rl/                              # RSL-RL train/play scripts
  tools/generate_assembly_usd_assets.py # assembly URDF-to-USD generation
  tools/preview_one_leg_assembled_pose.py # one_leg assembled pose preview
  tools/run_r1_pro_keyboard_teleop.py  # R1 Pro keyboard teleoperation
```

## Development

Run unit tests:

```bash
pytest tests
```

Run a Python syntax check:

```bash
python -m py_compile scripts/tools/preview_one_leg_assembled_pose.py
```

Use pre-commit:

```bash
pip install pre-commit
pre-commit run --all-files
```

For CPU/USD compatibility debugging, disable Fabric:

```bash
python scripts/zero_agent.py \
  --task=Assembly-Benchmark-Direct-v0 \
  --num_envs 1 --device cpu --disable_fabric --enable_cameras
```
