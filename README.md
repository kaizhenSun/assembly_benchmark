# Assembly Benchmark

Assembly Benchmark is an Isaac Lab extension for assembly-task experiments. It provides one generic R1 Pro assembly
environment, while individual assembly scenes are registered through `AssemblySpec`. The current default scene is
`one_leg`.

The GitHub Wiki is published from [docs/wiki](docs/wiki) at
https://github.com/kaizhenSun/assembly_benchmark/wiki.

## Overview

Registered tasks:

```text
Assembly-Benchmark-Direct-v0
Assembly-Benchmark-Cabinet-Direct-v0
Assembly-Benchmark-Chair-Direct-v0
Assembly-Benchmark-Desk-Direct-v0
Assembly-Benchmark-Drawer-Direct-v0
Assembly-Benchmark-Lamp-Direct-v0
Assembly-Benchmark-OneLeg-Direct-v0
Assembly-Benchmark-RoundTable-Direct-v0
Assembly-Benchmark-SquareTable-Direct-v0
Assembly-Benchmark-Stool-Direct-v0
```

`Assembly-Benchmark-Direct-v0` is the default generic entry point. It currently points to `one_leg`.
Every registered assembly also gets an explicit `Assembly-Benchmark-<Name>-Direct-v0` task id.

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

Run any explicit scene after generating its USD assets:

```bash
python scripts/zero_agent.py \
  --task=Assembly-Benchmark-Chair-Direct-v0 \
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

Preview assembled target poses:

```bash
python scripts/tools/preview_assembly_assembled_pose.py --assembly one_leg --num_envs 1 --device cuda:0
python scripts/tools/preview_assembly_assembled_pose.py --assembly chair --num_envs 1 --device cuda:0
python scripts/tools/preview_assembly_assembled_pose.py --assembly desk --num_envs 1 --device cuda:0 --disable_markers
```

The preview tool defaults to visual-only `ghost` physics for assembly parts so target poses are not pushed apart by
collision depenetration. Use `--disable_markers` or a smaller `--marker_scale` when you want to inspect only the
assembled geometry. To check the old dynamic collision behavior:

```bash
python scripts/tools/preview_assembly_assembled_pose.py \
  --assembly chair --num_envs 1 --device cuda:0 --physics_mode dynamic
```

Preview a single relation and print measured relative poses:

```bash
python scripts/tools/preview_assembly_assembled_pose.py \
  --assembly chair --mode relation_child --relation_index 2 --target_index 0 --print_poses \
  --num_envs 1 --device cuda:0
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
for assembly in cabinet chair desk drawer lamp one_leg round_table square_table stool; do
  PYTHONPATH=source/assembly_benchmark \
    python scripts/tools/generate_assembly_usd_assets.py --assembly "$assembly" --overwrite
done
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

## Assembly Scenes

`one_leg` is the current default assembly scene. It contains R1 Pro, LabTable, a base tag, one square table top, and four
square table legs. Obstacles are not part of the current scene spec.

The current success condition uses the primary relation `square_table_top -> square_table_leg4`. The task succeeds when
leg4 matches any valid table-corner target pose relative to the tabletop.

`chair`, `square_table`, `desk`, `round_table`, `drawer`, `lamp`, `stool`, and `cabinet` are ported from FurnitureBench
in the same spec style. Source URDF/mesh/tag assets live under `assets/furniture/<assembly>`; generate USD assets before
launching a newly ported task in Isaac Lab.

The chair spec keeps all five FurnitureBench assembly relations:

```text
chair_seat -> chair_leg1
chair_seat -> chair_leg2
chair_seat -> chair_back
chair_seat -> chair_nut1
chair_seat -> chair_nut2
```

The generic RL environment currently uses the first relation as `primary_relation` for sparse success. Multi-relation
full-assembly success is not modeled yet.

Assembly parts default to `observe=False`. The default policy observation contains robot joint state, end-effector
poses, and assembly target poses, but not per-part root poses. Set `observe=True` on a part spec only when that part pose
should be exposed to the policy.

The Python API now uses generic `assembly` naming. The on-disk path `assets/furniture/...` is kept as a historical asset
location to avoid cascading USD reference migrations.

## Adding Assembly Scenes

New scenes should follow the existing spec-driven pattern used by `one_leg`:

1. Add one spec module, for example `assembly/<name>.py`, that returns an `AssemblySpec`.
2. Define a unique `scene_key`, asset paths, initial pose, body type, mass or density, and reset/observe policy for each
   part.
3. Define at least one `AssemblyRelationSpec` with a parent, child, and child target poses in the parent frame.
4. Register the scene in `assembly/registry.py`.
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

scripts/
  list_envs.py                         # list registered tasks
  zero_agent.py                        # zero-action smoke test
  random_agent.py                      # random-action smoke test
  rsl_rl/                              # RSL-RL train/play scripts
  tools/generate_assembly_usd_assets.py # assembly URDF-to-USD generation
  tools/preview_assembly_assembled_pose.py # assembled target-pose preview
  tools/run_r1_pro_keyboard_teleop.py  # R1 Pro keyboard teleoperation
```

## Development

Run unit tests:

```bash
PYTHONPATH=source/assembly_benchmark python -m pytest tests/test_one_leg_assembly.py -q
```

Run Python syntax checks:

```bash
python -m compileall -q source/assembly_benchmark/assembly_benchmark/assembly
python -m compileall -q scripts/tools/preview_assembly_assembled_pose.py
python -m compileall -q scripts/tools/generate_assembly_usd_assets.py
```

Run a lightweight assembly contract check without Isaac Lab:

```bash
PYTHONPATH=source/assembly_benchmark python - <<'PY'
from assembly_benchmark.assembly import available_assemblies, make_assembly

assert available_assemblies() == (
    "cabinet",
    "chair",
    "desk",
    "drawer",
    "lamp",
    "one_leg",
    "round_table",
    "square_table",
    "stool",
)
for assembly_name in available_assemblies():
    assembly = make_assembly(assembly_name)
    assert assembly.part_names[0] == "base_tag"
    assert assembly.reset_part_names == assembly.part_names[1:]
    assert assembly.assembly_relations
    for part in assembly.parts:
        assert part.urdf_path(assembly.asset_root).is_file(), part.urdf_path(assembly.asset_root)
print("assembly contract ok")
PY
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

