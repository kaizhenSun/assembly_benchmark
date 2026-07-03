# Assembly Benchmark

Assembly Benchmark is an Isaac Lab extension for assembly-task reinforcement learning experiments.

The extension registers these tasks:

```text
Assembly-Benchmark-Direct-v0
Assembly-Benchmark-OneLeg-Direct-v0
```

## Requirements

- Isaac Lab installed and usable from the command line.
- Python environment from Isaac Lab, conda, or uv.

See the Isaac Lab installation guide:
https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html

## Installation

Install this extension in editable mode:

```bash
python -m pip install -e source/assembly_benchmark
```

If Isaac Lab is not installed in the active Python environment, use Isaac Lab's launcher instead:

```bash
<PATH_TO_ISAACLAB>/isaaclab.sh -p -m pip install -e source/assembly_benchmark
```

## Common Commands

List registered environments:

```bash
python scripts/list_envs.py
```

Run a zero-action smoke test:

```bash
python scripts/zero_agent.py --task=Assembly-Benchmark-Direct-v0
```

Run a random-action smoke test:

```bash
python scripts/random_agent.py --task=Assembly-Benchmark-Direct-v0
```

Run the generic R1 Pro assembly task:

```bash
python scripts/zero_agent.py --task=Assembly-Benchmark-Direct-v0 --num_envs 1 --device cuda:0 --headless
```

Run the explicit one_leg scene entry:

```bash
python scripts/zero_agent.py --task=Assembly-Benchmark-OneLeg-Direct-v0 --num_envs 1 --device cuda:0 --headless
```

`Assembly-Benchmark-Direct-v0` is the default alias for the generic assembly environment. It currently points to the
`one_leg` assembly scene, while `Assembly-Benchmark-OneLeg-Direct-v0` is the explicit one_leg task id. Assembly assets
are described through `assembly_benchmark.assembly` part and assembly specs, so each part owns its USD/URDF paths, mass,
reset pose, and scene key in one place. The shared `AssemblyBenchmarkEnv` uses generated Isaac Lab scene/env cfg classes
for each registered assembly. The one_leg variant loads the R1 Pro, shared LabTable asset, base tag/obstacles, and five
square-table parts. It uses the existing R1 Pro bimanual Differential IK controller with torso participation enabled and
keeps the sparse one_leg success condition: the square-table top and leg4 are assembled when their relative pose matches
one of the four valid table-corner targets. It does not include the FurnitureBench scripted FSM, AprilTag perception,
camera observations, or data collection pipeline.

Assembly parts default to `observe=False`, so the one_leg policy observation contains robot joint state, end-effector
poses, and target assembly poses, but not per-part root poses. Opt in on a specific part spec only when that part pose
should be exposed to the policy.

The object support table is stored once as `assets/furniture/lab_table/lab_table.usd` and used by the generic assembly
task. It contains the tabletop and four static collision legs with the same geometry, material, and friction settings
that were previously generated in scene code.

The default one_leg assets are pre-generated USD files under `assets/furniture/one_leg/usd`. The five dynamic
square-table parts use PhysX SDF mesh colliders (`resolution=512`, `subgrid=8`, `margin=0.001`, `narrow_band=0.01`) for
tighter insertion contact. Static obstacles keep their composed mesh colliders, and `base_tag` remains visual-only. The
runtime task loads these USD assets directly and no longer depends on `/tmp/assembly_benchmark/furniture_usd_cache`.
Regenerate them from the source URDFs inside an Isaac Lab runtime:

```bash
python scripts/tools/generate_assembly_usd_assets.py --assembly one_leg --overwrite
```

The generator reads the same assembly spec used by the runtime scene. The Python API now uses generic assembly naming,
while the on-disk `assets/furniture/...` directory remains a historical asset path for compatibility with checked-in USD
references.

Run the hard-coded one_leg assembly demo:

```bash
python scripts/tools/run_r1_pro_one_leg_scripted_assembly.py --num_envs 1 --device cuda:0
```

The scripted demo drives the whole-body IK action interface with a FurnitureBench-style waypoint sequence. It uses
real simulated contact for grasping, transport, insertion, and release; it does not kinematically attach leg4 to the
gripper or snap the part into place. Waypoints are routed through conservative table-clearance poses to reduce
collisions with the tabletop and scene objects. The default grasp orientation is top-down; after lift, the script
reorients the held leg in the air before insertion. In GUI runs, coordinate-axis markers are shown for the five one_leg
parts, both gripper frames, and planned grasp/insert frames by default; pass `--disable_markers` to hide them or adjust
`--marker_scale`.

Run keyboard teleoperation for the generic R1 Pro assembly whole-body IK scene:

```bash
python scripts/tools/run_r1_pro_keyboard_teleop.py --num_envs 1 --device cuda:0
```

The generic assembly task already uses whole-body IK with torso participation enabled, while keeping the same 16D
bimanual action format: left target pose + left gripper + right target pose + right gripper.

To let the torso joints participate in one joint bimanual IK solve during teleoperation:

```bash
python scripts/tools/run_r1_pro_keyboard_teleop.py --num_envs 1 --device cuda:0 --include_torso_in_ik
```

To directly adjust the torso joint angles from the keyboard while keeping arm IK unchanged:

```bash
python scripts/tools/run_r1_pro_keyboard_teleop.py --num_envs 1 --device cuda:0 --enable_torso_keys
```

To print all current robot joint angles with the periodic teleop diagnostics:

```bash
python scripts/tools/run_r1_pro_keyboard_teleop.py --num_envs 1 --device cuda:0 --print_joint_angles
```

The teleop script requires a GUI window. Use `W/S`, `A/D`, and `Q/E` to translate the active gripper, `Z/X`,
`T/G`, and `C/V` to rotate it, `N` to cycle between left/right/both control, `K` to toggle the active gripper,
`R` to reset, and `ESC` to quit. With `--enable_torso_keys`, press `P` to toggle torso mode; in torso mode,
`W/S`, `A/D`, `Q/E`, and `Z/X` adjust `torso_joint1-4`. Direct torso keys are only for tasks that are not already using
torso IK. It sends the existing bimanual IK action format and uses normal gripper-object contact for object
interaction. `--print_joint_angles` follows `--print_interval` and prints env0 joint positions in radians.

## Adding Assembly Scenes

Assembly scenes are registered from Python specs:

1. Add a spec module that returns an `AssemblySpec` with unique part `scene_key` values, source URDF/USD paths, reset
   poses, and at least one assembly relation. Parts default to `observe=False`; opt in only for poses that should enter
   policy observations.
2. Register it through `assembly_benchmark.assembly.register_assembly(name, factory)`.
3. Generate USD assets with `scripts/tools/generate_assembly_usd_assets.py --assembly <name> --overwrite`.
4. Run the generated Isaac Lab task id `Assembly-Benchmark-<Name>-Direct-v0`.

The default task alias remains `Assembly-Benchmark-Direct-v0`; per-scene task ids make training runs and `list_envs.py`
output explicit.

The R1 Pro assembly task follows Isaac Lab's normal CUDA/Fabric defaults. For a headless GPU check, use performance
rendering:

```bash
python scripts/zero_agent.py \
  --task=Assembly-Benchmark-Direct-v0 \
  --num_envs 1 --device cuda:0 --headless --rendering_mode performance
```

Run the visual R1 Pro Differential IK demo:

```bash
python scripts/tools/run_r1_pro_diff_ik.py --num_envs 1 --device cuda:0
```

For explicit CPU/USD compatibility debugging, pass both `--device cpu` and `--disable_fabric`. Disabling Fabric routes
reads and writes through USD and can make GUI mesh updates appear out of sync with marker updates:

```bash
python scripts/zero_agent.py \
  --task=Assembly-Benchmark-Direct-v0 \
  --num_envs 1 --device cpu --disable_fabric
```

Train with RSL-RL:

```bash
python scripts/rsl_rl/train.py --task=Assembly-Benchmark-Direct-v0
```

Play a trained RSL-RL policy:

```bash
python scripts/rsl_rl/play.py --task=Assembly-Benchmark-Direct-v0
```

Other RL backends are available under `scripts/rl_games`, `scripts/sb3`, and `scripts/skrl`.

## Project Layout

```text
source/assembly_benchmark/
  assembly_benchmark/
    assets/robots/r1_pro/              # Galaxea R1 Pro URDF, meshes, configs, generated USD
    assembly/                          # Generic assembly part/spec registry and Isaac config helpers
    controllers/                       # R1 Pro joint and Differential IK controllers
    robots/                            # Isaac Lab ArticulationCfg definitions
    tasks/direct/
      assembly_benchmark/              # generic R1 Pro assembly task registration, environment, config
  config/extension.toml                # Isaac Lab extension metadata

scripts/
  list_envs.py                         # list available tasks
  zero_agent.py                        # zero-action smoke test
  random_agent.py                      # random-action smoke test
  rsl_rl/                              # RSL-RL train/play scripts
  tools/convert_r1_pro_urdf.py         # R1 Pro URDF to USD conversion
  tools/generate_assembly_usd_assets.py # assembly URDF assets to checked-in USD conversion
  tools/run_r1_pro_diff_ik.py          # Visual R1 Pro Differential IK demo
```

## R1 Pro Assets

R1 Pro assets were migrated from:

```text
/home/kaizhen/nju_ws/GalaxeaManipSim/galaxea_sim/assets/r1_pro
```

Generate or refresh the fixed-base USD asset with Isaac Lab Python:

```bash
python scripts/tools/convert_r1_pro_urdf.py --force --headless
```

The runtime robot config expects:

```text
source/assembly_benchmark/assembly_benchmark/assets/robots/r1_pro/r1_pro_fixed.usd
```

## Development

Format and check code with pre-commit:

```bash
pip install pre-commit
pre-commit run --all-files
```

For VS Code indexing, run the `setup_python_env` task from `Tasks: Run Task` and provide the absolute path to the
Isaac Sim installation when prompted.
