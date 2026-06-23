# Assembly Benchmark

Assembly Benchmark is an Isaac Lab extension for assembly-task reinforcement learning experiments.

The extension registers these tasks:

```text
Assembly-Benchmark-Direct-v0
Assembly-R1Pro-Joint-Direct-v0
Assembly-R1Pro-IK-Direct-v0
Assembly-R1Pro-BlocksStackEasy-Joint-Direct-v0
Assembly-R1Pro-BlocksStackEasy-IK-Direct-v0
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
python scripts/zero_agent.py --task=Assembly-R1Pro-Joint-Direct-v0 --num_envs 1
```

Run a random-action smoke test:

```bash
python scripts/random_agent.py --task=Assembly-Benchmark-Direct-v0
python scripts/random_agent.py --task=Assembly-R1Pro-Joint-Direct-v0 --num_envs 1
```

Run the R1 Pro IK smoke task:

```bash
python scripts/zero_agent.py --task=Assembly-R1Pro-IK-Direct-v0 --num_envs 1
```

Run the migrated R1 Pro BlocksStackEasy task shells:

```bash
python scripts/zero_agent.py --task=Assembly-R1Pro-BlocksStackEasy-Joint-Direct-v0 --num_envs 1 --headless
python scripts/zero_agent.py --task=Assembly-R1Pro-BlocksStackEasy-IK-Direct-v0 --num_envs 1 --headless
```

The BlocksStackEasy migration includes the R1 Pro, tabletop, two dynamic colored blocks, SceneCfg-defined block
reset poses, sparse stack-success reward, and timeout/success termination. It does not include the original
GalaxeaManipSim expert solution, demo collection, RelaxedIK, or camera observation pipeline.

Run the scripted IK physical auto-grasp demo for BlocksStackEasy:

```bash
python scripts/tools/run_r1_pro_blocks_stack_easy_auto_grasp.py --num_envs 1 --device cpu --disable_fabric
```

This default mode uses real gripper-object contact after reset: the script does not attach, carry, or snap blocks by
writing their poses. It is a scripted IK demo, not an RL policy or the original GalaxeaManipSim expert. For deterministic
debugging with the old pose-attachment helper:

```bash
python scripts/tools/run_r1_pro_blocks_stack_easy_auto_grasp.py --grasp_mode kinematic --num_envs 1 --device cpu --disable_fabric
```

R1 Pro smoke tasks default to CPU simulation in `zero_agent.py` and `random_agent.py` unless `--device` is passed
explicitly. For a low-memory GPU smoke test, prefer headless performance mode:

```bash
python scripts/zero_agent.py --task=Assembly-R1Pro-IK-Direct-v0 --num_envs 1 --device cuda:0 --headless --rendering_mode performance
```

Run a deterministic IK tracking check:

```bash
python scripts/tools/check_r1_pro_ik.py --device cuda:0 --headless --rendering_mode performance
```

Run the visual R1 Pro Differential IK demo:

```bash
python scripts/tools/run_r1_pro_diff_ik.py --num_envs 1 --device cpu --disable_fabric
```

Run the layered R1 Pro robot bring-up checks:

```bash
python scripts/tools/check_r1_pro_robot.py --mode all --device cuda:0 --headless --rendering_mode performance --strict
python scripts/tools/check_r1_pro_robot.py --mode all --device cpu --disable_fabric --headless --rendering_mode performance --strict
```

For real-gravity dynamics diagnostics, run without `--strict` first. Drift threshold misses are reported as warnings
because this path is for actuator and gravity-compensation tuning:

```bash
python scripts/tools/check_r1_pro_robot.py --mode dynamics --enable_gravity --device cuda:0 --headless --rendering_mode performance
```

The R1 Pro smoke tasks currently disable robot-link gravity to validate asset loading, joint control, and IK without
requiring Galaxea's original passive-force gravity compensation layer.
The actuator gains are tuned to also hold the fixed-base robot under `--enable_gravity` diagnostics.

If GUI GPU simulation reports `PxgCudaDeviceMemoryAllocator failed to allocate memory`, rerun with CPU:

```bash
python scripts/zero_agent.py --task=Assembly-R1Pro-IK-Direct-v0 --num_envs 1 --device cpu --disable_fabric
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
    controllers/                       # R1 Pro joint and Differential IK controllers
    robots/                            # Isaac Lab ArticulationCfg definitions
    tasks/direct/assembly_benchmark/   # task registration, environment, config, agents
    tasks/direct/r1_pro/               # R1 Pro smoke tasks
    tasks/direct/r1_pro_blocks_stack_easy/ # R1 Pro BlocksStackEasy task shells
  config/extension.toml                # Isaac Lab extension metadata

scripts/
  list_envs.py                         # list available tasks
  zero_agent.py                        # zero-action smoke test
  random_agent.py                      # random-action smoke test
  rsl_rl/                              # RSL-RL train/play scripts
  tools/convert_r1_pro_urdf.py         # R1 Pro URDF to USD conversion
  tools/check_r1_pro_robot.py          # R1 Pro metadata, joint, dynamics, and IK checks
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

For VS Code indexing, run the `setup_python_env` task from `Tasks: Run Task` and provide the absolute path to the Isaac Sim installation when prompted.
