# Project Structure

Assembly Benchmark separates assembly metadata, reusable runtime components, checked-in assets, and executable tools.
This page maps the functional files without expanding generated USD contents, meshes, textures, or Python caches.

## Repository Layout

```text
source/assembly_benchmark/
  assembly_benchmark/
    assembly/                         # assembly specs, relations, registry, and Isaac asset cfg conversion
    tasks/direct/assembly_benchmark/  # generated scene/env cfgs, Gym registration, runtime environment, RL cfgs
    robots/                           # R1 Pro articulation configuration and joint/link constants
    controllers/                      # bimanual joint-position and Differential IK controllers
    sensors/                          # VT-Refine-style tactile sensor and scene configuration helpers
    assets/
      furniture/                      # assembly URDF, mesh, tag, and generated USD assets
      robots/r1_pro/                  # R1 Pro URDF, meshes, and generated fixed-base USD
      sensors/vt_refine_tactile/      # gripper tactile-pad collision meshes
  config/extension.toml               # Isaac Lab extension metadata

scripts/
  list_envs.py                        # registered task discovery
  zero_agent.py                       # zero-action environment smoke test
  random_agent.py                     # random-action control smoke test
  tools/                              # asset conversion, previews, IK, teleoperation, scripted assembly
  diagnostics/                        # focused physics and contact diagnostics
  rsl_rl/                             # RSL-RL train/play wrappers
  rl_games/                           # RL-Games train/play wrappers
  sb3/                                # Stable-Baselines3 train/play wrappers
  skrl/                               # skrl train/play wrappers

tests/                                # assembly, robot, tactile, and asset contract tests
docs/wiki/                            # canonical GitHub Wiki source
.github/workflows/sync-wiki.yml       # publishes docs/wiki to the GitHub Wiki
```

## Runtime Flow

```text
AssemblySpec in assembly/<name>.py
  -> assembly/registry.py
  -> assembly/isaac.py converts part metadata to Isaac Lab asset cfgs
  -> assembly_benchmark_env_cfg.py builds scene and environment cfg classes
  -> task __init__.py registers default and explicit Gym task ids
  -> AssemblyBenchmarkEnv creates the scene and runtime state
  -> R1 Pro controller applies actions
  -> optional tactile sensors extend scene data and policy observations
```

All registered assemblies share this path. A new scene normally adds metadata and assets rather than a new environment
class.

## Core Package

### Assembly definitions

- `assembly/specs.py` defines parts, target poses, relations, validation, and asset-generation metadata.
- `assembly/<name>.py` contains the scene-specific parts, initial poses, physical properties, and assembly relations.
- `assembly/registry.py` maps stable assembly names to their spec factories.
- `assembly/isaac.py` turns a part spec into an Isaac Lab `AssetBaseCfg` or `RigidObjectCfg`.

### Task configuration and runtime

- `assembly_benchmark_env_cfg.py` defines the shared R1 Pro scene and generates one scene/env cfg class per registered
  assembly.
- The task package `__init__.py` registers the default alias and each explicit task id, together with RL backend cfgs.
- `assembly_benchmark_env.py` implements reset, observation, action, reward, success, and optional tactile integration.
- `agents/` stores PPO configurations consumed by the RL wrapper scripts.

### Robot, control, and sensing

- `robots/r1_pro.py` defines the R1 Pro articulation, joint groups, end-effector links, home pose, and actuator
  settings.
- `controllers/r1_pro.py` maps the 16D bimanual action to arm, gripper, and optional torso joint targets through
  Differential IK.
- `sensors/vt_refine_tactile.py` builds the four tactile arrays from SDF contact geometry and exposes optional policy
  observations.

## Assets

- `assets/furniture/<assembly>/` groups source URDFs, meshes and tags, plus generated USD folders for each assembly.
- `assets/furniture/lab_table/` contains the shared work table used by the base scene.
- `assets/robots/r1_pro/` contains the robot source model and fixed-base USD loaded by the environment.
- `assets/sensors/vt_refine_tactile/` contains the flat pad meshes attached to the gripper fingers.

Assembly USD assets are generated offline by `generate_assembly_usd_assets.py`. The R1 Pro USD is generated and
post-processed by `convert_r1_pro_urdf.py`. Runtime tasks load checked-in USD assets directly.

## Scripts

The main script groups are:

- **Smoke tests:** `list_envs.py`, `zero_agent.py`, and `random_agent.py` verify registration, scene creation, and
  control.
- **Visualization:** `preview_assembly_assembled_pose.py` inspects target relations and assembled geometry.
- **Robot tools:** `run_r1_pro_diff_ik.py`, `run_r1_pro_keyboard_teleop.py`, and
  `run_r1_pro_one_leg_scripted_assembly.py` exercise reusable robot control.
- **Asset tools:** `generate_assembly_usd_assets.py` and `convert_r1_pro_urdf.py` prepare runtime USD assets.
- **Diagnostics:** joint-response and compliant-contact scripts isolate controller or physics behavior.
- **RL wrappers:** each backend directory supplies matching train/play entry points while task-side agent cfgs remain
  under the environment package.

## Tests and Documentation

The tests mirror the public contracts:

- assembly registry, parts, relations, reset lists, observation lists, and asset paths;
- R1 Pro configuration and gripper defaults;
- tactile tensor helpers, scene injection, USD materials, and pad assets.

Dependency-light tests run without Isaac Sim. Runtime-specific tests use skip markers when simulator modules are not
available. Wiki source is maintained beside the code and synchronized by `.github/workflows/sync-wiki.yml`.

## Where to Make Changes

| Goal | Primary location |
| --- | --- |
| Add or change an assembly | `assembly/<name>.py`, `assembly/registry.py`, `assets/furniture/<name>/` |
| Change observations, rewards, resets, or success | `tasks/direct/assembly_benchmark/` |
| Change R1 Pro links or actuators | `robots/r1_pro.py`, `assets/robots/r1_pro/` |
| Change action-to-joint control | `controllers/r1_pro.py` |
| Change tactile computation or configuration | `sensors/`, `assets/sensors/` |
| Add a runnable workflow | `scripts/` |
| Change an RL backend configuration | task `agents/` plus the matching `scripts/<backend>/` wrapper |
| Update user documentation | `docs/wiki/`, then `_Sidebar.md` when adding or removing a page |
