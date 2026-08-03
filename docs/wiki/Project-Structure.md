# Project Structure

Assembly Benchmark separates assembly metadata, reusable runtime components, checked-in assets, and executable tools.
This page maps the functional files without expanding generated USD contents, meshes, textures, or Python caches.

## Repository Layout

```text
source/assembly_benchmark/
  assembly_benchmark/
    assembly/                         # shared specs, registry, and Isaac asset cfg conversion
      fabrica/                        # Beam specification and Fabrica specialist plans
      furniture/                      # furniture assembly specifications
    tasks/direct/assembly_benchmark/  # generated scene/env cfgs, Gym registration, runtime environment, RL cfgs
    tasks/direct/fabrica/             # Fabrica environment, tensor algorithms, and RL-Games cfg
    robots/                           # R1 Pro and Piper runtime configurations
    utils/                            # reusable conversion and source-expansion helpers
    controllers/                      # bimanual and generic single-arm Differential IK controllers
    sensors/                          # R1 Pro RGB-D/semantic camera configuration helpers
    assets/
      fabrica/                        # Fabrica assembly URDF, mesh, and generated USD assets
      furniture/                      # furniture assembly URDF, mesh, tag, and generated USD assets
      robots/r1_pro/                  # R1 Pro URDF, meshes, and generated fixed-base USD
      robots/piper/                   # Piper sources, reusable USD, and relation-paired fixed-plug USDs
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

tests/                                # assembly, robot, camera, and asset contract tests
docs/wiki/                            # canonical GitHub Wiki source
.github/workflows/sync-wiki.yml       # publishes docs/wiki to the GitHub Wiki
```

## Runtime Flow

```text
AssemblySpec in assembly/{fabrica,furniture}/<name>.py
  -> assembly/registry.py
  -> assembly/isaac.py converts part metadata to Isaac Lab asset cfgs
  -> assembly_benchmark_env_cfg.py builds scene and environment cfg classes
  -> task __init__.py registers default and explicit Gym task ids
  -> AssemblyBenchmarkEnv creates the scene and runtime state
  -> R1 Pro controller applies actions
  -> head RGB-D/semantic camera exposes scene sensor tensors
```

All registered assemblies share this path. A new scene normally adds metadata and assets rather than a new environment
class.

## Core Package

### Assembly definitions

- `assembly/specs.py` defines parts, target poses, relations, validation, and asset-generation metadata.
- `assembly/fabrica/` contains the Beam specification and Fabrica specialist relation plans.
- `assembly/furniture/` contains furniture-specific parts, initial poses, physical properties, and relations.
- `assembly/registry.py` maps stable assembly names to their spec factories.
- `assembly/isaac.py` turns a part spec into an Isaac Lab `AssetBaseCfg` or `RigidObjectCfg`.

### Task configuration and runtime

- `assembly_benchmark_env_cfg.py` defines the shared R1 Pro scene and generates one scene/env cfg class per registered
  assembly.
- The task package `__init__.py` registers the default alias and each explicit task id, together with RL backend cfgs.
- `assembly_benchmark_env.py` implements reset, observation, action, reward, and success behavior.
- `agents/` stores PPO configurations consumed by the RL wrapper scripts.
- `fabrica/` implements the original fixed-plug specialist semantics over a complete assembly graph, using paired
  heterogeneous assets, 3D residual actions, asymmetric observations, and single-arm Differential IK.
- `fabrica/fabrica_algo_utils.py` contains the task-local tensor geometry, reward, observation, termination, and
  metric helpers.

### Robot, control, and sensing

- `robots/r1_pro.py` defines the R1 Pro articulation, joint groups, end-effector links, home pose, and actuator
  settings.
- `robots/piper.py` defines the Piper articulation, six arm joints, master gripper joint, and generated-asset paths.
- `utils/piper_urdf.py` expands the packaged Piper Xacro without ROS for reusable and fixed-plug asset conversion.
- `controllers/r1_pro.py` maps the 16D bimanual action to arm, gripper, and optional torso joint targets through
  Differential IK.
- `controllers/single_arm.py` maps an absolute EE pose and one normalized gripper command through damped-least-squares
  IK, fixed EE/IK offsets, soft limits, and velocity-step limits.
- `sensors/r1_pro_camera.py` defines the R1 Pro head-camera metadata and creates its vectorized RGB-D/semantic sensor.

## Assets

- `assets/fabrica/<assembly>/` groups Fabrica source parts with their generated USDs and unique fixed-plug socket
  wrappers.
- `assets/furniture/<assembly>/` groups the remaining source URDFs, meshes and tags with generated USD folders.
- `assets/furniture/lab_table/` contains the shared work table used by the base scene.
- `assets/robots/r1_pro/` contains the robot source model and fixed-base USD loaded by the environment.
- `assets/robots/piper/` contains the MIT-licensed Piper sources, reusable fixed-base USD, and four fixed-plug Beam
  relation variants under `fixed_plug/beam/`. `NOTICE.md` records the upstream source and regeneration flow.

Assembly USD assets are generated offline by `generate_assembly_usd_assets.py`. Robot assets are generated and
post-processed by their scripts under `scripts/tools/`. Runtime tasks load checked-in USD assets directly.

## Scripts

The main script groups are:

- **Smoke tests:** `list_envs.py`, `zero_agent.py`, and `random_agent.py` verify registration, scene creation, and
  control.
- **Visualization:** `preview_assembly_assembled_pose.py` inspects target relations and assembled geometry.
- **Robot tools:** `run_r1_pro_diff_ik.py`, `run_piper_diff_ik.py`, `run_r1_pro_keyboard_teleop.py`, and
  `run_r1_pro_one_leg_scripted_assembly.py` exercise reusable robot control.
- **Asset tools:** `generate_assembly_usd_assets.py` and `convert_r1_pro_urdf.py` prepare runtime USD assets.
- **Piper/Fabrica tools:** `convert_piper_urdf.py` performs ROS-free source expansion,
  `generate_fabrica_fixedplug_assets.py` builds relation-paired robots and unique socket parts, and
  `run_fabrica_fixplug_openloop.py` validates every relation with the zero-residual policy.
- **Diagnostics:** joint-response scripts isolate controller and actuator behavior.
- **RL wrappers:** each backend directory supplies matching train/play entry points while task-side agent cfgs remain
  under the environment package.

## Tests and Documentation

The tests mirror the public contracts:

- assembly registry, parts, relations, reset lists, and asset paths;
- R1 Pro configuration, head-camera metadata, gripper defaults, and robot assets.
- Piper xacro expansion, all-relation FK and grasp coverage, generic single-arm IK, fixed-plug physics, and PPO
  actor/critic startup.

Dependency-light tests run without Isaac Sim. Runtime-specific tests use skip markers when simulator modules are not
available. Wiki source is maintained beside the code and synchronized by `.github/workflows/sync-wiki.yml`.

## Where to Make Changes

| Goal | Primary location |
| --- | --- |
| Add or change an assembly | `assembly/{furniture,fabrica}/<name>.py`, `assembly/registry.py`, `assets/{furniture,fabrica}/<name>/` |
| Change observations, rewards, resets, or success | `tasks/direct/assembly_benchmark/` |
| Change R1 Pro links or actuators | `robots/r1_pro.py`, `assets/robots/r1_pro/` |
| Change Piper links or actuators | `robots/piper.py`, `assets/robots/piper/` |
| Change action-to-joint control | `controllers/r1_pro.py`, `controllers/single_arm.py` |
| Change R1 Pro head-camera parameters | `sensors/r1_pro_camera.py` |
| Add a runnable workflow | `scripts/` |
| Change an RL backend configuration | task `agents/` plus the matching `scripts/<backend>/` wrapper |
| Update user documentation | `docs/wiki/`, then `_Sidebar.md` when adding or removing a page |
