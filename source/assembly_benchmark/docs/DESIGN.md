# Assembly Benchmark Design

This document describes the current architecture and extension model for Assembly Benchmark. The README is the quick
start entry point; this document is for adding assembly assets, maintaining the environment, and understanding the Isaac
Lab integration.

## Design Goals

- Provide one generic R1 Pro assembly environment: `AssemblyBenchmarkEnv`.
- Describe assembly scenes through Python `AssemblySpec` objects instead of duplicating environment code per asset.
- Follow Isaac Lab conventions for task registration, `@configclass`, `InteractiveSceneCfg`, and `DirectRLEnvCfg`.
- Let new scenes add a spec, register a factory, generate USD assets, and reuse the same environment.
- Keep `one_leg` as the default scene while also exposing explicit per-scene task ids.

## Non-Goals

- Do not introduce YAML or JSON scene parsing yet; scene definitions are Python specs.
- Do not move `assets/furniture/...` in this round, because generated USD files may contain internal references.
- Do not implement multi-relation or multi-stage success logic yet; reward and success use one primary relation.
- Do not make scripted demos, teleoperation, or pose preview scripts part of the training environment contract.

## Architecture

The main data flow is:

```text
AssemblySpec
  -> make_assembly_scene_cfg_class()
  -> make_assembly_env_cfg_class()
  -> Gym task registration
  -> AssemblyBenchmarkEnv runtime
```

`assembly_benchmark.assembly` describes and registers assembly assets.
`tasks/direct/assembly_benchmark` converts specs into Isaac Lab scene/env cfg classes and registers Gym tasks.
`AssemblyBenchmarkEnv` reads only generic `assembly_*` cfg fields and does not hard-code `one_leg` part names.

## Assembly Spec Model

`AssemblyPartSpec` describes one part:

- `scene_key`: unique part name in the Isaac scene. It must be a valid Python identifier.
- `asset_name` / `prim_name`: asset and prim names.
- `urdf_rel_path`: source URDF path relative to `asset_root`.
- `init_pos` / `init_rot`: reset pose.
- `body_type`: `visual`, `static`, or `dynamic`.
- `mass`: dynamic rigid-body mass.
- `observe`: whether the part root pose is included in policy observations. Defaults to `False`.
- `reset`: whether the runtime environment resets this part. Defaults to `True`.

`AssemblyRelationSpec` describes one assembly relation:

- `parent`: parent part scene key.
- `child`: child part scene key.
- `target_poses`: candidate child poses in the parent frame.
- `default_target_index`: target used by scripted demos by default.
- `pos_threshold` / `ori_bound`: success thresholds.

`AssemblySpec` describes a complete assembly scene and validates the contract:

- Part `scene_key` values must be unique.
- Relation `parent` and `child` values must exist in the part list.
- Each relation must define at least one target pose.
- `default_target_index` must be inside the `target_poses` range.

## Multi-Scene Registration

Assembly scenes are exposed through the registry:

```python
available_assemblies()
make_assembly(name)
register_assembly(name, factory)
```

The default registry contains `one_leg`. When the task package is imported, it iterates over `available_assemblies()` and
registers one explicit task id per scene:

```text
Assembly-Benchmark-<Name>-Direct-v0
```

The package also keeps a default alias:

```text
Assembly-Benchmark-Direct-v0
```

The default alias is controlled by `DEFAULT_ASSEMBLY_NAME`, which currently points to `one_leg`.

## Isaac Lab Cfg Generation

`AssemblyBenchmarkBaseSceneCfg` defines shared scene content for all assembly variants:

- ground
- dome light
- R1 Pro
- work camera
- LabTable

Because the shared scene includes an RGB camera, generic Isaac Lab runners must pass `--enable_cameras` when launching
assembly tasks. Dedicated tool scripts may set this flag internally when they own camera usage.

`make_assembly_scene_cfg_class(assembly, class_name)` injects the parts declared in an `AssemblySpec` into the base
scene cfg and returns an Isaac Lab `@configclass`.

`AssemblyBenchmarkEnvCfg` defines generic simulation, control, reward, reset, and IK parameters.

`make_assembly_env_cfg_class(assembly_name, class_name)` reads the spec and generates a concrete env cfg class with:

- `assembly_name`
- `assembly_part_names`
- `assembly_reset_part_names`
- `assembly_observation_part_names`
- `assembly_parent_part_name`
- `assembly_child_part_name`
- `assembled_target_positions`
- `assembled_target_quats`
- success thresholds

This keeps Isaac Lab's standard cfg entry point flow while sourcing scene-specific data from the registry.

## AssemblyBenchmarkEnv Runtime

`AssemblyBenchmarkEnv` is the only generic direct RL environment. Its runtime responsibilities are:

- Retrieve the robot and assembly parts from the scene.
- Build `assembly_parts_by_name`, `assembly_reset_parts`, and `assembly_observation_parts`.
- Initialize the R1 Pro bimanual Differential IK controller.
- Apply the 16D action format for left target pose, left gripper, right target pose, and right gripper.
- Reset the robot and resettable assembly parts.
- Build policy observations.
- Compute sparse reward, success, and done from the primary relation.

The environment does not:

- Generate USD assets.
- Decide which parts a new scene contains.
- Run a scripted assembly state machine.
- Evaluate multi-relation success conditions.

## one_leg Default Scene

`one_leg` is the default scene used to validate the multi-scene framework and R1 Pro assembly workflow.

The scene contains:

- `square_table_top`
- `square_table_leg1`
- `square_table_leg2`
- `square_table_leg3`
- `square_table_leg4`
- base tag and obstacles
- R1 Pro
- LabTable

The primary relation is:

```text
square_table_top -> square_table_leg4
```

`target_poses` define the four valid table-corner poses for the leg in the tabletop frame.
`preview_one_leg_assembled_pose.py` reads these target poses directly, converts them into world poses, and places the
parts there for visual validation.

## Asset Path Policy

The code API uses generic `assembly` naming. The on-disk asset directory still keeps the historical path:

```text
source/assembly_benchmark/assembly_benchmark/assets/furniture/
```

This is intentional. Generated USD files may contain internal references, so moving the directory should be handled as a
separate migration with dedicated validation.

Runtime environments load checked-in USD files directly. Source URDF files are used by the asset generation tool:

```bash
python scripts/tools/generate_assembly_usd_assets.py --assembly <name> --overwrite
```

## Observations, Rewards, and Success

The default policy observation contains:

- controlled robot joint positions
- controlled robot joint velocities
- left and right end-effector poses in the robot root frame
- root poses for parts with `observe=True`
- assembly target poses

Parts default to `observe=False` so new assets do not accidentally expand the observation space. Enable `observe=True`
only for parts that should be visible to the policy.

The current reward is sparse:

```text
reward = rew_scale_success * success
```

`success` is based on the primary relation:

1. Compute the current child pose in the parent frame.
2. Compare position error against all target poses.
3. Compare relative rotation error.
4. Succeed if any target satisfies the thresholds.

## Tooling Layers

Scripts are grouped by purpose:

- `list_envs.py`: check task registration.
- `zero_agent.py` / `random_agent.py`: environment smoke tests.
- `generate_assembly_usd_assets.py`: generate USD assets from assembly specs.
- `preview_one_leg_assembled_pose.py`: place `one_leg` parts directly to validate target poses.
- `run_r1_pro_one_leg_scripted_assembly.py`: run a scripted assembly demo through the control interface.
- `run_r1_pro_keyboard_teleop.py`: manually teleoperate R1 Pro.
- `scripts/rsl_rl`, `scripts/rl_games`, `scripts/sb3`, `scripts/skrl`: train and play policies.

Scripts may read environment internals for debugging, but they should not change the training interface of
`AssemblyBenchmarkEnv`.

## Adding an Assembly Scene

Minimal flow:

1. Add a spec module that returns a valid `AssemblySpec`.
2. Define one `AssemblyPartSpec` per part.
3. Define at least one `AssemblyRelationSpec`.
4. Register the scene with `register_assembly(name, factory)` or add it to the default factory table.
5. Generate USD assets.
6. Run `scripts/list_envs.py` and confirm `Assembly-Benchmark-<Name>-Direct-v0` appears.
7. Run a `zero_agent.py` smoke test.

Prefer reusing the generic environment. Add a new environment class only when control, reward, or scene composition
differs fundamentally from the generic assembly task.

## Testing Strategy

Lightweight tests should cover:

- `available_assemblies()` includes the scene.
- `make_assembly(name)` returns a valid spec.
- Duplicate `scene_key`, invalid relations, and invalid `default_target_index` raise errors.
- Part order, asset paths, reset part lists, and observation part lists match expectations.
- The task id appears in `scripts/list_envs.py`.

Isaac Lab smoke tests should cover:

```bash
python scripts/zero_agent.py \
  --task=Assembly-Benchmark-Direct-v0 \
  --num_envs 1 --device cuda:0 --headless --enable_cameras

python scripts/zero_agent.py \
  --task=Assembly-Benchmark-OneLeg-Direct-v0 \
  --num_envs 1 --device cuda:0 --headless --enable_cameras
```

New scenes should add a smoke test for their explicit task id.

## Current Limits and Future Work

Current limits:

- Success and reward use only the first primary relation.
- Observation space is maintained by the generic environment.
- Specs are Python-only and do not support external YAML/JSON.
- Asset directories have not been fully renamed to assembly terminology.

Possible future work:

- Multi-relation success conditions.
- Multi-stage assembly tasks.
- A more generic scripted demo interface.
- Scene-spec-based observation space validation.
- Full asset directory and USD reference migration.
