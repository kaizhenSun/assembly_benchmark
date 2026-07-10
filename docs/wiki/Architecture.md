# Architecture

Assembly Benchmark keeps scene-specific assembly data in Python specs and routes every registered scene through one
generic Isaac Lab direct RL environment.

## Data Flow

```text
AssemblySpec
  -> make_assembly_scene_cfg_class()
  -> make_assembly_env_cfg_class()
  -> Gym task registration
  -> optional gripper tactile scene injection
  -> AssemblyBenchmarkEnv runtime
```

`assembly_benchmark.assembly` describes and registers assembly assets. `tasks/direct/assembly_benchmark` converts specs
into Isaac Lab scene and environment cfg classes, then registers Gym tasks.

## Assembly Specs

`AssemblyPartSpec` describes one part:

- `scene_key`: unique part name in the Isaac scene.
- `asset_name` / `prim_name`: asset and prim names.
- `urdf_rel_path`: source URDF path relative to the asset root.
- `init_pos` / `init_rot`: reset pose.
- `body_type`: `visual`, `static`, or `dynamic`.
- `mass`: dynamic rigid-body mass.
- `observe`: whether the part root pose is included in policy observations.
- `reset`: whether the runtime environment resets the part.

`AssemblyRelationSpec` describes one relation:

- `parent`: parent part scene key.
- `child`: child part scene key.
- `target_poses`: candidate child poses in the parent frame.
- `default_target_index`: default target used by scripted tools.
- `pos_threshold` / `ori_bound`: success thresholds.

`AssemblySpec` validates the scene contract: unique part keys, valid relation endpoints, at least one target pose per
relation, and valid default target indices.

## Registry and Task IDs

Assembly scenes are exposed through the registry:

```python
available_assemblies()
make_assembly(name)
register_assembly(name, factory)
```

The package registers one explicit task id per assembly:

```text
Assembly-Benchmark-<Name>-Direct-v0
```

It also keeps the default alias:

```text
Assembly-Benchmark-Direct-v0
```

The default alias is controlled by `DEFAULT_ASSEMBLY_NAME`, which currently points to `one_leg`.

## Isaac Lab Cfg Generation

`AssemblyBenchmarkBaseSceneCfg` defines shared scene content:

- ground
- dome light
- R1 Pro
- work camera
- LabTable

`make_assembly_scene_cfg_class(assembly, class_name)` injects the parts from an `AssemblySpec` into the base scene cfg.

`make_assembly_env_cfg_class(assembly_name, class_name)` reads the spec and generates a concrete env cfg class with:

- assembly name and part names
- reset and observation part lists
- primary relation parent and child names
- assembled target positions and quaternions
- success thresholds

This keeps Isaac Lab's standard cfg entry point flow while avoiding one environment class per asset.

## Runtime Environment

`AssemblyBenchmarkEnv` is the only generic direct RL environment. It:

- retrieves the robot and assembly parts from the scene
- builds part lookup, reset, and observation lists
- initializes the R1 Pro bimanual Differential IK controller
- applies the 16D action format for both arms and grippers
- resets the robot and resettable parts
- builds policy observations
- optionally reads and appends normalized gripper tactile arrays
- computes sparse reward, success, and done from the primary relation

The environment does not generate USD assets, decide which parts a scene contains, run scripted assembly state machines,
or evaluate full multi-relation assembly success.

## Observations and Success

The default policy observation contains robot joint state, end-effector poses, observed part root poses, and assembly
target poses. Parts default to `observe=False` so new assets do not accidentally expand the observation space.

The current reward is sparse:

```text
reward = rew_scale_success * success
```

`success` is computed from the primary relation by comparing the current child pose in the parent frame against all
target poses.

## Tactile Sensor Integration

`assembly_benchmark.sensors` provides the VT-Refine-style sensor implementation, R1 Pro tactile-pad metadata, scene
configuration helpers, and force-grid utilities. Each of the four finger pads uses a `12 x 32` taxel grid. Every taxel
contains `[x, y, z, normal_force]`.

`AssemblyBenchmarkEnv` calls `configure_r1_pro_gripper_tactile_scene_cfg(cfg)` before the base environment constructs
the Isaac Lab scene. The following environment cfg fields control this path:

- `enable_r1_pro_gripper_tactile`: inject the four sensors without changing policy observations.
- `r1_pro_gripper_tactile_contact_part_names`: restrict SDF contact targets by assembly scene key; an empty tuple uses
  all resettable assembly parts.
- `append_r1_pro_gripper_tactile_to_policy`: inject the sensors, append normalized tactile points, and add 6144 values
  to `observation_space`.

Contact targets must contain PhysX SDF mesh colliders because the sensor evaluates penetration and gradients at each
taxel. The four tactile pad colliders and their compliant materials are part of the generated R1 Pro asset.

The table-mounted tactile pad and sensor use the same sensor class but a separate helper path.
`configure_table_tactile_scene_cfg()` must be called explicitly before scene construction; the generic environment does
not inject table tactile entities or append table tactile data to policy observations. See
[[Tactile Sensing|Tactile-Sensing]] for usage and limitations.
