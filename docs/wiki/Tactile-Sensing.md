# Tactile Sensing

Assembly Benchmark provides VT-Refine-style force-field tactile sensing for the four R1 Pro gripper fingers. Tactile
sensing is opt-in for Gym tasks, while the keyboard teleoperation tool enables the gripper sensors for its pressure
visualization by default.

## Sensor Layout and Data

The robot has two tactile pads per hand: `right_pad1`, `right_pad2`, `left_pad1`, and `left_pad2`. Each pad samples a
`12 x 32` grid with 2 mm taxel spacing. A taxel is returned as:

```text
[x, y, z, normal_force]
```

Positions are in the world frame by default. Pass `env_origins` to `get_tactile_points()` or
`get_r1_pro_gripper_tactile_points()` to express them relative to each environment origin. The combined helper returns
all four pads with shape `(num_envs, 1536, 4)`.

The sensor computes contact from penetration into SDF mesh colliders. Each configured contact target must therefore
contain a mesh with PhysX SDF collision approximation. Non-finite and negative force values are converted to zero.

## Robot Asset Integration

The R1 Pro URDF mounts one flat tactile-pad link on each gripper finger. Every pad has dedicated visual and collision
geometry, and the generated USD binds an independent compliant physics material to each tactile collider. The robot
configuration starts all gripper finger joints at `0.05`, the fully open position used by the environment, and uses the
tuned gripper actuator limits shipped with the tactile-enabled asset.

Source pad meshes live under `assets/sensors/vt_refine_tactile`. Regenerate the R1 Pro USD after changing its URDF,
tactile meshes, material authoring, or finger collision geometry.

```bash
python scripts/tools/convert_r1_pro_urdf.py --force --headless
```

## Environment Configuration

Tactile sensors and tactile policy observations are disabled in the default environment configuration. Enable the four
gripper sensors without changing the policy observation space:

```python
env_cfg.enable_r1_pro_gripper_tactile = True
```

Append the four tactile arrays to the policy observation with:

```python
env_cfg.append_r1_pro_gripper_tactile_to_policy = True
```

Setting `append_r1_pro_gripper_tactile_to_policy` also injects the sensors even when
`enable_r1_pro_gripper_tactile` is false. It adds `4 * 12 * 32 * 4 = 6144` values and updates the configured Gym
`observation_space` once.

By default, the sensors test contact against every part listed in `assembly_reset_part_names`. To restrict contact
targets, provide assembly scene keys:

```python
env_cfg.r1_pro_gripper_tactile_contact_part_names = ("square_table_top", "square_table_leg4")
```

Explicit names are validated against `assembly_part_names`; an unknown name raises `ValueError` during scene
configuration.

Policy observations use normalized force values. Each gripper pad is normalized independently by its current maximum
force and clamped to `[0, 1]`; zero-force pads remain zero. Use the sensor API with `normalize=False` when raw force
values are required:

```python
from assembly_benchmark.sensors import get_r1_pro_gripper_tactile_points

tactile_points = get_r1_pro_gripper_tactile_points(
    env.scene,
    normalize=False,
    env_origins=env.scene.env_origins,
)
```

`VtRefineTactileSensorCfg` exposes the taxel layout, pad bounds, contact targets, force-field stiffness, friction,
normal axis, and normalization parameters for custom sensor configurations.

## Keyboard Teleoperation Visualization

Keyboard teleoperation requires a GUI and currently supports only `--num_envs 1`. Its 3D tabletop pressure panel is
enabled by default:

```bash
python scripts/tools/run_r1_pro_keyboard_teleop.py \
  --num_envs 1 --device cuda:0 --enable_cameras
```

The panel displays one colored taxel-pillar grid per gripper pad. A zero `--tactile_pressure_scale` uses per-frame
normalized forces. A positive value maps that raw force value to the maximum pillar height and color.

Useful options include:

```text
--disable_tactile_pressure_view
--tactile_pressure_scale <force>
--tactile_pressure_update_interval <loops>
--tactile_3d_max_height <meters>
--tactile_3d_taxel_size <meters>
--tactile_3d_origin <x> <y> <z>
--tactile_3d_yaw <radians>
```

Periodic teleoperation diagnostics report maximum and mean penetration depth, active taxels, and maximum force for each
pad. Use `--print_interval 0` to disable periodic output.

## Compliant-Contact Diagnostic

The tactile pad colliders use independent compliant physics materials. Compare the shared low-level material authoring
helper with Isaac Lab's official `RigidBodyMaterialCfg` path:

```bash
python scripts/diagnostics/run_compliant_contact_diagnostic.py --headless --device cuda:0
```

The command drops identical spheres onto rigid and compliant pads and exits nonzero if the compliant cases do not soften
contact or if the two compliant authoring paths diverge. An optional SDF asset comparison is available:

```bash
python scripts/diagnostics/run_compliant_contact_diagnostic.py \
  --headless --device cuda:0 --include_sdf --sdf_asset path/to/contact_part.usd
```

## Table-Mounted Tactile Helpers

`make_table_tactile_pad_cfg()`, `make_table_tactile_sensor_cfg()`, and `configure_table_tactile_scene_cfg()` provide a
lower-level table-mounted sensor path. Set `enable_table_tactile` and optionally
`table_tactile_contact_part_names` before calling `configure_table_tactile_scene_cfg(env_cfg)`.

The generic `AssemblyBenchmarkEnv` does not call this table injection helper. Applications that use the table sensor
must invoke it before the Isaac Lab scene is constructed. Table tactile data is not appended to the generic policy
observation automatically.

## Troubleshooting

- Confirm `enable_r1_pro_gripper_tactile` or `append_r1_pro_gripper_tactile_to_policy` is true before constructing the
  environment.
- Confirm every contact target has an SDF mesh collider and is named in `assembly_part_names`.
- Check penetration-depth diagnostics first: zero depth produces zero tactile force.
- Use a positive `--tactile_pressure_scale` when comparing raw force magnitudes across frames; use zero for an
  automatically normalized visualization.
- Keep tactile observations disabled when an existing policy expects the original observation dimension.
