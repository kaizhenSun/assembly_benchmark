# Tactile Sensing

The R1 Pro has one optional tactile pad on each gripper finger. Every pad samples a `12 x 32` grid at 2 mm spacing and
returns 384 taxels in the form:

```text
[x, y, z, normal_force]
```

All four pads together return `(num_envs, 1536, 4)`. Tactile sensing and tactile policy observations are disabled in
the default task configuration.

## Enable Tactile Sensing

Enable all four gripper sensors without changing policy observations:

```python
env_cfg.enable_r1_pro_gripper_tactile = True
```

Append normalized tactile data to the policy observation:

```python
env_cfg.append_r1_pro_gripper_tactile_to_policy = True
```

This also enables the sensors and adds `4 * 12 * 32 * 4 = 6144` values to `observation_space`.

By default, tactile sensors check every resettable assembly part. Restrict contact targets with assembly scene keys:

```python
env_cfg.r1_pro_gripper_tactile_contact_part_names = ("square_table_leg4",)
```

Explicit names must exist in `assembly_part_names`. Contact targets must contain PhysX SDF mesh colliders.

## Teleoperation Pressure View

The keyboard teleoperation command in [[Running Tasks|Running-Tasks]] enables a four-panel 3D tactile pressure view by
default.

Useful options:

```text
--disable_tactile_pressure_view
--tactile_pressure_scale <force>
--tactile_pressure_update_interval <loops>
--tactile_3d_origin <x> <y> <z>
--tactile_3d_yaw <radians>
--print_interval <loops>
```

A zero pressure scale normalizes every frame. A positive value maps that raw force value to the maximum panel height
and color.

## Troubleshooting

- Enable tactile sensing before constructing the environment.
- Confirm every configured contact part has an SDF mesh collider.
- Zero penetration depth produces zero tactile force; inspect the periodic teleoperation diagnostics first.
- Keep tactile policy observations disabled when loading a policy trained for the original observation size.
