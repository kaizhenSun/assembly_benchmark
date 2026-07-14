# Tactile Sensing

The R1 Pro has one optional tactile pad on each gripper finger. Every pad samples a `12 x 32` grid at 2 mm spacing and
returns 384 taxels in the form:

```text
[x, y, z, normal_force]
```

All four pads together return `(num_envs, 1536, 4)`. Tactile sensing is disabled in the default task configuration
and is not appended to policy observations.

## Data Convention

The sensor API reports taxel positions in the world frame. Callers may provide environment origins to obtain positions
local to each replicated scene.

Normalized forces are scaled independently for each pad and environment, then clamped to `[0, 1]`. Raw sensor access
can keep the unnormalized normal force when absolute magnitudes are needed.

The tactile data path is:

```text
assembly SDF collider
  -> taxel penetration and surface gradient
  -> normal force field
  -> four pad tensors
  -> optional normalization or visualization
```

## Enable Tactile Sensing

The two environment fields cover the normal configuration cases:

| Field | Behavior |
| --- | --- |
| `enable_r1_pro_gripper_tactile` | Creates all four sensors without changing policy observations. |
| `r1_pro_gripper_tactile_contact_part_names` | Restricts contact checks to selected assembly scene keys. |

Example:

```python
env_cfg.enable_r1_pro_gripper_tactile = True
env_cfg.r1_pro_gripper_tactile_contact_part_names = ("square_table_leg4",)
```

Enabling tactile sensing does not change the fixed 28-dimensional policy observation.

By default, tactile sensors check every resettable assembly part. Explicit names must exist in `assembly_part_names`,
and every contact target must contain a PhysX SDF mesh collider.

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

Periodic output reports penetration depth, active taxels, and maximum force for each pad. Set `--print_interval 0` to
disable these diagnostics.

## Contact Diagnostic

The compliant-contact diagnostic compares rigid contact with the two compliant material authoring paths used by the
tactile assets:

```bash
python scripts/diagnostics/run_compliant_contact_diagnostic.py --headless --device cuda:0
```

The command exits nonzero when the compliant pads do not soften contact as expected or the two authoring paths diverge.

## Troubleshooting

- Enable tactile sensing before constructing the environment.
- Confirm every configured contact part has an SDF mesh collider.
- Zero penetration depth produces zero tactile force; inspect the periodic teleoperation diagnostics first.
- Use a positive `--tactile_pressure_scale` to compare raw magnitudes across frames; use zero for automatic scaling.
