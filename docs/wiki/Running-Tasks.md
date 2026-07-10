# Running Tasks

## List Registered Environments

```bash
python scripts/list_envs.py
```

## Smoke Tests

Run the default assembly task:

```bash
python scripts/zero_agent.py \
  --task=Assembly-Benchmark-Direct-v0 \
  --num_envs 1 --device cuda:0 --headless --enable_cameras
```

Run a random-action smoke test:

```bash
python scripts/random_agent.py \
  --task=Assembly-Benchmark-Direct-v0 \
  --num_envs 1 --device cuda:0 --headless --enable_cameras
```

Run the explicit `one_leg` scene:

```bash
python scripts/zero_agent.py \
  --task=Assembly-Benchmark-OneLeg-Direct-v0 \
  --num_envs 1 --device cuda:0 --headless --enable_cameras
```

Run another explicit scene after generating or checking its USD assets:

```bash
python scripts/zero_agent.py \
  --task=Assembly-Benchmark-Chair-Direct-v0 \
  --num_envs 1 --device cuda:0 --headless --enable_cameras
```

## Preview Assembled Poses

Preview assembled target poses:

```bash
python scripts/tools/preview_assembly_assembled_pose.py --assembly one_leg --num_envs 1 --device cuda:0
python scripts/tools/preview_assembly_assembled_pose.py --assembly chair --num_envs 1 --device cuda:0
python scripts/tools/preview_assembly_assembled_pose.py --assembly desk --num_envs 1 --device cuda:0 --disable_markers
```

The preview tool defaults to visual-only `ghost` physics so target poses are not pushed apart by collision
depenetration. To inspect dynamic collision behavior:

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

## Generate Assembly USD Assets

Generate one assembly:

```bash
python scripts/tools/generate_assembly_usd_assets.py --assembly <name> --overwrite
```

Regenerate the current assembly set:

```bash
for assembly in cabinet chair desk drawer lamp one_leg round_table square_table stool; do
  PYTHONPATH=source/assembly_benchmark \
    python scripts/tools/generate_assembly_usd_assets.py --assembly "$assembly" --overwrite
done
```

## Teleoperation and Scripted Demo

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

Keyboard teleoperation requires a GUI and supports only one environment. The four-panel R1 Pro gripper tactile pressure
view is enabled by default. Disable it when tactile visualization is not needed:

```bash
python scripts/tools/run_r1_pro_keyboard_teleop.py \
  --num_envs 1 --device cuda:0 --enable_cameras \
  --disable_tactile_pressure_view
```

By default, the pressure panel normalizes every frame. Use a positive raw-force scale, change the refresh interval, or
move and rotate the panel with:

```bash
python scripts/tools/run_r1_pro_keyboard_teleop.py \
  --num_envs 1 --device cuda:0 --enable_cameras \
  --tactile_pressure_scale 1.0 \
  --tactile_pressure_update_interval 2 \
  --tactile_3d_origin 0.90 -0.45 0.775 \
  --tactile_3d_yaw 0.0
```

See [[Tactile Sensing|Tactile-Sensing]] for all visualization options and tactile data conventions.

## Tactile Contact Diagnostics

Compare rigid contact with both supported compliant-material authoring paths:

```bash
python scripts/diagnostics/run_compliant_contact_diagnostic.py --headless --device cuda:0
```

Also run the comparison with an assembly-part SDF collider:

```bash
python scripts/diagnostics/run_compliant_contact_diagnostic.py \
  --headless --device cuda:0 --include_sdf --sdf_asset path/to/contact_part.usd
```

The SDF mode requires an existing USD asset. The diagnostic exits nonzero when its compliant-contact checks fail.

## RL Backends

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

## CPU and USD Compatibility Debugging

Disable Fabric when debugging CPU execution or USD synchronization:

```bash
python scripts/zero_agent.py \
  --task=Assembly-Benchmark-Direct-v0 \
  --num_envs 1 --device cpu --disable_fabric --enable_cameras
```

Disabling Fabric is a compatibility and debugging option; it can make GUI mesh updates diverge from the normal Fabric
runtime path.
