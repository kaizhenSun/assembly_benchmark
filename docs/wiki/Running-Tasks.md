# Running Tasks

`Assembly-Benchmark-Direct-v0` runs the default `one_leg` scene. Use
`Assembly-Benchmark-<Name>-Direct-v0` for a specific registered assembly. Generic runners require
`--enable_cameras` because the shared scene includes an RGB work camera.

## Smoke Test

```bash
python scripts/zero_agent.py \
  --task=Assembly-Benchmark-Direct-v0 \
  --num_envs 1 --device cuda:0 --headless --enable_cameras
```

Replace the task id with an explicit assembly task when needed.

## Preview an Assembly

```bash
python scripts/tools/preview_assembly_assembled_pose.py \
  --assembly one_leg --num_envs 1 --device cuda:0
```

The preview uses visual-only `ghost` physics by default so collision depenetration does not move assembled target poses.

## Generate Assembly Assets

Regenerate USD assets after changing an assembly URDF, mesh, tag texture, or asset layout:

```bash
python scripts/tools/generate_assembly_usd_assets.py --assembly <name> --overwrite
```

Source assets remain under `assets/furniture/<assembly>` for compatibility with references inside generated USD files.

## Scripted Assembly and Teleoperation

Run the `one_leg` scripted assembly demo:

```bash
python scripts/tools/run_r1_pro_one_leg_scripted_assembly.py \
  --num_envs 1 --device cuda:0 --enable_cameras
```

Run keyboard teleoperation:

```bash
python scripts/tools/run_r1_pro_keyboard_teleop.py \
  --num_envs 1 --device cuda:0 --enable_cameras
```

Keyboard teleoperation requires a GUI and supports one environment. Its tactile pressure view is enabled by default;
see [[Tactile Sensing|Tactile-Sensing]] for controls.

## Reinforcement Learning

Train with RSL-RL:

```bash
python scripts/rsl_rl/train.py \
  --task=Assembly-Benchmark-Direct-v0 \
  --num_envs 1 --device cuda:0 --headless --enable_cameras
```

Playback is available through `scripts/rsl_rl/play.py`. Other backends are under `scripts/rl_games`, `scripts/sb3`, and
`scripts/skrl`.

## Current Task Behavior

- Sparse success evaluates only the assembly's primary relation; full multi-relation completion is not modeled yet.
- Assembly part poses enter policy observations only when their specs set `observe=True`.
- Gripper tactile observations are disabled by default.

## CPU and USD Debugging

Disable Fabric when debugging CPU execution or USD synchronization:

```bash
python scripts/zero_agent.py \
  --task=Assembly-Benchmark-Direct-v0 \
  --num_envs 1 --device cpu --disable_fabric --enable_cameras
```
