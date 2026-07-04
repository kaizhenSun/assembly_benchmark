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
