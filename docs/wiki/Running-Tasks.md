# Running Tasks

`Assembly-Benchmark-Direct-v0` runs the default `one_leg` scene. Use
`Assembly-Benchmark-<Name>-Direct-v0` for a specific registered assembly. Generic runners require
`--enable_cameras` because the shared scene includes an RGB work camera and the R1 Pro head RGB-D/semantic camera.

For example, `one_leg` is exposed as `Assembly-Benchmark-OneLeg-Direct-v0`. Use `scripts/list_envs.py` after
installation to see the exact ids currently registered by the package.

## Smoke Test

```bash
python scripts/zero_agent.py \
  --task=Assembly-Benchmark-Direct-v0 \
  --num_envs 1 --device cuda:0 --headless --enable_cameras
```

Replace the task id with an explicit assembly task when needed.

Use the random-action runner to exercise the action and control pipeline:

```bash
python scripts/random_agent.py \
  --task=Assembly-Benchmark-Direct-v0 \
  --num_envs 1 --device cuda:0 --headless --enable_cameras
```

## Preview an Assembly

```bash
python scripts/tools/preview_assembly_assembled_pose.py \
  --assembly one_leg --num_envs 1 --device cuda:0
```

The preview uses visual-only `ghost` physics by default so collision depenetration does not move assembled target poses.

Useful preview options:

- `--mode all_relations`: show every registered assembly relation.
- `--mode relation_child --relation_index N --target_index N`: inspect one child and target pose.
- `--physics_mode dynamic`: keep dynamic collision behavior instead of ghost parts.
- `--print_poses`: print world and measured relative poses.
- `--disable_markers`: show assembled geometry without frame markers.

## Generate Assembly Assets

Regenerate USD assets after changing an assembly URDF, mesh, tag texture, or asset layout:

```bash
python scripts/tools/generate_assembly_usd_assets.py --assembly <name> --overwrite
```

Source assets remain under `assets/furniture/<assembly>` for compatibility with references inside generated USD files.
Runtime tasks load the generated USD files directly; they do not invoke the URDF importer.

## Scripted Assembly and Teleoperation

Run the `one_leg` scripted assembly demo:

```bash
python scripts/tools/run_r1_pro_one_leg_scripted_assembly.py \
  --num_envs 1 --device cuda:0 --enable_cameras
```

Add `--record_camera --camera_name head_camera` to record the R1 Pro head RGB view. See
[[Camera Sensing|Camera-Sensing]] for the RGB-D tensor interface and native-resolution performance considerations.

Run keyboard teleoperation:

```bash
python scripts/tools/run_r1_pro_keyboard_teleop.py \
  --num_envs 1 --device cuda:0 --enable_cameras
```

Keyboard teleoperation requires a GUI and supports one environment.

The separate `run_r1_pro_diff_ik.py` and `run_r1_pro_joint_response_diagnostic.py` tools help inspect controller targets
and low-level actuator response without running an RL policy.

## R1 Pro Beam 0-to-2 Insertion

`Assembly-Benchmark-Beam02-LeftInsert-Direct-v0` is a dedicated, camera-free training task. Beam plug 0 is fixed to
the R1 Pro left gripper. The bimanual whole-body Differential IK controller commands the torso and both arms: the
left hand follows the insertion path, while the right arm compensates torso motion to hold its reset end-effector pose.
The 3D action is a residual translation in the socket-aligned path frame. Actor observations contain the nominal
3D socket-to-plug displacement; the asymmetric critic also receives the actual displacement after the socket's
uniform ±3 mm reset noise.

Replay the analytic 20.2 mm straight insertion path with zero policy residuals:

```bash
python scripts/tools/run_r1_pro_beam02_geometric_insertion.py \
  --num_envs 1 --device cuda:0 --headless
```

Add `--show_path` in GUI mode to render eleven 1 mm markers along each nominal path. The baseline advances to the
next finite-segment waypoint and limits the combined baseline-plus-residual translation to 2 mm per policy step.

The task runs physics at 120 Hz and policy control at 30 Hz for 128-step episodes. It does not require
`--enable_cameras`. To regenerate the task-specific robot asset after changing the R1 Pro or Beam plug geometry, run:

```bash
python scripts/tools/generate_r1_pro_beam02_asset.py --overwrite
```

Train the migrated Fabrica-style asymmetric PPO policy with RL-Games:

```bash
python scripts/rl_games/train.py \
  --task=Assembly-Benchmark-Beam02-LeftInsert-Direct-v0 \
  --num_envs 1024 --device cuda:0 --headless
```

## Reinforcement Learning

Train with RSL-RL:

```bash
python scripts/rsl_rl/train.py \
  --task=Assembly-Benchmark-Direct-v0 \
  --num_envs 1 --device cuda:0 --headless --enable_cameras
```

Playback is available through `scripts/rsl_rl/play.py`. Other backends are under `scripts/rl_games`, `scripts/sb3`, and
`scripts/skrl`.

Play an RSL-RL checkpoint:

```bash
python scripts/rsl_rl/play.py \
  --task=Assembly-Benchmark-Direct-v0 \
  --num_envs 1 --device cuda:0 --headless --enable_cameras
```

## Current Task Behavior

- Sparse success evaluates only the assembly's primary relation; full multi-relation completion is not modeled yet.
- Policy observations contain only the positions and velocities of the 14 arm joints, for a fixed size of 28.

## CPU and USD Debugging

Disable Fabric when debugging CPU execution or USD synchronization:

```bash
python scripts/zero_agent.py \
  --task=Assembly-Benchmark-Direct-v0 \
  --num_envs 1 --device cpu --disable_fabric --enable_cameras
```
