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

Fabrica source assets live under `assets/fabrica/<assembly>`; other assembly assets remain under
`assets/furniture/<assembly>`. Runtime tasks load the generated USD files directly and do not invoke the URDF importer.

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

Run the isolated Piper + Pika2 Differential IK diagnostic interactively:

```bash
python scripts/tools/run_piper_diff_ik.py --num_envs 1 --device cuda:0
```

Enable keyboard teleoperation in the same isolated scene:

```bash
python scripts/tools/run_piper_diff_ik.py --teleop --num_envs 1 --device cuda:0
```

Use `W/S`, `A/D`, and `Q/E` for root-frame translation; `Z/X`, `T/G`, and `C/V` for rotation; `K` to toggle
Pika2; `R` to reset; and `Esc` to quit. The periodic `[TELEOP]` line reports target/current XYZ, signed XYZ position
errors and total position error in millimetres, orientation error, joint tracking error, and torque metrics. Adjust the
motion increments with `--pos_step` and `--rot_step`, and the reporting rate with `--print_interval`.

For a complete headless 6D accuracy cycle with PASS/FAIL thresholds:

```bash
python scripts/tools/run_piper_diff_ik.py \
  --num_envs 1 --device cuda:0 --headless \
  --max_steps 1920 --fast_exit
```

The diagnostic reports per-target position, orientation, arm-joint tracking, and torque metrics. Its default limits
are 5 mm position error, 3 degrees orientation error, and 0.05 rad maximum joint tracking error; the corresponding
CLI tolerance options can override them.

## Piper Fabrica Fixed-Plug Specialist

`Assembly-Benchmark-FabricaFixPlugTaskAssemble-Direct-v0` migrates Fabrica's specialist contract to Piper. The
complete Beam graph contains parts `0, 1, 2, 3, 6` and relations `0→2`, `1→3`, `2→6`, and `3→6`. One policy covers
all four relations. Environments use `env_id % 4` deterministic assignment, while each episode performs only its
assigned insertion rather than assembling all four relations sequentially. Each plug is fixed to `gripper_base`, so
grasp acquisition and dropping are outside this task.

The 3D path-frame policy residual is added to the unit direction toward the nominal target and multiplied by 5 mm.
The actor observes the nominal 3D error; the asymmetric critic additionally receives the true error under uniform
±3 mm socket-position noise. Robot and socket asset lists are paired deterministically, and the heterogeneous scene
uses `replicate_physics=False`.

Replay the noise-free baseline across every relation:

```bash
python scripts/tools/run_fabrica_fixplug_openloop.py \
  --num_envs 4 --device cuda:0 --headless
```

Add `--show_path` in GUI mode to render each checked-in ten-point path. The command exits nonzero unless every relation
finishes below 5 mm error with success rate 1.0, no invalid state, and no path deviation beyond 8 mm. It also reports
the largest progress regression as a controller-transient diagnostic. Use `--socket_noise 0.003` for a noisy
asset-pairing and finite-metrics check.

The checked-in Piper assets are self-contained and need neither ROS nor xacro at runtime. Fixed-plug robot variants
live under `assets/robots/piper/fixed_plug/`, while the deduplicated socket wrappers live with the Beam parts under
`assets/fabrica/beam/usd/fixed_plug_socket/`. Regenerate them with:

```bash
python scripts/tools/convert_piper_urdf.py --force --headless
python scripts/tools/generate_assembly_usd_assets.py --assembly beam --overwrite --headless
python scripts/tools/generate_fabrica_fixedplug_assets.py --assembly beam --overwrite --headless
```

The reusable `piper_fixed.usd` keeps the upstream gripper direction. The 90-degree rotation, 30 mm lift, relation
transforms, openings, and common `plug` body exist only in the fixed-plug variants.

Train the asymmetric specialist PPO policy with RL-Games:

```bash
python scripts/rl_games/train.py \
  --task=Assembly-Benchmark-FabricaFixPlugTaskAssemble-Direct-v0 \
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
