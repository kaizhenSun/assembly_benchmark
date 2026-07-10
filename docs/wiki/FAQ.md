# FAQ

## Why do generic runners need `--enable_cameras`?

The shared assembly scene includes an RGB work camera. Generic Isaac Lab runners must pass `--enable_cameras` so the
camera sensor is enabled correctly. Dedicated tool scripts may set this flag internally when they own the camera setup.

## What does `Assembly-Benchmark-Direct-v0` run?

`Assembly-Benchmark-Direct-v0` is the default generic task alias. It currently points to `one_leg`.

Use an explicit task id when you want a specific scene:

```text
Assembly-Benchmark-Chair-Direct-v0
Assembly-Benchmark-Desk-Direct-v0
Assembly-Benchmark-OneLeg-Direct-v0
```

## Why are source assets under `assets/furniture`?

The Python API now uses generic `assembly` naming, but the on-disk asset directory keeps the historical
`assets/furniture/...` path. Generated USD files may contain internal references, so renaming asset directories should be
handled as a separate migration with dedicated validation.

## When should I regenerate USD assets?

Regenerate USD assets after changing a source URDF, mesh, tag texture, or assembly asset layout:

```bash
python scripts/tools/generate_assembly_usd_assets.py --assembly <name> --overwrite
```

Regenerate all current assemblies:

```bash
for assembly in cabinet chair desk drawer lamp one_leg round_table square_table stool; do
  PYTHONPATH=source/assembly_benchmark \
    python scripts/tools/generate_assembly_usd_assets.py --assembly "$assembly" --overwrite
done
```

## Does the environment evaluate full assembly success?

Not yet. The generic RL environment currently uses the primary relation for sparse success. Multi-relation and
multi-stage full-assembly success are future work.

## Are per-part poses included in observations?

Only for parts with `observe=True`. Assembly parts default to `observe=False` so new scenes do not accidentally change
the policy observation space.

## Is gripper tactile sensing enabled by default?

No. Gym task configurations default to `enable_r1_pro_gripper_tactile=False` and
`append_r1_pro_gripper_tactile_to_policy=False`. The keyboard teleoperation tool enables the four gripper sensors for
its pressure panel unless `--disable_tactile_pressure_view` is passed.

## How does tactile sensing change the policy observation?

Setting `append_r1_pro_gripper_tactile_to_policy=True` appends four `12 x 32` arrays. Each taxel contains
`[x, y, z, normal_force]`, so the policy receives 6144 additional values and the configured `observation_space` is
updated automatically. Enabling sensors with only `enable_r1_pro_gripper_tactile=True` does not change observations.

Existing policies trained for the original observation dimension should keep tactile policy observations disabled.

## Why are all tactile forces zero?

Check that the sensors were enabled before environment construction and that the configured contact parts contain
PhysX SDF mesh colliders. An empty `r1_pro_gripper_tactile_contact_part_names` tuple targets all resettable assembly
parts; explicit names must be valid assembly scene keys. The teleoperation diagnostics report penetration depth and
active taxels, which can distinguish missing contact from a visualization scaling issue.

## Is the table-mounted tactile sensor part of every task?

No. The table pad and sensor are lower-level helpers. Call `configure_table_tactile_scene_cfg(env_cfg)` explicitly
before scene construction after setting `enable_table_tactile=True`. `AssemblyBenchmarkEnv` does not call this helper
and does not append table tactile data to policy observations.

## How is this wiki published?

The canonical source lives in `docs/wiki/` in the main repository. The `sync-wiki.yml` GitHub Actions workflow clones
the GitHub Wiki repository, replaces its contents with `docs/wiki/`, and pushes the result.

Maintainers need to enable the repository Wiki and add a repository secret named `WIKI_PUSH_TOKEN` with permission to
push to:

```text
https://github.com/kaizhenSun/assembly_benchmark.wiki.git
```

The published wiki is:

https://github.com/kaizhenSun/assembly_benchmark/wiki
