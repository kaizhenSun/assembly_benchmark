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
