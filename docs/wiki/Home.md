# Assembly Benchmark

Assembly Benchmark is an Isaac Lab extension for assembly-task experiments with R1 Pro. It provides one generic direct
RL environment while individual assembly scenes are described by Python `AssemblySpec` objects.

The default task points to the `one_leg` assembly. Other ported FurnitureBench-style scenes are registered as explicit
task ids and reuse the same environment implementation.

Four optional VT-Refine-style tactile pads cover the R1 Pro gripper fingers. They support policy observations,
teleoperation pressure visualization, and compliant-contact diagnostics without changing the default task observation
space.

## Quick Start

Install the extension in editable mode:

```bash
python -m pip install -e source/assembly_benchmark
```

List registered tasks:

```bash
python scripts/list_envs.py
```

Run the default zero-action smoke test:

```bash
python scripts/zero_agent.py \
  --task=Assembly-Benchmark-Direct-v0 \
  --num_envs 1 --device cuda:0 --headless --enable_cameras
```

## Registered Tasks

```text
Assembly-Benchmark-Direct-v0
Assembly-Benchmark-Cabinet-Direct-v0
Assembly-Benchmark-Chair-Direct-v0
Assembly-Benchmark-Desk-Direct-v0
Assembly-Benchmark-Drawer-Direct-v0
Assembly-Benchmark-Lamp-Direct-v0
Assembly-Benchmark-OneLeg-Direct-v0
Assembly-Benchmark-RoundTable-Direct-v0
Assembly-Benchmark-SquareTable-Direct-v0
Assembly-Benchmark-Stool-Direct-v0
```

`Assembly-Benchmark-Direct-v0` is the generic alias and currently resolves to `one_leg`. Every registered assembly also
gets an explicit `Assembly-Benchmark-<Name>-Direct-v0` task id.

## Documentation Map

- [[Installation]] covers requirements, editable install, and verification.
- [[Running Tasks|Running-Tasks]] collects smoke tests, preview tools, asset generation, and RL commands.
- [[Tactile Sensing|Tactile-Sensing]] covers gripper taxels, environment configuration, visualization, and diagnostics.
- [[Architecture]] explains the spec registry, generated Isaac Lab cfg classes, and runtime environment.
- [[Adding Assembly Scenes|Adding-Assembly-Scenes]] describes how to add a new assembly spec and validate it.
- [[FAQ]] covers cameras, default scene behavior, generated USD assets, and current limits.

## Source Documentation

The quick-start README is maintained in the main repository:

https://github.com/kaizhenSun/assembly_benchmark/blob/main/README.md

Architecture and design notes are maintained in [[Architecture]].

This wiki is generated from `docs/wiki/` in the main repository and is synchronized by GitHub Actions.
