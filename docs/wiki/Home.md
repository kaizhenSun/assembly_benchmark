# Assembly Benchmark

Assembly Benchmark is an Isaac Lab extension for R1 Pro and Piper assembly-task experiments. Assembly scenes are
described by Python `AssemblySpec` objects and run through direct RL environments.

The default task uses the `one_leg` assembly.

## Features

- Explicit Gym task ids for every registered assembly.
- Assembly target-pose preview, scripted assembly, and keyboard teleoperation tools.
- A self-contained Piper Fabrica specialist task covering all four Beam relations with fixed grasps, residual IK
  control, and asymmetric RL-Games PPO observations.
- Training and playback entry points for common RL backends.

## Assemblies

The registered assemblies are:

```text
beam
cabinet
chair
desk
drawer
lamp
one_leg
round_table
square_table
stool
```

`Assembly-Benchmark-Direct-v0` is the default alias for `one_leg`. Explicit tasks follow the pattern
`Assembly-Benchmark-<Name>-Direct-v0`.

The Piper fixed-plug training task uses `Assembly-Benchmark-FabricaFixPlugTaskAssemble-Direct-v0`; see
[[Running Tasks|Running-Tasks]] for open-loop replay, asset regeneration, and RL-Games commands.

## How It Works

Each assembly module defines parts and target relations in an `AssemblySpec`. The registry exposes those specs to the
task configuration layer, which builds an Isaac Lab scene and environment configuration for every assembly. All task
ids then run through the same `AssemblyBenchmarkEnv` and R1 Pro controller.

This keeps assembly geometry and target data separate from reusable control, observation, reward, and reset behavior.
See [[Project Structure|Project-Structure]] for the complete file map and runtime flow.

## Documentation Structure

The Wiki is organized as a short user guide:

- [[Installation]] — install and verify the extension.
- [[Running Tasks|Running-Tasks]] — launch tasks, tools, and RL entry points.
- [[Project Structure|Project-Structure]] — understand source modules, assets, scripts, and tests.

The source pages live in `docs/wiki/`, `_Sidebar.md` defines their navigation order, and changes are published to the
GitHub Wiki by the repository's sync workflow.
