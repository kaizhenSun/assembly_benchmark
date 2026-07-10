# Assembly Benchmark

Assembly Benchmark is an Isaac Lab extension for R1 Pro assembly-task experiments. Assembly scenes are described by
Python `AssemblySpec` objects and run through one generic direct RL environment.

The default task uses the `one_leg` assembly. Optional VT-Refine-style tactile sensing is available on all four R1 Pro
gripper fingers.

## Features

- Explicit Gym task ids for every registered assembly.
- Assembly target-pose preview, scripted assembly, and keyboard teleoperation tools.
- Training and playback entry points for common RL backends.
- Optional gripper tactile observations and pressure visualization.

## Assemblies

The registered assemblies are:

```text
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

## How It Works

Each assembly module defines parts and target relations in an `AssemblySpec`. The registry exposes those specs to the
task configuration layer, which builds an Isaac Lab scene and environment configuration for every assembly. All task
ids then run through the same `AssemblyBenchmarkEnv`, R1 Pro controller, and optional tactile sensors.

This keeps assembly geometry and target data separate from reusable control, observation, reward, and reset behavior.
See [[Project Structure|Project-Structure]] for the complete file map and runtime flow.

## Documentation Structure

The Wiki is organized as a short user guide:

- [[Installation]] — install and verify the extension.
- [[Running Tasks|Running-Tasks]] — launch tasks, tools, and RL entry points.
- [[Tactile Sensing|Tactile-Sensing]] — enable and visualize gripper tactile data.
- [[Project Structure|Project-Structure]] — understand source modules, assets, scripts, and tests.

The source pages live in `docs/wiki/`, `_Sidebar.md` defines their navigation order, and changes are published to the
GitHub Wiki by the repository's sync workflow.
