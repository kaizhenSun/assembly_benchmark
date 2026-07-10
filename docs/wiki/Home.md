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

## Documentation

- [[Installation]] — install and verify the extension.
- [[Running Tasks|Running-Tasks]] — launch tasks, tools, and RL entry points.
- [[Tactile Sensing|Tactile-Sensing]] — enable and visualize gripper tactile data.
