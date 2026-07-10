# Assembly Benchmark

Assembly Benchmark is an Isaac Lab extension for R1 Pro assembly-task experiments. It provides one generic direct RL
environment, while individual assembly scenes are described and registered through Python `AssemblySpec` objects. The
default task uses the `one_leg` scene, and the repository also includes FurnitureBench-derived assembly assets.

The R1 Pro model includes optional VT-Refine-style tactile sensing on all four gripper fingers. Tactile data can be
visualized during keyboard teleoperation or appended to policy observations without changing the default task
configuration.

## Documentation

The canonical documentation is maintained in [`docs/wiki`](docs/wiki) and published on the
[GitHub Wiki](https://github.com/kaizhenSun/assembly_benchmark/wiki).

- [Installation](docs/wiki/Installation.md)
- [Running Tasks](docs/wiki/Running-Tasks.md)
- [Assembly Scenes](docs/wiki/Assembly-Scenes.md)
- [Tactile Sensing](docs/wiki/Tactile-Sensing.md)
- [Architecture](docs/wiki/Architecture.md)
- [Adding Assembly Scenes](docs/wiki/Adding-Assembly-Scenes.md)
- [Development](docs/wiki/Development.md)
- [FAQ](docs/wiki/FAQ.md)
