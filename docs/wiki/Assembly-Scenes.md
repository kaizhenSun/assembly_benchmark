# Assembly Scenes

Assembly Benchmark routes every registered scene through the generic `AssemblyBenchmarkEnv`. The default task alias is:

```text
Assembly-Benchmark-Direct-v0
```

It currently points to `one_leg`. Every registered assembly also receives an explicit
`Assembly-Benchmark-<Name>-Direct-v0` task id. See [[Running Tasks|Running-Tasks]] for launch and asset-generation
commands.

## One-Leg Scene

`one_leg` is the current default assembly scene. It contains R1 Pro, LabTable, a base tag, one square table top, and
four square table legs. Obstacles are not part of the current scene spec.

The primary relation is:

```text
square_table_top -> square_table_leg4
```

The task succeeds when leg 4 matches any valid table-corner target pose relative to the tabletop.

## FurnitureBench-Derived Scenes

`chair`, `square_table`, `desk`, `round_table`, `drawer`, `lamp`, `stool`, and `cabinet` use the same spec-driven
structure. Source URDF, mesh, and tag assets live under `assets/furniture/<assembly>`. Generate or refresh their USD
assets before launching a scene when the checked-in USD is unavailable or out of date.

The chair spec, for example, keeps all five FurnitureBench assembly relations:

```text
chair_seat -> chair_leg1
chair_seat -> chair_leg2
chair_seat -> chair_back
chair_seat -> chair_nut1
chair_seat -> chair_nut2
```

## Current Evaluation Scope

The generic RL environment uses the first relation as `primary_relation` for sparse success. Multi-relation,
multi-stage full-assembly success is not modeled yet.

Assembly parts default to `observe=False`. The default policy observation contains robot joint state, end-effector
poses, and assembly target poses, but not per-part root poses. Set `observe=True` only when a part pose should be
exposed to the policy.

The Python API uses generic `assembly` naming. The on-disk `assets/furniture/...` path is retained as a historical asset
location to avoid cascading migrations of references inside generated USD files.
