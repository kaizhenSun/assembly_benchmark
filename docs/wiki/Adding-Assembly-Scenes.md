# Adding Assembly Scenes

New scenes should follow the existing spec-driven pattern used by `one_leg`.

## Minimal Flow

1. Add one spec module, for example `assembly/<name>.py`, that returns an `AssemblySpec`.
2. Define one `AssemblyPartSpec` per part.
3. Give each part a unique `scene_key`, asset path, initial pose, body type, mass or density, and reset/observe policy.
4. Define at least one `AssemblyRelationSpec` with a parent, child, and child target poses in the parent frame.
5. Register the scene in `assembly/registry.py`.
6. Generate USD assets.
7. Confirm the task id appears in `scripts/list_envs.py`.
8. Run a zero-action smoke test for the explicit task id.

## Generate Assets

```bash
python scripts/tools/generate_assembly_usd_assets.py --assembly <name> --overwrite
```

The generated task id should use this pattern:

```text
Assembly-Benchmark-<Name>-Direct-v0
```

For example, an assembly named `chair` is exposed as:

```text
Assembly-Benchmark-Chair-Direct-v0
```

## Validation Commands

List task ids:

```bash
python scripts/list_envs.py
```

Run the scene:

```bash
python scripts/zero_agent.py \
  --task=Assembly-Benchmark-<Name>-Direct-v0 \
  --num_envs 1 --device cuda:0 --headless --enable_cameras
```

Preview assembled target poses:

```bash
python scripts/tools/preview_assembly_assembled_pose.py \
  --assembly <name> --num_envs 1 --device cuda:0
```

## Spec Guidelines

- Prefer reusing the generic `AssemblyBenchmarkEnv`.
- Add a new environment class only when control, reward, or scene composition differs fundamentally.
- Keep `scene_key` values stable once generated USD assets reference them.
- Keep source assets under the current `assets/furniture/<assembly>` layout unless doing a dedicated asset migration.
- Set `observe=True` only for parts whose root poses should be exposed to the policy.
- Remember that current sparse success uses the primary relation, not full multi-relation completion.

## Lightweight Tests

Good tests for a new scene include:

- `available_assemblies()` includes the scene.
- `make_assembly(name)` returns a valid spec.
- invalid duplicate `scene_key` values raise errors.
- invalid relation endpoints raise errors.
- invalid `default_target_index` values raise errors.
- dynamic parts declare the expected mass or density.
- reset and observation part lists match expectations.
- the explicit task id appears in `scripts/list_envs.py`.
