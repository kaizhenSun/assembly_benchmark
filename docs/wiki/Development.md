# Development

## Project Layout

```text
source/assembly_benchmark/
  assembly_benchmark/
    assembly/                          # AssemblySpec, part specs, registry, and Isaac cfg helpers
    assets/robots/r1_pro/              # R1 Pro URDF, meshes, configuration, and generated USD
    assets/sensors/vt_refine_tactile/  # tactile-pad collision meshes
    controllers/                       # R1 Pro joint and Differential IK controllers
    robots/                            # Isaac Lab ArticulationCfg definitions
    sensors/                           # VT-Refine-style tactile sensor and scene cfg helpers
    tasks/direct/assembly_benchmark/   # environment, cfg generation, and task registration
  config/extension.toml                # Isaac Lab extension metadata

scripts/
  diagnostics/run_compliant_contact_diagnostic.py # tactile compliant-contact validation
  list_envs.py                         # list registered tasks
  zero_agent.py                        # zero-action smoke test
  random_agent.py                      # random-action smoke test
  rsl_rl/                              # RSL-RL train/play scripts
  tools/convert_r1_pro_urdf.py         # R1 Pro conversion and tactile material authoring
  tools/generate_assembly_usd_assets.py # assembly URDF-to-USD generation
  tools/preview_assembly_assembled_pose.py # assembled target-pose preview
  tools/run_r1_pro_keyboard_teleop.py  # R1 Pro keyboard teleoperation

tests/                                 # pytest unit and Isaac Sim-gated integration tests
docs/wiki/                             # canonical GitHub Wiki source
```

## Tests

Run the complete test suite:

```bash
python -m pytest tests
```

Run the dependency-light assembly and tactile tests directly from the source tree:

```bash
PYTHONPATH=source/assembly_benchmark python -m pytest tests/test_one_leg_assembly.py -q
PYTHONPATH=source/assembly_benchmark python -m pytest \
  tests/test_vt_refine_tactile.py tests/test_r1_pro_tactile.py \
  tests/test_r1_pro_tactile_assets.py tests/test_r1_pro_robot_cfg.py -q
```

Isaac Sim-only tests use skip markers when their runtime dependencies are unavailable.

## Syntax Checks

```bash
python -m compileall -q source/assembly_benchmark/assembly_benchmark/assembly
python -m compileall -q source/assembly_benchmark/assembly_benchmark/sensors
python -m compileall -q scripts/tools/preview_assembly_assembled_pose.py
python -m compileall -q scripts/tools/generate_assembly_usd_assets.py
```

## Lightweight Assembly Contract Check

The registry and source asset contract can be checked without launching Isaac Sim:

```bash
PYTHONPATH=source/assembly_benchmark python - <<'PY'
from assembly_benchmark.assembly import available_assemblies, make_assembly

assert available_assemblies() == (
    "cabinet",
    "chair",
    "desk",
    "drawer",
    "lamp",
    "one_leg",
    "round_table",
    "square_table",
    "stool",
)
for assembly_name in available_assemblies():
    assembly = make_assembly(assembly_name)
    assert assembly.part_names[0] == "base_tag"
    assert assembly.reset_part_names == assembly.part_names[1:]
    assert assembly.assembly_relations
    for part in assembly.parts:
        assert part.urdf_path(assembly.asset_root).is_file(), part.urdf_path(assembly.asset_root)
print("assembly contract ok")
PY
```

## Pre-Commit

Install and run the repository hooks:

```bash
python -m pip install pre-commit
pre-commit run --all-files
```

The hooks run Ruff, formatting, codespell, license, and file-hygiene checks. Added files larger than 2 MB must be stored
through Git LFS.
