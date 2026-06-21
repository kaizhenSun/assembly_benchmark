# Assembly Benchmark

Assembly Benchmark is an Isaac Lab extension for assembly-task reinforcement learning experiments.

The extension registers the task:

```text
Assembly-Benchmark-Direct-v0
```

## Requirements

- Isaac Lab installed and usable from the command line.
- Python environment from Isaac Lab, conda, or uv.

See the Isaac Lab installation guide:
https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html

## Installation

Install this extension in editable mode:

```bash
python -m pip install -e source/assembly_benchmark
```

If Isaac Lab is not installed in the active Python environment, use Isaac Lab's launcher instead:

```bash
<PATH_TO_ISAACLAB>/isaaclab.sh -p -m pip install -e source/assembly_benchmark
```

## Common Commands

List registered environments:

```bash
python scripts/list_envs.py
```

Run a zero-action smoke test:

```bash
python scripts/zero_agent.py --task=Assembly-Benchmark-Direct-v0
```

Run a random-action smoke test:

```bash
python scripts/random_agent.py --task=Assembly-Benchmark-Direct-v0
```

Train with RSL-RL:

```bash
python scripts/rsl_rl/train.py --task=Assembly-Benchmark-Direct-v0
```

Play a trained RSL-RL policy:

```bash
python scripts/rsl_rl/play.py --task=Assembly-Benchmark-Direct-v0
```

Other RL backends are available under `scripts/rl_games`, `scripts/sb3`, and `scripts/skrl`.

## Project Layout

```text
source/assembly_benchmark/
  assembly_benchmark/
    tasks/direct/assembly_benchmark/   # task registration, environment, config, agents
  config/extension.toml                # Isaac Lab extension metadata

scripts/
  list_envs.py                         # list available tasks
  zero_agent.py                        # zero-action smoke test
  random_agent.py                      # random-action smoke test
  rsl_rl/                              # RSL-RL train/play scripts
```

## Development

Format and check code with pre-commit:

```bash
pip install pre-commit
pre-commit run --all-files
```

For VS Code indexing, run the `setup_python_env` task from `Tasks: Run Task` and provide the absolute path to the Isaac Sim installation when prompted.
