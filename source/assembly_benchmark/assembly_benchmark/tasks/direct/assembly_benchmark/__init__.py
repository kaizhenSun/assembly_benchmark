# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Generic R1 Pro assembly benchmark task."""

import gymnasium as gym

from assembly_benchmark.assembly import available_assemblies

from . import agents
from .assembly_benchmark_env_cfg import (
    DEFAULT_ASSEMBLY_NAME,
    assembly_env_cfg_class_name,
    assembly_task_id,
)


def _register_assembly_task(task_id: str, env_cfg_class_name: str) -> None:
    gym.register(
        id=task_id,
        entry_point=f"{__name__}.assembly_benchmark_env:AssemblyBenchmarkEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": (
                f"{__name__}.assembly_benchmark_env_cfg:{env_cfg_class_name}"
            ),
            "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
            "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PPORunnerCfg",
            "skrl_amp_cfg_entry_point": f"{agents.__name__}:skrl_amp_cfg.yaml",
            "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
            "sb3_cfg_entry_point": f"{agents.__name__}:sb3_ppo_cfg.yaml",
        },
    )


_register_assembly_task(
    "Assembly-Benchmark-Direct-v0",
    assembly_env_cfg_class_name(DEFAULT_ASSEMBLY_NAME),
)

for assembly_name in available_assemblies():
    _register_assembly_task(
        assembly_task_id(assembly_name),
        assembly_env_cfg_class_name(assembly_name),
    )
