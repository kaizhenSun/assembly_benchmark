# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Fabrica fixed-plug specialist tasks."""

import gymnasium as gym

from . import agents

TASK_ID = "Assembly-Benchmark-FabricaFixPlugTaskAssemble-Direct-v0"

gym.register(
    id=TASK_ID,
    entry_point=f"{__name__}.fabrica_fixplug_task_assemble:FabricaFixPlugTaskAssemble",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.fabrica_fixplug_task_assemble_cfg:FabricaFixPlugTaskAssembleCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:FabricaFixPlugTaskAssemblePPO.yaml",
    },
)

from .fabrica_fixplug_task_assemble import FabricaFixPlugTaskAssemble
from .fabrica_fixplug_task_assemble_cfg import (
    FabricaFixPlugTaskAssembleCfg,
    FabricaFixPlugTaskAssembleSceneCfg,
)

__all__ = [
    "FabricaFixPlugTaskAssemble",
    "FabricaFixPlugTaskAssembleCfg",
    "FabricaFixPlugTaskAssembleSceneCfg",
    "TASK_ID",
]
