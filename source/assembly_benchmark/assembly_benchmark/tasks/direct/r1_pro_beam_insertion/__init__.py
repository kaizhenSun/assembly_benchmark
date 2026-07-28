# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Dedicated R1 Pro Beam 0-to-2 left-arm insertion task."""

import gymnasium as gym

from . import agents

TASK_ID = "Assembly-Benchmark-Beam02-LeftInsert-Direct-v0"

gym.register(
    id=TASK_ID,
    entry_point=f"{__name__}.r1_pro_beam_insertion_env:R1ProBeam02InsertionEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.r1_pro_beam_insertion_env_cfg:R1ProBeam02InsertionEnvCfg"
        ),
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
    },
)

__all__ = ["TASK_ID"]
