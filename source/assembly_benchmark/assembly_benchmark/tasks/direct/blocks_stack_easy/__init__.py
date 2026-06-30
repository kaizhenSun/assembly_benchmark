# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""R1 Pro BlocksStackEasy direct task shells."""

import gymnasium as gym

gym.register(
    id="Assembly-R1Pro-BlocksStackEasy-Joint-Direct-v0",
    entry_point=f"{__name__}.blocks_stack_easy_env:BlocksStackEasyEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.blocks_stack_easy_env_cfg:BlocksStackEasyJointEnvCfg",
    },
)

gym.register(
    id="Assembly-R1Pro-BlocksStackEasy-IK-Direct-v0",
    entry_point=f"{__name__}.blocks_stack_easy_env:BlocksStackEasyEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.blocks_stack_easy_env_cfg:BlocksStackEasyIKEnvCfg",
    },
)
