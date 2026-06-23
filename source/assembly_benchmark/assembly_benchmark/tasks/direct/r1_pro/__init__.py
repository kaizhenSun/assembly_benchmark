# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""R1 Pro direct smoke-test environments."""

import gymnasium as gym

gym.register(
    id="Assembly-R1Pro-Joint-Direct-v0",
    entry_point=f"{__name__}.r1_pro_env:AssemblyR1ProEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.r1_pro_env_cfg:AssemblyR1ProJointEnvCfg",
    },
)

gym.register(
    id="Assembly-R1Pro-IK-Direct-v0",
    entry_point=f"{__name__}.r1_pro_env:AssemblyR1ProEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.r1_pro_env_cfg:AssemblyR1ProIKEnvCfg",
    },
)
