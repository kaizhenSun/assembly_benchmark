# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""R1 Pro FurnitureBench one_leg scene-loading task."""

import gymnasium as gym

gym.register(
    id="Assembly-R1Pro-OneLegScene-Direct-v0",
    entry_point=f"{__name__}.one_leg_scene_env:R1ProOneLegSceneEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.one_leg_scene_env_cfg:R1ProOneLegSceneEnvCfg",
    },
)
