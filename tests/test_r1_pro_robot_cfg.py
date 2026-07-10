# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import importlib.util

import pytest


@pytest.mark.skipif(importlib.util.find_spec("carb") is None, reason="Isaac Sim runtime is not available")
def test_r1_pro_gripper_home_position_is_fully_open() -> None:
    from assembly_benchmark.robots.r1_pro import (
        R1_PRO_CFG,
        R1_PRO_GRIPPER_HOME_POS,
        R1_PRO_GRIPPER_JOINT_NAMES,
    )

    assert R1_PRO_GRIPPER_HOME_POS == 0.05
    assert R1_PRO_CFG.init_state.joint_pos is not None
    assert {
        joint_name: R1_PRO_CFG.init_state.joint_pos[joint_name] for joint_name in R1_PRO_GRIPPER_JOINT_NAMES
    } == {joint_name: R1_PRO_GRIPPER_HOME_POS for joint_name in R1_PRO_GRIPPER_JOINT_NAMES}
