# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass

from assembly_benchmark.robots.r1_pro import (
    R1_PRO_CFG,
    R1_PRO_LEFT_ARM_JOINT_NAMES,
    R1_PRO_LEFT_EE_LINK_NAME,
    R1_PRO_LEFT_GRIPPER_JOINT_NAMES,
    R1_PRO_LEFT_IK_LINK_NAME,
    R1_PRO_RIGHT_ARM_JOINT_NAMES,
    R1_PRO_RIGHT_EE_LINK_NAME,
    R1_PRO_RIGHT_GRIPPER_JOINT_NAMES,
    R1_PRO_RIGHT_IK_LINK_NAME,
)


@configclass
class AssemblyR1ProEnvCfg(DirectRLEnvCfg):
    """Base configuration for R1 Pro direct smoke-test environments."""

    decimation = 2
    episode_length_s = 5.0

    action_space = 16
    observation_space = 50
    state_space = 0

    sim: SimulationCfg = SimulationCfg(dt=1 / 120, render_interval=decimation)
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=16, env_spacing=4.0, replicate_physics=True)
    robot_cfg = R1_PRO_CFG.replace(prim_path="/World/envs/env_.*/Robot")

    control_mode = "joint"
    left_arm_joint_names = R1_PRO_LEFT_ARM_JOINT_NAMES
    right_arm_joint_names = R1_PRO_RIGHT_ARM_JOINT_NAMES
    left_gripper_joint_names = R1_PRO_LEFT_GRIPPER_JOINT_NAMES
    right_gripper_joint_names = R1_PRO_RIGHT_GRIPPER_JOINT_NAMES
    left_ee_link_name = R1_PRO_LEFT_EE_LINK_NAME
    right_ee_link_name = R1_PRO_RIGHT_EE_LINK_NAME
    left_ik_link_name = R1_PRO_LEFT_IK_LINK_NAME
    right_ik_link_name = R1_PRO_RIGHT_IK_LINK_NAME

    arm_action_scale = 0.5
    gripper_min = 0.0
    gripper_max = 0.05

    rew_scale_alive = 0.1
    rew_scale_action_penalty = -0.01
    rew_scale_joint_limit = -0.05


@configclass
class AssemblyR1ProJointEnvCfg(AssemblyR1ProEnvCfg):
    """R1 Pro joint-position control smoke task."""

    control_mode = "joint"


@configclass
class AssemblyR1ProIKEnvCfg(AssemblyR1ProEnvCfg):
    """R1 Pro bimanual Differential IK smoke task."""

    control_mode = "ik"
