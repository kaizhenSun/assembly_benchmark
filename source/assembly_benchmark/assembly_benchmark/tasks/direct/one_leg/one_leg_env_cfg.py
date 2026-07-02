# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from isaaclab.envs import DirectRLEnvCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass

from assembly_benchmark.assets.furniture.lab_table import LAB_TABLE_SURFACE_Z
from assembly_benchmark.robots.r1_pro import (
    R1_PRO_LEFT_ARM_JOINT_NAMES,
    R1_PRO_LEFT_EE_LINK_NAME,
    R1_PRO_LEFT_GRIPPER_JOINT_NAMES,
    R1_PRO_LEFT_IK_LINK_NAME,
    R1_PRO_RIGHT_ARM_JOINT_NAMES,
    R1_PRO_RIGHT_EE_LINK_NAME,
    R1_PRO_RIGHT_GRIPPER_JOINT_NAMES,
    R1_PRO_RIGHT_IK_LINK_NAME,
    R1_PRO_TORSO_JOINT_NAMES,
)
from assembly_benchmark.tasks.direct.one_leg_scene.one_leg_scene_env_cfg import OneLegSceneCfg


@configclass
class OneLegWholeBodyIKEnvCfg(DirectRLEnvCfg):
    """R1 Pro FurnitureBench one_leg assembly task with whole-body IK control."""

    decimation = 4
    episode_length_s = 50.0

    action_space = 16
    observation_space = 121
    state_space = 0

    sim: SimulationCfg = SimulationCfg(dt=1 / 240, render_interval=decimation)
    scene: OneLegSceneCfg = OneLegSceneCfg(num_envs=16, env_spacing=4.0, replicate_physics=True)

    table_surface_z = LAB_TABLE_SURFACE_Z

    control_mode = "ik"
    torso_joint_names = R1_PRO_TORSO_JOINT_NAMES
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
    include_torso_in_ik = True

    assembled_pos_threshold = (0.010, 0.005, 0.010)
    assembled_ori_bound = 0.94
    rew_scale_success = 1.0
