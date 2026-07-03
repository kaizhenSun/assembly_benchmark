# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import CameraCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass

from assembly_benchmark.assets.furniture.lab_table import LAB_TABLE_SURFACE_Z, make_lab_table_cfg
from assembly_benchmark.assembly import make_assembly
from assembly_benchmark.assembly.isaac import make_assembly_part_cfg
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
    R1_PRO_CFG,
)


DEFAULT_ASSEMBLY = make_assembly("one_leg")
DEFAULT_ASSEMBLY_RELATION = DEFAULT_ASSEMBLY.primary_relation
DEFAULT_ASSEMBLY_RESET_PART_NAMES = DEFAULT_ASSEMBLY.reset_part_names
DEFAULT_ASSEMBLY_OBSERVATION_PART_NAMES = DEFAULT_ASSEMBLY.observation_part_names


@configclass
class AssemblyBenchmarkSceneCfg(InteractiveSceneCfg):
    """Generic assembly scene layout for R1 Pro assembly tasks."""

    ground = AssetBaseCfg(
        prim_path="/World/defaultGroundPlane",
        spawn=sim_utils.GroundPlaneCfg(),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
    )

    dome_light = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75)),
    )

    robot = R1_PRO_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    front_left_work_camera = CameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base_link/front_left_work_camera",
        update_period=0.0,
        height=480,
        width=640,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0,
            focus_distance=400.0,
            horizontal_aperture=20.955,
            clipping_range=(0.1, 20.0),
        ),
        offset=CameraCfg.OffsetCfg(
            pos=(2.5, 2.5, 2.5),
            rot=(0.458631, 0.187409, 0.099606, -0.862910),
            convention="world",
        ),
    )
    """RGB work camera for scene-level monitoring; videos are recorded through Gym RecordVideo."""

    lab_table = make_lab_table_cfg()

    base_tag = make_assembly_part_cfg(DEFAULT_ASSEMBLY, "base_tag")
    obstacle_front = make_assembly_part_cfg(DEFAULT_ASSEMBLY, "obstacle_front")
    obstacle_right = make_assembly_part_cfg(DEFAULT_ASSEMBLY, "obstacle_right")
    obstacle_left = make_assembly_part_cfg(DEFAULT_ASSEMBLY, "obstacle_left")
    square_table_top = make_assembly_part_cfg(DEFAULT_ASSEMBLY, "square_table_top")
    square_table_leg1 = make_assembly_part_cfg(DEFAULT_ASSEMBLY, "square_table_leg1")
    square_table_leg2 = make_assembly_part_cfg(DEFAULT_ASSEMBLY, "square_table_leg2")
    square_table_leg3 = make_assembly_part_cfg(DEFAULT_ASSEMBLY, "square_table_leg3")
    square_table_leg4 = make_assembly_part_cfg(DEFAULT_ASSEMBLY, "square_table_leg4")


@configclass
class AssemblyBenchmarkEnvCfg(DirectRLEnvCfg):
    """Generic R1 Pro assembly task with whole-body IK control."""

    decimation = 4
    episode_length_s = 50.0

    action_space = 16
    observation_space = 121
    state_space = 0

    sim: SimulationCfg = SimulationCfg(dt=1 / 240, render_interval=decimation)
    scene: AssemblyBenchmarkSceneCfg = AssemblyBenchmarkSceneCfg(
        num_envs=16, env_spacing=4.0, replicate_physics=True
    )

    table_surface_z = LAB_TABLE_SURFACE_Z
    assembly_name = DEFAULT_ASSEMBLY.name
    assembly_reset_part_names = DEFAULT_ASSEMBLY_RESET_PART_NAMES
    assembly_observation_part_names = DEFAULT_ASSEMBLY_OBSERVATION_PART_NAMES
    assembly_parent_part_name = DEFAULT_ASSEMBLY_RELATION.parent
    assembly_child_part_name = DEFAULT_ASSEMBLY_RELATION.child
    assembled_target_positions = tuple(target.pos for target in DEFAULT_ASSEMBLY_RELATION.target_poses)
    assembled_target_quats = tuple(target.quat for target in DEFAULT_ASSEMBLY_RELATION.target_poses)
    scripted_target_index = DEFAULT_ASSEMBLY_RELATION.default_target_index

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

    assembled_pos_threshold = DEFAULT_ASSEMBLY_RELATION.pos_threshold
    assembled_ori_bound = DEFAULT_ASSEMBLY_RELATION.ori_bound
    rew_scale_success = 1.0
