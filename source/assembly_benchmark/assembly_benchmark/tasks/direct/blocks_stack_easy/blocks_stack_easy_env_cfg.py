# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass

from assembly_benchmark.assets.furniture.lab_table import LAB_TABLE_SURFACE_Z, make_lab_table_cfg
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
    R1_PRO_TORSO_JOINT_NAMES,
)


@configclass
class BlocksStackEasySceneCfg(InteractiveSceneCfg):
    """BlocksStackEasy scene migrated from GalaxeaManipSim."""

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

    lab_table = make_lab_table_cfg()

    block1 = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Block1",
        spawn=sim_utils.CuboidCfg(
            size=(0.04, 0.04, 0.04),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(max_depenetration_velocity=5.0, disable_gravity=False),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.05),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=0.5, dynamic_friction=0.5, restitution=0.6
            ),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.38, 0.15, 1.15), rot=(1.0, 0.0, 0.0, 0.0)
        ),
    )

    block2 = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Block2",
        spawn=sim_utils.CuboidCfg(
            size=(0.04, 0.04, 0.04),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(max_depenetration_velocity=5.0, disable_gravity=False),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.05),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=0.5, dynamic_friction=0.5, restitution=0.6
            ),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.42, -0.15, 1.15), rot=(1.0, 0.0, 0.0, 0.0)
        ),
    )


@configclass
class BlocksStackEasyEnvCfg(DirectRLEnvCfg):
    """Base configuration for R1 Pro BlocksStackEasy task shells."""

    decimation = 2
    episode_length_s = 350 / 15

    action_space = 16
    observation_space = 78
    state_space = 0

    sim: SimulationCfg = SimulationCfg(dt=1 / 120, render_interval=decimation)
    scene: BlocksStackEasySceneCfg = BlocksStackEasySceneCfg(
        num_envs=16, env_spacing=4.0, replicate_physics=True
    )

    table_surface_z = LAB_TABLE_SURFACE_Z

    control_mode = "joint"
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
    include_torso_in_ik = False

    success_tolerance = (0.025, 0.025, 0.01)

    rew_scale_success = 1.0


@configclass
class BlocksStackEasyJointEnvCfg(BlocksStackEasyEnvCfg):
    """R1 Pro BlocksStackEasy joint-position control task shell."""

    control_mode = "joint"


@configclass
class BlocksStackEasyIKEnvCfg(BlocksStackEasyEnvCfg):
    """R1 Pro BlocksStackEasy Differential IK control task shell."""

    control_mode = "ik"
