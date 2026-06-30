# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for the Galaxea R1 Pro robot."""

from __future__ import annotations

from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

R1_PRO_ASSET_DIR = Path(__file__).resolve().parents[1] / "assets" / "robots" / "r1_pro"
R1_PRO_URDF_PATH = R1_PRO_ASSET_DIR / "robot.urdf"
R1_PRO_USD_PATH = R1_PRO_ASSET_DIR / "r1_pro_fixed.usd"
R1_PRO_MESH_DIR = R1_PRO_ASSET_DIR / "meshes"

R1_PRO_TORSO_JOINT_NAMES = [f"torso_joint{i}" for i in range(1, 5)]
R1_PRO_LEFT_ARM_JOINT_NAMES = [f"left_arm_joint{i}" for i in range(1, 8)]
R1_PRO_RIGHT_ARM_JOINT_NAMES = [f"right_arm_joint{i}" for i in range(1, 8)]
R1_PRO_LEFT_GRIPPER_JOINT_NAMES = [f"left_gripper_finger_joint{i}" for i in range(1, 3)]
R1_PRO_RIGHT_GRIPPER_JOINT_NAMES = [f"right_gripper_finger_joint{i}" for i in range(1, 3)]
R1_PRO_ARM_JOINT_NAMES = R1_PRO_LEFT_ARM_JOINT_NAMES + R1_PRO_RIGHT_ARM_JOINT_NAMES
R1_PRO_GRIPPER_JOINT_NAMES = R1_PRO_LEFT_GRIPPER_JOINT_NAMES + R1_PRO_RIGHT_GRIPPER_JOINT_NAMES
R1_PRO_CONTROLLED_JOINT_NAMES = R1_PRO_ARM_JOINT_NAMES + R1_PRO_GRIPPER_JOINT_NAMES

R1_PRO_TORSO_HOME_POS = [0.8000, -1.8600, -1.3999, 0.0000]
R1_PRO_LEFT_ARM_HOME_POS = [-0.4, 1.3, -0.7, -1.57, 1.3, -0.4, -0.8]
R1_PRO_RIGHT_ARM_HOME_POS = [-0.4, -1.3, 0.7, -1.57, -1.3, -0.4, 0.8]

R1_PRO_LEFT_EE_LINK_NAME = "left_gripper_link"
R1_PRO_RIGHT_EE_LINK_NAME = "right_gripper_link"
R1_PRO_LEFT_IK_LINK_NAME = "left_arm_link7"
R1_PRO_RIGHT_IK_LINK_NAME = "right_arm_link7"
R1_PRO_HEAD_CAMERA_LINK_NAME = "zed_link"
R1_PRO_LEFT_WRIST_CAMERA_LINK_NAME = "left_realsense_link"
R1_PRO_RIGHT_WRIST_CAMERA_LINK_NAME = "right_realsense_link"

R1_PRO_HOME_JOINT_POS = {
    **dict(zip(R1_PRO_TORSO_JOINT_NAMES, R1_PRO_TORSO_HOME_POS, strict=True)),
    **dict(zip(R1_PRO_LEFT_ARM_JOINT_NAMES, R1_PRO_LEFT_ARM_HOME_POS, strict=True)),
    **dict(zip(R1_PRO_RIGHT_ARM_JOINT_NAMES, R1_PRO_RIGHT_ARM_HOME_POS, strict=True)),
    **{name: 0.0 for name in R1_PRO_GRIPPER_JOINT_NAMES},
}

R1_PRO_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=str(R1_PRO_USD_PATH),
        activate_contact_sensors=False,
        # NOTE:
        # 关闭机器人本体重力可以让位置控制/IK 遥操作更容易稳定：
        # 各 link 不会因自重下垂，隐式关节执行器不需要额外重力补偿或更仔细的增益调参，
        # 就能更容易跟踪目标。
        #
        # 这适合做运动学、接触和键盘遥操作调试，但会带来 sim2real gap。
        # 真实机器人一定受重力影响，因此动力学验证、负载测试和 sim2real 实验最终应开启
        # 机器人重力，并配合合适的 actuator gains / 重力补偿。
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=True,
            max_depenetration_velocity=5.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
            solver_position_iteration_count=16,
            solver_velocity_iteration_count=4,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.0),
        joint_pos=R1_PRO_HOME_JOINT_POS,
    ),
    actuators={
        "torso": ImplicitActuatorCfg(
            joint_names_expr=["torso_joint[1-4]"],
            effort_limit_sim=3500.0,
            velocity_limit_sim=1.0,
            stiffness=16000.0,
            damping=1800.0,
            armature=0.08,
        ),
        "left_arm": ImplicitActuatorCfg(
            joint_names_expr=["left_arm_joint[1-7]"],
            effort_limit_sim=3000.0,
            velocity_limit_sim={
                "left_arm_joint[1-2]": 7.1209,
                "left_arm_joint[3-4]": 8.3776,
                "left_arm_joint[5-7]": 10.4720,
            },
            stiffness=10000.0,
            damping=1000.0,
            armature=0.04,
        ),
        "right_arm": ImplicitActuatorCfg(
            joint_names_expr=["right_arm_joint[1-7]"],
            effort_limit_sim=3000.0,
            velocity_limit_sim={
                "right_arm_joint[1-2]": 7.1209,
                "right_arm_joint[3-4]": 8.3776,
                "right_arm_joint[5-7]": 10.4720,
            },
            stiffness=10000.0,
            damping=1000.0,
            armature=0.04,
        ),
        "left_gripper": ImplicitActuatorCfg(
            joint_names_expr=["left_gripper_finger_joint[1-2]"],
            effort_limit_sim=500.0,
            velocity_limit_sim=0.25,
            stiffness=5000.0,
            damping=100.0,
            armature=0.001,
        ),
        "right_gripper": ImplicitActuatorCfg(
            joint_names_expr=["right_gripper_finger_joint[1-2]"],
            effort_limit_sim=500.0,
            velocity_limit_sim=0.25,
            stiffness=5000.0,
            damping=100.0,
            armature=0.001,
        ),
    },
    soft_joint_pos_limit_factor=1.0,
)
"""Configuration for the fixed-base Galaxea R1 Pro robot."""
