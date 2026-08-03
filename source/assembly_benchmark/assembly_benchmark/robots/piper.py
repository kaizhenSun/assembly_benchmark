# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for the AgileX Piper arm with a Pika2 parallel gripper."""

from __future__ import annotations

from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

PIPER_ASSET_DIR = Path(__file__).resolve().parents[1] / "assets" / "robots" / "piper"
PIPER_BASE_URDF_PATH = PIPER_ASSET_DIR / "piper_description.urdf"
PIPER_GRIPPER_URDF_PATH = PIPER_ASSET_DIR / "pika2_gripper.urdf"
# Compatibility alias retained for downstream callers of the former Xacro-based integration.
PIPER_GRIPPER_XACRO_PATH = PIPER_GRIPPER_URDF_PATH
PIPER_USD_PATH = PIPER_ASSET_DIR / "piper_fixed.usd"
PIPER_FIXED_PLUG_ASSET_DIR = PIPER_ASSET_DIR / "fixed_plug"

PIPER_ARM_JOINT_NAMES = [f"joint{index}" for index in range(1, 7)]
PIPER_GRIPPER_JOINT_NAMES = ["left_joint", "right_joint"]
PIPER_EE_LINK_NAME = "gripper_base_link"
PIPER_IK_LINK_NAME = "link6"
PIPER_GRIPPER_MIN = 0.0
PIPER_GRIPPER_MAX = 0.1
PIPER_DEFAULT_GRIPPER_OPENING = 0.0762
PIPER_GRIPPER_CLOSED_JOINT_POS = (0.0, 0.0)
PIPER_GRIPPER_OPEN_JOINT_POS = (0.05, -0.05)
PIPER_HOME_JOINT_POS = (0.0, 1.345086792, -0.673184517, 0.0, 0.986160525, 0.0)


def piper_fixed_plug_usd_path(assembly_name: str, relation_key: str) -> Path:
    """Return the generated Piper fixed-plug USD path for one assembly relation."""
    asset_dir = PIPER_FIXED_PLUG_ASSET_DIR / assembly_name / relation_key
    return asset_dir / "piper_fixed_plug.usd"


def piper_gripper_joint_positions(opening: float) -> tuple[float, float]:
    """Map total Pika2 jaw opening to the signed left and right prismatic joints."""
    if not PIPER_GRIPPER_MIN <= opening <= PIPER_GRIPPER_MAX:
        raise ValueError(f"Piper Pika2 opening must be in [{PIPER_GRIPPER_MIN}, {PIPER_GRIPPER_MAX}], got {opening}.")
    ratio = (opening - PIPER_GRIPPER_MIN) / (PIPER_GRIPPER_MAX - PIPER_GRIPPER_MIN)
    return tuple(
        closed + ratio * (opened - closed)
        for closed, opened in zip(PIPER_GRIPPER_CLOSED_JOINT_POS, PIPER_GRIPPER_OPEN_JOINT_POS, strict=True)
    )


PIPER_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=str(PIPER_USD_PATH),
        activate_contact_sensors=False,
        semantic_tags=[("class", "robot")],
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=True,
            max_depenetration_velocity=5.0,
            solver_position_iteration_count=192,
            solver_velocity_iteration_count=1,
        ),
        collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.001, rest_offset=0.0),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=192,
            solver_velocity_iteration_count=1,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        joint_pos={
            **dict(zip(PIPER_ARM_JOINT_NAMES, PIPER_HOME_JOINT_POS, strict=True)),
            **dict(
                zip(
                    PIPER_GRIPPER_JOINT_NAMES,
                    piper_gripper_joint_positions(PIPER_DEFAULT_GRIPPER_OPENING),
                    strict=True,
                )
            ),
        },
    ),
    actuators={
        "arm": ImplicitActuatorCfg(
            joint_names_expr=["joint[1-6]"],
            effort_limit_sim=100.0,
            velocity_limit_sim=5.0,
            stiffness=400.0,
            damping=40.0,
            armature=0.01,
        ),
        "gripper": ImplicitActuatorCfg(
            joint_names_expr=PIPER_GRIPPER_JOINT_NAMES,
            effort_limit_sim=100.0,
            velocity_limit_sim=1.0,
            stiffness=100.0,
            damping=10.0,
            armature=0.001,
        ),
    },
    soft_joint_pos_limit_factor=1.0,
)
"""Configuration for the fixed-base AgileX Piper robot."""

__all__ = [
    "PIPER_ARM_JOINT_NAMES",
    "PIPER_ASSET_DIR",
    "PIPER_BASE_URDF_PATH",
    "PIPER_CFG",
    "PIPER_DEFAULT_GRIPPER_OPENING",
    "PIPER_EE_LINK_NAME",
    "PIPER_FIXED_PLUG_ASSET_DIR",
    "PIPER_GRIPPER_CLOSED_JOINT_POS",
    "PIPER_GRIPPER_JOINT_NAMES",
    "PIPER_GRIPPER_MAX",
    "PIPER_GRIPPER_MIN",
    "PIPER_GRIPPER_OPEN_JOINT_POS",
    "PIPER_GRIPPER_URDF_PATH",
    "PIPER_GRIPPER_XACRO_PATH",
    "PIPER_HOME_JOINT_POS",
    "PIPER_IK_LINK_NAME",
    "PIPER_USD_PATH",
    "piper_fixed_plug_usd_path",
    "piper_gripper_joint_positions",
]
