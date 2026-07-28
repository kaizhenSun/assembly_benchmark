# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for the R1 Pro left-arm Beam 0-to-2 insertion task."""

from __future__ import annotations

import math

import gymnasium as gym
import numpy as np

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import PhysxCfg, SimulationCfg
from isaaclab.sim.spawners.materials.physics_materials_cfg import RigidBodyMaterialCfg
from isaaclab.utils import configclass

from assembly_benchmark.assembly import make_assembly
from assembly_benchmark.assembly.isaac import make_assembly_part_cfg
from assembly_benchmark.assets.furniture.lab_table import make_lab_table_cfg
from assembly_benchmark.beam02_grasp import BEAM02_GRIPPER_TO_PLUG_POS, BEAM02_GRIPPER_TO_PLUG_QUAT
from assembly_benchmark.planning import BEAM02_APPROACH_DISTANCE
from assembly_benchmark.robots.r1_pro import (
    R1_PRO_BEAM02_USD_PATH,
    R1_PRO_CFG,
    R1_PRO_GRIPPER_HOME_POS,
    R1_PRO_HOME_JOINT_POS,
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

BEAM_ASSEMBLY_NAME = "beam"
BEAM_PLUG_BODY_NAME = "beam_plug_0"
BEAM_SOCKET_SCENE_KEY = "beam_socket_2"
EPISODE_STEPS = 128
POLICY_FREQUENCY_HZ = 30
SIMULATION_FREQUENCY_HZ = 120
BEAM02_LEFT_GRIPPER_POS = 0.00635
BEAM02_GRIPPER_MIN = 0.0
BEAM02_GRIPPER_MAX = 0.05
BEAM02_LEFT_GRIPPER_ACTION = (
    2.0 * (BEAM02_LEFT_GRIPPER_POS - BEAM02_GRIPPER_MIN) / (BEAM02_GRIPPER_MAX - BEAM02_GRIPPER_MIN) - 1.0
)
BEAM02_RIGHT_GRIPPER_ACTION = (
    2.0 * (R1_PRO_GRIPPER_HOME_POS - BEAM02_GRIPPER_MIN) / (BEAM02_GRIPPER_MAX - BEAM02_GRIPPER_MIN) - 1.0
)
BEAM02_WHOLE_BODY_HOME_POS = (
    0.796419143480,
    -2.048491624328,
    -1.148481627732,
    -1.074710355439,
    0.333698777286,
    1.981330656756,
    -1.388202956678,
    -1.310799378479,
    1.265140892687,
    -0.316396494036,
    -0.654163365677,
)

_BEAM_ASSEMBLY = make_assembly(BEAM_ASSEMBLY_NAME)
_BEAM_RELATION = _BEAM_ASSEMBLY.primary_relation
if _BEAM_RELATION.parent != BEAM_SOCKET_SCENE_KEY or _BEAM_RELATION.child != BEAM_PLUG_BODY_NAME:
    raise ValueError(
        "The dedicated Beam 0-to-2 task requires the primary relation "
        f"{BEAM_SOCKET_SCENE_KEY} -> {BEAM_PLUG_BODY_NAME}."
    )

_BEAM_TARGET = _BEAM_RELATION.default_target_pose
_BEAM_SOCKET_CFG = make_assembly_part_cfg(_BEAM_ASSEMBLY, BEAM_SOCKET_SCENE_KEY)
if not isinstance(_BEAM_SOCKET_CFG, RigidObjectCfg):
    raise TypeError(f"{BEAM_SOCKET_SCENE_KEY} must be a resettable RigidObjectCfg.")


def _make_beam02_robot_cfg() -> ArticulationCfg:
    """Use the R1 Pro articulation with Beam plug 0 fixed to the left gripper."""
    if R1_PRO_CFG.spawn is None:
        raise ValueError("R1_PRO_CFG must define a USD spawn configuration.")
    home_joint_pos = dict(R1_PRO_HOME_JOINT_POS)
    whole_body_joint_names = R1_PRO_TORSO_JOINT_NAMES + R1_PRO_LEFT_ARM_JOINT_NAMES
    home_joint_pos.update(dict(zip(whole_body_joint_names, BEAM02_WHOLE_BODY_HOME_POS, strict=True)))
    home_joint_pos.update({name: BEAM02_LEFT_GRIPPER_POS for name in R1_PRO_LEFT_GRIPPER_JOINT_NAMES})
    return R1_PRO_CFG.replace(
        prim_path="{ENV_REGEX_NS}/Robot",
        spawn=R1_PRO_CFG.spawn.replace(usd_path=str(R1_PRO_BEAM02_USD_PATH)),
        init_state=R1_PRO_CFG.init_state.replace(joint_pos=home_joint_pos),
    )


@configclass
class R1ProBeam02InsertionSceneCfg(InteractiveSceneCfg):
    """Camera-free scene for vectorized Beam 0-to-2 insertion training."""

    ground = AssetBaseCfg(
        prim_path="/World/defaultGroundPlane",
        spawn=sim_utils.GroundPlaneCfg(semantic_tags=[("class", "ground")]),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
    )
    dome_light = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75)),
    )
    robot: ArticulationCfg = _make_beam02_robot_cfg()
    beam_socket_2: RigidObjectCfg = _BEAM_SOCKET_CFG
    lab_table = make_lab_table_cfg()


@configclass
class R1ProBeam02InsertionEnvCfg(DirectRLEnvCfg):
    """Direct RL configuration matching Fabrica's fixed-plug specialist task."""

    decimation = SIMULATION_FREQUENCY_HZ // POLICY_FREQUENCY_HZ
    episode_length_s = EPISODE_STEPS / POLICY_FREQUENCY_HZ

    action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)
    observation_space = 3
    state_space = 6

    sim: SimulationCfg = SimulationCfg(
        device="cuda:0",
        dt=1.0 / SIMULATION_FREQUENCY_HZ,
        render_interval=decimation,
        gravity=(0.0, 0.0, -9.81),
        physx=PhysxCfg(
            solver_type=1,
            max_position_iteration_count=192,
            max_velocity_iteration_count=1,
            bounce_threshold_velocity=0.2,
            friction_offset_threshold=0.01,
            friction_correlation_distance=0.00625,
            gpu_max_rigid_contact_count=2**23,
            gpu_max_rigid_patch_count=2**23,
            gpu_collision_stack_size=2**28,
            gpu_max_num_partitions=1,
        ),
        physics_material=RigidBodyMaterialCfg(static_friction=1.0, dynamic_friction=1.0),
    )
    scene: R1ProBeam02InsertionSceneCfg = R1ProBeam02InsertionSceneCfg(
        num_envs=1024,
        env_spacing=3.0,
        replicate_physics=True,
    )

    episode_steps = EPISODE_STEPS
    assembly_name = BEAM_ASSEMBLY_NAME
    plug_body_name = BEAM_PLUG_BODY_NAME
    socket_scene_key = BEAM_SOCKET_SCENE_KEY
    socket_prim_name = _BEAM_ASSEMBLY.part(BEAM_SOCKET_SCENE_KEY).prim_name
    tabletop_prim_path_suffix = "LabTable/Tabletop"
    assembled_target_pos = _BEAM_TARGET.pos
    assembled_target_quat = _BEAM_TARGET.quat

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
    gripper_min = BEAM02_GRIPPER_MIN
    gripper_max = BEAM02_GRIPPER_MAX
    left_gripper_action = BEAM02_LEFT_GRIPPER_ACTION
    right_gripper_action = BEAM02_RIGHT_GRIPPER_ACTION
    gripper_to_plug_pos = BEAM02_GRIPPER_TO_PLUG_POS
    gripper_to_plug_quat = BEAM02_GRIPPER_TO_PLUG_QUAT

    approach_axis_socket = (0.0, 0.0, 1.0)
    approach_distance = BEAM02_APPROACH_DISTANCE
    path_reference_distance = 0.02
    position_action_scale = 0.002
    socket_position_noise = (0.003, 0.003, 0.003)

    keypoint_reward_scale = 1.0e3
    keypoint_distance_cap = 0.03
    success_distance_threshold = 0.005
    deviation_distance_threshold = 0.008
    right_ee_position_drift_threshold = 0.002
    right_ee_orientation_drift_threshold = math.radians(2.0)
    right_ee_final_position_drift_threshold = 0.001
    right_ee_final_orientation_drift_threshold = math.radians(1.0)


__all__ = [
    "BEAM_ASSEMBLY_NAME",
    "BEAM_PLUG_BODY_NAME",
    "BEAM_SOCKET_SCENE_KEY",
    "BEAM02_GRIPPER_MAX",
    "BEAM02_GRIPPER_MIN",
    "BEAM02_LEFT_GRIPPER_ACTION",
    "BEAM02_LEFT_GRIPPER_POS",
    "BEAM02_RIGHT_GRIPPER_ACTION",
    "BEAM02_WHOLE_BODY_HOME_POS",
    "EPISODE_STEPS",
    "POLICY_FREQUENCY_HZ",
    "R1ProBeam02InsertionEnvCfg",
    "R1ProBeam02InsertionSceneCfg",
    "SIMULATION_FREQUENCY_HZ",
]
