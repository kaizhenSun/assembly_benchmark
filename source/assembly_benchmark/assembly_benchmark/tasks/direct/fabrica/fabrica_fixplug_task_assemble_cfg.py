# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for Fabrica's fixed-plug specialist assembly task."""

from __future__ import annotations

import gymnasium as gym
import numpy as np

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import PhysxCfg, SimulationCfg
from isaaclab.sim.spawners.materials.physics_materials_cfg import RigidBodyMaterialCfg
from isaaclab.utils import configclass

from assembly_benchmark.assembly import BEAM_FABRICA_PLAN, PIPER_FABRICA_BASE_POS, make_assembly
from assembly_benchmark.assets.furniture.lab_table import make_lab_table_cfg
from assembly_benchmark.robots.piper import (
    PIPER_ARM_JOINT_NAMES,
    PIPER_CFG,
    PIPER_EE_LINK_NAME,
    PIPER_GRIPPER_CLOSED_JOINT_POS,
    PIPER_GRIPPER_JOINT_NAMES,
    PIPER_GRIPPER_MAX,
    PIPER_GRIPPER_MIN,
    PIPER_GRIPPER_OPEN_JOINT_POS,
    PIPER_IK_LINK_NAME,
    piper_fixed_plug_usd_path,
    piper_gripper_joint_positions,
)

ASSEMBLY_PLAN = BEAM_FABRICA_PLAN
ASSEMBLY_SPEC = make_assembly(ASSEMBLY_PLAN.assembly_name)
RELATION_KEYS = ASSEMBLY_PLAN.relation_keys
EPISODE_STEPS = 128
POLICY_FREQUENCY_HZ = 30
SIMULATION_FREQUENCY_HZ = 120


def _robot_usd_paths() -> list[str]:
    return [
        str(piper_fixed_plug_usd_path(ASSEMBLY_PLAN.assembly_name, relation.key))
        for relation in ASSEMBLY_PLAN.relations
    ]


def _socket_usd_paths() -> list[str]:
    return [
        str(ASSEMBLY_SPEC.asset_root / "usd" / "fixed_plug_socket" / relation.socket_part_id / "socket.usd")
        for relation in ASSEMBLY_PLAN.relations
    ]


def _make_robot_cfg() -> ArticulationCfg:
    """Create a deterministic four-asset Piper articulation spawner."""
    spawn = PIPER_CFG.spawn
    if spawn is None:
        raise ValueError("PIPER_CFG must define a USD spawn configuration.")
    first_relation = ASSEMBLY_PLAN.relations[0]
    initial_joint_pos = dict(zip(PIPER_ARM_JOINT_NAMES, first_relation.piper_preassembly_joint_pos, strict=True))
    initial_joint_pos.update(
        zip(
            PIPER_GRIPPER_JOINT_NAMES,
            piper_gripper_joint_positions(first_relation.piper_gripper_opening),
            strict=True,
        )
    )
    multi_spawn = sim_utils.MultiUsdFileCfg(
        usd_path=_robot_usd_paths(),
        random_choice=False,
        activate_contact_sensors=spawn.activate_contact_sensors,
        semantic_tags=spawn.semantic_tags,
        rigid_props=spawn.rigid_props,
        collision_props=spawn.collision_props,
        articulation_props=spawn.articulation_props,
    )
    return PIPER_CFG.replace(
        prim_path="{ENV_REGEX_NS}/Robot",
        spawn=multi_spawn,
        init_state=PIPER_CFG.init_state.replace(
            pos=PIPER_FABRICA_BASE_POS,
            joint_pos=initial_joint_pos,
        ),
    )


def _make_socket_cfg() -> RigidObjectCfg:
    """Create the socket spawner paired with the fixed-plug robot list."""
    socket_part = ASSEMBLY_SPEC.part(ASSEMBLY_PLAN.part_scene_key(ASSEMBLY_PLAN.relations[0].socket_part_id))
    return RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Socket",
        spawn=sim_utils.MultiUsdFileCfg(
            usd_path=_socket_usd_paths(),
            random_choice=False,
            semantic_tags=[("class", "socket")],
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=True,
                kinematic_enabled=True,
                max_depenetration_velocity=5.0,
                solver_position_iteration_count=192,
                solver_velocity_iteration_count=1,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.001, rest_offset=0.0),
            mass_props=sim_utils.MassPropertiesCfg(density=1250.0),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=socket_part.init_pos, rot=socket_part.init_rot),
    )


@configclass
class FabricaFixPlugTaskAssembleSceneCfg(InteractiveSceneCfg):
    """Heterogeneous four-relation scene for Fabrica specialist training."""

    ground = AssetBaseCfg(
        prim_path="/World/defaultGroundPlane",
        spawn=sim_utils.GroundPlaneCfg(semantic_tags=[("class", "ground")]),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
    )
    dome_light = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75)),
    )
    robot: ArticulationCfg = _make_robot_cfg()
    socket: RigidObjectCfg = _make_socket_cfg()
    lab_table = make_lab_table_cfg()


@configclass
class FabricaFixPlugTaskAssembleCfg(DirectRLEnvCfg):
    """Direct RL configuration matching Fabrica's specialist contract."""

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
    scene: FabricaFixPlugTaskAssembleSceneCfg = FabricaFixPlugTaskAssembleSceneCfg(
        num_envs=1024,
        env_spacing=3.0,
        replicate_physics=False,
    )

    episode_steps = EPISODE_STEPS
    assembly_name = ASSEMBLY_PLAN.assembly_name
    relation_keys = RELATION_KEYS
    plug_body_name = "plug"
    socket_scene_key = "socket"
    socket_prim_path_suffix = "Socket/socket"
    tabletop_prim_path_suffix = "LabTable/Tabletop"
    socket_filtered_robot_body_names = (
        "link6",
        "gripper_base_link",
        "gripper_left_link",
        "gripper_right_link",
    )

    arm_joint_names = PIPER_ARM_JOINT_NAMES
    gripper_joint_names = PIPER_GRIPPER_JOINT_NAMES
    ee_link_name = PIPER_EE_LINK_NAME
    ik_link_name = PIPER_IK_LINK_NAME
    gripper_min = PIPER_GRIPPER_MIN
    gripper_max = PIPER_GRIPPER_MAX
    gripper_joint_closed_positions = PIPER_GRIPPER_CLOSED_JOINT_POS
    gripper_joint_open_positions = PIPER_GRIPPER_OPEN_JOINT_POS

    path_reference_distance = 0.02
    position_action_scale = 0.005
    socket_position_noise = (0.003, 0.003, 0.003)
    keypoint_reward_scale = 1.0e3
    keypoint_distance_cap = 0.03
    success_distance_threshold = 0.005
    deviation_distance_threshold = 0.008


__all__ = [
    "ASSEMBLY_PLAN",
    "EPISODE_STEPS",
    "FabricaFixPlugTaskAssembleCfg",
    "FabricaFixPlugTaskAssembleSceneCfg",
    "POLICY_FREQUENCY_HZ",
    "RELATION_KEYS",
    "SIMULATION_FREQUENCY_HZ",
]
