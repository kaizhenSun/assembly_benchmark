# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import CameraCfg
from isaaclab.sim import PhysxCfg, SimulationCfg
from isaaclab.sim.spawners.materials.physics_materials_cfg import RigidBodyMaterialCfg
from isaaclab.utils import configclass

from assembly_benchmark.assembly import available_assemblies, make_assembly
from assembly_benchmark.assembly.isaac import make_assembly_part_cfg
from assembly_benchmark.assembly.specs import AssemblySpec
from assembly_benchmark.assets.furniture.lab_table import LAB_TABLE_SURFACE_Z, make_lab_table_cfg
from assembly_benchmark.robots.r1_pro import (
    R1_PRO_ARM_JOINT_NAMES,
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
from assembly_benchmark.sensors import make_r1_pro_head_camera_cfg

DEFAULT_ASSEMBLY_NAME = "one_leg"
DEFAULT_NUM_ENVS = 16
DEFAULT_ENV_SPACING = 4.0
ARM_OBSERVATION_SIZE = 2 * len(R1_PRO_ARM_JOINT_NAMES)


def _assembly_class_prefix(assembly_name: str) -> str:
    """Convert an assembly registry name to a Python class/task component."""
    return "".join(part.capitalize() for part in assembly_name.replace("-", "_").split("_"))


def assembly_scene_cfg_class_name(assembly_name: str) -> str:
    """Return the generated scene cfg class name for an assembly."""
    return f"{_assembly_class_prefix(assembly_name)}AssemblyBenchmarkSceneCfg"


def assembly_env_cfg_class_name(assembly_name: str) -> str:
    """Return the generated env cfg class name for an assembly."""
    return f"{_assembly_class_prefix(assembly_name)}AssemblyBenchmarkEnvCfg"


def assembly_task_id(assembly_name: str) -> str:
    """Return the explicit Isaac Lab task id for an assembly."""
    return f"Assembly-Benchmark-{_assembly_class_prefix(assembly_name)}-Direct-v0"


@configclass
class AssemblyBenchmarkBaseSceneCfg(InteractiveSceneCfg):
    """Base R1 Pro scene shared by all assembly benchmark variants."""

    ground = AssetBaseCfg(
        prim_path="/World/defaultGroundPlane",
        spawn=sim_utils.GroundPlaneCfg(semantic_tags=[("class", "ground")]),
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

    head_camera = make_r1_pro_head_camera_cfg()
    """R1 Pro head-mounted RGB-D and semantic camera at the midpoint of the physical stereo pair."""

    lab_table = make_lab_table_cfg()


def make_assembly_scene_cfg_class(assembly: AssemblySpec, class_name: str) -> type[AssemblyBenchmarkBaseSceneCfg]:
    """Create an Isaac Lab scene cfg class by injecting assembly parts into the base scene."""
    existing_class = globals().get(class_name)
    if existing_class is not None:
        return existing_class

    annotations = {part.scene_key: AssetBaseCfg | RigidObjectCfg for part in assembly.parts}
    attrs = {
        "__module__": __name__,
        "__doc__": f"Scene cfg for the '{assembly.name}' assembly benchmark variant.",
        "__annotations__": annotations,
    }
    for part in assembly.parts:
        attrs[part.scene_key] = make_assembly_part_cfg(assembly, part.scene_key)

    return configclass(type(class_name, (AssemblyBenchmarkBaseSceneCfg,), attrs))


@configclass
class AssemblyBenchmarkEnvCfg(DirectRLEnvCfg):
    """Generic R1 Pro assembly task with whole-body IK control."""

    decimation = 4
    episode_length_s = 50.0

    action_space = 16
    observation_space = ARM_OBSERVATION_SIZE
    state_space = 0

    sim: SimulationCfg = SimulationCfg(
        device="cuda:0",
        dt=1 / 120,
        render_interval=decimation,
        gravity=(0.0, 0.0, -9.81),
        physx=PhysxCfg(
            solver_type=1,
            max_position_iteration_count=192,  # Important to avoid interpenetration.
            max_velocity_iteration_count=1,
            bounce_threshold_velocity=0.2,
            friction_offset_threshold=0.01,
            friction_correlation_distance=0.00625,
            gpu_max_rigid_contact_count=2**23,
            gpu_max_rigid_patch_count=2**23,
            gpu_collision_stack_size=2**28,
            gpu_max_num_partitions=1,  # Important for stable simulation.
        ),
        physics_material=RigidBodyMaterialCfg(
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
    )
    scene: AssemblyBenchmarkBaseSceneCfg = AssemblyBenchmarkBaseSceneCfg(
        num_envs=DEFAULT_NUM_ENVS,
        env_spacing=DEFAULT_ENV_SPACING,
        replicate_physics=True,
    )

    table_surface_z = LAB_TABLE_SURFACE_Z
    assembly_name = ""
    assembly_part_names = ()
    assembly_reset_part_names = ()
    assembly_parent_part_name = ""
    assembly_child_part_name = ""
    assembled_target_positions = ()
    assembled_target_quats = ()
    scripted_target_index = 0

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


def make_assembly_env_cfg_class(assembly_name: str, class_name: str) -> type[AssemblyBenchmarkEnvCfg]:
    """Create an Isaac Lab env cfg class for a registered assembly."""
    existing_class = globals().get(class_name)
    if existing_class is not None:
        return existing_class

    assembly = make_assembly(assembly_name)
    relation = assembly.primary_relation
    scene_class_name = assembly_scene_cfg_class_name(assembly.name)
    scene_cfg_cls = make_assembly_scene_cfg_class(assembly, scene_class_name)
    globals()[scene_class_name] = scene_cfg_cls

    attrs = {
        "__module__": __name__,
        "__doc__": f"Env cfg for the '{assembly.name}' assembly benchmark variant.",
        "__annotations__": {"scene": scene_cfg_cls},
        "scene": scene_cfg_cls(
            num_envs=DEFAULT_NUM_ENVS,
            env_spacing=DEFAULT_ENV_SPACING,
            replicate_physics=True,
        ),
        "assembly_name": assembly.name,
        "assembly_part_names": assembly.part_names,
        "assembly_reset_part_names": assembly.reset_part_names,
        "assembly_parent_part_name": relation.parent,
        "assembly_child_part_name": relation.child,
        "assembled_target_positions": tuple(target.pos for target in relation.target_poses),
        "assembled_target_quats": tuple(target.quat for target in relation.target_poses),
        "scripted_target_index": relation.default_target_index,
        "assembled_pos_threshold": relation.pos_threshold,
        "assembled_ori_bound": relation.ori_bound,
    }
    return configclass(type(class_name, (AssemblyBenchmarkEnvCfg,), attrs))


def _install_registered_assembly_cfg_classes() -> None:
    """Expose generated cfg classes as module globals for Isaac Lab entry points."""
    for assembly_name in available_assemblies():
        env_class_name = assembly_env_cfg_class_name(assembly_name)
        globals()[env_class_name] = make_assembly_env_cfg_class(assembly_name, env_class_name)


_install_registered_assembly_cfg_classes()


__all__ = [
    "AssemblyBenchmarkBaseSceneCfg",
    "AssemblyBenchmarkEnvCfg",
    "ARM_OBSERVATION_SIZE",
    "DEFAULT_ASSEMBLY_NAME",
    "assembly_env_cfg_class_name",
    "assembly_scene_cfg_class_name",
    "assembly_task_id",
    "make_assembly_env_cfg_class",
    "make_assembly_scene_cfg_class",
]
