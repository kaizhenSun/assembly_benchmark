# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Scripted IK auto-grasp demo for the R1 Pro BlocksStackEasy task.

This script is intentionally non-invasive: it does not change the RL task action
space or observation space. It drives the IK task with absolute EE pose actions.
By default, blocks are grasped through physical contact only; the legacy
kinematic attachment path is available as an explicit debug mode.

.. code-block:: bash

    python scripts/tools/run_r1_pro_blocks_stack_easy_auto_grasp.py --num_envs 1 --device cuda:0

"""

from __future__ import annotations

import argparse
import os
import sys
import traceback
from collections.abc import Callable

from isaaclab.app import AppLauncher


TASK_NAME = "Assembly-R1Pro-BlocksStackEasy-IK-Direct-v0"


parser = argparse.ArgumentParser(description="Run scripted R1 Pro BlocksStackEasy auto-grasp.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments. Only 1 is supported.")
parser.add_argument(
    "--disable_fabric",
    action="store_true",
    default=False,
    help="Debug/compatibility option: disable Fabric and use USD I/O, which may desync GUI mesh updates.",
)
parser.add_argument("--phase_steps", type=int, default=90, help="Number of simulation steps for each scripted phase.")
parser.add_argument("--enable_robot_gravity", action="store_true", help="Enable gravity on robot links for this run.")
parser.add_argument(
    "--include_torso_in_ik",
    action="store_true",
    help="Include torso joints in the bimanual IK solve for this scripted run.",
)
parser.add_argument("--torso_stiffness", type=float, default=None, help="Override torso actuator stiffness.")
parser.add_argument("--torso_damping", type=float, default=None, help="Override torso actuator damping.")
parser.add_argument("--torso_effort", type=float, default=None, help="Override torso actuator effort limit.")
parser.add_argument("--torso_armature", type=float, default=None, help="Override torso actuator armature.")
parser.add_argument("--arm_stiffness", type=float, default=None, help="Override both arm actuator stiffness.")
parser.add_argument("--arm_damping", type=float, default=None, help="Override both arm actuator damping.")
parser.add_argument("--arm_effort", type=float, default=None, help="Override both arm actuator effort limit.")
parser.add_argument("--arm_armature", type=float, default=None, help="Override both arm actuator armature.")
parser.add_argument("--gripper_stiffness", type=float, default=None, help="Override both gripper actuator stiffness.")
parser.add_argument("--gripper_damping", type=float, default=None, help="Override both gripper actuator damping.")
parser.add_argument("--gripper_effort", type=float, default=None, help="Override both gripper actuator effort limit.")
parser.add_argument("--gripper_armature", type=float, default=None, help="Override both gripper actuator armature.")
parser.add_argument(
    "--solver_position_iterations",
    type=int,
    default=None,
    help="Override articulation solver position iterations.",
)
parser.add_argument(
    "--solver_velocity_iterations",
    type=int,
    default=None,
    help="Override articulation solver velocity iterations.",
)
parser.add_argument(
    "--fast_exit",
    action="store_true",
    help="Exit the process immediately after the scripted run, avoiding slow Kit shutdown.",
)
parser.add_argument(
    "--grasp_mode",
    choices=("physical", "kinematic"),
    default="physical",
    help="Use real contact grasping by default, or the legacy kinematic attachment debug path.",
)
parser.add_argument(
    "--print_interval",
    type=int,
    default=30,
    help="Print phase diagnostics every N scripted steps. Use 0 to disable.",
)
parser.add_argument("--settle_steps", type=int, default=30, help="Steps to pause at the physical grasp pose.")
parser.add_argument("--close_steps", type=int, default=120, help="Steps used to close or open the gripper slowly.")
parser.add_argument("--lift_height", type=float, default=0.16, help="Physical lift height above the grasp/place pose.")
parser.add_argument(
    "--overhead_clearance",
    type=float,
    default=0.24,
    help="Extra vertical clearance used for high approach and transport waypoints.",
)
parser.add_argument(
    "--grasp_clearance",
    type=float,
    default=0.12,
    help="Vertical clearance above the final physical grasp pose for pre-grasp and retreat.",
)
parser.add_argument("--marker_scale", type=float, default=0.08, help="Scale of gripper frame markers.")
parser.add_argument("--disable_markers", action="store_true", help="Disable gripper frame marker visualization.")
parser.add_argument(
    "--grasp_orientation",
    choices=("current", "top_down"),
    default="current",
    help="Use the current gripper orientation by default, or force a top-down gripper frame.",
)
parser.add_argument(
    "--finger_center_offset_z",
    type=float,
    default=0.0,
    help="Additional z bias after aligning the physical finger center with the block center.",
)
parser.add_argument(
    "--finger_table_clearance",
    type=float,
    default=0.015,
    help="Minimum clearance between the lower finger collision face and the tabletop during grasp/place.",
)
parser.add_argument(
    "--finger_collision_half_height",
    type=float,
    default=0.02,
    help="Half height of the gripper finger collision boxes used for tabletop clearance planning.",
)
parser.add_argument(
    "--attach_offset_z",
    type=float,
    default=-0.05,
    help="Kinematic mode only: block z offset relative to the selected end-effector while attached.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

if args_cli.num_envs != 1:
    raise ValueError("The scripted BlocksStackEasy auto-grasp demo currently supports only --num_envs 1.")
if args_cli.phase_steps <= 0:
    raise ValueError("--phase_steps must be positive.")
for name in (
    "torso_stiffness",
    "torso_damping",
    "torso_effort",
    "torso_armature",
    "arm_stiffness",
    "arm_damping",
    "arm_effort",
    "arm_armature",
    "gripper_stiffness",
    "gripper_damping",
    "gripper_effort",
    "gripper_armature",
):
    value = getattr(args_cli, name)
    if value is not None and value <= 0.0:
        raise ValueError(f"--{name} must be positive.")
if args_cli.solver_position_iterations is not None and args_cli.solver_position_iterations <= 0:
    raise ValueError("--solver_position_iterations must be positive.")
if args_cli.solver_velocity_iterations is not None and args_cli.solver_velocity_iterations <= 0:
    raise ValueError("--solver_velocity_iterations must be positive.")
if args_cli.settle_steps <= 0:
    raise ValueError("--settle_steps must be positive.")
if args_cli.close_steps <= 0:
    raise ValueError("--close_steps must be positive.")
if args_cli.lift_height <= 0.0:
    raise ValueError("--lift_height must be positive.")
if args_cli.overhead_clearance <= 0.0:
    raise ValueError("--overhead_clearance must be positive.")
if args_cli.grasp_clearance <= 0.0:
    raise ValueError("--grasp_clearance must be positive.")
if args_cli.marker_scale <= 0.0:
    raise ValueError("--marker_scale must be positive.")
if args_cli.finger_table_clearance < 0.0:
    raise ValueError("--finger_table_clearance must be non-negative.")
if args_cli.finger_collision_half_height <= 0.0:
    raise ValueError("--finger_collision_half_height must be positive.")

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch

import isaaclab_tasks  # noqa: F401
from isaaclab.markers import VisualizationMarkers
from isaaclab.markers.config import CUBOID_MARKER_CFG, FRAME_MARKER_CFG, SPHERE_MARKER_CFG
from isaaclab.utils.math import (
    combine_frame_transforms,
    quat_apply,
    quat_apply_inverse,
    subtract_frame_transforms,
)
from isaaclab_tasks.utils import parse_env_cfg

import assembly_benchmark.tasks  # noqa: F401
from assembly_benchmark.assets.furniture.lab_table import LAB_TABLE_SURFACE_Z


OPEN_GRIPPER = 1.0
CLOSE_GRIPPER = -1.0
TOP_DOWN_GRIPPER_QUAT = (1.0, 0.0, 0.0, 0.0)
FINGER_COLLISION_BOX_SIZE = (0.06, 0.01, 0.04)
FINGER_COLLISION_OFFSETS = {
    "left_gripper_finger_link1": (0.0, -0.01, 0.0),
    "left_gripper_finger_link2": (0.0, 0.01, 0.0),
    "right_gripper_finger_link1": (0.0, -0.01, 0.0),
    "right_gripper_finger_link2": (0.0, 0.01, 0.0),
}
_MARKERS: tuple[
    VisualizationMarkers,
    VisualizationMarkers,
    VisualizationMarkers,
    VisualizationMarkers,
    VisualizationMarkers,
    VisualizationMarkers,
] | None = None
_PLANNED_GRASP_CENTER_POSE_B: torch.Tensor | None = None


def _override_actuator(actuator_cfg, **kwargs) -> None:
    for key, value in kwargs.items():
        if value is not None:
            setattr(actuator_cfg, key, value)


def _configure_robot_for_run(env_cfg) -> None:
    robot_cfg = env_cfg.scene.robot
    if args_cli.enable_robot_gravity:
        robot_cfg.spawn.rigid_props.disable_gravity = False
    if args_cli.solver_position_iterations is not None:
        robot_cfg.spawn.articulation_props.solver_position_iteration_count = args_cli.solver_position_iterations
    if args_cli.solver_velocity_iterations is not None:
        robot_cfg.spawn.articulation_props.solver_velocity_iteration_count = args_cli.solver_velocity_iterations
    _override_actuator(
        robot_cfg.actuators["torso"],
        stiffness=args_cli.torso_stiffness,
        damping=args_cli.torso_damping,
        effort_limit_sim=args_cli.torso_effort,
        armature=args_cli.torso_armature,
    )
    for actuator_name in ("left_arm", "right_arm"):
        _override_actuator(
            robot_cfg.actuators[actuator_name],
            stiffness=args_cli.arm_stiffness,
            damping=args_cli.arm_damping,
            effort_limit_sim=args_cli.arm_effort,
            armature=args_cli.arm_armature,
        )
    for actuator_name in ("left_gripper", "right_gripper"):
        _override_actuator(
            robot_cfg.actuators[actuator_name],
            stiffness=args_cli.gripper_stiffness,
            damping=args_cli.gripper_damping,
            effort_limit_sim=args_cli.gripper_effort,
            armature=args_cli.gripper_armature,
        )


def _make_action(
    left_pose: torch.Tensor, left_grip: float, right_pose: torch.Tensor, right_grip: float
) -> torch.Tensor:
    """Build one bimanual IK action."""
    actions = torch.zeros((1, 16), dtype=torch.float32, device=left_pose.device)
    actions[:, 0:7] = left_pose
    actions[:, 7] = left_grip
    actions[:, 8:15] = right_pose
    actions[:, 15] = right_grip
    return actions


def _step_without_auto_reset(env, actions: torch.Tensor):
    """Step the DirectRLEnv physics path without applying its automatic reset."""
    unwrapped = env.unwrapped
    actions = actions.to(unwrapped.device)
    unwrapped._pre_physics_step(actions)
    is_rendering = unwrapped.sim.has_gui() or unwrapped.sim.has_rtx_sensors()

    for _ in range(unwrapped.cfg.decimation):
        unwrapped._sim_step_counter += 1
        unwrapped._apply_action()
        unwrapped.scene.write_data_to_sim()
        unwrapped.sim.step(render=False)
        if unwrapped._sim_step_counter % unwrapped.cfg.sim.render_interval == 0 and is_rendering:
            unwrapped.sim.render()
        unwrapped.scene.update(dt=unwrapped.physics_dt)

    unwrapped.obs_buf = unwrapped._get_observations()
    reward = unwrapped._get_rewards()
    terminated, truncated = unwrapped._get_dones()
    return unwrapped.obs_buf, reward, terminated, truncated, unwrapped.extras


def _write_block_pose_env(unwrapped, block, pose_env: torch.Tensor) -> None:
    """Write a block pose expressed in the env frame into simulation coordinates."""
    if args_cli.grasp_mode == "physical":
        raise RuntimeError("Physical grasp mode forbids direct block pose writes after reset.")
    env_ids = torch.tensor([0], dtype=torch.long, device=unwrapped.device)
    pose_w = pose_env.clone()
    pose_w[:, :3] += unwrapped.scene.env_origins[env_ids]
    block.write_root_pose_to_sim(pose_w, env_ids=env_ids)
    block.write_root_velocity_to_sim(torch.zeros((1, 6), dtype=torch.float32, device=unwrapped.device), env_ids=env_ids)


def _carry_block_at_ee(unwrapped, block, active_arm: str, block_quat: torch.Tensor, attach_offset_z: float) -> None:
    """Keep a carried block attached under the selected end-effector."""
    left_ee_pose, right_ee_pose = unwrapped._get_ee_poses_in_root_frame()
    ee_pose = left_ee_pose if active_arm == "left" else right_ee_pose
    carried_pose = torch.zeros((1, 7), dtype=torch.float32, device=unwrapped.device)
    carried_pose[:, :3] = ee_pose[:, :3]
    carried_pose[:, 2] += attach_offset_z
    carried_pose[:, 3:7] = block_quat
    _write_block_pose_env(unwrapped, block, carried_pose)


def _pose_from_pos_quat(pos: torch.Tensor, quat: torch.Tensor) -> torch.Tensor:
    pose = torch.zeros((1, 7), dtype=torch.float32, device=pos.device)
    pose[:, :3] = pos
    pose[:, 3:7] = quat
    return pose


def _top_down_gripper_quat(device: str) -> torch.Tensor:
    return torch.tensor(TOP_DOWN_GRIPPER_QUAT, dtype=torch.float32, device=device).unsqueeze(0)


def _active_gripper_quat(left_pose: torch.Tensor, right_pose: torch.Tensor, active_arm: str, device: str) -> torch.Tensor:
    if args_cli.grasp_orientation == "top_down":
        return _top_down_gripper_quat(device)
    return left_pose[:, 3:7].clone() if active_arm == "left" else right_pose[:, 3:7].clone()


def _format_pose(pose: torch.Tensor) -> str:
    pose_cpu = pose[0].detach().cpu()
    pos = ", ".join(f"{value:.4f}" for value in pose_cpu[:3])
    quat = ", ".join(f"{value:.4f}" for value in pose_cpu[3:7])
    return f"pos=[{pos}] quat_wxyz=[{quat}]"


def _format_vec(vec: torch.Tensor) -> str:
    vec_cpu = vec[0].detach().cpu()
    values = ", ".join(f"{value:.4f}" for value in vec_cpu)
    return f"[{values}]"


def _max_controlled_joint_target_error(unwrapped) -> torch.Tensor:
    joint_pos = unwrapped.robot.data.joint_pos[:, unwrapped.controlled_joint_ids]
    return torch.max(torch.abs(joint_pos - unwrapped.joint_targets))


def _block_pose_env(unwrapped, block) -> torch.Tensor:
    return unwrapped._object_pose_in_env_frame(block)[0:1]


def _table_surface_z(unwrapped) -> float:
    return float(getattr(unwrapped.cfg, "table_surface_z", LAB_TABLE_SURFACE_Z))


def _table_safe_finger_center_z(unwrapped, desired_z: torch.Tensor) -> torch.Tensor:
    min_center_z = (
        _table_surface_z(unwrapped)
        + args_cli.finger_collision_half_height
        + args_cli.finger_table_clearance
    )
    min_center_z_tensor = torch.full_like(desired_z, min_center_z)
    return torch.maximum(desired_z, min_center_z_tensor)


def _body_position_in_root_frame(unwrapped, body_name: str) -> torch.Tensor:
    body_idx = unwrapped.robot.find_bodies(body_name)[0][0]
    body_pose_w = unwrapped.robot.data.body_pose_w[:, body_idx]
    root_pose_w = unwrapped.robot.data.root_pose_w
    body_pos_b, _ = subtract_frame_transforms(
        root_pose_w[:, :3],
        root_pose_w[:, 3:7],
        body_pose_w[:, :3],
        body_pose_w[:, 3:7],
    )
    return body_pos_b


def _finger_center_in_root_frame(unwrapped, active_arm: str) -> torch.Tensor:
    finger_1_pos = _body_position_in_root_frame(unwrapped, f"{active_arm}_gripper_finger_link1")
    finger_2_pos = _body_position_in_root_frame(unwrapped, f"{active_arm}_gripper_finger_link2")
    return 0.5 * (finger_1_pos + finger_2_pos)


def _finger_center_pose_in_root_frame(unwrapped, active_arm: str) -> torch.Tensor:
    left_ee_pose, right_ee_pose = unwrapped._get_ee_poses_in_root_frame()
    ee_pose = left_ee_pose if active_arm == "left" else right_ee_pose
    finger_center_pose = ee_pose.clone()
    finger_center_pose[:, :3] = _finger_center_in_root_frame(unwrapped, active_arm)
    return finger_center_pose


def _finger_center_pose_in_world_frame(unwrapped, active_arm: str) -> torch.Tensor:
    pose_b = _finger_center_pose_in_root_frame(unwrapped, active_arm)
    root_pose_w = unwrapped.robot.data.root_pose_w
    pos_w, quat_w = combine_frame_transforms(
        root_pose_w[:, :3],
        root_pose_w[:, 3:7],
        pose_b[:, :3],
        pose_b[:, 3:7],
    )
    return torch.cat((pos_w, quat_w), dim=-1)


def _finger_collision_box_poses_in_world_frame(unwrapped) -> tuple[torch.Tensor, torch.Tensor]:
    box_positions = []
    box_quats = []
    identity_quat = torch.tensor((1.0, 0.0, 0.0, 0.0), dtype=torch.float32, device=unwrapped.device).unsqueeze(0)

    for body_name, offset in FINGER_COLLISION_OFFSETS.items():
        body_idx = unwrapped.robot.find_bodies(body_name)[0][0]
        body_pose_w = unwrapped.robot.data.body_pose_w[:, body_idx]
        offset_pos = torch.tensor(offset, dtype=torch.float32, device=unwrapped.device).unsqueeze(0)
        box_pos_w, box_quat_w = combine_frame_transforms(
            body_pose_w[:, :3],
            body_pose_w[:, 3:7],
            offset_pos,
            identity_quat,
        )
        box_positions.append(box_pos_w)
        box_quats.append(box_quat_w)

    return torch.cat(box_positions, dim=0), torch.cat(box_quats, dim=0)


def _finger_center_offset_from_ee_for_quat(unwrapped, active_arm: str, target_quat_b: torch.Tensor) -> torch.Tensor:
    left_ee_pose, right_ee_pose = unwrapped._get_ee_poses_in_root_frame()
    ee_pose = left_ee_pose if active_arm == "left" else right_ee_pose
    offset_b = _finger_center_in_root_frame(unwrapped, active_arm) - ee_pose[:, :3]
    offset_ee = quat_apply_inverse(ee_pose[:, 3:7], offset_b)
    return quat_apply(target_quat_b, offset_ee)


def _planned_grasp_center_pose_in_world_frame(unwrapped) -> torch.Tensor | None:
    if _PLANNED_GRASP_CENTER_POSE_B is None:
        return None
    root_pose_w = unwrapped.robot.data.root_pose_w
    pos_w, quat_w = combine_frame_transforms(
        root_pose_w[:, :3],
        root_pose_w[:, 3:7],
        _PLANNED_GRASP_CENTER_POSE_B[:, :3],
        _PLANNED_GRASP_CENTER_POSE_B[:, 3:7],
    )
    return torch.cat((pos_w, quat_w), dim=-1)


def _set_planned_grasp_center_pose(pos_b: torch.Tensor, quat_b: torch.Tensor) -> None:
    global _PLANNED_GRASP_CENTER_POSE_B

    _PLANNED_GRASP_CENTER_POSE_B = _pose_from_pos_quat(pos_b, quat_b)


def _make_markers() -> tuple[
    VisualizationMarkers,
    VisualizationMarkers,
    VisualizationMarkers,
    VisualizationMarkers,
    VisualizationMarkers,
    VisualizationMarkers,
]:
    gripper_marker_cfg = FRAME_MARKER_CFG.copy()
    gripper_marker_cfg.markers["frame"].scale = (
        args_cli.marker_scale,
        args_cli.marker_scale,
        args_cli.marker_scale,
    )

    center_marker_cfg = FRAME_MARKER_CFG.copy()
    center_scale = args_cli.marker_scale * 0.7
    center_marker_cfg.markers["frame"].scale = (center_scale, center_scale, center_scale)

    planned_grasp_marker_cfg = SPHERE_MARKER_CFG.copy()
    planned_grasp_marker_cfg.markers["sphere"].radius = args_cli.marker_scale * 0.22
    planned_grasp_marker_cfg.markers["sphere"].visual_material.diffuse_color = (1.0, 0.85, 0.0)
    planned_grasp_marker = VisualizationMarkers(
        planned_grasp_marker_cfg.replace(prim_path="/Visuals/r1_pro_planned_grasp_center")
    )
    planned_grasp_marker.set_visibility(False)

    finger_collision_marker_cfg = CUBOID_MARKER_CFG.copy()
    finger_collision_marker_cfg.markers["cuboid"].size = FINGER_COLLISION_BOX_SIZE
    finger_collision_marker_cfg.markers["cuboid"].visual_material.diffuse_color = (0.0, 0.85, 1.0)
    finger_collision_marker_cfg.markers["cuboid"].visual_material.opacity = 0.35

    return (
        VisualizationMarkers(gripper_marker_cfg.replace(prim_path="/Visuals/r1_pro_left_gripper_frame")),
        VisualizationMarkers(gripper_marker_cfg.replace(prim_path="/Visuals/r1_pro_right_gripper_frame")),
        VisualizationMarkers(center_marker_cfg.replace(prim_path="/Visuals/r1_pro_left_finger_center_frame")),
        VisualizationMarkers(center_marker_cfg.replace(prim_path="/Visuals/r1_pro_right_finger_center_frame")),
        planned_grasp_marker,
        VisualizationMarkers(
            finger_collision_marker_cfg.replace(prim_path="/Visuals/r1_pro_finger_collision_boxes")
        ),
    )


def _visualize_markers(unwrapped) -> None:
    if _MARKERS is None:
        return

    (
        left_gripper_marker,
        right_gripper_marker,
        left_center_marker,
        right_center_marker,
        planned_grasp_marker,
        finger_collision_marker,
    ) = _MARKERS
    left_gripper_pose_w = unwrapped.robot.data.body_pose_w[:, unwrapped.left_ee_body_idx]
    right_gripper_pose_w = unwrapped.robot.data.body_pose_w[:, unwrapped.right_ee_body_idx]
    left_center_pose_w = _finger_center_pose_in_world_frame(unwrapped, "left")
    right_center_pose_w = _finger_center_pose_in_world_frame(unwrapped, "right")
    planned_grasp_pose_w = _planned_grasp_center_pose_in_world_frame(unwrapped)
    finger_collision_pos_w, finger_collision_quat_w = _finger_collision_box_poses_in_world_frame(unwrapped)

    left_gripper_marker.visualize(left_gripper_pose_w[:, :3], left_gripper_pose_w[:, 3:7])
    right_gripper_marker.visualize(right_gripper_pose_w[:, :3], right_gripper_pose_w[:, 3:7])
    left_center_marker.visualize(left_center_pose_w[:, :3], left_center_pose_w[:, 3:7])
    right_center_marker.visualize(right_center_pose_w[:, :3], right_center_pose_w[:, 3:7])
    finger_collision_marker.visualize(finger_collision_pos_w, finger_collision_quat_w)
    if planned_grasp_pose_w is None:
        planned_grasp_marker.set_visibility(False)
    else:
        planned_grasp_marker.set_visibility(True)
        planned_grasp_marker.visualize(planned_grasp_pose_w[:, :3])


def _run_phase(
    env,
    phase: str,
    block_label: str,
    block,
    block_target_pos: torch.Tensor,
    active_arm: str,
    target_pose: torch.Tensor,
    left_pose: torch.Tensor,
    right_pose: torch.Tensor,
    left_grip: float,
    right_grip: float,
    target_left_grip: float,
    target_right_grip: float,
    carried: bool,
    block_quat: torch.Tensor,
    global_step: int,
    steps: int | None = None,
    block_start_z: float | None = None,
    target_pose_fn: Callable[[], torch.Tensor] | None = None,
) -> tuple[torch.Tensor, torch.Tensor, float, float, int]:
    """Run one interpolated scripted phase."""
    unwrapped = env.unwrapped
    start_pose = left_pose.clone() if active_arm == "left" else right_pose.clone()
    start_left_grip = left_grip
    start_right_grip = right_grip
    phase_steps = args_cli.phase_steps if steps is None else steps

    for step_idx in range(phase_steps):
        if not simulation_app.is_running():
            break

        resolved_target_pose = target_pose_fn() if target_pose_fn is not None else target_pose
        alpha = float(step_idx + 1) / float(phase_steps)
        command_pose = start_pose.clone()
        command_pose[:, :3] = (1.0 - alpha) * start_pose[:, :3] + alpha * resolved_target_pose[:, :3]
        command_pose[:, 3:7] = resolved_target_pose[:, 3:7]
        if active_arm == "left":
            left_pose = command_pose
        else:
            right_pose = command_pose

        left_grip = (1.0 - alpha) * start_left_grip + alpha * target_left_grip
        right_grip = (1.0 - alpha) * start_right_grip + alpha * target_right_grip
        actions = _make_action(left_pose, left_grip, right_pose, right_grip)
        _step_without_auto_reset(env, actions)
        _visualize_markers(unwrapped)

        if carried:
            _carry_block_at_ee(unwrapped, block, active_arm, block_quat, args_cli.attach_offset_z)
            _visualize_markers(unwrapped)

        if args_cli.print_interval > 0 and global_step % args_cli.print_interval == 0:
            left_ee_pose, right_ee_pose = unwrapped._get_ee_poses_in_root_frame()
            ee_pose = left_ee_pose if active_arm == "left" else right_ee_pose
            ee_error = torch.linalg.norm(ee_pose[:, :3] - resolved_target_pose[:, :3], dim=-1).mean()
            block_pos = _block_pose_env(unwrapped, block)[:, :3]
            block_error_xyz = block_pos - block_target_pos
            block_error = torch.linalg.norm(block_error_xyz, dim=-1).mean()
            block_to_ee = torch.linalg.norm(block_pos - ee_pose[:, :3], dim=-1).mean()
            finger_center = _finger_center_in_root_frame(unwrapped, active_arm)
            finger_center_pose = _finger_center_pose_in_root_frame(unwrapped, active_arm)
            block_to_finger = torch.linalg.norm(block_pos - finger_center, dim=-1).mean()
            max_joint_error = _max_controlled_joint_target_error(unwrapped)
            finger_table_clearance = (
                finger_center[:, 2] - args_cli.finger_collision_half_height - _table_surface_z(unwrapped)
            ).mean()
            block_lift = float(block_pos[0, 2].item() - block_start_z) if block_start_z is not None else 0.0
            success = bool(unwrapped._success()[0].item())
            print(
                f"[INFO]: phase={phase} block={block_label} arm={active_arm} "
                f"ee_error={float(ee_error):.4f} m ee_block_dist={float(block_to_ee):.4f} m "
                f"finger_block_dist={float(block_to_finger):.4f} m "
                f"finger_table_clearance={float(finger_table_clearance):.4f} m "
                f"block_z={float(block_pos[0, 2]):.4f} m block_lift={block_lift:.4f} m "
                f"block_target_error={float(block_error):.4f} m "
                f"block_error_xyz={_format_vec(block_error_xyz)} "
                f"target_b={_format_pose(resolved_target_pose)} "
                f"block_b={_format_vec(block_pos)} "
                f"max_joint_target_error={float(max_joint_error):.4f} rad "
                f"finger_center_b={_format_pose(finger_center_pose)} "
                f"success={success}"
            )

        global_step += 1

    return left_pose, right_pose, left_grip, right_grip, global_step


def _move_block_kinematic(
    env,
    block_label: str,
    block,
    block_index: int,
    left_pose: torch.Tensor,
    right_pose: torch.Tensor,
    left_grip: float,
    right_grip: float,
    global_step: int,
) -> tuple[torch.Tensor, torch.Tensor, float, float, torch.Tensor, int]:
    """Move one block to its stack target with the legacy kinematic attachment path."""
    unwrapped = env.unwrapped
    block_pose = _block_pose_env(unwrapped, block)
    block_quat = block_pose[:, 3:7].clone()
    target_pos = unwrapped.block_target_positions[block_index].unsqueeze(0)
    target_pose = _pose_from_pos_quat(target_pos, block_quat)
    active_arm = "right" if float(block_pose[0, 1]) < 0.0 else "left"
    active_quat = left_pose[:, 3:7].clone() if active_arm == "left" else right_pose[:, 3:7].clone()

    ee_grasp_z = -args_cli.attach_offset_z
    grasp_pos = block_pose[:, :3].clone()
    grasp_pos[:, 2] += ee_grasp_z
    pre_grasp_pos = grasp_pos.clone()
    pre_grasp_pos[:, 2] += 0.15
    pre_place_pos = target_pos.clone()
    pre_place_pos[:, 2] += ee_grasp_z + 0.15
    place_pos = target_pos.clone()
    place_pos[:, 2] += ee_grasp_z

    pre_grasp_pose = _pose_from_pos_quat(pre_grasp_pos, active_quat)
    grasp_pose = _pose_from_pos_quat(grasp_pos, active_quat)
    pre_place_pose = _pose_from_pos_quat(pre_place_pos, active_quat)
    place_pose = _pose_from_pos_quat(place_pos, active_quat)

    if active_arm == "left":
        left_grip = OPEN_GRIPPER
    else:
        right_grip = OPEN_GRIPPER

    block_start_z = float(block_pose[0, 2].item())
    phases = (
        ("pre-grasp", pre_grasp_pose, False, OPEN_GRIPPER, args_cli.phase_steps),
        ("descend", grasp_pose, False, OPEN_GRIPPER, args_cli.phase_steps),
        ("close", grasp_pose, False, CLOSE_GRIPPER, args_cli.close_steps),
        ("attach/lift", pre_grasp_pose, True, CLOSE_GRIPPER, args_cli.phase_steps),
        ("pre-place", pre_place_pose, True, CLOSE_GRIPPER, args_cli.phase_steps),
        ("place-descend", place_pose, True, CLOSE_GRIPPER, args_cli.phase_steps),
        ("open/detach", place_pose, True, OPEN_GRIPPER, args_cli.close_steps),
    )

    for phase, phase_target_pose, carried, gripper_value, steps in phases:
        if active_arm == "left":
            target_left_grip = gripper_value
            target_right_grip = right_grip
        else:
            target_left_grip = left_grip
            target_right_grip = gripper_value
        left_pose, right_pose, left_grip, right_grip, global_step = _run_phase(
            env=env,
            phase=phase,
            block_label=block_label,
            block=block,
            block_target_pos=target_pos,
            active_arm=active_arm,
            target_pose=phase_target_pose,
            left_pose=left_pose,
            right_pose=right_pose,
            left_grip=left_grip,
            right_grip=right_grip,
            target_left_grip=target_left_grip,
            target_right_grip=target_right_grip,
            carried=carried,
            block_quat=block_quat,
            global_step=global_step,
            steps=steps,
            block_start_z=block_start_z,
        )

    _write_block_pose_env(unwrapped, block, target_pose)

    if active_arm == "left":
        target_left_grip = OPEN_GRIPPER
        target_right_grip = right_grip
    else:
        target_left_grip = left_grip
        target_right_grip = OPEN_GRIPPER
    left_pose, right_pose, left_grip, right_grip, global_step = _run_phase(
        env=env,
        phase="lift-away",
        block_label=block_label,
        block=block,
        block_target_pos=target_pos,
        active_arm=active_arm,
        target_pose=pre_place_pose,
        left_pose=left_pose,
        right_pose=right_pose,
        left_grip=left_grip,
        right_grip=right_grip,
        target_left_grip=target_left_grip,
        target_right_grip=target_right_grip,
        carried=False,
        block_quat=block_quat,
        global_step=global_step,
        block_start_z=block_start_z,
    )

    _write_block_pose_env(unwrapped, block, target_pose)
    return left_pose, right_pose, left_grip, right_grip, target_pose, global_step


def _move_block_physical(
    env,
    block_label: str,
    block,
    block_index: int,
    left_pose: torch.Tensor,
    right_pose: torch.Tensor,
    left_grip: float,
    right_grip: float,
    global_step: int,
) -> tuple[torch.Tensor, torch.Tensor, float, float, torch.Tensor, int]:
    """Move one block using real gripper-object contact only."""
    unwrapped = env.unwrapped
    block_pose = _block_pose_env(unwrapped, block)
    block_quat = block_pose[:, 3:7].clone()
    target_pos = unwrapped.block_target_positions[block_index].unsqueeze(0)
    target_pose = _pose_from_pos_quat(target_pos, block_quat)
    active_arm = "right" if float(block_pose[0, 1]) < 0.0 else "left"
    active_quat = _active_gripper_quat(left_pose, right_pose, active_arm, unwrapped.device)
    finger_center_offset = _finger_center_offset_from_ee_for_quat(unwrapped, active_arm, active_quat)
    block_start_z = float(block_pose[0, 2].item())
    current_pose = left_pose.clone() if active_arm == "left" else right_pose.clone()
    approach_clearance = max(args_cli.grasp_clearance, args_cli.overhead_clearance)
    transport_clearance = max(args_cli.lift_height, args_cli.overhead_clearance)

    def _current_grasp_pose(clearance: float = 0.0) -> torch.Tensor:
        current_block_pose = _block_pose_env(unwrapped, block)
        grasp_center_pos = current_block_pose[:, :3].clone()
        grasp_center_pos[:, 2] += args_cli.finger_center_offset_z
        grasp_center_pos[:, 2] = _table_safe_finger_center_z(unwrapped, grasp_center_pos[:, 2])
        _set_planned_grasp_center_pose(grasp_center_pos, active_quat)

        grasp_pos = grasp_center_pos - finger_center_offset
        grasp_pos[:, 2] += clearance
        return _pose_from_pos_quat(grasp_pos, active_quat)

    pre_grasp_pose = _current_grasp_pose(approach_clearance)
    _visualize_markers(unwrapped)
    raise_pos = current_pose[:, :3].clone()
    raise_pos[:, 2] = torch.maximum(
        raise_pos[:, 2],
        torch.tensor(pre_grasp_pose[0, 2].item(), dtype=torch.float32, device=unwrapped.device),
    )

    raise_pose = _pose_from_pos_quat(raise_pos, active_quat)

    print(
        f"[INFO]: physical grasp setup block={block_label} arm={active_arm} "
        f"grasp_orientation={args_cli.grasp_orientation} "
        f"approach_clearance={approach_clearance:.3f} m transport_clearance={transport_clearance:.3f} m "
        f"finger_table_clearance={args_cli.finger_table_clearance:.3f} m "
        f"planned_grasp_center_b={_format_pose(_PLANNED_GRASP_CENTER_POSE_B)} "
        f"initial_finger_center_b={_format_pose(_finger_center_pose_in_root_frame(unwrapped, active_arm))}"
    )

    grasp_phases: tuple[tuple[str, torch.Tensor, float, int, Callable[[], torch.Tensor] | None], ...] = (
        ("raise-arm", raise_pose, OPEN_GRIPPER, args_cli.phase_steps, None),
        (
            "move-above-grasp",
            pre_grasp_pose,
            OPEN_GRIPPER,
            args_cli.phase_steps,
            lambda: _current_grasp_pose(approach_clearance),
        ),
        (
            "descend-to-grasp",
            _current_grasp_pose(0.0),
            OPEN_GRIPPER,
            args_cli.phase_steps,
            lambda: _current_grasp_pose(0.0),
        ),
        (
            "settle",
            _current_grasp_pose(0.0),
            OPEN_GRIPPER,
            args_cli.settle_steps,
            lambda: _current_grasp_pose(0.0),
        ),
        (
            "close-slow",
            _current_grasp_pose(0.0),
            CLOSE_GRIPPER,
            args_cli.close_steps,
            lambda: _current_grasp_pose(0.0),
        ),
    )

    for phase, phase_target_pose, gripper_value, steps, target_pose_fn in grasp_phases:
        if active_arm == "left":
            target_left_grip = gripper_value
            target_right_grip = right_grip
        else:
            target_left_grip = left_grip
            target_right_grip = gripper_value
        left_pose, right_pose, left_grip, right_grip, global_step = _run_phase(
            env=env,
            phase=phase,
            block_label=block_label,
            block=block,
            block_target_pos=target_pos,
            active_arm=active_arm,
            target_pose=phase_target_pose,
            left_pose=left_pose,
            right_pose=right_pose,
            left_grip=left_grip,
            right_grip=right_grip,
            target_left_grip=target_left_grip,
            target_right_grip=target_right_grip,
            carried=False,
            block_quat=block_quat,
            global_step=global_step,
            steps=steps,
            block_start_z=block_start_z,
            target_pose_fn=target_pose_fn,
        )

    block_start_z = float(_block_pose_env(unwrapped, block)[0, 2].item())
    held_block_pos = _block_pose_env(unwrapped, block)[:, :3]
    held_finger_center_pos = _finger_center_in_root_frame(unwrapped, active_arm)
    block_offset_from_finger_center = held_block_pos - held_finger_center_pos
    place_center_pos = target_pos - block_offset_from_finger_center
    place_center_pos[:, 2] = _table_safe_finger_center_z(unwrapped, place_center_pos[:, 2])
    place_pos = place_center_pos - finger_center_offset
    place_above_pos = place_pos.clone()
    place_above_pos[:, 2] += transport_clearance
    place_pose = _pose_from_pos_quat(place_pos, active_quat)
    place_above_pose = _pose_from_pos_quat(place_above_pos, active_quat)
    grasp_pose = left_pose.clone() if active_arm == "left" else right_pose.clone()
    lift_pos = grasp_pose[:, :3].clone()
    lift_pos[:, 2] += transport_clearance
    lift_pose = _pose_from_pos_quat(lift_pos, active_quat)
    transport_phases = (
        ("lift-test", lift_pose, CLOSE_GRIPPER, args_cli.phase_steps),
        ("high-transport", place_above_pose, CLOSE_GRIPPER, args_cli.phase_steps),
        ("descend-to-place", place_pose, CLOSE_GRIPPER, args_cli.phase_steps),
        ("open", place_pose, OPEN_GRIPPER, args_cli.close_steps),
        ("retreat-up", place_above_pose, OPEN_GRIPPER, args_cli.phase_steps),
    )

    for phase, phase_target_pose, gripper_value, steps in transport_phases:
        if active_arm == "left":
            target_left_grip = gripper_value
            target_right_grip = right_grip
        else:
            target_left_grip = left_grip
            target_right_grip = gripper_value
        left_pose, right_pose, left_grip, right_grip, global_step = _run_phase(
            env=env,
            phase=phase,
            block_label=block_label,
            block=block,
            block_target_pos=target_pos,
            active_arm=active_arm,
            target_pose=phase_target_pose,
            left_pose=left_pose,
            right_pose=right_pose,
            left_grip=left_grip,
            right_grip=right_grip,
            target_left_grip=target_left_grip,
            target_right_grip=target_right_grip,
            carried=False,
            block_quat=block_quat,
            global_step=global_step,
            steps=steps,
            block_start_z=block_start_z,
        )

        block_pos = _block_pose_env(unwrapped, block)[:, :3]
        if phase == "lift-test":
            lift_delta = float(block_pos[0, 2].item() - block_start_z)
            min_lift = min(0.04, 0.25 * args_cli.lift_height)
            print(
                f"[INFO]: lift-test checkpoint block={block_label} "
                f"lift_delta={lift_delta:.4f} m min_lift={min_lift:.4f} m"
            )
            if lift_delta < min_lift:
                table_clearance = float(block_pos[0, 2].item() - _table_surface_z(unwrapped))
                failure_reason = (
                    f"Physical grasp failed during lift-test for {block_label}: "
                    f"block_lift={lift_delta:.4f} m, table_clearance={table_clearance:.4f} m. "
                    "The gripper did not physically hold the block."
                )
                print(f"[ERROR]: {failure_reason}", flush=True)
                setattr(unwrapped, "_scripted_auto_grasp_failure", failure_reason)
                return left_pose, right_pose, left_grip, right_grip, target_pose, global_step

    final_block_pos = _block_pose_env(unwrapped, block)[:, :3]
    final_error_xyz = final_block_pos - target_pos
    final_error = torch.linalg.norm(final_error_xyz, dim=-1).mean()
    success = bool(unwrapped._success()[0].item())
    print(
        f"[INFO]: physical block result block={block_label} "
        f"target_error={float(final_error):.4f} m "
        f"target_error_xyz={_format_vec(final_error_xyz)} "
        f"block_b={_format_vec(final_block_pos)} target_b={_format_vec(target_pos)} "
        f"final_finger_center_b={_format_pose(_finger_center_pose_in_root_frame(unwrapped, active_arm))} "
        f"success={success}"
    )
    return left_pose, right_pose, left_grip, right_grip, target_pose, global_step


def _move_block(
    env,
    block_label: str,
    block,
    block_index: int,
    left_pose: torch.Tensor,
    right_pose: torch.Tensor,
    left_grip: float,
    right_grip: float,
    global_step: int,
) -> tuple[torch.Tensor, torch.Tensor, float, float, torch.Tensor, int]:
    """Move one block with the selected grasp mode."""
    if args_cli.grasp_mode == "physical":
        return _move_block_physical(
            env, block_label, block, block_index, left_pose, right_pose, left_grip, right_grip, global_step
        )
    return _move_block_kinematic(
        env, block_label, block, block_index, left_pose, right_pose, left_grip, right_grip, global_step
    )


def _print_final_status(unwrapped) -> tuple[bool, float, bool, bool]:
    success = bool(unwrapped._success()[0].item())
    reward = float(unwrapped._get_rewards()[0].item())
    terminated, truncated = unwrapped._get_dones()
    terminated_value = bool(terminated[0].item())
    truncated_value = bool(truncated[0].item())
    print(
        "[INFO]: final "
        f"success={success} reward={reward:.1f} terminated={terminated_value} timeout={truncated_value}"
    )
    return success, reward, terminated_value, truncated_value


def main() -> int:
    global _MARKERS, _PLANNED_GRASP_CENTER_POSE_B

    env_cfg = parse_env_cfg(
        TASK_NAME,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
    )
    if args_cli.include_torso_in_ik:
        if not hasattr(env_cfg, "torso_joint_names") or not hasattr(env_cfg, "include_torso_in_ik"):
            raise RuntimeError(f"Task '{TASK_NAME}' does not expose R1 Pro torso IK configuration.")
        env_cfg.include_torso_in_ik = True
        env_cfg.observation_space += 2 * len(env_cfg.torso_joint_names)
    _configure_robot_for_run(env_cfg)
    env = gym.make(TASK_NAME, cfg=env_cfg)
    unwrapped = env.unwrapped

    try:
        print(f"[INFO]: Gym observation space: {env.observation_space}")
        print(f"[INFO]: Gym action space: {env.action_space}")
        print(f"[INFO]: Grasp mode: {args_cli.grasp_mode}")
        print(f"[INFO]: Grasp orientation: {args_cli.grasp_orientation}")
        print(f"[INFO]: Torso IK: enabled={args_cli.include_torso_in_ik}")
        print(f"[INFO]: Gripper frame markers: enabled={not args_cli.disable_markers}")
        print(
            "[INFO]: Robot dynamics: "
            f"gravity_enabled={args_cli.enable_robot_gravity} "
            f"torso_stiffness={unwrapped.cfg.scene.robot.actuators['torso'].stiffness} "
            f"torso_damping={unwrapped.cfg.scene.robot.actuators['torso'].damping} "
            f"torso_effort={unwrapped.cfg.scene.robot.actuators['torso'].effort_limit_sim} "
            f"arm_stiffness={unwrapped.cfg.scene.robot.actuators['left_arm'].stiffness} "
            f"arm_damping={unwrapped.cfg.scene.robot.actuators['left_arm'].damping} "
            f"arm_effort={unwrapped.cfg.scene.robot.actuators['left_arm'].effort_limit_sim} "
            f"gripper_stiffness={unwrapped.cfg.scene.robot.actuators['left_gripper'].stiffness} "
            f"gripper_damping={unwrapped.cfg.scene.robot.actuators['left_gripper'].damping} "
            f"gripper_effort={unwrapped.cfg.scene.robot.actuators['left_gripper'].effort_limit_sim}"
        )
        env.reset()
        _PLANNED_GRASP_CENTER_POSE_B = None
        _MARKERS = None if args_cli.disable_markers else _make_markers()
        _visualize_markers(unwrapped)

        with torch.inference_mode():
            left_pose, right_pose = unwrapped._get_ee_poses_in_root_frame()
            left_pose = left_pose.clone()
            right_pose = right_pose.clone()
            left_grip = OPEN_GRIPPER
            right_grip = OPEN_GRIPPER
            global_step = 0

            block1_final_pose = None
            block2_final_pose = None
            left_pose, right_pose, left_grip, right_grip, block1_final_pose, global_step = _move_block(
                env=env,
                block_label="block1",
                block=unwrapped.block1,
                block_index=0,
                left_pose=left_pose,
                right_pose=right_pose,
                left_grip=left_grip,
                right_grip=right_grip,
                global_step=global_step,
            )
            failure_reason = getattr(unwrapped, "_scripted_auto_grasp_failure", None)
            if failure_reason is not None:
                _print_final_status(unwrapped)
                print(f"[ERROR]: {failure_reason}")
                return 1

            left_pose, right_pose, left_grip, right_grip, block2_final_pose, global_step = _move_block(
                env=env,
                block_label="block2",
                block=unwrapped.block2,
                block_index=1,
                left_pose=left_pose,
                right_pose=right_pose,
                left_grip=left_grip,
                right_grip=right_grip,
                global_step=global_step,
            )
            failure_reason = getattr(unwrapped, "_scripted_auto_grasp_failure", None)
            if failure_reason is not None:
                _print_final_status(unwrapped)
                print(f"[ERROR]: {failure_reason}")
                return 1

            if args_cli.grasp_mode == "kinematic":
                _write_block_pose_env(unwrapped, unwrapped.block1, block1_final_pose)
                _write_block_pose_env(unwrapped, unwrapped.block2, block2_final_pose)
            success, reward, terminated_value, _ = _print_final_status(unwrapped)
            if not success or reward < 1.0 or not terminated_value:
                print(
                    f"[ERROR]: BlocksStackEasy {args_cli.grasp_mode} auto-grasp "
                    "did not reach the success condition."
                )
                return 1
            return 0
    finally:
        env.close()


if __name__ == "__main__":
    exit_code = 0
    try:
        exit_code = main()
    except Exception:
        exit_code = 1
        traceback.print_exc()
    finally:
        if exit_code == 0 and args_cli.fast_exit:
            sys.stdout.flush()
            sys.stderr.flush()
            os._exit(0)
        if exit_code == 0:
            simulation_app.close()
        else:
            print(f"[ERROR]: exiting with code {exit_code}", flush=True)
            sys.stdout.flush()
            sys.stderr.flush()
            os._exit(exit_code)
    sys.exit(exit_code)
