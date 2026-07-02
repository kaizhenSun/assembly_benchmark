# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Scripted physical assembly demo for the R1 Pro FurnitureBench one_leg task.

The script drives the normal whole-body IK action interface and keeps the RL
environment API unchanged. It does not kinematically attach or snap leg4 after
reset: the leg is grasped, lifted, transported, inserted, and released through
normal simulation dynamics.

.. code-block:: bash

    python scripts/tools/run_r1_pro_one_leg_scripted_assembly.py --num_envs 1 --device cuda:0

"""

from __future__ import annotations

import argparse
import os
import sys
import traceback
from collections.abc import Callable

from isaaclab.app import AppLauncher


TASK_NAME = "Assembly-R1Pro-OneLeg-WholeBodyIK-Direct-v0"


parser = argparse.ArgumentParser(description="Run scripted R1 Pro one_leg physical assembly.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments. Only 1 is supported.")
parser.add_argument(
    "--disable_fabric",
    action="store_true",
    default=False,
    help="Debug/compatibility option: disable Fabric and use USD I/O, which may desync GUI mesh updates.",
)
parser.add_argument("--phase_steps", type=int, default=120, help="Number of simulation steps for each scripted phase.")
parser.add_argument("--close_steps", type=int, default=160, help="Steps used to close or open the gripper.")
parser.add_argument("--settle_steps", type=int, default=30, help="Steps to pause at grasp or insertion poses.")
parser.add_argument("--lift_height", type=float, default=0.14, help="Lift-test height above the physical grasp pose.")
parser.add_argument(
    "--overhead_clearance",
    type=float,
    default=0.24,
    help="Vertical clearance used for high approach, lift, and transport waypoints.",
)
parser.add_argument(
    "--insert_clearance",
    type=float,
    default=0.04,
    help="Leg clearance above the final assembled pose before the final insertion push.",
)
parser.add_argument(
    "--insert_push_depth",
    type=float,
    default=0.015,
    help="Small downward insertion push after reaching the pre-insert pose.",
)
parser.add_argument(
    "--finger_center_offset_z",
    type=float,
    default=0.0,
    help="Additional z bias after aligning the physical finger center with the leg origin.",
)
parser.add_argument(
    "--finger_table_clearance",
    type=float,
    default=0.02,
    help="Minimum clearance between the lower finger collision face and the tabletop.",
)
parser.add_argument(
    "--finger_collision_half_height",
    type=float,
    default=0.02,
    help="Half height of the gripper finger collision boxes used for tabletop clearance planning.",
)
parser.add_argument(
    "--gripper_orientation",
    choices=("top_down", "current", "assembled"),
    default="top_down",
    help=(
        "Use a top-down grasp frame by default; current keeps the initial active gripper orientation, "
        "and assembled keeps the old assembled-leg orientation for debugging."
    ),
)
parser.add_argument("--marker_scale", type=float, default=0.06, help="Scale of part and gripper frame markers.")
parser.add_argument("--disable_markers", action="store_true", help="Disable part and gripper frame visualization.")
parser.add_argument(
    "--print_interval",
    type=int,
    default=30,
    help="Print phase diagnostics every N scripted steps. Use 0 to disable.",
)
parser.add_argument(
    "--fast_exit",
    action="store_true",
    help="Exit the process immediately after the scripted run, avoiding slow Kit shutdown.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

if args_cli.num_envs != 1:
    raise ValueError("The scripted one_leg assembly demo currently supports only --num_envs 1.")
if args_cli.phase_steps <= 0:
    raise ValueError("--phase_steps must be positive.")
if args_cli.close_steps <= 0:
    raise ValueError("--close_steps must be positive.")
if args_cli.settle_steps <= 0:
    raise ValueError("--settle_steps must be positive.")
if args_cli.lift_height <= 0.0:
    raise ValueError("--lift_height must be positive.")
if args_cli.overhead_clearance <= 0.0:
    raise ValueError("--overhead_clearance must be positive.")
if args_cli.insert_clearance < 0.0:
    raise ValueError("--insert_clearance must be non-negative.")
if args_cli.insert_push_depth < 0.0:
    raise ValueError("--insert_push_depth must be non-negative.")
if args_cli.finger_table_clearance < 0.0:
    raise ValueError("--finger_table_clearance must be non-negative.")
if args_cli.finger_collision_half_height <= 0.0:
    raise ValueError("--finger_collision_half_height must be positive.")
if args_cli.marker_scale <= 0.0:
    raise ValueError("--marker_scale must be positive.")
if args_cli.print_interval < 0:
    raise ValueError("--print_interval must be non-negative.")

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch

import isaaclab_tasks  # noqa: F401
from isaaclab.markers import VisualizationMarkers
from isaaclab.markers.config import FRAME_MARKER_CFG
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
LEG4_TARGET_POS_TOP = (-0.05625, 0.046875, -0.05625)
IDENTITY_QUAT = (1.0, 0.0, 0.0, 0.0)
TOP_DOWN_GRIPPER_QUAT = (1.0, 0.0, 0.0, 0.0)
_MARKERS: tuple[
    VisualizationMarkers,
    VisualizationMarkers,
    VisualizationMarkers,
    VisualizationMarkers,
] | None = None
_PLANNED_GRASP_POSE_B: torch.Tensor | None = None
_PLANNED_INSERT_POSE_B: torch.Tensor | None = None


def _make_action(
    left_pose: torch.Tensor, left_grip: float, right_pose: torch.Tensor, right_grip: float
) -> torch.Tensor:
    """Build one bimanual whole-body IK action."""
    actions = torch.zeros((1, 16), dtype=torch.float32, device=left_pose.device)
    actions[:, 0:7] = left_pose
    actions[:, 7] = left_grip
    actions[:, 8:15] = right_pose
    actions[:, 15] = right_grip
    return actions


def _step_without_auto_reset(env, actions: torch.Tensor):
    """Step the DirectRLEnv physics path without applying automatic reset."""
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


def _pose_from_pos_quat(pos: torch.Tensor, quat: torch.Tensor) -> torch.Tensor:
    pose = torch.zeros((1, 7), dtype=torch.float32, device=pos.device)
    pose[:, :3] = pos
    pose[:, 3:7] = quat
    return pose


def _quat_tensor(device: torch.device, values: tuple[float, float, float, float]) -> torch.Tensor:
    return torch.tensor((values,), dtype=torch.float32, device=device)


def _normalize_quat(quat: torch.Tensor) -> torch.Tensor:
    return quat / torch.linalg.norm(quat, dim=-1, keepdim=True).clamp_min(1.0e-8)


def _quat_mul(first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
    first_w, first_x, first_y, first_z = first.unbind(dim=-1)
    second_w, second_x, second_y, second_z = second.unbind(dim=-1)
    return torch.stack(
        (
            first_w * second_w - first_x * second_x - first_y * second_y - first_z * second_z,
            first_w * second_x + first_x * second_w + first_y * second_z - first_z * second_y,
            first_w * second_y - first_x * second_z + first_y * second_w + first_z * second_x,
            first_w * second_z + first_x * second_y - first_y * second_x + first_z * second_w,
        ),
        dim=-1,
    )


def _quat_inv(quat: torch.Tensor) -> torch.Tensor:
    conjugate = quat.clone()
    conjugate[..., 1:] *= -1.0
    norm_sq = torch.sum(quat * quat, dim=-1, keepdim=True).clamp_min(1.0e-8)
    return conjugate / norm_sq


def _nlerp_quat(start: torch.Tensor, target: torch.Tensor, alpha: float) -> torch.Tensor:
    start = _normalize_quat(start)
    target = _normalize_quat(target)
    target = torch.where(torch.sum(start * target, dim=-1, keepdim=True) < 0.0, -target, target)
    return _normalize_quat((1.0 - alpha) * start + alpha * target)


def _quat_angle_error(current: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    current = _normalize_quat(current)
    target = _normalize_quat(target)
    dot = torch.abs(torch.sum(current * target, dim=-1)).clamp(max=1.0)
    return 2.0 * torch.acos(dot)


def _env_ids(unwrapped) -> torch.Tensor:
    return torch.tensor([0], dtype=torch.long, device=unwrapped.device)


def _world_pose_to_env_pose(unwrapped, pose_w: torch.Tensor) -> torch.Tensor:
    env_pose = pose_w.clone()
    env_pose[:, :3] -= unwrapped.scene.env_origins[_env_ids(unwrapped)]
    return env_pose


def _leg_pose_env(unwrapped) -> torch.Tensor:
    return unwrapped._object_pose_in_env_frame(unwrapped.square_table_leg4)[0:1]


def _active_ee_pose(unwrapped, active_arm: str) -> torch.Tensor:
    left_pose, right_pose = unwrapped._get_ee_poses_in_root_frame()
    return left_pose.clone() if active_arm == "left" else right_pose.clone()


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


def _finger_center_offset_from_ee_for_quat(
    unwrapped, active_arm: str, target_quat_b: torch.Tensor
) -> torch.Tensor:
    ee_pose = _active_ee_pose(unwrapped, active_arm)
    offset_b = _finger_center_in_root_frame(unwrapped, active_arm) - ee_pose[:, :3]
    offset_ee = quat_apply_inverse(ee_pose[:, 3:7], offset_b)
    return quat_apply(target_quat_b, offset_ee)


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


def _make_markers() -> tuple[
    VisualizationMarkers,
    VisualizationMarkers,
    VisualizationMarkers,
    VisualizationMarkers,
]:
    part_marker_cfg = FRAME_MARKER_CFG.copy()
    part_marker_cfg.markers["frame"].scale = (
        args_cli.marker_scale,
        args_cli.marker_scale,
        args_cli.marker_scale,
    )

    gripper_marker_cfg = FRAME_MARKER_CFG.copy()
    gripper_scale = args_cli.marker_scale * 1.25
    gripper_marker_cfg.markers["frame"].scale = (gripper_scale, gripper_scale, gripper_scale)

    planned_marker_cfg = FRAME_MARKER_CFG.copy()
    planned_scale = args_cli.marker_scale * 1.6
    planned_marker_cfg.markers["frame"].scale = (planned_scale, planned_scale, planned_scale)

    planned_grasp_marker = VisualizationMarkers(
        planned_marker_cfg.replace(prim_path="/Visuals/one_leg_planned_grasp_frame")
    )
    planned_grasp_marker.set_visibility(False)
    planned_insert_marker = VisualizationMarkers(
        planned_marker_cfg.replace(prim_path="/Visuals/one_leg_planned_insert_frame")
    )
    planned_insert_marker.set_visibility(False)

    return (
        VisualizationMarkers(part_marker_cfg.replace(prim_path="/Visuals/one_leg_part_frames")),
        VisualizationMarkers(gripper_marker_cfg.replace(prim_path="/Visuals/one_leg_gripper_frames")),
        planned_grasp_marker,
        planned_insert_marker,
    )


def _part_frame_poses_w(unwrapped) -> torch.Tensor:
    part_poses = (
        unwrapped.square_table_top.data.root_pose_w[0:1],
        unwrapped.square_table_leg1.data.root_pose_w[0:1],
        unwrapped.square_table_leg2.data.root_pose_w[0:1],
        unwrapped.square_table_leg3.data.root_pose_w[0:1],
        unwrapped.square_table_leg4.data.root_pose_w[0:1],
    )
    return torch.cat(part_poses, dim=0)


def _gripper_frame_poses_w(unwrapped) -> torch.Tensor:
    left_gripper_pose_w = unwrapped.robot.data.body_pose_w[:, unwrapped.left_ee_body_idx]
    right_gripper_pose_w = unwrapped.robot.data.body_pose_w[:, unwrapped.right_ee_body_idx]
    return torch.cat((left_gripper_pose_w, right_gripper_pose_w), dim=0)


def _root_pose_to_world_pose(unwrapped, pose_b: torch.Tensor | None) -> torch.Tensor | None:
    if pose_b is None:
        return None
    root_pose_w = unwrapped.robot.data.root_pose_w
    pos_w, quat_w = combine_frame_transforms(
        root_pose_w[:, :3],
        root_pose_w[:, 3:7],
        pose_b[:, :3],
        pose_b[:, 3:7],
    )
    return torch.cat((pos_w, quat_w), dim=-1)


def _set_planned_grasp_pose(pose_b: torch.Tensor | None) -> None:
    global _PLANNED_GRASP_POSE_B

    _PLANNED_GRASP_POSE_B = None if pose_b is None else pose_b.clone()


def _set_planned_insert_pose(pose_b: torch.Tensor | None) -> None:
    global _PLANNED_INSERT_POSE_B

    _PLANNED_INSERT_POSE_B = None if pose_b is None else pose_b.clone()


def _visualize_markers(unwrapped) -> None:
    if _MARKERS is None:
        return

    part_marker, gripper_marker, planned_grasp_marker, planned_insert_marker = _MARKERS
    part_poses_w = _part_frame_poses_w(unwrapped)
    gripper_poses_w = _gripper_frame_poses_w(unwrapped)
    part_marker.visualize(part_poses_w[:, :3], part_poses_w[:, 3:7])
    gripper_marker.visualize(gripper_poses_w[:, :3], gripper_poses_w[:, 3:7])
    planned_grasp_pose_w = _root_pose_to_world_pose(unwrapped, _PLANNED_GRASP_POSE_B)
    planned_insert_pose_w = _root_pose_to_world_pose(unwrapped, _PLANNED_INSERT_POSE_B)
    if planned_grasp_pose_w is None:
        planned_grasp_marker.set_visibility(False)
    else:
        planned_grasp_marker.set_visibility(True)
        planned_grasp_marker.visualize(planned_grasp_pose_w[:, :3], planned_grasp_pose_w[:, 3:7])
    if planned_insert_pose_w is None:
        planned_insert_marker.set_visibility(False)
    else:
        planned_insert_marker.set_visibility(True)
        planned_insert_marker.visualize(planned_insert_pose_w[:, :3], planned_insert_pose_w[:, 3:7])


def _ee_pose_for_finger_center(
    unwrapped, active_arm: str, finger_center_pos: torch.Tensor, target_quat_b: torch.Tensor
) -> torch.Tensor:
    safe_center_pos = finger_center_pos.clone()
    safe_center_pos[:, 2] = _table_safe_finger_center_z(unwrapped, safe_center_pos[:, 2])
    finger_center_offset = _finger_center_offset_from_ee_for_quat(unwrapped, active_arm, target_quat_b)
    return _pose_from_pos_quat(safe_center_pos - finger_center_offset, target_quat_b)


def _finger_center_pose_in_root_frame(unwrapped, active_arm: str) -> torch.Tensor:
    pose = _active_ee_pose(unwrapped, active_arm)
    pose[:, :3] = _finger_center_in_root_frame(unwrapped, active_arm)
    return pose


def _held_leg_relative_to_finger(
    unwrapped, active_arm: str, leg_pose_b: torch.Tensor | None = None
) -> tuple[torch.Tensor, torch.Tensor]:
    finger_pose_b = _finger_center_pose_in_root_frame(unwrapped, active_arm)
    leg_pose_b = _leg_pose_env(unwrapped) if leg_pose_b is None else leg_pose_b
    return subtract_frame_transforms(
        finger_pose_b[:, :3],
        finger_pose_b[:, 3:7],
        leg_pose_b[:, :3],
        leg_pose_b[:, 3:7],
    )


def _ee_pose_for_held_leg_pose(
    unwrapped,
    active_arm: str,
    desired_leg_pose_b: torch.Tensor,
    leg_pos_in_finger: torch.Tensor,
    leg_quat_in_finger: torch.Tensor,
) -> torch.Tensor:
    desired_finger_quat = _normalize_quat(_quat_mul(desired_leg_pose_b[:, 3:7], _quat_inv(leg_quat_in_finger)))
    desired_finger_pos = desired_leg_pose_b[:, :3] - quat_apply(desired_finger_quat, leg_pos_in_finger)
    return _ee_pose_for_finger_center(unwrapped, active_arm, desired_finger_pos, desired_finger_quat)


def _grasp_quat(unwrapped, active_pose: torch.Tensor, final_leg_pose_env: torch.Tensor) -> torch.Tensor:
    if args_cli.gripper_orientation == "top_down":
        return _quat_tensor(unwrapped.device, TOP_DOWN_GRIPPER_QUAT)
    if args_cli.gripper_orientation == "assembled":
        return final_leg_pose_env[:, 3:7].clone()
    return active_pose[:, 3:7].clone()


def _final_leg_pose_env(unwrapped) -> torch.Tensor:
    top_pose_w = unwrapped.square_table_top.data.root_pose_w[0:1]
    target_pos_top = torch.tensor((LEG4_TARGET_POS_TOP,), dtype=torch.float32, device=unwrapped.device)
    target_quat_top = torch.tensor((IDENTITY_QUAT,), dtype=torch.float32, device=unwrapped.device)
    target_pos_w, target_quat_w = combine_frame_transforms(
        top_pose_w[:, :3],
        top_pose_w[:, 3:7],
        target_pos_top,
        target_quat_top,
    )
    return _world_pose_to_env_pose(unwrapped, torch.cat((target_pos_w, target_quat_w), dim=-1))


def _format_pose(pose: torch.Tensor) -> str:
    pose_cpu = pose[0].detach().cpu()
    pos = ", ".join(f"{value:.4f}" for value in pose_cpu[:3])
    quat = ", ".join(f"{value:.4f}" for value in pose_cpu[3:7])
    return f"pos=[{pos}] quat_wxyz=[{quat}]"


def _format_vec(vec: torch.Tensor) -> str:
    vec_cpu = vec[0].detach().cpu()
    values = ", ".join(f"{value:.4f}" for value in vec_cpu)
    return f"[{values}]"


def _final_relative_pose_diagnostics(unwrapped) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    rel_pos, rel_quat = unwrapped._assembled_relative_pose()
    target_pos = torch.tensor((LEG4_TARGET_POS_TOP,), dtype=torch.float32, device=unwrapped.device)
    pos_error = rel_pos - target_pos
    return rel_pos, rel_quat, pos_error


def _run_phase(
    env,
    phase: str,
    active_arm: str,
    target_pose: torch.Tensor,
    left_pose: torch.Tensor,
    right_pose: torch.Tensor,
    left_grip: float,
    right_grip: float,
    target_left_grip: float,
    target_right_grip: float,
    global_step: int,
    final_leg_pose_env: torch.Tensor,
    steps: int | None = None,
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
        command_pose[:, 3:7] = _nlerp_quat(start_pose[:, 3:7], resolved_target_pose[:, 3:7], alpha)
        if active_arm == "left":
            left_pose = command_pose
        else:
            right_pose = command_pose

        left_grip = (1.0 - alpha) * start_left_grip + alpha * target_left_grip
        right_grip = (1.0 - alpha) * start_right_grip + alpha * target_right_grip
        actions = _make_action(left_pose, left_grip, right_pose, right_grip)
        _step_without_auto_reset(env, actions)
        _visualize_markers(unwrapped)

        if args_cli.print_interval > 0 and global_step % args_cli.print_interval == 0:
            ee_pose = _active_ee_pose(unwrapped, active_arm)
            ee_error = torch.linalg.norm(ee_pose[:, :3] - resolved_target_pose[:, :3], dim=-1).mean()
            ee_ori_error = _quat_angle_error(ee_pose[:, 3:7], resolved_target_pose[:, 3:7]).mean()
            finger_center = _finger_center_in_root_frame(unwrapped, active_arm)
            table_clearance = (
                finger_center[:, 2]
                - args_cli.finger_collision_half_height
                - _table_surface_z(unwrapped)
            ).mean()
            leg_error_xyz = _leg_pose_env(unwrapped)[:, :3] - final_leg_pose_env[:, :3]
            leg_error = torch.linalg.norm(leg_error_xyz, dim=-1).mean()
            _, rel_quat, _ = _final_relative_pose_diagnostics(unwrapped)
            leg_ori_error = _quat_angle_error(
                rel_quat,
                _quat_tensor(unwrapped.device, IDENTITY_QUAT),
            ).mean()
            leg_pos_in_finger, leg_quat_in_finger = _held_leg_relative_to_finger(unwrapped, active_arm)
            finger_leg_dist = torch.linalg.norm(_leg_pose_env(unwrapped)[:, :3] - finger_center, dim=-1).mean()
            success = bool(unwrapped._success()[0].item())
            print(
                f"[INFO]: phase={phase} arm={active_arm} "
                f"ee_error={float(ee_error):.4f} m "
                f"ee_ori_error={float(ee_ori_error):.4f} rad "
                f"finger_leg_dist={float(finger_leg_dist):.4f} m "
                f"finger_table_clearance={float(table_clearance):.4f} m "
                f"leg_target_error={float(leg_error):.4f} m "
                f"leg_ori_error={float(leg_ori_error):.4f} rad "
                f"leg_error_xyz={_format_vec(leg_error_xyz)} "
                f"leg_in_finger_pos={_format_vec(leg_pos_in_finger)} "
                f"leg_in_finger_quat={_format_vec(leg_quat_in_finger)} "
                f"target_b={_format_pose(resolved_target_pose)} "
                f"leg_b={_format_pose(_leg_pose_env(unwrapped))} "
                f"success={success}"
            )

        global_step += 1

    return left_pose, right_pose, left_grip, right_grip, global_step


def _print_final_status(unwrapped) -> tuple[bool, float, bool, bool]:
    success = bool(unwrapped._success()[0].item())
    reward = float(unwrapped._get_rewards()[0].item())
    terminated, truncated = unwrapped._get_dones()
    terminated_value = bool(terminated[0].item())
    truncated_value = bool(truncated[0].item())
    rel_pos, rel_quat, pos_error = _final_relative_pose_diagnostics(unwrapped)
    rel_ori_error = _quat_angle_error(rel_quat, _quat_tensor(unwrapped.device, IDENTITY_QUAT))
    print(
        "[INFO]: final "
        f"success={success} reward={reward:.1f} terminated={terminated_value} timeout={truncated_value} "
        f"rel_pos={_format_vec(rel_pos)} rel_quat={_format_vec(rel_quat)} "
        f"rel_pos_error={_format_vec(pos_error)} rel_ori_error={float(rel_ori_error[0]):.4f} rad"
    )
    return success, reward, terminated_value, truncated_value


def _run_scripted_assembly(env) -> int:
    global _MARKERS

    unwrapped = env.unwrapped
    env.reset()
    _set_planned_grasp_pose(None)
    _set_planned_insert_pose(None)
    _MARKERS = None if args_cli.disable_markers else _make_markers()
    _visualize_markers(unwrapped)

    with torch.inference_mode():
        left_pose, right_pose = unwrapped._get_ee_poses_in_root_frame()
        left_pose = left_pose.clone()
        right_pose = right_pose.clone()
        left_grip = OPEN_GRIPPER
        right_grip = OPEN_GRIPPER
        global_step = 0

        leg_pose = _leg_pose_env(unwrapped)
        active_arm = "left" if float(leg_pose[0, 1]) > 0.0 else "right"
        active_pose = left_pose if active_arm == "left" else right_pose
        final_leg_pose_env = _final_leg_pose_env(unwrapped)
        grasp_quat = _grasp_quat(unwrapped, active_pose, final_leg_pose_env)

        def current_grasp_pose(clearance: float) -> torch.Tensor:
            current_leg_pose = _leg_pose_env(unwrapped)
            grasp_center_pos = current_leg_pose[:, :3].clone()
            grasp_center_pos[:, 2] += args_cli.finger_center_offset_z + clearance
            grasp_pose = _ee_pose_for_finger_center(unwrapped, active_arm, grasp_center_pos, grasp_quat)
            if clearance <= 0.0:
                _set_planned_grasp_pose(grasp_pose)
            return grasp_pose

        pre_grasp_pose = current_grasp_pose(args_cli.overhead_clearance)
        current_grasp_pose(0.0)
        raise_pos = active_pose[:, :3].clone()
        raise_pos[:, 2] = torch.maximum(
            raise_pos[:, 2],
            torch.tensor(pre_grasp_pose[0, 2].item(), dtype=torch.float32, device=unwrapped.device),
        )
        raise_pose = _pose_from_pos_quat(raise_pos, grasp_quat)

        print(
            f"[INFO]: physical one_leg setup arm={active_arm} "
            f"orientation={args_cli.gripper_orientation} "
            f"frame_markers={not args_cli.disable_markers} "
            f"grasp_quat={_format_vec(grasp_quat)} "
            f"leg_initial_b={_format_pose(leg_pose)} "
            f"final_leg_b={_format_pose(final_leg_pose_env)} "
            f"table_surface_z={_table_surface_z(unwrapped):.4f} m"
        )

        grasp_phases: tuple[tuple[str, torch.Tensor, float, int, Callable[[], torch.Tensor] | None], ...] = (
            ("raise-arm", raise_pose, OPEN_GRIPPER, args_cli.phase_steps, None),
            (
                "move-above-grasp",
                pre_grasp_pose,
                OPEN_GRIPPER,
                args_cli.phase_steps,
                lambda: current_grasp_pose(args_cli.overhead_clearance),
            ),
            (
                "descend-to-grasp",
                current_grasp_pose(0.0),
                OPEN_GRIPPER,
                args_cli.phase_steps,
                lambda: current_grasp_pose(0.0),
            ),
            (
                "settle-at-grasp",
                current_grasp_pose(0.0),
                OPEN_GRIPPER,
                args_cli.settle_steps,
                lambda: current_grasp_pose(0.0),
            ),
            (
                "close",
                current_grasp_pose(0.0),
                CLOSE_GRIPPER,
                args_cli.close_steps,
                lambda: current_grasp_pose(0.0),
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
                active_arm=active_arm,
                target_pose=phase_target_pose,
                left_pose=left_pose,
                right_pose=right_pose,
                left_grip=left_grip,
                right_grip=right_grip,
                target_left_grip=target_left_grip,
                target_right_grip=target_right_grip,
                global_step=global_step,
                final_leg_pose_env=final_leg_pose_env,
                steps=steps,
                target_pose_fn=target_pose_fn,
            )

        leg_z_before_lift = float(_leg_pose_env(unwrapped)[0, 2].item())
        lift_center = _finger_center_in_root_frame(unwrapped, active_arm).clone()
        lift_center[:, 2] += args_cli.lift_height
        lift_pose = _ee_pose_for_finger_center(unwrapped, active_arm, lift_center, grasp_quat)
        left_pose, right_pose, left_grip, right_grip, global_step = _run_phase(
            env=env,
            phase="lift-test",
            active_arm=active_arm,
            target_pose=lift_pose,
            left_pose=left_pose,
            right_pose=right_pose,
            left_grip=left_grip,
            right_grip=right_grip,
            target_left_grip=left_grip,
            target_right_grip=right_grip,
            global_step=global_step,
            final_leg_pose_env=final_leg_pose_env,
        )

        leg_pose_after_lift = _leg_pose_env(unwrapped)
        lift_delta = float(leg_pose_after_lift[0, 2].item() - leg_z_before_lift)
        min_lift = min(0.035, 0.25 * args_cli.lift_height)
        print(f"[INFO]: lift-test checkpoint lift_delta={lift_delta:.4f} m min_lift={min_lift:.4f} m")
        if lift_delta < min_lift:
            print("[ERROR]: physical grasp failed: leg4 did not lift with the gripper.")
            _print_final_status(unwrapped)
            return 1

        leg_pos_in_finger, leg_quat_in_finger = _held_leg_relative_to_finger(
            unwrapped, active_arm, leg_pose_after_lift
        )
        reorient_leg_pose = leg_pose_after_lift.clone()
        high_insert_z = torch.maximum(
            final_leg_pose_env[:, 2] + args_cli.overhead_clearance + args_cli.insert_clearance,
            torch.full_like(final_leg_pose_env[:, 2], _table_surface_z(unwrapped) + args_cli.overhead_clearance),
        )
        reorient_leg_pose[:, 2] = torch.maximum(reorient_leg_pose[:, 2], high_insert_z)
        reorient_leg_pose[:, 3:7] = final_leg_pose_env[:, 3:7]
        reorient_pose = _ee_pose_for_held_leg_pose(
            unwrapped,
            active_arm,
            reorient_leg_pose,
            leg_pos_in_finger,
            leg_quat_in_finger,
        )
        _set_planned_insert_pose(reorient_pose)
        print(
            "[INFO]: held-leg relation after lift "
            f"leg_in_finger_pos={_format_vec(leg_pos_in_finger)} "
            f"leg_in_finger_quat={_format_vec(leg_quat_in_finger)} "
            f"reorient_target_b={_format_pose(reorient_pose)}"
        )
        left_pose, right_pose, left_grip, right_grip, global_step = _run_phase(
            env=env,
            phase="reorient-for-insert",
            active_arm=active_arm,
            target_pose=reorient_pose,
            left_pose=left_pose,
            right_pose=right_pose,
            left_grip=left_grip,
            right_grip=right_grip,
            target_left_grip=left_grip,
            target_right_grip=right_grip,
            global_step=global_step,
            final_leg_pose_env=final_leg_pose_env,
        )

        leg_pose_after_reorient = _leg_pose_env(unwrapped)
        leg_pos_in_finger, leg_quat_in_finger = _held_leg_relative_to_finger(
            unwrapped, active_arm, leg_pose_after_reorient
        )
        pre_insert_leg_pose = final_leg_pose_env.clone()
        pre_insert_leg_pose[:, 2] += args_cli.overhead_clearance + args_cli.insert_clearance
        insert_leg_pose = final_leg_pose_env.clone()
        insert_leg_pose[:, 2] += args_cli.insert_clearance
        seat_leg_pose = final_leg_pose_env.clone()
        seat_leg_pose[:, 2] -= args_cli.insert_push_depth

        pre_insert_pose = _ee_pose_for_held_leg_pose(
            unwrapped,
            active_arm,
            pre_insert_leg_pose,
            leg_pos_in_finger,
            leg_quat_in_finger,
        )
        insert_pose = _ee_pose_for_held_leg_pose(
            unwrapped,
            active_arm,
            insert_leg_pose,
            leg_pos_in_finger,
            leg_quat_in_finger,
        )
        seat_pose = _ee_pose_for_held_leg_pose(
            unwrapped,
            active_arm,
            seat_leg_pose,
            leg_pos_in_finger,
            leg_quat_in_finger,
        )
        _set_planned_insert_pose(seat_pose)
        print(
            "[INFO]: insert planning "
            f"insert_quat={_format_vec(seat_pose[:, 3:7])} "
            f"leg_in_finger_pos={_format_vec(leg_pos_in_finger)} "
            f"leg_in_finger_quat={_format_vec(leg_quat_in_finger)} "
            f"pre_insert_b={_format_pose(pre_insert_pose)} "
            f"seat_b={_format_pose(seat_pose)}"
        )

        transport_phases = (
            ("move-above-insert", pre_insert_pose, CLOSE_GRIPPER, args_cli.phase_steps),
            ("descend-to-pre-insert", insert_pose, CLOSE_GRIPPER, args_cli.phase_steps),
            ("seat-insert", seat_pose, CLOSE_GRIPPER, args_cli.phase_steps),
            ("hold-insert", seat_pose, CLOSE_GRIPPER, args_cli.settle_steps),
            ("open", seat_pose, OPEN_GRIPPER, args_cli.close_steps),
            ("retreat", pre_insert_pose, OPEN_GRIPPER, args_cli.phase_steps),
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
                active_arm=active_arm,
                target_pose=phase_target_pose,
                left_pose=left_pose,
                right_pose=right_pose,
                left_grip=left_grip,
                right_grip=right_grip,
                target_left_grip=target_left_grip,
                target_right_grip=target_right_grip,
                global_step=global_step,
                final_leg_pose_env=final_leg_pose_env,
                steps=steps,
            )

        success, reward, terminated_value, _ = _print_final_status(unwrapped)
        if not success or reward < 1.0 or not terminated_value:
            print("[ERROR]: scripted one_leg physical assembly did not reach the success condition.")
            return 1
        return 0


def main() -> int:
    env_cfg = parse_env_cfg(
        TASK_NAME,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
    )
    env = gym.make(TASK_NAME, cfg=env_cfg)

    try:
        print(f"[INFO]: Gym observation space: {env.observation_space}")
        print(f"[INFO]: Gym action space: {env.action_space}")
        print(f"[INFO]: Part/gripper frame markers: enabled={not args_cli.disable_markers}")
        return _run_scripted_assembly(env)
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
