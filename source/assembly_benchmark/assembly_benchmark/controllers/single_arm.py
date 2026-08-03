# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Robot-agnostic single-arm Differential IK controller."""

from __future__ import annotations

import math

import torch

from isaaclab.assets import Articulation
from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg
from isaaclab.utils.math import combine_frame_transforms, matrix_from_quat, quat_inv, subtract_frame_transforms


class SingleArmDifferentialIKController:
    """Map an absolute EE pose and normalized gripper command to joint position targets."""

    action_dim = 8

    def __init__(
        self,
        robot: Articulation,
        arm_joint_names: list[str],
        gripper_joint_names: list[str],
        ee_link_name: str,
        ik_link_name: str,
        gripper_min: float,
        gripper_max: float,
        num_envs: int,
        device: str,
        control_dt: float | None = None,
        gripper_joint_closed_positions: tuple[float, ...] | None = None,
        gripper_joint_open_positions: tuple[float, ...] | None = None,
    ) -> None:
        if control_dt is not None and (not math.isfinite(control_dt) or control_dt <= 0.0):
            raise ValueError(f"control_dt must be finite and positive when provided, got {control_dt}.")
        if gripper_max <= gripper_min:
            raise ValueError("gripper_max must be greater than gripper_min.")
        self.robot = robot
        self.device = device
        self.num_envs = num_envs
        self.control_dt = control_dt
        self.gripper_min = gripper_min
        self.gripper_max = gripper_max
        self.arm_joint_ids, resolved_arm_names = robot.find_joints(arm_joint_names, preserve_order=True)
        self.gripper_joint_ids, resolved_gripper_names = robot.find_joints(gripper_joint_names, preserve_order=True)
        if list(resolved_arm_names) != arm_joint_names:
            raise RuntimeError(f"Could not resolve arm joints in order: {arm_joint_names} -> {resolved_arm_names}.")
        if list(resolved_gripper_names) != gripper_joint_names:
            raise RuntimeError(
                f"Could not resolve gripper joints in order: {gripper_joint_names} -> {resolved_gripper_names}."
            )
        self.joint_ids = self.arm_joint_ids + self.gripper_joint_ids
        gripper_joint_count = len(self.gripper_joint_ids)
        if gripper_joint_closed_positions is None:
            gripper_joint_closed_positions = (gripper_min,) * gripper_joint_count
        if gripper_joint_open_positions is None:
            gripper_joint_open_positions = (gripper_max,) * gripper_joint_count
        if (
            len(gripper_joint_closed_positions) != gripper_joint_count
            or len(gripper_joint_open_positions) != gripper_joint_count
        ):
            raise ValueError(
                "Gripper closed/open position counts must match the resolved gripper joints: "
                f"{gripper_joint_count} joints, closed={gripper_joint_closed_positions}, "
                f"open={gripper_joint_open_positions}."
            )
        if not all(math.isfinite(value) for value in (*gripper_joint_closed_positions, *gripper_joint_open_positions)):
            raise ValueError("Gripper closed/open joint positions must be finite.")
        self.gripper_joint_closed_positions = torch.tensor(
            gripper_joint_closed_positions, dtype=torch.float32, device=device
        ).unsqueeze(0)
        self.gripper_joint_open_positions = torch.tensor(
            gripper_joint_open_positions, dtype=torch.float32, device=device
        ).unsqueeze(0)

        controller_cfg = DifferentialIKControllerCfg(
            command_type="pose",
            use_relative_mode=False,
            ik_method="dls",
            ik_params={"lambda_val": 0.08},
        )
        self.ik = DifferentialIKController(controller_cfg, num_envs=num_envs, device=device)
        self.pose_command = torch.zeros((num_envs, 7), device=device)
        self.pose_command_initialized = torch.zeros(num_envs, dtype=torch.bool, device=device)

        self.ee_body_idx = self._find_unique_body(ee_link_name)
        self.ik_body_idx = self._find_unique_body(ik_link_name)
        self.ik_jacobian_idx = self._body_to_jacobian_idx(self.ik_body_idx)

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        """Reset controller commands for all or selected environments."""
        self.ik.reset(env_ids)
        if env_ids is None:
            self.pose_command_initialized[:] = False
        else:
            self.pose_command_initialized[env_ids] = False

    def compute(self, actions: torch.Tensor) -> torch.Tensor:
        """Compute controlled joint targets for an ``[xyz, wxyz, gripper]`` command."""
        if actions.ndim != 2 or actions.shape != (self.num_envs, self.action_dim):
            raise ValueError(
                f"Expected actions with shape ({self.num_envs}, {self.action_dim}), got {tuple(actions.shape)}."
            )
        if not torch.all(torch.isfinite(actions)):
            raise ValueError("Single-arm IK actions must contain only finite values.")

        ee_command = actions[:, :7]
        root_pose_w = self.robot.data.root_pose_w
        ee_pose_w = self.robot.data.body_pose_w[:, self.ee_body_idx]
        ik_pose_w = self.robot.data.body_pose_w[:, self.ik_body_idx]
        ee_pos_b, ee_quat_b = subtract_frame_transforms(
            root_pose_w[:, :3], root_pose_w[:, 3:7], ee_pose_w[:, :3], ee_pose_w[:, 3:7]
        )
        ik_pos_b, ik_quat_b = subtract_frame_transforms(
            root_pose_w[:, :3], root_pose_w[:, 3:7], ik_pose_w[:, :3], ik_pose_w[:, 3:7]
        )
        ee_command = self._resolve_pose_command(ee_command, ee_pos_b, ee_quat_b)

        ik_to_ee_pos, ik_to_ee_quat = subtract_frame_transforms(ik_pos_b, ik_quat_b, ee_pos_b, ee_quat_b)
        ee_to_ik_pos, ee_to_ik_quat = subtract_frame_transforms(ik_to_ee_pos, ik_to_ee_quat)
        ik_command_pos, ik_command_quat = combine_frame_transforms(
            ee_command[:, :3], ee_command[:, 3:7], ee_to_ik_pos, ee_to_ik_quat
        )
        self.ik.set_command(torch.cat((ik_command_pos, ik_command_quat), dim=-1))

        jacobian = self._jacobian_in_root_frame(root_pose_w)
        arm_joint_pos = self.robot.data.joint_pos[:, self.arm_joint_ids]
        arm_target = self.ik.compute(
            ik_pos_b,
            ik_quat_b,
            jacobian,
            arm_joint_pos,
        )
        arm_target = self._clamp_to_limits(arm_target, self.arm_joint_ids)
        gripper_target = self._expand_gripper_action(actions[:, 7])
        gripper_target = self._clamp_to_limits(gripper_target, self.gripper_joint_ids)
        target = torch.cat((arm_target, gripper_target), dim=-1)
        return self._apply_control_step_limits(target)

    def _resolve_pose_command(
        self,
        command: torch.Tensor,
        ee_pos_b: torch.Tensor,
        ee_quat_b: torch.Tensor,
    ) -> torch.Tensor:
        zero_pose = torch.linalg.vector_norm(command, dim=-1) < 1.0e-6
        command = command.clone()
        quat = command[:, 3:7]
        quat_norm = torch.linalg.vector_norm(quat, dim=-1, keepdim=True)
        command[:, 3:7] = torch.where(
            quat_norm > 1.0e-6,
            quat / quat_norm.clamp_min(1.0e-6),
            ee_quat_b,
        )

        uninitialized_hold = zero_pose & ~self.pose_command_initialized
        self.pose_command[uninitialized_hold, :3] = ee_pos_b[uninitialized_hold]
        self.pose_command[uninitialized_hold, 3:7] = ee_quat_b[uninitialized_hold]
        self.pose_command_initialized[uninitialized_hold] = True

        commanded = ~zero_pose
        self.pose_command[commanded] = command[commanded]
        self.pose_command_initialized[commanded] = True
        return self.pose_command.clone()

    def _jacobian_in_root_frame(self, root_pose_w: torch.Tensor) -> torch.Tensor:
        jacobian = self.robot.root_physx_view.get_jacobians()[:, self.ik_jacobian_idx, :, self.arm_joint_ids].clone()
        base_rot_matrix = matrix_from_quat(quat_inv(root_pose_w[:, 3:7]))
        jacobian[:, :3, :] = torch.bmm(base_rot_matrix, jacobian[:, :3, :])
        jacobian[:, 3:, :] = torch.bmm(base_rot_matrix, jacobian[:, 3:, :])
        return jacobian

    def _expand_gripper_action(self, action: torch.Tensor) -> torch.Tensor:
        opening = (action.clamp(-1.0, 1.0) + 1.0) * 0.5
        opening = opening * (self.gripper_max - self.gripper_min) + self.gripper_min
        return self.gripper_targets_from_opening(opening)

    def gripper_targets_from_opening(self, opening: torch.Tensor) -> torch.Tensor:
        """Map total jaw opening values to robot-specific gripper joint positions."""
        if opening.ndim != 1:
            raise ValueError(f"Expected one gripper opening per environment, got shape {tuple(opening.shape)}.")
        if not torch.all(torch.isfinite(opening)):
            raise ValueError("Gripper openings must contain only finite values.")
        ratio = (opening.clamp(self.gripper_min, self.gripper_max) - self.gripper_min) / (
            self.gripper_max - self.gripper_min
        )
        return self.gripper_joint_closed_positions + ratio.unsqueeze(-1) * (
            self.gripper_joint_open_positions - self.gripper_joint_closed_positions
        )

    def _clamp_to_limits(self, target: torch.Tensor, joint_ids: list[int]) -> torch.Tensor:
        limits = self.robot.data.soft_joint_pos_limits[:, joint_ids]
        return torch.clamp(target, min=limits[..., 0], max=limits[..., 1])

    def _apply_control_step_limits(self, target: torch.Tensor) -> torch.Tensor:
        if self.control_dt is None:
            return target
        joint_pos = self.robot.data.joint_pos[:, self.joint_ids]
        soft_limits = self.robot.data.soft_joint_pos_limits[:, self.joint_ids]
        max_step = self.robot.data.joint_vel_limits[:, self.joint_ids].clamp_min(0.0) * self.control_dt
        lower = torch.maximum(soft_limits[..., 0], joint_pos - max_step)
        upper = torch.minimum(soft_limits[..., 1], joint_pos + max_step)
        empty_interval = lower > upper
        nearest_soft_limit = torch.clamp(joint_pos, min=soft_limits[..., 0], max=soft_limits[..., 1])
        lower = torch.where(empty_interval, nearest_soft_limit, lower)
        upper = torch.where(empty_interval, nearest_soft_limit, upper)
        return torch.maximum(torch.minimum(target, upper), lower)

    def _find_unique_body(self, body_name: str) -> int:
        body_ids, body_names = self.robot.find_bodies(body_name, preserve_order=True)
        if len(body_ids) != 1 or list(body_names) != [body_name]:
            raise RuntimeError(f"Expected body '{body_name}' exactly once, got ids {body_ids} and names {body_names}.")
        return body_ids[0]

    def _body_to_jacobian_idx(self, body_idx: int) -> int:
        jacobian_body_count = self.robot.root_physx_view.get_jacobians().shape[1]
        body_count = len(self.robot.body_names)
        if jacobian_body_count == body_count:
            return body_idx
        if jacobian_body_count == body_count - 1:
            if body_idx == 0:
                raise RuntimeError("The fixed-base root body has no Jacobian row.")
            return body_idx - 1
        raise RuntimeError(
            f"Cannot map body index {body_idx} to Jacobian row: "
            f"{jacobian_body_count} Jacobian rows for {body_count} bodies."
        )


__all__ = ["SingleArmDifferentialIKController"]
