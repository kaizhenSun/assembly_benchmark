# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Bimanual controllers for Galaxea R1 Pro smoke-test environments."""

from __future__ import annotations

import torch

from isaaclab.assets import Articulation
from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg
from isaaclab.utils.math import combine_frame_transforms, matrix_from_quat, quat_inv, subtract_frame_transforms


class BimanualJointPositionController:
    """Maps compact bimanual arm and gripper actions to R1 Pro joint position targets."""

    action_dim = 16

    def __init__(
        self,
        robot: Articulation,
        left_arm_joint_names: list[str],
        right_arm_joint_names: list[str],
        left_gripper_joint_names: list[str],
        right_gripper_joint_names: list[str],
        arm_action_scale: float,
        gripper_min: float,
        gripper_max: float,
    ):
        self.robot = robot
        self.arm_action_scale = arm_action_scale
        self.gripper_min = gripper_min
        self.gripper_max = gripper_max

        self.left_arm_joint_ids, _ = robot.find_joints(left_arm_joint_names, preserve_order=True)
        self.right_arm_joint_ids, _ = robot.find_joints(right_arm_joint_names, preserve_order=True)
        self.left_gripper_joint_ids, _ = robot.find_joints(left_gripper_joint_names, preserve_order=True)
        self.right_gripper_joint_ids, _ = robot.find_joints(right_gripper_joint_names, preserve_order=True)

        self.arm_joint_ids = self.left_arm_joint_ids + self.right_arm_joint_ids
        self.gripper_joint_ids = self.left_gripper_joint_ids + self.right_gripper_joint_ids
        self.joint_ids = (
            self.left_arm_joint_ids
            + self.left_gripper_joint_ids
            + self.right_arm_joint_ids
            + self.right_gripper_joint_ids
        )

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        """Reset controller state."""
        return None

    def compute(self, actions: torch.Tensor) -> torch.Tensor:
        """Compute position targets for controlled joints."""
        if actions.shape[-1] != self.action_dim:
            raise ValueError(f"Expected action dimension {self.action_dim}, got {actions.shape[-1]}.")

        joint_pos = self.robot.data.default_joint_pos.clone()
        left_arm_target = (
            self.robot.data.default_joint_pos[:, self.left_arm_joint_ids] + actions[:, 0:7] * self.arm_action_scale
        )
        left_gripper_target = self._expand_gripper_action(actions[:, 7], len(self.left_gripper_joint_ids))
        right_arm_target = (
            self.robot.data.default_joint_pos[:, self.right_arm_joint_ids] + actions[:, 8:15] * self.arm_action_scale
        )
        right_gripper_target = self._expand_gripper_action(actions[:, 15], len(self.right_gripper_joint_ids))

        joint_pos[:, self.left_arm_joint_ids] = self._clamp_to_limits(left_arm_target, self.left_arm_joint_ids)
        joint_pos[:, self.left_gripper_joint_ids] = self._clamp_to_limits(
            left_gripper_target, self.left_gripper_joint_ids
        )
        joint_pos[:, self.right_arm_joint_ids] = self._clamp_to_limits(right_arm_target, self.right_arm_joint_ids)
        joint_pos[:, self.right_gripper_joint_ids] = self._clamp_to_limits(
            right_gripper_target, self.right_gripper_joint_ids
        )

        return joint_pos[:, self.joint_ids]

    def _expand_gripper_action(self, action: torch.Tensor, num_joints: int) -> torch.Tensor:
        target = (action.clamp(-1.0, 1.0) + 1.0) * 0.5 * (self.gripper_max - self.gripper_min) + self.gripper_min
        return target.unsqueeze(-1).repeat(1, num_joints)

    def _clamp_to_limits(self, joint_pos: torch.Tensor, joint_ids: list[int]) -> torch.Tensor:
        limits = self.robot.data.soft_joint_pos_limits[:, joint_ids]
        return torch.clamp(joint_pos, min=limits[..., 0], max=limits[..., 1])


class BimanualDifferentialIKController(BimanualJointPositionController):
    """Maps bimanual pose commands to arm joint targets with Isaac Lab Differential IK."""

    def __init__(
        self,
        robot: Articulation,
        left_arm_joint_names: list[str],
        right_arm_joint_names: list[str],
        left_gripper_joint_names: list[str],
        right_gripper_joint_names: list[str],
        left_ee_link_name: str,
        right_ee_link_name: str,
        left_ik_link_name: str,
        right_ik_link_name: str,
        arm_action_scale: float,
        gripper_min: float,
        gripper_max: float,
        num_envs: int,
        device: str,
    ):
        super().__init__(
            robot=robot,
            left_arm_joint_names=left_arm_joint_names,
            right_arm_joint_names=right_arm_joint_names,
            left_gripper_joint_names=left_gripper_joint_names,
            right_gripper_joint_names=right_gripper_joint_names,
            arm_action_scale=arm_action_scale,
            gripper_min=gripper_min,
            gripper_max=gripper_max,
        )
        controller_cfg = DifferentialIKControllerCfg(
            command_type="pose", use_relative_mode=False, ik_method="dls", ik_params={"lambda_val": 0.08}
        )
        self.left_ik = DifferentialIKController(controller_cfg, num_envs=num_envs, device=device)
        self.right_ik = DifferentialIKController(controller_cfg, num_envs=num_envs, device=device)
        self.left_pose_command = torch.zeros(num_envs, 7, device=device)
        self.right_pose_command = torch.zeros(num_envs, 7, device=device)
        self.left_pose_command_initialized = torch.zeros(num_envs, dtype=torch.bool, device=device)
        self.right_pose_command_initialized = torch.zeros(num_envs, dtype=torch.bool, device=device)

        self.left_ee_body_idx = robot.find_bodies(left_ee_link_name)[0][0]
        self.right_ee_body_idx = robot.find_bodies(right_ee_link_name)[0][0]
        self.left_ik_body_idx = robot.find_bodies(left_ik_link_name)[0][0]
        self.right_ik_body_idx = robot.find_bodies(right_ik_link_name)[0][0]
        self.left_ik_jacobian_idx = self._body_to_jacobian_idx(self.left_ik_body_idx)
        self.right_ik_jacobian_idx = self._body_to_jacobian_idx(self.right_ik_body_idx)

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        self.left_ik.reset(env_ids)
        self.right_ik.reset(env_ids)
        if env_ids is None:
            self.left_pose_command_initialized[:] = False
            self.right_pose_command_initialized[:] = False
        else:
            self.left_pose_command_initialized[env_ids] = False
            self.right_pose_command_initialized[env_ids] = False

    def compute(self, actions: torch.Tensor) -> torch.Tensor:
        if actions.shape[-1] != self.action_dim:
            raise ValueError(f"Expected action dimension {self.action_dim}, got {actions.shape[-1]}.")

        joint_pos = self.robot.data.default_joint_pos.clone()
        left_arm_target = self._compute_arm_target(
            controller=self.left_ik,
            command=actions[:, 0:7],
            ee_body_idx=self.left_ee_body_idx,
            ik_body_idx=self.left_ik_body_idx,
            ik_jacobian_idx=self.left_ik_jacobian_idx,
            arm_joint_ids=self.left_arm_joint_ids,
            pose_command=self.left_pose_command,
            pose_command_initialized=self.left_pose_command_initialized,
        )
        left_gripper_target = self._expand_gripper_action(actions[:, 7], len(self.left_gripper_joint_ids))
        right_arm_target = self._compute_arm_target(
            controller=self.right_ik,
            command=actions[:, 8:15],
            ee_body_idx=self.right_ee_body_idx,
            ik_body_idx=self.right_ik_body_idx,
            ik_jacobian_idx=self.right_ik_jacobian_idx,
            arm_joint_ids=self.right_arm_joint_ids,
            pose_command=self.right_pose_command,
            pose_command_initialized=self.right_pose_command_initialized,
        )
        right_gripper_target = self._expand_gripper_action(actions[:, 15], len(self.right_gripper_joint_ids))

        joint_pos[:, self.left_arm_joint_ids] = self._clamp_to_limits(left_arm_target, self.left_arm_joint_ids)
        joint_pos[:, self.left_gripper_joint_ids] = self._clamp_to_limits(
            left_gripper_target, self.left_gripper_joint_ids
        )
        joint_pos[:, self.right_arm_joint_ids] = self._clamp_to_limits(right_arm_target, self.right_arm_joint_ids)
        joint_pos[:, self.right_gripper_joint_ids] = self._clamp_to_limits(
            right_gripper_target, self.right_gripper_joint_ids
        )

        return joint_pos[:, self.joint_ids]

    def _compute_arm_target(
        self,
        controller: DifferentialIKController,
        command: torch.Tensor,
        ee_body_idx: int,
        ik_body_idx: int,
        ik_jacobian_idx: int,
        arm_joint_ids: list[int],
        pose_command: torch.Tensor,
        pose_command_initialized: torch.Tensor,
    ) -> torch.Tensor:
        ee_pose_w = self.robot.data.body_pose_w[:, ee_body_idx]
        ik_pose_w = self.robot.data.body_pose_w[:, ik_body_idx]
        root_pose_w = self.robot.data.root_pose_w
        ee_pos_b, ee_quat_b = subtract_frame_transforms(
            root_pose_w[:, 0:3], root_pose_w[:, 3:7], ee_pose_w[:, 0:3], ee_pose_w[:, 3:7]
        )
        ik_pos_b, ik_quat_b = subtract_frame_transforms(
            root_pose_w[:, 0:3], root_pose_w[:, 3:7], ik_pose_w[:, 0:3], ik_pose_w[:, 3:7]
        )

        ee_command = self._resolve_pose_command(command, ee_pos_b, ee_quat_b, pose_command, pose_command_initialized)
        ik_to_ee_pos, ik_to_ee_quat = subtract_frame_transforms(ik_pos_b, ik_quat_b, ee_pos_b, ee_quat_b)
        ee_to_ik_pos, ee_to_ik_quat = subtract_frame_transforms(ik_to_ee_pos, ik_to_ee_quat)
        ik_command_pos, ik_command_quat = combine_frame_transforms(
            ee_command[:, 0:3], ee_command[:, 3:7], ee_to_ik_pos, ee_to_ik_quat
        )
        controller.set_command(torch.cat((ik_command_pos, ik_command_quat), dim=-1))

        jacobian = self.robot.root_physx_view.get_jacobians()[:, ik_jacobian_idx, :, arm_joint_ids].clone()
        base_rot_matrix = matrix_from_quat(quat_inv(root_pose_w[:, 3:7]))
        jacobian[:, :3, :] = torch.bmm(base_rot_matrix, jacobian[:, :3, :])
        jacobian[:, 3:, :] = torch.bmm(base_rot_matrix, jacobian[:, 3:, :])

        joint_pos = self.robot.data.joint_pos[:, arm_joint_ids]
        return controller.compute(ik_pos_b, ik_quat_b, jacobian, joint_pos)

    def _body_to_jacobian_idx(self, body_idx: int) -> int:
        jacobian_body_count = self.robot.root_physx_view.get_jacobians().shape[1]
        body_count = len(self.robot.body_names)
        if jacobian_body_count == body_count:
            return body_idx
        if jacobian_body_count == body_count - 1:
            return body_idx - 1
        raise RuntimeError(
            f"Cannot map body index {body_idx} to Jacobian row: "
            f"{jacobian_body_count} Jacobian rows for {body_count} bodies."
        )

    def _resolve_pose_command(
        self,
        command: torch.Tensor,
        ee_pos: torch.Tensor,
        ee_quat: torch.Tensor,
        pose_command: torch.Tensor,
        pose_command_initialized: torch.Tensor,
    ) -> torch.Tensor:
        zero_pose = torch.linalg.norm(command, dim=1) < 1.0e-6
        command = self._sanitize_pose_command(command, ee_pos, ee_quat)

        uninitialized_zero_pose = zero_pose & ~pose_command_initialized
        pose_command[uninitialized_zero_pose, 0:3] = ee_pos[uninitialized_zero_pose]
        pose_command[uninitialized_zero_pose, 3:7] = ee_quat[uninitialized_zero_pose]
        pose_command_initialized[uninitialized_zero_pose] = True

        nonzero_pose = ~zero_pose
        pose_command[nonzero_pose] = command[nonzero_pose]
        pose_command_initialized[nonzero_pose] = True

        return pose_command.clone()

    @staticmethod
    def _sanitize_pose_command(command: torch.Tensor, ee_pos: torch.Tensor, ee_quat: torch.Tensor) -> torch.Tensor:
        command = command.clone()
        quat = command[:, 3:7]
        quat_norm = torch.linalg.norm(quat, dim=1, keepdim=True)
        valid_quat = quat_norm > 1.0e-6
        command[:, 3:7] = torch.where(valid_quat, quat / torch.clamp(quat_norm, min=1.0e-6), ee_quat)
        return command
