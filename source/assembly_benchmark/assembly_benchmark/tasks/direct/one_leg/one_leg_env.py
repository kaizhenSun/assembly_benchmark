# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from collections.abc import Sequence

import torch

from isaaclab.envs import DirectRLEnv
from isaaclab.utils.math import matrix_from_quat, subtract_frame_transforms

from assembly_benchmark.controllers import BimanualDifferentialIKController

from .one_leg_env_cfg import OneLegWholeBodyIKEnvCfg


class OneLegEnv(DirectRLEnv):
    """R1 Pro FurnitureBench one_leg assembly task using whole-body IK."""

    cfg: OneLegWholeBodyIKEnvCfg

    def __init__(
        self, cfg: OneLegWholeBodyIKEnvCfg, render_mode: str | None = None, **kwargs
    ):
        super().__init__(cfg, render_mode, **kwargs)

        self.controller = BimanualDifferentialIKController(
            robot=self.robot,
            left_arm_joint_names=self.cfg.left_arm_joint_names,
            right_arm_joint_names=self.cfg.right_arm_joint_names,
            left_gripper_joint_names=self.cfg.left_gripper_joint_names,
            right_gripper_joint_names=self.cfg.right_gripper_joint_names,
            left_ee_link_name=self.cfg.left_ee_link_name,
            right_ee_link_name=self.cfg.right_ee_link_name,
            left_ik_link_name=self.cfg.left_ik_link_name,
            right_ik_link_name=self.cfg.right_ik_link_name,
            arm_action_scale=self.cfg.arm_action_scale,
            gripper_min=self.cfg.gripper_min,
            gripper_max=self.cfg.gripper_max,
            num_envs=self.num_envs,
            device=self.device,
            torso_joint_names=self.cfg.torso_joint_names,
            include_torso_in_ik=True,
        )

        self.controlled_joint_ids = self.controller.joint_ids
        self.left_ee_body_idx = self.robot.find_bodies(self.cfg.left_ee_link_name)[0][0]
        self.right_ee_body_idx = self.robot.find_bodies(self.cfg.right_ee_link_name)[0][0]
        self.actions = torch.zeros((self.num_envs, self.cfg.action_space), device=self.device)
        self.joint_targets = self.robot.data.default_joint_pos[:, self.controlled_joint_ids].clone()

        self.assembled_target_positions = torch.tensor(
            (
                (-0.05625, 0.046875, -0.05625),
                (0.05625, 0.046875, -0.05625),
                (-0.05625, 0.046875, 0.05625),
                (0.05625, 0.046875, 0.05625),
            ),
            dtype=torch.float32,
            device=self.device,
        )
        target_quat = torch.tensor((1.0, 0.0, 0.0, 0.0), dtype=torch.float32, device=self.device)
        self.assembled_target_poses = torch.cat(
            (self.assembled_target_positions, target_quat.repeat(4, 1)),
            dim=-1,
        )
        self.assembled_pos_threshold = torch.tensor(
            self.cfg.assembled_pos_threshold, dtype=torch.float32, device=self.device
        )

        if len(self.controlled_joint_ids) != 22:
            raise RuntimeError(
                "OneLeg whole-body IK expected 22 controlled joints "
                f"(torso + both arms + both grippers), got {len(self.controlled_joint_ids)}."
            )

    def _setup_scene(self) -> None:
        self.robot = self.scene["robot"]
        self.square_table_top = self.scene["square_table_top"]
        self.square_table_leg1 = self.scene["square_table_leg1"]
        self.square_table_leg2 = self.scene["square_table_leg2"]
        self.square_table_leg3 = self.scene["square_table_leg3"]
        self.square_table_leg4 = self.scene["square_table_leg4"]
        self.furniture_parts = (
            self.square_table_top,
            self.square_table_leg1,
            self.square_table_leg2,
            self.square_table_leg3,
            self.square_table_leg4,
        )
        self.sim.set_camera_view(eye=(2.2, 1.6, 1.7), target=(0.65, 0.0, 0.9))

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self.actions = actions.clone()
        self.joint_targets = self.controller.compute(self.actions)

    def _apply_action(self) -> None:
        self.robot.set_joint_position_target(self.joint_targets, joint_ids=self.controlled_joint_ids)

    def _get_observations(self) -> dict:
        left_ee_pose_b, right_ee_pose_b = self._get_ee_poses_in_root_frame()
        furniture_poses = [self._object_pose_in_env_frame(part) for part in self.furniture_parts]
        target_poses = self.assembled_target_poses.flatten().repeat(self.num_envs, 1)
        obs = torch.cat(
            (
                self.robot.data.joint_pos[:, self.controlled_joint_ids],
                self.robot.data.joint_vel[:, self.controlled_joint_ids],
                left_ee_pose_b,
                right_ee_pose_b,
                *furniture_poses,
                target_poses,
            ),
            dim=-1,
        )
        return {"policy": obs}

    def _get_rewards(self) -> torch.Tensor:
        return self.cfg.rew_scale_success * self._success().float()

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        return self._success(), time_out

    def _reset_idx(self, env_ids: Sequence[int] | None) -> None:
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES
        super()._reset_idx(env_ids)

        joint_pos = self.robot.data.default_joint_pos[env_ids].clone()
        joint_vel = self.robot.data.default_joint_vel[env_ids].clone()
        default_root_state = self.robot.data.default_root_state[env_ids].clone()
        default_root_state[:, :3] += self.scene.env_origins[env_ids]

        self.robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self.robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)
        self.robot.set_joint_position_target(joint_pos, env_ids=env_ids)
        self.controller.reset(env_ids)

        for part in self.furniture_parts:
            self._reset_rigid_object_to_default(part, env_ids)

    def _get_ee_poses_in_root_frame(self) -> tuple[torch.Tensor, torch.Tensor]:
        root_pose_w = self.robot.data.root_pose_w
        left_ee_pose_w = self.robot.data.body_pose_w[:, self.left_ee_body_idx]
        right_ee_pose_w = self.robot.data.body_pose_w[:, self.right_ee_body_idx]
        left_pos_b, left_quat_b = subtract_frame_transforms(
            root_pose_w[:, :3],
            root_pose_w[:, 3:7],
            left_ee_pose_w[:, :3],
            left_ee_pose_w[:, 3:7],
        )
        right_pos_b, right_quat_b = subtract_frame_transforms(
            root_pose_w[:, :3],
            root_pose_w[:, 3:7],
            right_ee_pose_w[:, :3],
            right_ee_pose_w[:, 3:7],
        )
        return (
            torch.cat((left_pos_b, left_quat_b), dim=-1),
            torch.cat((right_pos_b, right_quat_b), dim=-1),
        )

    def _object_pose_in_env_frame(self, asset) -> torch.Tensor:
        pose = asset.data.root_pose_w.clone()
        pose[:, :3] -= self.scene.env_origins
        return pose

    def _assembled_relative_pose(self) -> tuple[torch.Tensor, torch.Tensor]:
        top_pose_w = self.square_table_top.data.root_pose_w
        leg_pose_w = self.square_table_leg4.data.root_pose_w
        return subtract_frame_transforms(
            top_pose_w[:, :3],
            top_pose_w[:, 3:7],
            leg_pose_w[:, :3],
            leg_pose_w[:, 3:7],
        )

    def _success(self) -> torch.Tensor:
        rel_pos, rel_quat = self._assembled_relative_pose()
        pos_error = torch.abs(rel_pos.unsqueeze(1) - self.assembled_target_positions.unsqueeze(0))
        pos_match = torch.all(pos_error <= self.assembled_pos_threshold.view(1, 1, 3), dim=-1)

        rel_rot = matrix_from_quat(rel_quat)
        rel_rot_diag = torch.diagonal(rel_rot, dim1=-2, dim2=-1)
        ori_match = torch.all(rel_rot_diag >= self.cfg.assembled_ori_bound, dim=-1)
        return torch.any(pos_match, dim=1) & ori_match

    def _reset_rigid_object_to_default(self, asset, env_ids: torch.Tensor) -> None:
        root_state = asset.data.default_root_state[env_ids].clone()
        root_state[:, :3] += self.scene.env_origins[env_ids]
        root_state[:, 7:] = 0.0
        asset.write_root_pose_to_sim(root_state[:, :7], env_ids)
        asset.write_root_velocity_to_sim(root_state[:, 7:], env_ids)
