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
from assembly_benchmark.sensors import configure_r1_pro_gripper_tactile_scene_cfg

from .assembly_benchmark_env_cfg import AssemblyBenchmarkEnvCfg


def _quat_inv(quat: torch.Tensor) -> torch.Tensor:
    conjugate = quat.clone()
    conjugate[..., 1:] *= -1.0
    norm_sq = torch.sum(quat * quat, dim=-1, keepdim=True).clamp_min(1.0e-8)
    return conjugate / norm_sq


def _quat_mul(first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
    w1, x1, y1, z1 = first.unbind(dim=-1)
    w2, x2, y2, z2 = second.unbind(dim=-1)
    return torch.stack(
        (
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ),
        dim=-1,
    )


class AssemblyBenchmarkEnv(DirectRLEnv):
    """Generic R1 Pro assembly task using whole-body IK."""

    cfg: AssemblyBenchmarkEnvCfg

    def __init__(self, cfg: AssemblyBenchmarkEnvCfg, render_mode: str | None = None, **kwargs):
        configure_r1_pro_gripper_tactile_scene_cfg(cfg)
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
            include_torso_in_ik=self.cfg.include_torso_in_ik,
        )

        self.controlled_joint_ids = self.controller.joint_ids
        self.arm_joint_ids = self.controller.arm_joint_ids
        self.left_ee_body_idx = self.robot.find_bodies(self.cfg.left_ee_link_name)[0][0]
        self.right_ee_body_idx = self.robot.find_bodies(self.cfg.right_ee_link_name)[0][0]
        self.actions = torch.zeros((self.num_envs, self.cfg.action_space), device=self.device)
        self.joint_targets = self.robot.data.default_joint_pos[:, self.controlled_joint_ids].clone()

        self.assembled_target_positions = torch.tensor(
            self.cfg.assembled_target_positions, dtype=torch.float32, device=self.device
        )
        self.assembled_target_quats = torch.tensor(
            self.cfg.assembled_target_quats, dtype=torch.float32, device=self.device
        )
        self.assembled_target_poses = torch.cat(
            (self.assembled_target_positions, self.assembled_target_quats),
            dim=-1,
        )
        self.scripted_target_index = self.cfg.scripted_target_index
        self.scripted_target_pose = self.assembled_target_poses[
            self.scripted_target_index : self.scripted_target_index + 1
        ].clone()
        self.assembled_pos_threshold = torch.tensor(
            self.cfg.assembled_pos_threshold, dtype=torch.float32, device=self.device
        )

        if len(self.controlled_joint_ids) != 22:
            raise RuntimeError(
                "AssemblyBenchmark whole-body IK expected 22 controlled joints "
                f"(torso + both arms + both grippers), got {len(self.controlled_joint_ids)}."
            )
        expected_arm_joint_count = len(self.cfg.left_arm_joint_names) + len(self.cfg.right_arm_joint_names)
        if len(self.arm_joint_ids) != expected_arm_joint_count:
            raise RuntimeError(
                f"AssemblyBenchmark expected {expected_arm_joint_count} arm joints, got {len(self.arm_joint_ids)}."
            )

    def _setup_scene(self) -> None:
        self.robot = self.scene["robot"]
        self.assembly_parts_by_name = {name: self.scene[name] for name in self.cfg.assembly_part_names}
        self.assembly_reset_parts = tuple(
            self.assembly_parts_by_name[name] for name in self.cfg.assembly_reset_part_names
        )
        for name, part in self.assembly_parts_by_name.items():
            setattr(self, name, part)
        self.assembly_parent_part = self.assembly_parts_by_name[self.cfg.assembly_parent_part_name]
        self.assembly_child_part = self.assembly_parts_by_name[self.cfg.assembly_child_part_name]
        self.sim.set_camera_view(eye=(2.2, 1.6, 1.7), target=(0.65, 0.0, 0.9))

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self.actions = actions.clone()
        self.joint_targets = self.controller.compute(self.actions)

    def _apply_action(self) -> None:
        self.robot.set_joint_position_target(self.joint_targets, joint_ids=self.controlled_joint_ids)

    def _get_observations(self) -> dict:
        obs = torch.cat(
            (
                self.robot.data.joint_pos[:, self.arm_joint_ids],
                self.robot.data.joint_vel[:, self.arm_joint_ids],
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

        for part in self.assembly_reset_parts:
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

    def _assembled_relative_pose(self) -> tuple[torch.Tensor, torch.Tensor]:
        parent_pose_w = self.assembly_parent_part.data.root_pose_w
        child_pose_w = self.assembly_child_part.data.root_pose_w
        return subtract_frame_transforms(
            parent_pose_w[:, :3],
            parent_pose_w[:, 3:7],
            child_pose_w[:, :3],
            child_pose_w[:, 3:7],
        )

    def _success(self) -> torch.Tensor:
        rel_pos, rel_quat = self._assembled_relative_pose()
        pos_error = torch.abs(rel_pos.unsqueeze(1) - self.assembled_target_positions.unsqueeze(0))
        pos_match = torch.all(pos_error <= self.assembled_pos_threshold.view(1, 1, 3), dim=-1)

        target_error_quat = _quat_mul(
            _quat_inv(self.assembled_target_quats).unsqueeze(0),
            rel_quat.unsqueeze(1),
        )
        rel_rot = matrix_from_quat(target_error_quat.reshape(-1, 4)).reshape(
            self.num_envs, len(self.assembled_target_quats), 3, 3
        )
        rel_rot_diag = torch.diagonal(rel_rot, dim1=-2, dim2=-1)
        ori_match = torch.all(rel_rot_diag >= self.cfg.assembled_ori_bound, dim=-1)
        return torch.any(pos_match & ori_match, dim=1)

    def _reset_rigid_object_to_default(self, asset, env_ids: torch.Tensor) -> None:
        root_state = asset.data.default_root_state[env_ids].clone()
        root_state[:, :3] += self.scene.env_origins[env_ids]
        root_state[:, 7:] = 0.0
        asset.write_root_pose_to_sim(root_state[:, :7], env_ids)
        asset.write_root_velocity_to_sim(root_state[:, 7:], env_ids)
