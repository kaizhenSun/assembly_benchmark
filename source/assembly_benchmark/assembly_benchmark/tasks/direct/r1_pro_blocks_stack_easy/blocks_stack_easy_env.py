# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from collections.abc import Sequence

import torch

from isaaclab.envs import DirectRLEnv
from isaaclab.utils.math import subtract_frame_transforms

from assembly_benchmark.controllers import (
    BimanualDifferentialIKController,
    BimanualJointPositionController,
)

from .blocks_stack_easy_env_cfg import R1ProBlocksStackEasyEnvCfg


class R1ProBlocksStackEasyEnv(DirectRLEnv):
    """R1 Pro BlocksStackEasy task shell migrated from GalaxeaManipSim."""

    cfg: R1ProBlocksStackEasyEnvCfg

    def __init__(
        self, cfg: R1ProBlocksStackEasyEnvCfg, render_mode: str | None = None, **kwargs
    ):
        super().__init__(cfg, render_mode, **kwargs)

        if self.cfg.control_mode == "joint":
            self.controller = BimanualJointPositionController(
                robot=self.robot,
                left_arm_joint_names=self.cfg.left_arm_joint_names,
                right_arm_joint_names=self.cfg.right_arm_joint_names,
                left_gripper_joint_names=self.cfg.left_gripper_joint_names,
                right_gripper_joint_names=self.cfg.right_gripper_joint_names,
                arm_action_scale=self.cfg.arm_action_scale,
                gripper_min=self.cfg.gripper_min,
                gripper_max=self.cfg.gripper_max,
            )
        elif self.cfg.control_mode == "ik":
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
            )
        else:
            raise ValueError(f"Unsupported R1 Pro control mode: {self.cfg.control_mode}")

        self.controlled_joint_ids = self.controller.joint_ids
        self.left_ee_body_idx = self.robot.find_bodies(self.cfg.left_ee_link_name)[0][0]
        self.right_ee_body_idx = self.robot.find_bodies(self.cfg.right_ee_link_name)[0][0]
        self.actions = torch.zeros((self.num_envs, self.cfg.action_space), device=self.device)
        self.joint_targets = self.robot.data.default_joint_pos[:, self.controlled_joint_ids].clone()

        block_half_size = 0.5 * float(self.cfg.scene.block1.spawn.size[2])
        table_surface_z = (
            float(self.cfg.scene.tabletop.init_state.pos[2])
            + 0.5 * float(self.cfg.scene.tabletop.spawn.size[2])
        )
        block1_init_pos = torch.tensor(
            self.cfg.scene.block1.init_state.pos, dtype=torch.float32, device=self.device
        )
        block2_init_pos = torch.tensor(
            self.cfg.scene.block2.init_state.pos, dtype=torch.float32, device=self.device
        )
        target_xy = 0.5 * (block1_init_pos[:2] + block2_init_pos[:2])
        block1_target_z = torch.tensor(
            table_surface_z + block_half_size, dtype=torch.float32, device=self.device
        )
        block2_target_z = torch.tensor(
            table_surface_z + 3.0 * block_half_size, dtype=torch.float32, device=self.device
        )
        self.block_target_positions = torch.stack(
            (
                torch.stack((target_xy[0], target_xy[1], block1_target_z)),
                torch.stack((target_xy[0], target_xy[1], block2_target_z)),
            ),
            dim=0,
        )
        target_quat = torch.tensor((1.0, 0.0, 0.0, 0.0), dtype=torch.float32, device=self.device)
        self.block_target_poses = torch.cat(
            (
                self.block_target_positions,
                target_quat.repeat(2, 1),
            ),
            dim=-1,
        )
        self.success_tolerance = torch.tensor(
            self.cfg.success_tolerance, dtype=torch.float32, device=self.device
        )

    def _setup_scene(self) -> None:
        self.robot = self.scene["robot"]
        self.block1 = self.scene["block1"]
        self.block2 = self.scene["block2"]
        self.sim.set_camera_view(eye=(2.2, 1.6, 1.7), target=(0.7, 0.0, 0.9))

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self.actions = actions.clone()
        self.joint_targets = self.controller.compute(self.actions)

    def _apply_action(self) -> None:
        self.robot.set_joint_position_target(self.joint_targets, joint_ids=self.controlled_joint_ids)

    def _get_observations(self) -> dict:
        left_ee_pose_b, right_ee_pose_b = self._get_ee_poses_in_root_frame()
        block1_pose = self._object_pose_in_env_frame(self.block1)
        block2_pose = self._object_pose_in_env_frame(self.block2)
        target_poses = self.block_target_poses.flatten().repeat(self.num_envs, 1)
        obs = torch.cat(
            (
                self.robot.data.joint_pos[:, self.controlled_joint_ids],
                self.robot.data.joint_vel[:, self.controlled_joint_ids],
                left_ee_pose_b,
                right_ee_pose_b,
                block1_pose,
                block2_pose,
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

        self._reset_rigid_object_to_default(self.block1, env_ids)
        self._reset_rigid_object_to_default(self.block2, env_ids)

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

    def _success(self) -> torch.Tensor:
        block1_pos = self._object_pose_in_env_frame(self.block1)[:, :3]
        block2_pos = self._object_pose_in_env_frame(self.block2)[:, :3]
        block1_success = torch.all(
            torch.abs(block1_pos - self.block_target_positions[0]) < self.success_tolerance,
            dim=-1,
        )
        block2_success = torch.all(
            torch.abs(block2_pos - self.block_target_positions[1]) < self.success_tolerance,
            dim=-1,
        )
        return block1_success & block2_success

    def _reset_rigid_object_to_default(self, asset, env_ids: torch.Tensor) -> None:
        root_state = asset.data.default_root_state[env_ids].clone()
        root_state[:, :3] += self.scene.env_origins[env_ids]
        root_state[:, 7:] = 0.0
        asset.write_root_pose_to_sim(root_state[:, :7], env_ids)
        asset.write_root_velocity_to_sim(root_state[:, 7:], env_ids)
