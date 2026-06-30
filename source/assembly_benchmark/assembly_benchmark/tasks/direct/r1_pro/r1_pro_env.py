# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from collections.abc import Sequence

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
from isaaclab.utils.math import subtract_frame_transforms

from assembly_benchmark.controllers import BimanualDifferentialIKController, BimanualJointPositionController

from .r1_pro_env_cfg import AssemblyR1ProEnvCfg


class AssemblyR1ProEnv(DirectRLEnv):
    """Direct R1 Pro smoke environment for validating assets and controllers."""

    cfg: AssemblyR1ProEnvCfg

    def __init__(self, cfg: AssemblyR1ProEnvCfg, render_mode: str | None = None, **kwargs):
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
                torso_joint_names=self.cfg.torso_joint_names,
                include_torso_in_ik=self.cfg.include_torso_in_ik,
            )
        else:
            raise ValueError(f"Unsupported R1 Pro control mode: {self.cfg.control_mode}")

        self.controlled_joint_ids = self.controller.joint_ids
        self.left_ee_body_idx = self.robot.find_bodies(self.cfg.left_ee_link_name)[0][0]
        self.right_ee_body_idx = self.robot.find_bodies(self.cfg.right_ee_link_name)[0][0]
        self.actions = torch.zeros((self.num_envs, self.cfg.action_space), device=self.device)
        self.joint_targets = self.robot.data.default_joint_pos[:, self.controlled_joint_ids].clone()

    def _setup_scene(self):
        self.robot = Articulation(self.cfg.robot_cfg)
        self.scene.clone_environments(copy_from_source=False)
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[])
        self.scene.articulations["robot"] = self.robot

        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)
        self.sim.set_camera_view(eye=(3.0, 3.0, 2.0), target=(0.0, 0.0, 0.8))

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self.actions = actions.clone()
        self.joint_targets = self.controller.compute(self.actions)

    def _apply_action(self) -> None:
        self.robot.set_joint_position_target(self.joint_targets, joint_ids=self.controlled_joint_ids)

    def _get_observations(self) -> dict:
        left_ee_pose_b, right_ee_pose_b = self._get_ee_poses_in_root_frame()
        obs = torch.cat(
            (
                self.robot.data.joint_pos[:, self.controlled_joint_ids],
                self.robot.data.joint_vel[:, self.controlled_joint_ids],
                left_ee_pose_b,
                right_ee_pose_b,
            ),
            dim=-1,
        )
        return {"policy": obs}

    def _get_rewards(self) -> torch.Tensor:
        action_penalty = torch.sum(torch.square(self.actions), dim=-1)
        joint_limit_penalty = self._joint_limit_penalty()
        return (
            torch.full((self.num_envs,), self.cfg.rew_scale_alive, device=self.device)
            + self.cfg.rew_scale_action_penalty * action_penalty
            + self.cfg.rew_scale_joint_limit * joint_limit_penalty
        )

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        finite_joint_state = torch.isfinite(self.robot.data.joint_pos).all(dim=1)
        finite_joint_state &= torch.isfinite(self.robot.data.joint_vel).all(dim=1)
        invalid_state = ~finite_joint_state
        return invalid_state, time_out

    def _reset_idx(self, env_ids: Sequence[int] | None):
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

    def _get_ee_poses_in_root_frame(self) -> tuple[torch.Tensor, torch.Tensor]:
        root_pose_w = self.robot.data.root_pose_w
        left_ee_pose_w = self.robot.data.body_pose_w[:, self.left_ee_body_idx]
        right_ee_pose_w = self.robot.data.body_pose_w[:, self.right_ee_body_idx]
        left_pos_b, left_quat_b = subtract_frame_transforms(
            root_pose_w[:, :3], root_pose_w[:, 3:7], left_ee_pose_w[:, :3], left_ee_pose_w[:, 3:7]
        )
        right_pos_b, right_quat_b = subtract_frame_transforms(
            root_pose_w[:, :3], root_pose_w[:, 3:7], right_ee_pose_w[:, :3], right_ee_pose_w[:, 3:7]
        )
        return torch.cat((left_pos_b, left_quat_b), dim=-1), torch.cat((right_pos_b, right_quat_b), dim=-1)

    def _joint_limit_penalty(self) -> torch.Tensor:
        joint_pos = self.robot.data.joint_pos[:, self.controlled_joint_ids]
        limits = self.robot.data.soft_joint_pos_limits[:, self.controlled_joint_ids]
        lower_violation = torch.clamp(limits[..., 0] - joint_pos, min=0.0)
        upper_violation = torch.clamp(joint_pos - limits[..., 1], min=0.0)
        return torch.sum(lower_violation + upper_violation, dim=-1)
