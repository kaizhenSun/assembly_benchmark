# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from collections.abc import Sequence

import torch

from isaaclab.envs import DirectRLEnv

from .one_leg_scene_env_cfg import R1ProOneLegSceneEnvCfg


class R1ProOneLegSceneEnv(DirectRLEnv):
    """Minimal environment that loads the R1 Pro FurnitureBench one_leg scene."""

    cfg: R1ProOneLegSceneEnvCfg

    def __init__(
        self, cfg: R1ProOneLegSceneEnvCfg, render_mode: str | None = None, **kwargs
    ):
        super().__init__(cfg, render_mode, **kwargs)
        self.actions = torch.zeros((self.num_envs, self.cfg.action_space), device=self.device)

    def _setup_scene(self) -> None:
        self.robot = self.scene["robot"]
        self.furniture_parts = (
            self.scene["square_table_top"],
            self.scene["square_table_leg1"],
            self.scene["square_table_leg2"],
            self.scene["square_table_leg3"],
            self.scene["square_table_leg4"],
        )
        self.sim.set_camera_view(eye=(2.2, 1.6, 1.7), target=(0.55, 0.0, 1.0))

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self.actions = actions.clone()

    def _apply_action(self) -> None:
        pass

    def _get_observations(self) -> dict:
        return {
            "policy": torch.zeros(
                (self.num_envs, self.cfg.observation_space), device=self.device
            )
        }

    def _get_rewards(self) -> torch.Tensor:
        return torch.zeros(self.num_envs, device=self.device)

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        terminated = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        return terminated, time_out

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

        for part in self.furniture_parts:
            self._reset_rigid_object_to_default(part, env_ids)

    def _reset_rigid_object_to_default(self, asset, env_ids: torch.Tensor) -> None:
        root_state = asset.data.default_root_state[env_ids].clone()
        root_state[:, :3] += self.scene.env_origins[env_ids]
        root_state[:, 7:] = 0.0
        asset.write_root_pose_to_sim(root_state[:, :7], env_ids)
        asset.write_root_velocity_to_sim(root_state[:, 7:], env_ids)
