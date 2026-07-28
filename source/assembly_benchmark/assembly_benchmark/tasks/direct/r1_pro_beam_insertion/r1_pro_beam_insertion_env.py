# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""R1 Pro left-arm Beam 0-to-2 geometric insertion environment."""

from __future__ import annotations

from collections.abc import Sequence

import torch

from pxr import UsdPhysics

from isaaclab.envs import DirectRLEnv
from isaaclab.utils.math import (
    combine_frame_transforms,
    quat_apply,
    quat_apply_inverse,
    subtract_frame_transforms,
)

from assembly_benchmark.controllers import BimanualDifferentialIKController

from .r1_pro_beam_insertion_env_cfg import R1ProBeam02InsertionEnvCfg
from .task_geometry import (
    build_asymmetric_observations,
    clip_vector_norm,
    dense_insertion_reward,
    episode_timeout_mask,
    invalid_state_mask,
    next_linear_waypoint_step,
    pose_drift,
    project_points_to_segments,
    update_latched_outcomes,
)


class R1ProBeam02InsertionEnv(DirectRLEnv):
    """Insert Fabrica Beam plug 0 into socket 2 with R1 Pro's torso and left arm.

    The policy outputs a 3D residual in the socket-aligned path frame. A zero residual follows an
    analytic straight-line baseline from the pre-insertion pose to the nominal assembled pose.
    """

    cfg: R1ProBeam02InsertionEnvCfg

    def __init__(self, cfg: R1ProBeam02InsertionEnvCfg, render_mode: str | None = None, **kwargs) -> None:
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
            control_dt=self.step_dt,
        )
        if len(self.controller.joint_ids) != 22:
            raise RuntimeError(
                "Beam02 bimanual whole-body IK requires torso, both arms, and both grippers, "
                f"got {len(self.controller.joint_ids)}."
            )

        self.plug_body_idx = self._find_unique_body(self.cfg.plug_body_name)
        self.right_ee_body_idx = self._find_unique_body(self.cfg.right_ee_link_name)
        self.controlled_joint_ids = self.controller.joint_ids
        self.actions = torch.zeros((self.num_envs, *self.single_action_space.shape), device=self.device)
        self.controller_actions = torch.zeros((self.num_envs, self.controller.action_dim), device=self.device)
        self.joint_targets = self.robot.data.default_joint_pos[:, self.controlled_joint_ids].clone()

        self._assembled_target_pos = self._repeat_tensor(self.cfg.assembled_target_pos)
        self._assembled_target_quat = self._repeat_tensor(self.cfg.assembled_target_quat)
        self._approach_axis_socket = self._repeat_tensor(self.cfg.approach_axis_socket)
        self._socket_position_noise_limit = self._repeat_tensor(self.cfg.socket_position_noise)
        self._gripper_to_plug_pos = self._repeat_tensor(self.cfg.gripper_to_plug_pos)
        self._gripper_to_plug_quat = self._repeat_tensor(self.cfg.gripper_to_plug_quat)
        self.path_scale = self.cfg.approach_distance / self.cfg.path_reference_distance

        self.nominal_socket_pose_w = self._default_socket_pose_w()
        self.actual_socket_pose_w = self.nominal_socket_pose_w.clone()
        self.socket_position_noise = torch.zeros((self.num_envs, 3), device=self.device)
        self.nominal_goal_pose_w = self._goal_pose_from_socket(self.nominal_socket_pose_w)
        self.actual_goal_pose_w = self.nominal_goal_pose_w.clone()
        world_approach_axis = quat_apply(self.nominal_socket_pose_w[:, 3:7], self._approach_axis_socket)
        self.path_start_pos_w = self.nominal_goal_pose_w[:, :3] + self.cfg.approach_distance * world_approach_axis

        self.insertion_successes = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.deviations = torch.zeros_like(self.insertion_successes)
        self.invalid_states = torch.zeros_like(self.insertion_successes)
        self.keypoint_distance = torch.zeros(self.num_envs, device=self.device)
        self.physical_error_distance = torch.zeros(self.num_envs, device=self.device)
        self.deviation_distance = torch.zeros(self.num_envs, device=self.device)
        self.axial_progress = torch.zeros(self.num_envs, device=self.device)
        self.max_cross_track_distance = torch.zeros(self.num_envs, device=self.device)
        self.right_ee_position_drift = torch.zeros(self.num_envs, device=self.device)
        self.right_ee_orientation_drift = torch.zeros(self.num_envs, device=self.device)
        self.max_right_ee_position_drift = torch.zeros(self.num_envs, device=self.device)
        self.max_right_ee_orientation_drift = torch.zeros(self.num_envs, device=self.device)
        self._last_keypoint_reward = torch.zeros(self.num_envs, device=self.device)

    def _setup_scene(self) -> None:
        self.robot = self.scene["robot"]
        self.socket = self.scene[self.cfg.socket_scene_key]
        self._author_socket_table_collision_filter()
        self.sim.set_camera_view(eye=(1.35, 1.10, 1.25), target=(0.55, 0.30, 0.80))

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self.invalid_states |= invalid_state_mask(actions)
        safe_actions = torch.nan_to_num(actions, nan=0.0, posinf=1.0, neginf=-1.0)
        self.actions = safe_actions.clamp(-1.0, 1.0).clone()
        plug_pose_w = self._plug_pose_w()
        baseline_step_w = self._geometric_baseline_step(plug_pose_w[:, :3])

        residual_step_path = self.actions * self.cfg.position_action_scale
        residual_step_w = quat_apply(self.nominal_socket_pose_w[:, 3:7], residual_step_path)

        command_step_w = clip_vector_norm(
            baseline_step_w + residual_step_w,
            self.cfg.position_action_scale,
        )
        plug_target_pos_w = plug_pose_w[:, :3] + command_step_w
        plug_target_pose_w = torch.cat((plug_target_pos_w, self.nominal_goal_pose_w[:, 3:7]), dim=-1)
        gripper_target_pose_b = self._plug_target_to_gripper_pose_in_robot_root(plug_target_pose_w)
        self.controller_actions.zero_()
        self.controller_actions[:, 0:7] = gripper_target_pose_b
        self.controller_actions[:, 7] = self.cfg.left_gripper_action
        # A zero right pose is the controller's per-environment hold-current-pose sentinel.
        self.controller_actions[:, 15] = self.cfg.right_gripper_action
        self.joint_targets = self.controller.compute(self.controller_actions)

    def _apply_action(self) -> None:
        self.robot.set_joint_position_target(self.joint_targets, joint_ids=self.controlled_joint_ids)

    def _get_observations(self) -> dict[str, torch.Tensor]:
        self._refresh_actual_goal_pose()
        plug_pos_w = self._plug_pose_w()[:, :3]
        nominal_delta_path = self._delta_to_path_frame(self.nominal_goal_pose_w[:, :3] - plug_pos_w)
        actual_delta_path = self._delta_to_path_frame(self.actual_goal_pose_w[:, :3] - plug_pos_w)
        policy_observation, critic_state = build_asymmetric_observations(nominal_delta_path, actual_delta_path)
        return {"policy": policy_observation, "critic": critic_state}

    def _get_rewards(self) -> torch.Tensor:
        self.extras.pop("log", None)
        self._refresh_actual_goal_pose()
        reward = self._update_task_metrics()
        self._last_keypoint_reward.copy_(reward)
        return reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        # Match Fabrica for task outcomes, but terminate unrecoverable non-finite simulator states.
        self.invalid_states |= self._current_invalid_state()
        terminated = self.invalid_states.clone()
        # DirectRLEnv increments episode_length_buf before calling this hook, so this yields exactly
        # cfg.episode_steps policy actions instead of truncating one action early.
        time_out = episode_timeout_mask(self.episode_length_buf, self.max_episode_length)
        return terminated, time_out

    def _reset_idx(self, env_ids: Sequence[int] | None) -> None:
        if env_ids is None:
            env_ids_tensor = torch.arange(self.num_envs, dtype=torch.long, device=self.device)
        elif isinstance(env_ids, torch.Tensor):
            env_ids_tensor = env_ids.to(device=self.device, dtype=torch.long)
        else:
            env_ids_tensor = torch.as_tensor(env_ids, dtype=torch.long, device=self.device)

        self._log_completed_episodes(env_ids_tensor)
        super()._reset_idx(env_ids_tensor)

        robot_joint_pos = self.robot.data.default_joint_pos[env_ids_tensor].clone()
        robot_joint_vel = self.robot.data.default_joint_vel[env_ids_tensor].clone()
        robot_root_state = self.robot.data.default_root_state[env_ids_tensor].clone()
        robot_root_state[:, :3] += self.scene.env_origins[env_ids_tensor]
        self.robot.write_root_pose_to_sim(robot_root_state[:, :7], env_ids_tensor)
        self.robot.write_root_velocity_to_sim(robot_root_state[:, 7:], env_ids_tensor)
        self.robot.write_joint_state_to_sim(robot_joint_pos, robot_joint_vel, None, env_ids_tensor)
        self.robot.set_joint_position_target(robot_joint_pos, env_ids=env_ids_tensor)
        self.controller.reset(env_ids_tensor)

        noise_path = 2.0 * torch.rand((len(env_ids_tensor), 3), device=self.device) - 1.0
        noise_path *= self._socket_position_noise_limit[env_ids_tensor]
        socket_pose_w = self.nominal_socket_pose_w[env_ids_tensor].clone()
        socket_pose_w[:, :3] += quat_apply(socket_pose_w[:, 3:7], noise_path)
        socket_velocity_w = torch.zeros((len(env_ids_tensor), 6), device=self.device)
        self.socket.write_root_pose_to_sim(socket_pose_w, env_ids_tensor)
        self.socket.write_root_velocity_to_sim(socket_velocity_w, env_ids_tensor)

        self.actual_socket_pose_w[env_ids_tensor] = socket_pose_w
        self.actual_goal_pose_w[env_ids_tensor] = self._goal_pose_from_socket(socket_pose_w)
        self.socket_position_noise[env_ids_tensor] = noise_path
        self.actions[env_ids_tensor] = 0.0
        self.controller_actions[env_ids_tensor] = 0.0
        self.joint_targets[env_ids_tensor] = robot_joint_pos[:, self.controlled_joint_ids]
        self.insertion_successes[env_ids_tensor] = False
        self.deviations[env_ids_tensor] = False
        self.invalid_states[env_ids_tensor] = False
        self.keypoint_distance[env_ids_tensor] = 0.0
        self.physical_error_distance[env_ids_tensor] = 0.0
        self.deviation_distance[env_ids_tensor] = 0.0
        self.axial_progress[env_ids_tensor] = 0.0
        self.max_cross_track_distance[env_ids_tensor] = 0.0
        self.right_ee_position_drift[env_ids_tensor] = 0.0
        self.right_ee_orientation_drift[env_ids_tensor] = 0.0
        self.max_right_ee_position_drift[env_ids_tensor] = 0.0
        self.max_right_ee_orientation_drift[env_ids_tensor] = 0.0
        self._last_keypoint_reward[env_ids_tensor] = 0.0

    def _default_socket_pose_w(self) -> torch.Tensor:
        pose_w = self.socket.data.default_root_state[:, :7].clone()
        pose_w[:, :3] += self.scene.env_origins
        return pose_w

    def _goal_pose_from_socket(self, socket_pose_w: torch.Tensor) -> torch.Tensor:
        env_count = socket_pose_w.shape[0]
        target_pos = self._assembled_target_pos[:env_count]
        target_quat = self._assembled_target_quat[:env_count]
        goal_pos, goal_quat = combine_frame_transforms(
            socket_pose_w[:, :3], socket_pose_w[:, 3:7], target_pos, target_quat
        )
        return torch.cat((goal_pos, goal_quat), dim=-1)

    def _refresh_actual_goal_pose(self) -> None:
        self.actual_socket_pose_w.copy_(self.socket.data.root_pose_w)
        self.actual_goal_pose_w.copy_(self._goal_pose_from_socket(self.actual_socket_pose_w))

    def _plug_target_to_gripper_pose_in_robot_root(self, plug_target_pose_w: torch.Tensor) -> torch.Tensor:
        plug_to_gripper_pos, plug_to_gripper_quat = subtract_frame_transforms(
            self._gripper_to_plug_pos, self._gripper_to_plug_quat
        )
        gripper_target_pos_w, gripper_target_quat_w = combine_frame_transforms(
            plug_target_pose_w[:, :3],
            plug_target_pose_w[:, 3:7],
            plug_to_gripper_pos,
            plug_to_gripper_quat,
        )
        gripper_target_pos_b, gripper_target_quat_b = subtract_frame_transforms(
            self.robot.data.root_pose_w[:, :3],
            self.robot.data.root_pose_w[:, 3:7],
            gripper_target_pos_w,
            gripper_target_quat_w,
        )
        return torch.cat((gripper_target_pos_b, gripper_target_quat_b), dim=-1)

    def _delta_to_path_frame(self, delta_w: torch.Tensor) -> torch.Tensor:
        delta_path = quat_apply_inverse(self.nominal_socket_pose_w[:, 3:7], delta_w)
        delta_path[:, 2] /= self.path_scale
        return delta_path

    def _geometric_baseline_step(self, plug_pos_w: torch.Tensor) -> torch.Tensor:
        return next_linear_waypoint_step(
            plug_pos_w,
            self.path_start_pos_w,
            self.nominal_goal_pose_w[:, :3],
            self.cfg.position_action_scale,
        )

    def _project_onto_nominal_path(self, plug_pos_w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return project_points_to_segments(
            plug_pos_w,
            self.path_start_pos_w,
            self.nominal_goal_pose_w[:, :3],
        )

    def _update_task_metrics(self) -> torch.Tensor:
        plug_pos_w = self._plug_pose_w()[:, :3]
        true_error_w = self.actual_goal_pose_w[:, :3] - plug_pos_w
        # Reward and success thresholds are physical metres; axial normalization is observation-only.
        reward, self.keypoint_distance = dense_insertion_reward(
            true_error_w,
            self.cfg.keypoint_reward_scale,
            self.cfg.keypoint_distance_cap,
        )
        self.physical_error_distance = self.keypoint_distance.clone()

        closest, progress = self._project_onto_nominal_path(plug_pos_w)
        self.axial_progress = torch.nan_to_num(progress, nan=0.0).clamp(0.0, 1.0)
        cross_track = torch.linalg.vector_norm(plug_pos_w - closest, dim=-1)
        self.deviation_distance = torch.nan_to_num(
            cross_track,
            nan=self.cfg.deviation_distance_threshold,
            posinf=self.cfg.deviation_distance_threshold,
            neginf=self.cfg.deviation_distance_threshold,
        )
        self.max_cross_track_distance = torch.maximum(
            self.max_cross_track_distance,
            self.deviation_distance,
        )
        self.insertion_successes, self.deviations = update_latched_outcomes(
            self.insertion_successes,
            self.deviations,
            self.keypoint_distance,
            self.deviation_distance,
            self.cfg.success_distance_threshold,
            self.cfg.deviation_distance_threshold,
        )
        self._update_right_ee_drift()
        return reward

    def _update_right_ee_drift(self) -> None:
        root_pose_w = self.robot.data.root_pose_w
        right_ee_pose_w = self.robot.data.body_pose_w[:, self.right_ee_body_idx]
        right_ee_pos_b, right_ee_quat_b = subtract_frame_transforms(
            root_pose_w[:, :3],
            root_pose_w[:, 3:7],
            right_ee_pose_w[:, :3],
            right_ee_pose_w[:, 3:7],
        )
        initialized = self.controller.right_pose_command_initialized
        target_pose_b = self.controller.right_pose_command
        current_pose_b = torch.cat((right_ee_pos_b, right_ee_quat_b), dim=-1)
        position_drift, orientation_drift = pose_drift(current_pose_b, target_pose_b)
        self.right_ee_position_drift = torch.where(initialized, position_drift, 0.0)
        self.right_ee_orientation_drift = torch.where(initialized, orientation_drift, 0.0)
        self.max_right_ee_position_drift = torch.maximum(self.max_right_ee_position_drift, self.right_ee_position_drift)
        self.max_right_ee_orientation_drift = torch.maximum(
            self.max_right_ee_orientation_drift, self.right_ee_orientation_drift
        )

    def _current_invalid_state(self) -> torch.Tensor:
        state_tensors = (
            self.robot.data.root_pose_w,
            self.robot.data.joint_pos,
            self.robot.data.joint_vel,
            self._plug_pose_w(),
            self.socket.data.root_pose_w,
        )
        return invalid_state_mask(*state_tensors)

    def _log_completed_episodes(self, env_ids: torch.Tensor) -> None:
        if hasattr(self, "extras"):
            self.extras.pop("log", None)
        if not hasattr(self, "insertion_successes"):
            return
        completed_ids = env_ids[self.episode_length_buf[env_ids] > 0]
        if len(completed_ids) == 0:
            return

        keypoint_reward = self._last_keypoint_reward[completed_ids].mean()
        keypoint_distance = self.keypoint_distance[completed_ids].mean()
        physical_error_distance = self.physical_error_distance[completed_ids].mean()
        insertion_successes = self.insertion_successes[completed_ids].float().mean()
        deviation = self.deviations[completed_ids].float().mean()
        invalid_state = self.invalid_states[completed_ids].float().mean()
        axial_progress = self.axial_progress[completed_ids].mean()
        max_cross_track = self.max_cross_track_distance[completed_ids].max()
        right_ee_position_drift = self.right_ee_position_drift[completed_ids].mean()
        right_ee_orientation_drift = self.right_ee_orientation_drift[completed_ids].mean()
        max_right_ee_position_drift = self.max_right_ee_position_drift[completed_ids].max()
        max_right_ee_orientation_drift = self.max_right_ee_orientation_drift[completed_ids].max()
        self.extras.update(
            {
                "keypoint_reward": keypoint_reward,
                "keypoint_distance": keypoint_distance,
                "insertion_successes": insertion_successes,
                "deviation": deviation,
                "invalid_state": invalid_state,
                "log": {
                    "Metrics/keypoint_reward": keypoint_reward,
                    "Metrics/keypoint_distance": keypoint_distance,
                    "Metrics/final_true_error": keypoint_distance,
                    "Metrics/final_physical_error": physical_error_distance,
                    "Metrics/insertion_successes": insertion_successes,
                    "Metrics/deviation": deviation,
                    "Metrics/invalid_state": invalid_state,
                    "Metrics/axial_progress": axial_progress,
                    "Metrics/max_cross_track_distance": max_cross_track,
                    "Metrics/right_ee_position_drift": right_ee_position_drift,
                    "Metrics/right_ee_orientation_drift": right_ee_orientation_drift,
                    "Metrics/max_right_ee_position_drift": max_right_ee_position_drift,
                    "Metrics/max_right_ee_orientation_drift": max_right_ee_orientation_drift,
                },
            }
        )

    def _author_socket_table_collision_filter(self) -> None:
        source_env_path = self.scene.env_prim_paths[0]
        source_socket_path = f"{source_env_path}/{self.cfg.socket_prim_name}"
        source_tabletop_path = f"{source_env_path}/{self.cfg.tabletop_prim_path_suffix}"
        source_socket = self.scene.stage.GetPrimAtPath(source_socket_path)
        source_tabletop = self.scene.stage.GetPrimAtPath(source_tabletop_path)
        if not source_socket.IsValid() or not source_tabletop.IsValid():
            raise RuntimeError(
                "Cannot author Beam02 socket-to-table collision filtering: "
                f"socket='{source_socket_path}', tabletop='{source_tabletop_path}'."
            )

        filtered_pairs = UsdPhysics.FilteredPairsAPI.Apply(source_socket).CreateFilteredPairsRel()
        filtered_pairs.SetTargets([source_tabletop.GetPath()])

        for env_path in self.scene.env_prim_paths:
            socket_path = f"{env_path}/{self.cfg.socket_prim_name}"
            tabletop_path = f"{env_path}/{self.cfg.tabletop_prim_path_suffix}"
            socket = self.scene.stage.GetPrimAtPath(socket_path)
            tabletop = self.scene.stage.GetPrimAtPath(tabletop_path)
            if not socket.IsValid() or not tabletop.IsValid():
                raise RuntimeError(
                    "Beam02 collision-filter prim did not propagate to "
                    f"socket='{socket_path}', tabletop='{tabletop_path}'."
                )
            targets = UsdPhysics.FilteredPairsAPI(socket).GetFilteredPairsRel().GetTargets()
            if tabletop.GetPath() not in targets:
                raise RuntimeError(
                    "Beam02 socket-to-table collision filter did not propagate to "
                    f"socket='{socket_path}', tabletop='{tabletop_path}'."
                )

    def _plug_pose_w(self) -> torch.Tensor:
        return self.robot.data.body_pose_w[:, self.plug_body_idx]

    def _find_unique_body(self, body_name: str) -> int:
        body_ids, body_names = self.robot.find_bodies(body_name, preserve_order=True)
        if len(body_ids) != 1 or list(body_names) != [body_name]:
            raise RuntimeError(
                f"Expected robot body '{body_name}' exactly once, got ids {body_ids} and names {body_names}."
            )
        return body_ids[0]

    def _repeat_tensor(self, values: Sequence[float]) -> torch.Tensor:
        return torch.tensor(values, dtype=torch.float32, device=self.device).repeat(self.num_envs, 1)


__all__ = ["R1ProBeam02InsertionEnv"]
