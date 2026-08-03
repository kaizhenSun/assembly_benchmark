# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Fabrica fixed-plug specialist environment for one complete assembly graph."""

from __future__ import annotations

from collections.abc import Sequence

import torch

from pxr import UsdPhysics

from isaaclab.envs import DirectRLEnv
from isaaclab.utils.math import combine_frame_transforms, quat_apply, subtract_frame_transforms

from assembly_benchmark.assembly import load_fabrica_assembly_plan
from assembly_benchmark.controllers import SingleArmDifferentialIKController

from .fabrica_algo_utils import (
    build_asymmetric_observations,
    dense_insertion_reward,
    do_deltapos_path_transform,
    episode_timeout_mask,
    invalid_state_mask,
    preprocess_fabrica_actions,
    project_points_to_paths,
    scale_path_observation,
    update_latched_outcomes,
)
from .fabrica_fixplug_task_assemble_cfg import FabricaFixPlugTaskAssembleCfg


class FabricaFixPlugTaskAssemble(DirectRLEnv):
    """Train one specialist over all fixed-plug relations in an assembly.

    Environments are assigned a relation by ``env_id % relation_count``. Each episode performs one
    insertion, matching Fabrica's specialist task rather than sequentially building the assembly.
    """

    cfg: FabricaFixPlugTaskAssembleCfg

    def __init__(self, cfg: FabricaFixPlugTaskAssembleCfg, render_mode: str | None = None, **kwargs) -> None:
        super().__init__(cfg, render_mode, **kwargs)

        self.assembly_plan = load_fabrica_assembly_plan(self.cfg.assembly_name)
        self.relation_count = len(self.assembly_plan.relations)
        self.relation_ids = torch.arange(self.num_envs, device=self.device) % self.relation_count
        self.relation_keys = self.assembly_plan.relation_keys

        self.controller = SingleArmDifferentialIKController(
            robot=self.robot,
            arm_joint_names=self.cfg.arm_joint_names,
            gripper_joint_names=self.cfg.gripper_joint_names,
            ee_link_name=self.cfg.ee_link_name,
            ik_link_name=self.cfg.ik_link_name,
            gripper_min=self.cfg.gripper_min,
            gripper_max=self.cfg.gripper_max,
            num_envs=self.num_envs,
            device=self.device,
            control_dt=self.step_dt,
            gripper_joint_closed_positions=self.cfg.gripper_joint_closed_positions,
            gripper_joint_open_positions=self.cfg.gripper_joint_open_positions,
        )
        expected_joint_count = len(self.cfg.arm_joint_names) + len(self.cfg.gripper_joint_names)
        if len(self.controller.joint_ids) != expected_joint_count:
            raise RuntimeError(
                f"Fabrica Piper IK requires {expected_joint_count} controlled joints, "
                f"got {len(self.controller.joint_ids)}."
            )

        self.plug_body_idx = self._find_unique_body(self.cfg.plug_body_name)
        self.controlled_joint_ids = self.controller.joint_ids
        self.actions = torch.zeros((self.num_envs, *self.single_action_space.shape), device=self.device)
        self.controller_actions = torch.zeros((self.num_envs, self.controller.action_dim), device=self.device)
        self.joint_targets = self.robot.data.default_joint_pos[:, self.controlled_joint_ids].clone()

        self._preassembly_joint_pos = self._relation_tensor("piper_preassembly_joint_pos")
        self._gripper_opening = self._relation_tensor("piper_gripper_opening").squeeze(-1)
        self._gripper_action = self._opening_to_action(self._gripper_opening)
        self._gripper_to_plug_pos = self._relation_tensor("piper_gripper_to_plug_pos")
        self._gripper_to_plug_quat = self._relation_tensor("piper_gripper_to_plug_quat")
        self._socket_position_noise_limit = self._repeat_tensor(self.cfg.socket_position_noise)

        target_positions, target_quaternions = self._relation_target_poses()
        self._assembled_target_pos = target_positions[self.relation_ids]
        self._assembled_target_quat = target_quaternions[self.relation_ids]

        self.nominal_socket_pose_w = self._default_socket_pose_w()
        self.actual_socket_pose_w = self.nominal_socket_pose_w.clone()
        self.socket_position_noise = torch.zeros((self.num_envs, 3), device=self.device)
        self.nominal_goal_pose_w = self._goal_pose_from_socket(self.nominal_socket_pose_w)
        self.actual_goal_pose_w = self.nominal_goal_pose_w.clone()

        self.disassembly_path_w = self._build_disassembly_paths(self.nominal_goal_pose_w)
        self.insertion_path_w = torch.flip(self.disassembly_path_w, dims=(1,))
        self.path_goal_pos_w = self.insertion_path_w[:, -1]
        self.path_preinsert_pos_w = self.insertion_path_w[:, 0]
        relation_path_lengths = torch.tensor(
            [relation.path_length for relation in self.assembly_plan.relations],
            dtype=torch.float32,
            device=self.device,
        )
        self.path_scale = relation_path_lengths[self.relation_ids] / self.cfg.path_reference_distance

        self.insertion_successes = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.deviations = torch.zeros_like(self.insertion_successes)
        self.invalid_states = torch.zeros_like(self.insertion_successes)
        self.keypoint_distance = torch.zeros(self.num_envs, device=self.device)
        self.physical_error_distance = torch.zeros(self.num_envs, device=self.device)
        self.deviation_distance = torch.zeros(self.num_envs, device=self.device)
        self.axial_progress = torch.zeros(self.num_envs, device=self.device)
        self.max_cross_track_distance = torch.zeros(self.num_envs, device=self.device)
        self._last_keypoint_reward = torch.zeros(self.num_envs, device=self.device)

    def _setup_scene(self) -> None:
        self.robot = self.scene["robot"]
        self.socket = self.scene[self.cfg.socket_scene_key]
        self._author_socket_collision_filters()
        self.sim.set_camera_view(eye=(1.10, 0.95, 1.35), target=(0.50, 0.30, 0.88))

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self.invalid_states |= invalid_state_mask(actions)
        self.actions = torch.nan_to_num(actions, nan=0.0, posinf=1.0, neginf=-1.0).clamp(-1.0, 1.0)
        plug_pose_w = self._plug_pose_w()
        command_step_w = preprocess_fabrica_actions(
            self.actions,
            plug_pose_w[:, :3],
            self.nominal_goal_pose_w[:, :3],
            self.path_goal_pos_w,
            self.path_preinsert_pos_w,
            self.path_scale,
            self.cfg.position_action_scale,
        )
        plug_target_pose_w = torch.cat((plug_pose_w[:, :3] + command_step_w, self.nominal_goal_pose_w[:, 3:7]), dim=-1)
        gripper_target_pose_b = self._plug_target_to_gripper_pose_in_robot_root(plug_target_pose_w)

        self.controller_actions.zero_()
        self.controller_actions[:, :7] = gripper_target_pose_b
        self.controller_actions[:, 7] = self._gripper_action
        self.joint_targets = self.controller.compute(self.controller_actions)

    def _apply_action(self) -> None:
        self.robot.set_joint_position_target(self.joint_targets, joint_ids=self.controlled_joint_ids)

    def _get_observations(self) -> dict[str, torch.Tensor]:
        self._refresh_actual_goal_pose()
        plug_pos_w = self._plug_pose_w()[:, :3]
        nominal_error = self._scaled_path_error(self.nominal_goal_pose_w[:, :3] - plug_pos_w)
        true_error = self._scaled_path_error(self.actual_goal_pose_w[:, :3] - plug_pos_w)
        policy_observation, critic_state = build_asymmetric_observations(nominal_error, true_error)
        return {"policy": policy_observation, "critic": critic_state}

    def _get_rewards(self) -> torch.Tensor:
        self.extras.pop("log", None)
        self._refresh_actual_goal_pose()
        reward = self._update_task_metrics()
        self._last_keypoint_reward.copy_(reward)
        return reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        self.invalid_states |= self._current_invalid_state()
        terminated = self.invalid_states.clone()
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
        robot_joint_pos[:, self.controller.arm_joint_ids] = self._preassembly_joint_pos[env_ids_tensor]
        robot_joint_pos[:, self.controller.gripper_joint_ids] = self.controller.gripper_targets_from_opening(
            self._gripper_opening[env_ids_tensor]
        )
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
        self.actual_goal_pose_w[env_ids_tensor] = self._goal_pose_from_socket(socket_pose_w, env_ids_tensor)
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
        self._last_keypoint_reward[env_ids_tensor] = 0.0

    def _default_socket_pose_w(self) -> torch.Tensor:
        pose_w = self.socket.data.default_root_state[:, :7].clone()
        pose_w[:, :3] += self.scene.env_origins
        return pose_w

    def _goal_pose_from_socket(self, socket_pose_w: torch.Tensor, env_ids: torch.Tensor | None = None) -> torch.Tensor:
        if env_ids is None:
            target_pos = self._assembled_target_pos
            target_quat = self._assembled_target_quat
        else:
            target_pos = self._assembled_target_pos[env_ids]
            target_quat = self._assembled_target_quat[env_ids]
        goal_pos, goal_quat = combine_frame_transforms(
            socket_pose_w[:, :3], socket_pose_w[:, 3:7], target_pos, target_quat
        )
        return torch.cat((goal_pos, goal_quat), dim=-1)

    def _refresh_actual_goal_pose(self) -> None:
        self.actual_socket_pose_w.copy_(self.socket.data.root_pose_w)
        self.actual_goal_pose_w.copy_(self._goal_pose_from_socket(self.actual_socket_pose_w))

    def _build_disassembly_paths(self, goal_pose_w: torch.Tensor) -> torch.Tensor:
        relation_paths = torch.tensor(
            [[pose[:3] for pose in relation.disassembly_path] for relation in self.assembly_plan.relations],
            dtype=torch.float32,
            device=self.device,
        )
        local_paths = relation_paths[self.relation_ids]
        goal_pos = goal_pose_w[:, None, :3].expand_as(local_paths)
        goal_quat = goal_pose_w[:, None, 3:7].expand(-1, local_paths.shape[1], -1)
        return goal_pos + quat_apply(goal_quat.reshape(-1, 4), local_paths.reshape(-1, 3)).reshape_as(local_paths)

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

    def _scaled_path_error(self, error_w: torch.Tensor) -> torch.Tensor:
        error_path = do_deltapos_path_transform(error_w, self.path_goal_pos_w, self.path_preinsert_pos_w)
        return scale_path_observation(error_path, self.path_scale)

    def _update_task_metrics(self) -> torch.Tensor:
        plug_pos_w = self._plug_pose_w()[:, :3]
        true_error_w = self.actual_goal_pose_w[:, :3] - plug_pos_w
        scaled_true_error = self._scaled_path_error(true_error_w)
        reward, self.keypoint_distance = dense_insertion_reward(
            scaled_true_error, self.cfg.keypoint_reward_scale, self.cfg.keypoint_distance_cap
        )
        self.physical_error_distance = torch.linalg.vector_norm(true_error_w, dim=-1)

        closest, progress = project_points_to_paths(plug_pos_w, self.insertion_path_w)
        self.axial_progress = torch.nan_to_num(progress, nan=0.0).clamp(0.0, 1.0)
        cross_track = torch.linalg.vector_norm(plug_pos_w - closest, dim=-1)
        self.deviation_distance = torch.nan_to_num(
            cross_track,
            nan=self.cfg.deviation_distance_threshold,
            posinf=self.cfg.deviation_distance_threshold,
            neginf=self.cfg.deviation_distance_threshold,
        )
        self.max_cross_track_distance = torch.maximum(self.max_cross_track_distance, self.deviation_distance)
        self.insertion_successes, self.deviations = update_latched_outcomes(
            self.insertion_successes,
            self.deviations,
            self.keypoint_distance,
            self.deviation_distance,
            self.cfg.success_distance_threshold,
            self.cfg.deviation_distance_threshold,
        )
        return reward

    def _current_invalid_state(self) -> torch.Tensor:
        return invalid_state_mask(
            self.robot.data.root_pose_w,
            self.robot.data.joint_pos,
            self.robot.data.joint_vel,
            self._plug_pose_w(),
            self.socket.data.root_pose_w,
        )

    def _log_completed_episodes(self, env_ids: torch.Tensor) -> None:
        if hasattr(self, "extras"):
            self.extras.pop("log", None)
        if not hasattr(self, "insertion_successes"):
            return
        completed_ids = env_ids[self.episode_length_buf[env_ids] > 0]
        if len(completed_ids) == 0:
            return

        log = self._metric_log(completed_ids)
        for relation_id, relation_key in enumerate(self.relation_keys):
            relation_env_ids = completed_ids[self.relation_ids[completed_ids] == relation_id]
            if len(relation_env_ids) == 0:
                continue
            prefix = f"Relations/{relation_key}"
            relation_log = self._metric_log(relation_env_ids)
            log.update({f"{prefix}/{name.removeprefix('Metrics/')}": value for name, value in relation_log.items()})
        self.extras.update(
            {
                "keypoint_reward": log["Metrics/keypoint_reward"],
                "keypoint_distance": log["Metrics/final_true_error"],
                "insertion_successes": log["Metrics/insertion_successes"],
                "deviation": log["Metrics/deviation"],
                "invalid_state": log["Metrics/invalid_state"],
                "log": log,
            }
        )

    def _metric_log(self, env_ids: torch.Tensor) -> dict[str, torch.Tensor]:
        return {
            "Metrics/keypoint_reward": self._last_keypoint_reward[env_ids].mean(),
            "Metrics/keypoint_distance": self.keypoint_distance[env_ids].mean(),
            "Metrics/final_true_error": self.keypoint_distance[env_ids].mean(),
            "Metrics/final_physical_error": self.physical_error_distance[env_ids].mean(),
            "Metrics/insertion_successes": self.insertion_successes[env_ids].float().mean(),
            "Metrics/deviation": self.deviations[env_ids].float().mean(),
            "Metrics/invalid_state": self.invalid_states[env_ids].float().mean(),
            "Metrics/axial_progress": self.axial_progress[env_ids].mean(),
            "Metrics/max_cross_track_distance": self.max_cross_track_distance[env_ids].max(),
        }

    def _author_socket_collision_filters(self) -> None:
        for env_path in self.scene.env_prim_paths:
            socket_path = f"{env_path}/{self.cfg.socket_prim_path_suffix}"
            tabletop_path = f"{env_path}/{self.cfg.tabletop_prim_path_suffix}"
            robot_body_paths = [
                f"{env_path}/Robot/{body_name}" for body_name in self.cfg.socket_filtered_robot_body_names
            ]
            socket = self.scene.stage.GetPrimAtPath(socket_path)
            tabletop = self.scene.stage.GetPrimAtPath(tabletop_path)
            robot_bodies = [self.scene.stage.GetPrimAtPath(path) for path in robot_body_paths]
            if not socket.IsValid() or not tabletop.IsValid() or not all(body.IsValid() for body in robot_bodies):
                raise RuntimeError(
                    "Cannot author Fabrica socket collision filtering: "
                    f"socket='{socket_path}', tabletop='{tabletop_path}', robot_bodies={robot_body_paths}."
                )
            targets = [tabletop.GetPath(), *(body.GetPath() for body in robot_bodies)]
            UsdPhysics.FilteredPairsAPI.Apply(socket).CreateFilteredPairsRel().SetTargets(targets)
            UsdPhysics.FilteredPairsAPI.Apply(tabletop).CreateFilteredPairsRel().SetTargets(
                [body.GetPath() for body in robot_bodies]
            )

    def _relation_tensor(self, attribute: str) -> torch.Tensor:
        values = [getattr(relation, attribute) for relation in self.assembly_plan.relations]
        tensor = torch.as_tensor(values, dtype=torch.float32, device=self.device)
        if tensor.ndim == 1:
            tensor = tensor.unsqueeze(-1)
        return tensor[self.relation_ids]

    def _relation_target_poses(self) -> tuple[torch.Tensor, torch.Tensor]:
        positions = []
        quaternions = []
        assembly = self.assembly_plan
        from assembly_benchmark.assembly import make_assembly

        spec = make_assembly(assembly.assembly_name)
        relation_by_pair = {(relation.child, relation.parent): relation for relation in spec.assembly_relations}
        for relation in assembly.relations:
            plug_key = assembly.part_scene_key(relation.plug_part_id)
            socket_key = assembly.part_scene_key(relation.socket_part_id)
            target = relation_by_pair[(plug_key, socket_key)].default_target_pose
            positions.append(target.pos)
            quaternions.append(target.quat)
        return (
            torch.tensor(positions, dtype=torch.float32, device=self.device),
            torch.tensor(quaternions, dtype=torch.float32, device=self.device),
        )

    def _opening_to_action(self, opening: torch.Tensor) -> torch.Tensor:
        return 2.0 * (opening - self.cfg.gripper_min) / (self.cfg.gripper_max - self.cfg.gripper_min) - 1.0

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


__all__ = ["FabricaFixPlugTaskAssemble"]
