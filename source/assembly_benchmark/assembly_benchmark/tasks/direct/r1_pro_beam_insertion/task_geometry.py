# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Batched geometric operations for the Beam02 insertion task."""

from __future__ import annotations

import torch


def build_asymmetric_observations(
    nominal_error: torch.Tensor,
    true_error: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build the 3D actor observation and 6D asymmetric critic state."""
    if nominal_error.shape != true_error.shape or nominal_error.shape[-1] != 3:
        raise ValueError(
            "nominal_error and true_error must have the same shape ending in 3, "
            f"got {nominal_error.shape} and {true_error.shape}."
        )
    return nominal_error, torch.cat((nominal_error, true_error), dim=-1)


def clip_vector_norm(vectors: torch.Tensor, max_norm: float) -> torch.Tensor:
    """Clip vectors to a maximum norm while leaving zero vectors unchanged."""
    if max_norm <= 0.0:
        raise ValueError(f"max_norm must be positive, got {max_norm}.")
    vector_norm = torch.linalg.vector_norm(vectors, dim=-1, keepdim=True)
    scale = torch.clamp(max_norm / vector_norm.clamp_min(1.0e-12), max=1.0)
    return vectors * scale


def dense_insertion_reward(
    true_error: torch.Tensor,
    reward_scale: float,
    distance_cap: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return Fabrica's capped negative-linear reward and its true-error norm."""
    if true_error.shape[-1] != 3:
        raise ValueError(f"true_error must end in three coordinates, got {true_error.shape}.")
    if reward_scale < 0.0 or distance_cap <= 0.0:
        raise ValueError("reward_scale must be non-negative and distance_cap must be positive.")

    distance = torch.linalg.vector_norm(true_error, dim=-1)
    safe_distance = torch.nan_to_num(distance, nan=distance_cap, posinf=distance_cap, neginf=distance_cap)
    return -reward_scale * safe_distance.clamp(max=distance_cap), safe_distance


def pose_drift(current_pose: torch.Tensor, target_pose: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return position and shortest-path quaternion drift for batched wxyz poses."""
    position_drift = torch.linalg.vector_norm(current_pose[:, :3] - target_pose[:, :3], dim=-1)
    current_quat = current_pose[:, 3:7]
    target_quat = target_pose[:, 3:7]
    current_quat = current_quat / torch.linalg.vector_norm(current_quat, dim=-1, keepdim=True).clamp_min(1.0e-8)
    target_quat = target_quat / torch.linalg.vector_norm(target_quat, dim=-1, keepdim=True).clamp_min(1.0e-8)
    quaternion_dot = torch.sum(current_quat * target_quat, dim=-1).abs().clamp(0.0, 1.0)
    orientation_drift = 2.0 * torch.acos(quaternion_dot)
    return position_drift, orientation_drift


def episode_timeout_mask(episode_lengths: torch.Tensor, max_episode_steps: int) -> torch.Tensor:
    """Return time-out flags after exactly ``max_episode_steps`` policy actions."""
    if max_episode_steps <= 0:
        raise ValueError(f"max_episode_steps must be positive, got {max_episode_steps}.")
    return episode_lengths >= max_episode_steps


def update_latched_outcomes(
    success_latch: torch.Tensor,
    deviation_latch: torch.Tensor,
    true_error_distance: torch.Tensor,
    cross_track_distance: torch.Tensor,
    success_threshold: float,
    deviation_threshold: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Latch insertion success and path-deviation outcomes independently per environment."""
    if success_threshold <= 0.0 or deviation_threshold <= 0.0:
        raise ValueError("success_threshold and deviation_threshold must be positive.")
    expected_shape = success_latch.shape
    if (
        deviation_latch.shape != expected_shape
        or true_error_distance.shape != expected_shape
        or cross_track_distance.shape != expected_shape
    ):
        raise ValueError("All latch and distance tensors must have the same batch shape.")
    return (
        success_latch | (true_error_distance < success_threshold),
        deviation_latch | (cross_track_distance > deviation_threshold),
    )


def invalid_state_mask(*state_tensors: torch.Tensor) -> torch.Tensor:
    """Return one invalid flag per environment for a collection of batched states."""
    if not state_tensors:
        raise ValueError("At least one state tensor is required.")
    batch_size = state_tensors[0].shape[0]
    if any(state.ndim < 2 or state.shape[0] != batch_size for state in state_tensors):
        raise ValueError("State tensors must share a batch dimension and contain at least one state dimension.")
    valid = torch.ones(batch_size, dtype=torch.bool, device=state_tensors[0].device)
    for state in state_tensors:
        valid &= torch.all(torch.isfinite(state).reshape(batch_size, -1), dim=-1)
    return ~valid


def project_points_to_segments(
    points: torch.Tensor,
    segment_starts: torch.Tensor,
    segment_ends: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Project batched points onto finite segments and return progress from start to end."""
    segment = segment_ends - segment_starts
    relative = points - segment_starts
    progress = torch.sum(relative * segment, dim=-1, keepdim=True) / torch.sum(
        segment * segment, dim=-1, keepdim=True
    ).clamp_min(1.0e-12)
    progress = progress.clamp(0.0, 1.0)
    projected_points = segment_starts + progress * segment
    return projected_points, progress.squeeze(-1)


def next_linear_waypoint_step(
    points: torch.Tensor,
    segment_starts: torch.Tensor,
    segment_ends: torch.Tensor,
    max_step: float,
) -> torch.Tensor:
    """Return the correction toward the next bounded waypoint on a finite segment."""
    projected_points, progress = project_points_to_segments(points, segment_starts, segment_ends)
    segment = segment_ends - segment_starts
    segment_length = torch.linalg.vector_norm(segment, dim=-1, keepdim=True)
    segment_direction = segment / segment_length.clamp_min(1.0e-12)
    remaining_distance = (1.0 - progress).unsqueeze(-1) * segment_length
    advance_distance = remaining_distance.clamp(max=max_step)
    next_waypoint = projected_points + advance_distance * segment_direction
    return next_waypoint - points


__all__ = [
    "build_asymmetric_observations",
    "clip_vector_norm",
    "dense_insertion_reward",
    "episode_timeout_mask",
    "invalid_state_mask",
    "next_linear_waypoint_step",
    "pose_drift",
    "project_points_to_segments",
    "update_latched_outcomes",
]
