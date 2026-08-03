# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Dependency-light batched geometry for Fabrica specialist policies."""

from __future__ import annotations

import torch


def build_asymmetric_observations(
    nominal_error: torch.Tensor,
    true_error: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build Fabrica's 3D actor observation and 6D asymmetric critic state."""
    if nominal_error.shape != true_error.shape or nominal_error.shape[-1] != 3:
        raise ValueError(
            "nominal_error and true_error must have the same shape ending in 3, "
            f"got {nominal_error.shape} and {true_error.shape}."
        )
    return nominal_error, torch.cat((nominal_error, true_error), dim=-1)


def normalize_vectors(vectors: torch.Tensor) -> torch.Tensor:
    """Normalize finite vectors while mapping zero vectors to zero."""
    if vectors.shape[-1] != 3:
        raise ValueError(f"vectors must end in three coordinates, got {vectors.shape}.")
    norms = torch.linalg.vector_norm(vectors, dim=-1, keepdim=True)
    return torch.where(norms > 1.0e-12, vectors / norms.clamp_min(1.0e-12), torch.zeros_like(vectors))


def clip_vector_norm(vectors: torch.Tensor, max_norm: float) -> torch.Tensor:
    """Clip vectors to a maximum norm while leaving zero vectors unchanged."""
    if max_norm <= 0.0:
        raise ValueError(f"max_norm must be positive, got {max_norm}.")
    vector_norm = torch.linalg.vector_norm(vectors, dim=-1, keepdim=True)
    scale = torch.clamp(max_norm / vector_norm.clamp_min(1.0e-12), max=1.0)
    return vectors * scale


def path_frame_rotation(path_goal: torch.Tensor, path_preinsert: torch.Tensor) -> torch.Tensor:
    """Return rotations mapping each goal-directed path axis onto negative Z."""
    if path_goal.shape != path_preinsert.shape or path_goal.ndim != 2 or path_goal.shape[-1] != 3:
        raise ValueError("path_goal and path_preinsert must both have shape (N, 3).")
    direction = normalize_vectors(path_goal - path_preinsert)
    if torch.any(torch.linalg.vector_norm(direction, dim=-1) < 0.5):
        raise ValueError("Fabrica paths must have distinct goal and pre-insertion points.")

    target = torch.zeros_like(direction)
    target[:, 2] = -1.0
    cross = torch.linalg.cross(direction, target, dim=-1)
    cosine = torch.sum(direction * target, dim=-1)
    sine = torch.linalg.vector_norm(cross, dim=-1)

    skew = torch.zeros((direction.shape[0], 3, 3), dtype=direction.dtype, device=direction.device)
    skew[:, 0, 1] = -cross[:, 2]
    skew[:, 0, 2] = cross[:, 1]
    skew[:, 1, 0] = cross[:, 2]
    skew[:, 1, 2] = -cross[:, 0]
    skew[:, 2, 0] = -cross[:, 1]
    skew[:, 2, 1] = cross[:, 0]
    identity = torch.eye(3, dtype=direction.dtype, device=direction.device).expand(direction.shape[0], -1, -1)
    factor = ((1.0 - cosine) / sine.square().clamp_min(1.0e-12)).view(-1, 1, 1)
    general = identity + skew + torch.bmm(skew, skew) * factor

    seed_x = torch.zeros_like(direction)
    seed_x[:, 0] = 1.0
    seed_y = torch.zeros_like(direction)
    seed_y[:, 1] = 1.0
    seed = torch.where((direction[:, :1].abs() > 0.9), seed_y, seed_x)
    opposite_axis = normalize_vectors(torch.linalg.cross(direction, seed, dim=-1))
    opposite = 2.0 * opposite_axis.unsqueeze(-1) * opposite_axis.unsqueeze(-2) - identity
    parallel = sine <= 1.0e-6
    parallel_rotation = torch.where((cosine >= 0.0).view(-1, 1, 1), identity, opposite)
    return torch.where(parallel.view(-1, 1, 1), parallel_rotation, general)


def do_deltapos_path_transform(
    delta_position: torch.Tensor,
    path_goal: torch.Tensor,
    path_preinsert: torch.Tensor,
) -> torch.Tensor:
    """Transform world-frame deltas into Fabrica's path-aligned frame."""
    rotation = path_frame_rotation(path_goal, path_preinsert)
    return torch.bmm(rotation, delta_position.unsqueeze(-1)).squeeze(-1)


def undo_deltapos_path_transform(
    delta_path: torch.Tensor,
    path_goal: torch.Tensor,
    path_preinsert: torch.Tensor,
) -> torch.Tensor:
    """Transform path-aligned deltas back into the world frame."""
    rotation = path_frame_rotation(path_goal, path_preinsert)
    return torch.bmm(rotation.transpose(1, 2), delta_path.unsqueeze(-1)).squeeze(-1)


def scale_path_observation(delta_path: torch.Tensor, path_scale: torch.Tensor) -> torch.Tensor:
    """Apply Fabrica's 20 mm reference-length normalization to path-frame errors."""
    if delta_path.shape[-1] != 3 or path_scale.shape != delta_path.shape[:-1]:
        raise ValueError("delta_path must end in 3 and path_scale must match its batch shape.")
    scaled = delta_path.clone()
    scaled[..., 2] /= path_scale.clamp_min(1.0e-12)
    return scaled


def preprocess_fabrica_actions(
    policy_actions_path: torch.Tensor,
    plug_position: torch.Tensor,
    nominal_goal_position: torch.Tensor,
    path_goal: torch.Tensor,
    path_preinsert: torch.Tensor,
    path_scale: torch.Tensor,
    position_action_scale: float,
) -> torch.Tensor:
    """Combine Fabrica policy residuals with its analytic unit-vector insertion action."""
    if position_action_scale <= 0.0:
        raise ValueError(f"position_action_scale must be positive, got {position_action_scale}.")
    if policy_actions_path.shape != plug_position.shape or plug_position.shape != nominal_goal_position.shape:
        raise ValueError("actions, plug positions, and nominal goals must have identical (N, 3) shapes.")
    residual_path = policy_actions_path.clone()
    residual_path[:, 2] *= path_scale
    residual_world = undo_deltapos_path_transform(residual_path, path_goal, path_preinsert)
    analytic_world = normalize_vectors(nominal_goal_position - plug_position)
    return (residual_world + analytic_world) * position_action_scale


def dense_insertion_reward(
    true_error: torch.Tensor,
    reward_scale: float,
    distance_cap: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return Fabrica's capped negative-linear reward and true-error norm."""
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
    """Latch insertion success and path-deviation outcomes per environment."""
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
    """Project batched points onto finite segments and return normalized progress."""
    segment = segment_ends - segment_starts
    relative = points - segment_starts
    progress = torch.sum(relative * segment, dim=-1, keepdim=True) / torch.sum(
        segment * segment, dim=-1, keepdim=True
    ).clamp_min(1.0e-12)
    progress = progress.clamp(0.0, 1.0)
    projected_points = segment_starts + progress * segment
    return projected_points, progress.squeeze(-1)


def project_points_to_paths(points: torch.Tensor, paths: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Project each point onto a batched polyline and return arc-length progress."""
    if points.ndim != 2 or points.shape[-1] != 3 or paths.ndim != 3 or paths.shape[0] != points.shape[0]:
        raise ValueError("points must have shape (N, 3) and paths must have shape (N, L, 3).")
    if paths.shape[1] < 2:
        raise ValueError("Each Fabrica path must contain at least two points.")
    starts = paths[:, :-1]
    segments = paths[:, 1:] - starts
    lengths = torch.linalg.vector_norm(segments, dim=-1)
    relative = points.unsqueeze(1) - starts
    factors = torch.sum(relative * segments, dim=-1) / lengths.square().clamp_min(1.0e-12)
    factors = factors.clamp(0.0, 1.0)
    projections = starts + factors.unsqueeze(-1) * segments
    distances = torch.linalg.vector_norm(projections - points.unsqueeze(1), dim=-1)
    segment_index = torch.argmin(distances, dim=-1)
    batch_index = torch.arange(points.shape[0], device=points.device)
    closest = projections[batch_index, segment_index]
    cumulative = torch.cat((torch.zeros_like(lengths[:, :1]), torch.cumsum(lengths, dim=-1)), dim=-1)
    arc = (
        cumulative[batch_index, segment_index]
        + factors[batch_index, segment_index] * lengths[batch_index, segment_index]
    )
    progress = arc / cumulative[:, -1].clamp_min(1.0e-12)
    return closest, progress.clamp(0.0, 1.0)


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
    next_waypoint = projected_points + remaining_distance.clamp(max=max_step) * segment_direction
    return next_waypoint - points


__all__ = [
    "build_asymmetric_observations",
    "clip_vector_norm",
    "dense_insertion_reward",
    "do_deltapos_path_transform",
    "episode_timeout_mask",
    "invalid_state_mask",
    "next_linear_waypoint_step",
    "normalize_vectors",
    "path_frame_rotation",
    "pose_drift",
    "preprocess_fabrica_actions",
    "project_points_to_paths",
    "project_points_to_segments",
    "scale_path_observation",
    "undo_deltapos_path_transform",
    "update_latched_outcomes",
]
