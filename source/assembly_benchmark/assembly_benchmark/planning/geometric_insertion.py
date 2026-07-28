# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Analytic straight-line insertion paths without simulator dependencies."""

from __future__ import annotations

import math
from dataclasses import dataclass

from assembly_benchmark.assembly.beam import INSERTION_PATH_LENGTH

Vec3 = tuple[float, float, float]
Quat = tuple[float, float, float, float]

BEAM02_APPROACH_DISTANCE = INSERTION_PATH_LENGTH
"""Beam 0-to-2 pre-insertion clearance measured from the Fabrica geometry, in metres."""


def _normalize(vector: Vec3, label: str) -> Vec3:
    norm = math.sqrt(sum(value * value for value in vector))
    if not math.isfinite(norm) or norm <= 1.0e-12:
        raise ValueError(f"{label} must be finite and non-zero.")
    return tuple(value / norm for value in vector)  # type: ignore[return-value]


def _normalize_quaternion(quaternion: Quat) -> Quat:
    norm = math.sqrt(sum(value * value for value in quaternion))
    if not math.isfinite(norm) or norm <= 1.0e-12:
        raise ValueError("assembled_quaternion must be finite and non-zero.")
    return tuple(value / norm for value in quaternion)  # type: ignore[return-value]


def _rotate(quaternion: Quat, vector: Vec3) -> Vec3:
    """Rotate a vector with a normalized ``wxyz`` quaternion."""
    _, qx, qy, qz = quaternion
    vx, vy, vz = vector
    tx = 2.0 * (qy * vz - qz * vy)
    ty = 2.0 * (qz * vx - qx * vz)
    tz = 2.0 * (qx * vy - qy * vx)
    qw = quaternion[0]
    return (
        vx + qw * tx + qy * tz - qz * ty,
        vy + qw * ty + qz * tx - qx * tz,
        vz + qw * tz + qx * ty - qy * tx,
    )


@dataclass(frozen=True)
class LinearInsertionPath:
    """A translation-only insertion segment expressed from an assembled pose.

    ``progress=0`` is the pre-insertion pose and ``progress=1`` is the assembled pose. The approach
    axis is expressed in the assembled/socket frame, so the path remains valid when that frame is rotated.
    """

    assembled_position: Vec3
    assembled_quaternion: Quat = (1.0, 0.0, 0.0, 0.0)
    approach_axis: Vec3 = (0.0, 0.0, 1.0)
    approach_distance: float = BEAM02_APPROACH_DISTANCE

    def __post_init__(self) -> None:
        if len(self.assembled_position) != 3 or not all(math.isfinite(value) for value in self.assembled_position):
            raise ValueError("assembled_position must contain three finite values.")
        if not math.isfinite(self.approach_distance) or self.approach_distance <= 0.0:
            raise ValueError("approach_distance must be finite and positive.")
        object.__setattr__(self, "assembled_quaternion", _normalize_quaternion(self.assembled_quaternion))
        object.__setattr__(self, "approach_axis", _normalize(self.approach_axis, "approach_axis"))

    @property
    def world_approach_axis(self) -> Vec3:
        """Return the unit approach axis in the stationary frame."""
        return _rotate(self.assembled_quaternion, self.approach_axis)

    @property
    def start_position(self) -> Vec3:
        """Return the collision-free pre-insertion endpoint."""
        return tuple(
            goal + self.approach_distance * axis
            for goal, axis in zip(self.assembled_position, self.world_approach_axis, strict=True)
        )  # type: ignore[return-value]

    def position(self, progress: float) -> Vec3:
        """Interpolate a position on the segment for progress in ``[0, 1]``."""
        if not math.isfinite(progress) or not 0.0 <= progress <= 1.0:
            raise ValueError(f"progress must be finite and in [0, 1], got {progress}.")
        remaining_distance = (1.0 - progress) * self.approach_distance
        return tuple(
            goal + remaining_distance * axis
            for goal, axis in zip(self.assembled_position, self.world_approach_axis, strict=True)
        )  # type: ignore[return-value]

    def sample(self, waypoint_count: int) -> tuple[Vec3, ...]:
        """Sample equally spaced waypoints, including both endpoints."""
        if waypoint_count < 2:
            raise ValueError(f"waypoint_count must be at least 2, got {waypoint_count}.")
        return tuple(self.position(index / (waypoint_count - 1)) for index in range(waypoint_count))

    def distance(self, point: Vec3) -> float:
        """Return the Euclidean distance from a point to the finite insertion segment."""
        if len(point) != 3 or not all(math.isfinite(value) for value in point):
            raise ValueError("point must contain three finite values.")
        segment = tuple(start - goal for start, goal in zip(self.start_position, self.assembled_position, strict=True))
        relative = tuple(value - goal for value, goal in zip(point, self.assembled_position, strict=True))
        segment_norm_sq = sum(value * value for value in segment)
        projection = sum(value * axis for value, axis in zip(relative, segment, strict=True)) / segment_norm_sq
        projection = min(max(projection, 0.0), 1.0)
        closest = tuple(goal + projection * axis for goal, axis in zip(self.assembled_position, segment, strict=True))
        return math.sqrt(sum((value - target) ** 2 for value, target in zip(point, closest, strict=True)))

    def bounded_step_to_goal(self, point: Vec3, step_size: float) -> Vec3:
        """Return a straight-line step toward the assembled endpoint without overshoot."""
        if not math.isfinite(step_size) or step_size <= 0.0:
            raise ValueError("step_size must be finite and positive.")
        delta = tuple(goal - value for goal, value in zip(self.assembled_position, point, strict=True))
        distance = math.sqrt(sum(value * value for value in delta))
        if distance <= step_size:
            return delta  # type: ignore[return-value]
        return tuple(step_size * value / distance for value in delta)  # type: ignore[return-value]


__all__ = ["BEAM02_APPROACH_DISTANCE", "LinearInsertionPath"]
