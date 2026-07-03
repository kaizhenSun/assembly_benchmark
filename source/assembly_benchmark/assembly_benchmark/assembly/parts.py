# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Concrete assembly part specifications."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .specs import AssemblyPartSpec, Quat, Vec3


@dataclass(frozen=True)
class BaseTagPart(AssemblyPartSpec):
    """Base tag visual asset."""

    def __init__(self, init_pos: Vec3, init_rot: Quat = (1.0, 0.0, 0.0, 0.0)):
        object.__setattr__(self, "scene_key", "base_tag")
        object.__setattr__(self, "asset_name", "base_tag")
        object.__setattr__(self, "prim_name", "BaseTag")
        object.__setattr__(self, "urdf_rel_path", Path("urdf/base_tag.urdf"))
        object.__setattr__(self, "init_pos", init_pos)
        object.__setattr__(self, "init_rot", init_rot)
        object.__setattr__(self, "body_type", "visual")
        object.__setattr__(self, "mass", None)
        object.__setattr__(self, "observe", False)
        object.__setattr__(self, "reset", False)
        object.__setattr__(self, "tag_ids", (0, 1, 2, 3))
        object.__setattr__(self, "reset_footprint_xy", None)


@dataclass(frozen=True)
class ObstaclePart(AssemblyPartSpec):
    """Static obstacle asset."""

    def __init__(
        self,
        scene_key: str,
        asset_name: str,
        prim_name: str,
        urdf_rel_path: str,
        init_pos: Vec3,
        init_rot: Quat,
    ):
        object.__setattr__(self, "scene_key", scene_key)
        object.__setattr__(self, "asset_name", asset_name)
        object.__setattr__(self, "prim_name", prim_name)
        object.__setattr__(self, "urdf_rel_path", Path(urdf_rel_path))
        object.__setattr__(self, "init_pos", init_pos)
        object.__setattr__(self, "init_rot", init_rot)
        object.__setattr__(self, "body_type", "static")
        object.__setattr__(self, "mass", None)
        object.__setattr__(self, "observe", False)
        object.__setattr__(self, "reset", False)
        object.__setattr__(self, "tag_ids", ())
        object.__setattr__(self, "reset_footprint_xy", None)


@dataclass(frozen=True)
class SquareTableTopPart(AssemblyPartSpec):
    """Square-table top used by the default one_leg assembly."""

    half_width: float = 0.08125

    def __init__(self):
        object.__setattr__(self, "scene_key", "square_table_top")
        object.__setattr__(self, "asset_name", "square_table_top")
        object.__setattr__(self, "prim_name", "SquareTableTop")
        object.__setattr__(
            self,
            "urdf_rel_path",
            Path("urdf/square_table/square_table_top.urdf"),
        )
        object.__setattr__(self, "init_pos", (0.7415, 0.0, 0.790625))
        object.__setattr__(self, "init_rot", (0.5, 0.5, -0.5, -0.5))
        object.__setattr__(self, "body_type", "dynamic")
        object.__setattr__(self, "mass", 0.151)
        object.__setattr__(self, "observe", False)
        object.__setattr__(self, "reset", True)
        object.__setattr__(self, "tag_ids", (4, 5, 6, 7))
        object.__setattr__(self, "reset_footprint_xy", (0.1625, 0.1625))
        object.__setattr__(self, "half_width", 0.08125)


@dataclass(frozen=True)
class SquareTableLegPart(AssemblyPartSpec):
    """Square-table leg used by the default one_leg assembly."""

    half_width: float = 0.015

    def __init__(
        self,
        index: int,
        init_pos: Vec3,
        init_rot: Quat = (0.5, 0.5, 0.5, -0.5),
    ):
        scene_key = f"square_table_leg{index}"
        object.__setattr__(self, "scene_key", scene_key)
        object.__setattr__(self, "asset_name", scene_key)
        object.__setattr__(self, "prim_name", f"SquareTableLeg{index}")
        object.__setattr__(
            self,
            "urdf_rel_path",
            Path(f"urdf/square_table/{scene_key}.urdf"),
        )
        object.__setattr__(self, "init_pos", init_pos)
        object.__setattr__(self, "init_rot", init_rot)
        object.__setattr__(self, "body_type", "dynamic")
        object.__setattr__(self, "mass", 0.0231)
        object.__setattr__(self, "observe", False)
        object.__setattr__(self, "reset", True)
        tag_start = 8 + (index - 1) * 4
        object.__setattr__(self, "tag_ids", tuple(range(tag_start, tag_start + 4)))
        object.__setattr__(self, "reset_footprint_xy", (0.05, 0.0875))
        object.__setattr__(self, "half_width", 0.015)
