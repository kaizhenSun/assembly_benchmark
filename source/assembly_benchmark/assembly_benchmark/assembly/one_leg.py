# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Default one_leg assembly specification."""

from __future__ import annotations

from pathlib import Path

from .parts import BaseTagPart, ObstaclePart, SquareTableLegPart, SquareTableTopPart
from .specs import AssemblyRelationSpec, AssemblySpec, AssemblyTargetPose


ONE_LEG_ASSET_ROOT = Path(__file__).resolve().parents[1] / "assets" / "furniture" / "one_leg"
ROT_Z_90 = (0.70710678, 0.0, 0.0, 0.70710678)


class OneLegAssemblySpec(AssemblySpec):
    """Default one_leg assembly: square tabletop plus one target leg."""

    def __init__(self):
        super().__init__(
            name="one_leg",
            asset_root=ONE_LEG_ASSET_ROOT,
            parts=(
                BaseTagPart(init_pos=(0.5015, 0.0, 0.775)),
                ObstaclePart(
                    scene_key="obstacle_front",
                    asset_name="obstacle_front",
                    prim_name="ObstacleFront",
                    urdf_rel_path="urdf/obstacle_front.urdf",
                    init_pos=(0.8815, 0.0, 0.79),
                    init_rot=ROT_Z_90,
                ),
                ObstaclePart(
                    scene_key="obstacle_right",
                    asset_name="obstacle_side",
                    prim_name="ObstacleRight",
                    urdf_rel_path="urdf/obstacle_side.urdf",
                    init_pos=(0.8065, -0.175, 0.79),
                    init_rot=ROT_Z_90,
                ),
                ObstaclePart(
                    scene_key="obstacle_left",
                    asset_name="obstacle_side",
                    prim_name="ObstacleLeft",
                    urdf_rel_path="urdf/obstacle_side.urdf",
                    init_pos=(0.8065, 0.175, 0.79),
                    init_rot=ROT_Z_90,
                ),
                SquareTableTopPart(),
                SquareTableLegPart(index=1, init_pos=(0.5715, -0.2, 0.79)),
                SquareTableLegPart(index=2, init_pos=(0.5715, -0.12, 0.79)),
                SquareTableLegPart(index=3, init_pos=(0.5715, 0.12, 0.79)),
                SquareTableLegPart(index=4, init_pos=(0.5715, 0.2, 0.79)),
            ),
            assembly_relations=(
                AssemblyRelationSpec(
                    parent="square_table_top",
                    child="square_table_leg4",
                    target_poses=(
                        AssemblyTargetPose(pos=(-0.05625, 0.046875, -0.05625)),
                        AssemblyTargetPose(pos=(0.05625, 0.046875, -0.05625)),
                        AssemblyTargetPose(pos=(-0.05625, 0.046875, 0.05625)),
                        AssemblyTargetPose(pos=(0.05625, 0.046875, 0.05625)),
                    ),
                    default_target_index=0,
                    pos_threshold=(0.010, 0.005, 0.010),
                    ori_bound=0.94,
                ),
            ),
        )


def make_one_leg_assembly() -> OneLegAssemblySpec:
    """Create the default one_leg assembly specification."""
    return OneLegAssemblySpec()
