# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Default one_leg assembly specification."""

from __future__ import annotations

from ..specs import (
    AssemblyPartSpec,
    AssemblyRelationSpec,
    AssemblySpec,
    Quat,
    Vec3,
    assembly_asset_root,
    make_base_tag_part,
    make_dynamic_part,
    make_relation,
)

ASSEMBLY_NAME = "one_leg"
ASSET_ROOT = assembly_asset_root(ASSEMBLY_NAME)
ONE_LEG_ASSET_ROOT = ASSET_ROOT

SQUARE_TABLE_TOP_ROT: Quat = (0.5, 0.5, -0.5, -0.5)
SQUARE_TABLE_LEG_ROT: Quat = (0.5, 0.5, 0.5, -0.5)

BASE_TAG_POS: Vec3 = (0.5015, 0.0, 0.775)
TABLE_TOP_INIT_POS: Vec3 = (0.7415, 0.0, 0.790625)
LEG_INIT_POSITIONS: tuple[Vec3, ...] = (
    (0.5715, -0.2, 0.79),
    (0.5715, -0.12, 0.79),
    (0.5715, 0.12, 0.79),
    (0.5715, 0.2, 0.79),
)

TABLE_LEG_TARGET_POSITIONS: tuple[Vec3, ...] = (
    (-0.05625, 0.046875, -0.05625),
    (0.05625, 0.046875, -0.05625),
    (-0.05625, 0.046875, 0.05625),
    (0.05625, 0.046875, 0.05625),
)


def make_square_table_top_part() -> AssemblyPartSpec:
    """Create the square table top."""
    return make_dynamic_part(
        scene_key="square_table_top",
        asset_name="square_table_top",
        prim_name="SquareTableTop",
        urdf_rel_path="urdf/square_table/square_table_top.urdf",
        init_pos=TABLE_TOP_INIT_POS,
        init_rot=SQUARE_TABLE_TOP_ROT,
        mass=0.151,
        tag_ids=(4, 5, 6, 7),
        reset_footprint_xy=(0.1625, 0.1625),
    )


def make_square_table_leg_part(index: int, init_pos: Vec3) -> AssemblyPartSpec:
    """Create one square table leg."""
    scene_key = f"square_table_leg{index}"
    tag_start = 8 + (index - 1) * 4
    return make_dynamic_part(
        scene_key=scene_key,
        asset_name=scene_key,
        prim_name=f"SquareTableLeg{index}",
        urdf_rel_path=f"urdf/square_table/{scene_key}.urdf",
        init_pos=init_pos,
        init_rot=SQUARE_TABLE_LEG_ROT,
        # mass=0.531,
        mass=0.0231,
        tag_ids=tuple(range(tag_start, tag_start + 4)),
        reset_footprint_xy=(0.05, 0.0875),
    )


PARTS: tuple[AssemblyPartSpec, ...] = (
    make_base_tag_part(init_pos=BASE_TAG_POS),
    make_square_table_top_part(),
    *(make_square_table_leg_part(index=index, init_pos=pos) for index, pos in enumerate(LEG_INIT_POSITIONS, start=1)),
)

RELATIONS: tuple[AssemblyRelationSpec, ...] = (
    make_relation(
        parent="square_table_top",
        child="square_table_leg4",
        target_positions=TABLE_LEG_TARGET_POSITIONS,
    ),
)


class OneLegAssemblySpec(AssemblySpec):
    """Default one_leg assembly: square tabletop plus one target leg."""

    def __init__(self):
        super().__init__(
            name=ASSEMBLY_NAME,
            asset_root=ASSET_ROOT,
            parts=PARTS,
            assembly_relations=RELATIONS,
        )


def make_one_leg_assembly() -> OneLegAssemblySpec:
    """Create the default one_leg assembly specification."""
    return OneLegAssemblySpec()
