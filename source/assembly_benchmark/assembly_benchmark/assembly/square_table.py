# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Default square_table assembly specification."""

from __future__ import annotations

from .specs import (
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


ASSEMBLY_NAME = "square_table"
ASSET_ROOT = assembly_asset_root(ASSEMBLY_NAME)
SQUARE_TABLE_ASSET_ROOT = ASSET_ROOT

BASE_TAG_POS: Vec3 = (0.5015, 0.0, 0.775)

SQUARE_TABLE_TOP_ROT: Quat = (0.5, 0.5, -0.5, -0.5)
SQUARE_TABLE_LEG_ROT: Quat = (0.5, 0.5, 0.5, -0.5)

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


def make_square_table_part(
    *,
    scene_key: str,
    prim_name: str,
    init_pos: Vec3,
    init_rot: Quat,
    density: float,
    tag_ids: tuple[int, ...],
) -> AssemblyPartSpec:
    """Create one dynamic square-table part."""
    return make_dynamic_part(
        scene_key=scene_key,
        asset_name=scene_key,
        prim_name=prim_name,
        urdf_rel_path=f"urdf/square_table/{scene_key}.urdf",
        init_pos=init_pos,
        init_rot=init_rot,
        density=density,
        tag_ids=tag_ids,
    )


def make_square_table_top_part() -> AssemblyPartSpec:
    """Create the square table top."""
    return make_square_table_part(
        scene_key="square_table_top",
        prim_name="SquareTableTop",
        init_pos=TABLE_TOP_INIT_POS,
        init_rot=SQUARE_TABLE_TOP_ROT,
        density=498.68,
        tag_ids=(4, 5, 6, 7),
    )


def make_square_table_leg_part(index: int, init_pos: Vec3) -> AssemblyPartSpec:
    """Create one square table leg."""
    scene_key = f"square_table_leg{index}"
    tag_start = 8 + (index - 1) * 4
    return make_square_table_part(
        scene_key=scene_key,
        prim_name=f"SquareTableLeg{index}",
        init_pos=init_pos,
        init_rot=SQUARE_TABLE_LEG_ROT,
        density=369.98,
        tag_ids=tuple(range(tag_start, tag_start + 4)),
    )


PARTS: tuple[AssemblyPartSpec, ...] = (
    make_base_tag_part(init_pos=BASE_TAG_POS),
    make_square_table_top_part(),
    *(
        make_square_table_leg_part(index=index, init_pos=pos)
        for index, pos in enumerate(LEG_INIT_POSITIONS, start=1)
    ),
)

RELATIONS: tuple[AssemblyRelationSpec, ...] = (
    *(
        make_relation(
            parent="square_table_top",
            child=f"square_table_leg{index}",
            target_positions=TABLE_LEG_TARGET_POSITIONS,
            default_target_index=index - 1,
        )
        for index in range(1, 5)
    ),
)


class SquareTableAssemblySpec(AssemblySpec):
    """Default square_table assembly: tabletop plus four legs."""

    def __init__(self):
        super().__init__(
            name=ASSEMBLY_NAME,
            asset_root=ASSET_ROOT,
            parts=PARTS,
            assembly_relations=RELATIONS,
        )


def make_square_table_assembly() -> SquareTableAssemblySpec:
    """Create the default square_table assembly specification."""
    return SquareTableAssemblySpec()
