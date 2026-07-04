# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Default round_table assembly specification."""

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


ASSEMBLY_NAME = "round_table"
ASSET_ROOT = assembly_asset_root(ASSEMBLY_NAME)
ROUND_TABLE_ASSET_ROOT = ASSET_ROOT

BASE_TAG_POS: Vec3 = (0.5015, 0.0, 0.775)

ROUND_TABLE_TOP_INIT_POS: Vec3 = (0.7415, 0.0, 0.776)
ROUND_TABLE_LEG_INIT_POS: Vec3 = (0.6015, 0.13, 0.79625)
ROUND_TABLE_BASE_INIT_POS: Vec3 = (0.6215, -0.12, 0.80375)

ROUND_TABLE_TOP_ROT: Quat = (0.7071067812, 0.0, 0.0, -0.7071067812)
ROUND_TABLE_LEG_ROT: Quat = (0.0, 0.7071067812, 0.7071067812, 0.0)
ROUND_TABLE_BASE_ROT: Quat = (0.7071067812, 0.0, 0.0, 0.7071067812)

ROUND_TABLE_LEG_TARGET_POSITIONS: tuple[Vec3, ...] = ((0.0, 0.0, 0.044375),)
ROUND_TABLE_LEG_TARGET_QUATS: tuple[Quat, ...] = (
    (0.0308435646, 0.0308435646, -0.7064337722, -0.7064337722),
)
ROUND_TABLE_BASE_TARGET_POSITIONS: tuple[Vec3, ...] = ((0.0, 0.053125, 0.0),)
ROUND_TABLE_BASE_TARGET_QUATS: tuple[Quat, ...] = (
    (0.5, -0.5, 0.5, 0.5),
)


def make_round_table_part(
    *,
    scene_key: str,
    prim_name: str,
    init_pos: Vec3,
    init_rot: Quat,
    density: float,
    tag_ids: tuple[int, ...],
) -> AssemblyPartSpec:
    """Create one dynamic round-table part."""
    return make_dynamic_part(
        scene_key=scene_key,
        asset_name=scene_key,
        prim_name=prim_name,
        urdf_rel_path=f"urdf/round_table/{scene_key}.urdf",
        init_pos=init_pos,
        init_rot=init_rot,
        density=density,
        tag_ids=tag_ids,
    )


PARTS: tuple[AssemblyPartSpec, ...] = (
    make_base_tag_part(init_pos=BASE_TAG_POS),
    make_round_table_part(
        scene_key="round_table_top",
        prim_name="RoundTableTop",
        init_pos=ROUND_TABLE_TOP_INIT_POS,
        init_rot=ROUND_TABLE_TOP_ROT,
        density=472.34,
        tag_ids=tuple(range(24, 32)),
    ),
    make_round_table_part(
        scene_key="round_table_leg",
        prim_name="RoundTableLeg",
        init_pos=ROUND_TABLE_LEG_INIT_POS,
        init_rot=ROUND_TABLE_LEG_ROT,
        density=414.52,
        tag_ids=(32, 33, 34, 35),
    ),
    make_round_table_part(
        scene_key="round_table_base",
        prim_name="RoundTableBase",
        init_pos=ROUND_TABLE_BASE_INIT_POS,
        init_rot=ROUND_TABLE_BASE_ROT,
        density=533.11,
        tag_ids=tuple(range(40, 48)),
    ),
)

RELATIONS: tuple[AssemblyRelationSpec, ...] = (
    make_relation(
        parent="round_table_top",
        child="round_table_leg",
        target_positions=ROUND_TABLE_LEG_TARGET_POSITIONS,
        target_quats=ROUND_TABLE_LEG_TARGET_QUATS,
    ),
    make_relation(
        parent="round_table_leg",
        child="round_table_base",
        target_positions=ROUND_TABLE_BASE_TARGET_POSITIONS,
        target_quats=ROUND_TABLE_BASE_TARGET_QUATS,
    ),
)


class RoundTableAssemblySpec(AssemblySpec):
    """Default round_table assembly: top, center leg, and base."""

    def __init__(self):
        super().__init__(
            name=ASSEMBLY_NAME,
            asset_root=ASSET_ROOT,
            parts=PARTS,
            assembly_relations=RELATIONS,
        )


def make_round_table_assembly() -> RoundTableAssemblySpec:
    """Create the default round_table assembly specification."""
    return RoundTableAssemblySpec()
