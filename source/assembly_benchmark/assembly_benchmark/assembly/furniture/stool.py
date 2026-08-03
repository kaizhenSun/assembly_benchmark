# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Default stool assembly specification."""

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

ASSEMBLY_NAME = "stool"
ASSET_ROOT = assembly_asset_root(ASSEMBLY_NAME)
STOOL_ASSET_ROOT = ASSET_ROOT

BASE_TAG_POS: Vec3 = (0.5015, 0.0, 0.775)

STOOL_SEAT_INIT_POS: Vec3 = (0.7915, 0.0, 0.79)
STOOL_LEG_INIT_POSITIONS: tuple[Vec3, ...] = (
    (0.6815, -0.1, 0.7925),
    (0.6815, 0.0, 0.7925),
    (0.6815, 0.1, 0.7925),
)

STOOL_SEAT_ROT: Quat = (0.5, 0.5, -0.5, -0.5)
STOOL_LEG_ROT: Quat = (0.0, 0.7071067812, 0.7071067812, 0.0)

STOOL_LEG_TARGET_POSITIONS: tuple[Vec3, ...] = (
    (0.0, 0.045, 0.040542),
    (0.0351104019, 0.045, -0.020271),
    (-0.0351104019, 0.045, -0.020271),
)
STOOL_LEG_TARGET_QUATS: tuple[Quat, ...] = (
    (0.7071067812, 0.0, 0.7071067812, 0.0),
    (0.2588190451, 0.0, -0.9659258263, 0.0),
    (0.9659258263, 0.0, -0.2588190451, 0.0),
)


def make_stool_part(
    *,
    scene_key: str,
    prim_name: str,
    init_pos: Vec3,
    init_rot: Quat,
    density: float,
    tag_ids: tuple[int, ...],
) -> AssemblyPartSpec:
    """Create one dynamic stool part."""
    return make_dynamic_part(
        scene_key=scene_key,
        asset_name=scene_key,
        prim_name=prim_name,
        urdf_rel_path=f"urdf/stool/{scene_key}.urdf",
        init_pos=init_pos,
        init_rot=init_rot,
        density=density,
        tag_ids=tag_ids,
    )


PARTS: tuple[AssemblyPartSpec, ...] = (
    make_base_tag_part(init_pos=BASE_TAG_POS),
    make_stool_part(
        scene_key="stool_seat",
        prim_name="StoolSeat",
        init_pos=STOOL_SEAT_INIT_POS,
        init_rot=STOOL_SEAT_ROT,
        density=553.93,
        tag_ids=(143, 144, 145, 146),
    ),
    *(
        make_stool_part(
            scene_key=f"stool_leg{index}",
            prim_name=f"StoolLeg{index}",
            init_pos=pos,
            init_rot=STOOL_LEG_ROT,
            density=333.66,
            tag_ids=tuple(range(147 + (index - 1) * 4, 151 + (index - 1) * 4)),
        )
        for index, pos in enumerate(STOOL_LEG_INIT_POSITIONS, start=1)
    ),
)

RELATIONS: tuple[AssemblyRelationSpec, ...] = (
    *(
        make_relation(
            parent="stool_seat",
            child=f"stool_leg{index}",
            target_positions=STOOL_LEG_TARGET_POSITIONS,
            target_quats=STOOL_LEG_TARGET_QUATS,
            default_target_index=index - 1,
        )
        for index in range(1, 4)
    ),
)


class StoolAssemblySpec(AssemblySpec):
    """Default stool assembly: seat plus three legs."""

    def __init__(self):
        super().__init__(
            name=ASSEMBLY_NAME,
            asset_root=ASSET_ROOT,
            parts=PARTS,
            assembly_relations=RELATIONS,
        )


def make_stool_assembly() -> StoolAssemblySpec:
    """Create the default stool assembly specification."""
    return StoolAssemblySpec()
