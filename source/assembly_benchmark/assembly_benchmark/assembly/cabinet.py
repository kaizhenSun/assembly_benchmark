# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Default cabinet assembly specification."""

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


ASSEMBLY_NAME = "cabinet"
ASSET_ROOT = assembly_asset_root(ASSEMBLY_NAME)
CABINET_ASSET_ROOT = ASSET_ROOT

BASE_TAG_POS: Vec3 = (0.5015, 0.0, 0.775)

CABINET_BODY_INIT_POS: Vec3 = (0.7415, 0.0, 0.80375)
CABINET_DOOR_LEFT_INIT_POS: Vec3 = (0.5815, 0.12, 0.78625)
CABINET_DOOR_RIGHT_INIT_POS: Vec3 = (0.5815, 0.2, 0.78625)
CABINET_TOP_INIT_POS: Vec3 = (0.6515, -0.15, 0.79)

CABINET_BODY_ROT: Quat = (0.5, 0.5, 0.5, -0.5)
CABINET_DOOR_ROT: Quat = (0.0, 0.7071067812, 0.7071067812, 0.0)
CABINET_TOP_ROT: Quat = (0.7071067812, -0.7071067812, 0.0, 0.0)

CABINET_DOOR_TARGET_QUAT: Quat = (0.7071067812, 0.0, 0.7071067812, 0.0)
CABINET_TOP_TARGET_QUAT: Quat = (0.7071067812, 0.0, -0.7071067812, 0.0)


def make_cabinet_part(
    *,
    scene_key: str,
    prim_name: str,
    init_pos: Vec3,
    init_rot: Quat,
    density: float,
    tag_ids: tuple[int, ...],
) -> AssemblyPartSpec:
    """Create one dynamic cabinet part."""
    return make_dynamic_part(
        scene_key=scene_key,
        asset_name=scene_key,
        prim_name=prim_name,
        urdf_rel_path=f"urdf/cabinet/{scene_key}.urdf",
        init_pos=init_pos,
        init_rot=init_rot,
        density=density,
        tag_ids=tag_ids,
    )


PARTS: tuple[AssemblyPartSpec, ...] = (
    make_base_tag_part(init_pos=BASE_TAG_POS),
    make_cabinet_part(
        scene_key="cabinet_body",
        prim_name="CabinetBody",
        init_pos=CABINET_BODY_INIT_POS,
        init_rot=CABINET_BODY_ROT,
        density=583.11,
        tag_ids=(134, 135, 136, 137, 138),
    ),
    make_cabinet_part(
        scene_key="cabinet_door_left",
        prim_name="CabinetDoorLeft",
        init_pos=CABINET_DOOR_LEFT_INIT_POS,
        init_rot=CABINET_DOOR_ROT,
        density=412.52,
        tag_ids=(139, 140),
    ),
    make_cabinet_part(
        scene_key="cabinet_door_right",
        prim_name="CabinetDoorRight",
        init_pos=CABINET_DOOR_RIGHT_INIT_POS,
        init_rot=CABINET_DOOR_ROT,
        density=412.52,
        tag_ids=(141, 142),
    ),
    make_cabinet_part(
        scene_key="cabinet_top",
        prim_name="CabinetTop",
        init_pos=CABINET_TOP_INIT_POS,
        init_rot=CABINET_TOP_ROT,
        density=312.89,
        tag_ids=(129, 130, 131, 132, 133),
    ),
)

RELATIONS: tuple[AssemblyRelationSpec, ...] = (
    make_relation(
        parent="cabinet_body",
        child="cabinet_door_right",
        target_positions=((-0.0275, -0.0375, -0.025),),
        target_quats=(CABINET_DOOR_TARGET_QUAT,),
    ),
    make_relation(
        parent="cabinet_body",
        child="cabinet_door_left",
        target_positions=((-0.02275, -0.0375, 0.025),),
        target_quats=(CABINET_DOOR_TARGET_QUAT,),
    ),
    make_relation(
        parent="cabinet_body",
        child="cabinet_top",
        target_positions=((0.0, -0.07750, 0.0),),
        target_quats=(CABINET_TOP_TARGET_QUAT,),
    ),
)


class CabinetAssemblySpec(AssemblySpec):
    """Default cabinet assembly: body, two doors, and top."""

    def __init__(self):
        super().__init__(
            name=ASSEMBLY_NAME,
            asset_root=ASSET_ROOT,
            parts=PARTS,
            assembly_relations=RELATIONS,
        )


def make_cabinet_assembly() -> CabinetAssemblySpec:
    """Create the default cabinet assembly specification."""
    return CabinetAssemblySpec()
