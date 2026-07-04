# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Default lamp assembly specification."""

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


ASSEMBLY_NAME = "lamp"
ASSET_ROOT = assembly_asset_root(ASSEMBLY_NAME)
LAMP_ASSET_ROOT = ASSET_ROOT

BASE_TAG_POS: Vec3 = (0.5015, 0.0, 0.775)

LAMP_BASE_INIT_POS: Vec3 = (0.6415, 0.05, 0.795)
LAMP_BULB_INIT_POS: Vec3 = (0.6315, 0.18, 0.791)
LAMP_HOOD_INIT_POS: Vec3 = (0.6615, -0.12, 0.823583)

LAMP_BASE_ROT: Quat = (0.5, -0.5, -0.5, 0.5)
LAMP_BULB_ROT: Quat = (0.0512648872, 0.7052459935, 0.7052459935, 0.0512648872)
LAMP_HOOD_ROT: Quat = LAMP_BASE_ROT

LAMP_BULB_TARGET_POSITIONS: tuple[Vec3, ...] = ((0.0, -0.0678, 0.0),)
LAMP_BULB_TARGET_QUATS: tuple[Quat, ...] = ((0.0, 0.0, 0.0, 1.0),)
LAMP_HOOD_TARGET_POSITIONS: tuple[Vec3, ...] = ((0.0, -0.088324, 0.0),)


def make_lamp_part(
    *,
    scene_key: str,
    prim_name: str,
    init_pos: Vec3,
    init_rot: Quat,
    density: float,
    tag_ids: tuple[int, ...],
) -> AssemblyPartSpec:
    """Create one dynamic lamp part."""
    return make_dynamic_part(
        scene_key=scene_key,
        asset_name=scene_key,
        prim_name=prim_name,
        urdf_rel_path=f"urdf/lamp/{scene_key}.urdf",
        init_pos=init_pos,
        init_rot=init_rot,
        density=density,
        tag_ids=tag_ids,
    )


PARTS: tuple[AssemblyPartSpec, ...] = (
    make_base_tag_part(init_pos=BASE_TAG_POS),
    make_lamp_part(
        scene_key="lamp_base",
        prim_name="LampBase",
        init_pos=LAMP_BASE_INIT_POS,
        init_rot=LAMP_BASE_ROT,
        density=341.54,
        tag_ids=(169, 170, 171, 172, 173),
    ),
    make_lamp_part(
        scene_key="lamp_bulb",
        prim_name="LampBulb",
        init_pos=LAMP_BULB_INIT_POS,
        init_rot=LAMP_BULB_ROT,
        density=545.09,
        tag_ids=(174, 177, 176, 175),
    ),
    make_lamp_part(
        scene_key="lamp_hood",
        prim_name="LampHood",
        init_pos=LAMP_HOOD_INIT_POS,
        init_rot=LAMP_HOOD_ROT,
        density=762.31,
        tag_ids=(163, 164, 165, 166, 167, 168),
    ),
)

RELATIONS: tuple[AssemblyRelationSpec, ...] = (
    make_relation(
        parent="lamp_base",
        child="lamp_bulb",
        target_positions=LAMP_BULB_TARGET_POSITIONS,
        target_quats=LAMP_BULB_TARGET_QUATS,
    ),
    make_relation(
        parent="lamp_base",
        child="lamp_hood",
        target_positions=LAMP_HOOD_TARGET_POSITIONS,
        ori_bound=-1.0,
    ),
)


class LampAssemblySpec(AssemblySpec):
    """Default lamp assembly: base, bulb, and hood."""

    def __init__(self):
        super().__init__(
            name=ASSEMBLY_NAME,
            asset_root=ASSET_ROOT,
            parts=PARTS,
            assembly_relations=RELATIONS,
        )


def make_lamp_assembly() -> LampAssemblySpec:
    """Create the default lamp assembly specification."""
    return LampAssemblySpec()
