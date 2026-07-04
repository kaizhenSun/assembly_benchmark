# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Default drawer assembly specification."""

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


ASSEMBLY_NAME = "drawer"
ASSET_ROOT = assembly_asset_root(ASSEMBLY_NAME)
DRAWER_ASSET_ROOT = ASSET_ROOT

BASE_TAG_POS: Vec3 = (0.5015, 0.0, 0.775)

DRAWER_BOX_INIT_POS: Vec3 = (0.6315, 0.08, 0.81625)
DRAWER_CONTAINER_TOP_INIT_POS: Vec3 = (0.6215, -0.11, 0.8005)
DRAWER_CONTAINER_BOTTOM_INIT_POS: Vec3 = (0.7515, -0.11, 0.8005)

DRAWER_BOX_ROT: Quat = (0.5, -0.5, 0.5, -0.5)
DRAWER_CONTAINER_ROT: Quat = (0.5, -0.5, -0.5, 0.5)

DRAWER_CONTAINER_TARGET_POSITIONS: tuple[Vec3, ...] = (
    (0.0, -0.0345, 0.008),
    (0.0, 0.0105, 0.008),
)


def make_drawer_part(
    *,
    scene_key: str,
    prim_name: str,
    init_pos: Vec3,
    init_rot: Quat,
    density: float,
    tag_ids: tuple[int, ...],
) -> AssemblyPartSpec:
    """Create one dynamic drawer part."""
    return make_dynamic_part(
        scene_key=scene_key,
        asset_name=scene_key,
        prim_name=prim_name,
        urdf_rel_path=f"urdf/drawer/{scene_key}.urdf",
        init_pos=init_pos,
        init_rot=init_rot,
        density=density,
        tag_ids=tag_ids,
    )


PARTS: tuple[AssemblyPartSpec, ...] = (
    make_base_tag_part(init_pos=BASE_TAG_POS),
    make_drawer_part(
        scene_key="drawer_box",
        prim_name="DrawerBox",
        init_pos=DRAWER_BOX_INIT_POS,
        init_rot=DRAWER_BOX_ROT,
        density=683.47,
        tag_ids=(48, 49, 50, 52),
    ),
    make_drawer_part(
        scene_key="drawer_container_top",
        prim_name="DrawerContainerTop",
        init_pos=DRAWER_CONTAINER_TOP_INIT_POS,
        init_rot=DRAWER_CONTAINER_ROT,
        density=639.10,
        tag_ids=(53, 54, 55, 56, 57),
    ),
    make_drawer_part(
        scene_key="drawer_container_bottom",
        prim_name="DrawerContainerBottom",
        init_pos=DRAWER_CONTAINER_BOTTOM_INIT_POS,
        init_rot=DRAWER_CONTAINER_ROT,
        density=639.10,
        tag_ids=(58, 59, 60, 61, 62),
    ),
)

RELATIONS: tuple[AssemblyRelationSpec, ...] = (
    make_relation(
        parent="drawer_box",
        child="drawer_container_top",
        target_positions=DRAWER_CONTAINER_TARGET_POSITIONS,
    ),
    make_relation(
        parent="drawer_box",
        child="drawer_container_bottom",
        target_positions=DRAWER_CONTAINER_TARGET_POSITIONS,
        default_target_index=1,
    ),
)


class DrawerAssemblySpec(AssemblySpec):
    """Default drawer assembly: box plus two container rails."""

    def __init__(self):
        super().__init__(
            name=ASSEMBLY_NAME,
            asset_root=ASSET_ROOT,
            parts=PARTS,
            assembly_relations=RELATIONS,
        )


def make_drawer_assembly() -> DrawerAssemblySpec:
    """Create the default drawer assembly specification."""
    return DrawerAssemblySpec()
