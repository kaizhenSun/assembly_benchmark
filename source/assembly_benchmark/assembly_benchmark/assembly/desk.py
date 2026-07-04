# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Default desk assembly specification."""

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


ASSEMBLY_NAME = "desk"
ASSET_ROOT = assembly_asset_root(ASSEMBLY_NAME)
DESK_ASSET_ROOT = ASSET_ROOT

BASE_TAG_POS: Vec3 = (0.5015, 0.0, 0.775)

DESK_TOP_ROT: Quat = (0.5, 0.5, -0.5, -0.5)
DESK_LEG_ROT: Quat = (0.5, 0.5, 0.5, -0.5)

DESK_TOP_INIT_POS: Vec3 = (0.7415, 0.0, 0.79092)
DESK_LEG_INIT_POSITIONS: tuple[Vec3, ...] = (
    (0.5715, -0.15, 0.7925),
    (0.5715, -0.08, 0.7925),
    (0.5715, 0.08, 0.7925),
    (0.5715, 0.15, 0.7925),
)

DESK_LEG_TARGET_POSITIONS: tuple[Vec3, ...] = (
    (-0.080, 0.065625, -0.050),
    (-0.080, 0.065625, 0.050),
    (0.080, 0.065625, -0.050),
    (0.080, 0.065625, 0.050),
)


def make_desk_part(
    *,
    scene_key: str,
    prim_name: str,
    init_pos: Vec3,
    init_rot: Quat,
    density: float,
    tag_ids: tuple[int, ...],
) -> AssemblyPartSpec:
    """Create one dynamic desk part."""
    return make_dynamic_part(
        scene_key=scene_key,
        asset_name=scene_key,
        prim_name=prim_name,
        urdf_rel_path=f"urdf/desk/{scene_key}.urdf",
        init_pos=init_pos,
        init_rot=init_rot,
        density=density,
        tag_ids=tag_ids,
    )


PARTS: tuple[AssemblyPartSpec, ...] = (
    make_base_tag_part(init_pos=BASE_TAG_POS),
    make_desk_part(
        scene_key="desk_top",
        prim_name="DeskTop",
        init_pos=DESK_TOP_INIT_POS,
        init_rot=DESK_TOP_ROT,
        density=492.98,
        tag_ids=(109, 110, 111, 112),
    ),
    *(
        make_desk_part(
            scene_key=f"desk_leg{index}",
            prim_name=f"DeskLeg{index}",
            init_pos=pos,
            init_rot=DESK_LEG_ROT,
            density=308.92,
            tag_ids=tuple(range(113 + (index - 1) * 4, 117 + (index - 1) * 4)),
        )
        for index, pos in enumerate(DESK_LEG_INIT_POSITIONS, start=1)
    ),
)

RELATIONS: tuple[AssemblyRelationSpec, ...] = (
    *(
        make_relation(
            parent="desk_top",
            child=f"desk_leg{index}",
            target_positions=DESK_LEG_TARGET_POSITIONS,
            default_target_index=index - 1,
        )
        for index in range(1, 5)
    ),
)


class DeskAssemblySpec(AssemblySpec):
    """Default desk assembly: desktop plus four legs."""

    def __init__(self):
        super().__init__(
            name=ASSEMBLY_NAME,
            asset_root=ASSET_ROOT,
            parts=PARTS,
            assembly_relations=RELATIONS,
        )


def make_desk_assembly() -> DeskAssemblySpec:
    """Create the default desk assembly specification."""
    return DeskAssemblySpec()
