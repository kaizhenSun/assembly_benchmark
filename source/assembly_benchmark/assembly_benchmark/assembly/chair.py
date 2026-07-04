# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Default chair assembly specification."""

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


ASSEMBLY_NAME = "chair"
ASSET_ROOT = assembly_asset_root(ASSEMBLY_NAME)
CHAIR_ASSET_ROOT = ASSET_ROOT

SQRT_HALF = 0.7071067812

BASE_TAG_POS: Vec3 = (0.5015, 0.0, 0.775)

CHAIR_SEAT_INIT_POS: Vec3 = (0.7715, 0.03, 0.79)
CHAIR_LEG1_INIT_POS: Vec3 = (0.6215, 0.01, 0.79)
CHAIR_LEG2_INIT_POS: Vec3 = (0.6215, 0.07, 0.79)
CHAIR_BACK_INIT_POS: Vec3 = (0.7015, -0.10, 0.79)
CHAIR_NUT1_INIT_POS: Vec3 = (0.6215, 0.13, 0.79)
CHAIR_NUT2_INIT_POS: Vec3 = (0.6215, 0.19, 0.79)

CHAIR_SEAT_ROT: Quat = (0.5, 0.5, -0.5, -0.5)
CHAIR_LEG_ROT: Quat = (0.0, SQRT_HALF, SQRT_HALF, 0.0)
CHAIR_BACK_ROT: Quat = (SQRT_HALF, 0.0, 0.0, -SQRT_HALF)
CHAIR_NUT_ROT: Quat = (0.5, -0.5, -0.5, 0.5)

CHAIR_LEG_TARGET_ROT: Quat = (SQRT_HALF, 0.0, SQRT_HALF, 0.0)
CHAIR_NUT_TARGET_ROT: Quat = (SQRT_HALF, -SQRT_HALF, 0.0, 0.0)

CHAIR_LEG_TARGET_POSITIONS: tuple[Vec3, ...] = (
    (-0.03375, 0.045, -0.01875),
    (0.03375, 0.045, -0.01875),
)
CHAIR_LEG_TARGET_QUATS: tuple[Quat, ...] = (
    CHAIR_LEG_TARGET_ROT,
    CHAIR_LEG_TARGET_ROT,
)
CHAIR_BACK_TARGET_POSITIONS: tuple[Vec3, ...] = (
    (0.0, -0.0325, 0.05025),
)
CHAIR_BACK_TARGET_QUATS: tuple[Quat, ...] = (
    (1.0, 0.0, 0.0, 0.0),
)
CHAIR_NUT_TARGET_POSITIONS: tuple[Vec3, ...] = (
    (0.035, 0.0, 0.0795),
    (-0.035, 0.0, 0.0795),
)
CHAIR_NUT_TARGET_QUATS: tuple[Quat, ...] = (
    CHAIR_NUT_TARGET_ROT,
    CHAIR_NUT_TARGET_ROT,
)


def make_chair_part(
    *,
    scene_key: str,
    prim_name: str,
    init_pos: Vec3,
    init_rot: Quat,
    mass: float,
    tag_ids: tuple[int, ...],
    reset_footprint_xy: tuple[float, float],
) -> AssemblyPartSpec:
    """Create one dynamic chair part."""
    return make_dynamic_part(
        scene_key=scene_key,
        asset_name=scene_key,
        prim_name=prim_name,
        urdf_rel_path=f"urdf/chair/{scene_key}.urdf",
        init_pos=init_pos,
        init_rot=init_rot,
        mass=mass,
        tag_ids=tag_ids,
        reset_footprint_xy=reset_footprint_xy,
    )


PARTS: tuple[AssemblyPartSpec, ...] = (
    make_base_tag_part(init_pos=BASE_TAG_POS),
    make_chair_part(
        scene_key="chair_seat",
        prim_name="ChairSeat",
        init_pos=CHAIR_SEAT_INIT_POS,
        init_rot=CHAIR_SEAT_ROT,
        mass=0.06187,
        tag_ids=(79, 80, 81, 82),
        reset_footprint_xy=(0.10, 0.11875),
    ),
    make_chair_part(
        scene_key="chair_leg1",
        prim_name="ChairLeg1",
        init_pos=CHAIR_LEG1_INIT_POS,
        init_rot=CHAIR_LEG_ROT,
        mass=0.02244,
        tag_ids=(91, 92, 93, 94),
        reset_footprint_xy=(0.03, 0.085),
    ),
    make_chair_part(
        scene_key="chair_leg2",
        prim_name="ChairLeg2",
        init_pos=CHAIR_LEG2_INIT_POS,
        init_rot=CHAIR_LEG_ROT,
        mass=0.02244,
        tag_ids=(95, 96, 97, 98),
        reset_footprint_xy=(0.03, 0.085),
    ),
    make_chair_part(
        scene_key="chair_back",
        prim_name="ChairBack",
        init_pos=CHAIR_BACK_INIT_POS,
        init_rot=CHAIR_BACK_ROT,
        mass=0.12316,
        tag_ids=(83, 84, 85, 86, 87, 88, 89, 90),
        reset_footprint_xy=(0.10, 0.21),
    ),
    make_chair_part(
        scene_key="chair_nut1",
        prim_name="ChairNut1",
        init_pos=CHAIR_NUT1_INIT_POS,
        init_rot=CHAIR_NUT_ROT,
        mass=0.01015,
        tag_ids=(99, 100, 101, 102, 103),
        reset_footprint_xy=(0.03, 0.03),
    ),
    make_chair_part(
        scene_key="chair_nut2",
        prim_name="ChairNut2",
        init_pos=CHAIR_NUT2_INIT_POS,
        init_rot=CHAIR_NUT_ROT,
        mass=0.01015,
        tag_ids=(104, 105, 106, 107, 108),
        reset_footprint_xy=(0.03, 0.03),
    ),
)

RELATIONS: tuple[AssemblyRelationSpec, ...] = (
    make_relation(
        parent="chair_seat",
        child="chair_leg1",
        target_positions=CHAIR_LEG_TARGET_POSITIONS,
        target_quats=CHAIR_LEG_TARGET_QUATS,
        default_target_index=1,
    ),
    make_relation(
        parent="chair_seat",
        child="chair_leg2",
        target_positions=CHAIR_LEG_TARGET_POSITIONS,
        target_quats=CHAIR_LEG_TARGET_QUATS,
    ),
    make_relation(
        parent="chair_seat",
        child="chair_back",
        target_positions=CHAIR_BACK_TARGET_POSITIONS,
        target_quats=CHAIR_BACK_TARGET_QUATS,
    ),
    make_relation(
        parent="chair_seat",
        child="chair_nut1",
        target_positions=CHAIR_NUT_TARGET_POSITIONS,
        target_quats=CHAIR_NUT_TARGET_QUATS,
        default_target_index=1,
    ),
    make_relation(
        parent="chair_seat",
        child="chair_nut2",
        target_positions=CHAIR_NUT_TARGET_POSITIONS,
        target_quats=CHAIR_NUT_TARGET_QUATS,
    ),
)


class ChairAssemblySpec(AssemblySpec):
    """Default chair assembly: seat, two legs, back, and two nuts."""

    def __init__(self):
        super().__init__(
            name=ASSEMBLY_NAME,
            asset_root=ASSET_ROOT,
            parts=PARTS,
            assembly_relations=RELATIONS,
        )


def make_chair_assembly() -> ChairAssemblySpec:
    """Create the default chair assembly specification."""
    return ChairAssemblySpec()
