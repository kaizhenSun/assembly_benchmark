# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Complete Fabrica Beam assembly specification."""

from __future__ import annotations

from ..specs import (
    IDENTITY_QUAT,
    AssemblyPartSpec,
    AssemblyRelationSpec,
    AssemblySpec,
    Vec3,
    assembly_asset_root,
    make_dynamic_part,
    make_kinematic_part,
    make_relation,
)

ASSEMBLY_NAME = "beam"
ASSET_ROOT = assembly_asset_root(ASSEMBLY_NAME)
BEAM_ASSET_ROOT = ASSET_ROOT

BEAM_PART_IDS = ("0", "1", "2", "3", "6")
BEAM_RELATION_PART_IDS = (("0", "2"), ("1", "3"), ("2", "6"), ("3", "6"))
BEAM_DENSITY = 1250.0
BEAM_PLUG_FRICTION = 1.0
BEAM_SOCKET_FRICTION = 0.5

SOCKET_NOMINAL_POS: Vec3 = (0.55, 0.30, 0.775)
INSERTION_CLEARANCE = 0.0075
PLUG_MIN_Z = 0.0634999069
SOCKET_MAX_Z = 0.0761999446
INSERTION_PATH_LENGTH = SOCKET_MAX_Z - PLUG_MIN_Z + INSERTION_CLEARANCE
PLUG_PREINSERT_POS: Vec3 = (
    SOCKET_NOMINAL_POS[0],
    SOCKET_NOMINAL_POS[1],
    SOCKET_NOMINAL_POS[2] + INSERTION_PATH_LENGTH,
)


def beam_part_scene_key(part_id: str) -> str:
    """Return the role-neutral scene key for one Beam part."""
    if part_id not in BEAM_PART_IDS:
        raise ValueError(f"Unsupported Beam part id '{part_id}'. Expected one of {BEAM_PART_IDS}.")
    return f"beam_part_{part_id}"


def make_beam_part(part_id: str) -> AssemblyPartSpec:
    """Create one part from the complete Beam assembly graph."""
    scene_key = beam_part_scene_key(part_id)
    common = {
        "scene_key": scene_key,
        "asset_name": scene_key,
        "prim_name": "".join(token.capitalize() for token in scene_key.split("_")),
        "urdf_rel_path": f"urdf/{scene_key}.urdf",
        "init_rot": IDENTITY_QUAT,
        "density": BEAM_DENSITY,
    }
    if part_id in ("0", "1"):
        return make_dynamic_part(
            **common,
            init_pos=PLUG_PREINSERT_POS,
            friction=BEAM_PLUG_FRICTION,
        )
    return make_kinematic_part(
        **common,
        init_pos=SOCKET_NOMINAL_POS,
        friction=BEAM_SOCKET_FRICTION,
    )


def make_beam_plug_part() -> AssemblyPartSpec:
    """Compatibility helper returning Beam part 0."""
    return make_beam_part("0")


def make_beam_socket_part() -> AssemblyPartSpec:
    """Compatibility helper returning Beam part 2."""
    return make_beam_part("2")


PARTS: tuple[AssemblyPartSpec, ...] = tuple(make_beam_part(part_id) for part_id in BEAM_PART_IDS)

RELATIONS: tuple[AssemblyRelationSpec, ...] = tuple(
    make_relation(
        parent=beam_part_scene_key(socket_id),
        child=beam_part_scene_key(plug_id),
        target_positions=((0.0, 0.0, 0.0),),
    )
    for plug_id, socket_id in BEAM_RELATION_PART_IDS
)


class BeamAssemblySpec(AssemblySpec):
    """Fabrica Beam assembly with five parts and four insertion relations."""

    def __init__(self):
        super().__init__(
            name=ASSEMBLY_NAME,
            asset_root=ASSET_ROOT,
            parts=PARTS,
            assembly_relations=RELATIONS,
        )


def make_beam_assembly() -> BeamAssemblySpec:
    """Create the complete Fabrica Beam assembly specification."""
    return BeamAssemblySpec()
