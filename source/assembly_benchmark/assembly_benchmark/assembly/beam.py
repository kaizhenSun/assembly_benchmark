# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Fabrica beam 0-to-2 insertion assembly specification."""

from __future__ import annotations

from .specs import (
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


def make_beam_plug_part() -> AssemblyPartSpec:
    """Create Fabrica beam part 0 as the dynamic insertion plug."""
    return make_dynamic_part(
        scene_key="beam_plug_0",
        asset_name="beam_part_0",
        prim_name="BeamPlug0",
        urdf_rel_path="urdf/beam/beam_part_0.urdf",
        init_pos=PLUG_PREINSERT_POS,
        init_rot=IDENTITY_QUAT,
        density=BEAM_DENSITY,
        friction=BEAM_PLUG_FRICTION,
    )


def make_beam_socket_part() -> AssemblyPartSpec:
    """Create Fabrica beam part 2 as a resettable kinematic socket."""
    return make_kinematic_part(
        scene_key="beam_socket_2",
        asset_name="beam_part_2",
        prim_name="BeamSocket2",
        urdf_rel_path="urdf/beam/beam_part_2.urdf",
        init_pos=SOCKET_NOMINAL_POS,
        init_rot=IDENTITY_QUAT,
        density=BEAM_DENSITY,
        friction=BEAM_SOCKET_FRICTION,
    )


PARTS: tuple[AssemblyPartSpec, ...] = (
    make_beam_socket_part(),
    make_beam_plug_part(),
)

RELATIONS: tuple[AssemblyRelationSpec, ...] = (
    make_relation(
        parent="beam_socket_2",
        child="beam_plug_0",
        target_positions=((0.0, 0.0, 0.0),),
    ),
)


class BeamAssemblySpec(AssemblySpec):
    """Fabrica beam part 0 inserted into part 2."""

    def __init__(self):
        super().__init__(
            name=ASSEMBLY_NAME,
            asset_root=ASSET_ROOT,
            parts=PARTS,
            assembly_relations=RELATIONS,
        )


def make_beam_assembly() -> BeamAssemblySpec:
    """Create the Fabrica beam 0-to-2 assembly specification."""
    return BeamAssemblySpec()
