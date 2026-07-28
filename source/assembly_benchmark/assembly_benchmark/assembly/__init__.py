# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Assembly specifications for assembly_benchmark tasks."""

from .beam import BeamAssemblySpec, make_beam_assembly
from .cabinet import CabinetAssemblySpec, make_cabinet_assembly
from .chair import ChairAssemblySpec, make_chair_assembly
from .desk import DeskAssemblySpec, make_desk_assembly
from .drawer import DrawerAssemblySpec, make_drawer_assembly
from .lamp import LampAssemblySpec, make_lamp_assembly
from .one_leg import OneLegAssemblySpec, make_one_leg_assembly
from .registry import available_assemblies, make_assembly, register_assembly
from .round_table import RoundTableAssemblySpec, make_round_table_assembly
from .specs import (
    ASSEMBLY_ASSET_ROOT,
    DEFAULT_ORI_BOUND,
    DEFAULT_POS_THRESHOLD,
    DEFAULT_TARGET_INDEX,
    AssemblyPartSpec,
    AssemblyRelationSpec,
    AssemblySpec,
    AssemblyTargetPose,
    PartBodyType,
    Quat,
    Vec3,
    assembly_asset_root,
    make_base_tag_part,
    make_dynamic_part,
    make_kinematic_part,
    make_part,
    make_relation,
    make_visual_part,
)
from .square_table import SquareTableAssemblySpec, make_square_table_assembly
from .stool import StoolAssemblySpec, make_stool_assembly

__all__ = [
    "AssemblyPartSpec",
    "AssemblyRelationSpec",
    "AssemblySpec",
    "AssemblyTargetPose",
    "ASSEMBLY_ASSET_ROOT",
    "DEFAULT_ORI_BOUND",
    "DEFAULT_POS_THRESHOLD",
    "DEFAULT_TARGET_INDEX",
    "BeamAssemblySpec",
    "CabinetAssemblySpec",
    "ChairAssemblySpec",
    "DeskAssemblySpec",
    "DrawerAssemblySpec",
    "LampAssemblySpec",
    "OneLegAssemblySpec",
    "PartBodyType",
    "Quat",
    "RoundTableAssemblySpec",
    "SquareTableAssemblySpec",
    "StoolAssemblySpec",
    "Vec3",
    "available_assemblies",
    "assembly_asset_root",
    "make_assembly",
    "make_base_tag_part",
    "make_beam_assembly",
    "make_cabinet_assembly",
    "make_chair_assembly",
    "make_desk_assembly",
    "make_drawer_assembly",
    "make_dynamic_part",
    "make_kinematic_part",
    "make_lamp_assembly",
    "make_one_leg_assembly",
    "make_part",
    "make_relation",
    "make_round_table_assembly",
    "make_square_table_assembly",
    "make_stool_assembly",
    "make_visual_part",
    "register_assembly",
]
