# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Assembly specifications for assembly_benchmark tasks."""

from .one_leg import OneLegAssemblySpec, make_one_leg_assembly
from .registry import available_assemblies, make_assembly, register_assembly
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
    make_part,
    make_relation,
    make_visual_part,
)

__all__ = [
    "AssemblyPartSpec",
    "AssemblyRelationSpec",
    "AssemblySpec",
    "AssemblyTargetPose",
    "ASSEMBLY_ASSET_ROOT",
    "DEFAULT_ORI_BOUND",
    "DEFAULT_POS_THRESHOLD",
    "DEFAULT_TARGET_INDEX",
    "OneLegAssemblySpec",
    "PartBodyType",
    "Quat",
    "Vec3",
    "available_assemblies",
    "assembly_asset_root",
    "make_assembly",
    "make_base_tag_part",
    "make_dynamic_part",
    "make_one_leg_assembly",
    "make_part",
    "make_relation",
    "make_visual_part",
    "register_assembly",
]
