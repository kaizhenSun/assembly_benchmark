# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Assembly specifications for assembly_benchmark tasks."""

from .one_leg import OneLegAssemblySpec, make_one_leg_assembly
from .registry import available_assemblies, make_assembly, register_assembly
from .specs import AssemblyPartSpec, AssemblyRelationSpec, AssemblySpec, AssemblyTargetPose

__all__ = [
    "AssemblyPartSpec",
    "AssemblyRelationSpec",
    "AssemblySpec",
    "AssemblyTargetPose",
    "OneLegAssemblySpec",
    "available_assemblies",
    "make_assembly",
    "make_one_leg_assembly",
    "register_assembly",
]
