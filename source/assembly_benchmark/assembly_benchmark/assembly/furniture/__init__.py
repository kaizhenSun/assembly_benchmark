# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Furniture assembly specifications."""

from .cabinet import CabinetAssemblySpec, make_cabinet_assembly
from .chair import ChairAssemblySpec, make_chair_assembly
from .desk import DeskAssemblySpec, make_desk_assembly
from .drawer import DrawerAssemblySpec, make_drawer_assembly
from .lamp import LampAssemblySpec, make_lamp_assembly
from .one_leg import OneLegAssemblySpec, make_one_leg_assembly
from .round_table import RoundTableAssemblySpec, make_round_table_assembly
from .square_table import SquareTableAssemblySpec, make_square_table_assembly
from .stool import StoolAssemblySpec, make_stool_assembly

__all__ = [
    "CabinetAssemblySpec",
    "ChairAssemblySpec",
    "DeskAssemblySpec",
    "DrawerAssemblySpec",
    "LampAssemblySpec",
    "OneLegAssemblySpec",
    "RoundTableAssemblySpec",
    "SquareTableAssemblySpec",
    "StoolAssemblySpec",
    "make_cabinet_assembly",
    "make_chair_assembly",
    "make_desk_assembly",
    "make_drawer_assembly",
    "make_lamp_assembly",
    "make_one_leg_assembly",
    "make_round_table_assembly",
    "make_square_table_assembly",
    "make_stool_assembly",
]
