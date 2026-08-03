# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Registry for assembly specifications."""

from __future__ import annotations

from collections.abc import Callable

from .fabrica import BeamAssemblySpec, make_beam_assembly
from .furniture import (
    CabinetAssemblySpec,
    ChairAssemblySpec,
    DeskAssemblySpec,
    DrawerAssemblySpec,
    LampAssemblySpec,
    OneLegAssemblySpec,
    RoundTableAssemblySpec,
    SquareTableAssemblySpec,
    StoolAssemblySpec,
    make_cabinet_assembly,
    make_chair_assembly,
    make_desk_assembly,
    make_drawer_assembly,
    make_lamp_assembly,
    make_one_leg_assembly,
    make_round_table_assembly,
    make_square_table_assembly,
    make_stool_assembly,
)
from .specs import AssemblySpec

_ASSEMBLY_FACTORIES: dict[str, Callable[[], AssemblySpec]] = {
    "beam": make_beam_assembly,
    "cabinet": make_cabinet_assembly,
    "chair": make_chair_assembly,
    "desk": make_desk_assembly,
    "drawer": make_drawer_assembly,
    "lamp": make_lamp_assembly,
    "one_leg": make_one_leg_assembly,
    "round_table": make_round_table_assembly,
    "square_table": make_square_table_assembly,
    "stool": make_stool_assembly,
}


def available_assemblies() -> tuple[str, ...]:
    """Return registered assembly names in deterministic order."""
    return tuple(sorted(_ASSEMBLY_FACTORIES))


def register_assembly(name: str, factory: Callable[[], AssemblySpec]) -> None:
    """Register an assembly specification factory for scene/task generation."""
    if name in _ASSEMBLY_FACTORIES:
        raise ValueError(f"Assembly '{name}' is already registered.")

    assembly = factory()
    if assembly.name != name:
        raise ValueError(f"Assembly factory registered as '{name}' returned spec named '{assembly.name}'.")

    _ASSEMBLY_FACTORIES[name] = factory


def make_assembly(name: str) -> AssemblySpec:
    """Create an assembly specification by name."""
    try:
        return _ASSEMBLY_FACTORIES[name]()
    except KeyError as exc:
        available = ", ".join(available_assemblies())
        raise KeyError(f"Unknown assembly '{name}'. Available assemblies: {available}") from exc


__all__ = [
    "BeamAssemblySpec",
    "CabinetAssemblySpec",
    "ChairAssemblySpec",
    "DeskAssemblySpec",
    "DrawerAssemblySpec",
    "LampAssemblySpec",
    "OneLegAssemblySpec",
    "RoundTableAssemblySpec",
    "SquareTableAssemblySpec",
    "StoolAssemblySpec",
    "available_assemblies",
    "make_beam_assembly",
    "make_cabinet_assembly",
    "make_chair_assembly",
    "make_desk_assembly",
    "make_drawer_assembly",
    "make_lamp_assembly",
    "make_assembly",
    "make_one_leg_assembly",
    "make_round_table_assembly",
    "make_square_table_assembly",
    "make_stool_assembly",
    "register_assembly",
]
