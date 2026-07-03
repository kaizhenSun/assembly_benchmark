# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Registry for assembly specifications."""

from __future__ import annotations

from collections.abc import Callable

from .one_leg import OneLegAssemblySpec, make_one_leg_assembly
from .specs import AssemblySpec


_ASSEMBLY_FACTORIES: dict[str, Callable[[], AssemblySpec]] = {
    "one_leg": make_one_leg_assembly,
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
        raise ValueError(
            f"Assembly factory registered as '{name}' returned spec named '{assembly.name}'."
        )

    _ASSEMBLY_FACTORIES[name] = factory


def make_assembly(name: str) -> AssemblySpec:
    """Create an assembly specification by name."""
    try:
        return _ASSEMBLY_FACTORIES[name]()
    except KeyError as exc:
        available = ", ".join(available_assemblies())
        raise KeyError(f"Unknown assembly '{name}'. Available assemblies: {available}") from exc


__all__ = [
    "OneLegAssemblySpec",
    "available_assemblies",
    "make_assembly",
    "make_one_leg_assembly",
    "register_assembly",
]
