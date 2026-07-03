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


def make_assembly(name: str) -> AssemblySpec:
    """Create an assembly specification by name."""
    try:
        return _ASSEMBLY_FACTORIES[name]()
    except KeyError as exc:
        available = ", ".join(sorted(_ASSEMBLY_FACTORIES))
        raise KeyError(f"Unknown assembly '{name}'. Available assemblies: {available}") from exc


__all__ = ["OneLegAssemblySpec", "make_assembly", "make_one_leg_assembly"]
