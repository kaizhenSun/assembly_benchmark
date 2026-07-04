# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Isaac Lab config helpers for assembly specifications."""

from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg

from .specs import AssemblyPartSpec, AssemblySpec


def make_assembly_part_spawn_cfg(
    assembly: AssemblySpec, part: AssemblyPartSpec
) -> sim_utils.UsdFileCfg:
    """Create a USD spawn config for an assembly part."""
    usd_path = str(part.usd_path(assembly.asset_root))
    mass_props = None
    if part.mass is not None:
        mass_props = sim_utils.MassPropertiesCfg(mass=part.mass)
    elif part.density is not None:
        mass_props = sim_utils.MassPropertiesCfg(density=part.density)

    if part.body_type == "visual":
        return sim_utils.UsdFileCfg(usd_path=usd_path)
    if part.body_type == "static":
        return sim_utils.UsdFileCfg(
            usd_path=usd_path,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=True,
                max_depenetration_velocity=5.0,
            ),
        )
    if part.body_type == "dynamic":
        return sim_utils.UsdFileCfg(
            usd_path=usd_path,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                max_depenetration_velocity=5.0,
            ),
            mass_props=mass_props,
        )
    raise ValueError(f"Unsupported body type '{part.body_type}' for part '{part.scene_key}'.")


def make_assembly_part_cfg(assembly: AssemblySpec, scene_key: str) -> AssetBaseCfg | RigidObjectCfg:
    """Create an Isaac Lab scene asset config for an assembly part."""
    part = assembly.part(scene_key)
    prim_path = f"{{ENV_REGEX_NS}}/{part.prim_name}"
    spawn = make_assembly_part_spawn_cfg(assembly, part)
    if part.body_type == "dynamic":
        return RigidObjectCfg(
            prim_path=prim_path,
            spawn=spawn,
            init_state=RigidObjectCfg.InitialStateCfg(pos=part.init_pos, rot=part.init_rot),
        )
    return AssetBaseCfg(
        prim_path=prim_path,
        spawn=spawn,
        init_state=AssetBaseCfg.InitialStateCfg(pos=part.init_pos, rot=part.init_rot),
    )
