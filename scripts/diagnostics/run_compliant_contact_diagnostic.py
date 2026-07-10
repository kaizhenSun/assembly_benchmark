# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Compare the R1 Pro tactile compliant material authoring paths under a constant load.

The diagnostic drops identical spheres onto a rigid pad and two compliant pads. The
compliant pads use the shared R1 Pro material values but are authored through either
Isaac Lab's :class:`RigidBodyMaterialCfg` or the low-level helper used by the R1 Pro
converter. An assembly-part SDF comparison can be enabled separately.

.. code-block:: bash

    python scripts/diagnostics/run_compliant_contact_diagnostic.py --headless --device cuda:0
    python scripts/diagnostics/run_compliant_contact_diagnostic.py --headless \
        --include_sdf --sdf_asset path/to/contact_part.usd

"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Compare R1 Pro tactile compliant-contact material authoring paths.")
parser.add_argument("--steps", type=int, default=2400, help="Number of simulation steps.")
parser.add_argument("--dt", type=float, default=1.0 / 240.0, help="Physics time step in seconds.")
parser.add_argument("--sphere_mass", type=float, default=0.1, help="Mass of every test object in kilograms.")
parser.add_argument(
    "--include_sdf",
    action="store_true",
    help="Also compare the material paths using an assembly-part SDF collider.",
)
parser.add_argument(
    "--sdf_asset",
    type=Path,
    default=None,
    help="USD asset used by --include_sdf; omitted from the default primitive-only diagnostic.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


import torch
from assembly_benchmark.sensors import (
    R1_PRO_GRIPPER_TACTILE_MATERIAL,
    TactileContactMaterialSpec,
    apply_tactile_compliant_material,
)

from pxr import UsdShade

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObject, RigidObjectCfg

PAD_SIZE = (0.20, 0.20, 0.20)
PAD_TOP_Z = PAD_SIZE[2] / 2.0
SPHERE_RADIUS = 0.025
SPHERE_START_Z = PAD_TOP_Z + SPHERE_RADIUS + 0.04
NOMINAL_CONTACT_CENTER_Z = PAD_TOP_Z + SPHERE_RADIUS
SDF_OBJECT_START_Z = 0.45


@dataclass(frozen=True)
class ContactCase:
    """One constant-load material-authoring case."""

    name: str
    authoring: str


CASES = (
    ContactCase("rigid", "none"),
    ContactCase("official_material", "official"),
    ContactCase("shared_authoring", "shared"),
)


def _resolved_material(stage, collider_path: str) -> tuple[str, float | None]:
    """Return the resolved physics material path and compliant stiffness."""
    collider_prim = stage.GetPrimAtPath(collider_path)
    material, _ = UsdShade.MaterialBindingAPI(collider_prim).ComputeBoundMaterial("physics")
    if not material:
        return "<scene default>", None
    material_prim = material.GetPrim()
    stiffness_attr = material_prim.GetAttribute("physxMaterial:compliantContactStiffness")
    stiffness = stiffness_attr.Get() if stiffness_attr.IsValid() else None
    return str(material.GetPath()), stiffness


def _official_material_cfg(spec: TactileContactMaterialSpec) -> sim_utils.RigidBodyMaterialCfg:
    """Translate the shared material values to Isaac Lab's official config type."""
    return sim_utils.RigidBodyMaterialCfg(
        static_friction=spec.static_friction,
        dynamic_friction=spec.dynamic_friction,
        restitution=spec.restitution,
        compliant_contact_stiffness=spec.stiffness,
        compliant_contact_damping=spec.damping,
    )


def _spawn_pad(stage, case: ContactCase, pad_path: str, position: tuple[float, float, float]) -> None:
    """Spawn one static pad and apply the requested material-authoring path."""
    spec = R1_PRO_GRIPPER_TACTILE_MATERIAL
    collider_path = f"{pad_path}/geometry/mesh"
    physics_material = _official_material_cfg(spec) if case.authoring == "official" else None

    pad_cfg = sim_utils.CuboidCfg(
        size=PAD_SIZE,
        collision_props=sim_utils.CollisionPropertiesCfg(
            collision_enabled=True,
            contact_offset=spec.contact_offset,
            rest_offset=spec.rest_offset,
        ),
        physics_material=physics_material,
    )
    pad_cfg.func(pad_path, pad_cfg, translation=position)

    if case.authoring == "shared":
        collider_prim = stage.GetPrimAtPath(collider_path)
        apply_tactile_compliant_material(
            stage,
            collider_prim,
            f"{pad_path}/materials/tactile_compliant_material",
            spec,
        )

    material_path, resolved_stiffness = _resolved_material(stage, collider_path)
    print(
        f"[BINDING] pad={pad_path} case={case.name} authoring={case.authoring} material={material_path} "
        f"stiffness={resolved_stiffness}"
    )


def _spawn_case(stage, case: ContactCase, x_pos: float) -> RigidObject:
    """Spawn one static pad and its gravity-loaded sphere."""
    pad_path = f"/World/Pads/{case.name}"
    _spawn_pad(stage, case, pad_path, (x_pos, 0.0, 0.0))

    sphere_cfg = RigidObjectCfg(
        prim_path=f"/World/Spheres/{case.name}",
        spawn=sim_utils.SphereCfg(
            radius=SPHERE_RADIUS,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                linear_damping=20.0,
                angular_damping=20.0,
                max_depenetration_velocity=5.0,
                solver_position_iteration_count=16,
                solver_velocity_iteration_count=4,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=args_cli.sphere_mass),
            collision_props=sim_utils.CollisionPropertiesCfg(
                collision_enabled=True,
                contact_offset=R1_PRO_GRIPPER_TACTILE_MATERIAL.contact_offset,
                rest_offset=R1_PRO_GRIPPER_TACTILE_MATERIAL.rest_offset,
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(x_pos, 0.0, SPHERE_START_Z)),
    )
    return RigidObject(sphere_cfg)


def _spawn_sdf_case(stage, case: ContactCase, x_pos: float) -> RigidObject:
    """Spawn one static pad and an assembly part using an SDF collider."""
    pad_path = f"/World/SdfPads/{case.name}"
    _spawn_pad(stage, case, pad_path, (x_pos, 0.5, 0.0))

    object_cfg = RigidObjectCfg(
        prim_path=f"/World/SdfObjects/{case.name}",
        spawn=sim_utils.UsdFileCfg(
            usd_path=str(args_cli.sdf_asset),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                linear_damping=20.0,
                angular_damping=20.0,
                max_depenetration_velocity=5.0,
                solver_position_iteration_count=16,
                solver_velocity_iteration_count=4,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=args_cli.sphere_mass),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(x_pos, 0.5, SDF_OBJECT_START_Z)),
    )
    return RigidObject(object_cfg)


def _evaluate_results(samples: dict[str, torch.Tensor]) -> bool:
    """Print steady-state sphere displacement results and validate both material paths."""
    penetrations_mm: dict[str, float] = {}
    for case in CASES:
        case_samples = samples[case.name]
        mean_z = float(case_samples.mean())
        std_z = float(case_samples.std())
        penetration_mm = (NOMINAL_CONTACT_CENTER_Z - mean_z) * 1000.0
        penetrations_mm[case.name] = penetration_mm
        escaped = mean_z < -PAD_SIZE[2] / 2.0 - SPHERE_RADIUS
        print(
            f"[RESULT] case={case.name} authoring={case.authoring} mean_z={mean_z:.6f} m "
            f"std_z={std_z:.6f} m penetration={penetration_mm:.3f} mm escaped={escaped}"
        )

    rigid_penetration = penetrations_mm["rigid"]
    official_penetration = penetrations_mm["official_material"]
    shared_penetration = penetrations_mm["shared_authoring"]
    checks = {
        "official_material_is_softer_than_rigid": official_penetration > rigid_penetration + 0.2,
        "shared_authoring_is_softer_than_rigid": shared_penetration > rigid_penetration + 0.2,
        "shared_and_official_authoring_match": abs(shared_penetration - official_penetration) < 0.25,
    }
    for name, passed in checks.items():
        print(f"[CHECK] {name}={'PASS' if passed else 'FAIL'}")
    return all(checks.values())


def _evaluate_sdf_results(samples: dict[str, torch.Tensor]) -> bool:
    """Validate both material paths against an assembly-part SDF collider."""
    mean_z = {name: float(values.mean()) for name, values in samples.items()}
    rigid_z = mean_z["rigid"]
    drop_from_rigid_mm: dict[str, float] = {}
    for case in CASES:
        case_samples = samples[case.name]
        std_z = float(case_samples.std())
        drop_mm = (rigid_z - mean_z[case.name]) * 1000.0
        drop_from_rigid_mm[case.name] = drop_mm
        print(
            f"[SDF_RESULT] case={case.name} authoring={case.authoring} mean_root_z={mean_z[case.name]:.6f} m "
            f"std_z={std_z:.6f} m drop_from_rigid={drop_mm:.3f} mm"
        )

    official_drop = drop_from_rigid_mm["official_material"]
    shared_drop = drop_from_rigid_mm["shared_authoring"]
    checks = {
        "sdf_official_material_is_softer_than_rigid": official_drop > 0.05,
        "sdf_shared_authoring_is_softer_than_rigid": shared_drop > 0.05,
        "sdf_shared_and_official_authoring_match": abs(shared_drop - official_drop) < 0.25,
    }
    for name, passed in checks.items():
        print(f"[CHECK] {name}={'PASS' if passed else 'FAIL'}")
    return all(checks.values())


def _stack_samples(samples: dict[str, list[torch.Tensor]]) -> dict[str, torch.Tensor]:
    """Stack the per-step samples for every case."""
    return {name: torch.stack(values) for name, values in samples.items()}


def main() -> int:
    """Run the diagnostic and return a process exit status."""
    if args_cli.steps < 240:
        raise ValueError("--steps must be at least 240 so a steady-state window can be measured.")
    if args_cli.dt <= 0.0:
        raise ValueError("--dt must be positive.")
    if args_cli.sphere_mass <= 0.0:
        raise ValueError("--sphere_mass must be positive.")
    if args_cli.include_sdf and args_cli.sdf_asset is None:
        raise ValueError("--include_sdf requires --sdf_asset.")
    if args_cli.sdf_asset is not None and not args_cli.sdf_asset.is_file():
        raise FileNotFoundError(f"SDF diagnostic asset does not exist: {args_cli.sdf_asset}")

    sim_cfg = sim_utils.SimulationCfg(dt=args_cli.dt, device=args_cli.device)
    sim = sim_utils.SimulationContext(sim_cfg)
    sim.set_camera_view(eye=(0.5, -2.5, 1.2), target=(0.5, 0.0, 0.0))

    stage = sim.stage
    sim_utils.create_prim("/World/Pads", "Xform")
    sim_utils.create_prim("/World/Spheres", "Xform")
    spacing = 0.35
    spheres = {case.name: _spawn_case(stage, case, index * spacing) for index, case in enumerate(CASES)}

    sdf_objects: dict[str, RigidObject] = {}
    if args_cli.include_sdf:
        sim_utils.create_prim("/World/SdfPads", "Xform")
        sim_utils.create_prim("/World/SdfObjects", "Xform")
        sdf_objects = {case.name: _spawn_sdf_case(stage, case, index * spacing) for index, case in enumerate(CASES)}

    sim.reset()
    settling_window = min(240, args_cli.steps // 4)
    samples: dict[str, list[torch.Tensor]] = {case.name: [] for case in CASES}
    sdf_samples: dict[str, list[torch.Tensor]] = {case.name: [] for case in CASES}
    for step in range(args_cli.steps):
        sim.step(render=False)
        for case_name, sphere in spheres.items():
            sphere.update(args_cli.dt)
            if step >= args_cli.steps - settling_window:
                samples[case_name].append(sphere.data.root_pos_w[0, 2].detach().cpu().clone())
        for case_name, sdf_object in sdf_objects.items():
            sdf_object.update(args_cli.dt)
            if step >= args_cli.steps - settling_window:
                sdf_samples[case_name].append(sdf_object.data.root_pos_w[0, 2].detach().cpu().clone())

    primitive_passed = _evaluate_results(_stack_samples(samples))
    sdf_passed = _evaluate_sdf_results(_stack_samples(sdf_samples)) if args_cli.include_sdf else True
    passed = primitive_passed and sdf_passed
    print(f"[SUMMARY] compliant_contact_diagnostic={'PASS' if passed else 'FAIL'}")
    return 0 if passed else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    finally:
        simulation_app.close()
