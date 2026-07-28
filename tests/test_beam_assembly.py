# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
from assembly_benchmark.assembly import AssemblySpec, make_assembly, make_kinematic_part
from assembly_benchmark.assembly.beam import (
    BEAM_DENSITY,
    BEAM_PLUG_FRICTION,
    BEAM_SOCKET_FRICTION,
    INSERTION_PATH_LENGTH,
    PLUG_PREINSERT_POS,
    SOCKET_NOMINAL_POS,
)

ISAAC_SIM_RUNTIME_AVAILABLE = (
    importlib.util.find_spec("carb") is not None and importlib.util.find_spec("pxr") is not None
)


def _mesh_stats(path: Path) -> tuple[int, int, tuple[float, float, float], tuple[float, float, float]]:
    vertices: list[tuple[float, float, float]] = []
    face_count = 0
    for line in path.read_text().splitlines():
        if line.startswith("v "):
            vertices.append(tuple(float(value) for value in line.split()[1:4]))
        elif line.startswith("f "):
            face_count += 1

    lower = tuple(min(vertex[axis] for vertex in vertices) for axis in range(3))
    upper = tuple(max(vertex[axis] for vertex in vertices) for axis in range(3))
    return len(vertices), face_count, lower, upper


def test_beam_pair_spec_contract() -> None:
    assembly = make_assembly("beam")

    assert assembly.part_names == ("beam_socket_2", "beam_plug_0")
    assert assembly.reset_part_names == assembly.part_names

    plug = assembly.part("beam_plug_0")
    socket = assembly.part("beam_socket_2")
    assert plug.asset_name == "beam_part_0"
    assert plug.body_type == "dynamic"
    assert plug.density == BEAM_DENSITY
    assert plug.friction == BEAM_PLUG_FRICTION
    assert plug.init_pos == PLUG_PREINSERT_POS
    assert socket.asset_name == "beam_part_2"
    assert socket.body_type == "kinematic"
    assert socket.density == BEAM_DENSITY
    assert socket.friction == BEAM_SOCKET_FRICTION
    assert socket.init_pos == SOCKET_NOMINAL_POS

    relation = assembly.primary_relation
    assert relation.parent == socket.scene_key
    assert relation.child == plug.scene_key
    assert relation.default_target_pose.pos == (0.0, 0.0, 0.0)
    assert relation.default_target_pose.quat == (1.0, 0.0, 0.0, 0.0)


def test_beam_analytic_preinsert_offset() -> None:
    assert pytest.approx(0.0202000377) == INSERTION_PATH_LENGTH
    assert PLUG_PREINSERT_POS[:2] == SOCKET_NOMINAL_POS[:2]
    assert PLUG_PREINSERT_POS[2] - SOCKET_NOMINAL_POS[2] == pytest.approx(INSERTION_PATH_LENGTH)


def test_beam_meshes_are_metre_scaled_without_recentering() -> None:
    asset_root = make_assembly("beam").asset_root
    mesh_root = asset_root / "mesh" / "beam"

    plug_stats = _mesh_stats(mesh_root / "beam_part_0.obj")
    socket_stats = _mesh_stats(mesh_root / "beam_part_2.obj")

    assert plug_stats[:2] == (3502, 7004)
    assert plug_stats[2] == pytest.approx((-0.0508000878, -0.0381000033, 0.0634999069))
    assert plug_stats[3] == pytest.approx((-0.0381000664, 0.0381000033, 0.0787399049))
    assert socket_stats[:2] == (2452, 4904)
    assert socket_stats[2] == pytest.approx((-0.0502029070, -0.0060016625, -0.0000000534))
    assert socket_stats[3] == pytest.approx((-0.0387000033, 0.0060016612, 0.0761999446))


def test_kinematic_part_generation_contract() -> None:
    assembly = make_assembly("beam")
    assets = {asset.asset_name: asset for asset in assembly.usd_generation_assets()}

    assert assets["beam_part_0"].is_dynamic
    assert assets["beam_part_0"].requires_free_root
    assert assets["beam_part_2"].is_kinematic
    assert assets["beam_part_2"].requires_free_root

    bad_part = make_kinematic_part(
        scene_key="bad_socket",
        asset_name="bad_socket",
        prim_name="BadSocket",
        urdf_rel_path="urdf/bad_socket.urdf",
        init_pos=(0.0, 0.0, 0.0),
        init_rot=(1.0, 0.0, 0.0, 0.0),
    )
    with pytest.raises(ValueError, match="exactly one of mass or density"):
        AssemblySpec(name="bad_kinematic", asset_root=Path("/tmp"), parts=(bad_part,), assembly_relations=())


def test_beam_asset_provenance_is_packaged() -> None:
    asset_root = make_assembly("beam").asset_root
    notice = (asset_root / "NOTICE.md").read_text()

    assert "215a30fe51e59299588a5b5e417a9cb934fb393e" in notice
    assert "multiplied by `0.01`" in notice
    assert (asset_root / "LICENSE.Fabrica-MIT.txt").is_file()
    assert (asset_root / "LICENSE.Fabrica-Learning-BSD-3-Clause.txt").is_file()


@pytest.mark.skipif(not ISAAC_SIM_RUNTIME_AVAILABLE, reason="Isaac Sim runtime is not available")
def test_kinematic_part_uses_rigid_object_cfg() -> None:
    from assembly_benchmark.assembly.isaac import make_assembly_part_cfg

    from isaaclab.assets import RigidObjectCfg

    assembly = make_assembly("beam")
    cfg = make_assembly_part_cfg(assembly, "beam_socket_2")

    assert isinstance(cfg, RigidObjectCfg)
    assert cfg.spawn is not None
    assert cfg.spawn.rigid_props is not None
    assert cfg.spawn.rigid_props.kinematic_enabled is True
    assert cfg.spawn.rigid_props.disable_gravity is True


@pytest.mark.skipif(not ISAAC_SIM_RUNTIME_AVAILABLE, reason="Isaac Sim USD runtime is not available")
def test_generated_beam_pair_uses_sdf_and_part_friction() -> None:
    from pxr import PhysxSchema, Usd, UsdPhysics, UsdShade

    asset_root = make_assembly("beam").asset_root
    expected_friction = {"beam_part_0": BEAM_PLUG_FRICTION, "beam_part_2": BEAM_SOCKET_FRICTION}
    for asset_name, friction in expected_friction.items():
        stage = Usd.Stage.Open(str(asset_root / "usd" / asset_name / f"{asset_name}.usd"))
        assert stage is not None
        colliders = [
            prim
            for prim in stage.TraverseAll()
            if prim.HasAPI(UsdPhysics.CollisionAPI) and UsdPhysics.MeshCollisionAPI(prim)
        ]
        assert len(colliders) == 1
        collider = colliders[0]
        assert UsdPhysics.MeshCollisionAPI(collider).GetApproximationAttr().Get() == PhysxSchema.Tokens.sdf
        assert PhysxSchema.PhysxSDFMeshCollisionAPI(collider).GetSdfResolutionAttr().Get() == 512

        material = UsdShade.MaterialBindingAPI(collider).GetDirectBinding("physics").GetMaterial()
        material_api = UsdPhysics.MaterialAPI(material.GetPrim())
        assert material_api.GetStaticFrictionAttr().Get() == pytest.approx(friction)
        assert material_api.GetDynamicFrictionAttr().Get() == pytest.approx(friction)
