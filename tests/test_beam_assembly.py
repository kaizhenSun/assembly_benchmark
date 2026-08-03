# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
from assembly_benchmark.assembly import AssemblySpec, assembly_asset_root, make_assembly, make_kinematic_part
from assembly_benchmark.assembly.fabrica.beam import (
    BEAM_DENSITY,
    BEAM_PART_IDS,
    BEAM_PLUG_FRICTION,
    BEAM_RELATION_PART_IDS,
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


def test_complete_beam_assembly_spec_contract() -> None:
    assembly = make_assembly("beam")

    assert assembly.part_names == tuple(f"beam_part_{part_id}" for part_id in BEAM_PART_IDS)
    assert assembly.reset_part_names == assembly.part_names

    plug = assembly.part("beam_part_0")
    socket = assembly.part("beam_part_2")
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

    assert tuple((relation.child, relation.parent) for relation in assembly.assembly_relations) == tuple(
        (f"beam_part_{plug_id}", f"beam_part_{socket_id}") for plug_id, socket_id in BEAM_RELATION_PART_IDS
    )
    for relation in assembly.assembly_relations:
        assert relation.default_target_pose.pos == (0.0, 0.0, 0.0)
        assert relation.default_target_pose.quat == (1.0, 0.0, 0.0, 0.0)


def test_beam_assets_use_fabrica_layout() -> None:
    assembly = make_assembly("beam")
    asset_root = assembly.asset_root

    assert asset_root.name == "beam"
    assert asset_root.parent.name == "fabrica"
    assert assembly_asset_root("one_leg").parent.name == "furniture"
    assert not (asset_root.parent.parent / "furniture" / "beam").exists()
    for part in assembly.parts:
        assert part.urdf_path(asset_root).is_file()
        assert part.usd_path(asset_root).is_file()


def test_beam_analytic_preinsert_offset() -> None:
    assert pytest.approx(0.0202000377) == INSERTION_PATH_LENGTH
    assert PLUG_PREINSERT_POS[:2] == SOCKET_NOMINAL_POS[:2]
    assert PLUG_PREINSERT_POS[2] - SOCKET_NOMINAL_POS[2] == pytest.approx(INSERTION_PATH_LENGTH)


def test_beam_meshes_are_metre_scaled_without_recentering() -> None:
    asset_root = make_assembly("beam").asset_root
    mesh_root = asset_root / "mesh"

    expected_stats = {
        "0": ((3502, 7004), (-0.0508000878, -0.0381000033, 0.0634999069), (-0.0381000664, 0.0381000033, 0.0787399049)),
        "1": ((3502, 7004), (0.0380999130, -0.0381000033, 0.0634999069), (0.0507999344, 0.0381000033, 0.0787399049)),
        "2": ((2452, 4904), (-0.0502029070, -0.0060016625, -0.0000000534), (-0.0387000033, 0.0060016612, 0.0761999446)),
        "3": ((2452, 4904), (0.0386998474, -0.0060016624, -0.0000000533), (0.0502027512, 0.0060016613, 0.0761999447)),
        "6": (
            (12297, 24606),
            (-0.0762000042, -0.0088900244, -0.0000000229),
            (0.0762000042, 0.0088900250, 0.0126999771),
        ),
    }
    for part_id, (counts, lower, upper) in expected_stats.items():
        stats = _mesh_stats(mesh_root / f"beam_part_{part_id}.obj")
        assert stats[:2] == counts
        assert stats[2] == pytest.approx(lower)
        assert stats[3] == pytest.approx(upper)


def test_kinematic_part_generation_contract() -> None:
    assembly = make_assembly("beam")
    assets = {asset.asset_name: asset for asset in assembly.usd_generation_assets()}

    assert assets["beam_part_0"].is_dynamic
    assert assets["beam_part_0"].requires_free_root
    assert assets["beam_part_1"].is_dynamic
    assert assets["beam_part_2"].is_kinematic
    assert assets["beam_part_2"].requires_free_root
    assert assets["beam_part_3"].is_kinematic
    assert assets["beam_part_6"].is_kinematic

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
    for blob in (
        "3d2ee3952a22b7e5e7d91b1e7141dd589d99b4a4",
        "d61dfee8d01d807f181ed33e2b0ac08efec53f21",
        "9e2f09f1c36305e090819b24d01f26a83682260b",
        "783f1128d74f3a590eefaf5736cc8e8ff7ca3528",
        "6577f770e0002c8144d6bfb04b1099850701fe05",
    ):
        assert blob in notice
    assert "multiplied by `0.01`" in notice
    assert (asset_root / "LICENSE.Fabrica-MIT.txt").is_file()
    assert (asset_root / "LICENSE.Fabrica-Learning-BSD-3-Clause.txt").is_file()


@pytest.mark.skipif(not ISAAC_SIM_RUNTIME_AVAILABLE, reason="Isaac Sim runtime is not available")
def test_kinematic_part_uses_rigid_object_cfg() -> None:
    from assembly_benchmark.assembly.isaac import make_assembly_part_cfg

    from isaaclab.assets import RigidObjectCfg

    assembly = make_assembly("beam")
    cfg = make_assembly_part_cfg(assembly, "beam_part_2")

    assert isinstance(cfg, RigidObjectCfg)
    assert cfg.spawn is not None
    assert cfg.spawn.rigid_props is not None
    assert cfg.spawn.rigid_props.kinematic_enabled is True
    assert cfg.spawn.rigid_props.disable_gravity is True


@pytest.mark.skipif(not ISAAC_SIM_RUNTIME_AVAILABLE, reason="Isaac Sim USD runtime is not available")
def test_generated_beam_pair_uses_sdf_and_part_friction() -> None:
    from pxr import PhysxSchema, Usd, UsdPhysics, UsdShade

    asset_root = make_assembly("beam").asset_root
    expected_friction = {
        "beam_part_0": BEAM_PLUG_FRICTION,
        "beam_part_1": BEAM_PLUG_FRICTION,
        "beam_part_2": BEAM_SOCKET_FRICTION,
        "beam_part_3": BEAM_SOCKET_FRICTION,
        "beam_part_6": BEAM_SOCKET_FRICTION,
    }
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
