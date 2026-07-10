# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import importlib
import re
from math import isclose, sqrt
from pathlib import Path

import pytest

from assembly_benchmark.assembly import (
    DEFAULT_ORI_BOUND,
    DEFAULT_POS_THRESHOLD,
    AssemblyPartSpec,
    AssemblyRelationSpec,
    AssemblySpec,
    AssemblyTargetPose,
    available_assemblies,
    make_assembly,
    make_cabinet_assembly,
    make_chair_assembly,
    make_desk_assembly,
    make_drawer_assembly,
    make_lamp_assembly,
    make_one_leg_assembly,
    make_round_table_assembly,
    make_square_table_assembly,
    make_stool_assembly,
    register_assembly,
)


EXPECTED_ASSEMBLIES = (
    "cabinet",
    "chair",
    "desk",
    "drawer",
    "lamp",
    "one_leg",
    "round_table",
    "square_table",
    "stool",
)
ONE_LEG_PART_NAMES = (
    "base_tag",
    "square_table_top",
    "square_table_leg1",
    "square_table_leg2",
    "square_table_leg3",
    "square_table_leg4",
)
CHAIR_PART_NAMES = (
    "base_tag",
    "chair_seat",
    "chair_leg1",
    "chair_leg2",
    "chair_back",
    "chair_nut1",
    "chair_nut2",
)
EXPECTED_PART_NAMES = {
    "cabinet": ("base_tag", "cabinet_body", "cabinet_door_left", "cabinet_door_right", "cabinet_top"),
    "chair": CHAIR_PART_NAMES,
    "desk": ("base_tag", "desk_top", "desk_leg1", "desk_leg2", "desk_leg3", "desk_leg4"),
    "drawer": ("base_tag", "drawer_box", "drawer_container_top", "drawer_container_bottom"),
    "lamp": ("base_tag", "lamp_base", "lamp_bulb", "lamp_hood"),
    "one_leg": ONE_LEG_PART_NAMES,
    "round_table": ("base_tag", "round_table_top", "round_table_leg", "round_table_base"),
    "square_table": ONE_LEG_PART_NAMES,
    "stool": ("base_tag", "stool_seat", "stool_leg1", "stool_leg2", "stool_leg3"),
}
EXPECTED_RELATIONS = {
    "cabinet": (
        ("cabinet_body", "cabinet_door_right", 0, 1, DEFAULT_ORI_BOUND),
        ("cabinet_body", "cabinet_door_left", 0, 1, DEFAULT_ORI_BOUND),
        ("cabinet_body", "cabinet_top", 0, 1, DEFAULT_ORI_BOUND),
    ),
    "chair": (
        ("chair_seat", "chair_leg1", 1, 2, DEFAULT_ORI_BOUND),
        ("chair_seat", "chair_leg2", 0, 2, DEFAULT_ORI_BOUND),
        ("chair_seat", "chair_back", 0, 1, DEFAULT_ORI_BOUND),
        ("chair_seat", "chair_nut1", 1, 2, DEFAULT_ORI_BOUND),
        ("chair_seat", "chair_nut2", 0, 2, DEFAULT_ORI_BOUND),
    ),
    "desk": (
        ("desk_top", "desk_leg1", 0, 4, DEFAULT_ORI_BOUND),
        ("desk_top", "desk_leg2", 1, 4, DEFAULT_ORI_BOUND),
        ("desk_top", "desk_leg3", 2, 4, DEFAULT_ORI_BOUND),
        ("desk_top", "desk_leg4", 3, 4, DEFAULT_ORI_BOUND),
    ),
    "drawer": (
        ("drawer_box", "drawer_container_top", 0, 2, DEFAULT_ORI_BOUND),
        ("drawer_box", "drawer_container_bottom", 1, 2, DEFAULT_ORI_BOUND),
    ),
    "lamp": (
        ("lamp_base", "lamp_bulb", 0, 1, DEFAULT_ORI_BOUND),
        ("lamp_base", "lamp_hood", 0, 1, -1.0),
    ),
    "one_leg": (("square_table_top", "square_table_leg4", 0, 4, DEFAULT_ORI_BOUND),),
    "round_table": (
        ("round_table_top", "round_table_leg", 0, 1, DEFAULT_ORI_BOUND),
        ("round_table_leg", "round_table_base", 0, 1, DEFAULT_ORI_BOUND),
    ),
    "square_table": (
        ("square_table_top", "square_table_leg1", 0, 4, DEFAULT_ORI_BOUND),
        ("square_table_top", "square_table_leg2", 1, 4, DEFAULT_ORI_BOUND),
        ("square_table_top", "square_table_leg3", 2, 4, DEFAULT_ORI_BOUND),
        ("square_table_top", "square_table_leg4", 3, 4, DEFAULT_ORI_BOUND),
    ),
    "stool": (
        ("stool_seat", "stool_leg1", 0, 3, DEFAULT_ORI_BOUND),
        ("stool_seat", "stool_leg2", 1, 3, DEFAULT_ORI_BOUND),
        ("stool_seat", "stool_leg3", 2, 3, DEFAULT_ORI_BOUND),
    ),
}
ASSEMBLY_FACTORIES = {
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


def _test_part(scene_key: str) -> AssemblyPartSpec:
    return AssemblyPartSpec(
        scene_key=scene_key,
        asset_name=scene_key,
        prim_name=scene_key.title().replace("_", ""),
        urdf_rel_path=Path(f"urdf/{scene_key}.urdf"),
        init_pos=(0.0, 0.0, 0.0),
        init_rot=(1.0, 0.0, 0.0, 0.0),
        body_type="dynamic",
        mass=0.1,
    )


def _assert_normalized_quat(quat: tuple[float, float, float, float]) -> None:
    assert len(quat) == 4
    norm = sqrt(sum(component * component for component in quat))
    assert isclose(norm, 1.0, abs_tol=1.0e-6)


def _assert_tuple_close(
    actual: tuple[float, ...],
    expected: tuple[float, ...],
    *,
    abs_tol: float = 1.0e-10,
) -> None:
    assert len(actual) == len(expected)
    for actual_value, expected_value in zip(actual, expected, strict=True):
        assert isclose(actual_value, expected_value, abs_tol=abs_tol)


def test_assembly_registry_contract() -> None:
    assert available_assemblies() == EXPECTED_ASSEMBLIES
    for assembly_name, factory in ASSEMBLY_FACTORIES.items():
        with pytest.raises(ValueError, match="already registered"):
            register_assembly(assembly_name, factory)


def test_assembly_spec_rejects_duplicate_part_names() -> None:
    with pytest.raises(ValueError, match="duplicate part scene keys"):
        AssemblySpec(
            name="bad_duplicate",
            asset_root=Path("/tmp"),
            parts=(_test_part("part"), _test_part("part")),
            assembly_relations=(),
        )


def test_assembly_spec_rejects_unknown_relation_parts() -> None:
    with pytest.raises(ValueError, match="relation child 'missing'"):
        AssemblySpec(
            name="bad_relation",
            asset_root=Path("/tmp"),
            parts=(_test_part("parent"),),
            assembly_relations=(
                AssemblyRelationSpec(
                    parent="parent",
                    child="missing",
                    target_poses=(AssemblyTargetPose(pos=(0.0, 0.0, 0.0)),),
                ),
            ),
        )


def test_assembly_spec_rejects_invalid_default_target_index() -> None:
    with pytest.raises(ValueError, match="default target index"):
        AssemblySpec(
            name="bad_target",
            asset_root=Path("/tmp"),
            parts=(_test_part("parent"), _test_part("child")),
            assembly_relations=(
                AssemblyRelationSpec(
                    parent="parent",
                    child="child",
                    target_poses=(AssemblyTargetPose(pos=(0.0, 0.0, 0.0)),),
                    default_target_index=1,
                ),
            ),
        )


def test_assembly_spec_rejects_invalid_dynamic_physics_params() -> None:
    with pytest.raises(ValueError, match="exactly one of mass or density"):
        AssemblySpec(
            name="missing_physics",
            asset_root=Path("/tmp"),
            parts=(
                AssemblyPartSpec(
                    scene_key="part",
                    asset_name="part",
                    prim_name="Part",
                    urdf_rel_path=Path("urdf/part.urdf"),
                    init_pos=(0.0, 0.0, 0.0),
                    init_rot=(1.0, 0.0, 0.0, 0.0),
                    body_type="dynamic",
                ),
            ),
            assembly_relations=(),
        )

    with pytest.raises(ValueError, match="exactly one of mass or density"):
        AssemblySpec(
            name="duplicate_physics",
            asset_root=Path("/tmp"),
            parts=(
                AssemblyPartSpec(
                    scene_key="part",
                    asset_name="part",
                    prim_name="Part",
                    urdf_rel_path=Path("urdf/part.urdf"),
                    init_pos=(0.0, 0.0, 0.0),
                    init_rot=(1.0, 0.0, 0.0, 0.0),
                    body_type="dynamic",
                    mass=0.1,
                    density=100.0,
                ),
            ),
            assembly_relations=(),
        )


@pytest.mark.parametrize("assembly_name", EXPECTED_ASSEMBLIES)
def test_flat_assembly_module_exports(assembly_name: str) -> None:
    module = importlib.import_module(f"assembly_benchmark.assembly.{assembly_name}")
    factory = getattr(module, f"make_{assembly_name}_assembly")

    assert module.ASSEMBLY_NAME == assembly_name
    assert module.ASSET_ROOT.name == assembly_name
    assert module.PARTS
    assert module.RELATIONS
    assert factory().name == assembly_name


@pytest.mark.parametrize("assembly_name", EXPECTED_ASSEMBLIES)
def test_assembly_contract(assembly_name: str) -> None:
    assembly = make_assembly(assembly_name)

    assert assembly.name == assembly_name
    assert assembly.asset_root.name == assembly_name
    assert assembly.part_names[0] == "base_tag"
    assert assembly.observation_part_names == ()
    assert all(not part.observe for part in assembly.parts)

    scene_keys = [part.scene_key for part in assembly.parts]
    assert len(scene_keys) == len(set(scene_keys))
    for part in assembly.parts:
        _assert_normalized_quat(part.init_rot)
        if part.body_type == "dynamic":
            assert (part.mass is None) != (part.density is None)
            if part.mass is not None:
                assert part.mass > 0.0
            if part.density is not None:
                assert part.density > 0.0
            assert part.reset is True
        else:
            assert part.mass is None
            assert part.density is None
            assert part.reset is False

    base_tag = assembly.part("base_tag")
    assert base_tag.body_type == "visual"
    assert base_tag.tag_ids == (0, 1, 2, 3)

    for relation in assembly.assembly_relations:
        assert relation.parent in scene_keys
        assert relation.child in scene_keys
        assert 0 <= relation.default_target_index < len(relation.target_poses)
        assert relation.pos_threshold == DEFAULT_POS_THRESHOLD
        assert relation.ori_bound in (DEFAULT_ORI_BOUND, -1.0)
        for target_pose in relation.target_poses:
            _assert_normalized_quat(target_pose.quat)


@pytest.mark.parametrize("assembly_name", EXPECTED_ASSEMBLIES)
def test_assembly_part_order_contract(assembly_name: str) -> None:
    assembly = make_assembly(assembly_name)
    expected_part_names = EXPECTED_PART_NAMES[assembly_name]

    assert assembly.part_names == expected_part_names
    assert assembly.reset_part_names == expected_part_names[1:]


@pytest.mark.parametrize("assembly_name", EXPECTED_ASSEMBLIES)
def test_assembly_relation_order_contract(assembly_name: str) -> None:
    assembly = make_assembly(assembly_name)

    assert tuple(
        (
            relation.parent,
            relation.child,
            relation.default_target_index,
            len(relation.target_poses),
            relation.ori_bound,
        )
        for relation in assembly.assembly_relations
    ) == EXPECTED_RELATIONS[assembly_name]


@pytest.mark.parametrize("assembly_name", EXPECTED_ASSEMBLIES)
def test_source_asset_paths_exist(assembly_name: str) -> None:
    assembly = make_assembly(assembly_name)

    assert (assembly.asset_root / "mesh").is_dir()
    assert (assembly.asset_root / "urdf").is_dir()
    for part in assembly.parts:
        assert part.urdf_path(assembly.asset_root).is_file()


@pytest.mark.parametrize("assembly_name", EXPECTED_ASSEMBLIES)
def test_generated_usd_paths_exist_when_generated(assembly_name: str) -> None:
    assembly = make_assembly(assembly_name)

    if not (assembly.asset_root / "usd").is_dir():
        pytest.skip(f"{assembly_name} USD assets have not been generated in this checkout.")

    for part in assembly.parts:
        assert part.usd_path(assembly.asset_root).is_file()


@pytest.mark.parametrize("assembly_name", EXPECTED_ASSEMBLIES)
def test_usd_generation_asset_contract(assembly_name: str) -> None:
    assembly = make_assembly(assembly_name)
    generation_assets = {asset.asset_name: asset for asset in assembly.usd_generation_assets()}

    assert set(generation_assets) == {part.asset_name for part in assembly.parts}
    assert not generation_assets["base_tag"].is_dynamic
    for part in assembly.parts:
        asset = generation_assets[part.asset_name]
        assert asset.body_type == part.body_type
        assert asset.mass == part.mass
        assert asset.density == part.density
        if asset.is_dynamic:
            assert (asset.mass is None) != (asset.density is None)


@pytest.mark.parametrize("assembly_name", EXPECTED_ASSEMBLIES)
def test_urdf_mesh_references_are_self_contained(assembly_name: str) -> None:
    assembly = make_assembly(assembly_name)

    for urdf_path in (assembly.asset_root / "urdf").rglob("*.urdf"):
        for mesh_ref in re.findall(r'filename="([^"]+)"', urdf_path.read_text()):
            if mesh_ref.startswith(("package://", "/")):
                continue
            assert (urdf_path.parent / mesh_ref).resolve().is_file()


def test_one_leg_part_contract() -> None:
    assembly = make_assembly("one_leg")

    assert assembly.part_names == ONE_LEG_PART_NAMES
    assert assembly.reset_part_names == ONE_LEG_PART_NAMES[1:]

    top = assembly.part("square_table_top")
    leg4 = assembly.part("square_table_leg4")
    assert top.mass == 0.151
    assert leg4.mass == 0.0231
    assert top.init_pos == (0.7415, 0.0, 0.790625)
    assert leg4.init_pos == (0.5715, 0.2, 0.79)


def test_one_leg_relation_contract() -> None:
    relation = make_assembly("one_leg").primary_relation

    assert relation.parent == "square_table_top"
    assert relation.child == "square_table_leg4"
    assert tuple(target.pos for target in relation.target_poses) == (
        (-0.05625, 0.046875, -0.05625),
        (0.05625, 0.046875, -0.05625),
        (-0.05625, 0.046875, 0.05625),
        (0.05625, 0.046875, 0.05625),
    )


def test_one_leg_generation_assets() -> None:
    assembly = make_assembly("one_leg")
    generation_assets = {asset.asset_name: asset for asset in assembly.usd_generation_assets()}

    assert set(generation_assets) == set(ONE_LEG_PART_NAMES)
    assert generation_assets["square_table_top"].is_dynamic
    assert generation_assets["square_table_top"].mass == 0.151
    assert generation_assets["square_table_top"].density is None
    assert not generation_assets["base_tag"].is_dynamic


def test_chair_part_contract() -> None:
    assembly = make_assembly("chair")

    assert assembly.part_names == CHAIR_PART_NAMES
    assert assembly.reset_part_names == CHAIR_PART_NAMES[1:]

    expected_parts = {
        "chair_seat": ((0.7715, 0.03, 0.79), 0.06187, (79, 80, 81, 82)),
        "chair_leg1": ((0.6215, 0.01, 0.79), 0.02244, (91, 92, 93, 94)),
        "chair_leg2": ((0.6215, 0.07, 0.79), 0.02244, (95, 96, 97, 98)),
        "chair_back": ((0.7015, -0.10, 0.79), 0.12316, tuple(range(83, 91))),
        "chair_nut1": ((0.6215, 0.13, 0.79), 0.01015, tuple(range(99, 104))),
        "chair_nut2": ((0.6215, 0.19, 0.79), 0.01015, tuple(range(104, 109))),
    }
    for part_name, (init_pos, mass, tag_ids) in expected_parts.items():
        part = assembly.part(part_name)
        assert part.init_pos == init_pos
        assert part.mass == mass
        assert part.tag_ids == tag_ids


def test_chair_relation_contract() -> None:
    assembly = make_assembly("chair")
    relations = assembly.assembly_relations

    assert tuple((relation.parent, relation.child) for relation in relations) == (
        ("chair_seat", "chair_leg1"),
        ("chair_seat", "chair_leg2"),
        ("chair_seat", "chair_back"),
        ("chair_seat", "chair_nut1"),
        ("chair_seat", "chair_nut2"),
    )
    assert tuple(relation.default_target_index for relation in relations) == (1, 0, 0, 1, 0)

    leg_target_quat = (0.7071067812, 0.0, 0.7071067812, 0.0)
    nut_target_quat = (0.7071067812, -0.7071067812, 0.0, 0.0)
    expected_targets = (
        (
            ((-0.03375, 0.045, -0.01875), (0.03375, 0.045, -0.01875)),
            (leg_target_quat, leg_target_quat),
        ),
        (
            ((-0.03375, 0.045, -0.01875), (0.03375, 0.045, -0.01875)),
            (leg_target_quat, leg_target_quat),
        ),
        (((0.0, -0.0325, 0.05025),), ((1.0, 0.0, 0.0, 0.0),)),
        (
            ((0.035, 0.0, 0.0795), (-0.035, 0.0, 0.0795)),
            (nut_target_quat, nut_target_quat),
        ),
        (
            ((0.035, 0.0, 0.0795), (-0.035, 0.0, 0.0795)),
            (nut_target_quat, nut_target_quat),
        ),
    )

    for relation, (target_positions, target_quats) in zip(
        relations,
        expected_targets,
        strict=True,
    ):
        assert tuple(target.pos for target in relation.target_poses) == target_positions
        for target_pose, expected_quat in zip(relation.target_poses, target_quats, strict=True):
            _assert_tuple_close(target_pose.quat, expected_quat)


def test_chair_generation_assets() -> None:
    assembly = make_assembly("chair")
    generation_assets = {asset.asset_name: asset for asset in assembly.usd_generation_assets()}

    assert set(generation_assets) == set(CHAIR_PART_NAMES)
    assert not generation_assets["base_tag"].is_dynamic
    assert generation_assets["chair_seat"].mass == 0.06187
    assert generation_assets["chair_seat"].density is None
    assert generation_assets["chair_back"].mass == 0.12316
    assert generation_assets["chair_leg1"].mass == 0.02244
    assert generation_assets["chair_leg2"].mass == 0.02244
    assert generation_assets["chair_nut1"].mass == 0.01015
    assert generation_assets["chair_nut2"].mass == 0.01015
