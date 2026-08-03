# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import importlib.util
import math
from pathlib import Path
from xml.etree import ElementTree

import pytest
import torch
from assembly_benchmark.assembly.fabrica.plans import (
    BEAM_DISASSEMBLY_PATH,
    BEAM_FABRICA_PLAN,
    PIPER_FABRICA_BASE_POS,
    PIPER_FABRICA_GRIPPER_BASE_POS,
    PIPER_FABRICA_GRIPPER_BASE_ROTATION_RPY,
    assign_fabrica_relations,
    available_fabrica_assemblies,
    load_fabrica_assembly_plan,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
PIPER_URDF_PATH = REPO_ROOT / "source" / "assembly_benchmark" / "assembly_benchmark" / "utils" / "piper_urdf.py"
PIPER_URDF_SPEC = importlib.util.spec_from_file_location("piper_urdf", PIPER_URDF_PATH)
assert PIPER_URDF_SPEC is not None and PIPER_URDF_SPEC.loader is not None
PIPER_URDF = importlib.util.module_from_spec(PIPER_URDF_SPEC)
PIPER_URDF_SPEC.loader.exec_module(PIPER_URDF)
compose_piper_pika2_urdf = PIPER_URDF.compose_piper_pika2_urdf
set_piper_gripper_base_transform = PIPER_URDF.set_piper_gripper_base_transform

PIPER_ASSET_DIR = REPO_ROOT / "source" / "assembly_benchmark" / "assembly_benchmark" / "assets" / "robots" / "piper"
BEAM_MESH_DIR = (
    REPO_ROOT / "source" / "assembly_benchmark" / "assembly_benchmark" / "assets" / "fabrica" / "beam" / "mesh"
)
PIPER_BASE_URDF_PATH = PIPER_ASSET_DIR / "piper_description.urdf"
PIPER_GRIPPER_URDF_PATH = PIPER_ASSET_DIR / "pika2_gripper.urdf"
PIPER_ARM_JOINT_NAMES = [f"joint{index}" for index in range(1, 7)]
EXPECTED_PIPER_GRIPPER_WORLD_QUATS = {
    "0_2": (
        1.88596520563597e-08,
        -0.7071067635203778,
        -0.7071067988527163,
        -1.8859651044337178e-08,
    ),
    "1_3": (
        -0.05765063622328482,
        -0.7047527270303522,
        -0.7047527155170007,
        0.05765074030527316,
    ),
    "2_6": (
        0.08438791434846613,
        3.414997819528587e-07,
        -0.9964329781334014,
        2.761439965581758e-09,
    ),
    "3_6": (
        -0.16422890509255497,
        -0.6877707156418048,
        -0.6877711668665806,
        0.164228899602483,
    ),
}
EXPECTED_BEAM_CONTACT_Z_BOUNDS = {
    "0_2": (0.17636363500000002, 0.18831900000000001),
    "1_3": (0.1694549082920716, 0.1883139479095236),
    "2_6": (0.1503144923187837, 0.20137407242565047),
    "3_6": (0.15326204104197927, 0.20277439551184873),
}
EXPECTED_BEAM_TIP_OFFSETS = {
    "0_2": 0.0,
    "1_3": 0.0,
    "2_6": 0.025,
    "3_6": 0.025,
}


def _quat_multiply(first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
    w1, x1, y1, z1 = first.unbind()
    w2, x2, y2, z2 = second.unbind()
    return torch.stack(
        (
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        )
    )


def _quat_apply(quaternion: torch.Tensor, vector: torch.Tensor) -> torch.Tensor:
    quaternion = quaternion / torch.linalg.vector_norm(quaternion)
    vector_part = quaternion[1:]
    twice_cross = 2.0 * torch.linalg.cross(vector_part, vector)
    return vector + quaternion[0] * twice_cross + torch.linalg.cross(vector_part, twice_cross)


def _quat_apply_points(quaternion: torch.Tensor, points: torch.Tensor) -> torch.Tensor:
    quaternion = quaternion / torch.linalg.vector_norm(quaternion)
    vector_part = quaternion[1:].expand_as(points)
    twice_cross = 2.0 * torch.linalg.cross(vector_part, points, dim=-1)
    return points + quaternion[0] * twice_cross + torch.linalg.cross(vector_part, twice_cross, dim=-1)


def _obj_mesh(path: Path) -> tuple[torch.Tensor, torch.Tensor]:
    vertices = []
    faces = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("v "):
            vertices.append(tuple(float(value) for value in line.split()[1:4]))
        elif line.startswith("f "):
            vertex_ids = [int(value.split("/")[0]) - 1 for value in line.split()[1:]]
            faces.extend(
                (vertex_ids[0], vertex_ids[index], vertex_ids[index + 1]) for index in range(1, len(vertex_ids) - 1)
            )
    if not vertices:
        raise ValueError(f"OBJ mesh contains no vertices: {path}")
    if not faces:
        raise ValueError(f"OBJ mesh contains no faces: {path}")
    return torch.tensor(vertices, dtype=torch.float64), torch.tensor(faces, dtype=torch.long)


def _collada_triangles(path: Path) -> torch.Tensor:
    root = ElementTree.parse(path).getroot()
    namespace_uri = root.tag.removeprefix("{").split("}", maxsplit=1)[0]
    namespace = {"c": namespace_uri}
    triangles = []
    for geometry in root.findall(".//c:library_geometries/c:geometry", namespace):
        sources = {}
        for source in geometry.findall("./c:mesh/c:source", namespace):
            float_array = source.find("./c:float_array", namespace)
            accessor = source.find("./c:technique_common/c:accessor", namespace)
            if float_array is None or accessor is None:
                continue
            stride = int(accessor.get("stride", "1"))
            values = torch.tensor([float(value) for value in float_array.text.split()], dtype=torch.float64).reshape(
                -1, stride
            )
            sources[f"#{source.get('id')}"] = values

        vertices = {}
        for vertex_set in geometry.findall("./c:mesh/c:vertices", namespace):
            position_input = vertex_set.find("./c:input[@semantic='POSITION']", namespace)
            if position_input is not None:
                vertices[f"#{vertex_set.get('id')}"] = sources[position_input.get("source")][:, :3]

        for triangle_set in geometry.findall("./c:mesh/c:triangles", namespace):
            inputs = triangle_set.findall("./c:input", namespace)
            vertex_input = next(item for item in inputs if item.get("semantic") == "VERTEX")
            stride = max(int(item.get("offset", "0")) for item in inputs) + 1
            vertex_offset = int(vertex_input.get("offset", "0"))
            indices = torch.tensor(
                [int(value) for value in triangle_set.find("./c:p", namespace).text.split()], dtype=torch.long
            ).reshape(-1, stride)[:, vertex_offset]
            triangles.append(vertices[vertex_input.get("source")][indices.reshape(-1, 3)])
    if not triangles:
        raise ValueError(f"COLLADA mesh contains no triangles: {path}")
    return torch.cat(triangles)


def _triangle_normals(triangles: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    normals = torch.linalg.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    lengths = torch.linalg.vector_norm(normals, dim=-1)
    valid = lengths > 1.0e-12
    normals[valid] /= lengths[valid, None]
    return normals, valid


def _compose_pose(first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
    return torch.cat(
        (
            first[:3] + _quat_apply(first[3:7], second[:3]),
            _quat_multiply(first[3:7], second[3:7]),
        )
    )


def _rpy_quaternion(rpy: tuple[float, float, float]) -> torch.Tensor:
    roll, pitch, yaw = rpy
    qx = torch.tensor((math.cos(roll / 2.0), math.sin(roll / 2.0), 0.0, 0.0), dtype=torch.float64)
    qy = torch.tensor((math.cos(pitch / 2.0), 0.0, math.sin(pitch / 2.0), 0.0), dtype=torch.float64)
    qz = torch.tensor((math.cos(yaw / 2.0), 0.0, 0.0, math.sin(yaw / 2.0)), dtype=torch.float64)
    return _quat_multiply(_quat_multiply(qz, qy), qx)


def _link_poses(root: ElementTree.Element, joint_positions: dict[str, float]) -> dict[str, torch.Tensor]:
    link_names = {link.get("name") for link in root.findall("link")}
    child_names = {joint.find("child").get("link") for joint in root.findall("joint")}
    root_link = next(iter(link_names - child_names))
    poses = {root_link: torch.tensor((0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0), dtype=torch.float64)}
    pending = list(root.findall("joint"))
    while pending:
        progressed = False
        for joint in pending.copy():
            parent = joint.find("parent").get("link")
            if parent not in poses:
                continue
            child = joint.find("child").get("link")
            origin = joint.find("origin")
            xyz_text = origin.get("xyz", "0 0 0") if origin is not None else "0 0 0"
            rpy_text = origin.get("rpy", "0 0 0") if origin is not None else "0 0 0"
            xyz = tuple(float(value) for value in xyz_text.split())
            rpy = tuple(float(value) for value in rpy_text.split())
            parent_to_joint = torch.cat((torch.tensor(xyz, dtype=torch.float64), _rpy_quaternion(rpy)))
            motion = torch.tensor((0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0), dtype=torch.float64)
            value = joint_positions.get(joint.get("name"), 0.0)
            axis_element = joint.find("axis")
            axis = torch.tensor(
                tuple(
                    float(item)
                    for item in (axis_element.get("xyz", "1 0 0") if axis_element is not None else "1 0 0").split()
                ),
                dtype=torch.float64,
            )
            axis /= torch.linalg.vector_norm(axis)
            if joint.get("type") in ("revolute", "continuous"):
                motion[3] = math.cos(value / 2.0)
                motion[4:7] = axis * math.sin(value / 2.0)
            elif joint.get("type") == "prismatic":
                motion[:3] = axis * value
            poses[child] = _compose_pose(poses[parent], _compose_pose(parent_to_joint, motion))
            pending.remove(joint)
            progressed = True
        if not progressed:
            raise ValueError(f"Could not resolve joints: {[joint.get('name') for joint in pending]}")
    return poses


def _origin_pose(origin: ElementTree.Element | None) -> torch.Tensor:
    xyz_text = origin.get("xyz", "0 0 0") if origin is not None else "0 0 0"
    rpy_text = origin.get("rpy", "0 0 0") if origin is not None else "0 0 0"
    xyz = tuple(float(value) for value in xyz_text.split())
    rpy = tuple(float(value) for value in rpy_text.split())
    return torch.cat((torch.tensor(xyz, dtype=torch.float64), _rpy_quaternion(rpy)))


def _transform_triangles(pose: torch.Tensor, triangles: torch.Tensor) -> torch.Tensor:
    points = triangles.reshape(-1, 3)
    return (_quat_apply_points(pose[3:7], points) + pose[:3]).reshape_as(triangles)


def _pika2_inner_pad_bounds() -> tuple[tuple[float, float], tuple[float, float]]:
    root = ElementTree.parse(PIPER_GRIPPER_URDF_PATH).getroot()
    link_poses = _link_poses(root, {})
    pad_points = []
    for side, inward_direction in (("left", -1.0), ("right", 1.0)):
        link_name = f"gripper_{side}_link"
        visual = root.find(f"./link[@name='{link_name}']/visual")
        mesh_pose = _compose_pose(link_poses[link_name], _origin_pose(visual.find("origin")))
        triangles = _transform_triangles(
            mesh_pose,
            _collada_triangles(PIPER_ASSET_DIR / "meshes" / "dae" / f"{link_name}.dae"),
        )
        normals, valid = _triangle_normals(triangles)
        if side == "left":
            near_inner_surface = triangles[:, :, 1].amin(dim=1) <= 2.0e-4
        else:
            near_inner_surface = triangles[:, :, 1].amax(dim=1) >= -2.0e-4
        pad_points.append(triangles[valid & (normals[:, 1] * inward_direction > 0.95) & near_inner_surface])

    points = torch.cat(pad_points).reshape(-1, 3)
    x_bounds = (float(points[:, 0].amin()), float(points[:, 0].amax()))
    # Use the 0.1 mm nominal contact envelope measured from both symmetric DAE finger meshes.
    z_bounds = tuple(
        float(value) for value in (torch.round(torch.stack((points[:, 2].amin(), points[:, 2].amax())) * 1.0e4) / 1.0e4)
    )
    return x_bounds, z_bounds


def _beam_contact_bounds(relation, pad_x_bounds: tuple[float, float]) -> tuple[torch.Tensor, torch.Tensor]:
    vertices, faces = _obj_mesh(BEAM_MESH_DIR / f"beam_part_{relation.plug_part_id}.obj")
    quaternion = torch.tensor(relation.piper_gripper_to_plug_quat, dtype=torch.float64)
    position = torch.tensor(relation.piper_gripper_to_plug_pos, dtype=torch.float64)
    vertices_gripper = _quat_apply_points(quaternion, vertices) + position
    triangles = vertices_gripper[faces]
    normals, valid = _triangle_normals(triangles)
    part_lower = vertices_gripper.amin(dim=0)
    part_upper = vertices_gripper.amax(dim=0)
    contact_points = []
    for outward_direction, surface_y in ((1.0, part_upper[1]), (-1.0, part_lower[1])):
        triangle_center_y = triangles[:, :, 1].mean(dim=1)
        near_outer_surface = torch.abs(triangle_center_y - surface_y) < 2.0e-4
        overlaps_pad_x = (triangles[:, :, 0].amax(dim=1) > pad_x_bounds[0]) & (
            triangles[:, :, 0].amin(dim=1) < pad_x_bounds[1]
        )
        surface = triangles[valid & (normals[:, 1] * outward_direction > 0.9) & near_outer_surface & overlaps_pad_x]
        if not len(surface):
            raise ValueError(f"Beam relation {relation.key} has no side contact surface for the Pika2 pad.")
        contact_points.append(surface)
    return vertices_gripper, torch.cat(contact_points).reshape(-1, 3)


def test_beam_fabrica_plan_is_safe_complete_and_self_contained() -> None:
    assert available_fabrica_assemblies() == ("beam",)
    assert load_fabrica_assembly_plan("beam") is BEAM_FABRICA_PLAN
    assert BEAM_FABRICA_PLAN.relation_keys == ("0_2", "1_3", "2_6", "3_6")
    assert BEAM_FABRICA_PLAN.part_scene_key("6") == "beam_part_6"
    assert len(BEAM_DISASSEMBLY_PATH) == 10
    assert BEAM_DISASSEMBLY_PATH[0] == pytest.approx((0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0))
    assert BEAM_DISASSEMBLY_PATH[-1] == pytest.approx((0.0, 0.0, 0.0202000019, 1.0, 0.0, 0.0, 0.0))
    for relation in BEAM_FABRICA_PLAN.relations:
        assert relation.disassembly_path is BEAM_DISASSEMBLY_PATH
        assert relation.path_length == pytest.approx(0.0202000019)
        assert len(relation.piper_preassembly_joint_pos) == 6
        assert 0.011 < relation.piper_gripper_opening <= 0.0127


def test_relation_assignment_matches_original_specialist_rotation() -> None:
    assert assign_fabrica_relations(0, 4) == ()
    assert assign_fabrica_relations(10, 4) == (0, 1, 2, 3, 0, 1, 2, 3, 0, 1)
    with pytest.raises(ValueError, match="positive"):
        assign_fabrica_relations(4, 0)


def test_piper_and_pika2_urdfs_compose_without_ros_dependencies() -> None:
    root = compose_piper_pika2_urdf(PIPER_BASE_URDF_PATH, PIPER_GRIPPER_URDF_PATH, PIPER_ASSET_DIR).getroot()
    link_names = {link.get("name") for link in root.findall("link")}
    assert {"base_link", "link6", "gripper_base_link", "gripper_left_link", "gripper_right_link"} <= link_names
    assert "world" not in link_names

    base_joint = root.find("./joint[@name='gripper_base_joint']")
    assert base_joint is not None
    assert base_joint.find("parent").get("link") == "link6"
    assert base_joint.find("child").get("link") == "gripper_base_link"
    assert base_joint.find("origin").get("xyz") == "0 0 0"
    assert base_joint.find("origin").get("rpy") == "0 0 0"

    left_limit = root.find("./joint[@name='left_joint']/limit")
    right_limit = root.find("./joint[@name='right_joint']/limit")
    assert (float(left_limit.get("lower")), float(left_limit.get("upper"))) == pytest.approx((0.0, 0.05))
    assert (float(right_limit.get("lower")), float(right_limit.get("upper"))) == pytest.approx((-0.05, 0.0))
    for mesh in root.findall(".//mesh"):
        mesh_path = Path(mesh.get("filename"))
        assert mesh_path.is_absolute()
        assert mesh_path.is_file()


def test_all_piper_preassembly_poses_place_plug_at_same_world_anchor() -> None:
    tree = compose_piper_pika2_urdf(PIPER_BASE_URDF_PATH, PIPER_GRIPPER_URDF_PATH, PIPER_ASSET_DIR)
    set_piper_gripper_base_transform(
        tree.getroot(), PIPER_FABRICA_GRIPPER_BASE_POS, PIPER_FABRICA_GRIPPER_BASE_ROTATION_RPY
    )
    base_pose = torch.tensor((*PIPER_FABRICA_BASE_POS, 1.0, 0.0, 0.0, 0.0), dtype=torch.float64)
    expected_plug = torch.tensor((0.55, 0.30, 0.7952000377, 1.0, 0.0, 0.0, 0.0), dtype=torch.float64)
    joints = {joint.get("name"): joint for joint in tree.getroot().findall("joint")}

    for relation in BEAM_FABRICA_PLAN.relations:
        joint_positions = dict(zip(PIPER_ARM_JOINT_NAMES, relation.piper_preassembly_joint_pos, strict=True))
        for joint_name, value in joint_positions.items():
            limit = joints[joint_name].find("limit")
            assert float(limit.get("lower")) <= value <= float(limit.get("upper"))
        poses = _link_poses(tree.getroot(), joint_positions)
        gripper_world = _compose_pose(base_pose, poses["gripper_base_link"])
        expected_gripper_world_quat = torch.tensor(
            EXPECTED_PIPER_GRIPPER_WORLD_QUATS[relation.key], dtype=torch.float64
        )
        assert abs(float(torch.dot(gripper_world[3:7], expected_gripper_world_quat))) == pytest.approx(1.0, abs=2.0e-7)
        if relation.key == "1_3":
            joint5_upper = float(joints["joint5"].find("limit").get("upper"))
            assert joint5_upper - joint_positions["joint5"] >= math.radians(0.4)
        gripper_to_plug = torch.tensor(
            (*relation.piper_gripper_to_plug_pos, *relation.piper_gripper_to_plug_quat), dtype=torch.float64
        )
        plug_world = _compose_pose(gripper_world, gripper_to_plug)
        assert plug_world[:3].tolist() == pytest.approx(expected_plug[:3].tolist(), abs=2.0e-7)
        assert abs(float(torch.dot(plug_world[3:7], expected_plug[3:7]))) == pytest.approx(1.0, abs=2.0e-7)


def test_all_beam_plugs_use_the_pika2_inner_finger_pad_regions() -> None:
    pad_x_bounds, pad_z_bounds = _pika2_inner_pad_bounds()
    assert pad_z_bounds == pytest.approx((0.1208, 0.1887))
    for relation in BEAM_FABRICA_PLAN.relations:
        vertices_gripper, contact_points = _beam_contact_bounds(relation, pad_x_bounds)
        lower = vertices_gripper.amin(dim=0)
        upper = vertices_gripper.amax(dim=0)
        contact_lower = contact_points.amin(dim=0)
        contact_upper = contact_points.amax(dim=0)

        jaw_center_y = 0.5 * (lower[1] + upper[1])
        plug_thickness_y = upper[1] - lower[1]
        jaw_clearance = relation.piper_gripper_opening - plug_thickness_y
        contact_overlap_x = min(float(contact_upper[0]), pad_x_bounds[1]) - max(
            float(contact_lower[0]), pad_x_bounds[0]
        )
        contact_overlap_z = min(float(contact_upper[2]), pad_z_bounds[1]) - max(
            float(contact_lower[2]), pad_z_bounds[0]
        )

        assert abs(float(jaw_center_y)) < 1.0e-5
        assert -5.0e-8 <= float(jaw_clearance) <= 5.0e-4
        assert contact_overlap_x > 0.01
        assert contact_overlap_z > 0.01
        assert float(upper[2] - pad_z_bounds[1]) == pytest.approx(EXPECTED_BEAM_TIP_OFFSETS[relation.key], abs=1.0e-5)
        assert (float(contact_lower[2]), float(contact_upper[2])) == pytest.approx(
            EXPECTED_BEAM_CONTACT_Z_BOUNDS[relation.key], abs=1.0e-5
        )
