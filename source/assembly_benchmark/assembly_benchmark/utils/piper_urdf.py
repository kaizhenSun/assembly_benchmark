# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""ROS-free composition helpers for the packaged Piper arm and Pika2 gripper URDFs."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from xml.etree import ElementTree

PIPER_PACKAGE_URI_PREFIX = "package://agx_arm_description/agx_arm_urdf/piper/"
PIKA2_PACKAGE_URI_PREFIX = "package://agx_arm_description/meshes/"


def compose_piper_pika2_urdf(
    base_urdf_path: Path,
    gripper_urdf_path: Path,
    asset_dir: Path,
) -> ElementTree.ElementTree:
    """Compose the standalone Piper arm and Pika2 gripper into one self-contained URDF tree."""
    base_root = ElementTree.parse(base_urdf_path).getroot()
    gripper_root = ElementTree.parse(gripper_urdf_path).getroot()

    root = ElementTree.Element("robot", {"name": "piper_with_pika2"})
    for child in base_root:
        if child.tag == "link" and child.get("name") == "world":
            continue
        if child.tag == "joint" and child.get("name") == "world_to_base_link":
            continue
        root.append(deepcopy(child))
    for child in gripper_root:
        root.append(deepcopy(child))

    joint = ElementTree.SubElement(root, "joint", {"name": "gripper_base_joint", "type": "fixed"})
    ElementTree.SubElement(joint, "origin", {"xyz": "0 0 0", "rpy": "0 0 0"})
    ElementTree.SubElement(joint, "parent", {"link": "link6"})
    ElementTree.SubElement(joint, "child", {"link": "gripper_base_link"})

    _rewrite_package_mesh_paths(root, asset_dir)
    _validate_standalone_tree(root)
    return ElementTree.ElementTree(root)


def write_piper_pika2_urdf(
    output_path: Path,
    base_urdf_path: Path,
    gripper_urdf_path: Path,
    asset_dir: Path,
) -> None:
    """Write a standalone Piper + Pika2 URDF for conversion or inspection."""
    tree = compose_piper_pika2_urdf(base_urdf_path, gripper_urdf_path, asset_dir)
    ElementTree.indent(tree, space="    ")
    tree.write(output_path, encoding="utf-8", xml_declaration=True)


def expand_piper_xacro(
    base_urdf_path: Path,
    gripper_xacro_path: Path,
    asset_dir: Path,
) -> ElementTree.ElementTree:
    """Compatibility alias for callers of the former Piper-gripper Xacro expander."""
    return compose_piper_pika2_urdf(base_urdf_path, gripper_xacro_path, asset_dir)


def write_expanded_piper_urdf(
    output_path: Path,
    base_urdf_path: Path,
    gripper_xacro_path: Path,
    asset_dir: Path,
) -> None:
    """Compatibility alias for writing the composed Piper + Pika2 URDF."""
    write_piper_pika2_urdf(output_path, base_urdf_path, gripper_xacro_path, asset_dir)


def set_piper_gripper_base_transform(
    root: ElementTree.Element,
    position_xyz: tuple[float, float, float],
    rotation_rpy: tuple[float, float, float],
) -> None:
    """Set the fixed link6-to-Pika2-base transform on a composed Piper URDF tree."""
    joint = root.find("./joint[@name='gripper_base_joint']")
    if joint is None:
        raise ValueError("Composed Piper URDF is missing gripper_base_joint.")
    parent = joint.find("parent")
    child = joint.find("child")
    origin = joint.find("origin")
    if (
        parent is None
        or parent.get("link") != "link6"
        or child is None
        or child.get("link") != "gripper_base_link"
        or origin is None
    ):
        raise ValueError("Piper Pika2 gripper_base_joint does not match the expected fixed-joint contract.")
    if len(position_xyz) != 3 or len(rotation_rpy) != 3:
        raise ValueError(
            f"position_xyz and rotation_rpy must each contain three values, got {position_xyz} and {rotation_rpy}."
        )
    origin.set("xyz", " ".join(f"{value:.12g}" for value in position_xyz))
    origin.set("rpy", " ".join(f"{value:.12g}" for value in rotation_rpy))


def set_piper_gripper_base_rotation(
    root: ElementTree.Element,
    rotation_rpy: tuple[float, float, float],
) -> None:
    """Set only the link6-to-Pika2-base rotation while retaining its source position."""
    joint = root.find("./joint[@name='gripper_base_joint']")
    origin = None if joint is None else joint.find("origin")
    if origin is None:
        raise ValueError("Composed Piper URDF is missing gripper_base_joint origin.")
    position_xyz = tuple(float(value) for value in origin.get("xyz", "0 0 0").split())
    set_piper_gripper_base_transform(root, position_xyz, rotation_rpy)


def _rewrite_package_mesh_paths(root: ElementTree.Element, asset_dir: Path) -> None:
    mappings = (
        (PIPER_PACKAGE_URI_PREFIX, asset_dir),
        (PIKA2_PACKAGE_URI_PREFIX, asset_dir / "meshes" / "dae"),
    )
    for mesh in root.findall(".//mesh"):
        filename = mesh.get("filename")
        if filename is None:
            continue
        for prefix, local_root in mappings:
            if filename.startswith(prefix):
                mesh_path = (local_root / filename.removeprefix(prefix)).resolve()
                break
        else:
            raise ValueError(f"Unexpected Piper or Pika2 mesh URI: {filename}")
        if not mesh_path.is_file():
            raise FileNotFoundError(f"Missing packaged Piper or Pika2 mesh: {mesh_path}")
        mesh.set("filename", str(mesh_path))


def _validate_standalone_tree(root: ElementTree.Element) -> None:
    link_names = [link.get("name") for link in root.findall("link")]
    joint_names = [joint.get("name") for joint in root.findall("joint")]
    if len(link_names) != len(set(link_names)) or len(joint_names) != len(set(joint_names)):
        raise ValueError("Composed Piper URDF contains duplicate link or joint names.")
    required_links = {"base_link", "link6", "gripper_base_link", "gripper_left_link", "gripper_right_link"}
    if not required_links.issubset(link_names):
        raise ValueError(f"Composed Piper URDF is missing required links: {required_links - set(link_names)}.")
    if "world" in link_names or "world_to_base_link" in joint_names:
        raise ValueError("Composed Piper URDF must use base_link as its root.")
    expected_joints = {f"joint{index}" for index in range(1, 7)} | {
        "gripper_base_joint",
        "left_joint",
        "right_joint",
    }
    if not expected_joints.issubset(joint_names):
        raise ValueError(f"Composed Piper URDF is missing required joints: {expected_joints - set(joint_names)}.")


__all__ = [
    "compose_piper_pika2_urdf",
    "expand_piper_xacro",
    "set_piper_gripper_base_rotation",
    "set_piper_gripper_base_transform",
    "write_expanded_piper_urdf",
    "write_piper_pika2_urdf",
]
