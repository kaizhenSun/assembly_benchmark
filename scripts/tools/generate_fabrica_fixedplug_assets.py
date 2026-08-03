# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Generate relation-paired Piper + Pika2 fixed-plug assets for a Fabrica assembly."""

from __future__ import annotations

import argparse
import math
import shutil
import sys
import tempfile
import traceback
from pathlib import Path
from xml.etree import ElementTree

from isaaclab.app import AppLauncher

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = PROJECT_ROOT / "source" / "assembly_benchmark"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

parser = argparse.ArgumentParser(description="Generate Piper + Pika2 fixed-plug USDs for one Fabrica assembly.")
parser.add_argument("--assembly", default="beam", help="Checked-in Fabrica assembly plan to generate.")
parser.add_argument("--overwrite", action="store_true", help="Replace existing relation asset folders.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.headless = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from assembly_benchmark.assembly import (  # noqa: E402
    PIPER_FABRICA_GRIPPER_BASE_POS,
    PIPER_FABRICA_GRIPPER_BASE_ROTATION_RPY,
    load_fabrica_assembly_plan,
    make_assembly,
)
from assembly_benchmark.robots.piper import (  # noqa: E402
    PIPER_ASSET_DIR,
    PIPER_BASE_URDF_PATH,
    PIPER_GRIPPER_URDF_PATH,
    piper_fixed_plug_usd_path,
)
from assembly_benchmark.utils.piper_urdf import (  # noqa: E402
    set_piper_gripper_base_transform,
    write_piper_pika2_urdf,
)

from pxr import PhysxSchema, Sdf, Usd, UsdPhysics, UsdShade  # noqa: E402

from isaaclab.sim.converters import UrdfConverter, UrdfConverterCfg  # noqa: E402

PLUG_LINK_NAME = "plug"
PLUG_JOINT_NAME = "plug_joint"
PLUG_DENSITY = 1250.0
PLUG_FRICTION = 1.0
SOCKET_LINK_NAME = "socket"
SOCKET_FRICTION = 0.5
SDF_RESOLUTION = 512
SDF_SUBGRID_RESOLUTION = 8
SDF_MARGIN = 0.001
SDF_NARROW_BAND_THICKNESS = 0.01
PLUG_FILTERED_BODY_NAMES = (
    "link6",
    "gripper_base_link",
    "gripper_left_link",
    "gripper_right_link",
)


def _format_vec(values: tuple[float, ...]) -> str:
    return " ".join(f"{value:.12g}" for value in values)


def _quaternion_to_rpy(quaternion: tuple[float, float, float, float]) -> tuple[float, float, float]:
    w, x, y, z = quaternion
    norm = math.sqrt(w * w + x * x + y * y + z * z)
    if norm <= 1.0e-12:
        raise ValueError("The Piper Pika2 gripper-to-plug quaternion must have non-zero norm.")
    w, x, y, z = (value / norm for value in (w, x, y, z))
    roll = math.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    sin_pitch = max(-1.0, min(1.0, 2.0 * (w * y - z * x)))
    pitch = math.asin(sin_pitch)
    yaw = math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    return roll, pitch, yaw


def _rpy_to_quaternion(rotation_rpy: tuple[float, float, float]) -> tuple[float, float, float, float]:
    roll, pitch, yaw = rotation_rpy
    cr, sr = math.cos(0.5 * roll), math.sin(0.5 * roll)
    cp, sp = math.cos(0.5 * pitch), math.sin(0.5 * pitch)
    cy, sy = math.cos(0.5 * yaw), math.sin(0.5 * yaw)
    return (
        cr * cp * cy + sr * sp * sy,
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
    )


def _append_plug_link(
    root: ElementTree.Element,
    mesh_path: Path,
    gripper_to_plug_pos: tuple[float, float, float],
    gripper_to_plug_quat: tuple[float, float, float, float],
) -> None:
    link = ElementTree.SubElement(root, "link", {"name": PLUG_LINK_NAME})
    for element_name in ("visual", "collision"):
        element = ElementTree.SubElement(link, element_name)
        ElementTree.SubElement(element, "origin", {"xyz": "0 0 0", "rpy": "0 0 0"})
        geometry = ElementTree.SubElement(element, "geometry")
        ElementTree.SubElement(geometry, "mesh", {"filename": str(mesh_path.resolve())})

    joint = ElementTree.SubElement(root, "joint", {"name": PLUG_JOINT_NAME, "type": "fixed"})
    ElementTree.SubElement(
        joint,
        "origin",
        {
            "xyz": _format_vec(gripper_to_plug_pos),
            "rpy": _format_vec(_quaternion_to_rpy(gripper_to_plug_quat)),
        },
    )
    ElementTree.SubElement(joint, "parent", {"link": "gripper_base_link"})
    ElementTree.SubElement(joint, "child", {"link": PLUG_LINK_NAME})


def _write_augmented_urdf(path: Path, mesh_path: Path, relation) -> None:
    write_piper_pika2_urdf(path, PIPER_BASE_URDF_PATH, PIPER_GRIPPER_URDF_PATH, PIPER_ASSET_DIR)
    tree = ElementTree.parse(path)
    set_piper_gripper_base_transform(
        tree.getroot(),
        PIPER_FABRICA_GRIPPER_BASE_POS,
        PIPER_FABRICA_GRIPPER_BASE_ROTATION_RPY,
    )
    _append_plug_link(
        tree.getroot(),
        mesh_path,
        relation.piper_gripper_to_plug_pos,
        relation.piper_gripper_to_plug_quat,
    )
    ElementTree.indent(tree, space="    ")
    tree.write(path, encoding="utf-8", xml_declaration=True)


def _runtime_physics_stage_path(usd_path: Path) -> Path:
    physics_path = usd_path.parent / "configuration" / f"{usd_path.stem}_physics.usd"
    return physics_path if physics_path.exists() else usd_path


def _fix_nested_collision_meshes(usd_path: Path) -> None:
    stage_path = _runtime_physics_stage_path(usd_path)
    stage = Usd.Stage.Open(str(stage_path))
    if stage is None:
        raise RuntimeError(f"Failed to open generated Piper fixed-plug physics stage: {stage_path}")
    meshes = [
        prim for prim in stage.TraverseAll() if prim.GetTypeName() == "Mesh" and "/colliders/" in str(prim.GetPath())
    ]
    if not meshes:
        raise RuntimeError(f"No collision meshes found in generated fixed-plug USD: {stage_path}")
    for prim in stage.TraverseAll():
        if prim.GetTypeName() == "Mesh" or "/colliders/" not in str(prim.GetPath()):
            continue
        if prim.HasAPI(UsdPhysics.MeshCollisionAPI):
            prim.RemoveAPI(UsdPhysics.MeshCollisionAPI)
        if prim.HasAPI(UsdPhysics.CollisionAPI):
            prim.RemoveAPI(UsdPhysics.CollisionAPI)
    for mesh in meshes:
        UsdPhysics.CollisionAPI.Apply(mesh).CreateCollisionEnabledAttr(True)
        UsdPhysics.MeshCollisionAPI.Apply(mesh).CreateApproximationAttr(UsdPhysics.Tokens.convexHull)
    stage.GetRootLayer().Save()


def _find_body_prim(stage: Usd.Stage, body_name: str) -> Usd.Prim:
    candidates = [
        prim for prim in stage.TraverseAll() if prim.GetName() == body_name and prim.HasAPI(UsdPhysics.RigidBodyAPI)
    ]
    if len(candidates) != 1:
        raise RuntimeError(f"Expected one rigid body named '{body_name}', found {len(candidates)}.")
    return candidates[0]


def _make_plug_collider_editable(usd_path: Path) -> None:
    stage = Usd.Stage.Open(str(usd_path))
    if stage is None:
        raise RuntimeError(f"Failed to open generated fixed-plug stage: {usd_path}")
    plug = _find_body_prim(stage, PLUG_LINK_NAME)
    collision_root = stage.GetPrimAtPath(plug.GetPath().AppendChild("collisions"))
    if not collision_root.IsValid():
        raise RuntimeError(f"Could not find plug collision root under {plug.GetPath()}.")
    if collision_root.IsInstanceable():
        collision_root.SetInstanceable(False)
    stage.GetRootLayer().Save()


def _find_plug_collider(stage: Usd.Stage) -> Usd.Prim:
    return _find_body_collider(stage, PLUG_LINK_NAME)


def _find_body_collider(stage: Usd.Stage, body_name: str) -> Usd.Prim:
    colliders = [
        prim
        for prim in stage.TraverseAll()
        if f"/{body_name}/" in str(prim.GetPath())
        and prim.GetTypeName() == "Mesh"
        and prim.HasAPI(UsdPhysics.CollisionAPI)
        and UsdPhysics.MeshCollisionAPI(prim)
    ]
    if len(colliders) != 1:
        raise RuntimeError(f"Expected one mesh collider for {body_name}, found {len(colliders)}.")
    return colliders[0]


def _configure_plug_physics(usd_path: Path) -> None:
    stage = Usd.Stage.Open(str(usd_path))
    if stage is None:
        raise RuntimeError(f"Failed to reopen generated fixed-plug stage: {usd_path}")
    plug = _find_body_prim(stage, PLUG_LINK_NAME)
    for prim in Usd.PrimRange(plug):
        if prim.GetTypeName() != "Mesh":
            if prim.HasAPI(UsdPhysics.MeshCollisionAPI):
                prim.RemoveAPI(UsdPhysics.MeshCollisionAPI)
            if prim.HasAPI(UsdPhysics.CollisionAPI):
                prim.RemoveAPI(UsdPhysics.CollisionAPI)
    collider = _find_plug_collider(stage)
    UsdPhysics.MeshCollisionAPI.Apply(collider).CreateApproximationAttr(PhysxSchema.Tokens.sdf)
    sdf_api = PhysxSchema.PhysxSDFMeshCollisionAPI.Apply(collider)
    sdf_api.CreateSdfResolutionAttr(SDF_RESOLUTION)
    sdf_api.CreateSdfSubgridResolutionAttr(SDF_SUBGRID_RESOLUTION)
    sdf_api.CreateSdfMarginAttr(SDF_MARGIN)
    sdf_api.CreateSdfNarrowBandThicknessAttr(SDF_NARROW_BAND_THICKNESS)

    material = UsdShade.Material.Define(stage, collider.GetPath().GetParentPath().AppendChild("plugPhysicsMaterial"))
    material_api = UsdPhysics.MaterialAPI.Apply(material.GetPrim())
    material_api.CreateStaticFrictionAttr(PLUG_FRICTION)
    material_api.CreateDynamicFrictionAttr(PLUG_FRICTION)
    material_api.CreateRestitutionAttr(0.0)
    UsdShade.MaterialBindingAPI.Apply(collider).Bind(
        material,
        bindingStrength=UsdShade.Tokens.strongerThanDescendants,
        materialPurpose="physics",
    )
    targets = [_find_body_prim(stage, name).GetPath() for name in PLUG_FILTERED_BODY_NAMES]
    UsdPhysics.FilteredPairsAPI.Apply(plug).CreateFilteredPairsRel().SetTargets(targets)
    stage.GetRootLayer().Save()


def _validate_generated_usd(usd_path: Path) -> tuple[tuple[str, ...], tuple[str, ...]]:
    stage = Usd.Stage.Open(str(usd_path))
    if stage is None:
        raise RuntimeError(f"Failed to validate generated fixed-plug stage: {usd_path}")
    plug = _find_body_prim(stage, PLUG_LINK_NAME)
    collider = _find_plug_collider(stage)
    approximation = UsdPhysics.MeshCollisionAPI(collider).GetApproximationAttr().Get()
    sdf_resolution = PhysxSchema.PhysxSDFMeshCollisionAPI(collider).GetSdfResolutionAttr().Get()
    if approximation != PhysxSchema.Tokens.sdf or sdf_resolution != SDF_RESOLUTION:
        raise RuntimeError(
            f"Generated plug collider has approximation={approximation!r}, resolution={sdf_resolution!r}."
        )
    filtered_names = {target.name for target in UsdPhysics.FilteredPairsAPI(plug).GetFilteredPairsRel().GetTargets()}
    if filtered_names != set(PLUG_FILTERED_BODY_NAMES):
        raise RuntimeError(f"Generated plug collision filters are incomplete: {filtered_names}.")

    root = stage.GetDefaultPrim()
    joint_root = root.GetPath().AppendChild("joints")
    gripper_mount = UsdPhysics.FixedJoint(stage.GetPrimAtPath(joint_root.AppendChild("gripper_base_joint")))
    if not gripper_mount:
        raise RuntimeError("Generated fixed-plug USD is missing gripper_base_joint.")
    if [target.name for target in gripper_mount.GetBody0Rel().GetTargets()] != ["link6"]:
        raise RuntimeError("Generated Pika2 mount is not attached to link6.")
    if [target.name for target in gripper_mount.GetBody1Rel().GetTargets()] != ["gripper_base_link"]:
        raise RuntimeError("Generated Pika2 mount does not target gripper_base_link.")
    mount_position = tuple(gripper_mount.GetLocalPos0Attr().Get())
    if any(abs(actual - expected) > 1.0e-7 for actual, expected in zip(mount_position, PIPER_FABRICA_GRIPPER_BASE_POS)):
        raise RuntimeError(f"Generated Pika2 mount has unexpected position: {mount_position}.")
    mount_rotation = gripper_mount.GetLocalRot0Attr().Get()
    mount_quaternion = (mount_rotation.GetReal(), *mount_rotation.GetImaginary())
    expected_mount_quaternion = _rpy_to_quaternion(PIPER_FABRICA_GRIPPER_BASE_ROTATION_RPY)
    mount_dot = abs(
        sum(actual * expected for actual, expected in zip(mount_quaternion, expected_mount_quaternion, strict=True))
    )
    if abs(mount_dot - 1.0) > 5.0e-7:
        raise RuntimeError(f"Generated Pika2 mount has unexpected rotation: {mount_quaternion}.")

    plug_joint = UsdPhysics.FixedJoint(stage.GetPrimAtPath(joint_root.AppendChild(PLUG_JOINT_NAME)))
    if not plug_joint:
        raise RuntimeError(f"Generated fixed-plug USD is missing {PLUG_JOINT_NAME}.")
    if [target.name for target in plug_joint.GetBody0Rel().GetTargets()] != ["gripper_base_link"]:
        raise RuntimeError("Generated plug joint is not attached to gripper_base_link.")
    if [target.name for target in plug_joint.GetBody1Rel().GetTargets()] != [PLUG_LINK_NAME]:
        raise RuntimeError("Generated plug joint does not target the common plug body.")

    limits = {}
    for joint_name in ("left_joint", "right_joint"):
        joint = UsdPhysics.PrismaticJoint(stage.GetPrimAtPath(joint_root.AppendChild(joint_name)))
        if not joint:
            raise RuntimeError(f"Generated Piper + Pika2 asset is missing prismatic joint '{joint_name}'.")
        limits[joint_name] = (joint.GetLowerLimitAttr().Get(), joint.GetUpperLimitAttr().Get())
    left_lower, left_upper = limits["left_joint"]
    right_lower, right_upper = limits["right_joint"]
    if left_lower < -1.0e-6 or left_upper <= 0.0 or right_lower >= 0.0 or right_upper > 1.0e-6:
        raise RuntimeError(f"Generated Pika2 joint limits have unexpected signs: {limits}.")

    bodies = tuple(sorted(prim.GetName() for prim in stage.TraverseAll() if prim.HasAPI(UsdPhysics.RigidBodyAPI)))
    joints = tuple(
        sorted(
            prim.GetName()
            for prim in stage.TraverseAll()
            if prim.IsA(UsdPhysics.Joint) or prim.IsA(UsdPhysics.FixedJoint)
        )
    )
    return bodies, joints


def _configure_and_validate_socket(usd_path: Path) -> None:
    print(f"[INFO] Configuring common socket asset: {usd_path}", flush=True)
    stage = Usd.Stage.Open(str(usd_path))
    if stage is None:
        raise RuntimeError(f"Failed to open generated socket stage: {usd_path}")
    socket = _find_body_prim(stage, SOCKET_LINK_NAME)
    print(f"[INFO] Found socket body {socket.GetPath()}.", flush=True)
    collision_meshes = [
        prim for prim in Usd.PrimRange(socket) if prim.GetTypeName() == "Mesh" and "/collisions/" in str(prim.GetPath())
    ]
    if len(collision_meshes) != 1:
        raise RuntimeError(f"Expected one socket collision mesh, found {len(collision_meshes)}.")
    for prim in Usd.PrimRange(socket):
        if prim.GetTypeName() == "Mesh" or "/collisions/" not in str(prim.GetPath()):
            continue
        if prim.HasAPI(UsdPhysics.MeshCollisionAPI):
            prim.RemoveAPI(UsdPhysics.MeshCollisionAPI)
        if prim.HasAPI(UsdPhysics.CollisionAPI):
            prim.RemoveAPI(UsdPhysics.CollisionAPI)
    UsdPhysics.CollisionAPI.Apply(collision_meshes[0]).CreateCollisionEnabledAttr(True)
    UsdPhysics.MeshCollisionAPI.Apply(collision_meshes[0]).CreateApproximationAttr(UsdPhysics.Tokens.convexHull)

    visuals_path = socket.GetPath().AppendChild("visuals")
    if stage.GetPrimAtPath(visuals_path).IsValid():
        stage.RemovePrim(visuals_path)
    visuals = stage.DefinePrim(visuals_path, "Xform")
    visuals.GetReferences().ClearReferences()
    visuals.SetInstanceable(False)
    visual_mesh_path = visuals_path.AppendChild("mesh")
    if not Sdf.CopySpec(stage.GetRootLayer(), collision_meshes[0].GetPath(), stage.GetRootLayer(), visual_mesh_path):
        raise RuntimeError("Failed to derive a self-contained socket visual mesh from its collider.")
    visual_mesh = stage.GetPrimAtPath(visual_mesh_path)
    for api in (PhysxSchema.PhysxSDFMeshCollisionAPI, UsdPhysics.MeshCollisionAPI, UsdPhysics.CollisionAPI):
        if visual_mesh.HasAPI(api):
            visual_mesh.RemoveAPI(api)

    collider = _find_body_collider(stage, SOCKET_LINK_NAME)
    print(f"[INFO] Found socket body {socket.GetPath()} and collider {collider.GetPath()}.", flush=True)
    UsdPhysics.MeshCollisionAPI.Apply(collider).CreateApproximationAttr(PhysxSchema.Tokens.sdf)
    sdf_api = PhysxSchema.PhysxSDFMeshCollisionAPI.Apply(collider)
    sdf_api.CreateSdfResolutionAttr(SDF_RESOLUTION)
    sdf_api.CreateSdfSubgridResolutionAttr(SDF_SUBGRID_RESOLUTION)
    sdf_api.CreateSdfMarginAttr(SDF_MARGIN)
    sdf_api.CreateSdfNarrowBandThicknessAttr(SDF_NARROW_BAND_THICKNESS)

    for prim in Usd.PrimRange(socket):
        for relationship in prim.GetRelationships():
            if relationship.GetName().startswith("material:binding"):
                prim.RemoveProperty(relationship.GetName())

    material = UsdShade.Material.Define(stage, socket.GetPath().AppendChild("socketPhysicsMaterial"))
    material_api = UsdPhysics.MaterialAPI.Apply(material.GetPrim())
    material_api.CreateStaticFrictionAttr(SOCKET_FRICTION)
    material_api.CreateDynamicFrictionAttr(SOCKET_FRICTION)
    material_api.CreateRestitutionAttr(0.0)
    UsdShade.MaterialBindingAPI.Apply(collider).Bind(
        material,
        bindingStrength=UsdShade.Tokens.strongerThanDescendants,
        materialPurpose="physics",
    )
    stage.GetRootLayer().Save()
    print(f"[INFO] Saved socket physics overrides: {usd_path}", flush=True)

    if UsdPhysics.MeshCollisionAPI(collider).GetApproximationAttr().Get() != PhysxSchema.Tokens.sdf:
        raise RuntimeError("Generated socket collider is not SDF.")
    if PhysxSchema.PhysxSDFMeshCollisionAPI(collider).GetSdfResolutionAttr().Get() != SDF_RESOLUTION:
        raise RuntimeError("Generated socket collider does not use the configured SDF resolution.")
    print(f"[INFO] Validated common socket asset: {usd_path}", flush=True)


def _generate_socket_asset(output_path: Path, source_usd_path: Path) -> None:
    """Wrap an assembly part USD under the common ``socket`` body name.

    The source Beam parts intentionally keep their neutral ``beam_part_<id>`` names. Their composed
    collision subtree is copied beneath a freshly-authored ``/asset/socket`` rigid body so every
    relation has the hierarchy required by ``MultiUsdFileCfg`` without invoking a second URDF
    converter or retaining external material and prototype relationships.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    source_stage = Usd.Stage.Open(str(source_usd_path))
    if source_stage is None or not source_stage.GetDefaultPrim().IsValid():
        raise RuntimeError(f"Failed to open source socket USD with a default prim: {source_usd_path}")

    source_bodies = [prim for prim in source_stage.TraverseAll() if prim.HasAPI(UsdPhysics.RigidBodyAPI)]
    if len(source_bodies) != 1:
        raise RuntimeError(f"Expected one rigid body in source socket USD, found {len(source_bodies)}.")
    source_collision_root = source_stage.GetPrimAtPath(source_bodies[0].GetPath().AppendChild("collisions"))
    if not source_collision_root.IsValid():
        raise RuntimeError(f"Source socket USD has no collision subtree: {source_usd_path}")

    stage = Usd.Stage.CreateNew(str(output_path))
    asset = stage.DefinePrim("/asset", "Xform")
    socket = stage.DefinePrim(f"/asset/{SOCKET_LINK_NAME}", "Xform")
    UsdPhysics.RigidBodyAPI.Apply(socket).CreateRigidBodyEnabledAttr(True)
    UsdPhysics.MassAPI.Apply(socket).CreateDensityAttr(PLUG_DENSITY)
    flattened_source = source_stage.Flatten()
    destination_path = socket.GetPath().AppendChild("collisions")
    if not Sdf.CopySpec(flattened_source, source_collision_root.GetPath(), stage.GetRootLayer(), destination_path):
        raise RuntimeError(f"Failed to copy source socket collision subtree into {output_path}.")
    stage.SetDefaultPrim(asset)
    stage.GetRootLayer().Save()
    print(f"[INFO] Authored self-contained common socket: {output_path}", flush=True)
    _configure_and_validate_socket(output_path)


def _strip_converter_metadata(asset_dir: Path) -> None:
    for metadata_file in (asset_dir / ".asset_hash", asset_dir / "config.yaml"):
        metadata_file.unlink(missing_ok=True)


def _fixed_plug_socket_usd_path(assembly, socket_part_id: str) -> Path:
    """Return the common-topology socket path for one unique assembly part."""
    return assembly.asset_root / "usd" / "fixed_plug_socket" / socket_part_id / "socket.usd"


def _generate_relation_asset(assembly, relation) -> tuple[tuple[str, ...], tuple[str, ...]]:
    print(f"[INFO] Building {assembly.name} relation {relation.key} with Piper + Pika2.", flush=True)
    plan = load_fabrica_assembly_plan(assembly.name)
    plug_part = assembly.part(plan.part_scene_key(relation.plug_part_id))
    mesh_path = assembly.asset_root / "mesh" / f"{assembly.name}_part_{relation.plug_part_id}.obj"
    output_path = piper_fixed_plug_usd_path(assembly.name, relation.key)
    output_dir = output_path.parent
    if output_dir.exists() and not args_cli.overwrite:
        raise FileExistsError(f"Asset already exists: {output_path}. Pass --overwrite to regenerate it.")
    if not mesh_path.is_file() or not plug_part.urdf_path(assembly.asset_root).is_file():
        raise FileNotFoundError(f"Missing source asset for relation {relation.key}: {mesh_path}")

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{relation.key}_", dir=output_dir.parent) as temp_dir:
        staging_dir = Path(temp_dir) / relation.key
        staging_dir.mkdir()
        staged_output_path = staging_dir / output_path.name
        augmented_urdf = Path(temp_dir) / "piper_fixed_plug.urdf"
        _write_augmented_urdf(augmented_urdf, mesh_path, relation)
        cfg = UrdfConverterCfg(
            asset_path=str(augmented_urdf),
            usd_dir=str(staging_dir),
            usd_file_name=staged_output_path.name,
            force_usd_conversion=True,
            make_instanceable=True,
            fix_base=True,
            merge_fixed_joints=False,
            convert_mimic_joints_to_normal_joints=False,
            self_collision=False,
            collision_from_visuals=False,
            link_density=PLUG_DENSITY,
            joint_drive=UrdfConverterCfg.JointDriveCfg(
                target_type="position",
                drive_type="force",
                gains=UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=400.0, damping=40.0),
            ),
        )
        generated_path = Path(UrdfConverter(cfg).usd_path)
        if generated_path.resolve() != staged_output_path.resolve():
            raise RuntimeError(f"Unexpected generated fixed-plug USD path: {generated_path}")
        _fix_nested_collision_meshes(generated_path)
        _make_plug_collider_editable(generated_path)
        _configure_plug_physics(generated_path)
        topology = _validate_generated_usd(generated_path)
        _strip_converter_metadata(staging_dir)

        # Keep the checked-in relation intact if conversion or validation fails partway through.
        if output_dir.exists():
            shutil.rmtree(output_dir)
        shutil.move(str(staging_dir), str(output_dir))
    print(f"[INFO] Generated {assembly.name} relation {relation.key}: {output_path}")
    return topology


def _generate_socket_assets(assembly, plan) -> tuple[Path, ...]:
    """Generate one common-topology socket USD per unique socket part."""
    socket_part_ids = tuple(dict.fromkeys(relation.socket_part_id for relation in plan.relations))
    output_paths = []
    for part_id in socket_part_ids:
        socket_part = assembly.part(plan.part_scene_key(part_id))
        source_path = socket_part.usd_path(assembly.asset_root)
        output_path = _fixed_plug_socket_usd_path(assembly, part_id)
        if output_path.exists() and not args_cli.overwrite:
            raise FileExistsError(f"Asset already exists: {output_path}. Pass --overwrite to regenerate it.")
        if not source_path.is_file():
            raise FileNotFoundError(f"Missing source socket asset for part {part_id}: {source_path}")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix=f".{part_id}_", dir=output_path.parent) as temp_dir:
            staged_output_path = Path(temp_dir) / output_path.name
            _generate_socket_asset(staged_output_path, source_path)
            staged_output_path.replace(output_path)
        output_paths.append(output_path)
    return tuple(output_paths)


def main() -> None:
    print(f"[INFO] Loading Fabrica assembly plan '{args_cli.assembly}'.", flush=True)
    plan = load_fabrica_assembly_plan(args_cli.assembly)
    assembly = make_assembly(plan.assembly_name)
    topologies = [_generate_relation_asset(assembly, relation) for relation in plan.relations]
    if any(topology != topologies[0] for topology in topologies[1:]):
        raise RuntimeError("Generated fixed-plug relation assets do not share articulation topology.")
    socket_paths = _generate_socket_assets(assembly, plan)
    print(
        f"[INFO] Generated {len(topologies)} topology-compatible fixed-plug robots and "
        f"{len(socket_paths)} unique sockets for {assembly.name}."
    )


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        raise
    finally:
        simulation_app.close()
