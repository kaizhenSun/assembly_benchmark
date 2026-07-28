# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Generate the fixed-base R1 Pro asset with Fabrica beam plug 0 attached.

The source R1 Pro URDF remains the single robot description.  This tool creates
an augmented URDF in a temporary directory, imports it, and checks the resulting
USD into the task-specific asset directory.

.. code-block:: bash

    python scripts/tools/generate_r1_pro_beam02_asset.py --overwrite

"""

from __future__ import annotations

import argparse
import math
import sys
import tempfile
from pathlib import Path
from xml.etree import ElementTree

from isaaclab.app import AppLauncher

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = PROJECT_ROOT / "source" / "assembly_benchmark"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))


parser = argparse.ArgumentParser(description="Generate the R1 Pro beam 0 fixed-attachment USD asset.")
parser.add_argument(
    "--overwrite",
    "--force",
    action="store_true",
    dest="overwrite",
    help="Regenerate the USD even when the output already exists.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.headless = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from assembly_benchmark.assembly.specs import assembly_asset_root  # noqa: E402
from assembly_benchmark.beam02_grasp import (  # noqa: E402
    BEAM02_GRIPPER_TO_PLUG_POS,
    BEAM02_GRIPPER_TO_PLUG_QUAT,
)
from assembly_benchmark.robots.r1_pro import (  # noqa: E402
    R1_PRO_ASSET_DIR,
    R1_PRO_BEAM02_ASSET_DIR,
    R1_PRO_BEAM02_USD_PATH,
    R1_PRO_URDF_PATH,
)

from pxr import PhysxSchema, Usd, UsdPhysics, UsdShade  # noqa: E402

from isaaclab.sim.converters import UrdfConverter, UrdfConverterCfg  # noqa: E402

PLUG_LINK_NAME = "beam_plug_0"
PLUG_JOINT_NAME = "beam_plug_0_joint"
PLUG_MESH_PATH = assembly_asset_root("beam") / "mesh" / "beam" / "beam_part_0.obj"


def _quaternion_to_rpy(quaternion: tuple[float, float, float, float]) -> tuple[float, float, float]:
    """Convert a normalized wxyz quaternion to URDF fixed-axis roll, pitch, yaw."""
    w, x, y, z = quaternion
    norm = math.sqrt(w * w + x * x + y * y + z * z)
    if norm <= 1.0e-12:
        raise ValueError("The gripper-to-plug quaternion must have non-zero norm.")
    w, x, y, z = (value / norm for value in (w, x, y, z))
    roll = math.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    sin_pitch = max(-1.0, min(1.0, 2.0 * (w * y - z * x)))
    pitch = math.asin(sin_pitch)
    yaw = math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    return roll, pitch, yaw


# T_left_gripper_link_beam_plug_0. URDF uses xyz/rpy, while the dependency-free
# grasp module is the canonical runtime and asset-generation source in wxyz form.
PLUG_JOINT_POS = BEAM02_GRIPPER_TO_PLUG_POS
PLUG_JOINT_RPY = _quaternion_to_rpy(BEAM02_GRIPPER_TO_PLUG_QUAT)

# Mass properties computed from the watertight, metre-scaled mesh at
# 1250 kg/m^3.  Inertia is expressed at the mesh center of mass.
PLUG_MASS = 0.012872312627763116
PLUG_FRICTION = 1.0
PLUG_COM = (-0.0444500771, -0.0000000007, 0.0708890000)
PLUG_INERTIA = {
    "ixx": 5.68065433e-06,
    "ixy": -1.72550318e-12,
    "ixz": 6.87717961e-16,
    "iyy": 3.84433647e-07,
    "iyz": 2.00827207e-14,
    "izz": 5.65849204e-06,
}

SDF_RESOLUTION = 512
SDF_SUBGRID_RESOLUTION = 8
SDF_MARGIN = 0.001
SDF_NARROW_BAND_THICKNESS = 0.01

GRIPPER_COLLISION_PATHS = (
    "/colliders/left_gripper_link/left_gripper_link_collision",
    "/colliders/left_gripper_finger_link1/left_gripper_finger_link1_collision",
    "/colliders/left_gripper_finger_link2/left_gripper_finger_link2_collision",
    "/colliders/right_gripper_link/right_gripper_link_collision",
    "/colliders/right_gripper_finger_link1/right_gripper_finger_link1_collision",
    "/colliders/right_gripper_finger_link2/right_gripper_finger_link2_collision",
)

PLUG_FILTERED_BODY_NAMES = (
    "left_arm_link7",
    "left_gripper_link",
    "left_gripper_finger_link1",
    "left_gripper_finger_link2",
    "left_realsense_link",
)


def _format_vec(values: tuple[float, ...]) -> str:
    return " ".join(f"{value:.12g}" for value in values)


def _rewrite_mesh_paths(root: ElementTree.Element) -> None:
    """Make the canonical R1 mesh paths independent of the temporary URDF location."""
    for mesh in root.findall(".//mesh"):
        filename = mesh.get("filename")
        if filename is None:
            continue
        mesh_path = Path(filename)
        if not mesh_path.is_absolute():
            mesh.set("filename", str((R1_PRO_ASSET_DIR / mesh_path).resolve()))


def _append_plug_link(root: ElementTree.Element) -> None:
    if root.find(f"./link[@name='{PLUG_LINK_NAME}']") is not None:
        raise ValueError(f"Source URDF already contains link '{PLUG_LINK_NAME}'.")

    link = ElementTree.SubElement(root, "link", {"name": PLUG_LINK_NAME})
    inertial = ElementTree.SubElement(link, "inertial")
    ElementTree.SubElement(inertial, "origin", {"xyz": _format_vec(PLUG_COM), "rpy": "0 0 0"})
    ElementTree.SubElement(inertial, "mass", {"value": f"{PLUG_MASS:.16g}"})
    ElementTree.SubElement(inertial, "inertia", {key: f"{value:.16g}" for key, value in PLUG_INERTIA.items()})

    for element_name in ("visual", "collision"):
        element = ElementTree.SubElement(link, element_name)
        ElementTree.SubElement(element, "origin", {"xyz": "0 0 0", "rpy": "0 0 0"})
        geometry = ElementTree.SubElement(element, "geometry")
        ElementTree.SubElement(geometry, "mesh", {"filename": str(PLUG_MESH_PATH.resolve())})

    joint = ElementTree.SubElement(root, "joint", {"name": PLUG_JOINT_NAME, "type": "fixed"})
    ElementTree.SubElement(joint, "origin", {"xyz": _format_vec(PLUG_JOINT_POS), "rpy": _format_vec(PLUG_JOINT_RPY)})
    ElementTree.SubElement(joint, "parent", {"link": "left_gripper_link"})
    ElementTree.SubElement(joint, "child", {"link": PLUG_LINK_NAME})


def _write_augmented_urdf(path: Path) -> None:
    tree = ElementTree.parse(R1_PRO_URDF_PATH)
    root = tree.getroot()
    _rewrite_mesh_paths(root)
    _append_plug_link(root)
    tree.write(path, encoding="utf-8", xml_declaration=True)


def _runtime_physics_stage_path(usd_path: Path) -> Path:
    physics_path = usd_path.parent / "configuration" / f"{usd_path.stem}_physics.usd"
    return physics_path if physics_path.exists() else usd_path


def _find_single_mesh_under(stage: Usd.Stage, root_path: str, description: str) -> Usd.Prim:
    root = stage.GetPrimAtPath(root_path)
    if not root.IsValid():
        raise RuntimeError(f"Could not find {description} in generated USD: {root_path}")
    meshes = [prim for prim in Usd.PrimRange(root) if prim.GetTypeName() == "Mesh"]
    if len(meshes) != 1:
        raise RuntimeError(f"Expected exactly one mesh under {root_path}, found {len(meshes)}.")
    return meshes[0]


def _fix_gripper_collision_meshes(stage: Usd.Stage) -> None:
    for collision_path in GRIPPER_COLLISION_PATHS:
        mesh = _find_single_mesh_under(stage, collision_path, "gripper collision root")
        UsdPhysics.CollisionAPI.Apply(mesh).CreateCollisionEnabledAttr(True)
        UsdPhysics.MeshCollisionAPI.Apply(mesh).CreateApproximationAttr(UsdPhysics.Tokens.convexHull)


def _make_plug_collider_editable(stage: Usd.Stage) -> None:
    """Break the imported collision instance so task-specific APIs can be authored."""
    collision_root = stage.GetPrimAtPath(f"/r1_pro_with_gripper/{PLUG_LINK_NAME}/collisions")
    if not collision_root.IsValid():
        raise RuntimeError(f"Could not find the {PLUG_LINK_NAME} collision root in the generated USD.")
    if collision_root.IsInstanceable():
        collision_root.SetInstanceable(False)


def _convert_plug_collider_to_sdf(stage: Usd.Stage) -> None:
    plug_colliders = [
        prim
        for prim in stage.TraverseAll()
        if PLUG_LINK_NAME in str(prim.GetPath())
        and prim.HasAPI(UsdPhysics.CollisionAPI)
        and UsdPhysics.MeshCollisionAPI(prim)
    ]
    if len(plug_colliders) != 1:
        raise RuntimeError(f"Expected one mesh collider for {PLUG_LINK_NAME}, found {len(plug_colliders)}.")

    collider = plug_colliders[0]
    UsdPhysics.MeshCollisionAPI.Apply(collider).CreateApproximationAttr(PhysxSchema.Tokens.sdf)
    sdf_api = PhysxSchema.PhysxSDFMeshCollisionAPI.Apply(collider)
    sdf_api.CreateSdfResolutionAttr(SDF_RESOLUTION)
    sdf_api.CreateSdfSubgridResolutionAttr(SDF_SUBGRID_RESOLUTION)
    sdf_api.CreateSdfMarginAttr(SDF_MARGIN)
    sdf_api.CreateSdfNarrowBandThicknessAttr(SDF_NARROW_BAND_THICKNESS)

    material_path = plug_colliders[0].GetPath().GetParentPath().AppendChild("beamPlugPhysicsMaterial")
    material = UsdShade.Material.Define(stage, material_path)
    material_api = UsdPhysics.MaterialAPI.Apply(material.GetPrim())
    material_api.CreateStaticFrictionAttr(PLUG_FRICTION)
    material_api.CreateDynamicFrictionAttr(PLUG_FRICTION)
    material_api.CreateRestitutionAttr(0.0)
    UsdShade.MaterialBindingAPI.Apply(collider).Bind(
        material,
        bindingStrength=UsdShade.Tokens.strongerThanDescendants,
        materialPurpose="physics",
    )


def _find_body_prim(stage: Usd.Stage, body_name: str) -> Usd.Prim:
    candidates = [
        prim for prim in stage.TraverseAll() if prim.GetName() == body_name and prim.HasAPI(UsdPhysics.RigidBodyAPI)
    ]
    if len(candidates) != 1:
        raise RuntimeError(f"Expected one rigid body named '{body_name}', found {len(candidates)}.")
    return candidates[0]


def _author_plug_collision_filters(usd_path: Path) -> None:
    stage = Usd.Stage.Open(str(usd_path))
    if stage is None:
        raise RuntimeError(f"Failed to open generated USD stage: {usd_path}")

    plug = _find_body_prim(stage, PLUG_LINK_NAME)
    targets = [_find_body_prim(stage, name).GetPath() for name in PLUG_FILTERED_BODY_NAMES]
    filtered_pairs = UsdPhysics.FilteredPairsAPI.Apply(plug).CreateFilteredPairsRel()
    filtered_pairs.SetTargets(targets)
    stage.GetRootLayer().Save()


def _postprocess_usd(usd_path: Path) -> None:
    physics_path = _runtime_physics_stage_path(usd_path)
    physics_stage = Usd.Stage.Open(str(physics_path))
    if physics_stage is None:
        raise RuntimeError(f"Failed to open generated USD stage: {physics_path}")
    _fix_gripper_collision_meshes(physics_stage)
    physics_stage.GetRootLayer().Save()

    stage = Usd.Stage.Open(str(usd_path))
    if stage is None:
        raise RuntimeError(f"Failed to open generated USD stage: {usd_path}")
    _make_plug_collider_editable(stage)
    stage.GetRootLayer().Save()

    # Reopen after de-instancing so the collider descendants are editable rather than instance proxies.
    stage = Usd.Stage.Open(str(usd_path))
    if stage is None:
        raise RuntimeError(f"Failed to reopen generated USD stage: {usd_path}")
    _convert_plug_collider_to_sdf(stage)
    stage.GetRootLayer().Save()
    _author_plug_collision_filters(usd_path)


def _validate_generated_usd(usd_path: Path) -> None:
    """Fail generation if the task-critical plug physics contract is incomplete."""
    stage = Usd.Stage.Open(str(usd_path))
    if stage is None:
        raise RuntimeError(f"Failed to validate generated USD stage: {usd_path}")

    plug = _find_body_prim(stage, PLUG_LINK_NAME)
    plug_colliders = [
        prim
        for prim in stage.TraverseAll()
        if PLUG_LINK_NAME in str(prim.GetPath())
        and prim.HasAPI(UsdPhysics.CollisionAPI)
        and UsdPhysics.MeshCollisionAPI(prim)
    ]
    if len(plug_colliders) != 1:
        raise RuntimeError(f"Expected one generated plug collider, found {len(plug_colliders)}.")
    collider = plug_colliders[0]
    approximation = UsdPhysics.MeshCollisionAPI(collider).GetApproximationAttr().Get()
    sdf_resolution = PhysxSchema.PhysxSDFMeshCollisionAPI(collider).GetSdfResolutionAttr().Get()
    if approximation != PhysxSchema.Tokens.sdf or sdf_resolution != SDF_RESOLUTION:
        raise RuntimeError(
            f"Generated plug collider has approximation={approximation!r}, resolution={sdf_resolution!r}."
        )

    material = UsdShade.MaterialBindingAPI(collider).GetDirectBinding("physics").GetMaterial()
    material_api = UsdPhysics.MaterialAPI(material.GetPrim())
    friction = (material_api.GetStaticFrictionAttr().Get(), material_api.GetDynamicFrictionAttr().Get())
    if friction != (PLUG_FRICTION, PLUG_FRICTION):
        raise RuntimeError(f"Generated plug material has friction {friction}, expected {PLUG_FRICTION}.")

    filtered_targets = UsdPhysics.FilteredPairsAPI(plug).GetFilteredPairsRel().GetTargets()
    if {target.name for target in filtered_targets} != set(PLUG_FILTERED_BODY_NAMES):
        raise RuntimeError(f"Generated plug collision filters are incomplete: {filtered_targets}.")


def _strip_converter_metadata() -> None:
    for metadata_file in (
        R1_PRO_BEAM02_ASSET_DIR / ".asset_hash",
        R1_PRO_BEAM02_ASSET_DIR / "config.yaml",
    ):
        metadata_file.unlink(missing_ok=True)


def main() -> None:
    if not PLUG_MESH_PATH.is_file():
        raise FileNotFoundError(
            f"Missing metre-scaled beam plug mesh: {PLUG_MESH_PATH}. Add/generate the beam assembly assets first."
        )
    if R1_PRO_BEAM02_USD_PATH.exists() and not args_cli.overwrite:
        raise FileExistsError(f"Asset already exists: {R1_PRO_BEAM02_USD_PATH}. Pass --overwrite to regenerate it.")

    R1_PRO_BEAM02_ASSET_DIR.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="r1_pro_beam02_") as temp_dir:
        augmented_urdf = Path(temp_dir) / "r1_pro_beam02.urdf"
        _write_augmented_urdf(augmented_urdf)
        cfg = UrdfConverterCfg(
            asset_path=str(augmented_urdf),
            usd_dir=str(R1_PRO_BEAM02_ASSET_DIR),
            usd_file_name=R1_PRO_BEAM02_USD_PATH.name,
            force_usd_conversion=True,
            make_instanceable=True,
            fix_base=True,
            merge_fixed_joints=False,
            self_collision=False,
            collision_from_visuals=False,
            joint_drive=UrdfConverterCfg.JointDriveCfg(
                target_type="position",
                drive_type="force",
                gains=UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=1000.0, damping=200.0),
            ),
        )
        converter = UrdfConverter(cfg)
        generated_path = Path(converter.usd_path)

    if generated_path.resolve() != R1_PRO_BEAM02_USD_PATH.resolve():
        raise RuntimeError(f"Unexpected generated USD path: {generated_path}")
    _postprocess_usd(generated_path)
    _validate_generated_usd(generated_path)
    _strip_converter_metadata()
    print(f"[INFO] R1 Pro beam02 USD written to: {generated_path}")


try:
    main()
finally:
    try:
        simulation_app.close()
    finally:
        _strip_converter_metadata()
