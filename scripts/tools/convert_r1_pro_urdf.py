# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Convert the Galaxea R1 Pro URDF asset to a fixed-base USD asset."""

from __future__ import annotations

import argparse
import atexit
import sys
import time
from pathlib import Path

from isaaclab.app import AppLauncher

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = PROJECT_ROOT / "source" / "assembly_benchmark"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

parser = argparse.ArgumentParser(description="Convert R1 Pro URDF to USD.")
parser.add_argument("--force", action="store_true", help="Force USD regeneration even if the output exists.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from assembly_benchmark.robots.r1_pro import R1_PRO_ASSET_DIR, R1_PRO_URDF_PATH, R1_PRO_USD_PATH  # noqa: E402
from assembly_benchmark.sensors import (  # noqa: E402
    R1_PRO_GRIPPER_TACTILE_MATERIAL,
    R1_PRO_GRIPPER_TACTILE_PAD_SPECS,
    R1_PRO_GRIPPER_TACTILE_VISUAL_RGBA,
    apply_tactile_compliant_material,
)

from pxr import Gf, Usd, UsdPhysics, UsdShade  # noqa: E402

from isaaclab.sim.converters import UrdfConverter, UrdfConverterCfg  # noqa: E402

GRIPPER_COLLISION_PATHS = (
    "/colliders/left_gripper_link/left_gripper_link_collision",
    "/colliders/left_gripper_finger_link1/left_gripper_finger_link1_collision",
    "/colliders/left_gripper_finger_link2/left_gripper_finger_link2_collision",
    "/colliders/right_gripper_link/right_gripper_link_collision",
    "/colliders/right_gripper_finger_link1/right_gripper_finger_link1_collision",
    "/colliders/right_gripper_finger_link2/right_gripper_finger_link2_collision",
    *(pad_spec.usd_collider_root_path for pad_spec in R1_PRO_GRIPPER_TACTILE_PAD_SPECS),
)
_POSTPROCESS_COMPLETE = False


def _runtime_physics_stage_path(usd_path: Path) -> Path:
    physics_usd_path = usd_path.parent / "configuration" / f"{usd_path.stem}_physics.usd"
    return physics_usd_path if physics_usd_path.exists() else usd_path


def _runtime_base_stage_path(usd_path: Path) -> Path:
    base_usd_path = usd_path.parent / "configuration" / f"{usd_path.stem}_base.usd"
    return base_usd_path if base_usd_path.exists() else usd_path


def _find_single_mesh_under(stage: Usd.Stage, root_path: str, description: str) -> Usd.Prim:
    root = stage.GetPrimAtPath(root_path)
    if not root.IsValid():
        raise RuntimeError(f"Could not find {description} in generated USD: {root_path}")

    mesh_prims = [prim for prim in Usd.PrimRange(root) if prim.GetTypeName() == "Mesh"]
    if len(mesh_prims) != 1:
        raise RuntimeError(f"Expected exactly one mesh under {root_path}, found {len(mesh_prims)}")
    return mesh_prims[0]


def _fix_nested_collision_mesh_apis(usd_path: Path) -> None:
    """Apply collision APIs to the imported STL mesh prim when the importer nests it under Xforms."""
    stage_path = _runtime_physics_stage_path(usd_path)

    stage = Usd.Stage.Open(str(stage_path))
    if stage is None:
        raise RuntimeError(f"Failed to open USD stage: {stage_path}")

    fixed_mesh_paths: list[str] = []
    for collision_path in GRIPPER_COLLISION_PATHS:
        mesh_prim = _find_single_mesh_under(stage, collision_path, "gripper collision root")
        collision_api = UsdPhysics.CollisionAPI.Apply(mesh_prim)
        mesh_collision_api = UsdPhysics.MeshCollisionAPI.Apply(mesh_prim)
        collision_api.CreateCollisionEnabledAttr(True)
        mesh_collision_api.CreateApproximationAttr(UsdPhysics.Tokens.convexHull)
        applied_schemas = set(mesh_prim.GetAppliedSchemas())
        if not {"PhysicsCollisionAPI", "PhysicsMeshCollisionAPI"}.issubset(applied_schemas):
            raise RuntimeError(f"Failed to apply collision APIs to mesh prim: {mesh_prim.GetPath()}")
        fixed_mesh_paths.append(str(mesh_prim.GetPath()))

    stage.GetRootLayer().Save()
    print(f"[INFO] Fixed gripper collision APIs on {len(fixed_mesh_paths)} mesh prims in {stage_path}")


def _apply_gripper_tactile_pad_visual_materials(usd_path: Path) -> None:
    """Set the display color of all gripper tactile pads without changing collision materials."""
    stage_path = _runtime_base_stage_path(usd_path)

    stage = Usd.Stage.Open(str(stage_path))
    if stage is None:
        raise RuntimeError(f"Failed to open USD stage: {stage_path}")

    updated_shader_paths: list[str] = []
    for pad_spec in R1_PRO_GRIPPER_TACTILE_PAD_SPECS:
        mesh_prim = _find_single_mesh_under(
            stage,
            pad_spec.usd_visual_root_path,
            "gripper tactile pad visual root",
        )
        material_targets = UsdShade.MaterialBindingAPI(mesh_prim).GetDirectBindingRel().GetTargets()
        if len(material_targets) != 1:
            raise RuntimeError(
                f"Expected one visual material under {pad_spec.usd_visual_root_path}, found {len(material_targets)}"
            )

        material_prim = stage.GetPrimAtPath(material_targets[0])
        shader_prims = [prim for prim in Usd.PrimRange(material_prim) if prim.GetTypeName() == "Shader"]
        if len(shader_prims) != 1:
            raise RuntimeError(
                f"Expected one visual material shader under {material_prim.GetPath()}, found {len(shader_prims)}"
            )

        shader = UsdShade.Shader(shader_prims[0])
        diffuse_color_input = shader.GetInput("diffuse_color_constant")
        opacity_input = shader.GetInput("opacity_constant")
        if not diffuse_color_input or not opacity_input:
            raise RuntimeError(f"Tactile pad visual shader has unexpected inputs: {shader.GetPath()}")
        diffuse_color_input.Set(Gf.Vec3f(*R1_PRO_GRIPPER_TACTILE_VISUAL_RGBA[:3]))
        opacity_input.Set(R1_PRO_GRIPPER_TACTILE_VISUAL_RGBA[3])
        updated_shader_paths.append(str(shader.GetPath()))

    stage.GetRootLayer().Save()
    print(f"[INFO] Applied configured visual material to gripper tactile pads {updated_shader_paths} in {stage_path}")


def _apply_gripper_tactile_pad_compliant_materials(usd_path: Path) -> None:
    """Bind independent PhysX compliant contact materials to all tactile pad colliders."""
    stage_path = _runtime_physics_stage_path(usd_path)

    stage = Usd.Stage.Open(str(stage_path))
    if stage is None:
        raise RuntimeError(f"Failed to open USD stage: {stage_path}")

    bound_mesh_paths: list[str] = []
    for pad_spec in R1_PRO_GRIPPER_TACTILE_PAD_SPECS:
        mesh_prim = _find_single_mesh_under(
            stage,
            pad_spec.usd_collider_root_path,
            "gripper tactile pad collider root",
        )
        stage.DefinePrim(f"{pad_spec.usd_collider_root_path}/materials", "Scope")
        apply_tactile_compliant_material(
            stage,
            mesh_prim,
            pad_spec.usd_material_path,
            R1_PRO_GRIPPER_TACTILE_MATERIAL,
        )
        bound_mesh_paths.append(str(mesh_prim.GetPath()))

    stage.GetRootLayer().Save()
    print(f"[INFO] Applied compliant materials to gripper tactile pad colliders {bound_mesh_paths} in {stage_path}")


def _postprocess_r1_pro_usd(usd_path: Path) -> None:
    """Apply all runtime USD fixes that should survive URDF reconversion."""
    global _POSTPROCESS_COMPLETE
    _fix_nested_collision_mesh_apis(usd_path)
    _apply_gripper_tactile_pad_visual_materials(usd_path)
    _apply_gripper_tactile_pad_compliant_materials(usd_path)
    _POSTPROCESS_COMPLETE = True


def _strip_converter_metadata(asset_dir: Path) -> None:
    """Remove importer metadata that is not needed by the checked-in runtime asset."""
    for metadata_file in (asset_dir / ".asset_hash", asset_dir / "config.yaml"):
        metadata_file.unlink(missing_ok=True)


atexit.register(_strip_converter_metadata, R1_PRO_ASSET_DIR)


def main() -> None:
    """Convert the R1 Pro URDF to the runtime USD file."""
    R1_PRO_ASSET_DIR.mkdir(parents=True, exist_ok=True)
    cfg = UrdfConverterCfg(
        asset_path=str(R1_PRO_URDF_PATH),
        usd_dir=str(R1_PRO_ASSET_DIR),
        usd_file_name=R1_PRO_USD_PATH.name,
        force_usd_conversion=args_cli.force,
        make_instanceable=True,
        fix_base=True,
        # Keep fixed frames such as left_gripper_link, right_gripper_link, and camera mounts available for IK/sensors.
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
    _postprocess_r1_pro_usd(Path(converter.usd_path))
    _strip_converter_metadata(R1_PRO_ASSET_DIR)
    print(f"[INFO] R1 Pro USD written to: {converter.usd_path}", flush=True)


if __name__ == "__main__":
    try:
        main()
    finally:
        try:
            if not _POSTPROCESS_COMPLETE and R1_PRO_USD_PATH.exists():
                _postprocess_r1_pro_usd(R1_PRO_USD_PATH)
        finally:
            simulation_app.close()
            time.sleep(0.5)
            _strip_converter_metadata(R1_PRO_ASSET_DIR)
