# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Generate checked-in USD assets for the FurnitureBench one_leg scene.

This is an offline asset preparation tool. Runtime one_leg tasks load the
generated USD files directly and do not invoke the URDF importer.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from isaaclab.app import AppLauncher


REPO_ROOT = Path(__file__).resolve().parents[2]
ONE_LEG_ASSET_DIR = REPO_ROOT / "source" / "assembly_benchmark" / "assembly_benchmark" / "assets" / "furniture" / "one_leg"
ONE_LEG_URDF_DIR = ONE_LEG_ASSET_DIR / "urdf"
DEFAULT_OUTPUT_DIR = ONE_LEG_ASSET_DIR / "usd"

SDF_RESOLUTION = 512
SDF_SUBGRID_RESOLUTION = 8
SDF_MARGIN = 0.001
SDF_NARROW_BAND_THICKNESS = 0.01

STATIC_ASSETS = {
    "base_tag": ONE_LEG_URDF_DIR / "base_tag.urdf",
    "obstacle_front": ONE_LEG_URDF_DIR / "obstacle_front.urdf",
    "obstacle_side": ONE_LEG_URDF_DIR / "obstacle_side.urdf",
}
DYNAMIC_ASSETS = {
    "square_table_top": (ONE_LEG_URDF_DIR / "square_table" / "square_table_top.urdf", 0.151),
    "square_table_leg1": (ONE_LEG_URDF_DIR / "square_table" / "square_table_leg1.urdf", 0.0231),
    "square_table_leg2": (ONE_LEG_URDF_DIR / "square_table" / "square_table_leg2.urdf", 0.0231),
    "square_table_leg3": (ONE_LEG_URDF_DIR / "square_table" / "square_table_leg3.urdf", 0.0231),
    "square_table_leg4": (ONE_LEG_URDF_DIR / "square_table" / "square_table_leg4.urdf", 0.0231),
}


parser = argparse.ArgumentParser(description="Generate one_leg USD assets from source URDF files.")
parser.add_argument(
    "--output_dir",
    type=Path,
    default=DEFAULT_OUTPUT_DIR,
    help="Directory where generated asset folders are written.",
)
parser.add_argument(
    "--overwrite",
    action="store_true",
    help="Replace existing generated asset folders in the output directory.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.headless = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import isaaclab.sim as sim_utils


def _ensure_urdf_colliders_are_composed(usd_path: str) -> None:
    from pxr import Usd

    stage = Usd.Stage.Open(usd_path)
    if stage is None:
        raise RuntimeError(f"Failed to open generated USD: {usd_path}")

    changed = False
    for prim in stage.TraverseAll():
        if prim.GetName() == "collisions" and prim.IsInstanceable():
            prim.SetInstanceable(False)
            changed = True

    if changed:
        stage.GetRootLayer().Save()


def _set_attr_if_needed(attr, value) -> bool:
    if attr.Get() == value:
        return False
    attr.Set(value)
    return True


def _convert_composed_colliders_to_sdf(usd_path: str) -> None:
    from pxr import PhysxSchema, Usd, UsdPhysics

    _ensure_urdf_colliders_are_composed(usd_path)
    stage = Usd.Stage.Open(usd_path)
    if stage is None:
        raise RuntimeError(f"Failed to open generated USD: {usd_path}")

    changed = False
    for prim in stage.TraverseAll():
        if not prim.HasAPI(UsdPhysics.CollisionAPI):
            continue
        mesh_collision_api = UsdPhysics.MeshCollisionAPI(prim)
        if not mesh_collision_api:
            continue

        changed |= _set_attr_if_needed(mesh_collision_api.GetApproximationAttr(), PhysxSchema.Tokens.sdf)
        if not prim.HasAPI(PhysxSchema.PhysxSDFMeshCollisionAPI):
            sdf_api = PhysxSchema.PhysxSDFMeshCollisionAPI.Apply(prim)
            changed = True
        else:
            sdf_api = PhysxSchema.PhysxSDFMeshCollisionAPI(prim)

        changed |= _set_attr_if_needed(sdf_api.CreateSdfResolutionAttr(), SDF_RESOLUTION)
        changed |= _set_attr_if_needed(sdf_api.CreateSdfSubgridResolutionAttr(), SDF_SUBGRID_RESOLUTION)
        changed |= _set_attr_if_needed(sdf_api.CreateSdfMarginAttr(), SDF_MARGIN)
        changed |= _set_attr_if_needed(sdf_api.CreateSdfNarrowBandThicknessAttr(), SDF_NARROW_BAND_THICKNESS)

    if changed:
        stage.GetRootLayer().Save()


def _strip_converter_metadata(asset_dir: Path) -> None:
    for metadata_file in (asset_dir / ".asset_hash", asset_dir / "config.yaml"):
        metadata_file.unlink(missing_ok=True)


def _prepare_output_dir(asset_dir: Path, overwrite: bool) -> None:
    if not asset_dir.exists():
        asset_dir.mkdir(parents=True)
        return
    if not overwrite:
        raise FileExistsError(f"Generated asset already exists: {asset_dir}. Pass --overwrite to replace it.")
    shutil.rmtree(asset_dir)
    asset_dir.mkdir(parents=True)


def _generate_static_asset(asset_name: str, urdf_path: Path, output_dir: Path, overwrite: bool) -> None:
    asset_dir = output_dir / asset_name
    _prepare_output_dir(asset_dir, overwrite)
    cfg = sim_utils.UrdfFileCfg(
        asset_path=str(urdf_path),
        usd_dir=str(asset_dir),
        fix_base=True,
        make_instanceable=False,
        joint_drive=None,
        collision_from_visuals=False,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=True,
            max_depenetration_velocity=5.0,
        ),
    )
    converter = sim_utils.UrdfConverter(cfg)
    _ensure_urdf_colliders_are_composed(converter.usd_path)
    _strip_converter_metadata(asset_dir)
    print(f"[INFO] Generated static one_leg USD: {converter.usd_path}")


def _generate_dynamic_asset(asset_name: str, urdf_path: Path, mass: float, output_dir: Path, overwrite: bool) -> None:
    asset_dir = output_dir / asset_name
    _prepare_output_dir(asset_dir, overwrite)
    cfg = sim_utils.UrdfFileCfg(
        asset_path=str(urdf_path),
        usd_dir=str(asset_dir),
        fix_base=False,
        make_instanceable=False,
        joint_drive=None,
        collision_from_visuals=False,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=5.0,
        ),
        mass_props=sim_utils.MassPropertiesCfg(mass=mass),
    )
    converter = sim_utils.UrdfConverter(cfg)
    _convert_composed_colliders_to_sdf(converter.usd_path)
    _strip_converter_metadata(asset_dir)
    print(f"[INFO] Generated dynamic SDF one_leg USD: {converter.usd_path}")


def main() -> None:
    output_dir = args_cli.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    for asset_name, urdf_path in STATIC_ASSETS.items():
        _generate_static_asset(asset_name, urdf_path, output_dir, args_cli.overwrite)
    for asset_name, (urdf_path, mass) in DYNAMIC_ASSETS.items():
        _generate_dynamic_asset(asset_name, urdf_path, mass, output_dir, args_cli.overwrite)


try:
    main()
finally:
    simulation_app.close()
