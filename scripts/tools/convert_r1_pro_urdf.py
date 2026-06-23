#!/usr/bin/env python3
# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Convert the Galaxea R1 Pro URDF asset to a fixed-base USD asset."""

from __future__ import annotations

import argparse
import sys
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

from isaaclab.sim.converters import UrdfConverter, UrdfConverterCfg  # noqa: E402

from assembly_benchmark.robots.r1_pro import R1_PRO_ASSET_DIR, R1_PRO_URDF_PATH, R1_PRO_USD_PATH  # noqa: E402


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
    print(f"[INFO] R1 Pro USD written to: {converter.usd_path}")


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
