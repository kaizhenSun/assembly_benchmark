from __future__ import annotations

from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg

LAB_TABLE_ASSET_DIR = Path(__file__).resolve().parent
LAB_TABLE_USD_PATH = LAB_TABLE_ASSET_DIR / "lab_table.usd"

LAB_TABLE_SURFACE_Z = 0.775


def make_lab_table_cfg(prim_path: str = "{ENV_REGEX_NS}/LabTable") -> AssetBaseCfg:
    return AssetBaseCfg(
        prim_path=prim_path,
        spawn=sim_utils.UsdFileCfg(usd_path=str(LAB_TABLE_USD_PATH)),
    )
