# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass

from assembly_benchmark.assets.furniture.lab_table import LAB_TABLE_SURFACE_Z, make_lab_table_cfg
from assembly_benchmark.robots.r1_pro import R1_PRO_CFG

ONE_LEG_ASSET_DIR = Path(__file__).resolve().parents[3] / "assets" / "furniture" / "one_leg"
ONE_LEG_USD_DIR = ONE_LEG_ASSET_DIR / "usd"

ROT_Z_90 = (0.70710678, 0.0, 0.0, 0.70710678)
SQUARE_TABLE_LEG_ROT = (0.5, 0.5, 0.5, -0.5)


def _usd_path(asset_name: str) -> str:
    return str(ONE_LEG_USD_DIR / asset_name / f"{asset_name}.usd")


def _static_usd(asset_name: str) -> sim_utils.UsdFileCfg:
    return sim_utils.UsdFileCfg(
        usd_path=_usd_path(asset_name),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=True,
            max_depenetration_velocity=5.0,
        ),
    )


def _dynamic_usd(asset_name: str, mass: float) -> sim_utils.UsdFileCfg:
    return sim_utils.UsdFileCfg(
        usd_path=_usd_path(asset_name),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=5.0,
        ),
        mass_props=sim_utils.MassPropertiesCfg(mass=mass),
    )


@configclass
class OneLegSceneCfg(InteractiveSceneCfg):
    """FurnitureBench one_leg scene layout for R1 Pro."""

    ground = AssetBaseCfg(
        prim_path="/World/defaultGroundPlane",
        spawn=sim_utils.GroundPlaneCfg(),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
    )

    dome_light = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75)),
    )

    robot = R1_PRO_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    lab_table = make_lab_table_cfg()

    base_tag = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/BaseTag",
        spawn=sim_utils.UsdFileCfg(usd_path=_usd_path("base_tag")),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.5015, 0.0, 0.775)),
    )

    obstacle_front = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/ObstacleFront",
        spawn=_static_usd("obstacle_front"),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.8815, 0.0, 0.79), rot=ROT_Z_90),
    )

    obstacle_right = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/ObstacleRight",
        spawn=_static_usd("obstacle_side"),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.8065, -0.175, 0.79), rot=ROT_Z_90),
    )

    obstacle_left = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/ObstacleLeft",
        spawn=_static_usd("obstacle_side"),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.8065, 0.175, 0.79), rot=ROT_Z_90),
    )

    square_table_top = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/SquareTableTop",
        spawn=_dynamic_usd(
            "square_table_top",
            0.151,
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.7415, 0.0, 0.790625),
            rot=(0.5, 0.5, -0.5, -0.5),
        ),
    )

    square_table_leg1 = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/SquareTableLeg1",
        spawn=_dynamic_usd(
            "square_table_leg1",
            0.0231,
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.5715, -0.2, 0.79), rot=SQUARE_TABLE_LEG_ROT
        ),
    )

    square_table_leg2 = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/SquareTableLeg2",
        spawn=_dynamic_usd(
            "square_table_leg2",
            0.0231,
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.5715, -0.12, 0.79), rot=SQUARE_TABLE_LEG_ROT
        ),
    )

    square_table_leg3 = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/SquareTableLeg3",
        spawn=_dynamic_usd(
            "square_table_leg3",
            0.0231,
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.5715, 0.12, 0.79), rot=SQUARE_TABLE_LEG_ROT
        ),
    )

    square_table_leg4 = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/SquareTableLeg4",
        spawn=_dynamic_usd(
            "square_table_leg4",
            0.0231,
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.5715, 0.2, 0.79), rot=SQUARE_TABLE_LEG_ROT
        ),
    )


@configclass
class OneLegSceneEnvCfg(DirectRLEnvCfg):
    """Minimal task shell that only loads the FurnitureBench one_leg scene."""

    decimation = 2
    episode_length_s = 20.0

    action_space = 1
    observation_space = 1
    state_space = 0

    sim: SimulationCfg = SimulationCfg(dt=1 / 120, render_interval=decimation)
    scene: OneLegSceneCfg = OneLegSceneCfg(
        num_envs=16, env_spacing=4.0, replicate_physics=True
    )
    table_surface_z = LAB_TABLE_SURFACE_Z
