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

from assembly_benchmark.robots.r1_pro import R1_PRO_CFG

ONE_LEG_ASSET_DIR = Path(__file__).resolve().parents[3] / "assets" / "furniture" / "one_leg"
ONE_LEG_URDF_DIR = ONE_LEG_ASSET_DIR / "urdf"
ONE_LEG_USD_CACHE_DIR = Path("/tmp/assembly_benchmark/furniture_usd_cache/one_leg")

ROT_Z_90 = (0.70710678, 0.0, 0.0, 0.70710678)
SQUARE_TABLE_LEG_ROT = (0.5, 0.5, 0.5, -0.5)
TABLE_LEG_SIZE = (0.045, 0.045, 0.725)
TABLE_LEG_COLOR = (0.08, 0.08, 0.08)


def _table_leg(prim_path: str, pos: tuple[float, float, float]) -> AssetBaseCfg:
    return AssetBaseCfg(
        prim_path=prim_path,
        spawn=sim_utils.CuboidCfg(
            size=TABLE_LEG_SIZE,
            collision_props=sim_utils.CollisionPropertiesCfg(),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=0.8, dynamic_friction=0.8, restitution=0.6
            ),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=TABLE_LEG_COLOR),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=pos),
    )


def _static_urdf(asset_path: Path, cache_name: str) -> sim_utils.UrdfFileCfg:
    return sim_utils.UrdfFileCfg(
        asset_path=str(asset_path),
        usd_dir=str(ONE_LEG_USD_CACHE_DIR / cache_name),
        fix_base=True,
        make_instanceable=False,
        joint_drive=None,
        collision_from_visuals=False,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=True,
            max_depenetration_velocity=5.0,
        ),
    )


def _dynamic_urdf(asset_path: Path, cache_name: str, mass: float) -> sim_utils.UrdfFileCfg:
    return sim_utils.UrdfFileCfg(
        asset_path=str(asset_path),
        usd_dir=str(ONE_LEG_USD_CACHE_DIR / cache_name),
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

    tabletop = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Tabletop",
        spawn=sim_utils.CuboidCfg(
            size=(0.7, 1.2, 0.05),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=0.8, dynamic_friction=0.8, restitution=0.6
            ),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.03, 0.14, 0.11))
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.6, 0.0, 0.75)),
    )

    table_leg_front_left = _table_leg(
        "{ENV_REGEX_NS}/TableLegFrontLeft", (0.3, 0.55, 0.3625)
    )
    table_leg_front_right = _table_leg(
        "{ENV_REGEX_NS}/TableLegFrontRight", (0.9, 0.55, 0.3625)
    )
    table_leg_back_left = _table_leg(
        "{ENV_REGEX_NS}/TableLegBackLeft", (0.3, -0.55, 0.3625)
    )
    table_leg_back_right = _table_leg(
        "{ENV_REGEX_NS}/TableLegBackRight", (0.9, -0.55, 0.3625)
    )

    base_tag = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/BaseTag",
        spawn=_static_urdf(ONE_LEG_URDF_DIR / "base_tag.urdf", "base_tag"),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.5015, 0.0, 0.775)),
    )

    obstacle_front = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/ObstacleFront",
        spawn=_static_urdf(ONE_LEG_URDF_DIR / "obstacle_front.urdf", "obstacle_front"),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.8815, 0.0, 0.79), rot=ROT_Z_90),
    )

    obstacle_right = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/ObstacleRight",
        spawn=_static_urdf(ONE_LEG_URDF_DIR / "obstacle_side.urdf", "obstacle_side"),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.8065, -0.175, 0.79), rot=ROT_Z_90),
    )

    obstacle_left = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/ObstacleLeft",
        spawn=_static_urdf(ONE_LEG_URDF_DIR / "obstacle_side.urdf", "obstacle_side"),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.8065, 0.175, 0.79), rot=ROT_Z_90),
    )

    square_table_top = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/SquareTableTop",
        spawn=_dynamic_urdf(
            ONE_LEG_URDF_DIR / "square_table" / "square_table_top.urdf",
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
        spawn=_dynamic_urdf(
            ONE_LEG_URDF_DIR / "square_table" / "square_table_leg1.urdf",
            "square_table_leg1",
            0.0231,
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.5715, -0.2, 0.79), rot=SQUARE_TABLE_LEG_ROT
        ),
    )

    square_table_leg2 = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/SquareTableLeg2",
        spawn=_dynamic_urdf(
            ONE_LEG_URDF_DIR / "square_table" / "square_table_leg2.urdf",
            "square_table_leg2",
            0.0231,
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.5715, -0.12, 0.79), rot=SQUARE_TABLE_LEG_ROT
        ),
    )

    square_table_leg3 = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/SquareTableLeg3",
        spawn=_dynamic_urdf(
            ONE_LEG_URDF_DIR / "square_table" / "square_table_leg3.urdf",
            "square_table_leg3",
            0.0231,
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.5715, 0.12, 0.79), rot=SQUARE_TABLE_LEG_ROT
        ),
    )

    square_table_leg4 = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/SquareTableLeg4",
        spawn=_dynamic_urdf(
            ONE_LEG_URDF_DIR / "square_table" / "square_table_leg4.urdf",
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
