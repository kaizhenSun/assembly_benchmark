# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import importlib.util
from dataclasses import FrozenInstanceError

import pytest

ISAAC_SIM_RUNTIME_AVAILABLE = (
    importlib.util.find_spec("carb") is not None and importlib.util.find_spec("pxr") is not None
)


def test_sensor_public_api_only_exports_r1_pro_head_camera() -> None:
    import assembly_benchmark.sensors as sensors

    assert sensors.__all__ == (
        "R1_PRO_HEAD_CAMERA_SPEC",
        "R1ProHeadCameraSpec",
        "make_r1_pro_head_camera_cfg",
    )


def test_r1_pro_head_camera_spec_matches_asset_generation() -> None:
    from assembly_benchmark.sensors import R1_PRO_HEAD_CAMERA_SPEC, R1ProHeadCameraSpec

    spec = R1_PRO_HEAD_CAMERA_SPEC

    assert isinstance(spec, R1ProHeadCameraSpec)
    assert (spec.width, spec.height) == (1920, 1080)
    assert spec.frame_rate_hz == 30.0
    assert spec.horizontal_fov_deg == 118.0
    assert spec.published_vertical_fov_deg == 62.0
    assert spec.stereo_baseline_m == 0.120
    assert spec.depth_range_m == (0.1, 20.0)
    assert spec.optical_center_pos == (spec.stereo_baseline_m / 2.0, 0.0, 0.020)
    assert spec.default_prim_path == "{ENV_REGEX_NS}/Robot/zed_link/head_camera"


def test_r1_pro_head_camera_pinhole_projection_contract() -> None:
    from assembly_benchmark.sensors import R1_PRO_HEAD_CAMERA_SPEC

    spec = R1_PRO_HEAD_CAMERA_SPEC

    assert spec.update_period_s == pytest.approx(1.0 / 30.0)
    assert spec.horizontal_aperture == pytest.approx(79.88541515282488)
    assert spec.simulated_vertical_fov_deg == pytest.approx(86.22282983803575)
    assert spec.simulated_vertical_fov_deg != pytest.approx(spec.published_vertical_fov_deg)


def test_r1_pro_head_camera_spec_is_frozen() -> None:
    from assembly_benchmark.sensors import R1_PRO_HEAD_CAMERA_SPEC

    with pytest.raises(FrozenInstanceError):
        R1_PRO_HEAD_CAMERA_SPEC.width = 640  # type: ignore[misc]


def test_registered_assembly_part_semantic_labels_are_unique() -> None:
    from assembly_benchmark.assembly import available_assemblies, make_assembly

    for assembly_name in available_assemblies():
        assembly = make_assembly(assembly_name)
        labels = tuple(part.scene_key for part in assembly.parts)

        assert labels == assembly.part_names
        assert len(labels) == len(set(labels))


@pytest.mark.parametrize(
    ("assembly_name", "part_names"),
    [
        ("chair", ("chair_leg1", "chair_leg2", "chair_nut1", "chair_nut2")),
        ("desk", ("desk_leg1", "desk_leg2", "desk_leg3", "desk_leg4")),
        ("stool", ("stool_leg1", "stool_leg2", "stool_leg3")),
    ],
)
def test_same_type_parts_have_distinct_semantic_labels(assembly_name: str, part_names: tuple[str, ...]) -> None:
    from assembly_benchmark.assembly import make_assembly

    assembly = make_assembly(assembly_name)
    semantic_labels = tuple(assembly.part(part_name).scene_key for part_name in part_names)

    assert semantic_labels == part_names
    assert len(semantic_labels) == len(set(semantic_labels))


@pytest.mark.skipif(not ISAAC_SIM_RUNTIME_AVAILABLE, reason="Isaac Sim runtime is not available")
def test_r1_pro_head_camera_factory_builds_tiled_rgbd_semantic_cfg() -> None:
    from assembly_benchmark.sensors import R1_PRO_HEAD_CAMERA_SPEC, make_r1_pro_head_camera_cfg

    from isaaclab.sensors import TiledCameraCfg

    cfg = make_r1_pro_head_camera_cfg()

    assert isinstance(cfg, TiledCameraCfg)
    assert cfg.prim_path == R1_PRO_HEAD_CAMERA_SPEC.default_prim_path
    assert cfg.update_period == pytest.approx(1.0 / 30.0)
    assert (cfg.width, cfg.height) == (1920, 1080)
    assert cfg.data_types == ["rgb", "distance_to_image_plane", "semantic_segmentation"]
    assert cfg.depth_clipping_behavior == "zero"
    assert cfg.semantic_filter == ["class"]
    assert cfg.colorize_semantic_segmentation is False
    assert cfg.update_latest_camera_pose is True
    assert cfg.spawn is not None
    assert cfg.spawn.clipping_range == (0.1, 20.0)
    assert cfg.spawn.focal_length == 24.0
    assert cfg.spawn.horizontal_aperture == pytest.approx(R1_PRO_HEAD_CAMERA_SPEC.horizontal_aperture)
    assert cfg.offset.pos == (0.060, 0.0, 0.020)
    assert cfg.offset.rot == (1.0, 0.0, 0.0, 0.0)
    assert cfg.offset.convention == "ros"


@pytest.mark.skipif(not ISAAC_SIM_RUNTIME_AVAILABLE, reason="Isaac Sim runtime is not available")
def test_scene_assets_have_class_semantic_tags() -> None:
    from assembly_benchmark.assembly import available_assemblies, make_assembly
    from assembly_benchmark.assembly.isaac import make_assembly_part_spawn_cfg
    from assembly_benchmark.assets.furniture.lab_table import make_lab_table_cfg
    from assembly_benchmark.robots.r1_pro import R1_PRO_CFG
    from assembly_benchmark.tasks.direct.assembly_benchmark.assembly_benchmark_env_cfg import (
        AssemblyBenchmarkBaseSceneCfg,
    )

    assert R1_PRO_CFG.spawn is not None
    assert R1_PRO_CFG.spawn.semantic_tags == [("class", "robot")]
    assert make_lab_table_cfg().spawn.semantic_tags == [("class", "lab_table")]
    assert AssemblyBenchmarkBaseSceneCfg.ground.spawn.semantic_tags == [("class", "ground")]

    for assembly_name in available_assemblies():
        assembly = make_assembly(assembly_name)
        for part in assembly.parts:
            spawn = make_assembly_part_spawn_cfg(assembly, part)
            assert spawn.semantic_tags == [("class", part.scene_key)]
