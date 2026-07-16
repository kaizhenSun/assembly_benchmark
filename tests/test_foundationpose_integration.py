# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import numpy as np
import pytest

import scripts.tools.foundationpose as foundationpose_module
from scripts.tools.foundationpose import (
    BOUNDING_BOX_EDGES,
    FOUNDATIONPOSE_RESULT_PREFIX,
    canonicalize_pose_by_symmetry,
    depth_mm_uint16,
    ensure_foundationpose_container,
    local_bounding_box_corners,
    mesh_from_part_urdf,
    package_obj_mesh,
    parse_foundationpose_result,
    parse_foundationpose_visualization,
    pos_quat_from_pose_matrix,
    pose_errors,
    pose_matrix_from_pos_quat,
    project_camera_points,
    render_foundationpose_overlay,
    run_foundationpose_adapter,
    target_mask,
    target_semantic_ids,
    validate_pose_matrix,
)


def test_projection_and_box_geometry_are_deterministic() -> None:
    corners = local_bounding_box_corners(np.array(((-1.0, -2.0, -3.0), (1.0, 2.0, 3.0))))
    assert corners.tolist() == [
        [-1.0, -2.0, -3.0],
        [-1.0, -2.0, 3.0],
        [-1.0, 2.0, -3.0],
        [-1.0, 2.0, 3.0],
        [1.0, -2.0, -3.0],
        [1.0, -2.0, 3.0],
        [1.0, 2.0, -3.0],
        [1.0, 2.0, 3.0],
    ]
    assert len(BOUNDING_BOX_EDGES) == 12
    assert {index for edge in BOUNDING_BOX_EDGES for index in edge} == set(range(8))
    pixels = project_camera_points(np.array(((0.0, 0.0, 2.0), (1.0, -1.0, 2.0))), np.diag((100, 100, 1)))
    np.testing.assert_allclose(pixels, ((0.0, 0.0), (50.0, -50.0)))


def test_projection_rejects_non_positive_depth() -> None:
    with pytest.raises(ValueError, match="non-positive"):
        project_camera_points(np.array(((0.0, 0.0, -1.0),)), np.eye(3))


def test_overlay_has_input_size_uint8_and_uses_supplied_top_pose() -> None:
    rgb = np.full((64, 64, 3), 40, dtype=np.uint8)
    masks = {"top": np.zeros((64, 64), dtype=np.uint8), "leg": np.zeros((64, 64), dtype=np.uint8)}
    masks["top"][20:40, 20:40] = 255
    masks["leg"][42:52, 42:52] = 255
    camera_matrix = np.array(((100.0, 0.0, 32.0), (0.0, 100.0, 32.0), (0.0, 0.0, 1.0)))
    top_pose = pose_matrix_from_pos_quat(np.array((0.0, 0.0, 1.0)), np.array((2**-0.5, 0, 0, 2**-0.5)))
    leg_pose = pose_matrix_from_pos_quat(np.array((0.15, 0.15, 1.0)), np.array((1, 0, 0, 0)))
    geometry = {
        name: {
            "bbox_corners": local_bounding_box_corners(((-0.03, -0.03, -0.03), (0.03, 0.03, 0.03))),
            "axis_length_m": 0.1,
        }
        for name in ("top", "leg")
    }
    overlay = render_foundationpose_overlay(rgb, masks, camera_matrix, {"top": top_pose, "leg": leg_pose}, geometry)
    assert overlay.shape == rgb.shape
    assert overlay.dtype == np.uint8
    # A canonicalized +90 degree top pose sends its red local-X axis down in the image.
    assert overlay[42, 32].tolist() == [255, 0, 0]


def test_overlay_rejects_image_size_mismatch() -> None:
    pose = np.eye(4)
    pose[2, 3] = 1.0
    geometry = {"top": {"bbox_corners": np.zeros((8, 3)), "axis_length_m": 0.1}}
    with pytest.raises(ValueError, match="expected"):
        render_foundationpose_overlay(
            np.zeros((8, 8, 3), dtype=np.uint8),
            {"top": np.zeros((7, 8), dtype=np.uint8)},
            np.eye(3),
            {"top": pose},
            geometry,
        )


def test_foundationpose_entry_is_independent_from_ground_truth_script() -> None:
    tools_dir = Path(__file__).resolve().parents[1] / "scripts/tools"
    foundationpose_source = (tools_dir / "run_r1_pro_one_leg_foundationpose_assembly.py").read_text()
    scripted_source = (tools_dir / "run_r1_pro_one_leg_scripted_assembly.py").read_text()

    assert "run_r1_pro_one_leg_scripted_assembly" not in foundationpose_source
    assert "execv" not in foundationpose_source
    assert "def _run_assembly(" in foundationpose_source
    assert "foundationpose" not in scripted_source.lower()


def test_foundationpose_visualization_cli_and_planning_pose_wiring() -> None:
    source = (
        Path(__file__).resolve().parents[1] / "scripts/tools/run_r1_pro_one_leg_foundationpose_assembly.py"
    ).read_text()
    assert '"--disable_foundationpose_overlay"' in source
    assert '"--disable_foundationpose_markers"' in source
    assert '"--foundationpose_marker_scale"' in source
    assert "--foundationpose_marker_scale must be finite and positive" in source
    assert "_write_estimate_overlay(request_path, planning_poses_c, adapter_payload)" in source
    assert "_make_estimate_markers(camera_pose_w_cv, planning_poses_c" in source


def _completed(returncode: int = 0, stdout: str = "", stderr: str = "") -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(["docker"], returncode, stdout, stderr)


class _QueuedRunner:
    def __init__(self, *results: subprocess.CompletedProcess[str]):
        self.results = list(results)
        self.calls: list[tuple[list[str], float | None]] = []

    def __call__(self, command: list[str], timeout: float | None) -> subprocess.CompletedProcess[str]:
        self.calls.append((command, timeout))
        return self.results.pop(0)


def test_depth_and_semantic_inputs_follow_foundationpose_contract() -> None:
    depth = np.array([[[[0.0], [0.1234], [np.nan], [70.0]]]], dtype=np.float32)
    depth_mm = depth_mm_uint16(depth)
    assert depth_mm.dtype == np.uint16
    assert depth_mm.tolist() == [[0, 123, 0, 0]]

    info = {"idToLabels": {"0": "BACKGROUND", "7": "{'class': 'square_table_leg4'}"}}
    ids = target_semantic_ids(info, "square_table_leg4")
    mask = target_mask(np.array([[0, 7], [7, 0]], dtype=np.int32), ids, "square_table_leg4")
    assert ids == (7,)
    assert mask.tolist() == [[0, 255], [255, 0]]


def test_empty_semantic_mask_is_rejected() -> None:
    with pytest.raises(RuntimeError, match="is empty"):
        target_mask(np.zeros((2, 2), dtype=np.int32), (7,), "square_table_leg4")


def test_depth_without_valid_samples_is_rejected() -> None:
    with pytest.raises(RuntimeError, match="no valid"):
        depth_mm_uint16(np.zeros((1, 2, 2, 1), dtype=np.float32))


@pytest.mark.parametrize(
    "quat",
    [
        np.array([1.0, 0.0, 0.0, 0.0]),
        np.array([2**-0.5, 0.0, 0.0, 2**-0.5]),
        np.array([0.0, 1.0, 0.0, 0.0]),
    ],
)
def test_pose_matrix_round_trip_uses_wxyz_quaternions(quat: np.ndarray) -> None:
    transform = pose_matrix_from_pos_quat(np.array([1.0, -2.0, 3.0]), quat)
    position_out, quat_out = pos_quat_from_pose_matrix(transform)
    reconstructed = pose_matrix_from_pos_quat(position_out, quat_out)
    np.testing.assert_allclose(reconstructed, transform, atol=1.0e-7)


def test_camera_world_and_top_leg_transform_composition() -> None:
    world_camera = pose_matrix_from_pos_quat(np.array([0.5, -0.2, 1.0]), np.array([2**-0.5, 0.0, 0.0, 2**-0.5]))
    camera_top = pose_matrix_from_pos_quat(np.array([0.1, 0.2, 0.8]), np.array([1.0, 0.0, 0.0, 0.0]))
    top_leg = pose_matrix_from_pos_quat(np.array([-0.05625, 0.046875, -0.05625]), np.array([1, 0, 0, 0]))

    world_top = world_camera @ camera_top
    world_leg_goal = world_top @ top_leg
    np.testing.assert_allclose(world_leg_goal, world_camera @ camera_top @ top_leg)


@pytest.mark.parametrize("raw_quarter_turn", [0, 1, 2, 3])
def test_square_top_symmetry_is_canonicalized_without_changing_translation(raw_quarter_turn: int) -> None:
    canonical_rotation = np.array(((0, 0, -1), (-1, 0, 0), (0, 1, 0)), dtype=np.float64)
    symmetries = tuple(
        np.array(
            (
                (np.cos(angle), 0, np.sin(angle)),
                (0, 1, 0),
                (-np.sin(angle), 0, np.cos(angle)),
            )
        )
        for angle in (0.0, np.pi / 2.0, np.pi, 3.0 * np.pi / 2.0)
    )
    raw_pose = np.eye(4)
    raw_pose[:3, :3] = canonical_rotation @ symmetries[raw_quarter_turn]
    raw_pose[:3, 3] = (0.7, -0.1, 0.8)

    canonical_pose, symmetry_index, error = canonicalize_pose_by_symmetry(
        raw_pose,
        canonical_rotation,
        symmetries,
    )

    np.testing.assert_allclose(canonical_pose[:3, :3], canonical_rotation, atol=1.0e-7)
    np.testing.assert_allclose(canonical_pose[:3, 3], raw_pose[:3, 3])
    assert symmetry_index == (-raw_quarter_turn) % 4
    assert error == pytest.approx(0.0, abs=1.0e-7)


def test_pose_symmetry_canonicalization_rejects_invalid_inputs() -> None:
    with pytest.raises(ValueError, match="At least one"):
        canonicalize_pose_by_symmetry(np.eye(4), np.eye(3), ())
    with pytest.raises(ValueError, match="must be 3x3"):
        canonicalize_pose_by_symmetry(np.eye(4), np.eye(3), (np.eye(4),))


def test_opencv_camera_pose_converts_through_world_to_robot_root() -> None:
    world_root = pose_matrix_from_pos_quat(np.array([0.2, -0.1, 0.0]), np.array([1, 0, 0, 0]))
    world_camera_cv = pose_matrix_from_pos_quat(np.array([0.5, 0.3, 1.2]), np.array([2**-0.5, 0.0, 2**-0.5, 0.0]))
    camera_object = pose_matrix_from_pos_quat(np.array([0.0, 0.0, 0.8]), np.array([1, 0, 0, 0]))
    root_object = np.linalg.inv(world_root) @ world_camera_cv @ camera_object
    reconstructed_camera_object = np.linalg.inv(world_camera_cv) @ world_root @ root_object
    np.testing.assert_allclose(reconstructed_camera_object, camera_object, atol=1.0e-8)


def test_finger_leg_relation_stays_fixed_during_motion() -> None:
    root_leg = pose_matrix_from_pos_quat(np.array([0.6, 0.2, 0.8]), np.array([1, 0, 0, 0]))
    root_finger_at_grasp = pose_matrix_from_pos_quat(np.array([0.6, 0.2, 0.85]), np.array([1, 0, 0, 0]))
    finger_leg = np.linalg.inv(root_finger_at_grasp) @ root_leg
    root_finger_lifted = root_finger_at_grasp.copy()
    root_finger_lifted[2, 3] += 0.14
    propagated_leg = root_finger_lifted @ finger_leg
    np.testing.assert_allclose(propagated_leg[:3, 3], [0.6, 0.2, 0.94])


def test_adapter_result_parser_requires_exact_valid_objects() -> None:
    identity = np.eye(4)
    identity[2, 3] = 0.5
    payload = {
        "schema_version": 1,
        "poses": {"square_table_top": identity.tolist(), "square_table_leg4": identity.tolist()},
    }
    stdout = "log line\n" + FOUNDATIONPOSE_RESULT_PREFIX + json.dumps(payload) + "\n"
    poses = parse_foundationpose_result(stdout, ("square_table_top", "square_table_leg4"))
    assert set(poses) == {"square_table_top", "square_table_leg4"}

    payload["poses"]["square_table_leg4"][0][0] = 2.0
    with pytest.raises(ValueError, match="not orthonormal"):
        parse_foundationpose_result(
            FOUNDATIONPOSE_RESULT_PREFIX + json.dumps(payload),
            ("square_table_top", "square_table_leg4"),
        )

    with pytest.raises(RuntimeError, match="found 2"):
        parse_foundationpose_result(
            stdout + FOUNDATIONPOSE_RESULT_PREFIX + json.dumps(payload),
            ("square_table_top", "square_table_leg4"),
        )


def test_adapter_visualization_geometry_has_exact_dual_object_mapping() -> None:
    corners = local_bounding_box_corners(((-0.1, -0.2, -0.3), (0.1, 0.2, 0.3))).tolist()
    payload = {
        "visualization": {
            "square_table_top": {"bbox_corners": corners, "axis_length_m": 0.06},
            "square_table_leg4": {"bbox_corners": corners, "axis_length_m": 0.04},
        }
    }
    result = parse_foundationpose_visualization(payload, ("square_table_top", "square_table_leg4"))
    assert tuple(result) == ("square_table_top", "square_table_leg4")
    assert result["square_table_top"]["bbox_corners"].shape == (8, 3)


@pytest.mark.parametrize(
    "visualization, error",
    [
        (None, "missing"),
        ({"square_table_top": {"bbox_corners": [[0, 0, 0]] * 8, "axis_length_m": 0.1}}, "mismatch"),
        (
            {
                "square_table_top": {"bbox_corners": [[0, 0, 0]] * 8, "axis_length_m": 0.1},
                "square_table_leg4": {"bbox_corners": [[0, 0, float("nan")]] * 8, "axis_length_m": 0.1},
            },
            "finite 8x3",
        ),
    ],
)
def test_invalid_adapter_visualization_geometry_is_rejected(visualization: object, error: str) -> None:
    payload = {} if visualization is None else {"visualization": visualization}
    with pytest.raises(RuntimeError, match=error):
        parse_foundationpose_visualization(payload, ("square_table_top", "square_table_leg4"))


def test_container_missing_is_rejected(tmp_path: Path) -> None:
    runner = _QueuedRunner(_completed(1, stderr="No such container"))
    with pytest.raises(RuntimeError, match="does not exist"):
        ensure_foundationpose_container("foundationpose", tmp_path, Path("/opt/FoundationPose"), runner)


def test_stopped_container_is_started_and_mount_is_checked(tmp_path: Path) -> None:
    runner = _QueuedRunner(
        _completed(stdout=json.dumps({"Running": False})),
        _completed(stdout="foundationpose\n"),
        _completed(
            stdout=json.dumps(
                [
                    {"Source": str(tmp_path.parent), "Destination": str(tmp_path.parent)},
                    {"Source": "/opt/FoundationPose", "Destination": "/opt/FoundationPose"},
                ]
            )
        ),
    )
    ensure_foundationpose_container("foundationpose", tmp_path, Path("/opt/FoundationPose"), runner)
    assert runner.calls[1][0] == ["docker", "start", "foundationpose"]
    assert len(runner.calls) == 3


def test_container_mount_mismatch_is_rejected(tmp_path: Path) -> None:
    runner = _QueuedRunner(
        _completed(stdout=json.dumps({"Running": True})),
        _completed(stdout=json.dumps([{"Source": "/unrelated", "Destination": "/unrelated"}])),
    )
    with pytest.raises(RuntimeError, match="not visible"):
        ensure_foundationpose_container("foundationpose", tmp_path, Path("/opt/FoundationPose"), runner)


def test_host_command_timeout_has_clear_error(monkeypatch: pytest.MonkeyPatch) -> None:
    def timeout(*_args, **_kwargs):
        raise subprocess.TimeoutExpired(["docker", "exec"], 3.0)

    monkeypatch.setattr(foundationpose_module.subprocess, "run", timeout)
    with pytest.raises(RuntimeError, match="timed out after 3.0 s"):
        foundationpose_module.run_host_command(["docker", "exec"], 3.0)


def test_adapter_nonzero_exit_persists_logs(tmp_path: Path) -> None:
    runner = _QueuedRunner(_completed(17, stdout="partial", stderr="CUDA failed"))
    with pytest.raises(RuntimeError, match="exit code 17"):
        run_foundationpose_adapter(
            ["docker", "exec"], tmp_path, 10.0, ("square_table_top", "square_table_leg4"), runner
        )
    assert (tmp_path / "adapter_stdout.log").read_text() == "partial"
    assert (tmp_path / "adapter_stderr.log").read_text() == "CUDA failed"


def test_adapter_protocol_reports_one_initialization_and_two_registrations(tmp_path: Path) -> None:
    pose = np.eye(4)
    pose[2, 3] = 0.5
    payload = {
        "schema_version": 1,
        "poses": {"square_table_top": pose.tolist(), "square_table_leg4": pose.tolist()},
        "model_initializations": 1,
        "register_calls": 2,
    }
    stdout = "log\n" + FOUNDATIONPOSE_RESULT_PREFIX + json.dumps(payload) + "\n"
    runner = _QueuedRunner(_completed(stdout=stdout))
    poses, returned_payload, _, _ = run_foundationpose_adapter(
        ["docker", "exec"], tmp_path, 10.0, ("square_table_top", "square_table_leg4"), runner
    )
    assert set(poses) == {"square_table_top", "square_table_leg4"}
    assert returned_payload["model_initializations"] == 1
    assert runner.calls[0][1] == 10.0


def test_pose_errors_report_translation_and_rotation() -> None:
    ground_truth = pose_matrix_from_pos_quat(np.zeros(3), np.array([1, 0, 0, 0]))
    estimated = pose_matrix_from_pos_quat(np.array([0.01, 0.0, 0.0]), np.array([np.cos(0.05), 0.0, 0.0, np.sin(0.05)]))
    translation, rotation = pose_errors(estimated, ground_truth)
    assert translation == pytest.approx(0.01)
    assert rotation == pytest.approx(0.1)


def test_one_leg_mesh_packaging_is_self_contained(tmp_path: Path) -> None:
    asset_root = (
        Path(__file__).resolve().parents[1] / "source/assembly_benchmark/assembly_benchmark/assets/furniture/one_leg"
    )
    source_obj = mesh_from_part_urdf(asset_root / "urdf/square_table/square_table_leg4.urdf")
    output_obj = package_obj_mesh(source_obj, tmp_path / "mesh")
    assert output_obj.name == "textured_simple.obj"
    assert (output_obj.parent / "textured_simple.mtl").is_file()
    assert len(tuple((output_obj.parent / "textures").glob("*.png"))) == 4


def test_invalid_pose_homogeneous_row_is_rejected() -> None:
    pose = np.eye(4)
    pose[3, 3] = 2.0
    with pytest.raises(ValueError, match="homogeneous row"):
        validate_pose_matrix(pose, require_positive_z=False)


def test_reflected_pose_rotation_is_rejected() -> None:
    pose = np.eye(4)
    pose[0, 0] = -1.0
    with pytest.raises(ValueError, match="determinant"):
        validate_pose_matrix(pose, require_positive_z=False)


@pytest.mark.parametrize("invalid_z", [-0.1, 0.0, 21.0, np.nan])
def test_invalid_camera_depth_is_rejected(invalid_z: float) -> None:
    pose = np.eye(4)
    pose[2, 3] = invalid_z
    with pytest.raises(ValueError, match="finite 4x4|camera Z"):
        validate_pose_matrix(pose)
