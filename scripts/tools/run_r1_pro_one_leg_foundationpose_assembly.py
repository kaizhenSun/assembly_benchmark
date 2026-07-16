# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Estimate one-leg assembly poses with FoundationPose and execute a minimal insert sequence."""

from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
import time
import traceback
from dataclasses import dataclass
from pathlib import Path

from isaaclab.app import AppLauncher

TASK_NAME = "Assembly-Benchmark-OneLeg-Direct-v0"
TOP_NAME = "square_table_top"
LEG_NAME = "square_table_leg4"
OPEN_GRIPPER = 1.0
CLOSE_GRIPPER = -1.0
TOP_DOWN_QUAT = (1.0, 0.0, 0.0, 0.0)
FINGER_TABLE_CLEARANCE = 0.02
FINGER_COLLISION_HALF_HEIGHT = 0.02
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
TOP_CANONICAL_ROTATION_ROOT = (
    (0.0, 0.0, -1.0),
    (-1.0, 0.0, 0.0),
    (0.0, 1.0, 0.0),
)

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    "--disable_fabric",
    action="store_true",
    default=False,
    help="Disable Fabric and use USD I/O for debugging.",
)
parser.add_argument("--phase_steps", type=int, default=120, help="Action steps for each arm motion.")
parser.add_argument("--close_steps", type=int, default=160, help="Action steps for opening or closing the gripper.")
parser.add_argument("--settle_steps", type=int, default=30, help="Action steps for short stationary holds.")
parser.add_argument("--lift_height", type=float, default=0.14, help="Vertical lift after grasping the leg.")
parser.add_argument(
    "--overhead_clearance",
    type=float,
    default=0.24,
    help="Vertical clearance for grasp and insertion approach waypoints.",
)
parser.add_argument(
    "--insert_clearance",
    type=float,
    default=0.04,
    help="Clearance above the final leg pose before seating.",
)
parser.add_argument(
    "--insert_push_depth",
    type=float,
    default=0.015,
    help="Optional downward offset applied to the final seated leg target.",
)
parser.add_argument(
    "--finger_center_offset_z",
    type=float,
    default=0.0,
    help="Additional Z offset between the estimated leg origin and grasp finger center.",
)
parser.add_argument(
    "--foundationpose_container",
    default="foundationpose",
    help="FoundationPose Docker container name.",
)
parser.add_argument(
    "--foundationpose_timeout_s",
    type=float,
    default=300.0,
    help="Timeout for the single batched FoundationPose invocation.",
)
parser.add_argument(
    "--foundationpose_est_refine_iter",
    type=int,
    default=5,
    help="FoundationPose refinement iterations per object.",
)
parser.add_argument(
    "--foundationpose_warmup_steps",
    type=int,
    default=60,
    help="Static action steps before capturing the shared RGB-D frame.",
)
parser.add_argument(
    "--foundationpose_capture_root",
    type=Path,
    default=Path("logs/foundationpose"),
    help="Host capture root mounted at the same path in the FoundationPose container.",
)
parser.add_argument(
    "--foundationpose_repo",
    type=Path,
    default=Path("/home/kaizhen/assembly_ws/FoundationPose"),
    help="FoundationPose checkout mounted at the same path in its container.",
)
parser.add_argument(
    "--disable_foundationpose_overlay",
    action="store_true",
    help="Disable the saved RGB FoundationPose estimate overlay.",
)
parser.add_argument(
    "--disable_foundationpose_markers",
    action="store_true",
    help="Disable persistent Isaac Sim frames at the initial FoundationPose estimates.",
)
parser.add_argument(
    "--foundationpose_marker_scale",
    type=float,
    default=0.06,
    help="Axis length in meters for the persistent FoundationPose frame markers.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

for name in ("phase_steps", "close_steps", "settle_steps"):
    if getattr(args_cli, name) <= 0:
        raise ValueError(f"--{name} must be positive.")
if args_cli.lift_height <= 0.0 or args_cli.overhead_clearance <= 0.0:
    raise ValueError("--lift_height and --overhead_clearance must be positive.")
if args_cli.insert_clearance < 0.0 or args_cli.insert_push_depth < 0.0:
    raise ValueError("--insert_clearance and --insert_push_depth must be non-negative.")
if args_cli.foundationpose_timeout_s <= 0.0:
    raise ValueError("--foundationpose_timeout_s must be positive.")
if args_cli.foundationpose_est_refine_iter <= 0:
    raise ValueError("--foundationpose_est_refine_iter must be positive.")
if args_cli.foundationpose_warmup_steps < 0:
    raise ValueError("--foundationpose_warmup_steps must be non-negative.")
if not math.isfinite(args_cli.foundationpose_marker_scale) or args_cli.foundationpose_marker_scale <= 0.0:
    raise ValueError("--foundationpose_marker_scale must be finite and positive.")

args_cli.enable_cameras = True
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import assembly_benchmark.tasks  # noqa: F401, E402
import gymnasium as gym  # noqa: E402
import imageio.v3 as iio  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from assembly_benchmark.assembly import make_assembly  # noqa: E402
from assembly_benchmark.assets.furniture.lab_table import LAB_TABLE_SURFACE_Z  # noqa: E402
from foundationpose import (  # noqa: E402
    canonicalize_pose_by_symmetry,
    depth_mm_uint16,
    ensure_foundationpose_container,
    mesh_from_part_urdf,
    package_obj_mesh,
    parse_foundationpose_visualization,
    pos_quat_from_pose_matrix,
    pose_errors,
    pose_matrix_from_pos_quat,
    render_foundationpose_overlay,
    rgb_uint8,
    run_foundationpose_adapter,
    semantic_ids,
    single_camera_array,
    target_mask,
    target_semantic_ids,
)

from isaaclab.markers import VisualizationMarkers  # noqa: E402
from isaaclab.markers.config import FRAME_MARKER_CFG  # noqa: E402
from isaaclab.utils.math import (  # noqa: E402
    combine_frame_transforms,
    quat_apply,
    quat_apply_inverse,
    subtract_frame_transforms,
)

import isaaclab_tasks  # noqa: F401, E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402


@dataclass
class MotionState:
    """Commanded bimanual IK state for the single environment."""

    left_pose: torch.Tensor
    right_pose: torch.Tensor
    left_grip: float = OPEN_GRIPPER
    right_grip: float = OPEN_GRIPPER
    action_step: int = 0


@dataclass
class EstimateContext:
    """One-shot pose estimates and their persistent audit data."""

    run_dir: Path
    leg_pose_b: torch.Tensor
    leg_goal_pose_b: torch.Tensor
    metrics: dict[str, object]
    estimate_markers: tuple[VisualizationMarkers, VisualizationMarkers] | None = None


def _top_symmetry_rotations() -> tuple[np.ndarray, ...]:
    """Return the square tabletop's four object-local rotations about its normal."""
    rotations = []
    for quarter_turn in range(4):
        angle = quarter_turn * np.pi / 2.0
        cosine = float(np.cos(angle))
        sine = float(np.sin(angle))
        rotations.append(
            np.array(
                ((cosine, 0.0, sine), (0.0, 1.0, 0.0), (-sine, 0.0, cosine)),
                dtype=np.float64,
            )
        )
    return tuple(rotations)


def _normalize_quat(quat: torch.Tensor) -> torch.Tensor:
    return quat / torch.linalg.norm(quat, dim=-1, keepdim=True).clamp_min(1.0e-8)


def _quat_inv(quat: torch.Tensor) -> torch.Tensor:
    conjugate = quat.clone()
    conjugate[..., 1:] *= -1.0
    return conjugate / torch.sum(quat * quat, dim=-1, keepdim=True).clamp_min(1.0e-8)


def _quat_mul(first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
    w1, x1, y1, z1 = first.unbind(dim=-1)
    w2, x2, y2, z2 = second.unbind(dim=-1)
    return torch.stack(
        (
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ),
        dim=-1,
    )


def _nlerp_quat(start: torch.Tensor, target: torch.Tensor, alpha: float) -> torch.Tensor:
    start = _normalize_quat(start)
    target = _normalize_quat(target)
    target = torch.where(torch.sum(start * target, dim=-1, keepdim=True) < 0.0, -target, target)
    return _normalize_quat((1.0 - alpha) * start + alpha * target)


def _quat_angle_error(current: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    dot = torch.abs(torch.sum(_normalize_quat(current) * _normalize_quat(target), dim=-1)).clamp(max=1.0)
    return 2.0 * torch.acos(dot)


def _pose(pos: torch.Tensor, quat: torch.Tensor) -> torch.Tensor:
    return torch.cat((pos, _normalize_quat(quat)), dim=-1)


def _make_action(state: MotionState) -> torch.Tensor:
    actions = torch.zeros((1, 16), dtype=torch.float32, device=state.left_pose.device)
    actions[:, 0:7] = state.left_pose
    actions[:, 7] = state.left_grip
    actions[:, 8:15] = state.right_pose
    actions[:, 15] = state.right_grip
    return actions


def _step_without_reset(env, actions: torch.Tensor) -> None:
    """Advance DirectRLEnv without reward, done evaluation, or automatic reset."""
    unwrapped = env.unwrapped
    unwrapped._pre_physics_step(actions.to(unwrapped.device))
    is_rendering = unwrapped.sim.has_gui() or unwrapped.sim.has_rtx_sensors()
    for _ in range(int(unwrapped.cfg.decimation)):
        unwrapped._sim_step_counter += 1
        unwrapped._apply_action()
        unwrapped.scene.write_data_to_sim()
        unwrapped.sim.step(render=False)
        if unwrapped._sim_step_counter % unwrapped.cfg.sim.render_interval == 0 and is_rendering:
            unwrapped.sim.render()
        unwrapped.scene.update(dt=unwrapped.physics_dt)


def _active_ee_pose(unwrapped, active_arm: str) -> torch.Tensor:
    left_pose, right_pose = unwrapped._get_ee_poses_in_root_frame()
    return left_pose.clone() if active_arm == "left" else right_pose.clone()


def _body_position_in_root(unwrapped, body_name: str) -> torch.Tensor:
    body_index = unwrapped.robot.find_bodies(body_name)[0][0]
    body_pose_w = unwrapped.robot.data.body_pose_w[:, body_index]
    root_pose_w = unwrapped.robot.data.root_pose_w
    position_b, _ = subtract_frame_transforms(
        root_pose_w[:, :3],
        root_pose_w[:, 3:7],
        body_pose_w[:, :3],
        body_pose_w[:, 3:7],
    )
    return position_b


def _finger_center(unwrapped, active_arm: str) -> torch.Tensor:
    first = _body_position_in_root(unwrapped, f"{active_arm}_gripper_finger_link1")
    second = _body_position_in_root(unwrapped, f"{active_arm}_gripper_finger_link2")
    return 0.5 * (first + second)


def _finger_pose(unwrapped, active_arm: str) -> torch.Tensor:
    ee_pose = _active_ee_pose(unwrapped, active_arm)
    return _pose(_finger_center(unwrapped, active_arm), ee_pose[:, 3:7])


def _finger_offset_for_quat(unwrapped, active_arm: str, target_quat: torch.Tensor) -> torch.Tensor:
    ee_pose = _active_ee_pose(unwrapped, active_arm)
    offset_b = _finger_center(unwrapped, active_arm) - ee_pose[:, :3]
    offset_ee = quat_apply_inverse(ee_pose[:, 3:7], offset_b)
    return quat_apply(target_quat, offset_ee)


def _ee_for_finger_center(
    unwrapped,
    active_arm: str,
    desired_center: torch.Tensor,
    desired_quat: torch.Tensor,
) -> torch.Tensor:
    safe_center = desired_center.clone()
    minimum_z = LAB_TABLE_SURFACE_Z + FINGER_COLLISION_HALF_HEIGHT + FINGER_TABLE_CLEARANCE
    safe_center[:, 2] = torch.maximum(safe_center[:, 2], torch.full_like(safe_center[:, 2], minimum_z))
    offset = _finger_offset_for_quat(unwrapped, active_arm, desired_quat)
    return _pose(safe_center - offset, desired_quat)


def _finger_leg_relation(
    unwrapped,
    active_arm: str,
    estimated_leg_pose_b: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    finger_pose_b = _finger_pose(unwrapped, active_arm)
    return subtract_frame_transforms(
        finger_pose_b[:, :3],
        finger_pose_b[:, 3:7],
        estimated_leg_pose_b[:, :3],
        estimated_leg_pose_b[:, 3:7],
    )


def _propagated_leg_pose(
    unwrapped,
    active_arm: str,
    leg_pos_in_finger: torch.Tensor,
    leg_quat_in_finger: torch.Tensor,
) -> torch.Tensor:
    finger_pose_b = _finger_pose(unwrapped, active_arm)
    leg_pos_b, leg_quat_b = combine_frame_transforms(
        finger_pose_b[:, :3],
        finger_pose_b[:, 3:7],
        leg_pos_in_finger,
        leg_quat_in_finger,
    )
    return _pose(leg_pos_b, leg_quat_b)


def _ee_for_held_leg(
    unwrapped,
    active_arm: str,
    desired_leg_pose_b: torch.Tensor,
    leg_pos_in_finger: torch.Tensor,
    leg_quat_in_finger: torch.Tensor,
) -> torch.Tensor:
    finger_quat = _normalize_quat(_quat_mul(desired_leg_pose_b[:, 3:7], _quat_inv(leg_quat_in_finger)))
    finger_pos = desired_leg_pose_b[:, :3] - quat_apply(finger_quat, leg_pos_in_finger)
    return _ee_for_finger_center(unwrapped, active_arm, finger_pos, finger_quat)


def _run_phase(
    env,
    state: MotionState,
    active_arm: str,
    name: str,
    target_pose: torch.Tensor,
    target_grip: float,
    steps: int,
    context: EstimateContext,
) -> None:
    """Interpolate one arm pose and active gripper without consulting simulator object truth."""
    start_pose = state.left_pose.clone() if active_arm == "left" else state.right_pose.clone()
    start_grip = state.left_grip if active_arm == "left" else state.right_grip
    for index in range(steps):
        if not simulation_app.is_running():
            raise RuntimeError(f"Simulation stopped during phase '{name}'.")
        alpha = float(index + 1) / float(steps)
        command_pose = start_pose.clone()
        command_pose[:, :3] = (1.0 - alpha) * start_pose[:, :3] + alpha * target_pose[:, :3]
        command_pose[:, 3:7] = _nlerp_quat(start_pose[:, 3:7], target_pose[:, 3:7], alpha)
        command_grip = (1.0 - alpha) * start_grip + alpha * target_grip
        if active_arm == "left":
            state.left_pose = command_pose
            state.left_grip = command_grip
        else:
            state.right_pose = command_pose
            state.right_grip = command_grip
        _step_without_reset(env, _make_action(state))
        state.action_step += 1

    actual_pose = _active_ee_pose(env.unwrapped, active_arm)
    position_error = float(torch.linalg.norm(actual_pose[:, :3] - target_pose[:, :3], dim=-1)[0].item())
    orientation_error = float(_quat_angle_error(actual_pose[:, 3:7], target_pose[:, 3:7])[0].item())
    context.metrics.setdefault("stages", []).append(
        {
            "name": name,
            "action_step": state.action_step,
            "target_ee_pose_in_root": target_pose[0].detach().cpu().tolist(),
            "actual_ee_pose_in_root": actual_pose[0].detach().cpu().tolist(),
            "position_error_m": position_error,
            "orientation_error_rad": orientation_error,
            "active_gripper": target_grip,
        }
    )
    _write_metrics(context)
    print(
        f"[INFO]: phase={name} step={state.action_step} "
        f"ee_error={position_error:.4f}m ee_ori_error={orientation_error:.4f}rad"
    )


def _pose_tensor_to_matrix(pose: torch.Tensor) -> np.ndarray:
    values = pose[0].detach().cpu().numpy()
    return pose_matrix_from_pos_quat(values[:3], values[3:7])


def _matrix_to_pose_tensor(transform: np.ndarray, device: torch.device | str) -> torch.Tensor:
    position, quaternion = pos_quat_from_pose_matrix(transform)
    return torch.as_tensor(
        np.concatenate((position, quaternion)),
        dtype=torch.float32,
        device=device,
    ).reshape(1, 7)


def _asset_pose_w_matrix(asset) -> np.ndarray:
    pose_w = asset.data.root_pose_w[0].detach().cpu().numpy()
    return pose_matrix_from_pos_quat(pose_w[:3], pose_w[3:7])


def _camera_pose_w_cv(camera) -> np.ndarray:
    position = camera.data.pos_w[0].detach().cpu().numpy()
    quaternion = camera.data.quat_w_ros[0].detach().cpu().numpy()
    return pose_matrix_from_pos_quat(position, quaternion)


def _new_run_directory() -> Path:
    capture_root = args_cli.foundationpose_capture_root.resolve()
    capture_root.mkdir(parents=True, exist_ok=True)
    run_dir = capture_root / f"one_leg_online_{time.strftime('%Y%m%d_%H%M%S')}"
    if run_dir.exists():
        raise FileExistsError(f"FoundationPose run directory already exists: {run_dir}")
    run_dir.mkdir()
    adapter_source = Path(__file__).with_name("foundationpose_container_adapter.py")
    if not adapter_source.is_file():
        raise RuntimeError(f"FoundationPose container adapter does not exist: {adapter_source}")
    shutil.copy2(adapter_source, run_dir / adapter_source.name)
    return run_dir


def _capture_request(env, state: MotionState) -> tuple[Path, np.ndarray]:
    unwrapped = env.unwrapped
    hold_action = _make_action(state)
    for _ in range(args_cli.foundationpose_warmup_steps):
        _step_without_reset(env, hold_action)
        state.action_step += 1

    camera = unwrapped.scene["head_camera"]
    required = {"rgb", "distance_to_image_plane", "semantic_segmentation"}
    missing = required.difference(camera.data.output)
    if missing:
        raise RuntimeError(f"Head camera is missing outputs: {sorted(missing)}")

    rgb = rgb_uint8(camera.data.output["rgb"])
    depth_mm = depth_mm_uint16(camera.data.output["distance_to_image_plane"])
    semantic = semantic_ids(camera.data.output["semantic_segmentation"])
    if rgb.shape[:2] != depth_mm.shape or depth_mm.shape != semantic.shape:
        raise RuntimeError(
            f"Camera dimensions differ: rgb={rgb.shape}, depth={depth_mm.shape}, semantic={semantic.shape}."
        )
    camera_matrix = single_camera_array(camera.data.intrinsic_matrices, "intrinsic_matrices").astype(np.float64)
    if camera_matrix.shape != (3, 3) or not np.all(np.isfinite(camera_matrix)):
        raise RuntimeError(f"Invalid head-camera intrinsic matrix: {camera_matrix}")

    run_dir = _new_run_directory()
    rgb_path = run_dir / "rgb.png"
    depth_path = run_dir / "depth.png"
    camera_matrix_path = run_dir / "cam_K.txt"
    iio.imwrite(rgb_path, rgb)
    iio.imwrite(depth_path, depth_mm)
    np.savetxt(camera_matrix_path, camera_matrix, fmt="%.18e")
    camera_pose_w_cv = _camera_pose_w_cv(camera)
    np.savetxt(run_dir / "camera_pose_w_cv.txt", camera_pose_w_cv, fmt="%.18e")

    semantic_info = camera.data.info.get("semantic_segmentation")
    assembly = make_assembly("one_leg")
    objects = []
    for object_name in (TOP_NAME, LEG_NAME):
        object_dir = run_dir / "objects" / object_name
        object_dir.mkdir(parents=True)
        mask = target_mask(
            semantic,
            target_semantic_ids(semantic_info, object_name),
            object_name,
        )
        mask_path = object_dir / "mask.png"
        iio.imwrite(mask_path, mask)
        part = assembly.part(object_name)
        source_obj = mesh_from_part_urdf(part.urdf_path(assembly.asset_root))
        mesh_path = package_obj_mesh(source_obj, object_dir / "mesh")
        objects.append(
            {
                "name": object_name,
                "mask_path": str(mask_path),
                "mesh_path": str(mesh_path),
                "mask_pixels": int(np.count_nonzero(mask)),
            }
        )

    request = {
        "schema_version": 1,
        "rgb_path": str(rgb_path),
        "depth_path": str(depth_path),
        "camera_matrix_path": str(camera_matrix_path),
        "depth_scale_to_meters": 0.001,
        "objects": objects,
    }
    request_path = run_dir / "request.json"
    request_path.write_text(json.dumps(request, indent=2) + "\n", encoding="utf-8")
    return request_path, camera_pose_w_cv


def _invoke_adapter(request_path: Path) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    ensure_foundationpose_container(
        args_cli.foundationpose_container,
        request_path.parent,
        args_cli.foundationpose_repo,
    )
    adapter_path = request_path.parent / "foundationpose_container_adapter.py"
    command = [
        "docker",
        "exec",
        "--workdir",
        str(args_cli.foundationpose_repo.resolve()),
        args_cli.foundationpose_container,
        "python",
        str(adapter_path.resolve()),
        "--request",
        str(request_path.resolve()),
        "--est_refine_iter",
        str(args_cli.foundationpose_est_refine_iter),
        "--debug",
        "0",
    ]
    poses, payload, _, _ = run_foundationpose_adapter(
        command,
        request_path.parent,
        args_cli.foundationpose_timeout_s,
        (TOP_NAME, LEG_NAME),
    )
    return poses, payload


def _write_estimate_overlay(
    request_path: Path,
    poses_c: dict[str, np.ndarray],
    adapter_payload: dict[str, object],
) -> Path:
    """Render the exact camera-frame estimates used by planning over the registration frame."""
    try:
        request = json.loads(request_path.read_text(encoding="utf-8"))
        rgb = np.asarray(iio.imread(request["rgb_path"]))
        camera_matrix = np.asarray(np.loadtxt(request["camera_matrix_path"]), dtype=np.float64)
        object_items = request.get("objects")
        if not isinstance(object_items, list):
            raise ValueError("FoundationPose request 'objects' must be a list.")
        masks = {item["name"]: np.asarray(iio.imread(item["mask_path"])) for item in object_items}
        geometry = parse_foundationpose_visualization(adapter_payload, (TOP_NAME, LEG_NAME))
        overlay = render_foundationpose_overlay(
            rgb,
            masks,
            camera_matrix,
            {TOP_NAME: poses_c[TOP_NAME], LEG_NAME: poses_c[LEG_NAME]},
            geometry,
            labels={TOP_NAME: f"{TOP_NAME} (canonicalized pose)", LEG_NAME: LEG_NAME},
        )
        output_path = request_path.parent / "visualization" / "foundationpose_estimates.png"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        iio.imwrite(output_path, overlay)
    except Exception as exc:
        raise RuntimeError(f"Could not create FoundationPose estimate overlay: {exc}") from exc
    print(f"[INFO]: FoundationPose estimate overlay saved to: {output_path}")
    return output_path


def _make_estimate_markers(
    camera_pose_w_cv: np.ndarray,
    poses_c: dict[str, np.ndarray],
    device: torch.device | str,
) -> tuple[VisualizationMarkers, VisualizationMarkers]:
    """Create persistent world-space frames at the initial planning estimates."""
    marker_cfg = FRAME_MARKER_CFG.copy()
    scale = args_cli.foundationpose_marker_scale
    marker_cfg.markers["frame"].scale = (scale, scale, scale)
    top_marker = VisualizationMarkers(
        marker_cfg.replace(prim_path="/Visuals/foundationpose_estimates/square_table_top")
    )
    leg_marker = VisualizationMarkers(
        marker_cfg.replace(prim_path="/Visuals/foundationpose_estimates/square_table_leg4")
    )
    for marker, name in ((top_marker, TOP_NAME), (leg_marker, LEG_NAME)):
        pose_w = _matrix_to_pose_tensor(camera_pose_w_cv @ poses_c[name], device)
        marker.visualize(pose_w[:, :3], pose_w[:, 3:7])
    return top_marker, leg_marker


def _estimate_context(env, state: MotionState) -> EstimateContext:
    unwrapped = env.unwrapped
    request_path, camera_pose_w_cv = _capture_request(env, state)
    poses_c, adapter_payload = _invoke_adapter(request_path)
    run_dir = request_path.parent

    root_from_world = np.linalg.inv(_asset_pose_w_matrix(unwrapped.robot))
    raw_poses_b = {name: root_from_world @ camera_pose_w_cv @ pose for name, pose in poses_c.items()}
    top_symmetries = _top_symmetry_rotations()
    canonical_top_b, symmetry_index, canonical_alignment_error = canonicalize_pose_by_symmetry(
        raw_poses_b[TOP_NAME],
        np.asarray(TOP_CANONICAL_ROTATION_ROOT, dtype=np.float64),
        top_symmetries,
    )
    poses_b = dict(raw_poses_b)
    poses_b[TOP_NAME] = canonical_top_b
    canonical_top_c = poses_c[TOP_NAME].copy()
    canonical_top_c[:3, :3] = poses_c[TOP_NAME][:3, :3] @ top_symmetries[symmetry_index]
    planning_poses_c = dict(poses_c)
    planning_poses_c[TOP_NAME] = canonical_top_c
    target_top = _pose_tensor_to_matrix(unwrapped.scripted_target_pose)
    leg_goal_b = poses_b[TOP_NAME] @ target_top

    overlay_path = None
    if not args_cli.disable_foundationpose_overlay:
        overlay_path = _write_estimate_overlay(request_path, planning_poses_c, adapter_payload)
    estimate_markers = None
    if not args_cli.disable_foundationpose_markers:
        estimate_markers = _make_estimate_markers(camera_pose_w_cv, planning_poses_c, unwrapped.device)

    camera_from_world = np.linalg.inv(camera_pose_w_cv)
    truth_c = {
        TOP_NAME: camera_from_world @ _asset_pose_w_matrix(unwrapped.assembly_parent_part),
        LEG_NAME: camera_from_world @ _asset_pose_w_matrix(unwrapped.assembly_child_part),
    }
    object_metrics = {}
    for name in (TOP_NAME, LEG_NAME):
        translation_error, rotation_error = pose_errors(planning_poses_c[name], truth_c[name])
        object_metrics[name] = {
            "translation_error_m": translation_error,
            "rotation_error_rad": rotation_error,
            "estimated_pose_in_camera": planning_poses_c[name].tolist(),
            "ground_truth_pose_in_camera": truth_c[name].tolist(),
            "estimated_pose_in_root": poses_b[name].tolist(),
        }
    raw_top_translation_error, raw_top_rotation_error = pose_errors(poses_c[TOP_NAME], truth_c[TOP_NAME])
    object_metrics[TOP_NAME].update(
        {
            "raw_translation_error_m": raw_top_translation_error,
            "raw_rotation_error_rad": raw_top_rotation_error,
            "raw_estimated_pose_in_camera": poses_c[TOP_NAME].tolist(),
            "raw_estimated_pose_in_root": raw_poses_b[TOP_NAME].tolist(),
        }
    )

    metrics: dict[str, object] = {
        "schema_version": 1,
        "pose_source": "foundationpose",
        "ground_truth_used_for_planning": False,
        "visualization": {
            "overlay_enabled": not args_cli.disable_foundationpose_overlay,
            "overlay_path": None if overlay_path is None else str(overlay_path),
            "overlay_pose_types": {
                TOP_NAME: "foundationpose_symmetry_canonicalized",
                LEG_NAME: "foundationpose_registration",
            },
            "markers_enabled": not args_cli.disable_foundationpose_markers,
            "markers_created": estimate_markers is not None,
            "marker_scale_m": args_cli.foundationpose_marker_scale,
            "marker_pose_types": {
                TOP_NAME: "foundationpose_symmetry_canonicalized",
                LEG_NAME: "foundationpose_registration",
            },
        },
        "top_symmetry_canonicalization": {
            "basis": "fixed one_leg task orientation prior",
            "axis": "object_local_y",
            "quarter_turn_index": symmetry_index,
            "applied_angle_rad": symmetry_index * np.pi / 2.0,
            "canonical_alignment_error_rad": canonical_alignment_error,
        },
        "objects": object_metrics,
        "estimated_leg_goal_in_root": leg_goal_b.tolist(),
        "stages": [],
    }
    (run_dir / "foundationpose_result.json").write_text(
        json.dumps(adapter_payload, indent=2) + "\n",
        encoding="utf-8",
    )
    context = EstimateContext(
        run_dir=run_dir,
        leg_pose_b=_matrix_to_pose_tensor(poses_b[LEG_NAME], unwrapped.device),
        leg_goal_pose_b=_matrix_to_pose_tensor(leg_goal_b, unwrapped.device),
        metrics=metrics,
        estimate_markers=estimate_markers,
    )
    _write_metrics(context)
    print(
        f"[INFO]: estimates ready run_dir={run_dir} "
        f"top_error={object_metrics[TOP_NAME]['translation_error_m']:.4f}m "
        f"leg_error={object_metrics[LEG_NAME]['translation_error_m']:.4f}m"
    )
    return context


def _write_metrics(context: EstimateContext) -> None:
    (context.run_dir / "metrics.json").write_text(
        json.dumps(context.metrics, indent=2) + "\n",
        encoding="utf-8",
    )


def _write_final_metrics(context: EstimateContext, unwrapped) -> bool:
    success = bool(unwrapped._success()[0].item())
    relative_pos, relative_quat = unwrapped._assembled_relative_pose()
    position_error = relative_pos - unwrapped.scripted_target_pose[:, :3]
    orientation_error = _quat_angle_error(relative_quat, unwrapped.scripted_target_pose[:, 3:7])
    terminated, truncated = unwrapped._get_dones()
    context.metrics["final"] = {
        "success": success,
        "reward": float(unwrapped._get_rewards()[0].item()),
        "terminated": bool(terminated[0].item()),
        "truncated": bool(truncated[0].item()),
        "relative_position": relative_pos[0].detach().cpu().tolist(),
        "relative_quaternion_wxyz": relative_quat[0].detach().cpu().tolist(),
        "position_error": position_error[0].detach().cpu().tolist(),
        "orientation_error_rad": float(orientation_error[0].item()),
    }
    _write_metrics(context)
    print(
        f"[INFO]: final success={success} "
        f"position_error={position_error[0].detach().cpu().tolist()} "
        f"orientation_error={float(orientation_error[0].item()):.4f}rad"
    )
    return success


def _run_assembly(env) -> int:
    unwrapped = env.unwrapped
    env.reset()
    with torch.inference_mode():
        left_pose, right_pose = unwrapped._get_ee_poses_in_root_frame()
        state = MotionState(left_pose=left_pose.clone(), right_pose=right_pose.clone())
        context = _estimate_context(env, state)

        left_pose, right_pose = unwrapped._get_ee_poses_in_root_frame()
        state.left_pose = left_pose.clone()
        state.right_pose = right_pose.clone()
        active_arm = "left" if float(context.leg_pose_b[0, 1].item()) > 0.0 else "right"
        active_pose = state.left_pose if active_arm == "left" else state.right_pose
        grasp_quat = torch.tensor((TOP_DOWN_QUAT,), dtype=torch.float32, device=unwrapped.device)

        grasp_center = context.leg_pose_b[:, :3].clone()
        grasp_center[:, 2] += args_cli.finger_center_offset_z
        pregrasp_center = grasp_center.clone()
        pregrasp_center[:, 2] += args_cli.overhead_clearance
        grasp_pose = _ee_for_finger_center(unwrapped, active_arm, grasp_center, grasp_quat)
        pregrasp_pose = _ee_for_finger_center(unwrapped, active_arm, pregrasp_center, grasp_quat)
        raise_pos = active_pose[:, :3].clone()
        raise_pos[:, 2] = torch.maximum(raise_pos[:, 2], pregrasp_pose[:, 2])
        raise_pose = _pose(raise_pos, grasp_quat)

        print(
            f"[INFO]: assembly start arm={active_arm} "
            f"estimated_leg={context.leg_pose_b[0].detach().cpu().tolist()} "
            f"estimated_goal={context.leg_goal_pose_b[0].detach().cpu().tolist()}"
        )
        _run_phase(env, state, active_arm, "raise-arm", raise_pose, OPEN_GRIPPER, args_cli.phase_steps, context)
        _run_phase(
            env,
            state,
            active_arm,
            "move-above-grasp",
            pregrasp_pose,
            OPEN_GRIPPER,
            args_cli.phase_steps,
            context,
        )
        _run_phase(
            env,
            state,
            active_arm,
            "descend-to-grasp",
            grasp_pose,
            OPEN_GRIPPER,
            args_cli.phase_steps,
            context,
        )
        _run_phase(
            env,
            state,
            active_arm,
            "settle-at-grasp",
            grasp_pose,
            OPEN_GRIPPER,
            args_cli.settle_steps,
            context,
        )
        _run_phase(
            env,
            state,
            active_arm,
            "close-gripper",
            grasp_pose,
            CLOSE_GRIPPER,
            args_cli.close_steps,
            context,
        )

        leg_pos_in_finger, leg_quat_in_finger = _finger_leg_relation(
            unwrapped,
            active_arm,
            context.leg_pose_b,
        )
        context.metrics["estimated_leg_pose_in_finger"] = (
            torch.cat((leg_pos_in_finger, leg_quat_in_finger), dim=-1)[0].detach().cpu().tolist()
        )
        _write_metrics(context)

        lift_center = _finger_center(unwrapped, active_arm).clone()
        lift_center[:, 2] += args_cli.lift_height
        lift_pose = _ee_for_finger_center(unwrapped, active_arm, lift_center, grasp_quat)
        _run_phase(env, state, active_arm, "lift", lift_pose, CLOSE_GRIPPER, args_cli.phase_steps, context)

        propagated_leg = _propagated_leg_pose(
            unwrapped,
            active_arm,
            leg_pos_in_finger,
            leg_quat_in_finger,
        )
        reorient_leg = propagated_leg.clone()
        high_z = max(
            float(context.leg_goal_pose_b[0, 2].item()) + args_cli.overhead_clearance + args_cli.insert_clearance,
            LAB_TABLE_SURFACE_Z + args_cli.overhead_clearance,
        )
        reorient_leg[:, 2] = torch.maximum(reorient_leg[:, 2], torch.full_like(reorient_leg[:, 2], high_z))
        reorient_leg[:, 3:7] = context.leg_goal_pose_b[:, 3:7]
        reorient_pose = _ee_for_held_leg(
            unwrapped,
            active_arm,
            reorient_leg,
            leg_pos_in_finger,
            leg_quat_in_finger,
        )
        _run_phase(
            env,
            state,
            active_arm,
            "reorient",
            reorient_pose,
            CLOSE_GRIPPER,
            args_cli.phase_steps,
            context,
        )

        above_leg = context.leg_goal_pose_b.clone()
        above_leg[:, 2] += args_cli.overhead_clearance + args_cli.insert_clearance
        preinsert_leg = context.leg_goal_pose_b.clone()
        preinsert_leg[:, 2] += args_cli.insert_clearance
        seated_leg = context.leg_goal_pose_b.clone()
        seated_leg[:, 2] -= args_cli.insert_push_depth
        above_pose = _ee_for_held_leg(
            unwrapped,
            active_arm,
            above_leg,
            leg_pos_in_finger,
            leg_quat_in_finger,
        )
        preinsert_pose = _ee_for_held_leg(
            unwrapped,
            active_arm,
            preinsert_leg,
            leg_pos_in_finger,
            leg_quat_in_finger,
        )
        seated_pose = _ee_for_held_leg(
            unwrapped,
            active_arm,
            seated_leg,
            leg_pos_in_finger,
            leg_quat_in_finger,
        )
        _run_phase(
            env,
            state,
            active_arm,
            "move-above-insert",
            above_pose,
            CLOSE_GRIPPER,
            args_cli.phase_steps,
            context,
        )
        _run_phase(
            env,
            state,
            active_arm,
            "preinsert",
            preinsert_pose,
            CLOSE_GRIPPER,
            args_cli.phase_steps,
            context,
        )
        _run_phase(
            env,
            state,
            active_arm,
            "seat",
            seated_pose,
            CLOSE_GRIPPER,
            args_cli.phase_steps,
            context,
        )
        _run_phase(
            env,
            state,
            active_arm,
            "hold",
            seated_pose,
            CLOSE_GRIPPER,
            args_cli.settle_steps,
            context,
        )

        release_pose = _active_ee_pose(unwrapped, active_arm)
        _run_phase(
            env,
            state,
            active_arm,
            "release",
            release_pose,
            OPEN_GRIPPER,
            args_cli.close_steps,
            context,
        )
        retreat_center = _finger_center(unwrapped, active_arm).clone()
        retreat_center[:, 2] += max(0.04, args_cli.insert_clearance + 0.02)
        retreat_pose = _ee_for_finger_center(
            unwrapped,
            active_arm,
            retreat_center,
            release_pose[:, 3:7],
        )
        _run_phase(
            env,
            state,
            active_arm,
            "retreat",
            retreat_pose,
            OPEN_GRIPPER,
            args_cli.phase_steps,
            context,
        )

        return 0 if _write_final_metrics(context, unwrapped) else 1


def main() -> int:
    env_cfg = parse_env_cfg(
        TASK_NAME,
        device=args_cli.device,
        num_envs=1,
        use_fabric=not args_cli.disable_fabric,
    )
    env_cfg.scene.head_camera.width = CAMERA_WIDTH
    env_cfg.scene.head_camera.height = CAMERA_HEIGHT
    ensure_foundationpose_container(
        args_cli.foundationpose_container,
        args_cli.foundationpose_capture_root,
        args_cli.foundationpose_repo,
    )
    env = gym.make(TASK_NAME, cfg=env_cfg)
    try:
        return _run_assembly(env)
    finally:
        env.close()


if __name__ == "__main__":
    exit_code = 0
    try:
        exit_code = main()
    except Exception:
        exit_code = 1
        traceback.print_exc()
    finally:
        simulation_app.close()
    sys.exit(exit_code)
