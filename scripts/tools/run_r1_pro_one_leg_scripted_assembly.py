# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Scripted physical assembly demo for the R1 Pro FurnitureBench one_leg task.

The script drives the normal whole-body IK action interface and keeps the RL
environment API unchanged. It does not kinematically attach or snap leg4 after
reset: the leg is grasped, lifted, transported, inserted, rotated in place, and
released through normal simulation dynamics.

.. code-block:: bash

    python scripts/tools/run_r1_pro_one_leg_scripted_assembly.py --num_envs 1 --device cuda:0

"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time
import traceback
from collections.abc import Callable

from isaaclab.app import AppLauncher


TASK_NAME = "Assembly-R1Pro-OneLeg-WholeBodyIK-Direct-v0"
DEFAULT_SCREW_ROTATION = -math.pi


parser = argparse.ArgumentParser(description="Run scripted R1 Pro one_leg physical assembly.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments. Only 1 is supported.")
parser.add_argument(
    "--disable_fabric",
    action="store_true",
    default=False,
    help="Debug/compatibility option: disable Fabric and use USD I/O, which may desync GUI mesh updates.",
)
parser.add_argument("--phase_steps", type=int, default=120, help="Number of simulation steps for each scripted phase.")
parser.add_argument("--close_steps", type=int, default=160, help="Steps used to close or open the gripper.")
parser.add_argument("--settle_steps", type=int, default=30, help="Steps to pause at grasp or insertion poses.")
parser.add_argument("--lift_height", type=float, default=0.14, help="Lift-test height above the physical grasp pose.")
parser.add_argument(
    "--overhead_clearance",
    type=float,
    default=0.24,
    help="Vertical clearance used for high approach, lift, and transport waypoints.",
)
parser.add_argument(
    "--insert_clearance",
    type=float,
    default=0.04,
    help="Leg clearance above the final assembled pose before reaching the insertion target.",
)
parser.add_argument(
    "--insert_push_depth",
    type=float,
    default=0.0,
    help="Optional downward insertion push after reaching the pre-insert pose. Defaults to no extra push.",
)
parser.add_argument(
    "--screw_steps",
    type=int,
    default=120,
    help="Number of simulation steps used for each small-angle screw phase.",
)
parser.add_argument(
    "--screw_cycles",
    type=int,
    default=4,
    help="Number of continuous small-angle screw segments after insertion.",
)
parser.add_argument(
    "--screw_down_depth",
    type=float,
    default=0.0,
    help="Optional downward bias during the screw phase. Defaults to in-place rotation.",
)
parser.add_argument(
    "--screw_gripper_value",
    type=float,
    default=-1.0,
    help="Active gripper target used after insertion for continuous screw phases.",
)
parser.add_argument(
    "--screw_lift_abort_threshold",
    type=float,
    default=0.003,
    help="Abort post-insertion screw phases if the table top or leg4 z rises by more than this threshold.",
)
parser.add_argument(
    "--screw_min_follow_fraction",
    type=float,
    default=0.20,
    help="Abort remaining screw segments if leg4 rotates less than this fraction of one commanded segment.",
)
parser.add_argument(
    "--screw_grip_slip_tolerance",
    type=float,
    default=0.02,
    help="Abort remaining screw segments if leg4 position in the finger frame drifts by more than this many meters.",
)
parser.add_argument(
    "--screw_grip_ori_slip_tolerance",
    type=float,
    default=0.35,
    help="Abort if leg4 orientation in the finger frame drifts by more than this many radians.",
)
parser.add_argument(
    "--screw_rotation",
    type=float,
    default=DEFAULT_SCREW_ROTATION,
    help="Total root-frame Z screw rotation in radians, split evenly across screw cycles. Default is -pi.",
)
parser.add_argument(
    "--screw_regrasp_clearance",
    type=float,
    default=0.04,
    help="Vertical clearance used for the final retreat after the continuous screw path.",
)
parser.add_argument(
    "--finger_center_offset_z",
    type=float,
    default=0.0,
    help="Additional z bias after aligning the physical finger center with the leg origin.",
)
parser.add_argument(
    "--finger_table_clearance",
    type=float,
    default=0.02,
    help="Minimum clearance between the lower finger collision face and the tabletop.",
)
parser.add_argument(
    "--finger_collision_half_height",
    type=float,
    default=0.02,
    help="Half height of the gripper finger collision boxes used for tabletop clearance planning.",
)
parser.add_argument(
    "--gripper_orientation",
    choices=("top_down", "current", "assembled"),
    default="top_down",
    help=(
        "Use a top-down grasp frame by default; current keeps the initial active gripper orientation, "
        "and assembled keeps the old assembled-leg orientation for debugging."
    ),
)
parser.add_argument("--marker_scale", type=float, default=0.06, help="Scale of part and gripper frame markers.")
parser.add_argument("--disable_markers", action="store_true", help="Disable part and gripper frame visualization.")
parser.add_argument(
    "--print_interval",
    type=int,
    default=30,
    help="Print phase diagnostics every N scripted steps. Use 0 to disable.",
)
parser.add_argument(
    "--rate_interval",
    type=int,
    default=120,
    help="Print simulation rate diagnostics every N action steps. Use 0 to print only the final summary.",
)
parser.add_argument(
    "--record_camera",
    action="store_true",
    help="Record RGB frames from a scene camera to an MP4 video.",
)
parser.add_argument(
    "--camera_name",
    type=str,
    default="front_left_work_camera",
    help="Name of the scene camera to record when --record_camera is set.",
)
parser.add_argument(
    "--camera_video_path",
    type=str,
    default=None,
    help="Optional output MP4 path for --record_camera.",
)
parser.add_argument(
    "--camera_video_fps",
    type=float,
    default=0.0,
    help="Camera video FPS. Use 0 to infer it from the environment step rate.",
)
parser.add_argument(
    "--camera_frame_interval",
    type=int,
    default=1,
    help="Record one camera frame every N scripted action steps.",
)
parser.add_argument(
    "--fast_exit",
    action="store_true",
    help="Exit the process immediately after the scripted run, avoiding slow Kit shutdown.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

if args_cli.num_envs != 1:
    raise ValueError("The scripted one_leg assembly demo currently supports only --num_envs 1.")
if args_cli.phase_steps <= 0:
    raise ValueError("--phase_steps must be positive.")
if args_cli.close_steps <= 0:
    raise ValueError("--close_steps must be positive.")
if args_cli.settle_steps <= 0:
    raise ValueError("--settle_steps must be positive.")
if args_cli.lift_height <= 0.0:
    raise ValueError("--lift_height must be positive.")
if args_cli.overhead_clearance <= 0.0:
    raise ValueError("--overhead_clearance must be positive.")
if args_cli.insert_clearance < 0.0:
    raise ValueError("--insert_clearance must be non-negative.")
if args_cli.insert_push_depth < 0.0:
    raise ValueError("--insert_push_depth must be non-negative.")
if args_cli.screw_steps <= 0:
    raise ValueError("--screw_steps must be positive.")
if args_cli.screw_cycles <= 0:
    raise ValueError("--screw_cycles must be positive.")
if args_cli.screw_down_depth < 0.0:
    raise ValueError("--screw_down_depth must be non-negative.")
if not -1.0 <= args_cli.screw_gripper_value <= 1.0:
    raise ValueError("--screw_gripper_value must be in [-1, 1].")
if args_cli.screw_lift_abort_threshold < 0.0:
    raise ValueError("--screw_lift_abort_threshold must be non-negative.")
if not 0.0 <= args_cli.screw_min_follow_fraction <= 1.0:
    raise ValueError("--screw_min_follow_fraction must be in [0, 1].")
if args_cli.screw_grip_slip_tolerance < 0.0:
    raise ValueError("--screw_grip_slip_tolerance must be non-negative.")
if args_cli.screw_grip_ori_slip_tolerance < 0.0:
    raise ValueError("--screw_grip_ori_slip_tolerance must be non-negative.")
if args_cli.screw_regrasp_clearance < 0.0:
    raise ValueError("--screw_regrasp_clearance must be non-negative.")
if args_cli.finger_table_clearance < 0.0:
    raise ValueError("--finger_table_clearance must be non-negative.")
if args_cli.finger_collision_half_height <= 0.0:
    raise ValueError("--finger_collision_half_height must be positive.")
if args_cli.marker_scale <= 0.0:
    raise ValueError("--marker_scale must be positive.")
if args_cli.print_interval < 0:
    raise ValueError("--print_interval must be non-negative.")
if args_cli.rate_interval < 0:
    raise ValueError("--rate_interval must be non-negative.")
if args_cli.camera_video_fps < 0.0:
    raise ValueError("--camera_video_fps must be non-negative.")
if args_cli.camera_frame_interval <= 0:
    raise ValueError("--camera_frame_interval must be positive.")
if args_cli.record_camera:
    args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch

import isaaclab_tasks  # noqa: F401
from isaaclab.markers import VisualizationMarkers
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.utils.math import (
    combine_frame_transforms,
    quat_apply,
    quat_apply_inverse,
    subtract_frame_transforms,
)
from isaaclab_tasks.utils import parse_env_cfg

import assembly_benchmark.tasks  # noqa: F401
from assembly_benchmark.assets.furniture.lab_table import LAB_TABLE_SURFACE_Z


OPEN_GRIPPER = 1.0
CLOSE_GRIPPER = -1.0
LEG4_TARGET_POS_TOP = (-0.05625, 0.046875, -0.05625)
IDENTITY_QUAT = (1.0, 0.0, 0.0, 0.0)
TOP_DOWN_GRIPPER_QUAT = (1.0, 0.0, 0.0, 0.0)
_MARKERS: tuple[
    VisualizationMarkers,
    VisualizationMarkers,
    VisualizationMarkers,
    VisualizationMarkers,
    VisualizationMarkers,
] | None = None
_PLANNED_GRASP_POSE_B: torch.Tensor | None = None
_PLANNED_INSERT_POSE_B: torch.Tensor | None = None
_PLANNED_SCREW_POSE_B: torch.Tensor | None = None
_LIFT_GUARD_ABORTED = False
_SIM_RATE_STATS: "SimRateStats | None" = None
_CAMERA_VIDEO_RECORDER: "CameraVideoRecorder | None" = None


class SimRateStats:
    """Tracks wall-clock simulation throughput without changing control timing."""

    def __init__(self, sim_dt: float, decimation: int, render_interval: int):
        now = time.perf_counter()
        self.sim_dt = sim_dt
        self.decimation = decimation
        self.render_interval = render_interval
        self.start_time = now
        self.last_report_time = now
        self.action_steps = 0
        self.physics_steps = 0
        self.render_frames = 0
        self.last_action_steps = 0
        self.last_physics_steps = 0
        self.last_render_frames = 0

    def record_action_step(self, physics_steps: int, render_frames: int) -> None:
        self.action_steps += 1
        self.physics_steps += physics_steps
        self.render_frames += render_frames

    def _metrics(
        self,
        action_steps: int,
        physics_steps: int,
        render_frames: int,
        wall_elapsed_s: float,
    ) -> dict[str, float]:
        safe_elapsed = max(wall_elapsed_s, 1.0e-9)
        sim_time_s = physics_steps * self.sim_dt
        physics_fps = physics_steps / safe_elapsed
        action_fps = action_steps / safe_elapsed
        render_fps = render_frames / safe_elapsed
        return {
            "wall_elapsed_s": wall_elapsed_s,
            "sim_time_s": sim_time_s,
            "physics_fps": physics_fps,
            "action_fps": action_fps,
            "render_fps": render_fps,
            "wall_ms_per_physics_step": 1000.0 / physics_fps if physics_steps > 0 else 0.0,
            "wall_ms_per_action_step": 1000.0 / action_fps if action_steps > 0 else 0.0,
            "real_time_factor": sim_time_s / safe_elapsed,
        }

    def _format_common(self, metrics: dict[str, float]) -> str:
        return (
            f"physics_fps={metrics['physics_fps']:.1f} "
            f"action_fps={metrics['action_fps']:.1f} "
            f"render_fps={metrics['render_fps']:.1f} "
            f"wall_ms_per_physics_step={metrics['wall_ms_per_physics_step']:.3f} "
            f"wall_ms_per_action_step={metrics['wall_ms_per_action_step']:.3f} "
            f"real_time_factor={metrics['real_time_factor']:.3f} "
            f"sim_dt_s={self.sim_dt:.6f} "
            f"decimation={self.decimation}"
        )

    def print_window(self, phase: str) -> None:
        now = time.perf_counter()
        action_steps = self.action_steps - self.last_action_steps
        physics_steps = self.physics_steps - self.last_physics_steps
        render_frames = self.render_frames - self.last_render_frames
        wall_elapsed_s = now - self.last_report_time
        metrics = self._metrics(action_steps, physics_steps, render_frames, wall_elapsed_s)
        print(
            "[INFO]: sim_rate "
            f"phase={phase} "
            f"action_steps={self.action_steps} "
            f"physics_steps={self.physics_steps} "
            f"render_frames={self.render_frames} "
            f"{self._format_common(metrics)}"
        )
        self.last_report_time = now
        self.last_action_steps = self.action_steps
        self.last_physics_steps = self.physics_steps
        self.last_render_frames = self.render_frames

    def print_summary(self) -> None:
        now = time.perf_counter()
        wall_elapsed_s = now - self.start_time
        metrics = self._metrics(
            self.action_steps,
            self.physics_steps,
            self.render_frames,
            wall_elapsed_s,
        )
        print(
            "[INFO]: sim_rate_summary "
            f"action_steps={self.action_steps} "
            f"physics_steps={self.physics_steps} "
            f"render_frames={self.render_frames} "
            f"wall_elapsed_s={metrics['wall_elapsed_s']:.3f} "
            f"sim_time_s={metrics['sim_time_s']:.3f} "
            f"{self._format_common(metrics)} "
            f"render_interval={self.render_interval}"
        )


def _default_camera_video_path() -> str:
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    filename = f"{TASK_NAME}_{timestamp}.mp4"
    return os.path.abspath(os.path.join("logs", "scripted_assembly", "camera", filename))


def _infer_camera_video_fps(unwrapped) -> float:
    if args_cli.camera_video_fps > 0.0:
        return args_cli.camera_video_fps
    step_dt = getattr(
        unwrapped,
        "step_dt",
        float(unwrapped.cfg.sim.dt) * float(unwrapped.cfg.decimation),
    )
    return 1.0 / max(float(step_dt) * float(args_cli.camera_frame_interval), 1.0e-9)


class CameraVideoRecorder:
    """Writes RGB frames from an Isaac Lab scene camera to an MP4 file."""

    def __init__(self, camera_name: str, video_path: str, fps: float, frame_interval: int):
        self.camera_name = camera_name
        self.video_path = os.path.abspath(video_path)
        self.fps = fps
        self.frame_interval = frame_interval
        self.action_steps = 0
        self.frames_written = 0
        self._imageio = None
        self._writer = None

    def validate(self, unwrapped) -> None:
        self._camera(unwrapped)

    def record_step(self, unwrapped) -> None:
        self.action_steps += 1
        if (self.action_steps - 1) % self.frame_interval != 0:
            return
        frame = self._rgb_frame(unwrapped)
        self._ensure_writer()
        self._writer.append_data(frame)
        self.frames_written += 1

    def close(self) -> None:
        if self._writer is not None:
            self._writer.close()
            self._writer = None
        if self.frames_written > 0:
            print(
                "[INFO]: camera video saved "
                f"path={self.video_path} frames={self.frames_written} fps={self.fps:.2f}"
            )

    def _ensure_writer(self) -> None:
        if self._writer is not None:
            return
        try:
            import imageio.v2 as imageio
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "Camera video recording requires imageio and imageio-ffmpeg. "
                "Install them in the Isaac Lab environment, then rerun with --record_camera."
            ) from exc

        os.makedirs(os.path.dirname(self.video_path), exist_ok=True)
        self._imageio = imageio
        try:
            self._writer = self._imageio.get_writer(self.video_path, fps=self.fps)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to open camera video writer at {self.video_path}. "
                "Ensure imageio-ffmpeg is installed and the output path is writable."
            ) from exc

    def _camera(self, unwrapped):
        try:
            return unwrapped.scene[self.camera_name]
        except KeyError as exc:
            scene_keys = getattr(unwrapped.scene, "keys", lambda: [])()
            available = ", ".join(str(key) for key in scene_keys)
            raise RuntimeError(
                f"Scene camera '{self.camera_name}' was not found. Available scene entities: {available}"
            ) from exc

    def _rgb_frame(self, unwrapped):
        camera = self._camera(unwrapped)
        if "rgb" not in camera.data.output:
            raise RuntimeError(f"Scene camera '{self.camera_name}' does not provide an 'rgb' output.")

        frame = camera.data.output["rgb"]
        if frame.ndim == 4:
            frame = frame[0]
        if frame.ndim != 3 or frame.shape[-1] < 3:
            raise RuntimeError(
                f"Expected camera '{self.camera_name}' rgb output with shape HxWx3 or NxHxWx3, got {tuple(frame.shape)}."
            )

        frame = frame[..., :3].detach()
        if frame.dtype != torch.uint8:
            if frame.is_floating_point():
                max_value = float(frame.max().item()) if frame.numel() > 0 else 1.0
                if max_value <= 1.0:
                    frame = frame.clamp(0.0, 1.0) * 255.0
                else:
                    frame = frame.clamp(0.0, 255.0)
            else:
                frame = frame.clamp(0, 255)
            frame = frame.to(torch.uint8)
        return frame.cpu().contiguous().numpy()


def _maybe_print_sim_rate(phase: str, global_step: int) -> None:
    if args_cli.rate_interval <= 0 or global_step <= 0:
        return
    if global_step % args_cli.rate_interval != 0:
        return
    if _SIM_RATE_STATS is None:
        return
    _SIM_RATE_STATS.print_window(phase)


def _print_sim_rate_summary() -> None:
    if _SIM_RATE_STATS is None:
        return
    _SIM_RATE_STATS.print_summary()


def _make_action(
    left_pose: torch.Tensor, left_grip: float, right_pose: torch.Tensor, right_grip: float
) -> torch.Tensor:
    """Build one bimanual whole-body IK action."""
    actions = torch.zeros((1, 16), dtype=torch.float32, device=left_pose.device)
    actions[:, 0:7] = left_pose
    actions[:, 7] = left_grip
    actions[:, 8:15] = right_pose
    actions[:, 15] = right_grip
    return actions


def _step_without_auto_reset(env, actions: torch.Tensor):
    """Step the DirectRLEnv physics path without applying automatic reset."""
    unwrapped = env.unwrapped
    actions = actions.to(unwrapped.device)
    unwrapped._pre_physics_step(actions)
    is_rendering = unwrapped.sim.has_gui() or unwrapped.sim.has_rtx_sensors()
    render_frames = 0
    physics_steps = int(unwrapped.cfg.decimation)

    for _ in range(physics_steps):
        unwrapped._sim_step_counter += 1
        unwrapped._apply_action()
        unwrapped.scene.write_data_to_sim()
        unwrapped.sim.step(render=False)
        if unwrapped._sim_step_counter % unwrapped.cfg.sim.render_interval == 0 and is_rendering:
            unwrapped.sim.render()
            render_frames += 1
        unwrapped.scene.update(dt=unwrapped.physics_dt)

    unwrapped.obs_buf = unwrapped._get_observations()
    reward = unwrapped._get_rewards()
    terminated, truncated = unwrapped._get_dones()
    if _SIM_RATE_STATS is not None:
        _SIM_RATE_STATS.record_action_step(physics_steps=physics_steps, render_frames=render_frames)
    if _CAMERA_VIDEO_RECORDER is not None:
        _CAMERA_VIDEO_RECORDER.record_step(unwrapped)
    return unwrapped.obs_buf, reward, terminated, truncated, unwrapped.extras


def _pose_from_pos_quat(pos: torch.Tensor, quat: torch.Tensor) -> torch.Tensor:
    pose = torch.zeros((1, 7), dtype=torch.float32, device=pos.device)
    pose[:, :3] = pos
    pose[:, 3:7] = quat
    return pose


def _quat_tensor(device: torch.device, values: tuple[float, float, float, float]) -> torch.Tensor:
    return torch.tensor((values,), dtype=torch.float32, device=device)


def _normalize_quat(quat: torch.Tensor) -> torch.Tensor:
    return quat / torch.linalg.norm(quat, dim=-1, keepdim=True).clamp_min(1.0e-8)


def _quat_mul(first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
    first_w, first_x, first_y, first_z = first.unbind(dim=-1)
    second_w, second_x, second_y, second_z = second.unbind(dim=-1)
    return torch.stack(
        (
            first_w * second_w - first_x * second_x - first_y * second_y - first_z * second_z,
            first_w * second_x + first_x * second_w + first_y * second_z - first_z * second_y,
            first_w * second_y - first_x * second_z + first_y * second_w + first_z * second_x,
            first_w * second_z + first_x * second_y - first_y * second_x + first_z * second_w,
        ),
        dim=-1,
    )


def _axis_angle_quat(device: torch.device, axis: tuple[float, float, float], angle: float) -> torch.Tensor:
    axis_tensor = torch.tensor((axis,), dtype=torch.float32, device=device)
    axis_tensor = axis_tensor / torch.linalg.norm(axis_tensor, dim=-1, keepdim=True).clamp_min(1.0e-8)
    half_angle = 0.5 * angle
    quat = torch.zeros((1, 4), dtype=torch.float32, device=device)
    quat[:, 0] = math.cos(half_angle)
    quat[:, 1:4] = axis_tensor * math.sin(half_angle)
    return _normalize_quat(quat)


def _quat_inv(quat: torch.Tensor) -> torch.Tensor:
    conjugate = quat.clone()
    conjugate[..., 1:] *= -1.0
    norm_sq = torch.sum(quat * quat, dim=-1, keepdim=True).clamp_min(1.0e-8)
    return conjugate / norm_sq


def _nlerp_quat(start: torch.Tensor, target: torch.Tensor, alpha: float) -> torch.Tensor:
    start = _normalize_quat(start)
    target = _normalize_quat(target)
    target = torch.where(torch.sum(start * target, dim=-1, keepdim=True) < 0.0, -target, target)
    return _normalize_quat((1.0 - alpha) * start + alpha * target)


def _quat_angle_error(current: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    current = _normalize_quat(current)
    target = _normalize_quat(target)
    dot = torch.abs(torch.sum(current * target, dim=-1)).clamp(max=1.0)
    return 2.0 * torch.acos(dot)


def _env_ids(unwrapped) -> torch.Tensor:
    return torch.tensor([0], dtype=torch.long, device=unwrapped.device)


def _world_pose_to_env_pose(unwrapped, pose_w: torch.Tensor) -> torch.Tensor:
    env_pose = pose_w.clone()
    env_pose[:, :3] -= unwrapped.scene.env_origins[_env_ids(unwrapped)]
    return env_pose


def _leg_pose_env(unwrapped) -> torch.Tensor:
    return unwrapped._object_pose_in_env_frame(unwrapped.square_table_leg4)[0:1]


def _active_ee_pose(unwrapped, active_arm: str) -> torch.Tensor:
    left_pose, right_pose = unwrapped._get_ee_poses_in_root_frame()
    return left_pose.clone() if active_arm == "left" else right_pose.clone()


def _body_position_in_root_frame(unwrapped, body_name: str) -> torch.Tensor:
    body_idx = unwrapped.robot.find_bodies(body_name)[0][0]
    body_pose_w = unwrapped.robot.data.body_pose_w[:, body_idx]
    root_pose_w = unwrapped.robot.data.root_pose_w
    body_pos_b, _ = subtract_frame_transforms(
        root_pose_w[:, :3],
        root_pose_w[:, 3:7],
        body_pose_w[:, :3],
        body_pose_w[:, 3:7],
    )
    return body_pos_b


def _finger_center_in_root_frame(unwrapped, active_arm: str) -> torch.Tensor:
    finger_1_pos = _body_position_in_root_frame(unwrapped, f"{active_arm}_gripper_finger_link1")
    finger_2_pos = _body_position_in_root_frame(unwrapped, f"{active_arm}_gripper_finger_link2")
    return 0.5 * (finger_1_pos + finger_2_pos)


def _finger_center_offset_from_ee_for_quat(
    unwrapped, active_arm: str, target_quat_b: torch.Tensor
) -> torch.Tensor:
    ee_pose = _active_ee_pose(unwrapped, active_arm)
    offset_b = _finger_center_in_root_frame(unwrapped, active_arm) - ee_pose[:, :3]
    offset_ee = quat_apply_inverse(ee_pose[:, 3:7], offset_b)
    return quat_apply(target_quat_b, offset_ee)


def _table_surface_z(unwrapped) -> float:
    return float(getattr(unwrapped.cfg, "table_surface_z", LAB_TABLE_SURFACE_Z))


def _table_safe_finger_center_z(unwrapped, desired_z: torch.Tensor) -> torch.Tensor:
    min_center_z = (
        _table_surface_z(unwrapped)
        + args_cli.finger_collision_half_height
        + args_cli.finger_table_clearance
    )
    min_center_z_tensor = torch.full_like(desired_z, min_center_z)
    return torch.maximum(desired_z, min_center_z_tensor)


def _make_markers() -> tuple[
    VisualizationMarkers,
    VisualizationMarkers,
    VisualizationMarkers,
    VisualizationMarkers,
    VisualizationMarkers,
]:
    part_marker_cfg = FRAME_MARKER_CFG.copy()
    part_marker_cfg.markers["frame"].scale = (
        args_cli.marker_scale,
        args_cli.marker_scale,
        args_cli.marker_scale,
    )

    gripper_marker_cfg = FRAME_MARKER_CFG.copy()
    gripper_scale = args_cli.marker_scale * 1.25
    gripper_marker_cfg.markers["frame"].scale = (gripper_scale, gripper_scale, gripper_scale)

    planned_marker_cfg = FRAME_MARKER_CFG.copy()
    planned_scale = args_cli.marker_scale * 1.6
    planned_marker_cfg.markers["frame"].scale = (planned_scale, planned_scale, planned_scale)

    planned_grasp_marker = VisualizationMarkers(
        planned_marker_cfg.replace(prim_path="/Visuals/one_leg_planned_grasp_frame")
    )
    planned_grasp_marker.set_visibility(False)
    planned_insert_marker = VisualizationMarkers(
        planned_marker_cfg.replace(prim_path="/Visuals/one_leg_planned_insert_frame")
    )
    planned_insert_marker.set_visibility(False)
    planned_screw_marker = VisualizationMarkers(
        planned_marker_cfg.replace(prim_path="/Visuals/one_leg_planned_screw_frame")
    )
    planned_screw_marker.set_visibility(False)

    return (
        VisualizationMarkers(part_marker_cfg.replace(prim_path="/Visuals/one_leg_part_frames")),
        VisualizationMarkers(gripper_marker_cfg.replace(prim_path="/Visuals/one_leg_gripper_frames")),
        planned_grasp_marker,
        planned_insert_marker,
        planned_screw_marker,
    )


def _part_frame_poses_w(unwrapped) -> torch.Tensor:
    part_poses = (
        unwrapped.square_table_top.data.root_pose_w[0:1],
        unwrapped.square_table_leg1.data.root_pose_w[0:1],
        unwrapped.square_table_leg2.data.root_pose_w[0:1],
        unwrapped.square_table_leg3.data.root_pose_w[0:1],
        unwrapped.square_table_leg4.data.root_pose_w[0:1],
    )
    return torch.cat(part_poses, dim=0)


def _gripper_frame_poses_w(unwrapped) -> torch.Tensor:
    left_gripper_pose_w = unwrapped.robot.data.body_pose_w[:, unwrapped.left_ee_body_idx]
    right_gripper_pose_w = unwrapped.robot.data.body_pose_w[:, unwrapped.right_ee_body_idx]
    return torch.cat((left_gripper_pose_w, right_gripper_pose_w), dim=0)


def _root_pose_to_world_pose(unwrapped, pose_b: torch.Tensor | None) -> torch.Tensor | None:
    if pose_b is None:
        return None
    root_pose_w = unwrapped.robot.data.root_pose_w
    pos_w, quat_w = combine_frame_transforms(
        root_pose_w[:, :3],
        root_pose_w[:, 3:7],
        pose_b[:, :3],
        pose_b[:, 3:7],
    )
    return torch.cat((pos_w, quat_w), dim=-1)


def _set_planned_grasp_pose(pose_b: torch.Tensor | None) -> None:
    global _PLANNED_GRASP_POSE_B

    _PLANNED_GRASP_POSE_B = None if pose_b is None else pose_b.clone()


def _set_planned_insert_pose(pose_b: torch.Tensor | None) -> None:
    global _PLANNED_INSERT_POSE_B

    _PLANNED_INSERT_POSE_B = None if pose_b is None else pose_b.clone()


def _set_planned_screw_pose(pose_b: torch.Tensor | None) -> None:
    global _PLANNED_SCREW_POSE_B

    _PLANNED_SCREW_POSE_B = None if pose_b is None else pose_b.clone()


def _visualize_markers(unwrapped) -> None:
    if _MARKERS is None:
        return

    part_marker, gripper_marker, planned_grasp_marker, planned_insert_marker, planned_screw_marker = _MARKERS
    part_poses_w = _part_frame_poses_w(unwrapped)
    gripper_poses_w = _gripper_frame_poses_w(unwrapped)
    part_marker.visualize(part_poses_w[:, :3], part_poses_w[:, 3:7])
    gripper_marker.visualize(gripper_poses_w[:, :3], gripper_poses_w[:, 3:7])
    planned_grasp_pose_w = _root_pose_to_world_pose(unwrapped, _PLANNED_GRASP_POSE_B)
    planned_insert_pose_w = _root_pose_to_world_pose(unwrapped, _PLANNED_INSERT_POSE_B)
    planned_screw_pose_w = _root_pose_to_world_pose(unwrapped, _PLANNED_SCREW_POSE_B)
    if planned_grasp_pose_w is None:
        planned_grasp_marker.set_visibility(False)
    else:
        planned_grasp_marker.set_visibility(True)
        planned_grasp_marker.visualize(planned_grasp_pose_w[:, :3], planned_grasp_pose_w[:, 3:7])
    if planned_insert_pose_w is None:
        planned_insert_marker.set_visibility(False)
    else:
        planned_insert_marker.set_visibility(True)
        planned_insert_marker.visualize(planned_insert_pose_w[:, :3], planned_insert_pose_w[:, 3:7])
    if planned_screw_pose_w is None:
        planned_screw_marker.set_visibility(False)
    else:
        planned_screw_marker.set_visibility(True)
        planned_screw_marker.visualize(planned_screw_pose_w[:, :3], planned_screw_pose_w[:, 3:7])


def _ee_pose_for_finger_center(
    unwrapped, active_arm: str, finger_center_pos: torch.Tensor, target_quat_b: torch.Tensor
) -> torch.Tensor:
    safe_center_pos = finger_center_pos.clone()
    safe_center_pos[:, 2] = _table_safe_finger_center_z(unwrapped, safe_center_pos[:, 2])
    finger_center_offset = _finger_center_offset_from_ee_for_quat(unwrapped, active_arm, target_quat_b)
    return _pose_from_pos_quat(safe_center_pos - finger_center_offset, target_quat_b)


def _ee_pose_for_fixed_finger_center(
    unwrapped, active_arm: str, finger_center_pos: torch.Tensor, target_quat_b: torch.Tensor
) -> torch.Tensor:
    min_center_z = (
        _table_surface_z(unwrapped)
        + args_cli.finger_collision_half_height
        + args_cli.finger_table_clearance
    )
    current_z = float(finger_center_pos[0, 2].item())
    if current_z < min_center_z:
        print(
            "[WARN]: fixed screw finger center is below tabletop safety height; "
            f"current_z={current_z:.4f} safe_z={min_center_z:.4f}. Keeping current z for in-place screw."
        )
    finger_center_offset = _finger_center_offset_from_ee_for_quat(unwrapped, active_arm, target_quat_b)
    return _pose_from_pos_quat(finger_center_pos - finger_center_offset, target_quat_b)


def _target_finger_center_for_ee_pose(unwrapped, active_arm: str, target_pose_b: torch.Tensor) -> torch.Tensor:
    finger_center_offset = _finger_center_offset_from_ee_for_quat(
        unwrapped,
        active_arm,
        target_pose_b[:, 3:7],
    )
    return target_pose_b[:, :3] + finger_center_offset


def _finger_center_pose_in_root_frame(unwrapped, active_arm: str) -> torch.Tensor:
    pose = _active_ee_pose(unwrapped, active_arm)
    pose[:, :3] = _finger_center_in_root_frame(unwrapped, active_arm)
    return pose


def _held_leg_relative_to_finger(
    unwrapped, active_arm: str, leg_pose_b: torch.Tensor | None = None
) -> tuple[torch.Tensor, torch.Tensor]:
    finger_pose_b = _finger_center_pose_in_root_frame(unwrapped, active_arm)
    leg_pose_b = _leg_pose_env(unwrapped) if leg_pose_b is None else leg_pose_b
    return subtract_frame_transforms(
        finger_pose_b[:, :3],
        finger_pose_b[:, 3:7],
        leg_pose_b[:, :3],
        leg_pose_b[:, 3:7],
    )


def _ee_pose_for_held_leg_pose(
    unwrapped,
    active_arm: str,
    desired_leg_pose_b: torch.Tensor,
    leg_pos_in_finger: torch.Tensor,
    leg_quat_in_finger: torch.Tensor,
) -> torch.Tensor:
    desired_finger_quat = _normalize_quat(_quat_mul(desired_leg_pose_b[:, 3:7], _quat_inv(leg_quat_in_finger)))
    desired_finger_pos = desired_leg_pose_b[:, :3] - quat_apply(desired_finger_quat, leg_pos_in_finger)
    return _ee_pose_for_finger_center(unwrapped, active_arm, desired_finger_pos, desired_finger_quat)


def _ee_pose_for_held_leg_pose_without_lift(
    unwrapped,
    active_arm: str,
    desired_leg_pose_b: torch.Tensor,
    leg_pos_in_finger: torch.Tensor,
    leg_quat_in_finger: torch.Tensor,
) -> torch.Tensor:
    desired_finger_quat = _normalize_quat(_quat_mul(desired_leg_pose_b[:, 3:7], _quat_inv(leg_quat_in_finger)))
    desired_finger_pos = desired_leg_pose_b[:, :3] - quat_apply(desired_finger_quat, leg_pos_in_finger)
    return _ee_pose_for_fixed_finger_center(unwrapped, active_arm, desired_finger_pos, desired_finger_quat)


def _ee_pose_for_current_leg_alignment(
    unwrapped,
    active_arm: str,
    leg_pos_in_finger: torch.Tensor,
    leg_quat_in_finger: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    current_leg_pose = _leg_pose_env(unwrapped)
    aligned_finger_quat = _normalize_quat(_quat_mul(current_leg_pose[:, 3:7], _quat_inv(leg_quat_in_finger)))
    aligned_finger_center = current_leg_pose[:, :3] - quat_apply(aligned_finger_quat, leg_pos_in_finger)
    aligned_ee_pose = _ee_pose_for_fixed_finger_center(
        unwrapped,
        active_arm,
        aligned_finger_center,
        aligned_finger_quat,
    )
    return aligned_ee_pose, aligned_finger_center


def _grasp_quat(unwrapped, active_pose: torch.Tensor, final_leg_pose_env: torch.Tensor) -> torch.Tensor:
    if args_cli.gripper_orientation == "top_down":
        return _quat_tensor(unwrapped.device, TOP_DOWN_GRIPPER_QUAT)
    if args_cli.gripper_orientation == "assembled":
        return final_leg_pose_env[:, 3:7].clone()
    return active_pose[:, 3:7].clone()


def _final_leg_pose_env(unwrapped) -> torch.Tensor:
    top_pose_w = unwrapped.square_table_top.data.root_pose_w[0:1]
    target_pos_top = torch.tensor((LEG4_TARGET_POS_TOP,), dtype=torch.float32, device=unwrapped.device)
    target_quat_top = torch.tensor((IDENTITY_QUAT,), dtype=torch.float32, device=unwrapped.device)
    target_pos_w, target_quat_w = combine_frame_transforms(
        top_pose_w[:, :3],
        top_pose_w[:, 3:7],
        target_pos_top,
        target_quat_top,
    )
    return _world_pose_to_env_pose(unwrapped, torch.cat((target_pos_w, target_quat_w), dim=-1))


def _format_pose(pose: torch.Tensor) -> str:
    pose_cpu = pose[0].detach().cpu()
    pos = ", ".join(f"{value:.4f}" for value in pose_cpu[:3])
    quat = ", ".join(f"{value:.4f}" for value in pose_cpu[3:7])
    return f"pos=[{pos}] quat_wxyz=[{quat}]"


def _format_vec(vec: torch.Tensor) -> str:
    vec_cpu = vec[0].detach().cpu()
    values = ", ".join(f"{value:.4f}" for value in vec_cpu)
    return f"[{values}]"


def _final_relative_pose_diagnostics(unwrapped) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    rel_pos, rel_quat = unwrapped._assembled_relative_pose()
    target_pos = torch.tensor((LEG4_TARGET_POS_TOP,), dtype=torch.float32, device=unwrapped.device)
    pos_error = rel_pos - target_pos
    return rel_pos, rel_quat, pos_error


def _run_phase(
    env,
    phase: str,
    active_arm: str,
    target_pose: torch.Tensor,
    left_pose: torch.Tensor,
    right_pose: torch.Tensor,
    left_grip: float,
    right_grip: float,
    target_left_grip: float,
    target_right_grip: float,
    global_step: int,
    final_leg_pose_env: torch.Tensor,
    steps: int | None = None,
    target_pose_fn: Callable[[], torch.Tensor] | None = None,
    stop_on_success: bool = False,
    lift_guard: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, float, float, int]:
    """Run one interpolated scripted phase."""
    global _LIFT_GUARD_ABORTED

    _LIFT_GUARD_ABORTED = False
    unwrapped = env.unwrapped
    start_pose = left_pose.clone() if active_arm == "left" else right_pose.clone()
    start_left_grip = left_grip
    start_right_grip = right_grip
    phase_steps = args_cli.phase_steps if steps is None else steps
    lift_guard_top_z = float(unwrapped.square_table_top.data.root_pose_w[0, 2].item()) if lift_guard else 0.0
    lift_guard_leg_z = float(unwrapped.square_table_leg4.data.root_pose_w[0, 2].item()) if lift_guard else 0.0

    for step_idx in range(phase_steps):
        if not simulation_app.is_running():
            break

        resolved_target_pose = target_pose_fn() if target_pose_fn is not None else target_pose
        alpha = float(step_idx + 1) / float(phase_steps)
        command_pose = start_pose.clone()
        command_pose[:, :3] = (1.0 - alpha) * start_pose[:, :3] + alpha * resolved_target_pose[:, :3]
        command_pose[:, 3:7] = _nlerp_quat(start_pose[:, 3:7], resolved_target_pose[:, 3:7], alpha)
        if active_arm == "left":
            left_pose = command_pose
        else:
            right_pose = command_pose

        left_grip = (1.0 - alpha) * start_left_grip + alpha * target_left_grip
        right_grip = (1.0 - alpha) * start_right_grip + alpha * target_right_grip
        actions = _make_action(left_pose, left_grip, right_pose, right_grip)
        _step_without_auto_reset(env, actions)
        _visualize_markers(unwrapped)
        success = bool(unwrapped._success()[0].item())
        top_lift_delta = 0.0
        leg_lift_delta = 0.0
        if lift_guard:
            top_lift_delta = float(unwrapped.square_table_top.data.root_pose_w[0, 2].item()) - lift_guard_top_z
            leg_lift_delta = float(unwrapped.square_table_leg4.data.root_pose_w[0, 2].item()) - lift_guard_leg_z

        if args_cli.print_interval > 0 and global_step % args_cli.print_interval == 0:
            ee_pose = _active_ee_pose(unwrapped, active_arm)
            ee_error = torch.linalg.norm(ee_pose[:, :3] - resolved_target_pose[:, :3], dim=-1).mean()
            ee_ori_error = _quat_angle_error(ee_pose[:, 3:7], resolved_target_pose[:, 3:7]).mean()
            finger_center = _finger_center_in_root_frame(unwrapped, active_arm)
            target_finger_center = _target_finger_center_for_ee_pose(unwrapped, active_arm, resolved_target_pose)
            finger_center_delta_z = (finger_center[:, 2] - target_finger_center[:, 2]).mean()
            table_clearance = (
                finger_center[:, 2]
                - args_cli.finger_collision_half_height
                - _table_surface_z(unwrapped)
            ).mean()
            leg_error_xyz = _leg_pose_env(unwrapped)[:, :3] - final_leg_pose_env[:, :3]
            leg_error = torch.linalg.norm(leg_error_xyz, dim=-1).mean()
            _, rel_quat, _ = _final_relative_pose_diagnostics(unwrapped)
            leg_ori_error = _quat_angle_error(
                rel_quat,
                _quat_tensor(unwrapped.device, IDENTITY_QUAT),
            ).mean()
            leg_pos_in_finger, leg_quat_in_finger = _held_leg_relative_to_finger(unwrapped, active_arm)
            finger_leg_dist = torch.linalg.norm(_leg_pose_env(unwrapped)[:, :3] - finger_center, dim=-1).mean()
            print(
                f"[INFO]: phase={phase} arm={active_arm} "
                f"ee_error={float(ee_error):.4f} m "
                f"ee_ori_error={float(ee_ori_error):.4f} rad "
                f"finger_leg_dist={float(finger_leg_dist):.4f} m "
                f"finger_center_delta_z={float(finger_center_delta_z):.4f} m "
                f"finger_table_clearance={float(table_clearance):.4f} m "
                f"leg_target_error={float(leg_error):.4f} m "
                f"leg_ori_error={float(leg_ori_error):.4f} rad "
                f"leg_error_xyz={_format_vec(leg_error_xyz)} "
                f"leg_in_finger_pos={_format_vec(leg_pos_in_finger)} "
                f"leg_in_finger_quat={_format_vec(leg_quat_in_finger)} "
                f"target_b={_format_pose(resolved_target_pose)} "
                f"leg_b={_format_pose(_leg_pose_env(unwrapped))} "
                f"top_lift_delta={top_lift_delta:.4f} m "
                f"leg_lift_delta={leg_lift_delta:.4f} m "
                f"success={success}"
            )

        global_step += 1
        _maybe_print_sim_rate(phase, global_step)
        if lift_guard and max(top_lift_delta, leg_lift_delta) > args_cli.screw_lift_abort_threshold:
            _LIFT_GUARD_ABORTED = True
            print(
                "[WARN]: lift_guard_abort "
                f"phase={phase} step={step_idx + 1}/{phase_steps} "
                f"top_lift_delta={top_lift_delta:.4f} m "
                f"leg_lift_delta={leg_lift_delta:.4f} m "
                f"threshold={args_cli.screw_lift_abort_threshold:.4f} m; opening active gripper."
            )
            if active_arm == "left":
                left_grip = OPEN_GRIPPER
            else:
                right_grip = OPEN_GRIPPER
            actions = _make_action(left_pose, left_grip, right_pose, right_grip)
            _step_without_auto_reset(env, actions)
            _visualize_markers(unwrapped)
            global_step += 1
            _maybe_print_sim_rate(phase, global_step)
            break
        if stop_on_success and success:
            rel_pos, rel_quat, pos_error = _final_relative_pose_diagnostics(unwrapped)
            rel_ori_error = _quat_angle_error(rel_quat, _quat_tensor(unwrapped.device, IDENTITY_QUAT))
            print(
                "[INFO]: stop_on_success=success "
                f"phase={phase} step={step_idx + 1}/{phase_steps} "
                f"rel_pos={_format_vec(rel_pos)} "
                f"rel_pos_error={_format_vec(pos_error)} "
                f"rel_ori_error={float(rel_ori_error[0]):.4f} rad"
            )
            break

    return left_pose, right_pose, left_grip, right_grip, global_step


def _run_active_gripper_phase(
    env,
    phase: str,
    active_arm: str,
    target_pose: torch.Tensor,
    left_pose: torch.Tensor,
    right_pose: torch.Tensor,
    left_grip: float,
    right_grip: float,
    active_grip: float,
    global_step: int,
    final_leg_pose_env: torch.Tensor,
    steps: int | None = None,
    target_pose_fn: Callable[[], torch.Tensor] | None = None,
    stop_on_success: bool = False,
    lift_guard: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, float, float, int]:
    if active_arm == "left":
        target_left_grip = active_grip
        target_right_grip = right_grip
    else:
        target_left_grip = left_grip
        target_right_grip = active_grip
    return _run_phase(
        env=env,
        phase=phase,
        active_arm=active_arm,
        target_pose=target_pose,
        left_pose=left_pose,
        right_pose=right_pose,
        left_grip=left_grip,
        right_grip=right_grip,
        target_left_grip=target_left_grip,
        target_right_grip=target_right_grip,
        global_step=global_step,
        final_leg_pose_env=final_leg_pose_env,
        steps=steps,
        target_pose_fn=target_pose_fn,
        stop_on_success=stop_on_success,
        lift_guard=lift_guard,
    )


def _print_final_status(unwrapped) -> tuple[bool, float, bool, bool]:
    success = bool(unwrapped._success()[0].item())
    reward = float(unwrapped._get_rewards()[0].item())
    terminated, truncated = unwrapped._get_dones()
    terminated_value = bool(terminated[0].item())
    truncated_value = bool(truncated[0].item())
    rel_pos, rel_quat, pos_error = _final_relative_pose_diagnostics(unwrapped)
    rel_ori_error = _quat_angle_error(rel_quat, _quat_tensor(unwrapped.device, IDENTITY_QUAT))
    print(
        "[INFO]: final "
        f"success={success} reward={reward:.1f} terminated={terminated_value} timeout={truncated_value} "
        f"rel_pos={_format_vec(rel_pos)} rel_quat={_format_vec(rel_quat)} "
        f"rel_pos_error={_format_vec(pos_error)} rel_ori_error={float(rel_ori_error[0]):.4f} rad"
    )
    return success, reward, terminated_value, truncated_value


def _run_scripted_assembly(env) -> int:
    global _CAMERA_VIDEO_RECORDER, _MARKERS, _SIM_RATE_STATS

    unwrapped = env.unwrapped
    env.reset()
    _SIM_RATE_STATS = SimRateStats(
        sim_dt=float(unwrapped.cfg.sim.dt),
        decimation=int(unwrapped.cfg.decimation),
        render_interval=int(unwrapped.cfg.sim.render_interval),
    )
    _set_planned_grasp_pose(None)
    _set_planned_insert_pose(None)
    _set_planned_screw_pose(None)
    _MARKERS = None if args_cli.disable_markers else _make_markers()
    _visualize_markers(unwrapped)
    if args_cli.record_camera:
        video_path = args_cli.camera_video_path or _default_camera_video_path()
        _CAMERA_VIDEO_RECORDER = CameraVideoRecorder(
            camera_name=args_cli.camera_name,
            video_path=video_path,
            fps=_infer_camera_video_fps(unwrapped),
            frame_interval=args_cli.camera_frame_interval,
        )
        _CAMERA_VIDEO_RECORDER.validate(unwrapped)
        print(
            "[INFO]: recording scene camera "
            f"name={args_cli.camera_name} path={_CAMERA_VIDEO_RECORDER.video_path} "
            f"fps={_CAMERA_VIDEO_RECORDER.fps:.2f} frame_interval={args_cli.camera_frame_interval}"
        )

    with torch.inference_mode():
        left_pose, right_pose = unwrapped._get_ee_poses_in_root_frame()
        left_pose = left_pose.clone()
        right_pose = right_pose.clone()
        left_grip = OPEN_GRIPPER
        right_grip = OPEN_GRIPPER
        global_step = 0

        leg_pose = _leg_pose_env(unwrapped)
        active_arm = "left" if float(leg_pose[0, 1]) > 0.0 else "right"
        active_pose = left_pose if active_arm == "left" else right_pose
        final_leg_pose_env = _final_leg_pose_env(unwrapped)
        grasp_quat = _grasp_quat(unwrapped, active_pose, final_leg_pose_env)

        def current_grasp_pose(clearance: float) -> torch.Tensor:
            current_leg_pose = _leg_pose_env(unwrapped)
            grasp_center_pos = current_leg_pose[:, :3].clone()
            grasp_center_pos[:, 2] += args_cli.finger_center_offset_z + clearance
            grasp_pose = _ee_pose_for_finger_center(unwrapped, active_arm, grasp_center_pos, grasp_quat)
            if clearance <= 0.0:
                _set_planned_grasp_pose(grasp_pose)
            return grasp_pose

        pre_grasp_pose = current_grasp_pose(args_cli.overhead_clearance)
        current_grasp_pose(0.0)
        raise_pos = active_pose[:, :3].clone()
        raise_pos[:, 2] = torch.maximum(
            raise_pos[:, 2],
            torch.tensor(pre_grasp_pose[0, 2].item(), dtype=torch.float32, device=unwrapped.device),
        )
        raise_pose = _pose_from_pos_quat(raise_pos, grasp_quat)

        print(
            f"[INFO]: physical one_leg setup arm={active_arm} "
            f"orientation={args_cli.gripper_orientation} "
            f"frame_markers={not args_cli.disable_markers} "
            f"grasp_quat={_format_vec(grasp_quat)} "
            f"screw_cycles={args_cli.screw_cycles} "
            f"screw_rotation_total={args_cli.screw_rotation:.4f} rad "
            f"screw_rotation_per_cycle={args_cli.screw_rotation / float(args_cli.screw_cycles):.4f} rad "
            f"screw_down_depth={args_cli.screw_down_depth:.4f} m "
            f"screw_gripper_value={args_cli.screw_gripper_value:.2f} "
            f"screw_lift_abort_threshold={args_cli.screw_lift_abort_threshold:.4f} m "
            f"screw_min_follow_fraction={args_cli.screw_min_follow_fraction:.2f} "
            f"screw_grip_slip_tolerance={args_cli.screw_grip_slip_tolerance:.4f} m "
            f"screw_grip_ori_slip_tolerance={args_cli.screw_grip_ori_slip_tolerance:.4f} rad "
            f"rate_interval={args_cli.rate_interval} "
            f"sim_dt_s={_SIM_RATE_STATS.sim_dt:.6f} "
            f"decimation={_SIM_RATE_STATS.decimation} "
            f"leg_initial_b={_format_pose(leg_pose)} "
            f"final_leg_b={_format_pose(final_leg_pose_env)} "
            f"table_surface_z={_table_surface_z(unwrapped):.4f} m"
        )

        grasp_phases: tuple[tuple[str, torch.Tensor, float, int, Callable[[], torch.Tensor] | None], ...] = (
            ("raise-arm", raise_pose, OPEN_GRIPPER, args_cli.phase_steps, None),
            (
                "move-above-grasp",
                pre_grasp_pose,
                OPEN_GRIPPER,
                args_cli.phase_steps,
                lambda: current_grasp_pose(args_cli.overhead_clearance),
            ),
            (
                "descend-to-grasp",
                current_grasp_pose(0.0),
                OPEN_GRIPPER,
                args_cli.phase_steps,
                lambda: current_grasp_pose(0.0),
            ),
            (
                "settle-at-grasp",
                current_grasp_pose(0.0),
                OPEN_GRIPPER,
                args_cli.settle_steps,
                lambda: current_grasp_pose(0.0),
            ),
            (
                "close",
                current_grasp_pose(0.0),
                CLOSE_GRIPPER,
                args_cli.close_steps,
                lambda: current_grasp_pose(0.0),
            ),
        )

        for phase, phase_target_pose, gripper_value, steps, target_pose_fn in grasp_phases:
            if active_arm == "left":
                target_left_grip = gripper_value
                target_right_grip = right_grip
            else:
                target_left_grip = left_grip
                target_right_grip = gripper_value
            left_pose, right_pose, left_grip, right_grip, global_step = _run_phase(
                env=env,
                phase=phase,
                active_arm=active_arm,
                target_pose=phase_target_pose,
                left_pose=left_pose,
                right_pose=right_pose,
                left_grip=left_grip,
                right_grip=right_grip,
                target_left_grip=target_left_grip,
                target_right_grip=target_right_grip,
                global_step=global_step,
                final_leg_pose_env=final_leg_pose_env,
                steps=steps,
                target_pose_fn=target_pose_fn,
            )

        leg_z_before_lift = float(_leg_pose_env(unwrapped)[0, 2].item())
        lift_center = _finger_center_in_root_frame(unwrapped, active_arm).clone()
        lift_center[:, 2] += args_cli.lift_height
        lift_pose = _ee_pose_for_finger_center(unwrapped, active_arm, lift_center, grasp_quat)
        left_pose, right_pose, left_grip, right_grip, global_step = _run_phase(
            env=env,
            phase="lift-test",
            active_arm=active_arm,
            target_pose=lift_pose,
            left_pose=left_pose,
            right_pose=right_pose,
            left_grip=left_grip,
            right_grip=right_grip,
            target_left_grip=left_grip,
            target_right_grip=right_grip,
            global_step=global_step,
            final_leg_pose_env=final_leg_pose_env,
        )

        leg_pose_after_lift = _leg_pose_env(unwrapped)
        lift_delta = float(leg_pose_after_lift[0, 2].item() - leg_z_before_lift)
        min_lift = min(0.035, 0.25 * args_cli.lift_height)
        print(f"[INFO]: lift-test checkpoint lift_delta={lift_delta:.4f} m min_lift={min_lift:.4f} m")
        if lift_delta < min_lift:
            print("[ERROR]: physical grasp failed: leg4 did not lift with the gripper.")
            _print_final_status(unwrapped)
            return 1

        leg_pos_in_finger, leg_quat_in_finger = _held_leg_relative_to_finger(
            unwrapped, active_arm, leg_pose_after_lift
        )
        reorient_leg_pose = leg_pose_after_lift.clone()
        high_insert_z = torch.maximum(
            final_leg_pose_env[:, 2] + args_cli.overhead_clearance + args_cli.insert_clearance,
            torch.full_like(final_leg_pose_env[:, 2], _table_surface_z(unwrapped) + args_cli.overhead_clearance),
        )
        reorient_leg_pose[:, 2] = torch.maximum(reorient_leg_pose[:, 2], high_insert_z)
        reorient_leg_pose[:, 3:7] = final_leg_pose_env[:, 3:7]
        reorient_pose = _ee_pose_for_held_leg_pose(
            unwrapped,
            active_arm,
            reorient_leg_pose,
            leg_pos_in_finger,
            leg_quat_in_finger,
        )
        _set_planned_insert_pose(reorient_pose)
        print(
            "[INFO]: held-leg relation after lift "
            f"leg_in_finger_pos={_format_vec(leg_pos_in_finger)} "
            f"leg_in_finger_quat={_format_vec(leg_quat_in_finger)} "
            f"reorient_target_b={_format_pose(reorient_pose)}"
        )
        left_pose, right_pose, left_grip, right_grip, global_step = _run_phase(
            env=env,
            phase="reorient-for-insert",
            active_arm=active_arm,
            target_pose=reorient_pose,
            left_pose=left_pose,
            right_pose=right_pose,
            left_grip=left_grip,
            right_grip=right_grip,
            target_left_grip=left_grip,
            target_right_grip=right_grip,
            global_step=global_step,
            final_leg_pose_env=final_leg_pose_env,
        )

        leg_pose_after_reorient = _leg_pose_env(unwrapped)
        leg_pos_in_finger, leg_quat_in_finger = _held_leg_relative_to_finger(
            unwrapped, active_arm, leg_pose_after_reorient
        )
        pre_insert_leg_pose = final_leg_pose_env.clone()
        pre_insert_leg_pose[:, 2] += args_cli.overhead_clearance + args_cli.insert_clearance
        insert_leg_pose = final_leg_pose_env.clone()
        insert_leg_pose[:, 2] += args_cli.insert_clearance
        seat_leg_pose = final_leg_pose_env.clone()
        seat_leg_pose[:, 2] -= args_cli.insert_push_depth

        pre_insert_pose = _ee_pose_for_held_leg_pose(
            unwrapped,
            active_arm,
            pre_insert_leg_pose,
            leg_pos_in_finger,
            leg_quat_in_finger,
        )
        insert_pose = _ee_pose_for_held_leg_pose(
            unwrapped,
            active_arm,
            insert_leg_pose,
            leg_pos_in_finger,
            leg_quat_in_finger,
        )
        seat_pose = _ee_pose_for_held_leg_pose(
            unwrapped,
            active_arm,
            seat_leg_pose,
            leg_pos_in_finger,
            leg_quat_in_finger,
        )
        _set_planned_insert_pose(seat_pose)
        print(
            "[INFO]: insert planning "
            f"insert_quat={_format_vec(seat_pose[:, 3:7])} "
            f"leg_in_finger_pos={_format_vec(leg_pos_in_finger)} "
            f"leg_in_finger_quat={_format_vec(leg_quat_in_finger)} "
            f"pre_insert_b={_format_pose(pre_insert_pose)} "
            f"seat_b={_format_pose(seat_pose)}"
        )

        insert_phases = (
            ("move-above-insert", pre_insert_pose, CLOSE_GRIPPER, args_cli.phase_steps, False),
            ("descend-to-pre-insert", insert_pose, CLOSE_GRIPPER, args_cli.phase_steps, True),
            ("seat-insert", seat_pose, CLOSE_GRIPPER, args_cli.phase_steps, True),
        )

        insert_stopped_on_success = False
        for phase, phase_target_pose, gripper_value, steps, stop_on_success in insert_phases:
            if active_arm == "left":
                target_left_grip = gripper_value
                target_right_grip = right_grip
            else:
                target_left_grip = left_grip
                target_right_grip = gripper_value
            left_pose, right_pose, left_grip, right_grip, global_step = _run_phase(
                env=env,
                phase=phase,
                active_arm=active_arm,
                target_pose=phase_target_pose,
                left_pose=left_pose,
                right_pose=right_pose,
                left_grip=left_grip,
                right_grip=right_grip,
                target_left_grip=target_left_grip,
                target_right_grip=target_right_grip,
                global_step=global_step,
                final_leg_pose_env=final_leg_pose_env,
                steps=steps,
                stop_on_success=stop_on_success,
            )
            if stop_on_success and bool(unwrapped._success()[0].item()):
                insert_stopped_on_success = True
                break

        hold_pose = _active_ee_pose(unwrapped, active_arm)
        if active_arm == "left":
            left_pose = hold_pose.clone()
            target_left_grip = CLOSE_GRIPPER
            target_right_grip = right_grip
        else:
            right_pose = hold_pose.clone()
            target_left_grip = left_grip
            target_right_grip = CLOSE_GRIPPER
        print(
            "[INFO]: hold-insert target=current_ee "
            f"insert_stopped_on_success={insert_stopped_on_success} "
            f"hold_b={_format_pose(hold_pose)}"
        )
        left_pose, right_pose, left_grip, right_grip, global_step = _run_phase(
            env=env,
            phase="hold-insert",
            active_arm=active_arm,
            target_pose=hold_pose,
            left_pose=left_pose,
            right_pose=right_pose,
            left_grip=left_grip,
            right_grip=right_grip,
            target_left_grip=target_left_grip,
            target_right_grip=target_right_grip,
            global_step=global_step,
            final_leg_pose_env=final_leg_pose_env,
            steps=args_cli.settle_steps,
        )

        rel_pos_before_screw, rel_quat_before_screw, pos_error_before_screw = _final_relative_pose_diagnostics(
            unwrapped
        )
        rel_ori_error_before_screw = _quat_angle_error(
            rel_quat_before_screw,
            _quat_tensor(unwrapped.device, IDENTITY_QUAT),
        )
        print(
            "[INFO]: insert checkpoint before screw "
            f"rel_pos={_format_vec(rel_pos_before_screw)} "
            f"rel_pos_error={_format_vec(pos_error_before_screw)} "
            f"rel_ori_error={float(rel_ori_error_before_screw[0]):.4f} rad "
            f"success={bool(unwrapped._success()[0].item())}"
        )

        screw_leg_pos_in_finger, screw_leg_quat_in_finger = _held_leg_relative_to_finger(unwrapped, active_arm)
        screw_cycle_rotation = args_cli.screw_rotation / float(args_cli.screw_cycles)
        initial_screw_leg_pose = _leg_pose_env(unwrapped).clone()
        initial_aligned_ready_pose, _ = _ee_pose_for_current_leg_alignment(
            unwrapped,
            active_arm,
            screw_leg_pos_in_finger,
            screw_leg_quat_in_finger,
        )
        initial_screw_target_leg_pose = initial_screw_leg_pose.clone()
        initial_screw_target_leg_pose[:, 2] -= args_cli.screw_down_depth
        screw_target_leg_delta_xyz = initial_screw_target_leg_pose[:, :3] - initial_screw_leg_pose[:, :3]

        print(
            "[INFO]: screw execution planning "
            "screw_mode=continuous_no_regrasp "
            "screw_axis=root_z "
            "post_insert_regrasp=False "
            f"screw_cycles={args_cli.screw_cycles} "
            f"screw_rotation_total={args_cli.screw_rotation:.4f} rad "
            f"screw_rotation_per_cycle={screw_cycle_rotation:.4f} rad "
            f"screw_down_depth={args_cli.screw_down_depth:.4f} m "
            f"screw_gripper_value={args_cli.screw_gripper_value:.2f} "
            f"screw_min_follow_fraction={args_cli.screw_min_follow_fraction:.2f} "
            f"screw_grip_slip_tolerance={args_cli.screw_grip_slip_tolerance:.4f} m "
            f"screw_grip_ori_slip_tolerance={args_cli.screw_grip_ori_slip_tolerance:.4f} rad "
            f"screw_target_leg_delta_xyz={_format_vec(screw_target_leg_delta_xyz)} "
            f"screw_target_leg_delta_z={float(screw_target_leg_delta_xyz[0, 2]):.4f} m "
            f"leg_in_finger_pos={_format_vec(screw_leg_pos_in_finger)} "
            f"leg_in_finger_quat={_format_vec(screw_leg_quat_in_finger)} "
            f"leg_b={_format_pose(initial_screw_leg_pose)} "
            f"aligned_ready_b={_format_pose(initial_aligned_ready_pose)}"
        )

        screw_success = bool(unwrapped._success()[0].item())
        screw_cycles_completed = 0
        screw_lift_aborted = False
        screw_contact_lost = False
        if screw_success:
            print("[INFO]: screw cycles skipped reason=success_before_screw")

        def contact_lost_after_screw_phase(
            phase: str,
            screw_cycle: int,
            start_leg_pose: torch.Tensor,
            start_leg_pos_in_finger: torch.Tensor,
            start_leg_quat_in_finger: torch.Tensor,
        ) -> bool:
            current_leg_pose = _leg_pose_env(unwrapped).clone()
            current_leg_pos_in_finger, current_leg_quat_in_finger = _held_leg_relative_to_finger(
                unwrapped,
                active_arm,
            )
            actual_leg_rotation = _quat_angle_error(start_leg_pose[:, 3:7], current_leg_pose[:, 3:7])
            expected_leg_rotation = max(abs(screw_cycle_rotation), 1.0e-6)
            follow_fraction = actual_leg_rotation / expected_leg_rotation
            grip_pos_slip = torch.linalg.norm(current_leg_pos_in_finger - start_leg_pos_in_finger, dim=-1)
            grip_ori_slip = _quat_angle_error(start_leg_quat_in_finger, current_leg_quat_in_finger)
            leg_not_following = float(follow_fraction[0].item()) < args_cli.screw_min_follow_fraction
            grip_slipped = (
                float(grip_pos_slip[0].item()) > args_cli.screw_grip_slip_tolerance
                or float(grip_ori_slip[0].item()) > args_cli.screw_grip_ori_slip_tolerance
            )
            print(
                "[INFO]: screw engagement check "
                f"phase={phase} "
                f"cycle={screw_cycle}/{args_cli.screw_cycles} "
                f"actual_leg_rotation={float(actual_leg_rotation[0]):.4f} rad "
                f"expected_leg_rotation={expected_leg_rotation:.4f} rad "
                f"follow_fraction={float(follow_fraction[0]):.2f} "
                f"grip_pos_slip={float(grip_pos_slip[0]):.4f} m "
                f"grip_ori_slip={float(grip_ori_slip[0]):.4f} rad "
                f"leg_b={_format_pose(current_leg_pose)} "
                f"leg_in_finger_pos={_format_vec(current_leg_pos_in_finger)} "
                f"leg_in_finger_quat={_format_vec(current_leg_quat_in_finger)}"
            )
            if bool(unwrapped._success()[0].item()):
                return False
            if not (leg_not_following or grip_slipped):
                return False
            print(
                "[WARN]: screw_contact_lost_abort "
                f"phase={phase} "
                f"cycle={screw_cycle}/{args_cli.screw_cycles} "
                f"leg_not_following={leg_not_following} "
                f"grip_slipped={grip_slipped} "
                f"follow_fraction={float(follow_fraction[0]):.2f} "
                f"min_follow_fraction={args_cli.screw_min_follow_fraction:.2f} "
                f"grip_pos_slip={float(grip_pos_slip[0]):.4f} m "
                f"grip_pos_limit={args_cli.screw_grip_slip_tolerance:.4f} m "
                f"grip_ori_slip={float(grip_ori_slip[0]):.4f} rad "
                f"grip_ori_limit={args_cli.screw_grip_ori_slip_tolerance:.4f} rad"
            )
            return True

        screw_cycle_count = 0 if screw_success or screw_lift_aborted else args_cli.screw_cycles
        for screw_cycle in range(1, screw_cycle_count + 1):
            screw_start_leg_pose = _leg_pose_env(unwrapped).clone()
            cycle_leg_pos_in_finger, cycle_leg_quat_in_finger = _held_leg_relative_to_finger(
                unwrapped,
                active_arm,
            )
            aligned_ready_pose, aligned_ready_center = _ee_pose_for_current_leg_alignment(
                unwrapped,
                active_arm,
                cycle_leg_pos_in_finger,
                cycle_leg_quat_in_finger,
            )
            screw_delta_quat = _axis_angle_quat(unwrapped.device, (0.0, 0.0, 1.0), screw_cycle_rotation)
            screw_target_leg_pose = screw_start_leg_pose.clone()
            screw_target_leg_pose[:, 2] -= args_cli.screw_down_depth
            screw_target_leg_pose[:, 3:7] = _normalize_quat(
                _quat_mul(screw_delta_quat, screw_start_leg_pose[:, 3:7])
            )
            screw_pose = _ee_pose_for_held_leg_pose_without_lift(
                unwrapped,
                active_arm,
                screw_target_leg_pose,
                cycle_leg_pos_in_finger,
                cycle_leg_quat_in_finger,
            )
            _set_planned_screw_pose(screw_pose)
            screw_ori_delta = _quat_angle_error(screw_start_leg_pose[:, 3:7], screw_target_leg_pose[:, 3:7])
            screw_target_leg_delta_xyz = screw_target_leg_pose[:, :3] - screw_start_leg_pose[:, :3]
            finger_center_target_delta_xyz = (
                _target_finger_center_for_ee_pose(unwrapped, active_arm, screw_pose) - aligned_ready_center
            )
            if float(screw_target_leg_delta_xyz[0, 2].item()) > 1.0e-5:
                print(
                    "[WARN]: screw cycle target leg z moved upward; "
                    f"cycle={screw_cycle}/{args_cli.screw_cycles} "
                    f"delta_z={float(screw_target_leg_delta_xyz[0, 2]):.6f} m"
                )
            print(
                "[INFO]: screw cycle planning "
                f"cycle={screw_cycle}/{args_cli.screw_cycles} "
                f"cycle_rotation={screw_cycle_rotation:.4f} rad "
                f"screw_target_leg_delta_xyz={_format_vec(screw_target_leg_delta_xyz)} "
                f"screw_target_leg_delta_z={float(screw_target_leg_delta_xyz[0, 2]):.4f} m "
                f"finger_center_target_delta_xyz={_format_vec(finger_center_target_delta_xyz)} "
                f"ee_ori_delta={float(screw_ori_delta[0]):.4f} rad "
                f"cycle_leg_in_finger_pos={_format_vec(cycle_leg_pos_in_finger)} "
                f"cycle_leg_in_finger_quat={_format_vec(cycle_leg_quat_in_finger)} "
                f"leg_b={_format_pose(screw_start_leg_pose)} "
                f"screw_target_leg_b={_format_pose(screw_target_leg_pose)} "
                f"aligned_ready_b={_format_pose(aligned_ready_pose)} "
                f"screw_pose_b={_format_pose(screw_pose)}"
            )

            left_pose, right_pose, left_grip, right_grip, global_step = _run_active_gripper_phase(
                env=env,
                phase=f"screw-cycle-{screw_cycle}",
                active_arm=active_arm,
                target_pose=screw_pose,
                left_pose=left_pose,
                right_pose=right_pose,
                left_grip=left_grip,
                right_grip=right_grip,
                active_grip=args_cli.screw_gripper_value,
                global_step=global_step,
                final_leg_pose_env=final_leg_pose_env,
                steps=args_cli.screw_steps,
                stop_on_success=True,
                lift_guard=True,
            )
            screw_cycles_completed = screw_cycle
            if _LIFT_GUARD_ABORTED:
                screw_lift_aborted = True
                break
            if contact_lost_after_screw_phase(
                phase=f"screw-cycle-{screw_cycle}",
                screw_cycle=screw_cycle,
                start_leg_pose=screw_start_leg_pose,
                start_leg_pos_in_finger=cycle_leg_pos_in_finger,
                start_leg_quat_in_finger=cycle_leg_quat_in_finger,
            ):
                screw_contact_lost = True
                break
            screw_success = bool(unwrapped._success()[0].item())
            if screw_success:
                break

            left_pose, right_pose, left_grip, right_grip, global_step = _run_active_gripper_phase(
                env=env,
                phase=f"hold-screw-cycle-{screw_cycle}",
                active_arm=active_arm,
                target_pose=screw_pose,
                left_pose=left_pose,
                right_pose=right_pose,
                left_grip=left_grip,
                right_grip=right_grip,
                active_grip=args_cli.screw_gripper_value,
                global_step=global_step,
                final_leg_pose_env=final_leg_pose_env,
                steps=args_cli.settle_steps,
                stop_on_success=True,
                lift_guard=True,
            )
            if _LIFT_GUARD_ABORTED:
                screw_lift_aborted = True
                break
            if contact_lost_after_screw_phase(
                phase=f"hold-screw-cycle-{screw_cycle}",
                screw_cycle=screw_cycle,
                start_leg_pose=screw_start_leg_pose,
                start_leg_pos_in_finger=cycle_leg_pos_in_finger,
                start_leg_quat_in_finger=cycle_leg_quat_in_finger,
            ):
                screw_contact_lost = True
                break
            screw_success = bool(unwrapped._success()[0].item())
            if screw_success or screw_cycle == args_cli.screw_cycles:
                break

            print(
                "[INFO]: continuing continuous screw without regrasp "
                f"cycle={screw_cycle}/{args_cli.screw_cycles} "
                f"active_grip={args_cli.screw_gripper_value:.2f}"
            )

        print(
            "[INFO]: screw cycles complete "
            "screw_mode=continuous_no_regrasp "
            f"cycles_completed={screw_cycles_completed}/{args_cli.screw_cycles} "
            f"lift_guard_aborted={screw_lift_aborted} "
            f"contact_lost={screw_contact_lost} "
            f"success={screw_success}"
        )

        release_pose = _active_ee_pose(unwrapped, active_arm).clone()
        left_pose, right_pose, left_grip, right_grip, global_step = _run_active_gripper_phase(
            env=env,
            phase="screw-final-release",
            active_arm=active_arm,
            target_pose=release_pose,
            left_pose=left_pose,
            right_pose=right_pose,
            left_grip=left_grip,
            right_grip=right_grip,
            active_grip=OPEN_GRIPPER,
            global_step=global_step,
            final_leg_pose_env=final_leg_pose_env,
            steps=args_cli.close_steps,
        )
        retreat_center = _finger_center_in_root_frame(unwrapped, active_arm).clone()
        retreat_center[:, 2] += max(args_cli.screw_regrasp_clearance, args_cli.insert_clearance + 0.02)
        retreat_pose = _ee_pose_for_finger_center(unwrapped, active_arm, retreat_center, release_pose[:, 3:7])
        left_pose, right_pose, left_grip, right_grip, global_step = _run_active_gripper_phase(
            env=env,
            phase="retreat-after-screw",
            active_arm=active_arm,
            target_pose=retreat_pose,
            left_pose=left_pose,
            right_pose=right_pose,
            left_grip=left_grip,
            right_grip=right_grip,
            active_grip=OPEN_GRIPPER,
            global_step=global_step,
            final_leg_pose_env=final_leg_pose_env,
            steps=args_cli.phase_steps,
        )

        rel_pos_after_screw, rel_quat_after_screw, pos_error_after_screw = _final_relative_pose_diagnostics(
            unwrapped
        )
        rel_ori_error_after_screw = _quat_angle_error(
            rel_quat_after_screw,
            _quat_tensor(unwrapped.device, IDENTITY_QUAT),
        )
        print(
            "[INFO]: screw checkpoint after release "
            f"rel_pos={_format_vec(rel_pos_after_screw)} "
            f"rel_pos_error={_format_vec(pos_error_after_screw)} "
            f"rel_ori_error={float(rel_ori_error_after_screw[0]):.4f} rad "
            f"success={bool(unwrapped._success()[0].item())}"
        )
        z_error_before = torch.abs(pos_error_before_screw[:, 2])
        z_error_after = torch.abs(pos_error_after_screw[:, 2])
        if bool((z_error_after > z_error_before + 0.002).any().item()):
            print(
                "[WARN]: leg4 z error increased during screw; "
                f"before={float(pos_error_before_screw[0, 2]):.4f} m "
                f"after={float(pos_error_after_screw[0, 2]):.4f} m"
            )

        success, reward, terminated_value, _ = _print_final_status(unwrapped)
        _print_sim_rate_summary()
        if not success or reward < 1.0 or not terminated_value:
            print("[ERROR]: scripted one_leg physical assembly did not reach the success condition.")
            return 1
        return 0


def main() -> int:
    env_cfg = parse_env_cfg(
        TASK_NAME,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
    )
    env = gym.make(TASK_NAME, cfg=env_cfg)

    try:
        print(f"[INFO]: Gym observation space: {env.observation_space}")
        print(f"[INFO]: Gym action space: {env.action_space}")
        print(f"[INFO]: Part/gripper frame markers: enabled={not args_cli.disable_markers}")
        return _run_scripted_assembly(env)
    finally:
        if _CAMERA_VIDEO_RECORDER is not None:
            _CAMERA_VIDEO_RECORDER.close()
        env.close()


if __name__ == "__main__":
    exit_code = 0
    try:
        exit_code = main()
    except Exception:
        exit_code = 1
        traceback.print_exc()
    finally:
        if exit_code == 0 and args_cli.fast_exit:
            sys.stdout.flush()
            sys.stderr.flush()
            os._exit(0)
        if exit_code == 0:
            simulation_app.close()
        else:
            print(f"[ERROR]: exiting with code {exit_code}", flush=True)
            sys.stdout.flush()
            sys.stderr.flush()
            os._exit(exit_code)
    sys.exit(exit_code)
