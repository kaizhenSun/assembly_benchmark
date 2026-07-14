# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Keyboard teleoperation for the Galaxea R1 Pro bimanual IK task.

This script drives the existing 16D R1 Pro IK action interface with keyboard
delta commands. It defaults to the generic assembly benchmark task.

.. code-block:: bash

    python scripts/tools/run_r1_pro_keyboard_teleop.py --num_envs 1 --device cuda:0

"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass

from isaaclab.app import AppLauncher

DEFAULT_TASK_NAME = "Assembly-Benchmark-Direct-v0"
OPEN_GRIPPER = 1.0
CLOSE_GRIPPER = -1.0
CONTROL_MODES = ("left", "right", "both")
DEFAULT_TACTILE_3D_ORIGIN = (0.90, -0.45, 0.775)
TACTILE_3D_BASE_OFFSET = 0.001
TACTILE_3D_MIN_HEIGHT = 0.0005
TACTILE_3D_COLOR_STOPS = (
    (0.03, 0.08, 0.45),
    (0.05, 0.23, 0.80),
    (0.00, 0.55, 0.85),
    (0.00, 0.72, 0.48),
    (0.65, 0.82, 0.12),
    (1.00, 0.68, 0.08),
    (0.95, 0.26, 0.12),
)


parser = argparse.ArgumentParser(description="Run keyboard teleoperation for R1 Pro bimanual IK.")
parser.add_argument(
    "--task",
    type=str,
    default=DEFAULT_TASK_NAME,
    help="IK task to teleoperate. Defaults to Assembly-Benchmark-Direct-v0.",
)
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments. Only 1 is supported.")
parser.add_argument(
    "--disable_fabric",
    action="store_true",
    default=False,
    help="Debug/compatibility option: disable Fabric and use USD I/O, which may desync GUI mesh updates.",
)
parser.add_argument("--pos_step", type=float, default=0.002, help="Position delta per sim loop while a key is held.")
parser.add_argument(
    "--rot_step",
    type=float,
    default=0.015,
    help="Rotation-vector delta per sim loop while a key is held.",
)
parser.add_argument("--marker_scale", type=float, default=0.08, help="Scale of current/target gripper frame markers.")
parser.add_argument("--disable_markers", action="store_true", help="Disable gripper frame marker visualization.")
parser.add_argument(
    "--include_torso_in_ik",
    action="store_true",
    help="Include torso joints in a joint bimanual IK solve for this teleop run.",
)
parser.add_argument(
    "--enable_torso_keys",
    action="store_true",
    help="Enable a P-toggled direct torso joint mode for tasks that are not already using torso IK.",
)
parser.add_argument(
    "--torso_step",
    type=float,
    default=0.01,
    help="Joint-angle delta in radians per sim loop while a torso key is held.",
)
parser.add_argument(
    "--print_interval",
    type=int,
    default=60,
    help="Print end-effector tracking diagnostics every N control loops. Use 0 to disable.",
)
parser.add_argument(
    "--print_joint_angles",
    action="store_true",
    help="Print env0 current joint angles in radians with the periodic diagnostics.",
)
parser.add_argument(
    "--disable_tactile_pressure_view",
    action="store_true",
    help="Disable the R1 Pro gripper tactile pressure 3D panel.",
)
parser.add_argument(
    "--tactile_pressure_scale",
    type=float,
    default=0.0,
    help="Raw normal-force value mapped to max 3D height/color. Use 0 for per-frame normalized force.",
)
parser.add_argument(
    "--tactile_pressure_update_interval",
    type=int,
    default=1,
    help="Refresh tactile pressure 3D panel every N control loops.",
)
parser.add_argument(
    "--tactile_3d_max_height",
    type=float,
    default=0.03,
    help="Maximum 3D taxel pillar height in meters.",
)
parser.add_argument(
    "--tactile_3d_taxel_size",
    type=float,
    default=0.0016,
    help="3D taxel pillar width/depth in meters.",
)
parser.add_argument(
    "--tactile_3d_origin",
    type=float,
    nargs=3,
    default=DEFAULT_TACTILE_3D_ORIGIN,
    metavar=("X", "Y", "Z"),
    help="World position of the center of the tabletop tactile pressure panel baseline.",
)
parser.add_argument(
    "--tactile_3d_yaw",
    type=float,
    default=0.0,
    help="Yaw angle in radians for the tabletop tactile pressure panel.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

if args_cli.num_envs != 1:
    raise ValueError("Keyboard teleoperation currently supports only --num_envs 1.")
if getattr(args_cli, "headless", False):
    raise ValueError("Keyboard teleoperation requires a GUI window. Remove --headless to use keyboard input.")
if args_cli.pos_step <= 0.0:
    raise ValueError("--pos_step must be positive.")
if args_cli.rot_step <= 0.0:
    raise ValueError("--rot_step must be positive.")
if args_cli.torso_step <= 0.0:
    raise ValueError("--torso_step must be positive.")
if args_cli.marker_scale <= 0.0:
    raise ValueError("--marker_scale must be positive.")
if args_cli.print_interval < 0:
    raise ValueError("--print_interval must be non-negative.")
if args_cli.tactile_pressure_scale < 0.0:
    raise ValueError("--tactile_pressure_scale must be non-negative.")
if args_cli.tactile_pressure_update_interval <= 0:
    raise ValueError("--tactile_pressure_update_interval must be positive.")
if args_cli.tactile_3d_max_height <= 0.0:
    raise ValueError("--tactile_3d_max_height must be positive.")
if args_cli.tactile_3d_taxel_size <= 0.0:
    raise ValueError("--tactile_3d_taxel_size must be positive.")
if args_cli.enable_torso_keys and args_cli.include_torso_in_ik:
    raise ValueError(
        "--enable_torso_keys cannot be combined with --include_torso_in_ik because both write torso joint targets."
    )

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import assembly_benchmark.tasks  # noqa: F401
import gymnasium as gym
import torch
from assembly_benchmark.sensors import (
    DEFAULT_TABLE_TACTILE_ARRAY_SIZE,
    DEFAULT_TABLE_TACTILE_POINT_DISTANCE,
    R1_PRO_GRIPPER_TACTILE_PAD_SPECS,
    tactile_force_grid,
)

import isaaclab.sim as sim_utils
from isaaclab.devices import Se3Keyboard, Se3KeyboardCfg
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.utils.math import apply_delta_pose, combine_frame_transforms

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg


class TeleopState:
    """Mutable keyboard callback state."""

    def __init__(self) -> None:
        self.mode_index = 0
        self.left_grip = OPEN_GRIPPER
        self.right_grip = OPEN_GRIPPER
        self.torso_mode = False
        self.reset_requested = False
        self.quit_requested = False

    @property
    def mode(self) -> str:
        return CONTROL_MODES[self.mode_index]

    def cycle_mode(self) -> None:
        self.mode_index = (self.mode_index + 1) % len(CONTROL_MODES)
        print(f"[INFO]: Control mode: {self.mode}", flush=True)

    def toggle_torso_mode(self) -> None:
        self.torso_mode = not self.torso_mode
        label = "torso" if self.torso_mode else f"arm ({self.mode})"
        print(f"[INFO]: Keyboard mode: {label}", flush=True)

    def toggle_gripper(self) -> None:
        if self.mode == "left":
            self.left_grip = _toggle_grip(self.left_grip)
        elif self.mode == "right":
            self.right_grip = _toggle_grip(self.right_grip)
        else:
            next_value = (
                OPEN_GRIPPER if self.left_grip == CLOSE_GRIPPER and self.right_grip == CLOSE_GRIPPER else CLOSE_GRIPPER
            )
            self.left_grip = next_value
            self.right_grip = next_value
        print(
            f"[INFO]: Gripper state: left={_grip_label(self.left_grip)} right={_grip_label(self.right_grip)}",
            flush=True,
        )

    def request_reset(self) -> None:
        self.reset_requested = True
        print("[INFO]: Reset requested.", flush=True)

    def request_quit(self) -> None:
        self.quit_requested = True
        print("[INFO]: Quit requested.", flush=True)


def _toggle_grip(value: float) -> float:
    return OPEN_GRIPPER if value == CLOSE_GRIPPER else CLOSE_GRIPPER


def _grip_label(value: float) -> str:
    return "open" if value == OPEN_GRIPPER else "closed"


def _make_action(
    left_pose: torch.Tensor,
    left_grip: float,
    right_pose: torch.Tensor,
    right_grip: float,
) -> torch.Tensor:
    actions = torch.zeros((1, 16), dtype=torch.float32, device=left_pose.device)
    actions[:, 0:7] = left_pose
    actions[:, 7] = left_grip
    actions[:, 8:15] = right_pose
    actions[:, 15] = right_grip
    return actions


def _step_without_auto_reset(
    env,
    actions: torch.Tensor,
    extra_joint_targets: tuple[torch.Tensor, list[int]] | None = None,
):
    """Step the DirectRLEnv physics path without applying its automatic reset."""
    unwrapped = env.unwrapped
    actions = actions.to(unwrapped.device)
    if extra_joint_targets is not None:
        extra_target_pos, extra_joint_ids = extra_joint_targets
        extra_target_pos = extra_target_pos.to(unwrapped.device)
    unwrapped._pre_physics_step(actions)
    is_rendering = unwrapped.sim.has_gui() or unwrapped.sim.has_rtx_sensors()

    for _ in range(unwrapped.cfg.decimation):
        unwrapped._sim_step_counter += 1
        unwrapped._apply_action()
        if extra_joint_targets is not None:
            unwrapped.robot.set_joint_position_target(extra_target_pos, joint_ids=extra_joint_ids)
        unwrapped.scene.write_data_to_sim()
        unwrapped.sim.step(render=False)
        if unwrapped._sim_step_counter % unwrapped.cfg.sim.render_interval == 0 and is_rendering:
            unwrapped.sim.render()
        unwrapped.scene.update(dt=unwrapped.physics_dt)

    unwrapped.obs_buf = unwrapped._get_observations()
    reward = unwrapped._get_rewards()
    terminated, truncated = unwrapped._get_dones()
    return unwrapped.obs_buf, reward, terminated, truncated, unwrapped.extras


@dataclass(frozen=True, slots=True)
class GripperTactilePadDiagnostics:
    label: str
    max_penetration_depth: float = 0.0
    mean_penetration_depth: float = 0.0
    active_taxels: int = 0
    max_force: float = 0.0


@dataclass(frozen=True, slots=True)
class GripperTactileDiagnostics:
    pads: tuple[GripperTactilePadDiagnostics, ...]

    @property
    def max_penetration_depth(self) -> float:
        return max((pad.max_penetration_depth for pad in self.pads), default=0.0)

    @property
    def mean_penetration_depth(self) -> float:
        return sum(pad.mean_penetration_depth for pad in self.pads) / max(len(self.pads), 1)

    @property
    def active_taxels(self) -> int:
        return sum(pad.active_taxels for pad in self.pads)

    @property
    def max_force(self) -> float:
        return max((pad.max_force for pad in self.pads), default=0.0)


class GripperTactile3DView:
    """Tabletop 3D taxel-pillar view for all four R1 Pro gripper tactile pads."""

    tactile_array_size = DEFAULT_TABLE_TACTILE_ARRAY_SIZE

    def __init__(
        self,
        *,
        origin: tuple[float, float, float],
        yaw: float,
        pressure_scale: float,
        max_height: float,
        taxel_size: float,
        base_offset: float,
    ) -> None:
        self._pressure_scale = float(pressure_scale)
        self._max_height = float(max_height)
        self._taxel_size = float(taxel_size)
        self._base_offset = float(base_offset)
        self._num_color_bins = len(TACTILE_3D_COLOR_STOPS)
        self._num_pads = len(R1_PRO_GRIPPER_TACTILE_PAD_SPECS)
        self._origin = torch.tensor(origin, dtype=torch.float32)
        base_panel_xy = self._make_panel_xy(yaw)
        pad_width = max(pad_spec.size[0] for pad_spec in R1_PRO_GRIPPER_TACTILE_PAD_SPECS)
        pad_height = max(pad_spec.size[2] for pad_spec in R1_PRO_GRIPPER_TACTILE_PAD_SPECS)
        column_spacing = pad_width + 0.02
        row_spacing = pad_height + 0.02
        column_direction = torch.tensor((math.cos(float(yaw)), math.sin(float(yaw))), dtype=torch.float32)
        row_direction = torch.tensor((-math.sin(float(yaw)), math.cos(float(yaw))), dtype=torch.float32)
        self._panel_xy = torch.cat(
            [
                base_panel_xy
                + column_index * column_spacing * column_direction
                + row_index * row_spacing * row_direction
                for column_index, row_index in (
                    pad_spec.panel_grid_coordinate for pad_spec in R1_PRO_GRIPPER_TACTILE_PAD_SPECS
                )
            ]
        )
        self._marker = self._make_marker()
        self._diagnostics = GripperTactileDiagnostics(
            tuple(GripperTactilePadDiagnostics(pad_spec.label) for pad_spec in R1_PRO_GRIPPER_TACTILE_PAD_SPECS)
        )
        self._visualize_force_values(torch.zeros((self._num_pads, *self.tactile_array_size), dtype=torch.float32))

        pad_sizes = tuple((pad_spec.size[0], pad_spec.size[2]) for pad_spec in R1_PRO_GRIPPER_TACTILE_PAD_SPECS)
        layout = "/".join(
            ",".join(
                pad_spec.label
                for pad_spec in R1_PRO_GRIPPER_TACTILE_PAD_SPECS
                if pad_spec.panel_grid_coordinate[1] == row_index
            )
            for row_index in range(2)
        )
        print(
            "[INFO]: Tactile pressure 3D panels created on table "
            f"origin={tuple(float(v) for v in self._origin)} "
            f"layout={layout} "
            f"pad_sizes={pad_sizes} "
            f"taxels={self.tactile_array_size[0]}x{self.tactile_array_size[1]}",
            flush=True,
        )

    def update(self, scene) -> None:
        normalize = self._pressure_scale <= 0.0
        force_grids = []
        pad_diagnostics = []
        for pad_spec in R1_PRO_GRIPPER_TACTILE_PAD_SPECS:
            sensor = scene[pad_spec.sensor_name]
            tactile_points = sensor.get_tactile_points(normalize=normalize)
            force_grid = (
                tactile_force_grid(
                    tactile_points,
                    array_size=self.tactile_array_size,
                    clamp_max=1.0 if normalize else None,
                )[0]
                .detach()
                .cpu()
            )
            force_grids.append(force_grid)
            penetration_depth = getattr(sensor.data, "penetration_depth", None)
            if penetration_depth is None:
                max_depth, mean_depth, active_taxels = 0.0, 0.0, 0
            else:
                depths = penetration_depth.detach().reshape(-1)
                max_depth = float(depths.max().item())
                mean_depth = float(depths.mean().item())
                active_taxels = int(torch.count_nonzero(depths > 0.0).item())
            pad_diagnostics.append(
                GripperTactilePadDiagnostics(
                    label=pad_spec.label,
                    max_penetration_depth=max_depth,
                    mean_penetration_depth=mean_depth,
                    active_taxels=active_taxels,
                    max_force=float(force_grid.max().item()),
                )
            )
        stacked_force_grids = torch.stack(force_grids)
        self._diagnostics = GripperTactileDiagnostics(tuple(pad_diagnostics))
        if not normalize:
            stacked_force_grids = stacked_force_grids / max(self._pressure_scale, 1.0e-12)
        self._visualize_force_values(stacked_force_grids.clamp(0.0, 1.0))

    def close(self) -> None:
        if getattr(self, "_marker", None) is not None:
            self._marker.set_visibility(False)

    @property
    def diagnostics(self) -> GripperTactileDiagnostics:
        return self._diagnostics

    def _make_panel_xy(self, yaw: float) -> torch.Tensor:
        rows, cols = self.tactile_array_size
        axis_0 = torch.linspace(
            -DEFAULT_TABLE_TACTILE_POINT_DISTANCE * (rows - 1) / 2.0,
            DEFAULT_TABLE_TACTILE_POINT_DISTANCE * (rows - 1) / 2.0,
            rows,
            dtype=torch.float32,
        )
        axis_1 = torch.linspace(
            -DEFAULT_TABLE_TACTILE_POINT_DISTANCE * (cols - 1) / 2.0,
            DEFAULT_TABLE_TACTILE_POINT_DISTANCE * (cols - 1) / 2.0,
            cols,
            dtype=torch.float32,
        )
        grid_x, grid_y = torch.meshgrid(axis_0, axis_1, indexing="ij")
        local_xy = torch.stack((grid_x.reshape(-1), grid_y.reshape(-1)), dim=-1)
        cos_yaw = math.cos(float(yaw))
        sin_yaw = math.sin(float(yaw))
        rotation = torch.tensor(
            ((cos_yaw, -sin_yaw), (sin_yaw, cos_yaw)),
            dtype=torch.float32,
        )
        return local_xy @ rotation.T

    def _make_marker(self) -> VisualizationMarkers:
        markers = {}
        for index, color in enumerate(TACTILE_3D_COLOR_STOPS):
            markers[f"force_{index}"] = sim_utils.CuboidCfg(
                size=(1.0, 1.0, 1.0),
                visual_material=sim_utils.PreviewSurfaceCfg(
                    diffuse_color=color,
                    emissive_color=tuple(0.15 * value for value in color),
                    roughness=0.35,
                ),
            )
        return VisualizationMarkers(
            VisualizationMarkersCfg(
                prim_path="/Visuals/r1_pro_tactile_pressure_3d",
                markers=markers,
            )
        )

    def _visualize_force_values(self, force_values: torch.Tensor) -> None:
        values = force_values.reshape(-1).to(dtype=torch.float32)
        heights = TACTILE_3D_MIN_HEIGHT + values * self._max_height

        translations = torch.zeros((values.numel(), 3), dtype=torch.float32)
        translations[:, 0:2] = self._panel_xy
        translations += self._origin
        translations[:, 2] += self._base_offset + heights * 0.5

        scales = torch.empty_like(translations)
        scales[:, 0] = self._taxel_size
        scales[:, 1] = self._taxel_size
        scales[:, 2] = heights

        marker_indices = torch.clamp(
            torch.floor(values * self._num_color_bins).to(dtype=torch.int64),
            min=0,
            max=self._num_color_bins - 1,
        )
        self._marker.visualize(
            translations=translations,
            scales=scales,
            marker_indices=marker_indices,
        )


def _make_markers() -> tuple[VisualizationMarkers, VisualizationMarkers, VisualizationMarkers, VisualizationMarkers]:
    current_marker_cfg = FRAME_MARKER_CFG.copy()
    current_marker_cfg.markers["frame"].scale = (args_cli.marker_scale, args_cli.marker_scale, args_cli.marker_scale)

    target_marker_cfg = FRAME_MARKER_CFG.copy()
    target_scale = args_cli.marker_scale * 1.35
    target_marker_cfg.markers["frame"].scale = (target_scale, target_scale, target_scale)

    left_current = VisualizationMarkers(current_marker_cfg.replace(prim_path="/Visuals/r1_pro_teleop_left_current"))
    left_target = VisualizationMarkers(target_marker_cfg.replace(prim_path="/Visuals/r1_pro_teleop_left_target"))
    right_current = VisualizationMarkers(current_marker_cfg.replace(prim_path="/Visuals/r1_pro_teleop_right_current"))
    right_target = VisualizationMarkers(target_marker_cfg.replace(prim_path="/Visuals/r1_pro_teleop_right_target"))
    return left_current, left_target, right_current, right_target


def _visualize_markers(
    markers: tuple[VisualizationMarkers, VisualizationMarkers, VisualizationMarkers, VisualizationMarkers] | None,
    unwrapped,
    left_target_b: torch.Tensor,
    right_target_b: torch.Tensor,
) -> None:
    if markers is None:
        return

    left_current, left_target, right_current, right_target = markers
    robot = unwrapped.robot
    root_pose_w = robot.data.root_pose_w
    left_pose_w = robot.data.body_pose_w[:, unwrapped.left_ee_body_idx]
    right_pose_w = robot.data.body_pose_w[:, unwrapped.right_ee_body_idx]
    left_target_pos_w, left_target_quat_w = combine_frame_transforms(
        root_pose_w[:, :3], root_pose_w[:, 3:7], left_target_b[:, :3], left_target_b[:, 3:7]
    )
    right_target_pos_w, right_target_quat_w = combine_frame_transforms(
        root_pose_w[:, :3], root_pose_w[:, 3:7], right_target_b[:, :3], right_target_b[:, 3:7]
    )

    left_current.visualize(left_pose_w[:, :3], left_pose_w[:, 3:7])
    right_current.visualize(right_pose_w[:, :3], right_pose_w[:, 3:7])
    left_target.visualize(left_target_pos_w, left_target_quat_w)
    right_target.visualize(right_target_pos_w, right_target_quat_w)


def _apply_delta_to_target(target: torch.Tensor, delta_pose: torch.Tensor) -> torch.Tensor:
    target_pos, target_quat = apply_delta_pose(target[:, :3], target[:, 3:7], delta_pose)
    return torch.cat((target_pos, target_quat), dim=-1)


def _apply_keyboard_delta(
    left_target: torch.Tensor,
    right_target: torch.Tensor,
    delta_pose: torch.Tensor,
    mode: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    if float(torch.linalg.norm(delta_pose).item()) <= 1.0e-9:
        return left_target, right_target
    if mode in ("left", "both"):
        left_target = _apply_delta_to_target(left_target, delta_pose)
    if mode in ("right", "both"):
        right_target = _apply_delta_to_target(right_target, delta_pose)
    return left_target, right_target


def _resolve_torso_joint_ids(unwrapped) -> list[int]:
    if not hasattr(unwrapped.cfg, "torso_joint_names"):
        raise RuntimeError(f"Task '{args_cli.task}' does not expose R1 Pro torso_joint_names.")
    torso_joint_names = list(unwrapped.cfg.torso_joint_names)
    if len(torso_joint_names) != 4:
        raise RuntimeError(f"Expected 4 R1 Pro torso joints, got {len(torso_joint_names)}: {torso_joint_names}")
    torso_joint_ids, resolved_names = unwrapped.robot.find_joints(torso_joint_names, preserve_order=True)
    if len(torso_joint_ids) != len(torso_joint_names):
        raise RuntimeError(f"Could not resolve all torso joints. Requested={torso_joint_names}, found={resolved_names}")
    return torso_joint_ids


def _clamp_torso_targets(unwrapped, torso_targets: torch.Tensor, torso_joint_ids: list[int]) -> torch.Tensor:
    limits = unwrapped.robot.data.soft_joint_pos_limits[:, torso_joint_ids]
    return torch.clamp(torso_targets, min=limits[..., 0], max=limits[..., 1])


def _current_torso_targets(unwrapped, torso_joint_ids: list[int]) -> torch.Tensor:
    torso_targets = unwrapped.robot.data.joint_pos[:, torso_joint_ids].clone()
    return _clamp_torso_targets(unwrapped, torso_targets, torso_joint_ids)


def _apply_torso_keyboard_delta(
    unwrapped,
    torso_targets: torch.Tensor,
    torso_joint_ids: list[int],
    delta_pose: torch.Tensor,
) -> torch.Tensor:
    torso_axes = torch.stack((delta_pose[:, 0], delta_pose[:, 1], delta_pose[:, 2], delta_pose[:, 3]), dim=-1)
    if float(torch.linalg.norm(torso_axes).item()) <= 1.0e-9:
        return torso_targets
    torso_targets = torso_targets + torch.sign(torso_axes) * args_cli.torso_step
    return _clamp_torso_targets(unwrapped, torso_targets, torso_joint_ids)


def _reset_env_and_targets(env, teleop_interface: Se3Keyboard, state: TeleopState) -> tuple[torch.Tensor, torch.Tensor]:
    env.reset()
    teleop_interface.reset()
    state.left_grip = OPEN_GRIPPER
    state.right_grip = OPEN_GRIPPER
    left_target, right_target = env.unwrapped._get_ee_poses_in_root_frame()
    return left_target.clone(), right_target.clone()


def _print_bindings() -> None:
    print("[INFO]: Keyboard bindings:")
    print("[INFO]:   W/S A/D Q/E: move current arm in x/y/z")
    print("[INFO]:   Z/X T/G C/V: rotate current arm around x/y/z")
    if args_cli.enable_torso_keys:
        print("[INFO]:   P: toggle arm/torso keyboard mode")
        print("[INFO]:   Torso mode: W/S A/D Q/E Z/X adjust torso_joint1-4")
    print("[INFO]:   N: cycle control mode left/right/both")
    print("[INFO]:   K: toggle gripper for current control mode")
    print("[INFO]:   R: reset environment and targets")
    print("[INFO]:   ESC: quit")


def _format_joint_angles(unwrapped, env_id: int = 0) -> str:
    joint_pos = unwrapped.robot.data.joint_pos[env_id].detach().cpu()
    return ", ".join(
        f"{name}={float(pos):.4f}" for name, pos in zip(unwrapped.robot.joint_names, joint_pos, strict=True)
    )


def _print_diagnostics(
    unwrapped,
    left_target: torch.Tensor,
    right_target: torch.Tensor,
    state: TeleopState,
    tactile_view: GripperTactile3DView | None = None,
) -> None:
    left_pose, right_pose = unwrapped._get_ee_poses_in_root_frame()
    left_error = torch.linalg.norm(left_pose[:, :3] - left_target[:, :3], dim=-1).mean()
    right_error = torch.linalg.norm(right_pose[:, :3] - right_target[:, :3], dim=-1).mean()
    keyboard_mode = "torso" if state.torso_mode else state.mode
    print(
        f"[INFO]: mode={keyboard_mode} left_error={float(left_error):.4f} m "
        f"right_error={float(right_error):.4f} m "
        f"left_grip={_grip_label(state.left_grip)} right_grip={_grip_label(state.right_grip)}"
    )
    if tactile_view is not None:
        diagnostics = tactile_view.diagnostics
        max_depth = diagnostics.max_penetration_depth
        mean_depth = diagnostics.mean_penetration_depth
        print(
            f"[INFO]: tactile_max_penetration_depth={max_depth:.6f} m "
            f"({max_depth * 1000.0:.3f} mm) "
            f"tactile_mean_penetration_depth={mean_depth:.6f} m "
            f"active_taxels={diagnostics.active_taxels} "
            f"tactile_max_force={diagnostics.max_force:.6f}"
        )
        for pad in diagnostics.pads:
            print(
                f"[INFO]: tactile_{pad.label}_max_penetration_depth={pad.max_penetration_depth:.6f} m "
                f"({pad.max_penetration_depth * 1000.0:.3f} mm) "
                f"tactile_{pad.label}_mean_penetration_depth={pad.mean_penetration_depth:.6f} m "
                f"active_taxels={pad.active_taxels} tactile_max_force={pad.max_force:.6f}"
            )
    if args_cli.print_joint_angles:
        print(f"[INFO]: env_0_joint_angles(rad): {_format_joint_angles(unwrapped)}")


def _configure_tactile_view_env_cfg(env_cfg) -> None:
    if args_cli.disable_tactile_pressure_view:
        return

    env_cfg.enable_r1_pro_gripper_tactile = True


def main() -> int:
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
    )
    _configure_tactile_view_env_cfg(env_cfg)
    task_has_torso_ik = bool(getattr(env_cfg, "include_torso_in_ik", False))
    if args_cli.include_torso_in_ik:
        if not hasattr(env_cfg, "torso_joint_names") or not hasattr(env_cfg, "include_torso_in_ik"):
            raise RuntimeError(f"Task '{args_cli.task}' does not expose R1 Pro torso IK configuration.")
        if not task_has_torso_ik:
            env_cfg.include_torso_in_ik = True
    effective_torso_ik = bool(getattr(env_cfg, "include_torso_in_ik", False))
    if args_cli.enable_torso_keys and effective_torso_ik:
        raise RuntimeError(
            "--enable_torso_keys cannot be used when the selected task already has torso IK enabled, "
            "because both paths would write torso joint targets."
        )

    env = gym.make(args_cli.task, cfg=env_cfg)
    unwrapped = env.unwrapped
    tactile_view = None

    try:
        if not hasattr(unwrapped, "_get_ee_poses_in_root_frame"):
            raise RuntimeError(f"Task '{args_cli.task}' does not expose R1 Pro end-effector poses.")
        if getattr(unwrapped.cfg, "control_mode", None) != "ik":
            raise RuntimeError(f"Task '{args_cli.task}' must use R1 Pro IK control_mode.")
        if unwrapped.cfg.action_space != 16:
            raise RuntimeError(f"Task '{args_cli.task}' must use the 16D R1 Pro IK action interface.")

        state = TeleopState()
        torso_joint_ids = None
        torso_targets = None
        if args_cli.enable_torso_keys:
            torso_joint_ids = _resolve_torso_joint_ids(unwrapped)
            torso_targets = _current_torso_targets(unwrapped, torso_joint_ids)
        teleop_interface = Se3Keyboard(
            Se3KeyboardCfg(
                gripper_term=False,
                pos_sensitivity=args_cli.pos_step,
                rot_sensitivity=args_cli.rot_step,
                sim_device=unwrapped.device,
            )
        )
        teleop_interface.add_callback("N", state.cycle_mode)
        teleop_interface.add_callback("K", state.toggle_gripper)
        teleop_interface.add_callback("R", state.request_reset)
        teleop_interface.add_callback("ESCAPE", state.request_quit)
        if args_cli.enable_torso_keys:
            teleop_interface.add_callback("P", state.toggle_torso_mode)

        markers = None if args_cli.disable_markers else _make_markers()
        if not args_cli.disable_tactile_pressure_view:
            print("[INFO]: Creating tabletop tactile pressure 3D panel...", flush=True)
            tactile_view = GripperTactile3DView(
                origin=tuple(args_cli.tactile_3d_origin),
                yaw=args_cli.tactile_3d_yaw,
                pressure_scale=args_cli.tactile_pressure_scale,
                max_height=args_cli.tactile_3d_max_height,
                taxel_size=args_cli.tactile_3d_taxel_size,
                base_offset=TACTILE_3D_BASE_OFFSET,
            )
        left_target, right_target = _reset_env_and_targets(env, teleop_interface, state)
        if torso_joint_ids is not None:
            torso_targets = _current_torso_targets(unwrapped, torso_joint_ids)
        _visualize_markers(markers, unwrapped, left_target, right_target)
        if tactile_view is not None:
            tactile_view.update(unwrapped.scene)

        print(f"[INFO]: Gym observation space: {env.observation_space}")
        print(f"[INFO]: Gym action space: {env.action_space}")
        print(f"[INFO]: Task: {args_cli.task}")
        print(f"[INFO]: Position step: {args_cli.pos_step:.4f} m, rotation step: {args_cli.rot_step:.4f} rad")
        print(f"[INFO]: Torso IK: enabled={effective_torso_ik}")
        print(f"[INFO]: Torso keys: enabled={args_cli.enable_torso_keys}, step={args_cli.torso_step:.4f} rad")
        print(f"[INFO]: Tactile pressure view: enabled={tactile_view is not None}")
        print(f"[INFO]: Control mode: {state.mode}")
        _print_bindings()

        count = 0
        while simulation_app.is_running() and not state.quit_requested:
            if state.reset_requested:
                left_target, right_target = _reset_env_and_targets(env, teleop_interface, state)
                if torso_joint_ids is not None:
                    torso_targets = _current_torso_targets(unwrapped, torso_joint_ids)
                state.reset_requested = False
                _visualize_markers(markers, unwrapped, left_target, right_target)
                if tactile_view is not None:
                    tactile_view.update(unwrapped.scene)
                print("[INFO]: Environment reset complete.", flush=True)
                continue

            delta_pose = teleop_interface.advance().view(1, 6)
            if state.quit_requested or state.reset_requested:
                continue
            if state.torso_mode:
                if torso_joint_ids is None or torso_targets is None:
                    raise RuntimeError("Torso keyboard mode is active but torso control is not initialized.")
                torso_targets = _apply_torso_keyboard_delta(unwrapped, torso_targets, torso_joint_ids, delta_pose)
            else:
                left_target, right_target = _apply_keyboard_delta(left_target, right_target, delta_pose, state.mode)
            actions = _make_action(left_target, state.left_grip, right_target, state.right_grip)
            extra_joint_targets = None
            if torso_joint_ids is not None and torso_targets is not None:
                extra_joint_targets = (torso_targets, torso_joint_ids)
            _step_without_auto_reset(env, actions, extra_joint_targets=extra_joint_targets)
            _visualize_markers(markers, unwrapped, left_target, right_target)
            if tactile_view is not None and count % args_cli.tactile_pressure_update_interval == 0:
                tactile_view.update(unwrapped.scene)

            if args_cli.print_interval > 0 and count % args_cli.print_interval == 0:
                _print_diagnostics(unwrapped, left_target, right_target, state, tactile_view)
            count += 1

        return 0
    finally:
        if tactile_view is not None:
            tactile_view.close()
        env.close()


if __name__ == "__main__":
    try:
        with torch.inference_mode():
            raise SystemExit(main())
    finally:
        simulation_app.close()
