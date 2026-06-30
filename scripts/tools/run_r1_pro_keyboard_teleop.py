# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Keyboard teleoperation for the Galaxea R1 Pro bimanual IK task.

This script drives the existing 16D R1 Pro IK action interface with keyboard
delta commands. It defaults to the BlocksStackEasy scene so the grippers can
interact with the tabletop blocks through normal physics contact.

.. code-block:: bash

    python scripts/tools/run_r1_pro_keyboard_teleop.py --num_envs 1 --device cuda:0

"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher


DEFAULT_TASK_NAME = "Assembly-R1Pro-BlocksStackEasy-IK-Direct-v0"
OPEN_GRIPPER = 1.0
CLOSE_GRIPPER = -1.0
CONTROL_MODES = ("left", "right", "both")


parser = argparse.ArgumentParser(description="Run keyboard teleoperation for R1 Pro bimanual IK.")
parser.add_argument("--task", type=str, default=DEFAULT_TASK_NAME, help="IK task to teleoperate.")
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
    "--print_interval",
    type=int,
    default=60,
    help="Print end-effector tracking diagnostics every N control loops. Use 0 to disable.",
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
if args_cli.marker_scale <= 0.0:
    raise ValueError("--marker_scale must be positive.")
if args_cli.print_interval < 0:
    raise ValueError("--print_interval must be non-negative.")

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch

import isaaclab_tasks  # noqa: F401
from isaaclab.devices import Se3Keyboard, Se3KeyboardCfg
from isaaclab.markers import VisualizationMarkers
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.utils.math import apply_delta_pose, combine_frame_transforms
from isaaclab_tasks.utils import parse_env_cfg

import assembly_benchmark.tasks  # noqa: F401


class TeleopState:
    """Mutable keyboard callback state."""

    def __init__(self) -> None:
        self.mode_index = 0
        self.left_grip = OPEN_GRIPPER
        self.right_grip = OPEN_GRIPPER
        self.reset_requested = False
        self.quit_requested = False

    @property
    def mode(self) -> str:
        return CONTROL_MODES[self.mode_index]

    def cycle_mode(self) -> None:
        self.mode_index = (self.mode_index + 1) % len(CONTROL_MODES)
        print(f"[INFO]: Control mode: {self.mode}", flush=True)

    def toggle_gripper(self) -> None:
        if self.mode == "left":
            self.left_grip = _toggle_grip(self.left_grip)
        elif self.mode == "right":
            self.right_grip = _toggle_grip(self.right_grip)
        else:
            next_value = (
                OPEN_GRIPPER
                if self.left_grip == CLOSE_GRIPPER and self.right_grip == CLOSE_GRIPPER
                else CLOSE_GRIPPER
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


def _make_action(left_pose: torch.Tensor, left_grip: float, right_pose: torch.Tensor, right_grip: float) -> torch.Tensor:
    actions = torch.zeros((1, 16), dtype=torch.float32, device=left_pose.device)
    actions[:, 0:7] = left_pose
    actions[:, 7] = left_grip
    actions[:, 8:15] = right_pose
    actions[:, 15] = right_grip
    return actions


def _step_without_auto_reset(env, actions: torch.Tensor):
    """Step the DirectRLEnv physics path without applying its automatic reset."""
    unwrapped = env.unwrapped
    actions = actions.to(unwrapped.device)
    unwrapped._pre_physics_step(actions)
    is_rendering = unwrapped.sim.has_gui() or unwrapped.sim.has_rtx_sensors()

    for _ in range(unwrapped.cfg.decimation):
        unwrapped._sim_step_counter += 1
        unwrapped._apply_action()
        unwrapped.scene.write_data_to_sim()
        unwrapped.sim.step(render=False)
        if unwrapped._sim_step_counter % unwrapped.cfg.sim.render_interval == 0 and is_rendering:
            unwrapped.sim.render()
        unwrapped.scene.update(dt=unwrapped.physics_dt)

    unwrapped.obs_buf = unwrapped._get_observations()
    reward = unwrapped._get_rewards()
    terminated, truncated = unwrapped._get_dones()
    return unwrapped.obs_buf, reward, terminated, truncated, unwrapped.extras


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
    print("[INFO]:   N: cycle control mode left/right/both")
    print("[INFO]:   K: toggle gripper for current control mode")
    print("[INFO]:   R: reset environment and targets")
    print("[INFO]:   ESC: quit")


def _print_diagnostics(unwrapped, left_target: torch.Tensor, right_target: torch.Tensor, state: TeleopState) -> None:
    left_pose, right_pose = unwrapped._get_ee_poses_in_root_frame()
    left_error = torch.linalg.norm(left_pose[:, :3] - left_target[:, :3], dim=-1).mean()
    right_error = torch.linalg.norm(right_pose[:, :3] - right_target[:, :3], dim=-1).mean()
    print(
        f"[INFO]: mode={state.mode} left_error={float(left_error):.4f} m "
        f"right_error={float(right_error):.4f} m "
        f"left_grip={_grip_label(state.left_grip)} right_grip={_grip_label(state.right_grip)}"
    )


def main() -> int:
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
    )
    env = gym.make(args_cli.task, cfg=env_cfg)
    unwrapped = env.unwrapped

    try:
        if not hasattr(unwrapped, "_get_ee_poses_in_root_frame"):
            raise RuntimeError(f"Task '{args_cli.task}' does not expose R1 Pro end-effector poses.")
        if getattr(unwrapped.cfg, "control_mode", None) != "ik":
            raise RuntimeError(f"Task '{args_cli.task}' must use R1 Pro IK control_mode.")
        if unwrapped.cfg.action_space != 16:
            raise RuntimeError(f"Task '{args_cli.task}' must use the 16D R1 Pro IK action interface.")

        state = TeleopState()
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

        markers = None if args_cli.disable_markers else _make_markers()
        left_target, right_target = _reset_env_and_targets(env, teleop_interface, state)
        _visualize_markers(markers, unwrapped, left_target, right_target)

        print(f"[INFO]: Gym observation space: {env.observation_space}")
        print(f"[INFO]: Gym action space: {env.action_space}")
        print(f"[INFO]: Task: {args_cli.task}")
        print(f"[INFO]: Position step: {args_cli.pos_step:.4f} m, rotation step: {args_cli.rot_step:.4f} rad")
        print(f"[INFO]: Control mode: {state.mode}")
        _print_bindings()

        count = 0
        while simulation_app.is_running() and not state.quit_requested:
            if state.reset_requested:
                left_target, right_target = _reset_env_and_targets(env, teleop_interface, state)
                state.reset_requested = False
                _visualize_markers(markers, unwrapped, left_target, right_target)
                print("[INFO]: Environment reset complete.", flush=True)
                continue

            delta_pose = teleop_interface.advance().view(1, 6)
            if state.quit_requested or state.reset_requested:
                continue
            left_target, right_target = _apply_keyboard_delta(left_target, right_target, delta_pose, state.mode)
            actions = _make_action(left_target, state.left_grip, right_target, state.right_grip)
            _step_without_auto_reset(env, actions)
            _visualize_markers(markers, unwrapped, left_target, right_target)

            if args_cli.print_interval > 0 and count % args_cli.print_interval == 0:
                _print_diagnostics(unwrapped, left_target, right_target, state)
            count += 1

        return 0
    finally:
        env.close()


if __name__ == "__main__":
    try:
        with torch.inference_mode():
            raise SystemExit(main())
    finally:
        simulation_app.close()
