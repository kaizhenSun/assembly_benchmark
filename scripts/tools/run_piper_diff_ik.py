# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Measure Piper single-arm Differential IK accuracy in an isolated scene.

The tool loads the reusable Piper + Pika2 asset without Fabrica parts, commands
root-frame end-effector poses through the benchmark's single-arm controller,
and reports Cartesian, orientation, joint-tracking, and torque metrics.

.. code-block:: bash

    python scripts/tools/run_piper_diff_ik.py --num_envs 1 --device cuda:0

    python scripts/tools/run_piper_diff_ik.py --teleop --num_envs 1 --device cuda:0

    python scripts/tools/run_piper_diff_ik.py --num_envs 1 --device cuda:0 \
        --headless --max_steps 1920 --fast_exit

"""

from __future__ import annotations

import argparse
import copy
import math
import os
import sys
from dataclasses import dataclass

from isaaclab.app import AppLauncher

TARGET_COUNT = 8

parser = argparse.ArgumentParser(description="Run Piper single-arm Differential IK accuracy diagnostics.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of vectorized Piper environments.")
parser.add_argument(
    "--teleop",
    action="store_true",
    help="Use GUI keyboard teleoperation instead of the automatic target sequence.",
)
parser.add_argument(
    "--max_steps",
    type=int,
    default=0,
    help="Maximum diagnostic steps. Use 0 to run until the app closes.",
)
parser.add_argument("--hold_steps", type=int, default=240, help="Simulation steps to hold each IK target.")
parser.add_argument("--settle_steps", type=int, default=60, help="Steps to settle at the default pose.")
parser.add_argument(
    "--metric_warmup_steps",
    type=int,
    default=120,
    help="Steps skipped after each target switch before metrics are accumulated.",
)
parser.add_argument(
    "--position_tolerance",
    type=float,
    default=0.005,
    help="Maximum allowed post-warmup end-effector position error in metres.",
)
parser.add_argument(
    "--orientation_tolerance_deg",
    type=float,
    default=3.0,
    help="Maximum allowed post-warmup orientation error in degrees.",
)
parser.add_argument(
    "--joint_tolerance",
    type=float,
    default=0.05,
    help="Maximum allowed post-warmup arm joint target error in radians.",
)
parser.add_argument("--enable_gravity", action="store_true", help="Enable gravity on Piper links.")
parser.add_argument("--arm_stiffness", type=float, default=None, help="Override the arm actuator stiffness.")
parser.add_argument("--arm_damping", type=float, default=None, help="Override the arm actuator damping.")
parser.add_argument("--arm_effort", type=float, default=None, help="Override the arm actuator effort limit.")
parser.add_argument("--arm_armature", type=float, default=None, help="Override the arm actuator armature.")
parser.add_argument("--gripper_stiffness", type=float, default=None, help="Override the gripper actuator stiffness.")
parser.add_argument("--gripper_damping", type=float, default=None, help="Override the gripper actuator damping.")
parser.add_argument("--gripper_effort", type=float, default=None, help="Override the gripper actuator effort limit.")
parser.add_argument("--gripper_armature", type=float, default=None, help="Override the gripper actuator armature.")
parser.add_argument(
    "--solver_position_iterations",
    type=int,
    default=None,
    help="Override articulation solver position iterations.",
)
parser.add_argument(
    "--solver_velocity_iterations",
    type=int,
    default=None,
    help="Override articulation solver velocity iterations.",
)
parser.add_argument(
    "--disable_fabric",
    action="store_true",
    help="Disable Fabric and use USD I/O for debugging compatibility.",
)
parser.add_argument("--marker_scale", type=float, default=0.09, help="Current/target frame marker scale.")
parser.add_argument("--disable_markers", action="store_true", help="Disable frame marker visualization.")
parser.add_argument(
    "--pos_step", type=float, default=0.002, help="Teleoperation translation increment per loop in metres."
)
parser.add_argument(
    "--rot_step", type=float, default=0.015, help="Teleoperation rotation increment per loop in radians."
)
parser.add_argument(
    "--print_interval",
    type=int,
    default=60,
    help="Print live errors every N diagnostic steps. Use 0 to disable.",
)
parser.add_argument(
    "--fast_exit",
    action="store_true",
    help="Exit immediately after a finite diagnostic, avoiding slow Kit shutdown.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

if args_cli.num_envs <= 0:
    raise ValueError("--num_envs must be positive.")
if args_cli.teleop and args_cli.num_envs != 1:
    raise ValueError("Keyboard teleoperation currently supports only --num_envs 1.")
if args_cli.teleop and getattr(args_cli, "headless", False):
    raise ValueError("Keyboard teleoperation requires a GUI window. Remove --headless.")
if args_cli.max_steps < 0:
    raise ValueError("--max_steps must be non-negative.")
if args_cli.hold_steps <= 0:
    raise ValueError("--hold_steps must be positive.")
if args_cli.settle_steps < 0:
    raise ValueError("--settle_steps must be non-negative.")
if not 0 <= args_cli.metric_warmup_steps < args_cli.hold_steps:
    raise ValueError("--metric_warmup_steps must be non-negative and smaller than --hold_steps.")
if args_cli.print_interval < 0:
    raise ValueError("--print_interval must be non-negative.")
for name in (
    "position_tolerance",
    "orientation_tolerance_deg",
    "joint_tolerance",
    "marker_scale",
    "pos_step",
    "rot_step",
):
    if getattr(args_cli, name) <= 0.0:
        raise ValueError(f"--{name} must be positive.")
for name in (
    "arm_stiffness",
    "arm_damping",
    "arm_effort",
    "arm_armature",
    "gripper_stiffness",
    "gripper_damping",
    "gripper_effort",
    "gripper_armature",
):
    value = getattr(args_cli, name)
    if value is not None and value <= 0.0:
        raise ValueError(f"--{name} must be positive.")
if args_cli.solver_position_iterations is not None and args_cli.solver_position_iterations <= 0:
    raise ValueError("--solver_position_iterations must be positive.")
if args_cli.solver_velocity_iterations is not None and args_cli.solver_velocity_iterations <= 0:
    raise ValueError("--solver_velocity_iterations must be positive.")

cycle_steps = TARGET_COUNT * args_cli.hold_steps
if (
    not args_cli.teleop
    and args_cli.max_steps > 0
    and (args_cli.max_steps < cycle_steps or args_cli.max_steps % cycle_steps != 0)
):
    raise ValueError(
        f"Finite diagnostics must contain complete {cycle_steps}-step target cycles; got --max_steps "
        f"{args_cli.max_steps}."
    )

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch
from assembly_benchmark.controllers import SingleArmDifferentialIKController
from assembly_benchmark.robots.piper import (
    PIPER_ARM_JOINT_NAMES,
    PIPER_CFG,
    PIPER_DEFAULT_GRIPPER_OPENING,
    PIPER_EE_LINK_NAME,
    PIPER_GRIPPER_CLOSED_JOINT_POS,
    PIPER_GRIPPER_JOINT_NAMES,
    PIPER_GRIPPER_MAX,
    PIPER_GRIPPER_MIN,
    PIPER_GRIPPER_OPEN_JOINT_POS,
    PIPER_IK_LINK_NAME,
)

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.devices import Se3Keyboard, Se3KeyboardCfg
from isaaclab.markers import VisualizationMarkers
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import (
    apply_delta_pose,
    combine_frame_transforms,
    quat_from_euler_xyz,
    quat_mul,
    subtract_frame_transforms,
)


@dataclass(frozen=True)
class TargetSpec:
    """One pose offset relative to the settled Piper home end-effector."""

    label: str
    position_offset: tuple[float, float, float]
    rpy_offset_deg: tuple[float, float, float]


@dataclass
class TargetMetrics:
    """Post-warmup statistics for one repeated IK target."""

    count: int = 0
    position_error_sum: float = 0.0
    orientation_error_sum: float = 0.0
    joint_error_sum: float = 0.0
    max_position_error: float = 0.0
    max_orientation_error: float = 0.0
    max_joint_error: float = 0.0
    max_torque: float = 0.0
    max_torque_ratio: float = 0.0

    def update(
        self,
        position_error: torch.Tensor,
        orientation_error: torch.Tensor,
        joint_error: torch.Tensor,
        torque: torch.Tensor,
        torque_ratio: torch.Tensor,
    ) -> None:
        """Accumulate one vectorized simulation sample."""
        self.count += 1
        self.position_error_sum += float(position_error.mean())
        self.orientation_error_sum += float(orientation_error.mean())
        self.joint_error_sum += float(joint_error.mean())
        self.max_position_error = max(self.max_position_error, float(position_error.max()))
        self.max_orientation_error = max(self.max_orientation_error, float(orientation_error.max()))
        self.max_joint_error = max(self.max_joint_error, float(joint_error.max()))
        self.max_torque = max(self.max_torque, float(torque.max()))
        self.max_torque_ratio = max(self.max_torque_ratio, float(torque_ratio.max()))


class TeleopState:
    """Mutable keyboard callback state for Piper teleoperation."""

    def __init__(self, default_gripper_action: float) -> None:
        self.default_gripper_action = default_gripper_action
        self.gripper_action = default_gripper_action
        self.reset_requested = False
        self.quit_requested = False

    def toggle_gripper(self) -> None:
        self.gripper_action = 1.0 if self.gripper_action <= -1.0 else -1.0
        state = "open" if self.gripper_action > 0.0 else "closed"
        print(f"[INFO]: Gripper state: {state}", flush=True)

    def request_reset(self) -> None:
        self.reset_requested = True
        print("[INFO]: Reset requested.", flush=True)

    def request_quit(self) -> None:
        self.quit_requested = True
        print("[INFO]: Quit requested.", flush=True)

    def reset(self) -> None:
        self.gripper_action = self.default_gripper_action
        self.reset_requested = False


TARGET_SPECS = (
    TargetSpec("home", (0.0, 0.0, 0.0), (0.0, 0.0, 0.0)),
    TargetSpec("lift 80 mm", (0.0, 0.0, 0.08), (0.0, 0.0, 0.0)),
    TargetSpec("forward 60 mm", (0.06, 0.0, 0.03), (0.0, 0.0, 0.0)),
    TargetSpec("lateral 50 mm", (0.0, 0.05, 0.04), (0.0, 0.0, 0.0)),
    TargetSpec("local roll +10 deg", (0.0, 0.0, 0.0), (10.0, 0.0, 0.0)),
    TargetSpec("local pitch +10 deg", (0.0, 0.0, 0.0), (0.0, 10.0, 0.0)),
    TargetSpec("local yaw +10 deg", (0.0, 0.0, 0.0), (0.0, 0.0, 10.0)),
    TargetSpec("combined 6D", (0.04, -0.03, 0.05), (-8.0, 8.0, -8.0)),
)
assert len(TARGET_SPECS) == TARGET_COUNT


@configclass
class PiperDiffIKSceneCfg(InteractiveSceneCfg):
    """Minimal scene containing only the fixed-base Piper + Pika2 robot."""

    ground = AssetBaseCfg(
        prim_path="/World/defaultGroundPlane",
        spawn=sim_utils.GroundPlaneCfg(),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
    )
    dome_light = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75)),
    )
    robot = PIPER_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")


def _override_actuator(actuator_cfg, **kwargs) -> None:
    for key, value in kwargs.items():
        if value is not None:
            setattr(actuator_cfg, key, value)


def _build_scene_cfg() -> PiperDiffIKSceneCfg:
    scene_cfg = PiperDiffIKSceneCfg(num_envs=args_cli.num_envs, env_spacing=2.0, replicate_physics=True)
    robot_cfg = copy.deepcopy(PIPER_CFG).replace(prim_path="{ENV_REGEX_NS}/Robot")
    robot_cfg.spawn.rigid_props.disable_gravity = not args_cli.enable_gravity
    if args_cli.solver_position_iterations is not None:
        robot_cfg.spawn.articulation_props.solver_position_iteration_count = args_cli.solver_position_iterations
    if args_cli.solver_velocity_iterations is not None:
        robot_cfg.spawn.articulation_props.solver_velocity_iteration_count = args_cli.solver_velocity_iterations
    _override_actuator(
        robot_cfg.actuators["arm"],
        stiffness=args_cli.arm_stiffness,
        damping=args_cli.arm_damping,
        effort_limit_sim=args_cli.arm_effort,
        armature=args_cli.arm_armature,
    )
    _override_actuator(
        robot_cfg.actuators["gripper"],
        stiffness=args_cli.gripper_stiffness,
        damping=args_cli.gripper_damping,
        effort_limit_sim=args_cli.gripper_effort,
        armature=args_cli.gripper_armature,
    )
    scene_cfg.robot = robot_cfg
    return scene_cfg


def _reset_robot(sim: sim_utils.SimulationContext, scene: InteractiveScene) -> None:
    robot = scene["robot"]
    joint_pos = robot.data.default_joint_pos.clone()
    joint_vel = robot.data.default_joint_vel.clone()
    root_state = robot.data.default_root_state.clone()
    root_state[:, :3] += scene.env_origins
    robot.write_root_pose_to_sim(root_state[:, :7])
    robot.write_root_velocity_to_sim(root_state[:, 7:])
    robot.write_joint_state_to_sim(joint_pos, joint_vel)
    robot.set_joint_position_target(joint_pos)
    robot.reset()
    scene.write_data_to_sim()
    sim.step()
    scene.update(sim.get_physics_dt())


def _make_controller(sim: sim_utils.SimulationContext, scene: InteractiveScene) -> SingleArmDifferentialIKController:
    robot = scene["robot"]
    return SingleArmDifferentialIKController(
        robot=robot,
        arm_joint_names=PIPER_ARM_JOINT_NAMES,
        gripper_joint_names=PIPER_GRIPPER_JOINT_NAMES,
        ee_link_name=PIPER_EE_LINK_NAME,
        ik_link_name=PIPER_IK_LINK_NAME,
        gripper_min=PIPER_GRIPPER_MIN,
        gripper_max=PIPER_GRIPPER_MAX,
        num_envs=scene.num_envs,
        device=robot.device,
        control_dt=sim.get_physics_dt(),
        gripper_joint_closed_positions=PIPER_GRIPPER_CLOSED_JOINT_POS,
        gripper_joint_open_positions=PIPER_GRIPPER_OPEN_JOINT_POS,
    )


def _ee_pose_in_root(robot, ee_body_idx: int) -> torch.Tensor:
    root_pose_w = robot.data.root_pose_w
    ee_pose_w = robot.data.body_pose_w[:, ee_body_idx]
    position, quaternion = subtract_frame_transforms(
        root_pose_w[:, :3], root_pose_w[:, 3:7], ee_pose_w[:, :3], ee_pose_w[:, 3:7]
    )
    return torch.cat((position, quaternion), dim=-1)


def _make_targets(home_pose: torch.Tensor) -> tuple[tuple[str, torch.Tensor], ...]:
    targets = []
    for spec in TARGET_SPECS:
        target = home_pose.clone()
        target[:, :3] += torch.tensor(spec.position_offset, dtype=torch.float32, device=home_pose.device)
        rpy = torch.tensor(spec.rpy_offset_deg, dtype=torch.float32, device=home_pose.device) * (math.pi / 180.0)
        delta_quat = quat_from_euler_xyz(rpy[0:1], rpy[1:2], rpy[2:3]).expand(home_pose.shape[0], -1)
        target[:, 3:7] = quat_mul(home_pose[:, 3:7], delta_quat)
        targets.append((spec.label, target))
    return tuple(targets)


def _make_markers() -> tuple[VisualizationMarkers, VisualizationMarkers]:
    current_cfg = FRAME_MARKER_CFG.copy()
    current_cfg.markers["frame"].scale = (args_cli.marker_scale,) * 3
    target_cfg = FRAME_MARKER_CFG.copy()
    target_scale = 1.35 * args_cli.marker_scale
    target_cfg.markers["frame"].scale = (target_scale,) * 3
    current = VisualizationMarkers(current_cfg.replace(prim_path="/Visuals/piper_current"))
    target = VisualizationMarkers(target_cfg.replace(prim_path="/Visuals/piper_target"))
    return current, target


def _visualize_markers(
    markers: tuple[VisualizationMarkers, VisualizationMarkers] | None,
    robot,
    ee_body_idx: int,
    target_b: torch.Tensor,
) -> None:
    if markers is None:
        return
    current_marker, target_marker = markers
    root_pose_w = robot.data.root_pose_w
    ee_pose_w = robot.data.body_pose_w[:, ee_body_idx]
    target_pos_w, target_quat_w = combine_frame_transforms(
        root_pose_w[:, :3], root_pose_w[:, 3:7], target_b[:, :3], target_b[:, 3:7]
    )
    current_marker.visualize(ee_pose_w[:, :3], ee_pose_w[:, 3:7])
    target_marker.visualize(target_pos_w, target_quat_w)


def _orientation_error(current: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    current = current / torch.linalg.vector_norm(current, dim=-1, keepdim=True).clamp_min(1.0e-8)
    target = target / torch.linalg.vector_norm(target, dim=-1, keepdim=True).clamp_min(1.0e-8)
    dot = torch.sum(current * target, dim=-1).abs().clamp(max=1.0)
    return 2.0 * torch.acos(dot)


def _apply_keyboard_delta(target: torch.Tensor, delta_pose: torch.Tensor) -> torch.Tensor:
    if float(torch.linalg.vector_norm(delta_pose).item()) <= 1.0e-9:
        return target
    position, quaternion = apply_delta_pose(target[:, :3], target[:, 3:7], delta_pose)
    return torch.cat((position, quaternion), dim=-1)


def _arm_diagnostics(robot, arm_joint_ids: list[int]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    joint_error = torch.abs(
        robot.data.joint_pos[:, arm_joint_ids] - robot.data.joint_pos_target[:, arm_joint_ids]
    ).amax(dim=-1)
    torque = torch.abs(robot.data.applied_torque[:, arm_joint_ids]).amax(dim=-1)
    effort_limits = robot.data.joint_effort_limits[:, arm_joint_ids]
    torque_ratio = torch.where(
        effort_limits > 1.0e-6,
        torch.abs(robot.data.applied_torque[:, arm_joint_ids]) / effort_limits.clamp_min(1.0e-6),
        torch.zeros_like(effort_limits),
    ).amax(dim=-1)
    return joint_error, torque, torque_ratio


def _format_joint_angles(robot, arm_joint_ids: list[int], env_id: int = 0) -> str:
    positions = robot.data.joint_pos[env_id, arm_joint_ids].detach().cpu()
    names = [robot.joint_names[index] for index in arm_joint_ids]
    return ", ".join(f"{name}={position:.4f}" for name, position in zip(names, positions, strict=True))


def _mean_joint_property(robot, property_tensor: torch.Tensor, joint_ids: list[int]) -> float:
    return float(property_tensor[0, joint_ids].mean())


def _print_metadata(robot, controller: SingleArmDifferentialIKController) -> None:
    print(
        "[INFO]: Piper metadata: "
        f"fixed_base={robot.is_fixed_base}, joints={robot.num_joints}, bodies={robot.num_bodies}, "
        f"jacobians={tuple(robot.root_physx_view.get_jacobians().shape)}"
    )
    print(
        "[INFO]: IK configuration: "
        f"ee={PIPER_EE_LINK_NAME}, ik={PIPER_IK_LINK_NAME}, arm_joints={controller.arm_joint_ids}, "
        f"gripper_joints={controller.gripper_joint_ids}"
    )
    print(
        "[INFO]: Actuators: "
        f"arm(k={_mean_joint_property(robot, robot.data.default_joint_stiffness, controller.arm_joint_ids):.1f}, "
        f"d={_mean_joint_property(robot, robot.data.default_joint_damping, controller.arm_joint_ids):.1f}, "
        f"effort={_mean_joint_property(robot, robot.data.joint_effort_limits, controller.arm_joint_ids):.1f}) "
        "gripper("
        f"k={_mean_joint_property(robot, robot.data.default_joint_stiffness, controller.gripper_joint_ids):.1f}, "
        f"d={_mean_joint_property(robot, robot.data.default_joint_damping, controller.gripper_joint_ids):.1f}, "
        f"effort={_mean_joint_property(robot, robot.data.joint_effort_limits, controller.gripper_joint_ids):.1f})"
    )


def _print_teleop_bindings() -> None:
    print("[INFO]: Keyboard bindings:")
    print("[INFO]:   W/S A/D Q/E: move target along root-frame x/y/z")
    print("[INFO]:   Z/X T/G C/V: rotate target around x/y/z")
    print("[INFO]:   K: toggle Pika2 open/closed")
    print("[INFO]:   R: reset Piper and target")
    print("[INFO]:   ESC: quit")


def _print_teleop_diagnostics(
    robot,
    controller: SingleArmDifferentialIKController,
    target_pose: torch.Tensor,
    state: TeleopState,
) -> None:
    current_pose = _ee_pose_in_root(robot, controller.ee_body_idx)
    position_error_xyz = target_pose[:, :3] - current_pose[:, :3]
    position_error = torch.linalg.vector_norm(position_error_xyz, dim=-1)
    orientation_error = _orientation_error(current_pose[:, 3:7], target_pose[:, 3:7])
    joint_error, torque, torque_ratio = _arm_diagnostics(robot, controller.arm_joint_ids)
    error_mm = position_error_xyz[0] * 1000.0
    target = target_pose[0, :3]
    current = current_pose[0, :3]
    print(
        "[TELEOP]: "
        f"target_xyz=({float(target[0]):.4f}, {float(target[1]):.4f}, {float(target[2]):.4f}) m "
        f"current_xyz=({float(current[0]):.4f}, {float(current[1]):.4f}, {float(current[2]):.4f}) m "
        f"position_error_xyz=({float(error_mm[0]):+.3f}, {float(error_mm[1]):+.3f}, "
        f"{float(error_mm[2]):+.3f}) mm "
        f"position_error={float(position_error[0]) * 1000.0:.3f} mm "
        f"orientation_error={float(orientation_error[0]):.6f} rad "
        f"({math.degrees(float(orientation_error[0])):.3f} deg) "
        f"max_joint_error={float(joint_error[0]):.6f} rad "
        f"max_torque={float(torque[0]):.3f} Nm torque_ratio={float(torque_ratio[0]):.3f} "
        f"gripper_action={state.gripper_action:+.3f}"
    )


def _print_target_summaries(metrics: list[TargetMetrics]) -> None:
    for spec, metric in zip(TARGET_SPECS, metrics, strict=True):
        if metric.count == 0:
            print(f"[TARGET_SUMMARY]: target='{spec.label}' samples=0")
            continue
        print(
            f"[TARGET_SUMMARY]: target='{spec.label}' samples={metric.count} "
            f"mean_position_error={metric.position_error_sum / metric.count:.6f} m "
            f"max_position_error={metric.max_position_error:.6f} m "
            f"mean_orientation_error={metric.orientation_error_sum / metric.count:.6f} rad "
            f"({math.degrees(metric.orientation_error_sum / metric.count):.3f} deg) "
            f"max_orientation_error={metric.max_orientation_error:.6f} rad "
            f"({math.degrees(metric.max_orientation_error):.3f} deg) "
            f"mean_max_joint_error={metric.joint_error_sum / metric.count:.6f} rad "
            f"max_joint_error={metric.max_joint_error:.6f} rad "
            f"max_torque={metric.max_torque:.3f} Nm max_torque_ratio={metric.max_torque_ratio:.3f}"
        )


def _print_global_summary(metrics: list[TargetMetrics]) -> None:
    sample_count = sum(metric.count for metric in metrics)
    if sample_count == 0:
        print("[GLOBAL_SUMMARY]: samples=0")
        return
    position_error_sum = sum(metric.position_error_sum for metric in metrics)
    orientation_error_sum = sum(metric.orientation_error_sum for metric in metrics)
    joint_error_sum = sum(metric.joint_error_sum for metric in metrics)
    max_position_error = max(metric.max_position_error for metric in metrics)
    max_orientation_error = max(metric.max_orientation_error for metric in metrics)
    max_joint_error = max(metric.max_joint_error for metric in metrics)
    max_torque = max(metric.max_torque for metric in metrics)
    max_torque_ratio = max(metric.max_torque_ratio for metric in metrics)
    mean_orientation_error = orientation_error_sum / sample_count
    print(
        f"[GLOBAL_SUMMARY]: samples={sample_count} "
        f"mean_position_error={position_error_sum / sample_count:.6f} m "
        f"max_position_error={max_position_error:.6f} m "
        f"mean_orientation_error={mean_orientation_error:.6f} rad "
        f"({math.degrees(mean_orientation_error):.3f} deg) "
        f"max_orientation_error={max_orientation_error:.6f} rad "
        f"({math.degrees(max_orientation_error):.3f} deg) "
        f"mean_max_joint_error={joint_error_sum / sample_count:.6f} rad "
        f"max_joint_error={max_joint_error:.6f} rad "
        f"max_torque={max_torque:.3f} Nm max_torque_ratio={max_torque_ratio:.3f}"
    )


def _evaluate(metrics: list[TargetMetrics]) -> int:
    orientation_tolerance = math.radians(args_cli.orientation_tolerance_deg)
    failures = []
    for spec, metric in zip(TARGET_SPECS, metrics, strict=True):
        if metric.count == 0:
            failures.append(f"target='{spec.label}' has no post-warmup samples")
            continue
        if metric.max_position_error > args_cli.position_tolerance:
            failures.append(
                f"target='{spec.label}' position={metric.max_position_error:.6f} m > "
                f"{args_cli.position_tolerance:.6f} m"
            )
        if metric.max_orientation_error > orientation_tolerance:
            failures.append(
                f"target='{spec.label}' orientation={math.degrees(metric.max_orientation_error):.3f} deg > "
                f"{args_cli.orientation_tolerance_deg:.3f} deg"
            )
        if metric.max_joint_error > args_cli.joint_tolerance:
            failures.append(
                f"target='{spec.label}' joint={metric.max_joint_error:.6f} rad > {args_cli.joint_tolerance:.6f} rad"
            )
    if failures:
        for failure in failures:
            print(f"[FAIL]: {failure}")
        print(f"[SUMMARY]: status=FAIL failures={len(failures)}")
        return 1
    print(
        "[SUMMARY]: status=PASS "
        f"position_tolerance={args_cli.position_tolerance:.6f} m "
        f"orientation_tolerance={args_cli.orientation_tolerance_deg:.3f} deg "
        f"joint_tolerance={args_cli.joint_tolerance:.6f} rad"
    )
    return 0


def _run_keyboard_teleop(
    sim: sim_utils.SimulationContext,
    scene: InteractiveScene,
    controller: SingleArmDifferentialIKController,
    markers: tuple[VisualizationMarkers, VisualizationMarkers] | None,
    default_gripper_action: float,
) -> int:
    robot = scene["robot"]
    state = TeleopState(default_gripper_action)
    keyboard = Se3Keyboard(
        Se3KeyboardCfg(
            gripper_term=False,
            pos_sensitivity=args_cli.pos_step,
            rot_sensitivity=args_cli.rot_step,
            sim_device=robot.device,
        )
    )
    keyboard.add_callback("K", state.toggle_gripper)
    keyboard.add_callback("R", state.request_reset)
    keyboard.add_callback("ESCAPE", state.request_quit)
    target_pose = _ee_pose_in_root(robot, controller.ee_body_idx).clone()
    _visualize_markers(markers, robot, controller.ee_body_idx, target_pose)
    print(
        "[INFO]: Piper keyboard teleoperation ready: "
        f"dt={sim.get_physics_dt():.6f}, pos_step={args_cli.pos_step:.4f} m, "
        f"rot_step={args_cli.rot_step:.4f} rad, print_interval={args_cli.print_interval}"
    )
    _print_teleop_bindings()

    count = 0
    sim_dt = sim.get_physics_dt()
    while (
        simulation_app.is_running()
        and not state.quit_requested
        and (args_cli.max_steps == 0 or count < args_cli.max_steps)
    ):
        if state.reset_requested:
            _reset_robot(sim, scene)
            controller.reset()
            keyboard.reset()
            state.reset()
            target_pose = _ee_pose_in_root(robot, controller.ee_body_idx).clone()
            _visualize_markers(markers, robot, controller.ee_body_idx, target_pose)
            print("[INFO]: Piper and teleoperation target reset.", flush=True)
            continue

        delta_pose = keyboard.advance().view(1, 6)
        if state.quit_requested or state.reset_requested:
            continue
        target_pose = _apply_keyboard_delta(target_pose, delta_pose)
        actions = torch.zeros((1, controller.action_dim), device=robot.device)
        actions[:, :7] = target_pose
        actions[:, 7] = state.gripper_action
        joint_targets = controller.compute(actions)
        robot.set_joint_position_target(joint_targets, joint_ids=controller.joint_ids)
        scene.write_data_to_sim()
        sim.step()
        scene.update(sim_dt)
        _visualize_markers(markers, robot, controller.ee_body_idx, target_pose)

        if args_cli.print_interval > 0 and count % args_cli.print_interval == 0:
            _print_teleop_diagnostics(robot, controller, target_pose, state)
        count += 1

    print(f"[SUMMARY]: status=REPORT_ONLY mode=teleop steps={count}")
    return 0


def run_simulator(sim: sim_utils.SimulationContext, scene: InteractiveScene) -> int:
    robot = scene["robot"]
    controller = _make_controller(sim, scene)
    _print_metadata(robot, controller)
    markers = None if args_cli.disable_markers else _make_markers()
    gripper_action = (
        2.0 * (PIPER_DEFAULT_GRIPPER_OPENING - PIPER_GRIPPER_MIN) / (PIPER_GRIPPER_MAX - PIPER_GRIPPER_MIN) - 1.0
    )

    _reset_robot(sim, scene)
    controller.reset()
    hold_actions = torch.zeros((scene.num_envs, controller.action_dim), device=robot.device)
    hold_actions[:, 7] = gripper_action
    for _ in range(args_cli.settle_steps):
        joint_targets = controller.compute(hold_actions)
        robot.set_joint_position_target(joint_targets, joint_ids=controller.joint_ids)
        scene.write_data_to_sim()
        sim.step()
        scene.update(sim.get_physics_dt())

    if args_cli.teleop:
        return _run_keyboard_teleop(sim, scene, controller, markers, gripper_action)

    home_pose = _ee_pose_in_root(robot, controller.ee_body_idx)
    targets = _make_targets(home_pose)
    metrics = [TargetMetrics() for _ in targets]
    current_target_index = -1
    count = 0
    sim_dt = sim.get_physics_dt()
    print(
        "[INFO]: Diagnostic ready: "
        f"gravity_enabled={args_cli.enable_gravity}, num_envs={scene.num_envs}, dt={sim_dt:.6f}, "
        f"targets={len(targets)}, hold_steps={args_cli.hold_steps}, "
        f"metric_warmup_steps={args_cli.metric_warmup_steps}, gripper_opening={PIPER_DEFAULT_GRIPPER_OPENING:.4f} m"
    )

    while simulation_app.is_running() and (args_cli.max_steps == 0 or count < args_cli.max_steps):
        target_step = count % args_cli.hold_steps
        if target_step == 0:
            current_target_index = (current_target_index + 1) % len(targets)
            controller.reset()
            print(f"[INFO]: Switching IK target: {targets[current_target_index][0]}")

        label, target_pose = targets[current_target_index]
        actions = torch.zeros((scene.num_envs, controller.action_dim), device=robot.device)
        actions[:, :7] = target_pose
        actions[:, 7] = gripper_action
        joint_targets = controller.compute(actions)
        robot.set_joint_position_target(joint_targets, joint_ids=controller.joint_ids)
        scene.write_data_to_sim()
        sim.step()
        scene.update(sim_dt)
        _visualize_markers(markers, robot, controller.ee_body_idx, target_pose)

        current_pose = _ee_pose_in_root(robot, controller.ee_body_idx)
        position_error = torch.linalg.vector_norm(current_pose[:, :3] - target_pose[:, :3], dim=-1)
        orientation_error = _orientation_error(current_pose[:, 3:7], target_pose[:, 3:7])
        joint_error, torque, torque_ratio = _arm_diagnostics(robot, controller.arm_joint_ids)
        if target_step >= args_cli.metric_warmup_steps:
            metrics[current_target_index].update(position_error, orientation_error, joint_error, torque, torque_ratio)

        if args_cli.print_interval > 0 and count % args_cli.print_interval == 0:
            print(
                f"[INFO]: target='{label}' position_error={float(position_error.mean()):.6f} m "
                f"orientation_error={float(orientation_error.mean()):.6f} rad "
                f"({math.degrees(float(orientation_error.mean())):.3f} deg) "
                f"max_joint_error={float(joint_error.max()):.6f} rad "
                f"max_torque={float(torque.max()):.3f} Nm "
                f"max_torque_ratio={float(torque_ratio.max()):.3f}"
            )
            print(f"[INFO]: env_0_arm_angles(rad): {_format_joint_angles(robot, controller.arm_joint_ids)}")
        count += 1

    _print_target_summaries(metrics)
    _print_global_summary(metrics)
    if args_cli.max_steps == 0:
        print("[SUMMARY]: status=REPORT_ONLY reason=interactive_run")
        return 0
    return _evaluate(metrics)


def main() -> int:
    sim_cfg = sim_utils.SimulationCfg(
        dt=1.0 / 120.0,
        render_interval=2,
        device=args_cli.device,
        use_fabric=not args_cli.disable_fabric,
    )
    sim = sim_utils.SimulationContext(sim_cfg)
    sim.set_camera_view(eye=[1.2, 1.2, 1.0], target=[0.25, 0.0, 0.45])
    scene = InteractiveScene(_build_scene_cfg())
    sim.reset()
    scene.update(sim.get_physics_dt())
    return run_simulator(sim, scene)


if __name__ == "__main__":
    exit_code = 0
    try:
        with torch.inference_mode():
            exit_code = main()
        if args_cli.fast_exit and args_cli.max_steps > 0:
            sys.stdout.flush()
            sys.stderr.flush()
            os._exit(exit_code)
    finally:
        simulation_app.close()
    raise SystemExit(exit_code)
