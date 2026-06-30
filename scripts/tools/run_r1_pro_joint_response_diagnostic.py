# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Measure low-level R1 Pro joint step response under Isaac Lab physics.

This diagnostic is intentionally below the task/IK layer.  It writes joint
position targets directly, then reports per-joint response metrics such as
steady-state error, overshoot, rise time, settling time, and torque usage.

.. code-block:: bash

    python scripts/tools/run_r1_pro_joint_response_diagnostic.py --device cuda:0 --headless --enable_gravity

"""

from __future__ import annotations

import argparse
import copy
import math
import os

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Run R1 Pro low-level joint response diagnostics.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments. Only 1 is supported.")
parser.add_argument("--enable_gravity", action="store_true", help="Enable gravity on robot links for this run.")
parser.add_argument(
    "--joint_groups",
    type=str,
    default="all",
    help="Comma-separated joint groups to step: all, torso, left_arm, right_arm, left_gripper, right_gripper.",
)
parser.add_argument("--pre_steps", type=int, default=180, help="Steps to hold home target before the response step.")
parser.add_argument("--response_steps", type=int, default=360, help="Steps to record after applying the step target.")
parser.add_argument("--steady_steps", type=int, default=60, help="Tail steps used for steady-state metrics.")
parser.add_argument("--torso_step", type=float, default=0.08, help="Torso joint step size in rad.")
parser.add_argument("--arm_step", type=float, default=0.10, help="Arm joint step size in rad.")
parser.add_argument("--gripper_step", type=float, default=0.02, help="Gripper joint step size in rad.")
parser.add_argument(
    "--step_reference",
    choices=("actual", "home"),
    default="actual",
    help="Use the settled actual joint position or the home target as the step starting reference.",
)
parser.add_argument("--settling_tolerance", type=float, default=0.005, help="Absolute settling tolerance in rad.")
parser.add_argument(
    "--settling_ratio",
    type=float,
    default=0.05,
    help="Relative settling tolerance as a fraction of the commanded step.",
)
parser.add_argument("--torso_stiffness", type=float, default=None, help="Override torso actuator stiffness.")
parser.add_argument("--torso_damping", type=float, default=None, help="Override torso actuator damping.")
parser.add_argument("--torso_effort", type=float, default=None, help="Override torso actuator effort limit.")
parser.add_argument("--torso_armature", type=float, default=None, help="Override torso actuator armature.")
parser.add_argument("--arm_stiffness", type=float, default=None, help="Override both arm actuator stiffness.")
parser.add_argument("--arm_damping", type=float, default=None, help="Override both arm actuator damping.")
parser.add_argument("--arm_effort", type=float, default=None, help="Override both arm actuator effort limit.")
parser.add_argument("--arm_armature", type=float, default=None, help="Override both arm actuator armature.")
parser.add_argument("--gripper_stiffness", type=float, default=None, help="Override both gripper actuator stiffness.")
parser.add_argument("--gripper_damping", type=float, default=None, help="Override both gripper actuator damping.")
parser.add_argument("--gripper_effort", type=float, default=None, help="Override both gripper actuator effort limit.")
parser.add_argument("--gripper_armature", type=float, default=None, help="Override both gripper actuator armature.")
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
    default=False,
    help="Debug/compatibility option: disable Fabric and use USD I/O.",
)
parser.add_argument("--summary_only", action="store_true", help="Print group summaries and worst joints only.")
parser.add_argument("--top_k", type=int, default=8, help="Number of worst joints to print when --summary_only is set.")
parser.add_argument("--print_interval", type=int, default=0, help="Print live max error every N steps. Use 0 to disable.")
parser.add_argument(
    "--fast_exit",
    action="store_true",
    help="Exit the process immediately after diagnostics, avoiding slow Kit shutdown.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

if args_cli.num_envs != 1:
    raise ValueError("Joint response diagnostics currently support only --num_envs 1.")
if args_cli.pre_steps <= 0:
    raise ValueError("--pre_steps must be positive.")
if args_cli.response_steps <= 0:
    raise ValueError("--response_steps must be positive.")
if args_cli.steady_steps <= 0:
    raise ValueError("--steady_steps must be positive.")
if args_cli.steady_steps > args_cli.response_steps:
    raise ValueError("--steady_steps must be <= --response_steps.")
for name in ("torso_step", "arm_step", "gripper_step", "settling_tolerance", "settling_ratio"):
    if getattr(args_cli, name) < 0.0:
        raise ValueError(f"--{name} must be non-negative.")
for name in (
    "torso_stiffness",
    "torso_damping",
    "torso_effort",
    "torso_armature",
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
if args_cli.top_k <= 0:
    raise ValueError("--top_k must be positive.")

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.utils import configclass

from assembly_benchmark.robots.r1_pro import (
    R1_PRO_CFG,
    R1_PRO_LEFT_ARM_JOINT_NAMES,
    R1_PRO_LEFT_GRIPPER_JOINT_NAMES,
    R1_PRO_RIGHT_ARM_JOINT_NAMES,
    R1_PRO_RIGHT_GRIPPER_JOINT_NAMES,
    R1_PRO_TORSO_JOINT_NAMES,
)


@configclass
class R1ProJointResponseSceneCfg(InteractiveSceneCfg):
    """Minimal scene for direct joint response diagnostics."""

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


def _override_actuator(actuator_cfg, **kwargs) -> None:
    for key, value in kwargs.items():
        if value is not None:
            setattr(actuator_cfg, key, value)


def _build_scene_cfg() -> R1ProJointResponseSceneCfg:
    scene_cfg = R1ProJointResponseSceneCfg(num_envs=args_cli.num_envs, env_spacing=4.0, replicate_physics=True)
    robot_cfg = copy.deepcopy(R1_PRO_CFG).replace(prim_path="{ENV_REGEX_NS}/Robot")
    robot_cfg.spawn.rigid_props.disable_gravity = not args_cli.enable_gravity
    if args_cli.solver_position_iterations is not None:
        robot_cfg.spawn.articulation_props.solver_position_iteration_count = args_cli.solver_position_iterations
    if args_cli.solver_velocity_iterations is not None:
        robot_cfg.spawn.articulation_props.solver_velocity_iteration_count = args_cli.solver_velocity_iterations

    _override_actuator(
        robot_cfg.actuators["torso"],
        stiffness=args_cli.torso_stiffness,
        damping=args_cli.torso_damping,
        effort_limit_sim=args_cli.torso_effort,
        armature=args_cli.torso_armature,
    )
    for actuator_name in ("left_arm", "right_arm"):
        _override_actuator(
            robot_cfg.actuators[actuator_name],
            stiffness=args_cli.arm_stiffness,
            damping=args_cli.arm_damping,
            effort_limit_sim=args_cli.arm_effort,
            armature=args_cli.arm_armature,
        )
    for actuator_name in ("left_gripper", "right_gripper"):
        _override_actuator(
            robot_cfg.actuators[actuator_name],
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


def _joint_group_ids(robot) -> dict[str, list[int]]:
    group_names = {
        "torso": R1_PRO_TORSO_JOINT_NAMES,
        "left_arm": R1_PRO_LEFT_ARM_JOINT_NAMES,
        "right_arm": R1_PRO_RIGHT_ARM_JOINT_NAMES,
        "left_gripper": R1_PRO_LEFT_GRIPPER_JOINT_NAMES,
        "right_gripper": R1_PRO_RIGHT_GRIPPER_JOINT_NAMES,
    }
    group_ids = {}
    for group, names in group_names.items():
        joint_ids, _ = robot.find_joints(names, preserve_order=True)
        group_ids[group] = joint_ids
    return group_ids


def _selected_groups() -> list[str]:
    valid_groups = ("torso", "left_arm", "right_arm", "left_gripper", "right_gripper")
    if args_cli.joint_groups.strip() == "all":
        return list(valid_groups)
    groups = [group.strip() for group in args_cli.joint_groups.split(",") if group.strip()]
    invalid_groups = [group for group in groups if group not in valid_groups]
    if invalid_groups:
        raise ValueError(f"Unsupported --joint_groups entries: {invalid_groups}. Valid groups: {valid_groups}.")
    if not groups:
        raise ValueError("--joint_groups must select at least one group.")
    return groups


def _group_step_size(group: str) -> float:
    if group == "torso":
        return args_cli.torso_step
    if group.endswith("gripper"):
        return args_cli.gripper_step
    return args_cli.arm_step


def _selected_joint_ids(group_ids: dict[str, list[int]], groups: list[str]) -> tuple[list[int], dict[int, str]]:
    selected_joint_ids = []
    joint_to_group = {}
    for group in groups:
        if _group_step_size(group) <= 0.0:
            continue
        for joint_id in group_ids[group]:
            selected_joint_ids.append(joint_id)
            joint_to_group[joint_id] = group
    if not selected_joint_ids:
        raise RuntimeError("No joints selected for diagnostics. Check --joint_groups and step sizes.")
    return selected_joint_ids, joint_to_group


def _make_step_target(robot, group_ids: dict[str, list[int]], groups: list[str], reference_pos: torch.Tensor) -> torch.Tensor:
    home_target = robot.data.default_joint_pos.clone()
    step_target = home_target.clone()
    soft_limits = robot.data.soft_joint_pos_limits[0]

    for group in groups:
        step_size = _group_step_size(group)
        for local_index, joint_id in enumerate(group_ids[group]):
            if step_size <= 0.0:
                continue
            sign = 1.0 if local_index % 2 == 0 else -1.0
            reference_value = float(reference_pos[0, joint_id])
            lower = float(soft_limits[joint_id, 0])
            upper = float(soft_limits[joint_id, 1])
            desired = reference_value + sign * step_size
            if desired < lower or desired > upper:
                desired = reference_value - sign * step_size
            desired = min(max(desired, lower), upper)
            step_target[:, joint_id] = desired

    return step_target


def _step_sim(sim: sim_utils.SimulationContext, scene: InteractiveScene, target: torch.Tensor) -> None:
    robot = scene["robot"]
    robot.set_joint_position_target(target)
    scene.write_data_to_sim()
    sim.step()
    scene.update(sim.get_physics_dt())


def _collect_joint_state(robot, joint_ids: list[int]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return (
        robot.data.joint_pos[0, joint_ids].clone(),
        robot.data.joint_vel[0, joint_ids].clone(),
        robot.data.applied_torque[0, joint_ids].clone(),
    )


def _first_index(mask: torch.Tensor) -> int | None:
    indices = torch.nonzero(mask, as_tuple=False)
    if indices.numel() == 0:
        return None
    return int(indices[0, 0].item())


def _settling_index(error: torch.Tensor, threshold: float) -> int | None:
    settled = error <= threshold
    for index in range(settled.numel()):
        if bool(torch.all(settled[index:]).item()):
            return index
    return None


def _format_time(value: float) -> str:
    if math.isinf(value):
        return "inf"
    if math.isnan(value):
        return "nan"
    return f"{value:.4f}"


def _compute_metrics(
    robot,
    joint_ids: list[int],
    joint_to_group: dict[int, str],
    home_target: torch.Tensor,
    step_target: torch.Tensor,
    pre_positions: torch.Tensor,
    positions: torch.Tensor,
    velocities: torch.Tensor,
    torques: torch.Tensor,
    dt: float,
) -> list[dict[str, float | str]]:
    start_pos = pre_positions[-1]
    target_pos = step_target[0, joint_ids]
    command_delta = target_pos - start_pos
    command_mag = torch.abs(command_delta)
    command_sign = torch.where(command_delta >= 0.0, 1.0, -1.0)
    tail_count = min(args_cli.steady_steps, positions.shape[0])
    home_error_tail = torch.abs(pre_positions[-tail_count:] - home_target[0, joint_ids]).mean(dim=0)

    effort_limits = robot.data.joint_effort_limits[0, joint_ids].detach().clone()
    metrics = []
    for local_index, joint_id in enumerate(joint_ids):
        name = robot.joint_names[joint_id]
        magnitude = float(command_mag[local_index])
        if magnitude < 1.0e-8:
            continue

        signed_progress = (positions[:, local_index] - start_pos[local_index]) * command_sign[local_index]
        error_abs = torch.abs(positions[:, local_index] - target_pos[local_index])
        overshoot = max(float(torch.max(signed_progress - command_mag[local_index])), 0.0)
        rise_index = _first_index(signed_progress >= 0.9 * command_mag[local_index])
        settle_threshold = max(args_cli.settling_tolerance, args_cli.settling_ratio * magnitude)
        settle_index = _settling_index(error_abs, settle_threshold)
        effort_limit = float(effort_limits[local_index])
        torque_ratio = 0.0
        if effort_limit > 1.0e-6:
            torque_ratio = float(torch.max(torch.abs(torques[:, local_index])) / effort_limit)

        metrics.append(
            {
                "joint": name,
                "group": joint_to_group[joint_id],
                "delta": float(command_delta[local_index]),
                "home_steady_error": float(home_error_tail[local_index]),
                "final_error": float(error_abs[-1]),
                "steady_error": float(error_abs[-tail_count:].mean()),
                "overshoot": overshoot,
                "overshoot_pct": 100.0 * overshoot / magnitude,
                "rise_time": math.inf if rise_index is None else rise_index * dt,
                "settle_time": math.inf if settle_index is None else settle_index * dt,
                "max_velocity": float(torch.max(torch.abs(velocities[:, local_index]))),
                "max_torque_ratio": torque_ratio,
            }
        )
    return metrics


def _print_joint_metric(metric: dict[str, float | str]) -> None:
    print(
        "[JOINT_METRIC] "
        f"name={metric['joint']} group={metric['group']} "
        f"delta={metric['delta']:.5f} rad "
        f"home_steady_error={metric['home_steady_error']:.6f} rad "
        f"steady_error={metric['steady_error']:.6f} rad "
        f"final_error={metric['final_error']:.6f} rad "
        f"overshoot={metric['overshoot']:.6f} rad "
        f"overshoot_pct={metric['overshoot_pct']:.2f} "
        f"rise_time={_format_time(metric['rise_time'])} s "
        f"settle_time={_format_time(metric['settle_time'])} s "
        f"max_velocity={metric['max_velocity']:.4f} rad/s "
        f"max_torque_ratio={metric['max_torque_ratio']:.3f}"
    )


def _print_summaries(metrics: list[dict[str, float | str]]) -> None:
    groups = sorted({str(metric["group"]) for metric in metrics})
    for group in groups:
        group_metrics = [metric for metric in metrics if metric["group"] == group]
        finite_settles = [float(metric["settle_time"]) for metric in group_metrics if not math.isinf(metric["settle_time"])]
        settle_fail_count = len(group_metrics) - len(finite_settles)
        max_settle_time = max(finite_settles) if finite_settles else math.inf
        print(
            "[GROUP_SUMMARY] "
            f"group={group} joints={len(group_metrics)} "
            f"mean_home_steady_error={sum(float(m['home_steady_error']) for m in group_metrics) / len(group_metrics):.6f} rad "
            f"max_home_steady_error={max(float(m['home_steady_error']) for m in group_metrics):.6f} rad "
            f"mean_steady_error={sum(float(m['steady_error']) for m in group_metrics) / len(group_metrics):.6f} rad "
            f"max_steady_error={max(float(m['steady_error']) for m in group_metrics):.6f} rad "
            f"max_overshoot_pct={max(float(m['overshoot_pct']) for m in group_metrics):.2f} "
            f"max_settle_time={_format_time(max_settle_time)} s "
            f"settle_fail_count={settle_fail_count} "
            f"max_torque_ratio={max(float(m['max_torque_ratio']) for m in group_metrics):.3f}"
        )

    finite_settles = [float(metric["settle_time"]) for metric in metrics if not math.isinf(metric["settle_time"])]
    overall_settle_fail_count = len(metrics) - len(finite_settles)
    print(
        "[SUMMARY] "
        f"gravity_enabled={args_cli.enable_gravity} joints={len(metrics)} "
        f"mean_home_steady_error={sum(float(m['home_steady_error']) for m in metrics) / len(metrics):.6f} rad "
        f"max_home_steady_error={max(float(m['home_steady_error']) for m in metrics):.6f} rad "
        f"mean_steady_error={sum(float(m['steady_error']) for m in metrics) / len(metrics):.6f} rad "
        f"max_steady_error={max(float(m['steady_error']) for m in metrics):.6f} rad "
        f"max_overshoot_pct={max(float(m['overshoot_pct']) for m in metrics):.2f} "
        f"max_settle_time={_format_time(max(finite_settles) if finite_settles else math.inf)} s "
        f"settle_fail_count={overall_settle_fail_count} "
        f"max_torque_ratio={max(float(m['max_torque_ratio']) for m in metrics):.3f}"
    )


def _print_worst_joints(metrics: list[dict[str, float | str]]) -> None:
    by_steady = sorted(metrics, key=lambda metric: float(metric["steady_error"]), reverse=True)[: args_cli.top_k]
    by_overshoot = sorted(metrics, key=lambda metric: float(metric["overshoot_pct"]), reverse=True)[: args_cli.top_k]
    print("[WORST_STEADY_ERROR]")
    for metric in by_steady:
        _print_joint_metric(metric)
    print("[WORST_OVERSHOOT]")
    for metric in by_overshoot:
        _print_joint_metric(metric)


def _mean_joint_property(robot, property_tensor: torch.Tensor, joint_ids: list[int]) -> float:
    return float(property_tensor[0, joint_ids].mean())


def run_simulator(sim: sim_utils.SimulationContext, scene: InteractiveScene) -> None:
    robot = scene["robot"]
    group_ids = _joint_group_ids(robot)
    groups = _selected_groups()
    joint_ids, joint_to_group = _selected_joint_ids(group_ids, groups)
    home_target = robot.data.default_joint_pos.clone()

    _reset_robot(sim, scene)
    pre_positions = []
    response_positions = []
    response_velocities = []
    response_torques = []
    dt = sim.get_physics_dt()

    print(
        "[INFO]: Diagnostic config: "
        f"gravity_enabled={args_cli.enable_gravity} groups={groups} step_reference={args_cli.step_reference} "
        f"pre_steps={args_cli.pre_steps} "
        f"response_steps={args_cli.response_steps} dt={dt:.6f} "
        f"torso(k={_mean_joint_property(robot, robot.data.default_joint_stiffness, group_ids['torso']):.1f}, "
        f"d={_mean_joint_property(robot, robot.data.default_joint_damping, group_ids['torso']):.1f}, "
        f"effort={_mean_joint_property(robot, robot.data.joint_effort_limits, group_ids['torso']):.1f}) "
        f"arm(k={_mean_joint_property(robot, robot.data.default_joint_stiffness, group_ids['left_arm']):.1f}, "
        f"d={_mean_joint_property(robot, robot.data.default_joint_damping, group_ids['left_arm']):.1f}, "
        f"effort={_mean_joint_property(robot, robot.data.joint_effort_limits, group_ids['left_arm']):.1f}) "
        f"gripper(k={_mean_joint_property(robot, robot.data.default_joint_stiffness, group_ids['left_gripper']):.1f}, "
        f"d={_mean_joint_property(robot, robot.data.default_joint_damping, group_ids['left_gripper']):.1f}, "
        f"effort={_mean_joint_property(robot, robot.data.joint_effort_limits, group_ids['left_gripper']):.1f})"
    )

    for step in range(args_cli.pre_steps):
        _step_sim(sim, scene, home_target)
        pre_positions.append(robot.data.joint_pos[0, joint_ids].clone())
        if args_cli.print_interval > 0 and step % args_cli.print_interval == 0:
            max_error = torch.max(torch.abs(robot.data.joint_pos[0, joint_ids] - home_target[0, joint_ids]))
            print(f"[INFO]: pre_step={step} max_home_error={float(max_error):.6f} rad")

    reference_pos = robot.data.joint_pos.clone() if args_cli.step_reference == "actual" else home_target
    step_target = _make_step_target(robot, group_ids, groups, reference_pos)

    for step in range(args_cli.response_steps):
        _step_sim(sim, scene, step_target)
        pos, vel, torque = _collect_joint_state(robot, joint_ids)
        response_positions.append(pos)
        response_velocities.append(vel)
        response_torques.append(torque)
        if args_cli.print_interval > 0 and step % args_cli.print_interval == 0:
            max_error = torch.max(torch.abs(pos - step_target[0, joint_ids]))
            print(f"[INFO]: response_step={step} max_step_error={float(max_error):.6f} rad")

    metrics = _compute_metrics(
        robot=robot,
        joint_ids=joint_ids,
        joint_to_group=joint_to_group,
        home_target=home_target,
        step_target=step_target,
        pre_positions=torch.stack(pre_positions),
        positions=torch.stack(response_positions),
        velocities=torch.stack(response_velocities),
        torques=torch.stack(response_torques),
        dt=dt,
    )

    if args_cli.summary_only:
        _print_summaries(metrics)
        _print_worst_joints(metrics)
    else:
        for metric in metrics:
            _print_joint_metric(metric)
        _print_summaries(metrics)


def main() -> None:
    sim_cfg = sim_utils.SimulationCfg(
        dt=1.0 / 120.0,
        render_interval=2,
        device=args_cli.device,
        use_fabric=not args_cli.disable_fabric,
    )
    sim = sim_utils.SimulationContext(sim_cfg)
    sim.set_camera_view(eye=[3.0, 3.0, 2.0], target=[0.0, 0.0, 0.8])

    scene = InteractiveScene(_build_scene_cfg())
    sim.reset()
    scene.update(sim.get_physics_dt())
    run_simulator(sim, scene)


if __name__ == "__main__":
    try:
        with torch.inference_mode():
            main()
        if args_cli.fast_exit:
            os._exit(0)
    finally:
        simulation_app.close()
