# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Visual Differential IK demo for the Galaxea R1 Pro robot.

This script follows the structure of Isaac Lab's ``run_diff_ik.py`` tutorial, but
uses the local R1 Pro asset and the bimanual controller used by the benchmark
tasks.

.. code-block:: bash

    python scripts/tools/run_r1_pro_diff_ik.py --num_envs 1 --device cuda:0

"""

from __future__ import annotations

import argparse
import copy

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Run a visual R1 Pro bimanual Differential IK demo.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of R1 Pro environments to spawn.")
parser.add_argument("--hold_steps", type=int, default=240, help="Number of simulation steps to hold each IK target.")
parser.add_argument("--settle_steps", type=int, default=60, help="Number of steps to settle at the default pose.")
parser.add_argument("--enable_gravity", action="store_true", help="Enable gravity on robot links.")
parser.add_argument(
    "--disable_fabric",
    action="store_true",
    default=False,
    help="Debug/compatibility option: disable Fabric and use USD I/O, which may desync GUI mesh updates.",
)
parser.add_argument("--marker_scale", type=float, default=0.09, help="Scale of current/goal frame markers.")
parser.add_argument(
    "--print_interval",
    type=int,
    default=60,
    help="Print mean end-effector position error every N simulation steps. Use 0 to disable.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
if args_cli.hold_steps <= 0:
    raise ValueError("--hold_steps must be positive.")

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.markers import VisualizationMarkers
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import combine_frame_transforms, subtract_frame_transforms

from assembly_benchmark.controllers import BimanualDifferentialIKController
from assembly_benchmark.robots.r1_pro import (
    R1_PRO_CFG,
    R1_PRO_LEFT_ARM_JOINT_NAMES,
    R1_PRO_LEFT_EE_LINK_NAME,
    R1_PRO_LEFT_GRIPPER_JOINT_NAMES,
    R1_PRO_LEFT_IK_LINK_NAME,
    R1_PRO_RIGHT_ARM_JOINT_NAMES,
    R1_PRO_RIGHT_EE_LINK_NAME,
    R1_PRO_RIGHT_GRIPPER_JOINT_NAMES,
    R1_PRO_RIGHT_IK_LINK_NAME,
)


@configclass
class R1ProDiffIKSceneCfg(InteractiveSceneCfg):
    """Minimal scene for the R1 Pro visual IK demo."""

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


def _build_scene_cfg() -> R1ProDiffIKSceneCfg:
    scene_cfg = R1ProDiffIKSceneCfg(num_envs=args_cli.num_envs, env_spacing=4.0, replicate_physics=True)
    robot_cfg = copy.deepcopy(R1_PRO_CFG).replace(prim_path="{ENV_REGEX_NS}/Robot")
    robot_cfg.spawn.rigid_props.disable_gravity = not args_cli.enable_gravity
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


def _ee_poses_in_root(robot, left_body_idx: int, right_body_idx: int) -> tuple[torch.Tensor, torch.Tensor]:
    root_pose_w = robot.data.root_pose_w
    left_pose_w = robot.data.body_pose_w[:, left_body_idx]
    right_pose_w = robot.data.body_pose_w[:, right_body_idx]
    left_pos_b, left_quat_b = subtract_frame_transforms(
        root_pose_w[:, :3], root_pose_w[:, 3:7], left_pose_w[:, :3], left_pose_w[:, 3:7]
    )
    right_pos_b, right_quat_b = subtract_frame_transforms(
        root_pose_w[:, :3], root_pose_w[:, 3:7], right_pose_w[:, :3], right_pose_w[:, 3:7]
    )
    return torch.cat((left_pos_b, left_quat_b), dim=-1), torch.cat((right_pos_b, right_quat_b), dim=-1)


def _make_markers() -> tuple[VisualizationMarkers, VisualizationMarkers, VisualizationMarkers, VisualizationMarkers]:
    current_marker_cfg = FRAME_MARKER_CFG.copy()
    current_marker_cfg.markers["frame"].scale = (args_cli.marker_scale, args_cli.marker_scale, args_cli.marker_scale)

    target_marker_cfg = FRAME_MARKER_CFG.copy()
    target_scale = args_cli.marker_scale * 1.35
    target_marker_cfg.markers["frame"].scale = (target_scale, target_scale, target_scale)

    left_current = VisualizationMarkers(current_marker_cfg.replace(prim_path="/Visuals/r1_pro_left_current"))
    left_target = VisualizationMarkers(target_marker_cfg.replace(prim_path="/Visuals/r1_pro_left_target"))
    right_current = VisualizationMarkers(current_marker_cfg.replace(prim_path="/Visuals/r1_pro_right_current"))
    right_target = VisualizationMarkers(target_marker_cfg.replace(prim_path="/Visuals/r1_pro_right_target"))
    return left_current, left_target, right_current, right_target


def _visualize_markers(
    markers: tuple[VisualizationMarkers, VisualizationMarkers, VisualizationMarkers, VisualizationMarkers],
    robot,
    left_body_idx: int,
    right_body_idx: int,
    left_target_b: torch.Tensor,
    right_target_b: torch.Tensor,
) -> None:
    left_current, left_target, right_current, right_target = markers
    root_pose_w = robot.data.root_pose_w
    left_pose_w = robot.data.body_pose_w[:, left_body_idx]
    right_pose_w = robot.data.body_pose_w[:, right_body_idx]
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


def _make_controller(scene: InteractiveScene) -> BimanualDifferentialIKController:
    robot = scene["robot"]
    return BimanualDifferentialIKController(
        robot=robot,
        left_arm_joint_names=R1_PRO_LEFT_ARM_JOINT_NAMES,
        right_arm_joint_names=R1_PRO_RIGHT_ARM_JOINT_NAMES,
        left_gripper_joint_names=R1_PRO_LEFT_GRIPPER_JOINT_NAMES,
        right_gripper_joint_names=R1_PRO_RIGHT_GRIPPER_JOINT_NAMES,
        left_ee_link_name=R1_PRO_LEFT_EE_LINK_NAME,
        right_ee_link_name=R1_PRO_RIGHT_EE_LINK_NAME,
        left_ik_link_name=R1_PRO_LEFT_IK_LINK_NAME,
        right_ik_link_name=R1_PRO_RIGHT_IK_LINK_NAME,
        arm_action_scale=0.5,
        gripper_min=0.0,
        gripper_max=0.05,
        num_envs=scene.num_envs,
        device=robot.device,
    )


def _print_robot_metadata(scene: InteractiveScene, controller: BimanualDifferentialIKController) -> None:
    robot = scene["robot"]
    print(
        "[INFO]: R1 Pro metadata: "
        f"fixed_base={robot.is_fixed_base}, joints={robot.num_joints}, bodies={robot.num_bodies}, "
        f"jacobians={tuple(robot.root_physx_view.get_jacobians().shape)}"
    )
    print(
        "[INFO]: IK links: "
        f"left_ee={R1_PRO_LEFT_EE_LINK_NAME} left_ik={R1_PRO_LEFT_IK_LINK_NAME} "
        f"right_ee={R1_PRO_RIGHT_EE_LINK_NAME} right_ik={R1_PRO_RIGHT_IK_LINK_NAME}"
    )
    print(
        "[INFO]: Controller joints: "
        f"left_arm={controller.left_arm_joint_ids}, right_arm={controller.right_arm_joint_ids}, "
        f"controlled={controller.joint_ids}"
    )


def _format_joint_angles(robot, env_id: int = 0) -> str:
    joint_pos = robot.data.joint_pos[env_id].detach().cpu()
    return ", ".join(f"{name}={pos:.4f}" for name, pos in zip(robot.joint_names, joint_pos, strict=True))


def _target_offsets(device: str) -> list[tuple[str, torch.Tensor, torch.Tensor]]:
    return [
        (
            "home",
            torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32, device=device),
            torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32, device=device),
        ),
        (
            "lift both arms",
            torch.tensor([0.0, 0.0, 0.08], dtype=torch.float32, device=device),
            torch.tensor([0.0, 0.0, 0.08], dtype=torch.float32, device=device),
        ),
        (
            "move outward",
            torch.tensor([0.0, 0.05, 0.05], dtype=torch.float32, device=device),
            torch.tensor([0.0, -0.05, 0.05], dtype=torch.float32, device=device),
        ),
        (
            "reach forward",
            torch.tensor([0.06, 0.03, 0.03], dtype=torch.float32, device=device),
            torch.tensor([0.06, -0.03, 0.03], dtype=torch.float32, device=device),
        ),
        (
            "left up right down",
            torch.tensor([0.03, 0.02, 0.09], dtype=torch.float32, device=device),
            torch.tensor([0.03, -0.02, -0.02], dtype=torch.float32, device=device),
        ),
    ]


def run_simulator(sim: sim_utils.SimulationContext, scene: InteractiveScene) -> None:
    robot = scene["robot"]
    controller = _make_controller(scene)
    _print_robot_metadata(scene, controller)

    left_ee_body_idx = controller.left_ee_body_idx
    right_ee_body_idx = controller.right_ee_body_idx
    markers = _make_markers()

    _reset_robot(sim, scene)
    controller.reset()
    zero_actions = torch.zeros((scene.num_envs, controller.action_dim), device=robot.device)
    for _ in range(args_cli.settle_steps):
        joint_targets = controller.compute(zero_actions)
        robot.set_joint_position_target(joint_targets, joint_ids=controller.joint_ids)
        scene.write_data_to_sim()
        sim.step()
        scene.update(sim.get_physics_dt())

    left_home, right_home = _ee_poses_in_root(robot, left_ee_body_idx, right_ee_body_idx)
    offsets = _target_offsets(robot.device)
    current_goal_idx = -1
    count = 0
    sim_dt = sim.get_physics_dt()
    print("[INFO]: Setup complete. Visualizing current EE frames and commanded target frames.")

    while simulation_app.is_running():
        if count % args_cli.hold_steps == 0:
            current_goal_idx = (current_goal_idx + 1) % len(offsets)
            label, left_offset, right_offset = offsets[current_goal_idx]
            print(f"[INFO]: Switching IK target: {label}")
            controller.reset()

        label, left_offset, right_offset = offsets[current_goal_idx]
        left_target = left_home.clone()
        right_target = right_home.clone()
        left_target[:, :3] += left_offset
        right_target[:, :3] += right_offset

        actions = torch.zeros((scene.num_envs, controller.action_dim), device=robot.device)
        actions[:, 0:7] = left_target
        actions[:, 7] = 0.0
        actions[:, 8:15] = right_target
        actions[:, 15] = 0.0

        joint_targets = controller.compute(actions)
        robot.set_joint_position_target(joint_targets, joint_ids=controller.joint_ids)
        scene.write_data_to_sim()
        sim.step()
        scene.update(sim_dt)
        _visualize_markers(markers, robot, left_ee_body_idx, right_ee_body_idx, left_target, right_target)

        if args_cli.print_interval > 0 and count % args_cli.print_interval == 0:
            left_pose, right_pose = _ee_poses_in_root(robot, left_ee_body_idx, right_ee_body_idx)
            left_error = torch.linalg.norm(left_pose[:, :3] - left_target[:, :3], dim=-1).mean()
            right_error = torch.linalg.norm(right_pose[:, :3] - right_target[:, :3], dim=-1).mean()
            print(
                f"[INFO]: target='{label}' left_error={float(left_error):.4f} m "
                f"right_error={float(right_error):.4f} m"
            )
            print(f"[INFO]: env_0_joint_angles(rad): {_format_joint_angles(robot)}")

        count += 1


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
    finally:
        simulation_app.close()
