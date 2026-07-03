# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Preview the one_leg assembly target poses by placing parts directly.

This script is a visual/debug helper. It loads the one_leg assembly scene, keeps
the table top at its reset pose, and writes table-leg root poses from the
assembly target poses so the user can inspect whether the relative target
frames form a valid assembled table.

.. code-block:: bash

    python scripts/tools/preview_one_leg_assembled_pose.py --num_envs 1 --device cuda:0

"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher


TASK_NAME = "Assembly-Benchmark-OneLeg-Direct-v0"
FULL_TABLE_LEG_NAMES = (
    "square_table_leg1",
    "square_table_leg2",
    "square_table_leg3",
    "square_table_leg4",
)


parser = argparse.ArgumentParser(description="Preview one_leg assembled target poses.")
parser.add_argument("--task", type=str, default=TASK_NAME, help="Task to load.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments. Only 1 is supported.")
parser.add_argument(
    "--disable_fabric",
    action="store_true",
    default=False,
    help="Debug/compatibility option: disable Fabric and use USD I/O, which may desync GUI mesh updates.",
)
parser.add_argument(
    "--mode",
    choices=("full_table", "relation_child"),
    default="full_table",
    help="Preview all four table legs, or only the relation child at --target_index.",
)
parser.add_argument("--target_index", type=int, default=0, help="Target index used by relation_child mode.")
parser.add_argument("--settle_steps", type=int, default=30, help="Steps to refresh the assembled preview initially.")
parser.add_argument("--marker_scale", type=float, default=0.08, help="Scale of top/target frame markers.")
parser.add_argument("--disable_markers", action="store_true", help="Disable top and target frame markers.")
parser.add_argument("--print_poses", action="store_true", help="Print assembled world and relative poses.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

args_cli.enable_cameras = True

if args_cli.num_envs != 1:
    raise ValueError("one_leg assembled pose preview currently supports only --num_envs 1.")
if args_cli.settle_steps < 0:
    raise ValueError("--settle_steps must be non-negative.")
if args_cli.marker_scale <= 0.0:
    raise ValueError("--marker_scale must be positive.")

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch

import isaaclab_tasks  # noqa: F401
from isaaclab.markers import VisualizationMarkers
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.utils.math import combine_frame_transforms, subtract_frame_transforms
from isaaclab_tasks.utils import parse_env_cfg

import assembly_benchmark.tasks  # noqa: F401


def _make_markers() -> tuple[VisualizationMarkers, VisualizationMarkers]:
    top_marker_cfg = FRAME_MARKER_CFG.copy()
    top_marker_cfg.markers["frame"].scale = (
        args_cli.marker_scale,
        args_cli.marker_scale,
        args_cli.marker_scale,
    )

    target_marker_cfg = FRAME_MARKER_CFG.copy()
    target_scale = args_cli.marker_scale * 1.25
    target_marker_cfg.markers["frame"].scale = (target_scale, target_scale, target_scale)

    top_marker = VisualizationMarkers(top_marker_cfg.replace(prim_path="/Visuals/one_leg_preview_top_frame"))
    target_marker = VisualizationMarkers(
        target_marker_cfg.replace(prim_path="/Visuals/one_leg_preview_target_frames")
    )
    return top_marker, target_marker


def _target_poses_world(unwrapped) -> torch.Tensor:
    top_pose_w = unwrapped.assembly_parent_part.data.root_pose_w[0:1]
    target_poses_top = unwrapped.assembled_target_poses
    top_pos_w = top_pose_w[:, :3].repeat(target_poses_top.shape[0], 1)
    top_quat_w = top_pose_w[:, 3:7].repeat(target_poses_top.shape[0], 1)
    target_pos_w, target_quat_w = combine_frame_transforms(
        top_pos_w,
        top_quat_w,
        target_poses_top[:, :3],
        target_poses_top[:, 3:7],
    )
    return torch.cat((target_pos_w, target_quat_w), dim=-1)


def _preview_targets(unwrapped) -> tuple[tuple[str, ...], tuple[int, ...]]:
    target_poses_w = _target_poses_world(unwrapped)
    if args_cli.target_index < 0 or args_cli.target_index >= target_poses_w.shape[0]:
        raise ValueError(
            f"--target_index must be in [0, {target_poses_w.shape[0] - 1}], got {args_cli.target_index}."
        )

    if args_cli.mode == "relation_child":
        return (unwrapped.cfg.assembly_child_part_name,), (args_cli.target_index,)

    if len(FULL_TABLE_LEG_NAMES) > target_poses_w.shape[0]:
        raise RuntimeError(
            f"full_table mode requires at least {len(FULL_TABLE_LEG_NAMES)} target poses, "
            f"got {target_poses_w.shape[0]}."
        )
    missing = [name for name in FULL_TABLE_LEG_NAMES if not hasattr(unwrapped, name)]
    if missing:
        raise RuntimeError(f"full_table mode requires missing one_leg parts: {', '.join(missing)}.")
    return FULL_TABLE_LEG_NAMES, tuple(range(len(FULL_TABLE_LEG_NAMES)))


def _write_part_poses(unwrapped, part_names: tuple[str, ...], poses_w: torch.Tensor) -> None:
    env_ids = torch.tensor([0], dtype=torch.long, device=unwrapped.device)
    zero_velocity = torch.zeros((1, 6), dtype=torch.float32, device=unwrapped.device)

    for index, part_name in enumerate(part_names):
        part = getattr(unwrapped, part_name)
        pose_w = poses_w[index : index + 1]
        part.write_root_pose_to_sim(pose_w, env_ids)
        part.write_root_velocity_to_sim(zero_velocity, env_ids)


def _visualize_markers(
    markers: tuple[VisualizationMarkers, VisualizationMarkers] | None,
    unwrapped,
    target_poses_w: torch.Tensor,
) -> None:
    if markers is None:
        return
    top_marker, target_marker = markers
    top_pose_w = unwrapped.assembly_parent_part.data.root_pose_w[0:1]
    top_marker.visualize(top_pose_w[:, :3], top_pose_w[:, 3:7])
    target_marker.visualize(target_poses_w[:, :3], target_poses_w[:, 3:7])


def _print_preview_poses(unwrapped, part_names: tuple[str, ...]) -> None:
    top_pose_w = unwrapped.assembly_parent_part.data.root_pose_w[0]
    print(
        "[INFO]: top world pose "
        f"pos={top_pose_w[:3].detach().cpu().tolist()} quat={top_pose_w[3:7].detach().cpu().tolist()}",
        flush=True,
    )
    for part_name in part_names:
        part_pose_w = getattr(unwrapped, part_name).data.root_pose_w[0:1]
        rel_pos, rel_quat = subtract_frame_transforms(
            top_pose_w[None, :3],
            top_pose_w[None, 3:7],
            part_pose_w[:, :3],
            part_pose_w[:, 3:7],
        )
        print(
            f"[INFO]: {part_name} relative to top "
            f"pos={rel_pos[0].detach().cpu().tolist()} quat={rel_quat[0].detach().cpu().tolist()}",
            flush=True,
        )


def _refresh_preview(
    unwrapped,
    part_names: tuple[str, ...],
    target_indices: tuple[int, ...],
    markers: tuple[VisualizationMarkers, VisualizationMarkers] | None,
) -> None:
    target_poses_w = _target_poses_world(unwrapped)
    selected_target_poses_w = target_poses_w[list(target_indices)]
    _write_part_poses(unwrapped, part_names, selected_target_poses_w)
    unwrapped.scene.write_data_to_sim()
    _visualize_markers(markers, unwrapped, target_poses_w)


def _step_preview(unwrapped) -> None:
    unwrapped._sim_step_counter += 1
    unwrapped.sim.step(render=False)
    is_rendering = unwrapped.sim.has_gui() or unwrapped.sim.has_rtx_sensors()
    if unwrapped._sim_step_counter % unwrapped.cfg.sim.render_interval == 0 and is_rendering:
        unwrapped.sim.render()
    unwrapped.scene.update(dt=unwrapped.physics_dt)


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
        env.reset()
        if getattr(unwrapped.cfg, "assembly_name", None) != "one_leg":
            raise RuntimeError(f"Expected one_leg assembly, got '{getattr(unwrapped.cfg, 'assembly_name', None)}'.")

        part_names, target_indices = _preview_targets(unwrapped)
        markers = None if args_cli.disable_markers else _make_markers()

        for _ in range(args_cli.settle_steps + 1):
            _refresh_preview(unwrapped, part_names, target_indices, markers)
            _step_preview(unwrapped)

        _refresh_preview(unwrapped, part_names, target_indices, markers)
        if args_cli.print_poses:
            _print_preview_poses(unwrapped, part_names)
        if args_cli.mode == "relation_child":
            print(f"[INFO]: success={bool(unwrapped._success()[0].item())}", flush=True)

        print(f"[INFO]: Task: {args_cli.task}", flush=True)
        print(f"[INFO]: Mode: {args_cli.mode}", flush=True)
        print(f"[INFO]: Previewing parts: {', '.join(part_names)}", flush=True)
        print("[INFO]: Close the simulator window or press Ctrl+C to exit.", flush=True)

        while simulation_app.is_running():
            _refresh_preview(unwrapped, part_names, target_indices, markers)
            _step_preview(unwrapped)
    finally:
        env.close()

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    finally:
        simulation_app.close()
