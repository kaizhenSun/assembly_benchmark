# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Replay the zero-residual geometric path for R1 Pro Beam 0-to-2 insertion."""

import argparse

from isaaclab.app import AppLauncher

TASK_ID = "Assembly-Benchmark-Beam02-LeftInsert-Direct-v0"

parser = argparse.ArgumentParser(description="Run the R1 Pro Beam02 analytic insertion baseline.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of parallel insertion environments.")
parser.add_argument("--episodes", type=int, default=1, help="Number of synchronized vector episodes to replay.")
parser.add_argument("--print_interval", type=int, default=16, help="Print live metrics every N policy steps.")
parser.add_argument(
    "--show_path",
    action="store_true",
    help="Render eleven 1 mm sphere markers along each nominal insertion path.",
)
parser.add_argument(
    "--disable_fabric",
    action="store_true",
    default=False,
    help="Disable Fabric for USD synchronization debugging.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import assembly_benchmark.tasks  # noqa: F401
import gymnasium as gym
import torch

from isaaclab.markers import SPHERE_MARKER_CFG, VisualizationMarkers

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg


def _as_float(value) -> float:
    return float(value.item()) if isinstance(value, torch.Tensor) else float(value)


def main() -> bool:
    """Run zero policy residuals and return whether every requested episode passed."""
    if args_cli.episodes <= 0:
        raise ValueError("--episodes must be positive.")
    if args_cli.print_interval <= 0:
        raise ValueError("--print_interval must be positive.")

    env_cfg = parse_env_cfg(
        TASK_ID,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
    )
    env_cfg.seed = 42
    env_cfg.socket_position_noise = (0.0, 0.0, 0.0)
    env = gym.make(TASK_ID, cfg=env_cfg)
    task = env.unwrapped
    zero_residual = torch.zeros(env.action_space.shape, device=task.device)

    print(f"[INFO] Task: {TASK_ID}")
    print(
        f"[INFO] Zero residual follows a {task.cfg.approach_distance * 1000.0:.3f} mm straight path; "
        f"policy rate={1.0 / task.step_dt:.1f} Hz; socket noise is disabled."
    )
    if args_cli.show_path:
        marker_cfg = SPHERE_MARKER_CFG.copy()
        marker_cfg.prim_path = "/Visuals/Beam02Path"
        marker_cfg.markers["sphere"].radius = 0.001
        path_markers = VisualizationMarkers(marker_cfg)
        progress = torch.linspace(0.0, 1.0, 11, device=task.device).view(1, -1, 1)
        path_points = task.path_start_pos_w.unsqueeze(1) + progress * (
            task.nominal_goal_pose_w[:, None, :3] - task.path_start_pos_w.unsqueeze(1)
        )
        path_markers.visualize(path_points.reshape(-1, 3))
        print(
            "[INFO] --show_path: rendered 11 one-millimetre sphere markers per environment; "
            f"env_0 start={task.path_start_pos_w[0].tolist()}, goal={task.nominal_goal_pose_w[0, :3].tolist()}."
        )

    env.reset()
    vector_episodes = 0
    step = 0
    all_passed = True
    previous_progress = None
    max_progress_regression = 0.0
    observed_max_cross_track = 0.0
    try:
        while simulation_app.is_running() and vector_episodes < args_cli.episodes:
            with torch.inference_mode():
                _, reward, terminated, truncated, extras = env.step(zero_residual)
            step += 1
            done = terminated | truncated
            if torch.any(done) and not torch.all(done):
                print(
                    "[FAIL] Vector replay lost synchronization: "
                    f"done_envs={torch.nonzero(done, as_tuple=False).flatten().tolist()}."
                )
                all_passed = False
                break
            episode_ended = torch.all(done)

            if not episode_ended:
                current_progress = task.axial_progress.clone()
                if previous_progress is not None:
                    regression = torch.clamp(previous_progress - current_progress, min=0.0).max().item()
                    max_progress_regression = max(max_progress_regression, regression)
                previous_progress = current_progress
                observed_max_cross_track = max(
                    observed_max_cross_track,
                    task.max_cross_track_distance.max().item(),
                )

            if step % args_cli.print_interval == 0 and not episode_ended:
                print(
                    f"[step {step:04d}] physical_error={task.keypoint_distance.mean().item() * 1000.0:.3f} mm, "
                    f"progress={task.axial_progress.mean().item():.4f}, "
                    f"cross_track={task.deviation_distance.max().item() * 1000.0:.3f} mm, "
                    f"reward={reward.mean().item():.3f}, "
                    f"success_latch={task.insertion_successes.float().mean().item():.3f}, "
                    f"deviation_latch={task.deviations.float().mean().item():.3f}, "
                    f"right_drift={task.right_ee_position_drift.max().item() * 1000.0:.3f} mm/"
                    f"{torch.rad2deg(task.right_ee_orientation_drift.max()).item():.3f} deg"
                )

            if episode_ended:
                vector_episodes += 1
                metrics = extras.get("log", {})
                final_error = _as_float(metrics.get("Metrics/final_true_error", float("inf")))
                final_physical_error = _as_float(metrics.get("Metrics/final_physical_error", float("inf")))
                success = _as_float(metrics.get("Metrics/insertion_successes", 0.0))
                invalid_state = _as_float(metrics.get("Metrics/invalid_state", 1.0))
                final_progress = _as_float(metrics.get("Metrics/axial_progress", 0.0))
                right_position_drift = _as_float(metrics.get("Metrics/right_ee_position_drift", float("inf")))
                right_orientation_drift = _as_float(metrics.get("Metrics/right_ee_orientation_drift", float("inf")))
                max_right_position_drift = _as_float(metrics.get("Metrics/max_right_ee_position_drift", float("inf")))
                max_right_orientation_drift = _as_float(
                    metrics.get("Metrics/max_right_ee_orientation_drift", float("inf"))
                )
                observed_max_cross_track = max(
                    observed_max_cross_track,
                    _as_float(metrics.get("Metrics/max_cross_track_distance", float("inf"))),
                )
                episode_passed = (
                    final_error < task.cfg.success_distance_threshold
                    and success == 1.0
                    and invalid_state == 0.0
                    and max_progress_regression <= 1.0e-4
                    and observed_max_cross_track <= task.cfg.deviation_distance_threshold
                    and right_position_drift <= task.cfg.right_ee_final_position_drift_threshold
                    and right_orientation_drift <= task.cfg.right_ee_final_orientation_drift_threshold
                    and max_right_position_drift <= task.cfg.right_ee_position_drift_threshold
                    and max_right_orientation_drift <= task.cfg.right_ee_orientation_drift_threshold
                )
                all_passed &= episode_passed
                result = "PASS" if episode_passed else "FAIL"
                print(
                    f"[{result} episode {vector_episodes}] final_error={final_error * 1000.0:.3f} mm, "
                    f"physical_error={final_physical_error * 1000.0:.3f} mm, "
                    f"progress={final_progress:.4f}, max_progress_regression={max_progress_regression:.6f}, "
                    f"max_cross_track={observed_max_cross_track * 1000.0:.3f} mm, success={success:.3f}, "
                    f"invalid_state={invalid_state:.3f}, "
                    f"right_final={right_position_drift * 1000.0:.3f} mm/"
                    f"{torch.rad2deg(torch.tensor(right_orientation_drift)).item():.3f} deg, "
                    f"right_max={max_right_position_drift * 1000.0:.3f} mm/"
                    f"{torch.rad2deg(torch.tensor(max_right_orientation_drift)).item():.3f} deg"
                )
                previous_progress = None
                max_progress_regression = 0.0
                observed_max_cross_track = 0.0
    finally:
        env.close()
    return all_passed and vector_episodes == args_cli.episodes


if __name__ == "__main__":
    try:
        replay_passed = main()
    finally:
        simulation_app.close()
    raise SystemExit(0 if replay_passed else 1)
