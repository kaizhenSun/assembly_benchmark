# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Replay Fabrica's zero-residual fixed-plug insertion baseline."""

import argparse
import math
import os
import sys

from isaaclab.app import AppLauncher

TASK_ID = "Assembly-Benchmark-FabricaFixPlugTaskAssemble-Direct-v0"

parser = argparse.ArgumentParser(description="Run the Fabrica fixed-plug analytic insertion baseline.")
parser.add_argument("--num_envs", type=int, default=4, help="Number of parallel insertion environments.")
parser.add_argument("--episodes", type=int, default=1, help="Number of synchronized vector episodes to replay.")
parser.add_argument("--print_interval", type=int, default=16, help="Print live metrics every N policy steps.")
parser.add_argument("--show_path", action="store_true", help="Render markers along every nominal insertion path.")
parser.add_argument("--disable_fabric", action="store_true", help="Disable Fabric for USD synchronization debugging.")
parser.add_argument(
    "--socket_noise",
    type=float,
    default=0.0,
    help="Uniform socket-position noise half-width in metres; zero is the acceptance baseline.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import assembly_benchmark.tasks  # noqa: F401, E402
import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

from isaaclab.markers import SPHERE_MARKER_CFG, VisualizationMarkers  # noqa: E402

import isaaclab_tasks  # noqa: F401, E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402


def _as_float(value) -> float:
    return float(value.item()) if isinstance(value, torch.Tensor) else float(value)


def main() -> bool:
    """Run zero policy residuals and return whether every represented relation passed."""
    if args_cli.episodes <= 0 or args_cli.print_interval <= 0:
        raise ValueError("--episodes and --print_interval must be positive.")
    if args_cli.socket_noise < 0.0:
        raise ValueError("--socket_noise must be non-negative.")

    env_cfg = parse_env_cfg(
        TASK_ID,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
    )
    env_cfg.seed = 42
    env_cfg.socket_position_noise = (args_cli.socket_noise,) * 3
    relation_count = len(env_cfg.relation_keys)
    if args_cli.num_envs < relation_count:
        raise ValueError(f"--num_envs must be at least {relation_count} to cover every relation.")

    env = gym.make(TASK_ID, cfg=env_cfg)
    task = env.unwrapped
    zero_residual = torch.zeros(env.action_space.shape, device=task.device)
    print(f"[INFO] Task: {TASK_ID}")
    print(
        f"[INFO] Relations={task.relation_keys}; policy rate={1.0 / task.step_dt:.1f} Hz; "
        f"socket noise=±{args_cli.socket_noise * 1000.0:.1f} mm."
    )

    if args_cli.show_path:
        marker_cfg = SPHERE_MARKER_CFG.copy()
        marker_cfg.prim_path = "/Visuals/FabricaInsertionPaths"
        marker_cfg.markers["sphere"].radius = 0.001
        path_markers = VisualizationMarkers(marker_cfg)
        path_markers.visualize(task.insertion_path_w.reshape(-1, 3))
        print(f"[INFO] Rendered {task.insertion_path_w.shape[1]} path markers per environment.")

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
                print(f"[FAIL] Vector replay lost synchronization: {torch.nonzero(done).flatten().tolist()}.")
                return False
            episode_ended = bool(torch.all(done))

            if not episode_ended:
                current_progress = task.axial_progress.clone()
                if previous_progress is not None:
                    regression = torch.clamp(previous_progress - current_progress, min=0.0).max().item()
                    max_progress_regression = max(max_progress_regression, regression)
                previous_progress = current_progress
                observed_max_cross_track = max(observed_max_cross_track, task.max_cross_track_distance.max().item())

            if step % args_cli.print_interval == 0 and not episode_ended:
                print(
                    f"[step {step:04d}] true_error={task.keypoint_distance.mean().item() * 1000.0:.3f} mm, "
                    f"progress={task.axial_progress.mean().item():.4f}, "
                    f"cross_track={task.deviation_distance.max().item() * 1000.0:.3f} mm, "
                    f"reward={reward.mean().item():.3f}"
                )

            if episode_ended:
                vector_episodes += 1
                metrics = extras.get("log", {})
                episode_passed = True
                for relation_key in task.relation_keys:
                    prefix = f"Relations/{relation_key}"
                    final_error = _as_float(metrics.get(f"{prefix}/final_true_error", float("inf")))
                    success = _as_float(metrics.get(f"{prefix}/insertion_successes", 0.0))
                    invalid = _as_float(metrics.get(f"{prefix}/invalid_state", 1.0))
                    cross_track = _as_float(metrics.get(f"{prefix}/max_cross_track_distance", float("inf")))
                    metrics_are_finite = all(
                        math.isfinite(value) for value in (final_error, success, invalid, cross_track)
                    )
                    if args_cli.socket_noise == 0.0:
                        relation_passed = (
                            metrics_are_finite
                            and final_error < task.cfg.success_distance_threshold
                            and success == 1.0
                            and invalid == 0.0
                            and cross_track <= task.cfg.deviation_distance_threshold
                        )
                    else:
                        relation_passed = metrics_are_finite and invalid == 0.0
                    episode_passed &= relation_passed
                    print(
                        f"[{'PASS' if relation_passed else 'FAIL'} {relation_key}] "
                        f"final_error={final_error * 1000.0:.3f} mm, success={success:.3f}, "
                        f"cross_track={cross_track * 1000.0:.3f} mm, invalid={invalid:.3f}"
                    )
                if args_cli.socket_noise == 0.0:
                    episode_passed &= observed_max_cross_track <= task.cfg.deviation_distance_threshold
                all_passed &= episode_passed
                print(
                    f"[{'PASS' if episode_passed else 'FAIL'} episode {vector_episodes}] "
                    f"max_progress_regression={max_progress_regression:.6f}, "
                    f"max_cross_track={observed_max_cross_track * 1000.0:.3f} mm"
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
        sys.stdout.flush()
        sys.stderr.flush()
        if "replay_passed" in locals() and not replay_passed:
            os._exit(1)
        simulation_app.close()
    raise SystemExit(0)
