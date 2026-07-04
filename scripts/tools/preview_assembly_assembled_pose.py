# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Preview registered assembly target poses by placing parts directly.

This script is a visual/debug helper. It loads one registered assembly scene,
keeps each relation parent at its reset pose, and writes relation child root
poses from the assembly target poses so target frames can be inspected without
running a manipulation policy.

.. code-block:: bash

    python scripts/tools/preview_assembly_assembled_pose.py --assembly one_leg --num_envs 1 --device cuda:0
    python scripts/tools/preview_assembly_assembled_pose.py --assembly chair --num_envs 1 --device cuda:0
    python scripts/tools/preview_assembly_assembled_pose.py --assembly chair --physics_mode dynamic

"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

from isaaclab.app import AppLauncher


ONE_LEG_FULL_TARGET_PART_NAMES = (
    "square_table_leg1",
    "square_table_leg2",
    "square_table_leg3",
    "square_table_leg4",
)


def _assembly_class_prefix(assembly_name: str) -> str:
    """Convert an assembly registry name to a Python class/task component."""
    return "".join(part.capitalize() for part in assembly_name.replace("-", "_").split("_"))


def _assembly_task_id(assembly_name: str) -> str:
    """Return the explicit Isaac Lab task id for an assembly."""
    return f"Assembly-Benchmark-{_assembly_class_prefix(assembly_name)}-Direct-v0"


parser = argparse.ArgumentParser(description="Preview assembled target poses for a registered assembly.")
parser.add_argument("--assembly", type=str, default="one_leg", help="Registered assembly name to preview.")
parser.add_argument(
    "--task",
    type=str,
    default=None,
    help="Task to load. Defaults to the explicit task id generated from --assembly.",
)
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments. Only 1 is supported.")
parser.add_argument(
    "--disable_fabric",
    action="store_true",
    default=False,
    help="Debug/compatibility option: disable Fabric and use USD I/O, which may desync GUI mesh updates.",
)
parser.add_argument(
    "--mode",
    choices=("auto", "all_relations", "relation_child", "one_leg_full_targets"),
    default="auto",
    help="Preview all relation children, one relation child, or the one_leg four-target layout.",
)
parser.add_argument("--relation_index", type=int, default=0, help="Relation index used by relation_child mode.")
parser.add_argument("--target_index", type=int, default=0, help="Target index used by relation_child mode.")
parser.add_argument(
    "--physics_mode",
    choices=("ghost", "dynamic"),
    default="ghost",
    help="Use collision-free kinematic assembly parts for visual preview, or keep dynamic physics.",
)
parser.add_argument("--settle_steps", type=int, default=30, help="Steps to refresh the assembled preview initially.")
parser.add_argument("--marker_scale", type=float, default=0.03, help="Scale of parent/target frame markers.")
parser.add_argument("--disable_markers", action="store_true", help="Disable parent and target frame markers.")
parser.add_argument("--print_poses", action="store_true", help="Print assembled world and relative poses.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

args_cli.enable_cameras = True
if args_cli.task is None:
    args_cli.task = _assembly_task_id(args_cli.assembly)

if args_cli.num_envs != 1:
    raise ValueError("Assembly assembled pose preview currently supports only --num_envs 1.")
if args_cli.settle_steps < 0:
    raise ValueError("--settle_steps must be non-negative.")
if args_cli.marker_scale <= 0.0:
    raise ValueError("--marker_scale must be positive.")
if args_cli.relation_index < 0:
    raise ValueError("--relation_index must be non-negative.")
if args_cli.target_index < 0:
    raise ValueError("--target_index must be non-negative.")

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch

import isaaclab.sim as sim_utils
import isaaclab_tasks  # noqa: F401
from isaaclab.markers import VisualizationMarkers
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.utils.math import combine_frame_transforms, subtract_frame_transforms
from isaaclab_tasks.utils import parse_env_cfg

from assembly_benchmark.assembly import AssemblyRelationSpec, AssemblySpec, make_assembly

import assembly_benchmark.tasks  # noqa: F401


@dataclass(frozen=True)
class PreviewPlacement:
    """One child placement to refresh during preview."""

    relation_index: int
    parent: str
    child: str
    target_index: int
    target_pos: tuple[float, float, float]
    target_quat: tuple[float, float, float, float]

    @property
    def relation_label(self) -> str:
        """Return a compact relation label."""
        return f"{self.parent}->{self.child}"


def _configure_preview_physics(env_cfg) -> tuple[str, ...]:
    """Configure assembly parts for this preview run."""
    if args_cli.physics_mode == "dynamic":
        return ()

    part_names = tuple(env_cfg.assembly_reset_part_names)
    for part_name in part_names:
        if not hasattr(env_cfg.scene, part_name):
            raise RuntimeError(f"Preview physics could not find scene part '{part_name}' in env cfg.")

        part_cfg = getattr(env_cfg.scene, part_name)
        spawn_cfg = getattr(part_cfg, "spawn", None)
        if spawn_cfg is None:
            raise RuntimeError(f"Preview physics could not find spawn cfg for scene part '{part_name}'.")

        spawn_cfg.collision_props = sim_utils.CollisionPropertiesCfg(collision_enabled=False)
        if spawn_cfg.rigid_props is None:
            spawn_cfg.rigid_props = sim_utils.RigidBodyPropertiesCfg()
        spawn_cfg.rigid_props.disable_gravity = True
        spawn_cfg.rigid_props.kinematic_enabled = True

    return part_names


def _print_physics_mode(ghost_part_names: tuple[str, ...]) -> None:
    print(f"[INFO]: Preview physics mode: {args_cli.physics_mode}", flush=True)
    if args_cli.physics_mode == "ghost":
        print(
            "[INFO]: Ghost mode is visual/test-only: collisions are disabled for assembly reset parts.",
            flush=True,
        )
        print(f"[INFO]: Ghosted assembly parts: {', '.join(ghost_part_names)}", flush=True)
    else:
        print("[INFO]: Dynamic mode keeps assembly collisions and rigid-body physics enabled.", flush=True)


def _marker_prim_component(assembly_name: str) -> str:
    """Return a USD-prim-friendly path component."""
    return assembly_name.replace("-", "_")


def _make_markers(assembly_name: str) -> tuple[VisualizationMarkers, VisualizationMarkers]:
    parent_marker_cfg = FRAME_MARKER_CFG.copy()
    parent_marker_cfg.markers["frame"].scale = (
        args_cli.marker_scale,
        args_cli.marker_scale,
        args_cli.marker_scale,
    )

    target_marker_cfg = FRAME_MARKER_CFG.copy()
    target_scale = args_cli.marker_scale * 1.25
    target_marker_cfg.markers["frame"].scale = (target_scale, target_scale, target_scale)

    prim_component = _marker_prim_component(assembly_name)
    parent_marker = VisualizationMarkers(
        parent_marker_cfg.replace(prim_path=f"/Visuals/{prim_component}_preview_parent_frames")
    )
    target_marker = VisualizationMarkers(
        target_marker_cfg.replace(prim_path=f"/Visuals/{prim_component}_preview_target_frames")
    )
    return parent_marker, target_marker


def _resolve_mode(assembly: AssemblySpec) -> str:
    """Resolve auto mode for the selected assembly."""
    if args_cli.mode != "auto":
        return args_cli.mode
    if assembly.name == "one_leg":
        return "one_leg_full_targets"
    return "all_relations"


def _placement_from_relation(
    relation: AssemblyRelationSpec,
    relation_index: int,
    target_index: int,
    *,
    child: str | None = None,
) -> PreviewPlacement:
    """Create one preview placement from a relation target pose."""
    if target_index < 0 or target_index >= len(relation.target_poses):
        raise ValueError(
            f"--target_index must be in [0, {len(relation.target_poses) - 1}] for relation "
            f"{relation.parent}->{relation.child}, got {target_index}."
        )

    target_pose = relation.target_poses[target_index]
    return PreviewPlacement(
        relation_index=relation_index,
        parent=relation.parent,
        child=relation.child if child is None else child,
        target_index=target_index,
        target_pos=target_pose.pos,
        target_quat=target_pose.quat,
    )


def _preview_placements(assembly: AssemblySpec, mode: str) -> tuple[PreviewPlacement, ...]:
    """Return part placements for the selected preview mode."""
    if mode == "all_relations":
        return tuple(
            _placement_from_relation(relation, relation_index, relation.default_target_index)
            for relation_index, relation in enumerate(assembly.assembly_relations)
        )

    if mode == "relation_child":
        if args_cli.relation_index >= len(assembly.assembly_relations):
            raise ValueError(
                f"--relation_index must be in [0, {len(assembly.assembly_relations) - 1}] "
                f"for assembly '{assembly.name}', got {args_cli.relation_index}."
            )
        relation = assembly.assembly_relations[args_cli.relation_index]
        return (_placement_from_relation(relation, args_cli.relation_index, args_cli.target_index),)

    if mode == "one_leg_full_targets":
        if assembly.name != "one_leg":
            raise ValueError("one_leg_full_targets mode is only valid for the 'one_leg' assembly.")
        relation = assembly.primary_relation
        if len(relation.target_poses) < len(ONE_LEG_FULL_TARGET_PART_NAMES):
            raise RuntimeError(
                "one_leg_full_targets mode requires at least "
                f"{len(ONE_LEG_FULL_TARGET_PART_NAMES)} target poses, got {len(relation.target_poses)}."
            )
        return tuple(
            _placement_from_relation(relation, 0, target_index, child=part_name)
            for target_index, part_name in enumerate(ONE_LEG_FULL_TARGET_PART_NAMES)
        )

    raise ValueError(f"Unsupported preview mode '{mode}'.")


def _validate_scene_parts(unwrapped, placements: tuple[PreviewPlacement, ...]) -> None:
    """Validate that selected placement parts exist in the loaded scene."""
    duplicate_children = sorted(
        {
            placement.child
            for placement in placements
            if sum(other.child == placement.child for other in placements) > 1
        }
    )
    if duplicate_children:
        duplicates = ", ".join(duplicate_children)
        raise RuntimeError(f"Preview placements write the same child more than once: {duplicates}.")

    missing = sorted(
        {
            part_name
            for placement in placements
            for part_name in (placement.parent, placement.child)
            if not hasattr(unwrapped, part_name)
        }
    )
    if missing:
        raise RuntimeError(f"Preview requires missing scene parts: {', '.join(missing)}.")


def _warn_duplicate_selected_targets(placements: tuple[PreviewPlacement, ...], mode: str) -> None:
    """Warn if all-relations mode selects the same target pose for multiple children."""
    if mode != "all_relations":
        return

    children_by_target: dict[tuple[str, tuple[float, ...], tuple[float, ...]], list[str]] = {}
    for placement in placements:
        key = (placement.parent, placement.target_pos, placement.target_quat)
        children_by_target.setdefault(key, []).append(placement.child)

    for (parent, target_pos, target_quat), children in children_by_target.items():
        if len(children) < 2:
            continue
        print(
            "[WARN]: "
            f"Children of {parent} share the selected target pose "
            f"pos={target_pos} quat={target_quat}: {', '.join(children)}.",
            flush=True,
        )


def _placement_target_pose_world(unwrapped, placement: PreviewPlacement) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the parent pose and target child pose in world frame."""
    parent_pose_w = getattr(unwrapped, placement.parent).data.root_pose_w[0:1]
    target_pos = torch.tensor((placement.target_pos,), dtype=torch.float32, device=unwrapped.device)
    target_quat = torch.tensor((placement.target_quat,), dtype=torch.float32, device=unwrapped.device)
    target_pos_w, target_quat_w = combine_frame_transforms(
        parent_pose_w[:, :3],
        parent_pose_w[:, 3:7],
        target_pos,
        target_quat,
    )
    return parent_pose_w, torch.cat((target_pos_w, target_quat_w), dim=-1)


def _target_poses_world(
    unwrapped,
    placements: tuple[PreviewPlacement, ...],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return parent and target poses for all selected placements."""
    parent_poses_w = []
    target_poses_w = []
    for placement in placements:
        parent_pose_w, target_pose_w = _placement_target_pose_world(unwrapped, placement)
        parent_poses_w.append(parent_pose_w)
        target_poses_w.append(target_pose_w)
    return torch.cat(parent_poses_w, dim=0), torch.cat(target_poses_w, dim=0)


def _unique_parent_poses_world(
    placements: tuple[PreviewPlacement, ...],
    parent_poses_w: torch.Tensor,
) -> torch.Tensor:
    """Return one marker pose per parent part."""
    seen_parents: set[str] = set()
    unique_indices: list[int] = []
    for index, placement in enumerate(placements):
        if placement.parent in seen_parents:
            continue
        seen_parents.add(placement.parent)
        unique_indices.append(index)

    if len(unique_indices) == parent_poses_w.shape[0]:
        return parent_poses_w

    indices = torch.tensor(unique_indices, dtype=torch.long, device=parent_poses_w.device)
    return parent_poses_w.index_select(0, indices)


def _write_part_poses(
    unwrapped,
    placements: tuple[PreviewPlacement, ...],
    target_poses_w: torch.Tensor,
) -> None:
    """Write selected child poses into sim and clear velocities."""
    env_ids = torch.tensor([0], dtype=torch.long, device=unwrapped.device)
    zero_velocity = torch.zeros((1, 6), dtype=torch.float32, device=unwrapped.device)

    for index, placement in enumerate(placements):
        part = getattr(unwrapped, placement.child)
        pose_w = target_poses_w[index : index + 1]
        part.write_root_pose_to_sim(pose_w, env_ids)
        part.write_root_velocity_to_sim(zero_velocity, env_ids)


def _visualize_markers(
    markers: tuple[VisualizationMarkers, VisualizationMarkers] | None,
    placements: tuple[PreviewPlacement, ...],
    parent_poses_w: torch.Tensor,
    target_poses_w: torch.Tensor,
) -> None:
    if markers is None:
        return
    parent_marker, target_marker = markers
    unique_parent_poses_w = _unique_parent_poses_world(placements, parent_poses_w)
    parent_marker.visualize(unique_parent_poses_w[:, :3], unique_parent_poses_w[:, 3:7])
    target_marker.visualize(target_poses_w[:, :3], target_poses_w[:, 3:7])


def _print_preview_selection(
    task: str,
    assembly: AssemblySpec,
    mode: str,
    placements: tuple[PreviewPlacement, ...],
) -> None:
    print(f"[INFO]: Task: {task}", flush=True)
    print(f"[INFO]: Assembly: {assembly.name}", flush=True)
    print(f"[INFO]: Mode: {mode}", flush=True)
    for placement in placements:
        print(
            "[INFO]: "
            f"relation[{placement.relation_index}] {placement.relation_label} "
            f"target_index={placement.target_index} "
            f"target_pos={placement.target_pos} target_quat={placement.target_quat}",
            flush=True,
        )


def _print_preview_poses(unwrapped, placements: tuple[PreviewPlacement, ...]) -> None:
    print(f"[INFO]: assembly={unwrapped.cfg.assembly_name}", flush=True)
    for placement in placements:
        parent_pose_w = getattr(unwrapped, placement.parent).data.root_pose_w[0:1]
        child_pose_w = getattr(unwrapped, placement.child).data.root_pose_w[0:1]
        rel_pos, rel_quat = subtract_frame_transforms(
            parent_pose_w[:, :3],
            parent_pose_w[:, 3:7],
            child_pose_w[:, :3],
            child_pose_w[:, 3:7],
        )
        print(
            "[INFO]: "
            f"relation[{placement.relation_index}] {placement.relation_label} "
            f"target_index={placement.target_index} "
            f"target_pos={placement.target_pos} target_quat={placement.target_quat}",
            flush=True,
        )
        print(
            "[INFO]: "
            f"{placement.child} world "
            f"pos={child_pose_w[0, :3].detach().cpu().tolist()} "
            f"quat={child_pose_w[0, 3:7].detach().cpu().tolist()}",
            flush=True,
        )
        print(
            "[INFO]: "
            f"{placement.child} relative to {placement.parent} "
            f"pos={rel_pos[0].detach().cpu().tolist()} "
            f"quat={rel_quat[0].detach().cpu().tolist()}",
            flush=True,
        )


def _refresh_preview(
    unwrapped,
    placements: tuple[PreviewPlacement, ...],
    markers: tuple[VisualizationMarkers, VisualizationMarkers] | None,
) -> None:
    parent_poses_w, target_poses_w = _target_poses_world(unwrapped, placements)
    _write_part_poses(unwrapped, placements, target_poses_w)
    unwrapped.scene.write_data_to_sim()
    _visualize_markers(markers, placements, parent_poses_w, target_poses_w)


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
    ghost_part_names = _configure_preview_physics(env_cfg)
    _print_physics_mode(ghost_part_names)

    env = gym.make(args_cli.task, cfg=env_cfg)
    unwrapped = env.unwrapped

    try:
        env.reset()

        assembly_name = getattr(unwrapped.cfg, "assembly_name", None)
        if not assembly_name:
            raise RuntimeError("Loaded task does not expose cfg.assembly_name.")
        if assembly_name != args_cli.assembly:
            print(
                f"[WARN]: Loaded task assembly '{assembly_name}' differs from --assembly '{args_cli.assembly}'. "
                "Using the loaded task assembly.",
                flush=True,
            )
        assembly = make_assembly(assembly_name)
        mode = _resolve_mode(assembly)
        placements = _preview_placements(assembly, mode)
        _validate_scene_parts(unwrapped, placements)
        _warn_duplicate_selected_targets(placements, mode)
        _print_preview_selection(args_cli.task, assembly, mode, placements)

        markers = None if args_cli.disable_markers else _make_markers(assembly.name)

        for _ in range(args_cli.settle_steps + 1):
            _refresh_preview(unwrapped, placements, markers)
            _step_preview(unwrapped)

        _refresh_preview(unwrapped, placements, markers)
        if args_cli.print_poses:
            _print_preview_poses(unwrapped, placements)
        if mode == "relation_child" and placements[0].relation_index == 0:
            print(f"[INFO]: primary_success={bool(unwrapped._success()[0].item())}", flush=True)

        print(
            "[INFO]: Previewing placements: "
            + ", ".join(
                f"{placement.relation_label}[target={placement.target_index}]" for placement in placements
            ),
            flush=True,
        )
        print("[INFO]: Close the simulator window or press Ctrl+C to exit.", flush=True)

        while simulation_app.is_running():
            _refresh_preview(unwrapped, placements, markers)
            _step_preview(unwrapped)
    finally:
        env.close()

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    finally:
        simulation_app.close()
