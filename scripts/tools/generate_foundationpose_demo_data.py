# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Generate a FoundationPose-compatible RGB-D demo sequence from an assembly scene."""

from __future__ import annotations

import argparse
import ast
import json
import re
import shlex
import shutil
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Generate static FoundationPose demo data from Assembly Benchmark.")
parser.add_argument("--task", default="Assembly-Benchmark-OneLeg-Direct-v0", help="Isaac Lab task ID to capture.")
parser.add_argument(
    "--target_part",
    default="square_table_leg4",
    help="Registered assembly part whose semantic mask and mesh should be exported.",
)
parser.add_argument(
    "--output_dir",
    type=Path,
    default=None,
    help="Output scene directory. Defaults to logs/foundationpose/<assembly>_<part>_<timestamp>.",
)
parser.add_argument("--num_frames", type=int, default=30, help="Number of RGB-D frames to save.")
parser.add_argument("--warmup_steps", type=int, default=60, help="Action steps to run before capture.")
parser.add_argument("--frame_interval", type=int, default=2, help="Action steps between saved frames.")
parser.add_argument("--width", type=int, default=640, help="Captured image width in pixels.")
parser.add_argument("--height", type=int, default=480, help="Captured image height in pixels.")
parser.add_argument(
    "--mask_mode",
    choices=("first", "all"),
    default="first",
    help="Save only the initialization mask or one mask for every frame.",
)
parser.add_argument(
    "--disable_fabric",
    action="store_true",
    help="Disable Fabric and use USD I/O operations.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

if args_cli.num_frames <= 0:
    parser.error("--num_frames must be positive")
if args_cli.warmup_steps < 0:
    parser.error("--warmup_steps must be non-negative")
if args_cli.frame_interval <= 0:
    parser.error("--frame_interval must be positive")
if args_cli.width <= 0 or args_cli.height <= 0:
    parser.error("--width and --height must be positive")

# This script cannot operate without RTX camera sensors. Do not require callers to
# remember the generic AppLauncher flag.
args_cli.enable_cameras = True
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Imports below this point require the running simulator."""

import assembly_benchmark.tasks  # noqa: F401
import gymnasium as gym
import imageio.v3 as iio
import numpy as np
import torch
from assembly_benchmark.assembly import make_assembly

from isaaclab.utils.math import matrix_from_quat

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg

_CAMERA_NAME = "head_camera"
_RGB_DATA_TYPE = "rgb"
_DEPTH_DATA_TYPE = "distance_to_image_plane"
_SEMANTIC_DATA_TYPE = "semantic_segmentation"
_UINT16_MAX = np.iinfo(np.uint16).max


def _slug(value: str) -> str:
    """Return a path-safe lowercase identifier."""
    slug = re.sub(r"[^a-zA-Z0-9_-]+", "_", value).strip("_").lower()
    return slug or "scene"


def _default_output_dir(assembly_name: str, target_part: str) -> Path:
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    return Path("logs/foundationpose") / f"{_slug(assembly_name)}_{_slug(target_part)}_{timestamp}"


def _single_camera_array(value: torch.Tensor, name: str) -> np.ndarray:
    """Move the sole environment's camera buffer to a NumPy array."""
    if value.ndim == 0 or value.shape[0] != 1:
        raise RuntimeError(f"Expected '{name}' to have one leading environment dimension, got {tuple(value.shape)}.")
    return value[0].detach().cpu().numpy()


def _rgb_uint8(value: torch.Tensor) -> np.ndarray:
    rgb = _single_camera_array(value, _RGB_DATA_TYPE)
    if rgb.ndim != 3 or rgb.shape[-1] < 3:
        raise RuntimeError(f"Expected RGB/RGBA camera output, got shape {rgb.shape}.")
    rgb = rgb[..., :3]
    if np.issubdtype(rgb.dtype, np.floating):
        finite_max = float(np.nanmax(rgb)) if rgb.size else 1.0
        scale = 255.0 if finite_max <= 1.0 else 1.0
        rgb = np.nan_to_num(rgb, nan=0.0, posinf=255.0, neginf=0.0) * scale
    return np.clip(rgb, 0, 255).astype(np.uint8)


def _depth_mm_uint16(value: torch.Tensor) -> np.ndarray:
    depth_m = _single_camera_array(value, _DEPTH_DATA_TYPE)
    if depth_m.ndim == 3 and depth_m.shape[-1] == 1:
        depth_m = depth_m[..., 0]
    if depth_m.ndim != 2:
        raise RuntimeError(f"Expected a single-channel depth image, got shape {depth_m.shape}.")

    valid = np.isfinite(depth_m) & (depth_m >= 0.001) & (depth_m <= _UINT16_MAX / 1000.0)
    depth_mm = np.zeros(depth_m.shape, dtype=np.uint16)
    depth_mm[valid] = np.rint(depth_m[valid] * 1000.0).astype(np.uint16)
    return depth_mm


def _semantic_ids(value: torch.Tensor) -> np.ndarray:
    semantic = _single_camera_array(value, _SEMANTIC_DATA_TYPE)
    if semantic.ndim == 3 and semantic.shape[-1] == 1:
        semantic = semantic[..., 0]
    if semantic.ndim != 2:
        raise RuntimeError(f"Expected uncolorized semantic IDs, got shape {semantic.shape}.")
    return semantic


def _decode_maybe_serialized(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    stripped = value.strip()
    if not stripped:
        return value
    for decoder in (json.loads, ast.literal_eval):
        try:
            return decoder(stripped)
        except (ValueError, SyntaxError, TypeError, json.JSONDecodeError):
            pass
    return value


def _flatten_labels(value: Any) -> set[str]:
    value = _decode_maybe_serialized(value)
    if isinstance(value, dict):
        labels: set[str] = set()
        for nested in value.values():
            labels.update(_flatten_labels(nested))
        return labels
    if isinstance(value, (list, tuple, set)):
        labels = set()
        for nested in value:
            labels.update(_flatten_labels(nested))
        return labels
    return {str(value)}


def _target_semantic_ids(info: Any, target_part: str) -> tuple[int, ...]:
    """Resolve Replicator's semantic mapping to IDs for one exact class label."""
    info = _decode_maybe_serialized(info)
    if not isinstance(info, dict):
        raise RuntimeError(f"Semantic camera metadata is not a mapping: {type(info).__name__}.")
    id_to_labels = _decode_maybe_serialized(info.get("idToLabels", info))
    if not isinstance(id_to_labels, dict):
        raise RuntimeError("Semantic camera metadata does not contain an 'idToLabels' mapping.")

    target_ids: list[int] = []
    for raw_id, raw_labels in id_to_labels.items():
        if target_part not in _flatten_labels(raw_labels):
            continue
        try:
            target_ids.append(int(raw_id))
        except (TypeError, ValueError) as exc:
            raise RuntimeError(f"Semantic ID {raw_id!r} for '{target_part}' is not an integer.") from exc

    if not target_ids:
        available = sorted({label for labels in id_to_labels.values() for label in _flatten_labels(labels)})
        preview = ", ".join(available[:20])
        raise RuntimeError(f"Semantic label '{target_part}' was not found. Available labels include: {preview}")
    return tuple(sorted(set(target_ids)))


def _target_mask(semantic_ids: np.ndarray, target_ids: tuple[int, ...]) -> np.ndarray:
    return (np.isin(semantic_ids, target_ids).astype(np.uint8) * 255).astype(np.uint8)


def _validate_frame(rgb: np.ndarray, depth_mm: np.ndarray, mask: np.ndarray, *, require_mask: bool) -> None:
    if rgb.shape[:2] != depth_mm.shape or depth_mm.shape != mask.shape:
        raise RuntimeError(
            f"RGB, depth, and mask dimensions differ: rgb={rgb.shape}, depth={depth_mm.shape}, mask={mask.shape}."
        )
    if rgb.dtype != np.uint8 or depth_mm.dtype != np.uint16 or mask.dtype != np.uint8:
        raise RuntimeError(f"Unexpected output dtypes: rgb={rgb.dtype}, depth={depth_mm.dtype}, mask={mask.dtype}.")
    if require_mask and not np.any(mask):
        raise RuntimeError("The initialization mask is empty; the target part is not visible from the head camera.")
    if not np.any(depth_mm):
        raise RuntimeError("The depth image contains no valid uint16 millimeter samples.")


def _object_in_camera_pose(camera, target_asset) -> np.ndarray:
    """Return the target-local to OpenCV-camera transform as a 4x4 matrix."""
    root_pose_w = getattr(getattr(target_asset, "data", None), "root_pose_w", None)
    if root_pose_w is None:
        raise RuntimeError(
            f"Target scene asset '{args_cli.target_part}' does not expose a world root pose; "
            "ground-truth pose export requires a rigid assembly part."
        )
    if root_pose_w.shape[0] != 1:
        raise RuntimeError(f"Expected one target pose, got shape {tuple(root_pose_w.shape)}.")

    camera_pos_w = camera.data.pos_w[0]
    camera_quat_w_cv = camera.data.quat_w_ros[0]
    object_pos_w = root_pose_w[0, :3]
    object_quat_w = root_pose_w[0, 3:7]
    camera_rot_w_cv = matrix_from_quat(camera_quat_w_cv.unsqueeze(0))[0]
    object_rot_w = matrix_from_quat(object_quat_w.unsqueeze(0))[0]

    camera_rot_object = camera_rot_w_cv.transpose(0, 1) @ object_rot_w
    camera_pos_object = camera_rot_w_cv.transpose(0, 1) @ (object_pos_w - camera_pos_w)
    object_in_camera = torch.eye(4, dtype=root_pose_w.dtype, device=root_pose_w.device)
    object_in_camera[:3, :3] = camera_rot_object
    object_in_camera[:3, 3] = camera_pos_object
    pose = object_in_camera.detach().cpu().numpy().astype(np.float64)
    if pose.shape != (4, 4) or not np.all(np.isfinite(pose)):
        raise RuntimeError(f"Computed an invalid object-in-camera pose: {pose}")
    return pose


def _mesh_from_part_urdf(urdf_path: Path) -> Path:
    """Find the first visual OBJ referenced by an assembly part URDF."""
    try:
        root = ET.parse(urdf_path).getroot()
    except (ET.ParseError, OSError) as exc:
        raise RuntimeError(f"Could not read target part URDF: {urdf_path}") from exc
    filenames = [node.get("filename") for node in root.findall(".//visual/geometry/mesh")]
    obj_paths = []
    for filename in filenames:
        if not filename or filename.startswith("package://"):
            continue
        candidate = (urdf_path.parent / filename).resolve()
        if candidate.suffix.lower() == ".obj":
            obj_paths.append(candidate)
    if not obj_paths:
        raise RuntimeError(f"No visual OBJ mesh was found in {urdf_path}.")
    if not obj_paths[0].is_file():
        raise RuntimeError(f"Target OBJ mesh does not exist: {obj_paths[0]}")
    return obj_paths[0]


def _rewrite_material(material_path: Path, output_path: Path, texture_dir: Path) -> None:
    """Copy an MTL while making its referenced textures self-contained."""
    rewritten: list[str] = []
    copied_names: dict[str, Path] = {}
    for line in material_path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped.startswith("map_"):
            rewritten.append(line)
            continue
        tokens = shlex.split(stripped)
        if len(tokens) < 2:
            rewritten.append(line)
            continue
        source_texture = (material_path.parent / tokens[-1]).resolve()
        if not source_texture.is_file():
            raise RuntimeError(f"Material texture does not exist: {source_texture}")
        previous_source = copied_names.get(source_texture.name)
        if previous_source is not None and previous_source != source_texture:
            raise RuntimeError(f"Two material textures have the same filename: {source_texture.name}")
        copied_names[source_texture.name] = source_texture
        texture_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_texture, texture_dir / source_texture.name)
        tokens[-1] = f"textures/{source_texture.name}"
        rewritten.append(" ".join(tokens))
    output_path.write_text("\n".join(rewritten) + "\n", encoding="utf-8")


def _package_mesh(source_obj: Path, mesh_dir: Path) -> Path:
    """Copy an OBJ, its sole MTL, and referenced textures under standard names."""
    obj_lines = source_obj.read_text(encoding="utf-8").splitlines()
    material_refs = []
    for line in obj_lines:
        tokens = shlex.split(line.strip())
        if tokens and tokens[0] == "mtllib":
            material_refs.extend(tokens[1:])
    if len(material_refs) > 1:
        raise RuntimeError(f"Expected at most one MTL reference in {source_obj}, found {material_refs}.")

    mesh_dir.mkdir(parents=True, exist_ok=False)
    output_obj = mesh_dir / "textured_simple.obj"
    output_lines: list[str] = []
    for line in obj_lines:
        tokens = shlex.split(line.strip())
        if tokens and tokens[0] == "mtllib":
            output_lines.append("mtllib textured_simple.mtl")
        else:
            output_lines.append(line)
    output_obj.write_text("\n".join(output_lines) + "\n", encoding="utf-8")

    if material_refs:
        source_mtl = (source_obj.parent / material_refs[0]).resolve()
        if not source_mtl.is_file():
            raise RuntimeError(f"OBJ material file does not exist: {source_mtl}")
        _rewrite_material(source_mtl, mesh_dir / "textured_simple.mtl", mesh_dir / "textures")
    return output_obj


def _prepare_output_dir(output_dir: Path) -> None:
    if output_dir.exists():
        raise FileExistsError(f"Output directory already exists; refusing to overwrite it: {output_dir}")
    output_dir.mkdir(parents=True)
    (output_dir / "rgb").mkdir()
    (output_dir / "depth").mkdir()
    (output_dir / "masks").mkdir()
    (output_dir / "annotated_poses").mkdir()


def _save_frame(
    output_dir: Path,
    frame_index: int,
    rgb: np.ndarray,
    depth_mm: np.ndarray,
    mask: np.ndarray,
    object_in_camera: np.ndarray,
) -> None:
    stem = f"{frame_index:06d}.png"
    iio.imwrite(output_dir / "rgb" / stem, rgb)
    iio.imwrite(output_dir / "depth" / stem, depth_mm)
    if args_cli.mask_mode == "all" or frame_index == 0:
        iio.imwrite(output_dir / "masks" / stem, mask)
    np.savetxt(
        output_dir / "annotated_poses" / f"{frame_index:06d}.txt",
        object_in_camera,
        fmt="%.18e",
    )


def _write_metadata(
    output_dir: Path,
    *,
    assembly_name: str,
    camera_matrix: np.ndarray,
    source_obj: Path,
) -> None:
    metadata = {
        "format": "foundationpose_ycbineoat",
        "task": args_cli.task,
        "assembly": assembly_name,
        "target_part": args_cli.target_part,
        "camera": _CAMERA_NAME,
        "image_width": args_cli.width,
        "image_height": args_cli.height,
        "num_frames": args_cli.num_frames,
        "warmup_steps": args_cli.warmup_steps,
        "frame_interval": args_cli.frame_interval,
        "mask_mode": args_cli.mask_mode,
        "depth_encoding": "uint16_png",
        "depth_unit": "millimeter",
        "depth_scale_to_meters": 0.001,
        "camera_matrix": camera_matrix.tolist(),
        "mesh": "mesh/textured_simple.obj",
        "source_mesh": str(source_obj),
        "ground_truth_pose_directory": "annotated_poses",
        "ground_truth_pose_type": "object_in_camera",
        "ground_truth_pose_convention": "opencv_x_right_y_down_z_forward",
        "ground_truth_pose_unit": "meter",
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")


def _camera_buffers(camera) -> tuple[np.ndarray, np.ndarray, np.ndarray, Any]:
    missing = {
        _RGB_DATA_TYPE,
        _DEPTH_DATA_TYPE,
        _SEMANTIC_DATA_TYPE,
    }.difference(camera.data.output)
    if missing:
        raise RuntimeError(f"Camera '{_CAMERA_NAME}' is missing required outputs: {sorted(missing)}")
    rgb = _rgb_uint8(camera.data.output[_RGB_DATA_TYPE])
    depth_mm = _depth_mm_uint16(camera.data.output[_DEPTH_DATA_TYPE])
    semantic = _semantic_ids(camera.data.output[_SEMANTIC_DATA_TYPE])
    semantic_info = camera.data.info.get(_SEMANTIC_DATA_TYPE)
    return rgb, depth_mm, semantic, semantic_info


def main() -> None:
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=1,
        use_fabric=not args_cli.disable_fabric,
    )
    try:
        camera_cfg = getattr(env_cfg.scene, _CAMERA_NAME)
    except AttributeError as exc:
        raise RuntimeError(f"Task '{args_cli.task}' does not define scene camera '{_CAMERA_NAME}'.") from exc
    camera_cfg.width = args_cli.width
    camera_cfg.height = args_cli.height

    assembly_name = env_cfg.assembly_name
    assembly = make_assembly(assembly_name)
    if args_cli.target_part not in assembly.part_names:
        raise ValueError(
            f"Target part '{args_cli.target_part}' is not registered for assembly '{assembly_name}'. "
            f"Available parts: {', '.join(assembly.part_names)}"
        )
    part = assembly.part(args_cli.target_part)
    source_obj = _mesh_from_part_urdf(part.urdf_path(assembly.asset_root))
    output_dir = (args_cli.output_dir or _default_output_dir(assembly_name, args_cli.target_part)).resolve()
    if output_dir.exists():
        raise FileExistsError(f"Output directory already exists; refusing to overwrite it: {output_dir}")

    env = gym.make(args_cli.task, cfg=env_cfg)
    try:
        env.reset()
        actions = torch.zeros(env.action_space.shape, dtype=torch.float32, device=env.unwrapped.device)
        for _ in range(args_cli.warmup_steps):
            env.step(actions)

        camera = env.unwrapped.scene[_CAMERA_NAME]
        target_asset = env.unwrapped.scene[args_cli.target_part]
        target_ids: tuple[int, ...] | None = None
        camera_matrix: np.ndarray | None = None
        output_prepared = False
        for frame_index in range(args_cli.num_frames):
            for _ in range(args_cli.frame_interval):
                env.step(actions)

            rgb, depth_mm, semantic, semantic_info = _camera_buffers(camera)
            if target_ids is None:
                target_ids = _target_semantic_ids(semantic_info, args_cli.target_part)
            mask = _target_mask(semantic, target_ids)
            _validate_frame(rgb, depth_mm, mask, require_mask=frame_index == 0)
            object_in_camera = _object_in_camera_pose(camera, target_asset)

            if camera_matrix is None:
                camera_matrix = _single_camera_array(camera.data.intrinsic_matrices, "intrinsic_matrices")
                if camera_matrix.shape != (3, 3) or not np.all(np.isfinite(camera_matrix)):
                    raise RuntimeError(
                        f"Invalid camera intrinsic matrix: shape={camera_matrix.shape}, K={camera_matrix}"
                    )
            if not output_prepared:
                _prepare_output_dir(output_dir)
                _package_mesh(source_obj, output_dir / "mesh")
                np.savetxt(output_dir / "cam_K.txt", camera_matrix, fmt="%.18e")
                output_prepared = True
            _save_frame(output_dir, frame_index, rgb, depth_mm, mask, object_in_camera)
            print(
                f"[INFO]: captured frame {frame_index + 1}/{args_cli.num_frames} "
                f"valid_depth={np.count_nonzero(depth_mm)} target_pixels={np.count_nonzero(mask)} "
                f"object_translation_c={object_in_camera[:3, 3].tolist()}"
            )

        assert camera_matrix is not None
        _write_metadata(
            output_dir,
            assembly_name=assembly_name,
            camera_matrix=camera_matrix,
            source_obj=source_obj,
        )
        print(f"[INFO]: FoundationPose demo data saved to: {output_dir}")
        print(f"[INFO]: mesh file: {output_dir / 'mesh' / 'textured_simple.obj'}")
    finally:
        env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
