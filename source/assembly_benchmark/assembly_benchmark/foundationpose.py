# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""FoundationPose interchange helpers used by Assembly Benchmark tools."""

from __future__ import annotations

import ast
import json
import math
import shlex
import shutil
import subprocess
import xml.etree.ElementTree as ET
from collections.abc import Callable
from pathlib import Path
from typing import Any

import imageio.v3 as iio
import numpy as np

FOUNDATIONPOSE_RESULT_PREFIX = "FOUNDATIONPOSE_RESULT_JSON="
UINT16_MAX = np.iinfo(np.uint16).max
BOUNDING_BOX_EDGES = (
    (0, 1),
    (0, 2),
    (0, 4),
    (1, 3),
    (1, 5),
    (2, 3),
    (2, 6),
    (3, 7),
    (4, 5),
    (4, 6),
    (5, 7),
    (6, 7),
)
CommandRunner = Callable[[list[str], float | None], subprocess.CompletedProcess[str]]


def local_bounding_box_corners(bounds: Any) -> np.ndarray:
    """Return eight local AABB corners in deterministic binary XYZ order."""
    bounds = np.asarray(bounds, dtype=np.float64)
    if bounds.shape != (2, 3) or not np.all(np.isfinite(bounds)):
        raise ValueError(f"Mesh bounds must be finite 2x3, got {bounds}.")
    if np.any(bounds[1] < bounds[0]):
        raise ValueError("Mesh maximum bounds must not be smaller than minimum bounds.")
    return np.asarray(
        [
            (bounds[x, 0], bounds[y, 1], bounds[z, 2])
            for x, y, z in (
                (0, 0, 0),
                (0, 0, 1),
                (0, 1, 0),
                (0, 1, 1),
                (1, 0, 0),
                (1, 0, 1),
                (1, 1, 0),
                (1, 1, 1),
            )
        ],
        dtype=np.float64,
    )


def parse_foundationpose_visualization(
    payload: dict[str, Any], expected_objects: tuple[str, ...]
) -> dict[str, dict[str, Any]]:
    """Validate per-object local geometry returned for host-side visualization."""
    visualization = payload.get("visualization")
    if not isinstance(visualization, dict):
        raise RuntimeError("FoundationPose result is missing the 'visualization' geometry mapping.")
    missing = set(expected_objects).difference(visualization)
    extra = set(visualization).difference(expected_objects)
    if missing or extra:
        raise RuntimeError(
            f"FoundationPose visualization object mismatch: missing={sorted(missing)}, extra={sorted(extra)}"
        )
    result: dict[str, dict[str, Any]] = {}
    for name in expected_objects:
        item = visualization[name]
        if not isinstance(item, dict):
            raise RuntimeError(f"Visualization geometry for '{name}' must be a mapping.")
        corners = np.asarray(item.get("bbox_corners"), dtype=np.float64)
        axis_length = item.get("axis_length_m")
        if corners.shape != (8, 3) or not np.all(np.isfinite(corners)):
            raise RuntimeError(f"Visualization bbox_corners for '{name}' must be finite 8x3.")
        if not isinstance(axis_length, (int, float)) or not np.isfinite(axis_length) or axis_length <= 0.0:
            raise RuntimeError(f"Visualization axis_length_m for '{name}' must be finite and positive.")
        result[name] = {"bbox_corners": corners, "axis_length_m": float(axis_length)}
    return result


def project_camera_points(points: Any, camera_matrix: Any) -> np.ndarray:
    """Project finite positive-depth camera-frame points to image pixel coordinates."""
    points = np.asarray(points, dtype=np.float64)
    camera_matrix = np.asarray(camera_matrix, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or not np.all(np.isfinite(points)):
        raise ValueError(f"Camera points must be finite Nx3, got shape {points.shape}.")
    if camera_matrix.shape != (3, 3) or not np.all(np.isfinite(camera_matrix)):
        raise ValueError(f"Camera matrix must be finite 3x3, got {camera_matrix}.")
    if np.any(points[:, 2] <= 0.0):
        raise ValueError("Cannot project points with non-positive camera depth.")
    homogeneous = points @ camera_matrix.T
    pixels = homogeneous[:, :2] / homogeneous[:, 2:3]
    if not np.all(np.isfinite(pixels)):
        raise ValueError("Projected image points are not finite.")
    return pixels


def _transform_points(transform: Any, points: Any) -> np.ndarray:
    transform = validate_pose_matrix(transform)
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or not np.all(np.isfinite(points)):
        raise ValueError(f"Local points must be finite Nx3, got shape {points.shape}.")
    return points @ transform[:3, :3].T + transform[:3, 3]


def _draw_line(
    image: np.ndarray, start: np.ndarray, end: np.ndarray, color: tuple[int, int, int], width: int = 2
) -> None:
    delta = end - start
    count = max(1, int(np.ceil(np.max(np.abs(delta))))) + 1
    samples = np.rint(np.linspace(start, end, count)).astype(np.int64)
    height, image_width = image.shape[:2]
    radius = max(0, width // 2)
    for x, y in samples:
        x0, x1 = max(0, x - radius), min(image_width, x + radius + 1)
        y0, y1 = max(0, y - radius), min(height, y + radius + 1)
        if x0 < x1 and y0 < y1:
            image[y0:y1, x0:x1] = color


_FONT = {
    "A": ("01110", "10001", "10001", "11111", "10001", "10001", "10001"),
    "B": ("11110", "10001", "10001", "11110", "10001", "10001", "11110"),
    "C": ("01111", "10000", "10000", "10000", "10000", "10000", "01111"),
    "D": ("11110", "10001", "10001", "10001", "10001", "10001", "11110"),
    "E": ("11111", "10000", "10000", "11110", "10000", "10000", "11111"),
    "F": ("11111", "10000", "10000", "11110", "10000", "10000", "10000"),
    "G": ("01111", "10000", "10000", "10111", "10001", "10001", "01111"),
    "H": ("10001", "10001", "10001", "11111", "10001", "10001", "10001"),
    "I": ("11111", "00100", "00100", "00100", "00100", "00100", "11111"),
    "J": ("00111", "00010", "00010", "00010", "10010", "10010", "01100"),
    "K": ("10001", "10010", "10100", "11000", "10100", "10010", "10001"),
    "L": ("10000", "10000", "10000", "10000", "10000", "10000", "11111"),
    "M": ("10001", "11011", "10101", "10101", "10001", "10001", "10001"),
    "N": ("10001", "11001", "10101", "10011", "10001", "10001", "10001"),
    "O": ("01110", "10001", "10001", "10001", "10001", "10001", "01110"),
    "P": ("11110", "10001", "10001", "11110", "10000", "10000", "10000"),
    "Q": ("01110", "10001", "10001", "10001", "10101", "10010", "01101"),
    "R": ("11110", "10001", "10001", "11110", "10100", "10010", "10001"),
    "S": ("01111", "10000", "10000", "01110", "00001", "00001", "11110"),
    "T": ("11111", "00100", "00100", "00100", "00100", "00100", "00100"),
    "U": ("10001", "10001", "10001", "10001", "10001", "10001", "01110"),
    "V": ("10001", "10001", "10001", "10001", "10001", "01010", "00100"),
    "W": ("10001", "10001", "10001", "10101", "10101", "10101", "01010"),
    "X": ("10001", "10001", "01010", "00100", "01010", "10001", "10001"),
    "Y": ("10001", "10001", "01010", "00100", "00100", "00100", "00100"),
    "Z": ("11111", "00001", "00010", "00100", "01000", "10000", "11111"),
    "_": ("00000", "00000", "00000", "00000", "00000", "00000", "11111"),
    "(": ("00110", "01000", "10000", "10000", "10000", "01000", "00110"),
    ")": ("01100", "00010", "00001", "00001", "00001", "00010", "01100"),
    " ": ("00000",) * 7,
}


def _draw_label(image: np.ndarray, origin: np.ndarray, label: str, color: tuple[int, int, int]) -> None:
    x_start, y_start = np.rint(origin).astype(int)
    for index, character in enumerate(label.upper()):
        glyph = _FONT.get(character, _FONT[" "])
        x_offset = x_start + index * 6
        for row, bits in enumerate(glyph):
            for column, bit in enumerate(bits):
                if bit == "1":
                    y, x = y_start + row, x_offset + column
                    if 0 <= y < image.shape[0] and 0 <= x < image.shape[1]:
                        image[y, x] = color


def render_foundationpose_overlay(
    rgb: Any,
    masks: dict[str, Any],
    camera_matrix: Any,
    poses: dict[str, Any],
    visualization: dict[str, dict[str, Any]],
    labels: dict[str, str] | None = None,
) -> np.ndarray:
    """Render semantic masks, local AABBs, coordinate axes, and labels over one RGB frame."""
    image = np.asarray(rgb)
    if image.ndim != 3 or image.shape[2] < 3:
        raise ValueError(f"RGB overlay input must be HxWx3/4, got {image.shape}.")
    image = np.clip(image[..., :3], 0, 255).astype(np.uint8, copy=True)
    names = tuple(poses)
    if set(masks) != set(names) or set(visualization) != set(names):
        raise ValueError("Overlay masks, poses, and visualization geometry must contain the same objects.")
    colors = ((0, 220, 220), (220, 0, 220))
    for object_index, name in enumerate(names):
        mask = np.asarray(masks[name])
        if mask.shape != image.shape[:2]:
            raise ValueError(f"Overlay mask for '{name}' has shape {mask.shape}; expected {image.shape[:2]}.")
        selected = mask != 0
        color = np.asarray(colors[object_index % len(colors)], dtype=np.float64)
        image[selected] = np.rint(image[selected] * 0.6 + color * 0.4).astype(np.uint8)

        pose = validate_pose_matrix(poses[name])
        item = visualization[name]
        corners = np.asarray(item["bbox_corners"], dtype=np.float64)
        if corners.shape != (8, 3) or not np.all(np.isfinite(corners)):
            raise ValueError(f"Overlay bbox_corners for '{name}' must be finite 8x3.")
        axis_length = float(item["axis_length_m"])
        if not np.isfinite(axis_length) or axis_length <= 0.0:
            raise ValueError(f"Overlay axis_length_m for '{name}' must be finite and positive.")
        corner_pixels = project_camera_points(_transform_points(pose, corners), camera_matrix)
        for start, end in BOUNDING_BOX_EDGES:
            _draw_line(image, corner_pixels[start], corner_pixels[end], tuple(color.astype(int)), 2)
        axes = np.vstack((np.zeros(3), np.eye(3) * axis_length))
        axis_pixels = project_camera_points(_transform_points(pose, axes), camera_matrix)
        for endpoint, axis_color in zip(axis_pixels[1:], ((255, 0, 0), (0, 255, 0), (0, 0, 255)), strict=True):
            _draw_line(image, axis_pixels[0], endpoint, axis_color, 3)
        label = name if labels is None else labels.get(name, name)
        label_origin = np.array((np.min(corner_pixels[:, 0]), np.min(corner_pixels[:, 1]) - 9.0))
        _draw_label(image, label_origin, label, tuple(color.astype(int)))
    return image


def run_host_command(command: list[str], timeout: float | None = None) -> subprocess.CompletedProcess[str]:
    """Run one host-side container command with normalized missing-command and timeout errors."""
    try:
        return subprocess.run(command, capture_output=True, text=True, timeout=timeout, check=False)
    except FileNotFoundError as exc:
        raise RuntimeError(f"Required host command was not found: {command[0]}") from exc
    except subprocess.TimeoutExpired as exc:
        timeout_text = "the configured timeout" if timeout is None else f"{timeout:.1f} s"
        raise RuntimeError(f"Command timed out after {timeout_text}: {' '.join(command)}") from exc


def ensure_foundationpose_container(
    container: str,
    run_dir: Path,
    foundationpose_repo: Path,
    runner: CommandRunner = run_host_command,
) -> None:
    """Ensure the named container is running and can read a same-path capture directory."""
    state_result = runner(["docker", "inspect", "--format", "{{json .State}}", container], None)
    if state_result.returncode != 0:
        detail = state_result.stderr.strip() or state_result.stdout.strip()
        raise RuntimeError(
            f"FoundationPose container '{container}' does not exist or cannot be inspected: {detail}. "
            f"Create it with {foundationpose_repo / 'docker' / 'run_container.sh'}."
        )
    try:
        state = json.loads(state_result.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Docker returned invalid state JSON for container '{container}'.") from exc
    if not state.get("Running", False):
        start_result = runner(["docker", "start", container], None)
        if start_result.returncode != 0:
            detail = start_result.stderr.strip() or start_result.stdout.strip()
            raise RuntimeError(f"Failed to start FoundationPose container '{container}': {detail}")

    mounts_result = runner(["docker", "inspect", "--format", "{{json .Mounts}}", container], None)
    if mounts_result.returncode != 0:
        raise RuntimeError(f"Failed to inspect mounts for FoundationPose container '{container}'.")
    try:
        mounts = json.loads(mounts_result.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Docker returned invalid mount JSON for container '{container}'.") from exc
    required_paths = ((run_dir.resolve(), "Capture directory"), (foundationpose_repo.resolve(), "FoundationPose repo"))
    for required_path, label in required_paths:
        visible = False
        for mount in mounts:
            destination_text = mount.get("Destination")
            source_text = mount.get("Source")
            if not source_text or not destination_text:
                continue
            try:
                relative_path = required_path.relative_to(Path(source_text).resolve())
            except ValueError:
                continue
            container_path = (Path(destination_text) / relative_path).resolve()
            if container_path == required_path:
                visible = True
                break
        if not visible:
            raise RuntimeError(
                f"{label} {required_path} is not visible in container '{container}' at the same absolute path."
            )


def single_camera_array(value: Any, name: str) -> np.ndarray:
    """Convert a one-environment camera tensor or array to a NumPy image."""
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    value = np.asarray(value)
    if value.ndim == 0 or value.shape[0] != 1:
        raise RuntimeError(f"Expected '{name}' to have one leading environment dimension, got {value.shape}.")
    return value[0]


def rgb_uint8(value: Any) -> np.ndarray:
    """Convert an Isaac camera RGB/RGBA buffer to uint8 RGB."""
    rgb = single_camera_array(value, "rgb")
    if rgb.ndim != 3 or rgb.shape[-1] < 3:
        raise RuntimeError(f"Expected RGB/RGBA camera output, got shape {rgb.shape}.")
    rgb = rgb[..., :3]
    if np.issubdtype(rgb.dtype, np.floating):
        finite = rgb[np.isfinite(rgb)]
        finite_max = float(finite.max()) if finite.size else 1.0
        rgb = np.nan_to_num(rgb, nan=0.0, posinf=255.0, neginf=0.0)
        if finite_max <= 1.0:
            rgb = rgb * 255.0
    return np.clip(rgb, 0, 255).astype(np.uint8)


def depth_mm_uint16(value: Any) -> np.ndarray:
    """Encode meter-valued optical-axis depth as uint16 millimeters."""
    depth_m = single_camera_array(value, "distance_to_image_plane")
    if depth_m.ndim == 3 and depth_m.shape[-1] == 1:
        depth_m = depth_m[..., 0]
    if depth_m.ndim != 2:
        raise RuntimeError(f"Expected a single-channel depth image, got shape {depth_m.shape}.")
    valid = np.isfinite(depth_m) & (depth_m >= 0.001) & (depth_m <= UINT16_MAX / 1000.0)
    depth_mm = np.zeros(depth_m.shape, dtype=np.uint16)
    depth_mm[valid] = np.rint(depth_m[valid] * 1000.0).astype(np.uint16)
    if not np.any(depth_mm):
        raise RuntimeError("The depth image contains no valid uint16 millimeter samples.")
    return depth_mm


def semantic_ids(value: Any) -> np.ndarray:
    """Extract the uncolorized semantic ID plane from an Isaac camera buffer."""
    semantic = single_camera_array(value, "semantic_segmentation")
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
        return {label for nested in value.values() for label in _flatten_labels(nested)}
    if isinstance(value, (list, tuple, set)):
        return {label for nested in value for label in _flatten_labels(nested)}
    return {str(value)}


def target_semantic_ids(info: Any, target_label: str) -> tuple[int, ...]:
    """Resolve Replicator semantic metadata to IDs for one exact class label."""
    info = _decode_maybe_serialized(info)
    if not isinstance(info, dict):
        raise RuntimeError(f"Semantic camera metadata is not a mapping: {type(info).__name__}.")
    id_to_labels = _decode_maybe_serialized(info.get("idToLabels", info))
    if not isinstance(id_to_labels, dict):
        raise RuntimeError("Semantic camera metadata does not contain an 'idToLabels' mapping.")

    ids: list[int] = []
    for raw_id, raw_labels in id_to_labels.items():
        if target_label in _flatten_labels(raw_labels):
            try:
                ids.append(int(raw_id))
            except (TypeError, ValueError) as exc:
                raise RuntimeError(f"Semantic ID {raw_id!r} for '{target_label}' is not an integer.") from exc
    if not ids:
        available = sorted({label for labels in id_to_labels.values() for label in _flatten_labels(labels)})
        raise RuntimeError(
            f"Semantic label '{target_label}' was not found. Available labels include: {', '.join(available[:20])}"
        )
    return tuple(sorted(set(ids)))


def target_mask(semantic: np.ndarray, ids: tuple[int, ...], label: str) -> np.ndarray:
    """Create a visible-object 0/255 mask and reject empty targets."""
    mask = np.isin(semantic, ids).astype(np.uint8) * 255
    if not np.any(mask):
        raise RuntimeError(f"Semantic mask for '{label}' is empty; the object is not visible.")
    return mask


def mesh_from_part_urdf(urdf_path: Path) -> Path:
    """Find the first visual OBJ referenced by an assembly part URDF."""
    try:
        root = ET.parse(urdf_path).getroot()
    except (ET.ParseError, OSError) as exc:
        raise RuntimeError(f"Could not read target part URDF: {urdf_path}") from exc
    for node in root.findall(".//visual/geometry/mesh"):
        filename = node.get("filename")
        if not filename or filename.startswith("package://"):
            continue
        candidate = (urdf_path.parent / filename).resolve()
        if candidate.suffix.lower() == ".obj":
            if not candidate.is_file():
                raise RuntimeError(f"Target OBJ mesh does not exist: {candidate}")
            return candidate
    raise RuntimeError(f"No visual OBJ mesh was found in {urdf_path}.")


def _rewrite_material(material_path: Path, output_path: Path, texture_dir: Path) -> None:
    rewritten: list[str] = []
    copied_names: dict[str, Path] = {}
    for line in material_path.read_text(encoding="utf-8").splitlines():
        tokens = shlex.split(line.strip())
        if not tokens or not tokens[0].startswith("map_") or len(tokens) < 2:
            rewritten.append(line)
            continue
        source_texture = (material_path.parent / tokens[-1]).resolve()
        if not source_texture.is_file():
            raise RuntimeError(f"Material texture does not exist: {source_texture}")
        previous = copied_names.get(source_texture.name)
        if previous is not None and previous != source_texture:
            raise RuntimeError(f"Two material textures have the same filename: {source_texture.name}")
        copied_names[source_texture.name] = source_texture
        texture_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_texture, texture_dir / source_texture.name)
        tokens[-1] = f"textures/{source_texture.name}"
        rewritten.append(" ".join(tokens))
    output_path.write_text("\n".join(rewritten) + "\n", encoding="utf-8")


def package_obj_mesh(source_obj: Path, mesh_dir: Path) -> Path:
    """Copy an OBJ, its MTL, and textures into a self-contained directory."""
    obj_lines = source_obj.read_text(encoding="utf-8").splitlines()
    material_refs = [
        token
        for line in obj_lines
        for tokens in [shlex.split(line.strip())]
        if tokens and tokens[0] == "mtllib"
        for token in tokens[1:]
    ]
    if len(material_refs) > 1:
        raise RuntimeError(f"Expected at most one MTL reference in {source_obj}, found {material_refs}.")

    mesh_dir.mkdir(parents=True, exist_ok=False)
    output_obj = mesh_dir / "textured_simple.obj"
    output_lines = [
        "mtllib textured_simple.mtl" if (tokens := shlex.split(line.strip())) and tokens[0] == "mtllib" else line
        for line in obj_lines
    ]
    output_obj.write_text("\n".join(output_lines) + "\n", encoding="utf-8")
    if material_refs:
        source_mtl = (source_obj.parent / material_refs[0]).resolve()
        if not source_mtl.is_file():
            raise RuntimeError(f"OBJ material file does not exist: {source_mtl}")
        _rewrite_material(source_mtl, mesh_dir / "textured_simple.mtl", mesh_dir / "textures")
    return output_obj


def write_rgbd_frame(root: Path, rgb: np.ndarray, depth_mm: np.ndarray) -> tuple[Path, Path, Path]:
    """Write the common one-frame RGB-D input and return RGB, depth, and K placeholders."""
    rgb_path = root / "rgb.png"
    depth_path = root / "depth.png"
    camera_matrix_path = root / "cam_K.txt"
    iio.imwrite(rgb_path, rgb)
    iio.imwrite(depth_path, depth_mm)
    return rgb_path, depth_path, camera_matrix_path


def pose_matrix_from_pos_quat(pos: np.ndarray, quat_wxyz: np.ndarray) -> np.ndarray:
    """Build a homogeneous transform from position and a wxyz quaternion."""
    pos = np.asarray(pos, dtype=np.float64).reshape(3)
    quat = np.asarray(quat_wxyz, dtype=np.float64).reshape(4)
    norm = np.linalg.norm(quat)
    if not np.isfinite(norm) or norm < 1.0e-8:
        raise ValueError("Quaternion must be finite and non-zero.")
    w, x, y, z = quat / norm
    rotation = np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rotation
    transform[:3, 3] = pos
    return transform


def pos_quat_from_pose_matrix(transform: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Convert a homogeneous transform to position and a normalized wxyz quaternion."""
    transform = validate_pose_matrix(transform, require_positive_z=False)
    rotation = transform[:3, :3]
    trace = float(np.trace(rotation))
    if trace > 0.0:
        scale = math.sqrt(trace + 1.0) * 2.0
        quat = np.array(
            [
                0.25 * scale,
                (rotation[2, 1] - rotation[1, 2]) / scale,
                (rotation[0, 2] - rotation[2, 0]) / scale,
                (rotation[1, 0] - rotation[0, 1]) / scale,
            ]
        )
    else:
        index = int(np.argmax(np.diag(rotation)))
        next_index = (index + 1) % 3
        last_index = (index + 2) % 3
        scale = (
            math.sqrt(
                1.0 + rotation[index, index] - rotation[next_index, next_index] - rotation[last_index, last_index]
            )
            * 2.0
        )
        xyz = np.empty(3, dtype=np.float64)
        xyz[index] = 0.25 * scale
        xyz[next_index] = (rotation[next_index, index] + rotation[index, next_index]) / scale
        xyz[last_index] = (rotation[last_index, index] + rotation[index, last_index]) / scale
        w = (rotation[last_index, next_index] - rotation[next_index, last_index]) / scale
        quat = np.concatenate(([w], xyz))
    quat /= np.linalg.norm(quat)
    if quat[0] < 0.0:
        quat = -quat
    return transform[:3, 3].copy(), quat


def validate_pose_matrix(transform: Any, *, require_positive_z: bool = True) -> np.ndarray:
    """Validate and return one rigid 4x4 object-in-camera transform."""
    transform = np.asarray(transform, dtype=np.float64)
    if transform.shape != (4, 4) or not np.all(np.isfinite(transform)):
        raise ValueError(f"Pose must be a finite 4x4 matrix, got shape {transform.shape}.")
    if not np.allclose(transform[3], [0.0, 0.0, 0.0, 1.0], atol=1.0e-5):
        raise ValueError(f"Pose has an invalid homogeneous row: {transform[3].tolist()}")
    rotation = transform[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=2.0e-3):
        raise ValueError("Pose rotation is not orthonormal.")
    determinant = float(np.linalg.det(rotation))
    if not math.isclose(determinant, 1.0, abs_tol=2.0e-3):
        raise ValueError(f"Pose rotation determinant must be +1, got {determinant}.")
    if require_positive_z and not 0.001 <= float(transform[2, 3]) <= 20.0:
        raise ValueError(f"Object camera Z must be in [0.001, 20] m, got {transform[2, 3]}.")
    return transform


def foundationpose_result_payload(stdout: str) -> dict[str, Any]:
    """Extract the adapter's unique versioned JSON result line."""
    lines = [line for line in stdout.splitlines() if line.startswith(FOUNDATIONPOSE_RESULT_PREFIX)]
    if len(lines) != 1:
        raise RuntimeError(f"Expected one {FOUNDATIONPOSE_RESULT_PREFIX!r} line, found {len(lines)}.")
    try:
        payload = json.loads(lines[0][len(FOUNDATIONPOSE_RESULT_PREFIX) :])
    except json.JSONDecodeError as exc:
        raise RuntimeError("FoundationPose adapter returned invalid JSON.") from exc
    if (
        not isinstance(payload, dict)
        or payload.get("schema_version") != 1
        or not isinstance(payload.get("poses"), dict)
    ):
        raise RuntimeError("FoundationPose result has an unsupported schema.")
    return payload


def parse_foundationpose_result(stdout: str, expected_objects: tuple[str, ...]) -> dict[str, np.ndarray]:
    """Parse the adapter's unique JSON result line and validate all requested poses."""
    payload = foundationpose_result_payload(stdout)
    missing = set(expected_objects).difference(payload["poses"])
    extra = set(payload["poses"]).difference(expected_objects)
    if missing or extra:
        raise RuntimeError(f"FoundationPose result object mismatch: missing={sorted(missing)}, extra={sorted(extra)}")
    return {name: validate_pose_matrix(payload["poses"][name]) for name in expected_objects}


def run_foundationpose_adapter(
    command: list[str],
    run_dir: Path,
    timeout_s: float,
    expected_objects: tuple[str, ...],
    runner: CommandRunner = run_host_command,
) -> tuple[dict[str, np.ndarray], dict[str, Any], str, str]:
    """Run the container adapter once, persist logs, and validate its sentinel response."""
    result = runner(command, timeout_s)
    run_dir.joinpath("adapter_stdout.log").write_text(result.stdout, encoding="utf-8")
    run_dir.joinpath("adapter_stderr.log").write_text(result.stderr, encoding="utf-8")
    if result.returncode != 0:
        raise RuntimeError(
            f"FoundationPose adapter failed with exit code {result.returncode}. See {run_dir / 'adapter_stderr.log'}."
        )
    payload = foundationpose_result_payload(result.stdout)
    if payload.get("model_initializations") != 1:
        raise RuntimeError("FoundationPose adapter must initialize the shared models exactly once.")
    if payload.get("register_calls") != len(expected_objects):
        raise RuntimeError(
            f"FoundationPose adapter reported {payload.get('register_calls')!r} register calls; "
            f"expected {len(expected_objects)}."
        )
    poses = parse_foundationpose_result(result.stdout, expected_objects)
    return poses, payload, result.stdout, result.stderr


def pose_errors(estimated: np.ndarray, ground_truth: np.ndarray) -> tuple[float, float]:
    """Return translation error in meters and geodesic rotation error in radians."""
    estimated = validate_pose_matrix(estimated, require_positive_z=False)
    ground_truth = validate_pose_matrix(ground_truth, require_positive_z=False)
    translation = float(np.linalg.norm(estimated[:3, 3] - ground_truth[:3, 3]))
    relative_rotation = estimated[:3, :3].T @ ground_truth[:3, :3]
    cosine = np.clip((np.trace(relative_rotation) - 1.0) * 0.5, -1.0, 1.0)
    return translation, float(np.arccos(cosine))


def canonicalize_pose_by_symmetry(
    transform: np.ndarray,
    canonical_rotation: np.ndarray,
    symmetry_rotations: tuple[np.ndarray, ...],
) -> tuple[np.ndarray, int, float]:
    """Select the symmetry-equivalent pose closest to a canonical reference rotation.

    Symmetries are object-local rotations and are therefore post-multiplied onto the
    estimated rotation. The translation is never changed. The returned angle is the
    geodesic rotation distance to the canonical reference, in radians.
    """
    transform = validate_pose_matrix(transform, require_positive_z=False)
    canonical_rotation = np.asarray(canonical_rotation, dtype=np.float64)
    if canonical_rotation.shape != (3, 3):
        raise ValueError(f"Canonical rotation must be 3x3, got {canonical_rotation.shape}.")
    canonical_pose = np.eye(4, dtype=np.float64)
    canonical_pose[:3, :3] = canonical_rotation
    validate_pose_matrix(canonical_pose, require_positive_z=False)
    if not symmetry_rotations:
        raise ValueError("At least one symmetry rotation is required.")

    best_pose: np.ndarray | None = None
    best_index = -1
    best_angle = math.inf
    for index, symmetry_rotation in enumerate(symmetry_rotations):
        symmetry_rotation = np.asarray(symmetry_rotation, dtype=np.float64)
        if symmetry_rotation.shape != (3, 3):
            raise ValueError(f"Symmetry rotation {index} must be 3x3, got {symmetry_rotation.shape}.")
        symmetry_pose = np.eye(4, dtype=np.float64)
        symmetry_pose[:3, :3] = symmetry_rotation
        validate_pose_matrix(symmetry_pose, require_positive_z=False)

        candidate = transform.copy()
        candidate[:3, :3] = transform[:3, :3] @ symmetry_rotation
        relative_rotation = canonical_rotation.T @ candidate[:3, :3]
        cosine = np.clip((np.trace(relative_rotation) - 1.0) * 0.5, -1.0, 1.0)
        angle = float(np.arccos(cosine))
        if angle < best_angle:
            best_pose = candidate
            best_index = index
            best_angle = angle

    assert best_pose is not None
    return best_pose, best_index, best_angle


__all__ = [
    "BOUNDING_BOX_EDGES",
    "FOUNDATIONPOSE_RESULT_PREFIX",
    "canonicalize_pose_by_symmetry",
    "depth_mm_uint16",
    "ensure_foundationpose_container",
    "foundationpose_result_payload",
    "local_bounding_box_corners",
    "mesh_from_part_urdf",
    "package_obj_mesh",
    "parse_foundationpose_result",
    "parse_foundationpose_visualization",
    "pos_quat_from_pose_matrix",
    "pose_errors",
    "pose_matrix_from_pos_quat",
    "project_camera_points",
    "render_foundationpose_overlay",
    "rgb_uint8",
    "run_foundationpose_adapter",
    "run_host_command",
    "semantic_ids",
    "single_camera_array",
    "target_mask",
    "target_semantic_ids",
    "validate_pose_matrix",
    "write_rgbd_frame",
]
