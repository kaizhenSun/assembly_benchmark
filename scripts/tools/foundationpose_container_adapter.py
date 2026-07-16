# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Register multiple objects from one RGB-D frame inside a FoundationPose container."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import imageio.v2 as imageio
import numpy as np
import trimesh

# docker exec sets the FoundationPose checkout as the working directory. The adapter itself is
# staged beside the read-only capture so Python's default script path does not include that checkout.
sys.path.insert(0, str(Path.cwd()))

from estimater import (  # noqa: E402  # codespell:ignore estimater
    FoundationPose,
    PoseRefinePredictor,
    ScorePredictor,
    dr,
    set_logging_format,
    set_seed,
)

RESULT_PREFIX = "FOUNDATIONPOSE_RESULT_JSON="


def _visualization_geometry(mesh: trimesh.Trimesh, object_name: str) -> dict:
    """Build finite deterministic local geometry without serializing the full mesh."""
    bounds = np.asarray(mesh.bounds, dtype=np.float64)
    if bounds.shape != (2, 3) or not np.all(np.isfinite(bounds)) or np.any(bounds[1] < bounds[0]):
        raise ValueError(f"Mesh for '{object_name}' has invalid bounds: {bounds}.")
    corners = np.asarray(
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
    diagonal = float(np.linalg.norm(bounds[1] - bounds[0]))
    if not np.isfinite(diagonal) or diagonal <= 0.0:
        raise ValueError(f"Mesh for '{object_name}' has zero or invalid extent.")
    return {
        "bbox_corners": corners.tolist(),
        "axis_length_m": float(np.clip(diagonal * 0.35, 0.02, 0.15)),
    }


def _absolute_input_path(request_path: Path, value: object, field: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f"'{field}' must be a non-empty path string.")
    path = Path(value)
    if not path.is_absolute():
        path = request_path.parent / path
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(f"'{field}' does not exist: {path}")
    return path


def _load_request(request_path: Path) -> tuple[dict, np.ndarray, np.ndarray, np.ndarray]:
    request = json.loads(request_path.read_text(encoding="utf-8"))
    if request.get("schema_version") != 1:
        raise ValueError(f"Unsupported request schema_version: {request.get('schema_version')!r}")

    rgb_path = _absolute_input_path(request_path, request.get("rgb_path"), "rgb_path")
    depth_path = _absolute_input_path(request_path, request.get("depth_path"), "depth_path")
    camera_path = _absolute_input_path(request_path, request.get("camera_matrix_path"), "camera_matrix_path")
    rgb = np.asarray(imageio.imread(str(rgb_path)))
    if rgb.ndim != 3 or rgb.shape[2] < 3:
        raise ValueError(f"RGB image must be HxWx3/4, got {rgb.shape}.")
    rgb = np.ascontiguousarray(rgb[..., :3].astype(np.uint8, copy=False))

    depth_raw = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
    if depth_raw is None or depth_raw.ndim != 2:
        shape = None if depth_raw is None else depth_raw.shape
        raise ValueError(f"Depth image must be single-channel, got {shape}.")
    depth_scale = float(request.get("depth_scale_to_meters", 0.001))
    if not np.isfinite(depth_scale) or depth_scale <= 0.0:
        raise ValueError("'depth_scale_to_meters' must be finite and positive.")
    depth = depth_raw.astype(np.float32) * depth_scale
    depth[~np.isfinite(depth)] = 0.0
    if not np.any(depth >= 0.001):
        raise ValueError("Depth image contains no valid samples.")

    camera_matrix = np.asarray(np.loadtxt(camera_path), dtype=np.float64)
    if camera_matrix.shape != (3, 3) or not np.all(np.isfinite(camera_matrix)):
        raise ValueError(f"Camera matrix must be finite 3x3, got {camera_matrix}.")
    if rgb.shape[:2] != depth.shape:
        raise ValueError(f"RGB and depth dimensions differ: {rgb.shape[:2]} vs {depth.shape}.")
    return request, rgb, depth, camera_matrix


def _validate_pose(pose: object, object_name: str) -> np.ndarray:
    pose = np.asarray(pose, dtype=np.float64)
    if pose.shape != (4, 4) or not np.all(np.isfinite(pose)):
        raise RuntimeError(f"FoundationPose returned an invalid matrix for '{object_name}': {pose}.")
    if not np.allclose(pose[3], [0.0, 0.0, 0.0, 1.0], atol=1.0e-5):
        raise RuntimeError(f"FoundationPose returned an invalid homogeneous row for '{object_name}'.")
    return pose


def run(request_path: Path, est_refine_iter: int, debug: int) -> dict:
    request, rgb, depth, camera_matrix = _load_request(request_path)
    objects = request.get("objects")
    if not isinstance(objects, list) or not objects:
        raise ValueError("'objects' must be a non-empty list.")
    names = [item.get("name") for item in objects if isinstance(item, dict)]
    if len(names) != len(objects) or any(not isinstance(name, str) or not name for name in names):
        raise ValueError("Every request object must have a non-empty string 'name'.")
    if len(set(names)) != len(names):
        raise ValueError(f"Request object names must be unique: {names}.")

    initialization_started = time.perf_counter()
    scorer = ScorePredictor()
    refiner = PoseRefinePredictor()
    glctx = dr.RasterizeCudaContext()
    initialization_seconds = time.perf_counter() - initialization_started

    poses = {}
    visualization = {}
    registrations = []
    debug_root = Path("/tmp/foundationpose_assembly") / request_path.stem
    for item in objects:
        name = item["name"]
        mask_path = _absolute_input_path(request_path, item.get("mask_path"), f"objects[{name}].mask_path")
        mesh_path = _absolute_input_path(request_path, item.get("mesh_path"), f"objects[{name}].mesh_path")
        mask_raw = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
        if mask_raw is None or mask_raw.ndim != 2:
            raise ValueError(f"Mask for '{name}' must be single-channel.")
        mask = mask_raw != 0
        if mask.shape != depth.shape:
            raise ValueError(f"Mask for '{name}' has shape {mask.shape}; expected {depth.shape}.")
        if not np.any(mask):
            raise ValueError(f"Mask for '{name}' is empty.")
        if not np.any(mask & (depth >= 0.001)):
            raise ValueError(f"Mask for '{name}' contains no valid depth samples.")

        mesh = trimesh.load(str(mesh_path), force="mesh")
        if not isinstance(mesh, trimesh.Trimesh) or mesh.vertices.size == 0:
            raise ValueError(f"Mesh for '{name}' is empty or unsupported: {mesh_path}")
        visualization[name] = _visualization_geometry(mesh, name)
        estimator = FoundationPose(
            model_pts=mesh.vertices,
            model_normals=mesh.vertex_normals,
            mesh=mesh,
            scorer=scorer,
            refiner=refiner,
            debug_dir=str(debug_root / name),
            debug=debug,
            glctx=glctx,
        )
        started = time.perf_counter()
        pose = estimator.register(
            K=camera_matrix,
            rgb=rgb,
            depth=depth,
            ob_mask=mask,
            iteration=est_refine_iter,
        )
        poses[name] = _validate_pose(pose, name).tolist()
        registrations.append(
            {
                "name": name,
                "seconds": time.perf_counter() - started,
                "mask_pixels": int(np.count_nonzero(mask)),
            }
        )

    return {
        "schema_version": 1,
        "poses": poses,
        "visualization": visualization,
        "model_initializations": 1,
        "register_calls": len(registrations),
        "initialization_seconds": initialization_seconds,
        "registrations": registrations,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--est_refine_iter", type=int, default=5)
    parser.add_argument("--debug", type=int, default=0)
    args = parser.parse_args()
    if args.est_refine_iter <= 0:
        parser.error("--est_refine_iter must be positive")

    set_logging_format()
    set_seed(0)
    result = run(args.request.resolve(), args.est_refine_iter, args.debug)
    print(RESULT_PREFIX + json.dumps(result, separators=(",", ":")), flush=True)


if __name__ == "__main__":
    main()
