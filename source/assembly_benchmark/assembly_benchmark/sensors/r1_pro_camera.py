# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""RGB-D and semantic camera configuration for the Galaxea R1 Pro head."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab.sensors import TiledCameraCfg


@dataclass(frozen=True, slots=True)
class R1ProHeadCameraSpec:
    """Published R1 Pro head-camera metadata plus the simulation depth contract."""

    parent_link_name: str = "zed_link"
    prim_name: str = "head_camera"
    width: int = 1920
    height: int = 1080
    frame_rate_hz: float = 30.0
    horizontal_fov_deg: float = 118.0
    published_vertical_fov_deg: float = 62.0
    stereo_baseline_m: float = 0.120
    depth_range_m: tuple[float, float] = (0.1, 20.0)
    focal_length: float = 24.0
    optical_center_pos: tuple[float, float, float] = (0.060, 0.0, 0.020)
    optical_center_rot: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)

    @property
    def update_period_s(self) -> float:
        """Return the sensor update period matching the published frame rate."""
        return 1.0 / self.frame_rate_hz

    @property
    def horizontal_aperture(self) -> float:
        """Return the pinhole aperture that reproduces the published horizontal FOV."""
        return 2.0 * self.focal_length * math.tan(math.radians(self.horizontal_fov_deg / 2.0))

    @property
    def simulated_vertical_fov_deg(self) -> float:
        """Return the square-pixel vertical FOV implied by the resolution and horizontal FOV."""
        half_horizontal_fov = math.radians(self.horizontal_fov_deg / 2.0)
        half_vertical_fov = math.atan(math.tan(half_horizontal_fov) * self.height / self.width)
        return math.degrees(2.0 * half_vertical_fov)

    @property
    def default_prim_path(self) -> str:
        """Return the default cloned scene path below the R1 Pro head link."""
        return f"{{ENV_REGEX_NS}}/Robot/{self.parent_link_name}/{self.prim_name}"


R1_PRO_HEAD_CAMERA_SPEC = R1ProHeadCameraSpec()


def make_r1_pro_head_camera_cfg(
    prim_path: str | None = None,
    spec: R1ProHeadCameraSpec = R1_PRO_HEAD_CAMERA_SPEC,
) -> TiledCameraCfg:
    """Create the vectorized, head-mounted R1 Pro RGB-D and semantic camera configuration."""
    try:
        import isaaclab.sim as sim_utils
        from isaaclab.sensors import TiledCameraCfg
    except ModuleNotFoundError as exc:
        raise RuntimeError("Creating the R1 Pro head camera requires an Isaac Sim runtime.") from exc

    return TiledCameraCfg(
        prim_path=prim_path or spec.default_prim_path,
        update_period=spec.update_period_s,
        height=spec.height,
        width=spec.width,
        data_types=["rgb", "distance_to_image_plane", "semantic_segmentation"],
        depth_clipping_behavior="zero",
        semantic_filter=["class"],
        colorize_semantic_segmentation=False,
        update_latest_camera_pose=True,
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=spec.focal_length,
            horizontal_aperture=spec.horizontal_aperture,
            clipping_range=spec.depth_range_m,
        ),
        offset=TiledCameraCfg.OffsetCfg(
            pos=spec.optical_center_pos,
            rot=spec.optical_center_rot,
            convention="ros",
        ),
    )


__all__ = (
    "R1_PRO_HEAD_CAMERA_SPEC",
    "R1ProHeadCameraSpec",
    "make_r1_pro_head_camera_cfg",
)
