# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Public sensor API for Assembly Benchmark tasks."""

from .r1_pro_camera import R1_PRO_HEAD_CAMERA_SPEC, R1ProHeadCameraSpec, make_r1_pro_head_camera_cfg

__all__ = (
    "R1_PRO_HEAD_CAMERA_SPEC",
    "R1ProHeadCameraSpec",
    "make_r1_pro_head_camera_cfg",
)
