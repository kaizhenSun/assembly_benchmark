# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Dependency-free geometric constants for the R1 Pro Beam02 fixed grasp."""

BEAM02_GRIPPER_OFFSET_IN_PLUG = (0.0, 0.0285, 0.0300)
"""Offset applied to the gripper origin in the plug frame, in metres."""

BEAM02_GRIPPER_TO_PLUG_POS = (0.03823563633, 0.04445008697, -0.13810032123)
"""Position of ``beam_plug_0`` in ``left_gripper_link``."""

BEAM02_GRIPPER_TO_PLUG_QUAT = (0.70658029442, -0.02727860680, -0.02727860111, 0.70658052837)
"""Orientation of ``beam_plug_0`` in ``left_gripper_link`` as a wxyz quaternion."""

__all__ = [
    "BEAM02_GRIPPER_OFFSET_IN_PLUG",
    "BEAM02_GRIPPER_TO_PLUG_POS",
    "BEAM02_GRIPPER_TO_PLUG_QUAT",
]
