# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Controllers used by assembly_benchmark tasks."""

from .r1_pro import BimanualDifferentialIKController, BimanualJointPositionController

__all__ = ["BimanualDifferentialIKController", "BimanualJointPositionController"]
