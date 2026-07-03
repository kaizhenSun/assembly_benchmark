# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Assembly Benchmark Isaac Lab extension."""

# Register Gym environments.
try:
    from .tasks import *
except ModuleNotFoundError as exc:
    if exc.name != "isaaclab_tasks":
        raise

# Register UI extensions.
try:
    from .ui_extension_example import *
except ModuleNotFoundError as exc:
    if exc.name != "omni":
        raise
