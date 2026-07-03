# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Assembly Benchmark Isaac Lab extension."""

_OPTIONAL_TASK_IMPORT_DEPS = {"isaaclab", "isaaclab_tasks", "pxr"}

# Register Gym environments.
try:
    from .tasks import *
except ModuleNotFoundError as exc:
    if exc.name not in _OPTIONAL_TASK_IMPORT_DEPS:
        raise

# Register UI extensions.
try:
    from .ui_extension_example import *
except ModuleNotFoundError as exc:
    if exc.name != "omni":
        raise
