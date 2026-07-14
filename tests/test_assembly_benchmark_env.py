# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import importlib
import importlib.util
from types import SimpleNamespace

import pytest
import torch

ISAAC_SIM_RUNTIME_AVAILABLE = importlib.util.find_spec("carb") is not None


@pytest.mark.skipif(not ISAAC_SIM_RUNTIME_AVAILABLE, reason="Isaac Sim runtime is not available")
def test_all_assembly_env_cfgs_use_fixed_arm_observation_size() -> None:
    assembly_module = importlib.import_module("assembly_benchmark.assembly")
    env_cfg_module = importlib.import_module(
        "assembly_benchmark.tasks.direct.assembly_benchmark.assembly_benchmark_env_cfg"
    )

    assert env_cfg_module.ARM_OBSERVATION_SIZE == 28
    for assembly_name in assembly_module.available_assemblies():
        class_prefix = "".join(part.capitalize() for part in assembly_name.split("_"))
        cfg_class_name = env_cfg_module.assembly_env_cfg_class_name(assembly_name)
        cfg = getattr(env_cfg_module, cfg_class_name)()

        assert cfg_class_name == f"{class_prefix}AssemblyBenchmarkEnvCfg"
        assert env_cfg_module.assembly_task_id(assembly_name) == f"Assembly-Benchmark-{class_prefix}-Direct-v0"
        assert cfg.observation_space == env_cfg_module.ARM_OBSERVATION_SIZE
        for part_name in assembly_module.make_assembly(assembly_name).part_names:
            assert hasattr(cfg.scene, part_name)


@pytest.mark.skipif(not ISAAC_SIM_RUNTIME_AVAILABLE, reason="Isaac Sim runtime is not available")
def test_policy_observation_contains_arm_joint_positions_then_velocities() -> None:
    env_module = importlib.import_module("assembly_benchmark.tasks.direct.assembly_benchmark.assembly_benchmark_env")
    joint_pos = torch.arange(40, dtype=torch.float32).reshape(2, 20)
    joint_vel = joint_pos + 100.0
    arm_joint_ids = [0, 2, 4, 6, 8, 10, 12, 1, 3, 5, 7, 9, 11, 13]
    env = SimpleNamespace(
        robot=SimpleNamespace(data=SimpleNamespace(joint_pos=joint_pos, joint_vel=joint_vel)),
        arm_joint_ids=arm_joint_ids,
    )

    observation = env_module.AssemblyBenchmarkEnv._get_observations(env)["policy"]
    expected = torch.cat((joint_pos[:, arm_joint_ids], joint_vel[:, arm_joint_ids]), dim=-1)

    assert observation.shape == (2, 28)
    assert torch.equal(observation, expected)
