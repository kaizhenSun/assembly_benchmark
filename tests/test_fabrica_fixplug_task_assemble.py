# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = REPO_ROOT / "source" / "assembly_benchmark" / "assembly_benchmark"
TASK_ROOT = PACKAGE_ROOT / "tasks" / "direct" / "fabrica"
TASK_ID = "Assembly-Benchmark-FabricaFixPlugTaskAssemble-Direct-v0"
ISAAC_LAB_AVAILABLE = importlib.util.find_spec("isaaclab") is not None
ISAAC_SIM_RUNTIME_AVAILABLE = importlib.util.find_spec("carb") is not None


def test_fabrica_task_registration_and_source_contract() -> None:
    registration = (TASK_ROOT / "__init__.py").read_text(encoding="utf-8")
    cfg_source = (TASK_ROOT / "fabrica_fixplug_task_assemble_cfg.py").read_text(encoding="utf-8")
    env_source = (TASK_ROOT / "fabrica_fixplug_task_assemble.py").read_text(encoding="utf-8")
    replay_source = (REPO_ROOT / "scripts" / "tools" / "run_fabrica_fixplug_openloop.py").read_text(encoding="utf-8")

    assert TASK_ID in registration
    assert "class FabricaFixPlugTaskAssemble(DirectRLEnv)" in env_source
    assert "self.relation_ids = torch.arange(self.num_envs, device=self.device) % self.relation_count" in env_source
    assert "preprocess_fabrica_actions" in env_source
    assert 'return {"policy": policy_observation, "critic": critic_state}' in env_source
    assert "terminated = self.invalid_states.clone()" in env_source
    assert "replicate_physics=False" in cfg_source
    assert "random_choice=False" in cfg_source
    assert "position_action_scale = 0.005" in cfg_source
    assert "observation_space = 3" in cfg_source
    assert "state_space = 6" in cfg_source
    assert "EPISODE_STEPS = 128" in cfg_source
    assert "POLICY_FREQUENCY_HZ = 30" in cfg_source
    assert "SIMULATION_FREQUENCY_HZ = 120" in cfg_source
    for body_name in ("link6", "gripper_base_link", "gripper_left_link", "gripper_right_link"):
        assert f'        "{body_name}",' in cfg_source
    assert "UsdPhysics.FilteredPairsAPI.Apply(socket)" in env_source
    assert "UsdPhysics.FilteredPairsAPI.Apply(tabletop)" in env_source
    assert "for relation_key in task.relation_keys" in replay_source


def test_fabrica_rl_games_specialist_parameters() -> None:
    config_path = TASK_ROOT / "agents" / "FabricaFixPlugTaskAssemblePPO.yaml"
    data = yaml.safe_load(config_path.read_text(encoding="utf-8"))["params"]
    assert data["network"]["mlp"] == {
        "units": [256, 128, 64],
        "activation": "elu",
        "d2rl": False,
        "initializer": {"name": "default"},
        "regularizer": {"name": "None"},
    }
    continuous = data["network"]["space"]["continuous"]
    assert continuous["fixed_sigma"] is True
    config = data["config"]
    assert config["horizon_length"] == 32
    assert config["minibatch_size"] == 512
    assert config["mini_epochs"] == 8
    assert config["learning_rate"] == pytest.approx(1.0e-4)
    assert config["max_epochs"] == 1500
    central = config["central_value_config"]
    assert central["minibatch_size"] == 256
    assert central["mini_epochs"] == 4
    assert central["learning_rate"] == pytest.approx(3.0e-4)
    assert central["lr_schedule"] == "linear"


@pytest.mark.skipif(not ISAAC_SIM_RUNTIME_AVAILABLE, reason="Isaac Sim runtime is not available")
def test_fabrica_env_cfg_uses_paired_multi_usd_spawners() -> None:
    from assembly_benchmark.tasks.direct.fabrica.fabrica_fixplug_task_assemble_cfg import (
        FabricaFixPlugTaskAssembleCfg,
    )

    cfg = FabricaFixPlugTaskAssembleCfg()
    assert cfg.action_space.shape == (3,)
    assert cfg.observation_space == 3
    assert cfg.state_space == 6
    assert cfg.decimation == 4
    assert cfg.episode_length_s == pytest.approx(128 / 30)
    assert cfg.scene.replicate_physics is False
    assert cfg.scene.robot.spawn.random_choice is False
    assert cfg.scene.socket.spawn.random_choice is False
    assert len(cfg.scene.robot.spawn.usd_path) == 4
    assert len(cfg.scene.socket.spawn.usd_path) == 4
    assert tuple(Path(path).parent.name for path in cfg.scene.robot.spawn.usd_path) == ("0_2", "1_3", "2_6", "3_6")
    assert tuple(Path(path).parents[1].name for path in cfg.scene.socket.spawn.usd_path) == (
        "0_2",
        "1_3",
        "2_6",
        "3_6",
    )
    assert all(Path(path).name == "socket.usd" for path in cfg.scene.socket.spawn.usd_path)
