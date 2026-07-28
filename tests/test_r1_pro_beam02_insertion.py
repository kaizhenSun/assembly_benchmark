# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import ast
import importlib.util
import math
import struct
from pathlib import Path
from types import ModuleType
from xml.etree import ElementTree

import pytest
import torch
from assembly_benchmark.beam02_grasp import (
    BEAM02_GRIPPER_OFFSET_IN_PLUG,
    BEAM02_GRIPPER_TO_PLUG_POS,
    BEAM02_GRIPPER_TO_PLUG_QUAT,
)
from assembly_benchmark.planning import BEAM02_APPROACH_DISTANCE, LinearInsertionPath

REPO_ROOT = Path(__file__).resolve().parents[1]
TASK_ROOT = (
    REPO_ROOT / "source" / "assembly_benchmark" / "assembly_benchmark" / "tasks" / "direct" / "r1_pro_beam_insertion"
)
TASK_ID = "Assembly-Benchmark-Beam02-LeftInsert-Direct-v0"
ISAAC_SIM_RUNTIME_AVAILABLE = importlib.util.find_spec("carb") is not None


def _load_task_geometry() -> ModuleType:
    module_path = TASK_ROOT / "task_geometry.py"
    spec = importlib.util.spec_from_file_location("beam02_task_geometry", module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _quat_multiply(first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
    w1, x1, y1, z1 = first.unbind()
    w2, x2, y2, z2 = second.unbind()
    return torch.stack(
        (
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        )
    )


def _quat_apply(quaternion: torch.Tensor, vector: torch.Tensor) -> torch.Tensor:
    quaternion = quaternion / torch.linalg.vector_norm(quaternion)
    vector_part = quaternion[1:]
    twice_cross = 2.0 * torch.linalg.cross(vector_part, vector)
    return vector + quaternion[0] * twice_cross + torch.linalg.cross(vector_part, twice_cross)


def _compose_pose(first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
    return torch.cat(
        (
            first[:3] + _quat_apply(first[3:7], second[:3]),
            _quat_multiply(first[3:7], second[3:7]),
        )
    )


def _inverse_pose(pose: torch.Tensor) -> torch.Tensor:
    inverse_quaternion = pose[3:7].clone()
    inverse_quaternion[1:] *= -1.0
    inverse_quaternion /= torch.sum(pose[3:7] * pose[3:7])
    return torch.cat((_quat_apply(inverse_quaternion, -pose[:3]), inverse_quaternion))


def _rpy_quaternion(rpy: tuple[float, float, float]) -> torch.Tensor:
    roll, pitch, yaw = rpy
    qx = torch.tensor((math.cos(roll / 2.0), math.sin(roll / 2.0), 0.0, 0.0), dtype=torch.float64)
    qy = torch.tensor((math.cos(pitch / 2.0), 0.0, math.sin(pitch / 2.0), 0.0), dtype=torch.float64)
    qz = torch.tensor((math.cos(yaw / 2.0), 0.0, 0.0, math.sin(yaw / 2.0)), dtype=torch.float64)
    return _quat_multiply(_quat_multiply(qz, qy), qx)


def _read_literal_assignments(path: Path, names: set[str]) -> dict[str, object]:
    syntax_tree = ast.parse(path.read_text(encoding="utf-8"))
    values: dict[str, object] = {}
    for statement in syntax_tree.body:
        if not isinstance(statement, ast.Assign) or len(statement.targets) != 1:
            continue
        target = statement.targets[0]
        if isinstance(target, ast.Name) and target.id in names:
            values[target.id] = ast.literal_eval(statement.value)
    return values


def _read_binary_stl_vertices(path: Path) -> list[torch.Tensor]:
    data = path.read_bytes()
    triangle_count = struct.unpack_from("<I", data, 80)[0]
    vertices = []
    for triangle_index in range(triangle_count):
        vertex_offset = 84 + triangle_index * 50 + 12
        vertices.extend(
            torch.tensor(struct.unpack_from("<fff", data, vertex_offset + vertex_index * 12), dtype=torch.float64)
            for vertex_index in range(3)
        )
    return vertices


def _read_obj_vertices(path: Path) -> list[torch.Tensor]:
    return [
        torch.tensor(tuple(float(value) for value in line.split()[1:4]), dtype=torch.float64)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.startswith("v ")
    ]


def _transform_vertices(pose: torch.Tensor, vertices: list[torch.Tensor]) -> torch.Tensor:
    return torch.stack([pose[:3] + _quat_apply(pose[3:7], vertex) for vertex in vertices])


def _urdf_link_poses(urdf_path: Path, joint_positions: dict[str, float]) -> dict[str, torch.Tensor]:
    root = ElementTree.parse(urdf_path).getroot()
    link_names = {link.get("name") for link in root.findall("link")}
    child_names = {joint.find("child").get("link") for joint in root.findall("joint")}
    root_link = next(iter(link_names - child_names))
    poses = {root_link: torch.tensor((0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0), dtype=torch.float64)}
    pending = list(root.findall("joint"))

    while pending:
        made_progress = False
        for joint in pending.copy():
            parent = joint.find("parent").get("link")
            child = joint.find("child").get("link")
            if parent not in poses:
                continue

            origin = joint.find("origin")
            xyz_text = origin.get("xyz", "0 0 0") if origin is not None else "0 0 0"
            rpy_text = origin.get("rpy", "0 0 0") if origin is not None else "0 0 0"
            xyz = tuple(float(value) for value in xyz_text.split())
            rpy = tuple(float(value) for value in rpy_text.split())
            parent_to_joint = torch.cat((torch.tensor(xyz, dtype=torch.float64), _rpy_quaternion(rpy)))
            motion = torch.tensor((0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0), dtype=torch.float64)
            joint_value = joint_positions.get(joint.get("name"), 0.0)
            joint_type = joint.get("type")
            axis_element = joint.find("axis")
            axis = torch.tensor(
                tuple(
                    float(value)
                    for value in (axis_element.get("xyz", "1 0 0") if axis_element is not None else "1 0 0").split()
                ),
                dtype=torch.float64,
            )
            axis /= torch.linalg.vector_norm(axis)
            if joint_type in ("revolute", "continuous"):
                half_angle = joint_value / 2.0
                motion[3] = math.cos(half_angle)
                motion[4:7] = axis * math.sin(half_angle)
            elif joint_type == "prismatic":
                motion[:3] = axis * joint_value

            poses[child] = _compose_pose(poses[parent], _compose_pose(parent_to_joint, motion))
            pending.remove(joint)
            made_progress = True
        if not made_progress:
            unresolved = [joint.get("name") for joint in pending]
            raise ValueError(f"Could not resolve URDF joint tree: {unresolved}.")
    return poses


def test_beam02_geometric_path_has_expected_endpoints_and_spacing() -> None:
    path = LinearInsertionPath(assembled_position=(0.55, 0.30, 0.775))
    waypoints = path.sample(5)

    assert pytest.approx(0.0202000377) == BEAM02_APPROACH_DISTANCE
    assert path.start_position == pytest.approx((0.55, 0.30, 0.7952000377))
    assert path.position(1.0) == pytest.approx((0.55, 0.30, 0.775))
    assert waypoints[0] == pytest.approx(path.start_position)
    assert waypoints[-1] == pytest.approx(path.assembled_position)
    assert [first[2] - second[2] for first, second in zip(waypoints[:-1], waypoints[1:], strict=True)] == pytest.approx(
        [BEAM02_APPROACH_DISTANCE / 4.0] * 4
    )


def test_geometric_path_axis_follows_assembled_orientation() -> None:
    half_angle = math.sqrt(0.5)
    path = LinearInsertionPath(
        assembled_position=(1.0, 2.0, 3.0),
        assembled_quaternion=(half_angle, 0.0, half_angle, 0.0),
        approach_distance=0.02,
    )

    assert path.world_approach_axis == pytest.approx((1.0, 0.0, 0.0), abs=1.0e-7)
    assert path.start_position == pytest.approx((1.02, 2.0, 3.0), abs=1.0e-7)


def test_geometric_path_distance_uses_finite_segment() -> None:
    path = LinearInsertionPath(assembled_position=(0.0, 0.0, 0.0), approach_distance=0.02)

    assert path.distance((0.003, 0.004, 0.01)) == pytest.approx(0.005)
    assert path.distance((0.0, 0.0, -0.003)) == pytest.approx(0.003)
    assert path.distance((0.0, 0.0, 0.025)) == pytest.approx(0.005)
    assert path.bounded_step_to_goal((0.0, 0.0, 0.02), 0.002) == pytest.approx((0.0, 0.0, -0.002))
    assert path.bounded_step_to_goal(path.assembled_position, 0.002) == pytest.approx((0.0, 0.0, 0.0))


def test_beam02_se3_chain_and_generated_fixed_joint_round_trip() -> None:
    gripper_to_plug = torch.tensor((*BEAM02_GRIPPER_TO_PLUG_POS, *BEAM02_GRIPPER_TO_PLUG_QUAT), dtype=torch.float64)
    root_to_plug = torch.tensor(
        (0.55, 0.30, 0.7952000377, 1.0, 0.0, 0.0, 0.0),
        dtype=torch.float64,
    )

    root_to_gripper = _compose_pose(root_to_plug, _inverse_pose(gripper_to_plug))
    reconstructed_plug = _compose_pose(root_to_gripper, gripper_to_plug)

    generator_source = (REPO_ROOT / "scripts" / "tools" / "generate_r1_pro_beam02_asset.py").read_text()

    assert reconstructed_plug[:3].tolist() == pytest.approx(root_to_plug[:3].tolist(), abs=1.0e-9)
    assert abs(float(torch.dot(reconstructed_plug[3:7], root_to_plug[3:7]))) == pytest.approx(1.0, abs=1.0e-9)
    assert "PLUG_JOINT_POS = BEAM02_GRIPPER_TO_PLUG_POS" in generator_source
    assert "PLUG_JOINT_RPY = _quaternion_to_rpy(BEAM02_GRIPPER_TO_PLUG_QUAT)" in generator_source

    old_plug_to_gripper_pos = torch.tensor((-0.044450077216, -0.001025430503, 0.110637162613))
    plug_to_gripper = _inverse_pose(gripper_to_plug)
    assert (plug_to_gripper[:3] - old_plug_to_gripper_pos).tolist() == pytest.approx(
        BEAM02_GRIPPER_OFFSET_IN_PLUG, abs=5.0e-9
    )

    cfg_values = _read_literal_assignments(
        TASK_ROOT / "r1_pro_beam_insertion_env_cfg.py",
        {"BEAM02_WHOLE_BODY_HOME_POS"},
    )
    reset_positions = cfg_values["BEAM02_WHOLE_BODY_HOME_POS"]
    assert reset_positions[:4] == pytest.approx((0.796419143480, -2.048491624328, -1.148481627732, -1.074710355439))
    controlled_joint_names = tuple(f"torso_joint{index}" for index in range(1, 5)) + tuple(
        f"left_arm_joint{index}" for index in range(1, 8)
    )
    joint_positions = dict(zip(controlled_joint_names, reset_positions, strict=True))
    robot_urdf = (
        REPO_ROOT
        / "source"
        / "assembly_benchmark"
        / "assembly_benchmark"
        / "assets"
        / "robots"
        / "r1_pro"
        / "robot.urdf"
    )
    reset_gripper_pose = _urdf_link_poses(robot_urdf, joint_positions)["left_gripper_link"]
    assert reset_gripper_pose[:3].tolist() == pytest.approx(root_to_gripper[:3].tolist(), abs=2.0e-6)
    assert abs(float(torch.dot(reset_gripper_pose[3:7], root_to_gripper[3:7]))) == pytest.approx(1.0, abs=2.0e-6)

    identity = torch.tensor((0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0), dtype=torch.float64)
    for pose in (gripper_to_plug, root_to_gripper):
        round_trip = _compose_pose(pose, _inverse_pose(pose))
        assert round_trip.tolist() == pytest.approx(identity.tolist(), abs=1.0e-9)


def test_beam02_offset_grasp_preserves_grip_and_socket_clearance() -> None:
    cfg_values = _read_literal_assignments(
        TASK_ROOT / "r1_pro_beam_insertion_env_cfg.py",
        {"BEAM02_WHOLE_BODY_HOME_POS", "BEAM02_LEFT_GRIPPER_POS"},
    )
    reset_positions = cfg_values["BEAM02_WHOLE_BODY_HOME_POS"]
    controlled_names = tuple(f"torso_joint{index}" for index in range(1, 5)) + tuple(
        f"left_arm_joint{index}" for index in range(1, 8)
    )
    joint_positions = dict(zip(controlled_names, reset_positions, strict=True))
    joint_positions.update(
        {
            "left_gripper_finger_joint1": cfg_values["BEAM02_LEFT_GRIPPER_POS"],
            "left_gripper_finger_joint2": cfg_values["BEAM02_LEFT_GRIPPER_POS"],
        }
    )

    asset_root = REPO_ROOT / "source" / "assembly_benchmark" / "assembly_benchmark" / "assets"
    robot_urdf = asset_root / "robots" / "r1_pro" / "robot.urdf"
    link_poses = _urdf_link_poses(robot_urdf, joint_positions)
    root_to_plug = torch.tensor((0.55, 0.30, 0.7952000377, 1.0, 0.0, 0.0, 0.0), dtype=torch.float64)
    plug_to_root = _inverse_pose(root_to_plug)

    finger_vertices_in_plug = []
    mesh_root = asset_root / "robots" / "r1_pro" / "meshes"
    for finger_index in (1, 2):
        link_name = f"left_gripper_finger_link{finger_index}"
        vertices = _read_binary_stl_vertices(mesh_root / f"{link_name}_collision.STL")
        root_vertices = _transform_vertices(link_poses[link_name], vertices)
        finger_vertices_in_plug.append(
            torch.stack([plug_to_root[:3] + _quat_apply(plug_to_root[3:7], vertex) for vertex in root_vertices])
        )

    beam_mesh_root = asset_root / "furniture" / "beam" / "mesh" / "beam"
    plug_vertices = torch.stack(_read_obj_vertices(beam_mesh_root / "beam_part_0.obj"))
    socket_vertices = torch.stack(_read_obj_vertices(beam_mesh_root / "beam_part_2.obj"))
    worst_case_socket_max_y = socket_vertices[:, 1].max() + 0.003

    for finger_vertices in finger_vertices_in_plug:
        assert float(finger_vertices[:, 1].min() - worst_case_socket_max_y) >= 0.0048
        grip_overlap_y = min(float(finger_vertices[:, 1].max()), float(plug_vertices[:, 1].max())) - max(
            float(finger_vertices[:, 1].min()), float(plug_vertices[:, 1].min())
        )
        grip_overlap_z = min(float(finger_vertices[:, 2].max()), float(plug_vertices[:, 2].max())) - max(
            float(finger_vertices[:, 2].min()), float(plug_vertices[:, 2].min())
        )
        assert grip_overlap_y >= 0.0238
        plug_thickness_z = float(plug_vertices[:, 2].max() - plug_vertices[:, 2].min())
        assert grip_overlap_z == pytest.approx(plug_thickness_z, abs=2.0e-6)


def test_batched_waypoint_and_final_clip_are_zero_norm_safe() -> None:
    geometry = _load_task_geometry()
    starts = torch.tensor([[0.0, 0.0, 0.02], [0.0, 0.0, 0.02]])
    goals = torch.zeros((2, 3))
    points = torch.tensor([[0.0, 0.0, 0.02], [0.004, 0.0, 0.01]])

    baseline = geometry.next_linear_waypoint_step(points, starts, goals, 0.002)
    combined = baseline + torch.tensor([[0.0, 0.0, 0.0], [-0.002, 0.0, 0.0]])
    clipped = geometry.clip_vector_norm(combined, 0.002)
    zero = geometry.clip_vector_norm(torch.zeros((2, 3)), 0.002)

    assert baseline[0].tolist() == pytest.approx([0.0, 0.0, -0.002])
    assert torch.linalg.vector_norm(clipped, dim=-1).tolist() == pytest.approx([0.002, 0.002])
    assert torch.equal(zero, torch.zeros_like(zero))
    assert torch.all(torch.isfinite(zero))


def test_observation_reward_latch_and_invalid_state_contract() -> None:
    geometry = _load_task_geometry()
    nominal_error = torch.tensor(((0.001, -0.002, 0.020), (0.0, 0.0, 0.010)))
    true_error = torch.tensor(((0.003, 0.0, 0.004), (0.0, 0.0, 0.040)))

    policy, critic = geometry.build_asymmetric_observations(nominal_error, true_error)
    reward, true_distance = geometry.dense_insertion_reward(true_error, reward_scale=1000.0, distance_cap=0.03)
    success, deviation = geometry.update_latched_outcomes(
        torch.tensor((False, True)),
        torch.tensor((False, False)),
        true_distance,
        torch.tensor((0.009, 0.0)),
        success_threshold=0.005,
        deviation_threshold=0.008,
    )

    assert torch.equal(policy, nominal_error)
    assert torch.equal(critic, torch.cat((nominal_error, true_error), dim=-1))
    assert policy.shape == (2, 3)
    assert critic.shape == (2, 6)
    assert true_distance.tolist() == pytest.approx((0.005, 0.04))
    assert reward.tolist() == pytest.approx((-5.0, -30.0))
    assert success.tolist() == [False, True]
    assert deviation.tolist() == [True, False]

    latched_success, latched_deviation = geometry.update_latched_outcomes(
        success,
        deviation,
        torch.tensor((0.004, 0.004)),
        torch.zeros(2),
        success_threshold=0.005,
        deviation_threshold=0.008,
    )
    assert latched_success.tolist() == [True, True]
    assert latched_deviation.tolist() == [True, False]

    finite_state = torch.zeros((2, 3, 2))
    state_with_nan = torch.zeros((2, 4))
    state_with_nan[1, 2] = float("nan")
    assert geometry.invalid_state_mask(finite_state, state_with_nan).tolist() == [False, True]
    assert geometry.episode_timeout_mask(torch.tensor((126, 127, 128)), 128).tolist() == [False, False, True]


def test_pose_drift_is_quaternion_sign_invariant() -> None:
    geometry = _load_task_geometry()
    current = torch.tensor(
        (
            (0.001, -0.002, 0.003, 1.0, 0.0, 0.0, 0.0),
            (0.0, 0.0, 0.0, math.sqrt(0.5), 0.0, 0.0, math.sqrt(0.5)),
        )
    )
    target = torch.tensor(
        (
            (0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0),
            (0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0),
        )
    )

    position_drift, orientation_drift = geometry.pose_drift(current, target)

    assert position_drift.tolist() == pytest.approx((math.sqrt(14.0) * 0.001, 0.0))
    assert orientation_drift.tolist() == pytest.approx((0.0, math.pi / 2.0), abs=1.0e-6)


def test_task_registration_and_training_config_contract() -> None:
    registration_source = (TASK_ROOT / "__init__.py").read_text(encoding="utf-8")
    env_source = (TASK_ROOT / "r1_pro_beam_insertion_env.py").read_text(encoding="utf-8")
    cfg_source = (TASK_ROOT / "r1_pro_beam_insertion_env_cfg.py").read_text(encoding="utf-8")
    replay_source = (REPO_ROOT / "scripts" / "tools" / "run_r1_pro_beam02_geometric_insertion.py").read_text(
        encoding="utf-8"
    )
    agent_cfg = pytest.importorskip("yaml").safe_load(
        (TASK_ROOT / "agents" / "rl_games_ppo_cfg.yaml").read_text(encoding="utf-8")
    )

    assert TASK_ID in registration_source
    assert "BimanualDifferentialIKController" in env_source
    assert "self.controller_actions[:, 0:7] = gripper_target_pose_b" in env_source
    assert "self.controller_actions[:, 15] = self.cfg.right_gripper_action" in env_source
    assert "self.controller_actions[:, 8:15]" not in env_source
    assert "self.invalid_states |= invalid_state_mask(actions)" in env_source
    assert "safe_actions.clamp(-1.0, 1.0)" in env_source
    assert "build_asymmetric_observations(nominal_delta_path, actual_delta_path)" in env_source
    assert 'return {"policy": policy_observation, "critic": critic_state}' in env_source
    assert "UsdPhysics.FilteredPairsAPI.Apply" in env_source
    assert "_current_invalid_state" in env_source
    assert "dense_insertion_reward(\n            true_error_w," in env_source
    assert "residual_step_path[:, 2]" not in env_source
    assert "episode_timeout_mask(self.episode_length_buf, self.max_episode_length)" in env_source
    assert 'tabletop_prim_path_suffix = "LabTable/Tabletop"' in cfg_source
    assert "BEAM02_GRIPPER_TO_PLUG_POS" in cfg_source
    assert "BEAM02_WHOLE_BODY_HOME_POS" in cfg_source
    assert "env_cfg.socket_position_noise = (0.0, 0.0, 0.0)" in replay_source
    assert '"--show_path"' in replay_source
    assert 'marker_cfg.prim_path = "/Visuals/Beam02Path"' in replay_source
    assert 'marker_cfg.markers["sphere"].radius = 0.001' in replay_source
    assert "raise SystemExit(0 if replay_passed else 1)" in replay_source

    params = agent_cfg["params"]
    assert params["env"]["clip_actions"] == 1.0
    assert params["network"]["mlp"]["units"] == [256, 128, 64]
    assert params["network"]["space"]["continuous"]["mu_activation"] == "None"
    assert params["network"]["space"]["continuous"]["sigma_activation"] == "None"
    assert params["network"]["space"]["continuous"]["fixed_sigma"] is True
    assert params["config"]["horizon_length"] == 32
    assert params["config"]["minibatch_size"] == 512
    assert params["config"]["mini_epochs"] == 8
    assert params["config"]["entropy_coef"] == pytest.approx(0.003)
    assert params["config"]["critic_coef"] == 2
    assert params["config"]["gamma"] == pytest.approx(0.99)
    assert params["config"]["tau"] == pytest.approx(0.95)
    assert params["config"]["learning_rate"] == pytest.approx(1.0e-4)
    assert params["config"]["lr_schedule"] == "fixed"
    assert params["config"]["normalize_input"] is True
    assert params["config"]["normalize_value"] is True
    assert params["config"]["max_epochs"] == 1500
    central_value = params["config"]["central_value_config"]
    assert central_value["minibatch_size"] == 512
    assert central_value["mini_epochs"] == 8
    assert central_value["learning_rate"] == pytest.approx(1.0e-4)
    assert central_value["lr_schedule"] == "fixed"
    assert central_value["network"]["central_value"] is True


def test_rl_games_actor_and_central_critic_network_startup() -> None:
    yaml = pytest.importorskip("yaml")
    model_builder_module = pytest.importorskip("rl_games.algos_torch.model_builder")
    params = yaml.safe_load((TASK_ROOT / "agents" / "rl_games_ppo_cfg.yaml").read_text(encoding="utf-8"))["params"]

    actor_builder = model_builder_module.ModelBuilder().load(params)
    actor = actor_builder.build(
        {
            "actions_num": 3,
            "input_shape": (3,),
            "num_seqs": 1,
            "value_size": 1,
            "normalize_value": True,
            "normalize_input": True,
        }
    )
    actor_output = actor(
        {
            "obs": torch.zeros((4, 3)),
            "prev_actions": torch.zeros((4, 3)),
            "is_train": True,
        }
    )

    central_value_config = params["config"]["central_value_config"]
    critic_builder = model_builder_module.ModelBuilder().load(
        {
            "model": {"name": "central_value"},
            "network": central_value_config["network"],
        }
    )
    critic = critic_builder.build(
        {
            "actions_num": 3,
            "input_shape": (6,),
            "num_seqs": 1,
            "value_size": 1,
            "normalize_value": True,
            "normalize_input": True,
        }
    )
    critic_output = critic({"obs": torch.zeros((4, 6)), "is_train": True})

    assert actor_output["mus"].shape == (4, 3)
    assert actor_output["sigmas"].shape == (4, 3)
    assert actor_output["values"].shape == (4, 1)
    assert critic_output["values"].shape == (4, 1)


@pytest.mark.skipif(not ISAAC_SIM_RUNTIME_AVAILABLE, reason="Isaac Sim runtime is not available")
def test_beam02_env_cfg_matches_policy_and_timing_contract() -> None:
    import gymnasium as gym
    from assembly_benchmark.beam02_grasp import (
        BEAM02_GRIPPER_TO_PLUG_POS,
        BEAM02_GRIPPER_TO_PLUG_QUAT,
    )
    from assembly_benchmark.robots.r1_pro import (
        R1_PRO_LEFT_ARM_JOINT_NAMES,
        R1_PRO_LEFT_GRIPPER_JOINT_NAMES,
        R1_PRO_RIGHT_ARM_JOINT_NAMES,
        R1_PRO_RIGHT_GRIPPER_JOINT_NAMES,
        R1_PRO_TORSO_JOINT_NAMES,
    )
    from assembly_benchmark.tasks.direct.r1_pro_beam_insertion.r1_pro_beam_insertion_env_cfg import (
        BEAM02_LEFT_GRIPPER_ACTION,
        BEAM02_LEFT_GRIPPER_POS,
        BEAM02_RIGHT_GRIPPER_ACTION,
        BEAM02_WHOLE_BODY_HOME_POS,
        R1ProBeam02InsertionEnvCfg,
    )

    cfg = R1ProBeam02InsertionEnvCfg()

    assert isinstance(cfg.action_space, gym.spaces.Box)
    assert cfg.action_space.shape == (3,)
    assert cfg.action_space.low.tolist() == [-1.0, -1.0, -1.0]
    assert cfg.action_space.high.tolist() == [1.0, 1.0, 1.0]
    assert cfg.observation_space == 3
    assert cfg.state_space == 6
    assert cfg.sim.dt == pytest.approx(1.0 / 120.0)
    assert cfg.decimation == 4
    assert cfg.episode_steps == 128
    assert cfg.episode_length_s == pytest.approx(128.0 / 30.0)
    assert cfg.scene.num_envs == 1024
    assert cfg.position_action_scale == pytest.approx(0.002)
    assert cfg.socket_position_noise == pytest.approx((0.003, 0.003, 0.003))
    assert cfg.gripper_to_plug_pos == pytest.approx(BEAM02_GRIPPER_TO_PLUG_POS)
    assert cfg.gripper_to_plug_quat == pytest.approx(BEAM02_GRIPPER_TO_PLUG_QUAT)
    assert cfg.left_gripper_action == pytest.approx(BEAM02_LEFT_GRIPPER_ACTION)
    assert cfg.right_gripper_action == pytest.approx(BEAM02_RIGHT_GRIPPER_ACTION)
    assert cfg.left_gripper_action == pytest.approx(-0.746)
    assert cfg.right_gripper_action == pytest.approx(1.0)
    controlled_names = R1_PRO_TORSO_JOINT_NAMES + R1_PRO_LEFT_ARM_JOINT_NAMES
    assert [cfg.scene.robot.init_state.joint_pos[name] for name in controlled_names] == pytest.approx(
        BEAM02_WHOLE_BODY_HOME_POS
    )
    assert [cfg.scene.robot.init_state.joint_pos[name] for name in R1_PRO_LEFT_GRIPPER_JOINT_NAMES] == pytest.approx(
        [BEAM02_LEFT_GRIPPER_POS, BEAM02_LEFT_GRIPPER_POS]
    )
    assert cfg.right_arm_joint_names == R1_PRO_RIGHT_ARM_JOINT_NAMES
    assert cfg.right_gripper_joint_names == R1_PRO_RIGHT_GRIPPER_JOINT_NAMES
    assert str(cfg.scene.robot.spawn.usd_path).endswith("r1_pro_beam02/r1_pro_beam02_fixed.usd")
