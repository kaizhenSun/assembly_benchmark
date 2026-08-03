# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
CONTROLLER_PATH = REPO_ROOT / "source" / "assembly_benchmark" / "assembly_benchmark" / "controllers" / "single_arm.py"
ISAAC_SIM_RUNTIME_AVAILABLE = importlib.util.find_spec("carb") is not None


def _quat_mul(first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
    w1, x1, y1, z1 = first.unbind(dim=-1)
    w2, x2, y2, z2 = second.unbind(dim=-1)
    return torch.stack(
        (
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ),
        dim=-1,
    )


def _quat_inv(quaternion: torch.Tensor) -> torch.Tensor:
    result = quaternion.clone()
    result[..., 1:] *= -1.0
    return result / torch.sum(quaternion * quaternion, dim=-1, keepdim=True)


def _quat_matrix(quaternion: torch.Tensor) -> torch.Tensor:
    quaternion = quaternion / torch.linalg.vector_norm(quaternion, dim=-1, keepdim=True)
    w, x, y, z = quaternion.unbind(dim=-1)
    return torch.stack(
        (
            1 - 2 * (y * y + z * z),
            2 * (x * y - z * w),
            2 * (x * z + y * w),
            2 * (x * y + z * w),
            1 - 2 * (x * x + z * z),
            2 * (y * z - x * w),
            2 * (x * z - y * w),
            2 * (y * z + x * w),
            1 - 2 * (x * x + y * y),
        ),
        dim=-1,
    ).reshape(*quaternion.shape[:-1], 3, 3)


def _quat_apply(quaternion: torch.Tensor, vector: torch.Tensor) -> torch.Tensor:
    return torch.bmm(_quat_matrix(quaternion), vector.unsqueeze(-1)).squeeze(-1)


def _combine_frame_transforms(
    position_01: torch.Tensor,
    quaternion_01: torch.Tensor,
    position_12: torch.Tensor | None = None,
    quaternion_12: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    position_02 = position_01 if position_12 is None else position_01 + _quat_apply(quaternion_01, position_12)
    quaternion_02 = quaternion_01 if quaternion_12 is None else _quat_mul(quaternion_01, quaternion_12)
    return position_02, quaternion_02


def _subtract_frame_transforms(
    position_01: torch.Tensor,
    quaternion_01: torch.Tensor,
    position_02: torch.Tensor | None = None,
    quaternion_02: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    quaternion_10 = _quat_inv(quaternion_01)
    position_12 = (
        _quat_apply(quaternion_10, -position_01)
        if position_02 is None
        else _quat_apply(quaternion_10, position_02 - position_01)
    )
    quaternion_12 = quaternion_10 if quaternion_02 is None else _quat_mul(quaternion_10, quaternion_02)
    return position_12, quaternion_12


class _FakeDifferentialIKControllerCfg:
    def __init__(self, command_type, use_relative_mode, ik_method, ik_params):
        self.command_type = command_type
        self.use_relative_mode = use_relative_mode
        self.ik_method = ik_method
        self.ik_params = ik_params


class _FakeDifferentialIKController:
    def __init__(self, cfg, num_envs, device):
        self.cfg = cfg
        self.num_envs = num_envs
        self.device = device
        self.command = None
        self.compute_args = None
        self.reset_env_ids = None

    def reset(self, env_ids=None):
        self.reset_env_ids = env_ids

    def set_command(self, command):
        self.command = command.clone()

    def compute(self, *args):
        self.compute_args = args
        return args[-1] + 0.5


@pytest.fixture
def controller_module(monkeypatch):
    isaaclab_module = ModuleType("isaaclab")
    assets_module = ModuleType("isaaclab.assets")
    controllers_module = ModuleType("isaaclab.controllers")
    utils_module = ModuleType("isaaclab.utils")
    math_module = ModuleType("isaaclab.utils.math")
    assets_module.Articulation = object
    controllers_module.DifferentialIKController = _FakeDifferentialIKController
    controllers_module.DifferentialIKControllerCfg = _FakeDifferentialIKControllerCfg
    math_module.combine_frame_transforms = _combine_frame_transforms
    math_module.matrix_from_quat = _quat_matrix
    math_module.quat_inv = _quat_inv
    math_module.subtract_frame_transforms = _subtract_frame_transforms
    monkeypatch.setitem(sys.modules, "isaaclab", isaaclab_module)
    monkeypatch.setitem(sys.modules, "isaaclab.assets", assets_module)
    monkeypatch.setitem(sys.modules, "isaaclab.controllers", controllers_module)
    monkeypatch.setitem(sys.modules, "isaaclab.utils", utils_module)
    monkeypatch.setitem(sys.modules, "isaaclab.utils.math", math_module)

    spec = importlib.util.spec_from_file_location("_single_arm_controller_under_test", CONTROLLER_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _FakeRootPhysXView:
    def __init__(self, jacobians: torch.Tensor):
        self.jacobians = jacobians

    def get_jacobians(self) -> torch.Tensor:
        return self.jacobians


class _FakeRobot:
    def __init__(self, controller_module):
        self.joint_names = [f"joint{index}" for index in range(1, 7)] + ["left_joint", "right_joint"]
        self.body_names = ["base_link", "link6", "gripper_base_link"]
        self._joint_ids = {name: index for index, name in enumerate(self.joint_names)}
        self._body_ids = {name: index for index, name in enumerate(self.body_names)}
        num_envs = 2
        root_pos_w = torch.tensor(((0.0, 0.0, 0.0), (1.0, -2.0, 0.3)))
        half_sqrt_two = 2.0**-0.5
        root_quat_w = torch.tensor(((1.0, 0.0, 0.0, 0.0), (half_sqrt_two, 0.0, 0.0, half_sqrt_two)))
        root_pose_w = torch.cat((root_pos_w, root_quat_w), dim=-1)
        identity = torch.tensor((1.0, 0.0, 0.0, 0.0)).repeat(num_envs, 1)
        body_pose_w = torch.zeros((num_envs, len(self.body_names), 7))
        body_pose_w[:, :, 3] = 1.0
        body_pose_w[:, 0] = root_pose_w
        for body_index, body_position in ((1, (0.1, 0.0, 0.2)), (2, (0.1, 0.0, 0.25))):
            position_w, quaternion_w = _combine_frame_transforms(
                root_pos_w,
                root_quat_w,
                torch.tensor(body_position).repeat(num_envs, 1),
                identity,
            )
            body_pose_w[:, body_index] = torch.cat((position_w, quaternion_w), dim=-1)

        joint_pos = torch.zeros((num_envs, len(self.joint_names)))
        soft_limits = torch.empty((num_envs, len(self.joint_names), 2))
        soft_limits[..., 0] = -1.0
        soft_limits[..., 1] = 1.0
        soft_limits[:, -2, 0] = 0.0
        soft_limits[:, -2, 1] = 0.05
        soft_limits[:, -1, 0] = -0.05
        soft_limits[:, -1, 1] = 0.0
        velocity_limits = torch.full_like(joint_pos, 0.5)
        velocity_limits[:, -2:] = 0.05
        self.data = SimpleNamespace(
            root_pose_w=root_pose_w,
            body_pose_w=body_pose_w,
            joint_pos=joint_pos,
            soft_joint_pos_limits=soft_limits,
            joint_vel_limits=velocity_limits,
        )

        jacobian_b = torch.arange(num_envs * 2 * 6 * len(self.joint_names), dtype=torch.float32).reshape(
            num_envs, 2, 6, len(self.joint_names)
        )
        root_rotation = controller_module.matrix_from_quat(root_quat_w)
        jacobian_w = jacobian_b.clone()
        jacobian_w[:, :, :3] = torch.einsum("nij,nmjk->nmik", root_rotation, jacobian_b[:, :, :3])
        jacobian_w[:, :, 3:] = torch.einsum("nij,nmjk->nmik", root_rotation, jacobian_b[:, :, 3:])
        self.root_physx_view = _FakeRootPhysXView(jacobian_w)
        self.expected_jacobian_b = jacobian_b

    def find_joints(self, names, preserve_order=False):
        del preserve_order
        resolved = [name for name in names if name in self._joint_ids]
        return [self._joint_ids[name] for name in resolved], resolved

    def find_bodies(self, name, preserve_order=False):
        del preserve_order
        return ([self._body_ids[name]], [name]) if name in self._body_ids else ([], [])


def _make_controller(module, robot, control_dt=None):
    return module.SingleArmDifferentialIKController(
        robot=robot,
        arm_joint_names=[f"joint{index}" for index in range(1, 7)],
        gripper_joint_names=["left_joint", "right_joint"],
        ee_link_name="gripper_base_link",
        ik_link_name="link6",
        gripper_min=0.0,
        gripper_max=0.1,
        num_envs=2,
        device="cpu",
        control_dt=control_dt,
        gripper_joint_closed_positions=(0.0, 0.0),
        gripper_joint_open_positions=(0.05, -0.05),
    )


def test_single_arm_jacobian_index_and_dls_configuration(controller_module) -> None:
    robot = _FakeRobot(controller_module)
    controller = _make_controller(controller_module, robot)
    jacobian = controller._jacobian_in_root_frame(robot.data.root_pose_w)

    assert controller.ee_body_idx == 2
    assert controller.ik_body_idx == 1
    assert controller.ik_jacobian_idx == 0
    assert jacobian.shape == (2, 6, 6)
    assert torch.allclose(jacobian, robot.expected_jacobian_b[:, 0, :, :6], atol=1.0e-5)
    assert controller.ik.cfg.ik_method == "dls"
    assert controller.ik.cfg.ik_params == {"lambda_val": 0.08}


def test_single_arm_ee_command_converts_to_ik_link_and_gripper_target(controller_module) -> None:
    robot = _FakeRobot(controller_module)
    controller = _make_controller(controller_module, robot)
    actions = torch.zeros((2, 8))
    actions[:, :7] = torch.tensor((0.2, 0.1, 0.4, 1.0, 0.0, 0.0, 0.0))
    actions[:, 7] = 0.524
    target = controller.compute(actions)

    assert target.shape == (2, 8)
    assert torch.allclose(controller.ik.command[:, :3], torch.tensor(((0.2, 0.1, 0.35),) * 2), atol=1.0e-6)
    assert torch.allclose(controller.ik.command[:, 3:7], torch.tensor(((1.0, 0.0, 0.0, 0.0),) * 2))
    assert torch.allclose(target[:, :6], torch.full((2, 6), 0.5))
    assert target[:, 6].tolist() == pytest.approx((0.0381, 0.0381))
    assert target[:, 7].tolist() == pytest.approx((-0.0381, -0.0381))


def test_single_arm_maps_total_opening_to_signed_pika2_joint_targets(controller_module) -> None:
    controller = _make_controller(controller_module, _FakeRobot(controller_module))

    targets = controller.gripper_targets_from_opening(torch.tensor((0.0, 0.0127)))
    assert targets[0].tolist() == pytest.approx((0.0, 0.0))
    assert targets[1].tolist() == pytest.approx((0.00635, -0.00635))
    assert torch.allclose(
        controller.gripper_targets_from_opening(torch.tensor((0.1, 0.1))),
        torch.tensor(((0.05, -0.05), (0.05, -0.05))),
    )


def test_single_arm_default_gripper_mapping_remains_backward_compatible(controller_module) -> None:
    robot = _FakeRobot(controller_module)
    controller = controller_module.SingleArmDifferentialIKController(
        robot=robot,
        arm_joint_names=[f"joint{index}" for index in range(1, 7)],
        gripper_joint_names=["left_joint"],
        ee_link_name="gripper_base_link",
        ik_link_name="link6",
        gripper_min=0.0,
        gripper_max=0.1,
        num_envs=2,
        device="cpu",
    )

    targets = controller.gripper_targets_from_opening(torch.tensor((0.0, 0.1)))
    assert torch.allclose(targets, torch.tensor(((0.0,), (0.1,))))


def test_single_arm_limits_velocity_reset_and_invalid_inputs(controller_module) -> None:
    robot = _FakeRobot(controller_module)
    controller = _make_controller(controller_module, robot, control_dt=0.1)
    large_target = torch.full((2, 8), 10.0)
    large_target[:, -1] = -10.0
    limited = controller._apply_control_step_limits(large_target)
    assert torch.allclose(limited[:, :6], torch.full((2, 6), 0.05))
    assert torch.allclose(limited[:, 6:], torch.tensor(((0.005, -0.005),) * 2))

    zero_actions = torch.zeros((2, 8))
    controller.compute(zero_actions)
    held_pose = controller.pose_command.clone()
    assert torch.allclose(held_pose[:, :3], torch.tensor(((0.1, 0.0, 0.25),) * 2), atol=1.0e-6)
    robot.data.body_pose_w[0, controller.ee_body_idx, 0] += 0.02
    controller.compute(zero_actions)
    assert torch.equal(controller.pose_command, held_pose)
    controller.reset(torch.tensor([0]))
    controller.compute(zero_actions)
    assert controller.pose_command[0, 0] == pytest.approx(0.12)
    assert torch.equal(controller.pose_command[1], held_pose[1])

    with pytest.raises(ValueError, match="shape"):
        controller.compute(torch.zeros((1, 8)))
    invalid = torch.zeros((2, 8))
    invalid[0, 0] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        controller.compute(invalid)
    with pytest.raises(ValueError, match="control_dt"):
        _make_controller(controller_module, robot, control_dt=0.0)
    with pytest.raises(ValueError, match="gripper_max"):
        controller_module.SingleArmDifferentialIKController(
            robot,
            [f"joint{index}" for index in range(1, 7)],
            ["left_joint", "right_joint"],
            "gripper_base_link",
            "link6",
            0.1,
            0.1,
            2,
            "cpu",
        )


@pytest.mark.skipif(not ISAAC_SIM_RUNTIME_AVAILABLE, reason="Isaac Sim runtime is not available")
def test_single_arm_controller_imports_in_isaac_sim_runtime() -> None:
    from assembly_benchmark.controllers import SingleArmDifferentialIKController

    assert SingleArmDifferentialIKController.action_dim == 8
