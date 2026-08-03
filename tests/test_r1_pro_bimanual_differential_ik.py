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
CONTROLLER_PATH = REPO_ROOT / "source" / "assembly_benchmark" / "assembly_benchmark" / "controllers" / "r1_pro.py"
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
        self.reset_env_ids = None

    def reset(self, env_ids=None):
        self.reset_env_ids = env_ids


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
    math_module.compute_pose_error = lambda current_pos, current_quat, *args, **kwargs: (
        torch.zeros_like(current_pos),
        torch.zeros_like(current_pos),
    )
    math_module.matrix_from_quat = _quat_matrix
    math_module.quat_inv = _quat_inv
    math_module.subtract_frame_transforms = _subtract_frame_transforms

    monkeypatch.setitem(sys.modules, "isaaclab", isaaclab_module)
    monkeypatch.setitem(sys.modules, "isaaclab.assets", assets_module)
    monkeypatch.setitem(sys.modules, "isaaclab.controllers", controllers_module)
    monkeypatch.setitem(sys.modules, "isaaclab.utils", utils_module)
    monkeypatch.setitem(sys.modules, "isaaclab.utils.math", math_module)

    module_spec = importlib.util.spec_from_file_location("_r1_pro_controller_under_test", CONTROLLER_PATH)
    assert module_spec is not None and module_spec.loader is not None
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    return module


class _FakeRootPhysXView:
    def __init__(self, jacobians: torch.Tensor):
        self._jacobians = jacobians

    def get_jacobians(self) -> torch.Tensor:
        return self._jacobians


class _FakeRobot:
    def __init__(self, controller_module):
        self.joint_names = [f"joint_{index}" for index in range(24)]
        self.body_names = ["base", "left_arm_link7", "left_gripper_link", "right_arm_link7", "right_gripper_link"]
        self._joint_ids_by_name = {
            **{f"torso_joint{index + 1}": index for index in range(4)},
            **{f"left_arm_joint{index + 1}": index + 4 for index in range(7)},
            **{f"left_gripper_finger_joint{index + 1}": index + 11 for index in range(2)},
            **{f"right_arm_joint{index + 1}": index + 13 for index in range(7)},
            **{f"right_gripper_finger_joint{index + 1}": index + 20 for index in range(2)},
        }
        self._body_ids_by_name = {name: index for index, name in enumerate(self.body_names)}

        num_envs = 2
        root_pos_w = torch.tensor(((0.0, 0.0, 0.0), (1.0, -2.0, 0.3)))
        half_sqrt_two = 2.0**-0.5
        root_quat_w = torch.tensor(((1.0, 0.0, 0.0, 0.0), (half_sqrt_two, 0.0, 0.0, half_sqrt_two)))
        root_pose_w = torch.cat((root_pos_w, root_quat_w), dim=-1)
        identity_quat = torch.tensor((1.0, 0.0, 0.0, 0.0)).repeat(num_envs, 1)

        body_pose_w = torch.zeros(num_envs, len(self.body_names), 7)
        body_pose_w[:, :, 3] = 1.0
        for body_index, position_b in (
            (1, (0.1, 0.2, 0.3)),
            (2, (0.1, 0.2, 0.4)),
            (3, (0.1, -0.2, 0.3)),
            (4, (0.1, -0.2, 0.4)),
        ):
            position_w, quaternion_w = _combine_frame_transforms(
                root_pos_w, root_quat_w, torch.tensor(position_b).repeat(num_envs, 1), identity_quat
            )
            body_pose_w[:, body_index] = torch.cat((position_w, quaternion_w), dim=-1)

        joint_pos = torch.zeros(num_envs, len(self.joint_names))
        soft_limits = torch.empty(num_envs, len(self.joint_names), 2)
        soft_limits[..., 0] = -1.0
        soft_limits[..., 1] = 1.0
        soft_limits[:, self._joint_ids_by_name["torso_joint1"], 1] = 0.02
        velocity_limits = torch.full((num_envs, len(self.joint_names)), 0.5)
        for name, joint_id in self._joint_ids_by_name.items():
            if "gripper" in name:
                velocity_limits[:, joint_id] = 0.05
        self.data = SimpleNamespace(
            root_pose_w=root_pose_w,
            body_pose_w=body_pose_w,
            joint_pos=joint_pos,
            default_joint_pos=joint_pos.clone(),
            soft_joint_pos_limits=soft_limits,
            joint_vel_limits=velocity_limits,
        )

        jacobian_b = torch.arange(num_envs * 4 * 6 * len(self.joint_names), dtype=torch.float32).reshape(
            num_envs, 4, 6, len(self.joint_names)
        )
        root_rotation = controller_module.matrix_from_quat(root_quat_w)
        jacobian_w = jacobian_b.clone()
        jacobian_w[:, :, :3] = torch.einsum("nij,nmjk->nmik", root_rotation, jacobian_b[:, :, :3])
        jacobian_w[:, :, 3:] = torch.einsum("nij,nmjk->nmik", root_rotation, jacobian_b[:, :, 3:])
        self.root_physx_view = _FakeRootPhysXView(jacobian_w)
        self.expected_jacobian_b = jacobian_b

    def find_joints(self, names, preserve_order=False):
        del preserve_order
        resolved_names = [name for name in names if name in self._joint_ids_by_name]
        return [self._joint_ids_by_name[name] for name in resolved_names], resolved_names

    def find_bodies(self, name, preserve_order=False):
        del preserve_order
        if name not in self._body_ids_by_name:
            return [], []
        return [self._body_ids_by_name[name]], [name]


def _make_controller(controller_module, robot, control_dt=None):
    return controller_module.BimanualDifferentialIKController(
        robot=robot,
        left_arm_joint_names=[f"left_arm_joint{index}" for index in range(1, 8)],
        right_arm_joint_names=[f"right_arm_joint{index}" for index in range(1, 8)],
        left_gripper_joint_names=[f"left_gripper_finger_joint{index}" for index in range(1, 3)],
        right_gripper_joint_names=[f"right_gripper_finger_joint{index}" for index in range(1, 3)],
        left_ee_link_name="left_gripper_link",
        right_ee_link_name="right_gripper_link",
        left_ik_link_name="left_arm_link7",
        right_ik_link_name="right_arm_link7",
        arm_action_scale=0.5,
        gripper_min=0.0,
        gripper_max=0.05,
        num_envs=2,
        device="cpu",
        torso_joint_names=[f"torso_joint{index}" for index in range(1, 5)],
        include_torso_in_ik=True,
        control_dt=control_dt,
    )


def _sample_controller_actions() -> torch.Tensor:
    actions = torch.zeros((2, 16))
    actions[:, 0:7] = torch.tensor((0.1, 0.2, 0.4, 1.0, 0.0, 0.0, 0.0))
    actions[:, 7] = -0.5
    actions[:, 15] = 1.0
    return actions


def test_bimanual_whole_body_jacobian_has_expected_task_blocks(controller_module) -> None:
    robot = _FakeRobot(controller_module)
    controller = _make_controller(controller_module, robot)
    jacobian = controller._combined_bimanual_jacobian(robot.data.root_pose_w)

    assert jacobian.shape == (2, 12, 18)
    assert torch.count_nonzero(jacobian[:, :6, 11:]) == 0
    assert torch.count_nonzero(jacobian[:, 6:, 4:11]) == 0
    assert torch.count_nonzero(jacobian[:, :6, :11]) > 0
    assert torch.count_nonzero(jacobian[:, 6:, :4]) > 0
    assert torch.count_nonzero(jacobian[:, 6:, 11:]) > 0


def test_zero_right_pose_latches_and_recaptures_per_environment(controller_module) -> None:
    robot = _FakeRobot(controller_module)
    controller = _make_controller(controller_module, robot)
    actions = _sample_controller_actions()

    target = controller.compute(actions)
    first_hold_pose = controller.right_pose_command.clone()
    assert target.shape == (2, 22)
    assert controller.right_pose_command_initialized.tolist() == [True, True]
    assert torch.allclose(first_hold_pose[:, :3], torch.tensor([[0.1, -0.2, 0.4]] * 2), atol=1.0e-6)

    robot.data.body_pose_w[0, controller.right_ee_body_idx, 0] += 0.02
    controller.compute(actions)
    assert torch.equal(controller.right_pose_command, first_hold_pose)

    env_ids = torch.tensor([0])
    controller.reset(env_ids)
    controller.compute(actions)
    assert controller.right_pose_command[0, 0] == pytest.approx(0.12)
    assert torch.equal(controller.right_pose_command[1], first_hold_pose[1])


def test_bimanual_gripper_targets_and_optional_step_limits(controller_module) -> None:
    robot = _FakeRobot(controller_module)
    legacy_controller = _make_controller(controller_module, robot)
    legacy_target = legacy_controller.compute(_sample_controller_actions())
    left_gripper_slice = slice(11, 13)
    right_gripper_slice = slice(20, 22)
    assert torch.allclose(legacy_target[:, left_gripper_slice], torch.full((2, 2), 0.0125), atol=1.0e-7)
    assert torch.allclose(legacy_target[:, right_gripper_slice], torch.full((2, 2), 0.05), atol=1.0e-7)

    limited_controller = _make_controller(controller_module, robot, control_dt=0.1)
    large_target = torch.full((2, 22), 10.0)
    limited_target = limited_controller._apply_control_step_limits(large_target, limited_controller.joint_ids)
    selected_velocity_limits = robot.data.joint_vel_limits[:, limited_controller.joint_ids]
    assert torch.all(limited_target <= selected_velocity_limits * 0.1 + 1.0e-7)
    assert limited_target[:, 0].tolist() == pytest.approx([0.02, 0.02])
    assert torch.allclose(limited_target[:, left_gripper_slice], torch.full((2, 2), 0.005), atol=1.0e-7)
    assert torch.allclose(limited_target[:, right_gripper_slice], torch.full((2, 2), 0.005), atol=1.0e-7)

    with pytest.raises(ValueError, match="control_dt"):
        _make_controller(controller_module, robot, control_dt=0.0)


def test_task_specific_left_whole_body_controller_is_not_exported() -> None:
    controller_source = CONTROLLER_PATH.read_text(encoding="utf-8")
    controller_init = CONTROLLER_PATH.with_name("__init__.py").read_text(encoding="utf-8")
    assert "R1ProLeftWholeBodyIKController" not in controller_source
    assert "R1ProLeftWholeBodyIKController" not in controller_init
    assert '"BimanualDifferentialIKController"' in controller_init


@pytest.mark.skipif(not ISAAC_SIM_RUNTIME_AVAILABLE, reason="Isaac Sim runtime is not available")
def test_bimanual_differential_ik_imports_in_isaac_sim_runtime() -> None:
    from assembly_benchmark.controllers import BimanualDifferentialIKController

    assert BimanualDifferentialIKController.action_dim == 16
