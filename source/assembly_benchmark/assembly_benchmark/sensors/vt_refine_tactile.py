# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""VT-Refine-style tactile force-field sensor for Isaac Lab scenes."""

from __future__ import annotations

import re
import struct
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import torch

TactilePadBounds = tuple[tuple[float, float, float], tuple[float, float, float]]


@dataclass(frozen=True, slots=True)
class TactileContactMaterialSpec:
    """PhysX compliant-contact parameters shared by tactile pad colliders."""

    stiffness: float
    damping: float
    acceleration_spring: bool
    static_friction: float
    dynamic_friction: float
    restitution: float
    contact_offset: float
    rest_offset: float


@dataclass(frozen=True, slots=True)
class R1ProGripperTactilePadSpec:
    """Canonical metadata for one R1 Pro gripper tactile pad."""

    hand: Literal["right", "left"]
    finger_index: Literal[1, 2]
    mesh_path: Path
    bounds: TactilePadBounds
    surface_sign: float

    def __post_init__(self) -> None:
        if self.hand not in ("right", "left"):
            raise ValueError(f"Unsupported gripper hand: {self.hand}")
        if self.finger_index not in (1, 2):
            raise ValueError(f"Unsupported gripper finger index: {self.finger_index}")
        if self.surface_sign not in (-1.0, 1.0):
            raise ValueError(f"surface_sign must be -1.0 or 1.0, got {self.surface_sign}")

    @property
    def label(self) -> str:
        return f"{self.hand}_pad{self.finger_index}"

    @property
    def sensor_name(self) -> str:
        return f"{self.hand}_gripper_tactile_sensor{self.finger_index}"

    @property
    def pad_link_name(self) -> str:
        return f"{self.hand}_gripper_tactile_pad{self.finger_index}"

    @property
    def parent_link_name(self) -> str:
        return f"{self.hand}_gripper_finger_link{self.finger_index}"

    @property
    def prim_path_expr(self) -> str:
        return f"{{ENV_REGEX_NS}}/Robot/.*{self.pad_link_name}"

    @property
    def size(self) -> tuple[float, float, float]:
        mins, maxs = self.bounds
        return tuple(max_value - min_value for min_value, max_value in zip(mins, maxs, strict=True))

    @property
    def normal_axis(self) -> tuple[float, float, float]:
        return (0.0, -self.surface_sign, 0.0)

    @property
    def panel_grid_coordinate(self) -> tuple[int, int]:
        return self.finger_index - 1, 0 if self.hand == "right" else 1

    @property
    def usd_visual_root_path(self) -> str:
        return f"/visuals/{self.pad_link_name}"

    @property
    def usd_collider_root_path(self) -> str:
        return f"/colliders/{self.pad_link_name}"

    @property
    def usd_material_path(self) -> str:
        return f"{self.usd_collider_root_path}/materials/{self.pad_link_name}_compliant_material"


try:
    from isaacsim.core.simulation_manager import SimulationManager
    from pxr import PhysxSchema, Sdf, Usd, UsdPhysics, UsdShade

    import isaaclab.sim as sim_utils
    from isaaclab.assets import RigidObjectCfg
    from isaaclab.sensors import SensorBase, SensorBaseCfg
    from isaaclab.utils import configclass
    from isaaclab.utils import math as math_utils

    _ISAAC_IMPORT_ERROR: ImportError | None = None
except ImportError as exc:
    sim_utils = None
    RigidObjectCfg = None
    math_utils = None
    SimulationManager = None
    PhysxSchema = None
    Sdf = None
    Usd = None
    UsdPhysics = None
    UsdShade = None
    _ISAAC_IMPORT_ERROR = exc

    class SensorBase:  # type: ignore[no-redef]
        """Fallback base class used when Isaac Sim runtime modules are unavailable."""

    @dataclass
    class SensorBaseCfg:  # type: ignore[no-redef]
        """Fallback config base used by pure helper tests."""

        class_type: type | None = None
        prim_path: str = ""
        update_period: float = 0.0
        history_length: int = 0
        debug_vis: bool = False

    def configclass(cls):  # type: ignore[no-redef]
        return dataclass(cls)


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TABLE_TACTILE_ARRAY_SIZE = (12, 32)
DEFAULT_TABLE_TACTILE_POINT_DISTANCE = 0.002
DEFAULT_TABLE_TACTILE_PAD_SIZE = (0.026, 0.06665, 0.003)
DEFAULT_TABLE_TACTILE_LOW_FORCE_THRESHOLD = 0.0002
DEFAULT_TABLE_TACTILE_LOW_FORCE_SCALE = 0.0008
DEFAULT_TABLE_TACTILE_COMPLIANT_CONTACT_STIFFNESS = 10.0
DEFAULT_TABLE_TACTILE_COMPLIANT_CONTACT_DAMPING = 1.0
DEFAULT_TABLE_TACTILE_NORMAL_CONTACT_STIFFNESS = 1.0
DEFAULT_TABLE_TACTILE_TANGENTIAL_CONTACT_STIFFNESS = 0.1
DEFAULT_TABLE_TACTILE_FRICTION_COEFFICIENT = 2.0
DEFAULT_TABLE_TACTILE_NORMAL_AXIS = (0.0, 0.0, -1.0)

R1_PRO_GRIPPER_TACTILE_MATERIAL = TactileContactMaterialSpec(
    stiffness=100.0,
    damping=0.01,
    acceleration_spring=False,
    static_friction=1.0,
    dynamic_friction=1.0,
    restitution=0.0,
    contact_offset=0.005,
    rest_offset=0.0,
)
R1_PRO_GRIPPER_TACTILE_VISUAL_RGBA = (0.42, 0.18, 0.07, 1.0)
R1_PRO_GRIPPER_TACTILE_NORMALIZATION_LOW_FORCE_THRESHOLD = 0.0

TABLE_TACTILE_PAD_POS = (0.5715, 0.0, 0.7765)
TABLE_TACTILE_PAD_PRIM_PATH = "{ENV_REGEX_NS}/TableTactilePad"
TABLE_TACTILE_SENSOR_PRIM_PATH = "{ENV_REGEX_NS}/TableTactilePad"

_ENV_REGEX_NS_PATTERN = re.compile(r"^(?P<env_ns>.*?/envs/env_[^/]+)")


def _require_isaac_lab() -> None:
    if _ISAAC_IMPORT_ERROR is not None:
        raise ImportError("Isaac Lab is required to instantiate VT-Refine tactile sensor scene entities.") from (
            _ISAAC_IMPORT_ERROR
        )


def tactile_point_count(array_size: Sequence[int]) -> int:
    """Return the number of tactile points for a rows/columns array."""
    if len(array_size) != 2:
        raise ValueError(f"Tactile array size must have two entries, got {tuple(array_size)}.")
    rows, cols = tuple(array_size)
    if rows <= 0 or cols <= 0:
        raise ValueError(f"Tactile array dimensions must be positive, got {tuple(array_size)}.")
    return int(rows) * int(cols)


def tactile_observation_size(array_size: Sequence[int]) -> int:
    """Return flattened ``[x, y, z, normal_force]`` tactile observation size."""
    return tactile_point_count(array_size) * 4


def tactile_force_grid(
    tactile_points: torch.Tensor,
    array_size: Sequence[int] = DEFAULT_TABLE_TACTILE_ARRAY_SIZE,
    clamp_max: float | None = None,
) -> torch.Tensor:
    """Return tactile normal force reshaped to the taxel grid."""
    num_points = tactile_point_count(array_size)
    if tactile_points.shape[-2:] != (num_points, 4):
        raise ValueError(
            f"Expected tactile points shape ending in ({num_points}, 4), got {tuple(tactile_points.shape)}."
        )

    posinf_value = 0.0 if clamp_max is None else clamp_max
    forces = torch.nan_to_num(tactile_points[..., 3], nan=0.0, posinf=posinf_value, neginf=0.0).clamp(min=0.0)
    if clamp_max is not None:
        forces = forces.clamp(max=clamp_max)
    return forces.reshape(*tactile_points.shape[:-2], int(array_size[0]), int(array_size[1]))


def _vertex_bounds(
    vertices: Sequence[Sequence[float]],
    mesh_path: str | Path,
) -> TactilePadBounds:
    if not vertices:
        raise ValueError(f"No vertices found in tactile pad mesh: {mesh_path}")
    mins = tuple(min(vertex[axis] for vertex in vertices) for axis in range(3))
    maxs = tuple(max(vertex[axis] for vertex in vertices) for axis in range(3))
    return mins, maxs


def read_stl_vertex_bounds(mesh_path: str | Path) -> TactilePadBounds:
    """Return binary or ASCII STL vertex min/max bounds."""
    mesh_file = Path(mesh_path)
    data = mesh_file.read_bytes()
    vertices: list[tuple[float, float, float]] = []
    if len(data) >= 84:
        triangle_count = struct.unpack_from("<I", data, 80)[0]
        if 84 + triangle_count * 50 == len(data):
            offset = 84
            for _ in range(triangle_count):
                offset += 12
                for _ in range(3):
                    vertices.append(struct.unpack_from("<fff", data, offset))
                    offset += 12
                offset += 2

    if not vertices:
        for line in data.decode("utf-8", errors="ignore").splitlines():
            parts = line.strip().split()
            if len(parts) == 4 and parts[0].lower() == "vertex":
                vertices.append((float(parts[1]), float(parts[2]), float(parts[3])))

    return _vertex_bounds(vertices, mesh_file)


def read_mesh_vertex_bounds(mesh_path: str | Path) -> TactilePadBounds:
    """Return tactile STL vertex min/max bounds."""
    mesh_file = Path(mesh_path)
    if mesh_file.suffix.lower() == ".stl":
        return read_stl_vertex_bounds(mesh_file)
    raise ValueError(f"Unsupported tactile pad mesh format: {mesh_file}")


_R1_PRO_GRIPPER_TACTILE_MESH_DIR = PACKAGE_ROOT / "assets" / "sensors" / "vt_refine_tactile"
_R1_PRO_GRIPPER_TACTILE_MESH_PATHS = {
    1: _R1_PRO_GRIPPER_TACTILE_MESH_DIR / "r1_pro_gripper_finger_link1_flat_pad.STL",
    2: _R1_PRO_GRIPPER_TACTILE_MESH_DIR / "r1_pro_gripper_finger_link2_flat_pad.STL",
}


def _make_r1_pro_gripper_tactile_pad_spec(
    hand: Literal["right", "left"],
    finger_index: Literal[1, 2],
    surface_sign: float,
) -> R1ProGripperTactilePadSpec:
    mesh_path = _R1_PRO_GRIPPER_TACTILE_MESH_PATHS[finger_index]
    return R1ProGripperTactilePadSpec(
        hand=hand,
        finger_index=finger_index,
        mesh_path=mesh_path,
        bounds=read_mesh_vertex_bounds(mesh_path),
        surface_sign=surface_sign,
    )


R1_PRO_GRIPPER_TACTILE_PAD_SPECS = (
    _make_r1_pro_gripper_tactile_pad_spec("right", 1, -1.0),
    _make_r1_pro_gripper_tactile_pad_spec("right", 2, 1.0),
    _make_r1_pro_gripper_tactile_pad_spec("left", 1, -1.0),
    _make_r1_pro_gripper_tactile_pad_spec("left", 2, 1.0),
)
R1_PRO_GRIPPER_TACTILE_OBSERVATION_SIZE = tactile_observation_size(DEFAULT_TABLE_TACTILE_ARRAY_SIZE) * len(
    R1_PRO_GRIPPER_TACTILE_PAD_SPECS
)


def apply_tactile_compliant_material(
    stage: Any,
    collider_prim: Any,
    material_path: str,
    spec: TactileContactMaterialSpec,
) -> Any:
    """Author and bind one independent compliant material to a tactile collider."""
    _require_isaac_lab()
    if not collider_prim.IsValid() or not collider_prim.HasAPI(UsdPhysics.CollisionAPI):
        raise RuntimeError(f"Expected tactile collision prim, got: {collider_prim.GetPath()}")

    physx_collision_api = PhysxSchema.PhysxCollisionAPI.Apply(collider_prim)
    physx_collision_api.CreateContactOffsetAttr().Set(spec.contact_offset)
    physx_collision_api.CreateRestOffsetAttr().Set(spec.rest_offset)

    material = UsdShade.Material.Define(stage, material_path)
    material_prim = material.GetPrim()
    physics_material_api = UsdPhysics.MaterialAPI.Apply(material_prim)
    physics_material_api.CreateStaticFrictionAttr().Set(spec.static_friction)
    physics_material_api.CreateDynamicFrictionAttr().Set(spec.dynamic_friction)
    physics_material_api.CreateRestitutionAttr().Set(spec.restitution)

    applied_schemas = list(material_prim.GetAppliedSchemas())
    if "PhysxMaterialAPI" not in applied_schemas:
        applied_schemas.append("PhysxMaterialAPI")
        material_prim.SetMetadata("apiSchemas", Sdf.TokenListOp.CreateExplicit(applied_schemas))
    material_prim.CreateAttribute("physxMaterial:compliantContactStiffness", Sdf.ValueTypeNames.Float).Set(
        spec.stiffness
    )
    material_prim.CreateAttribute("physxMaterial:compliantContactDamping", Sdf.ValueTypeNames.Float).Set(spec.damping)
    material_prim.CreateAttribute("physxMaterial:compliantContactAccelerationSpring", Sdf.ValueTypeNames.Bool).Set(
        spec.acceleration_spring
    )

    UsdShade.MaterialBindingAPI.Apply(collider_prim).Bind(
        material,
        bindingStrength=UsdShade.Tokens.strongerThanDescendants,
        materialPurpose="physics",
    )
    return material


def _validate_pad_size(pad_size: Sequence[float]) -> tuple[float, float, float]:
    if len(pad_size) != 3:
        raise ValueError(f"pad_size must have three entries, got {tuple(pad_size)}.")
    sizes = tuple(float(value) for value in pad_size)
    if any(value <= 0.0 for value in sizes):
        raise ValueError(f"pad_size entries must be positive, got {sizes}.")
    return sizes


def _validate_pad_bounds(
    pad_bounds: Sequence[Sequence[float]],
) -> TactilePadBounds:
    if len(pad_bounds) != 2:
        raise ValueError(f"pad_bounds must contain min/max points, got {tuple(pad_bounds)}.")
    mins = tuple(float(value) for value in pad_bounds[0])
    maxs = tuple(float(value) for value in pad_bounds[1])
    if len(mins) != 3 or len(maxs) != 3:
        raise ValueError(f"pad_bounds min/max points must be 3D, got {pad_bounds}.")
    if any(max_value <= min_value for min_value, max_value in zip(mins, maxs, strict=True)):
        raise ValueError(f"pad_bounds max values must be greater than min values, got {pad_bounds}.")
    return mins, maxs


def _validate_surface_axis(surface_axis: int) -> int:
    if surface_axis not in (0, 1, 2):
        raise ValueError(f"surface_axis must be 0, 1, or 2, got {surface_axis}.")
    return int(surface_axis)


def _validate_grid_axes(grid_axes: Sequence[int], surface_axis: int) -> tuple[int, int]:
    if len(grid_axes) != 2:
        raise ValueError(f"grid_axes must have two entries, got {tuple(grid_axes)}.")
    axes = tuple(int(axis) for axis in grid_axes)
    if len(set(axes)) != 2 or any(axis not in (0, 1, 2) for axis in axes):
        raise ValueError(f"grid_axes must contain two distinct axes from 0, 1, 2; got {axes}.")
    if surface_axis in axes:
        raise ValueError(f"grid_axes {axes} must not include surface_axis {surface_axis}.")
    return axes


def generate_tactile_points_on_rectangular_pad(
    *,
    pad_size: Sequence[float],
    surface_axis: int,
    surface_sign: float,
    grid_axes: Sequence[int],
    array_size: Sequence[int] = DEFAULT_TABLE_TACTILE_ARRAY_SIZE,
    point_distance: float = DEFAULT_TABLE_TACTILE_POINT_DISTANCE,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Generate centered taxel points on one face of a rectangular elastomer pad."""
    rows, cols = tuple(array_size)
    tactile_point_count(array_size)
    pad_size = _validate_pad_size(pad_size)
    surface_axis = _validate_surface_axis(surface_axis)
    grid_axis_0, grid_axis_1 = _validate_grid_axes(grid_axes, surface_axis)
    if point_distance <= 0.0:
        raise ValueError(f"point_distance must be positive, got {point_distance}.")
    if surface_sign == 0.0:
        raise ValueError("surface_sign must be non-zero.")

    axis_0_values = torch.linspace(
        -point_distance * (int(rows) - 1) / 2.0,
        point_distance * (int(rows) - 1) / 2.0,
        int(rows),
        device=device,
        dtype=dtype,
    )
    axis_1_values = torch.linspace(
        -point_distance * (int(cols) - 1) / 2.0,
        point_distance * (int(cols) - 1) / 2.0,
        int(cols),
        device=device,
        dtype=dtype,
    )
    grid_axis_0_values, grid_axis_1_values = torch.meshgrid(axis_0_values, axis_1_values, indexing="ij")

    points = torch.zeros((int(rows), int(cols), 3), dtype=dtype, device=device)
    points[..., surface_axis] = float(surface_sign) / abs(float(surface_sign)) * pad_size[surface_axis] / 2.0
    points[..., grid_axis_0] = grid_axis_0_values
    points[..., grid_axis_1] = grid_axis_1_values
    return points.reshape(-1, 3)


def generate_tactile_points_on_rectangular_pad_from_bounds(
    *,
    pad_bounds: Sequence[Sequence[float]],
    surface_axis: int,
    surface_sign: float,
    grid_axes: Sequence[int],
    array_size: Sequence[int] = DEFAULT_TABLE_TACTILE_ARRAY_SIZE,
    point_distance: float = DEFAULT_TABLE_TACTILE_POINT_DISTANCE,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Generate taxel points on one rectangular mesh-bound face."""
    mins, maxs = _validate_pad_bounds(pad_bounds)
    centered_points = generate_tactile_points_on_rectangular_pad(
        pad_size=tuple(max_value - min_value for min_value, max_value in zip(mins, maxs, strict=True)),
        surface_axis=surface_axis,
        surface_sign=surface_sign,
        grid_axes=grid_axes,
        array_size=array_size,
        point_distance=point_distance,
        device=device,
        dtype=dtype,
    )
    center = torch.tensor(
        [(min_value + max_value) / 2.0 for min_value, max_value in zip(mins, maxs, strict=True)],
        device=device,
        dtype=dtype,
    )
    return centered_points + center


def generate_table_tactile_points_local(
    pad_size: Sequence[float] | None = None,
    array_size: Sequence[int] = DEFAULT_TABLE_TACTILE_ARRAY_SIZE,
    point_distance: float = DEFAULT_TABLE_TACTILE_POINT_DISTANCE,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Generate VT-Refine-style taxel points on the table-mounted elastomer top surface."""
    return generate_tactile_points_on_rectangular_pad(
        pad_size=DEFAULT_TABLE_TACTILE_PAD_SIZE if pad_size is None else pad_size,
        surface_axis=2,
        surface_sign=1.0,
        grid_axes=(0, 1),
        array_size=array_size,
        point_distance=point_distance,
        device=device,
        dtype=dtype,
    )


def resolve_tactile_contact_part_names(
    label: str,
    configured_names: Sequence[str],
    default_part_names: Sequence[str],
    all_part_names: Sequence[str],
) -> tuple[str, ...]:
    """Resolve configured contact targets and validate they are assembly scene keys."""
    names = tuple(configured_names) if len(configured_names) > 0 else tuple(default_part_names)
    known_names = set(all_part_names)
    unknown_names = tuple(name for name in names if name not in known_names)
    if unknown_names:
        unknown = ", ".join(unknown_names)
        known = ", ".join(all_part_names)
        raise ValueError(f"Unknown {label} contact part(s): {unknown}. Known assembly parts: {known}.")
    return names


def resolve_table_tactile_contact_part_names(
    configured_names: Sequence[str],
    default_part_names: Sequence[str],
    all_part_names: Sequence[str],
) -> tuple[str, ...]:
    """Resolve configured table tactile contact targets and validate they are assembly scene keys."""
    return resolve_tactile_contact_part_names("table tactile", configured_names, default_part_names, all_part_names)


def normalize_tactile_normal_force(
    normal_force: torch.Tensor,
    *,
    low_force_threshold: float = DEFAULT_TABLE_TACTILE_LOW_FORCE_THRESHOLD,
    low_force_scale: float = DEFAULT_TABLE_TACTILE_LOW_FORCE_SCALE,
) -> torch.Tensor:
    """Normalize normal forces with the VT-Refine tactile-points convention."""
    values = torch.nan_to_num(normal_force, nan=0.0, posinf=0.0, neginf=0.0).clamp(min=0.0)
    max_force = values.max(dim=-1, keepdim=True).values
    scales = torch.where(
        max_force < low_force_threshold,
        torch.full_like(max_force, low_force_scale),
        max_force.clamp_min(1.0e-12),
    )
    return (values / scales).clamp(0.0, 1.0)


def _infer_env_regex_ns_from_sensor_prim_path(
    sensor_prim_path: str, configured_env_expr: str = "{ENV_REGEX_NS}"
) -> str:
    match = _ENV_REGEX_NS_PATTERN.match(sensor_prim_path)
    if match is not None:
        return match.group("env_ns")
    return sensor_prim_path.rsplit("/", 1)[0].rstrip("/") or configured_env_expr


@dataclass
class VtRefineTactileSensorData:
    """Data container for VT-Refine-style tactile points."""

    tactile_points_pos_w: torch.Tensor | None = None
    tactile_points_quat_w: torch.Tensor | None = None
    penetration_depth: torch.Tensor | None = None
    tactile_normal_force: torch.Tensor | None = None


@dataclass(slots=True)
class _ContactTargetRuntime:
    sdf_view: Any
    body_view: Any
    com_b: torch.Tensor
    mesh_pos_local: torch.Tensor
    mesh_quat_local: torch.Tensor


class VtRefineTactileSensor(SensorBase):
    """SDF force-field tactile sensor that outputs VT-Refine-style tactile points."""

    cfg: VtRefineTactileSensorCfg

    def __init__(self, cfg: Any):
        _require_isaac_lab()
        self._data = VtRefineTactileSensorData()
        self._physics_sim_view = None
        self._pad_body_view = None
        self._contact_targets: list[_ContactTargetRuntime] = []
        self._tactile_pos_local = None
        self._tactile_quat_local = None
        self._tactile_pos_expanded = None
        self._tactile_quat_expanded = None
        self._tactile_normal_axis_tensor = None
        self._pad_pos_w = None
        self._pad_quat_w = None
        self._pad_com_b = None
        super().__init__(cfg)

    @property
    def num_tactile_points(self) -> int:
        return tactile_point_count(self.cfg.tactile_array_size)

    @property
    def data(self) -> VtRefineTactileSensorData:
        self._update_outdated_buffers()
        return self._data

    def _initialize_impl(self) -> None:
        super()._initialize_impl()
        self._physics_sim_view = SimulationManager.get_physics_sim_view()
        self._generate_tactile_points()
        self._create_pad_view()
        self._create_contact_views()
        self._initialize_buffers()

    def _generate_tactile_points(self) -> None:
        if self.cfg.tactile_pad_bounds is None:
            self._tactile_pos_local = generate_tactile_points_on_rectangular_pad(
                pad_size=self.cfg.pad_size,
                surface_axis=self.cfg.tactile_surface_axis,
                surface_sign=self.cfg.tactile_surface_sign,
                grid_axes=self.cfg.tactile_grid_axes,
                array_size=self.cfg.tactile_array_size,
                point_distance=self.cfg.tactile_point_distance,
                device=self._device,
                dtype=torch.float32,
            )
        else:
            self._tactile_pos_local = generate_tactile_points_on_rectangular_pad_from_bounds(
                pad_bounds=self.cfg.tactile_pad_bounds,
                surface_axis=self.cfg.tactile_surface_axis,
                surface_sign=self.cfg.tactile_surface_sign,
                grid_axes=self.cfg.tactile_grid_axes,
                array_size=self.cfg.tactile_array_size,
                point_distance=self.cfg.tactile_point_distance,
                device=self._device,
                dtype=torch.float32,
            )
        self._tactile_quat_local = torch.tensor(
            [[1.0, 0.0, 0.0, 0.0]] * self.num_tactile_points,
            dtype=torch.float32,
            device=self._device,
        )
        self._tactile_pos_expanded = self._tactile_pos_local.unsqueeze(0).repeat(self._num_envs, 1, 1)
        self._tactile_quat_expanded = self._tactile_quat_local.unsqueeze(0).repeat(self._num_envs, 1, 1)
        self._tactile_normal_axis_tensor = torch.tensor(
            self.cfg.tactile_normal_axis,
            dtype=torch.float32,
            device=self._device,
        )

    def _create_pad_view(self) -> None:
        pad_paths = sim_utils.find_matching_prim_paths(self.cfg.prim_path)
        if len(pad_paths) == 0:
            raise RuntimeError(f"No tactile pad body found for prim path expression: {self.cfg.prim_path}")
        pad_path_pattern = self.cfg.prim_path.replace(".*", "*")
        self._pad_body_view = self._physics_sim_view.create_rigid_body_view([pad_path_pattern])
        self._pad_com_b = self._pad_body_view.get_coms().to(self._device).split([3, 4], dim=-1)[0]

    def _create_contact_views(self) -> None:
        env_regex_ns = _infer_env_regex_ns_from_sensor_prim_path(self.cfg.prim_path, self.cfg.env_prim_path_expr)
        num_query_points = self.num_tactile_points
        self._contact_targets = []

        for contact_prim_path_expr in self.cfg.contact_prim_paths_expr:
            formatted_expr = contact_prim_path_expr.format(ENV_REGEX_NS=env_regex_ns)
            contact_mesh, contact_body = self._find_contact_object_components(formatted_expr)
            mesh_path_pattern = contact_mesh.GetPath().pathString.replace("env_0", "env_*").replace(".*", "*")
            body_path_pattern = contact_body.GetPath().pathString.replace("env_0", "env_*").replace(".*", "*")
            sdf_view = self._physics_sim_view.create_sdf_shape_view(mesh_path_pattern, num_query_points)
            contact_body_view = self._physics_sim_view.create_rigid_body_view([body_path_pattern])
            mesh_pos, mesh_quat = sim_utils.resolve_prim_pose(contact_mesh, contact_body)
            self._contact_targets.append(
                _ContactTargetRuntime(
                    sdf_view=sdf_view,
                    body_view=contact_body_view,
                    com_b=contact_body_view.get_coms().to(self._device).split([3, 4], dim=-1)[0],
                    mesh_pos_local=torch.tensor(mesh_pos, dtype=torch.float32, device=self._device),
                    mesh_quat_local=torch.tensor(mesh_quat, dtype=torch.float32, device=self._device),
                )
            )

    def _find_contact_object_components(self, contact_prim_path_expr: str) -> tuple[Any, Any]:
        contact_object_prim = sim_utils.find_first_matching_prim(contact_prim_path_expr)
        if contact_object_prim is None:
            raise RuntimeError(f"No contact object prim found matching pattern: {contact_prim_path_expr}")

        def is_sdf_mesh(prim: Any) -> bool:
            return (
                prim.HasAPI(UsdPhysics.MeshCollisionAPI)
                and UsdPhysics.MeshCollisionAPI(prim).GetApproximationAttr().Get() == "sdf"
            )

        contact_mesh = sim_utils.get_first_matching_child_prim(contact_object_prim.GetPath(), predicate=is_sdf_mesh)
        if contact_mesh is None:
            raise RuntimeError(
                f"No SDF mesh found under contact object at path: {contact_object_prim.GetPath().pathString}"
            )

        contact_body = self._find_parent_rigid_body(contact_mesh)
        if contact_body is None:
            raise RuntimeError(
                f"No rigid body parent found for contact mesh at path: {contact_mesh.GetPath().pathString}"
            )
        return contact_mesh, contact_body

    @staticmethod
    def _find_parent_rigid_body(prim: Any) -> Any | None:
        current_prim = prim
        while current_prim is not None and current_prim.IsValid():
            if current_prim.HasAPI(UsdPhysics.RigidBodyAPI):
                return current_prim
            current_prim = current_prim.GetParent()
            if current_prim.GetPath() == "/":
                break
        return None

    def _initialize_buffers(self) -> None:
        self._data.tactile_points_pos_w = torch.zeros(
            self._num_envs, self.num_tactile_points, 3, dtype=torch.float32, device=self._device
        )
        self._data.tactile_points_quat_w = torch.zeros(
            self._num_envs, self.num_tactile_points, 4, dtype=torch.float32, device=self._device
        )
        self._data.penetration_depth = torch.zeros(
            self._num_envs, self.num_tactile_points, dtype=torch.float32, device=self._device
        )
        self._data.tactile_normal_force = torch.zeros(
            self._num_envs, self.num_tactile_points, dtype=torch.float32, device=self._device
        )

    def _update_buffers_impl(self, env_ids: Sequence[int] | slice) -> None:
        if isinstance(env_ids, torch.Tensor):
            env_ids = env_ids.to(device=self._device)
        self._update_tactile_points_pose(env_ids)
        self._data.penetration_depth[env_ids] = 0.0
        self._data.tactile_normal_force[env_ids] = 0.0
        for target in self._contact_targets:
            self._accumulate_contact_object_forces(
                env_ids,
                target.sdf_view,
                target.body_view,
                target.com_b,
                target.mesh_pos_local,
                target.mesh_quat_local,
            )

    def _update_tactile_points_pose(self, env_ids: Sequence[int] | slice) -> None:
        pad_pos_w, pad_quat_w = self._pad_body_view.get_transforms().split([3, 4], dim=-1)
        pad_quat_w = math_utils.convert_quat(pad_quat_w, to="wxyz")
        self._pad_pos_w = pad_pos_w
        self._pad_quat_w = pad_quat_w

        pad_pos_expanded = pad_pos_w.unsqueeze(1).expand(-1, self.num_tactile_points, -1)
        pad_quat_expanded = pad_quat_w.unsqueeze(1).expand(-1, self.num_tactile_points, -1)
        self._data.tactile_points_pos_w[env_ids] = (
            math_utils.quat_apply(pad_quat_expanded, self._tactile_pos_expanded) + pad_pos_expanded
        )[env_ids]
        self._data.tactile_points_quat_w[env_ids] = math_utils.quat_mul(pad_quat_expanded, self._tactile_quat_expanded)[
            env_ids
        ]

    def _accumulate_contact_object_forces(
        self,
        env_ids: Sequence[int] | slice,
        sdf_view: Any,
        body_view: Any,
        contact_com_b: torch.Tensor,
        mesh_pos_local: torch.Tensor,
        mesh_quat_local: torch.Tensor,
    ) -> None:
        object_pos_w, object_quat_w = body_view.get_transforms().split([3, 4], dim=-1)
        object_quat_w = math_utils.convert_quat(object_quat_w, to="wxyz")
        mesh_pos_body = mesh_pos_local.unsqueeze(0).expand(self._num_envs, -1)
        mesh_quat_body = mesh_quat_local.unsqueeze(0).expand(self._num_envs, -1)
        mesh_pos_w = math_utils.quat_apply(object_quat_w, mesh_pos_body) + object_pos_w
        mesh_quat_w = math_utils.quat_mul(object_quat_w, mesh_quat_body)

        mesh_quat_inv = math_utils.quat_inv(mesh_quat_w)
        mesh_quat_inv_expanded = mesh_quat_inv.unsqueeze(1).expand(-1, self.num_tactile_points, -1)
        mesh_pos_expanded = mesh_pos_w.unsqueeze(1).expand(-1, self.num_tactile_points, -1)
        points_mesh_local = math_utils.quat_apply(
            mesh_quat_inv_expanded, self._data.tactile_points_pos_w - mesh_pos_expanded
        )

        sdf_values_and_gradients = sdf_view.get_sdf_and_gradients(points_mesh_local)
        sdf_values = sdf_values_and_gradients[..., 3]
        sdf_gradients = sdf_values_and_gradients[..., :3]
        current_depth = (-sdf_values).clamp(min=0.0)
        if not (current_depth[env_ids] > 0.0).any():
            return

        normals_local = torch.nn.functional.normalize(sdf_gradients, dim=-1)
        mesh_quat_expanded = mesh_quat_w.unsqueeze(1).expand(-1, self.num_tactile_points, -1)
        normals_world = math_utils.quat_apply(mesh_quat_expanded, normals_local)

        pad_vel = self._pad_body_view.get_velocities()
        object_vel = body_view.get_velocities()
        pad_linvel_w_com = pad_vel[:, :3]
        pad_angvel_w = pad_vel[:, 3:]
        object_linvel_w_com = object_vel[:, :3]
        object_angvel_w = object_vel[:, 3:]
        pad_com_w_offset = math_utils.quat_apply(self._pad_quat_w, self._pad_com_b)
        object_com_w_offset = math_utils.quat_apply(object_quat_w, contact_com_b)
        pad_linvel_w = pad_linvel_w_com - torch.linalg.cross(pad_angvel_w, pad_com_w_offset)
        object_linvel_w = object_linvel_w_com - torch.linalg.cross(object_angvel_w, object_com_w_offset)

        pad_angvel_expanded = pad_angvel_w.unsqueeze(1).expand(-1, self.num_tactile_points, -1)
        pad_linvel_expanded = pad_linvel_w.unsqueeze(1).expand(-1, self.num_tactile_points, -1)
        tactile_offsets_w = math_utils.quat_apply(
            self._pad_quat_w.unsqueeze(1).expand(-1, self.num_tactile_points, -1),
            self._tactile_pos_expanded,
        )
        tactile_points_linvel = torch.linalg.cross(pad_angvel_expanded, tactile_offsets_w) + pad_linvel_expanded

        closest_points_mesh_local = points_mesh_local + current_depth.unsqueeze(-1) * normals_local
        object_angvel_expanded = object_angvel_w.unsqueeze(1).expand(-1, self.num_tactile_points, -1)
        object_linvel_expanded = object_linvel_w.unsqueeze(1).expand(-1, self.num_tactile_points, -1)
        mesh_offset_w = (mesh_pos_w - object_pos_w).unsqueeze(1).expand(-1, self.num_tactile_points, -1)
        closest_offsets_w = mesh_offset_w + math_utils.quat_apply(mesh_quat_expanded, closest_points_mesh_local)
        closest_points_linvel = torch.linalg.cross(object_angvel_expanded, closest_offsets_w) + object_linvel_expanded

        relative_linvel = tactile_points_linvel - closest_points_linvel
        relative_vt = relative_linvel - normals_world * (normals_world * relative_linvel).sum(dim=-1, keepdim=True)

        fn_norm = self.cfg.normal_contact_stiffness * current_depth
        fn = fn_norm.unsqueeze(-1) * normals_world
        relative_vt_norm = torch.linalg.norm(relative_vt, dim=-1)
        ft_static_norm = self.cfg.tangential_contact_stiffness * relative_vt_norm
        ft_dynamic_norm = self.cfg.friction_coefficient * fn_norm
        ft_norm = torch.minimum(ft_static_norm, ft_dynamic_norm)
        ft = -ft_norm.unsqueeze(-1) * relative_vt / relative_vt_norm.clamp(min=1.0e-9).unsqueeze(-1)

        force_world = fn + ft
        force_tactile = math_utils.quat_apply_inverse(self._data.tactile_points_quat_w, force_world)
        current_normal_force = (force_tactile @ self._tactile_normal_axis_tensor).clamp(min=0.0)
        stronger = current_depth[env_ids] > self._data.penetration_depth[env_ids]
        self._data.penetration_depth[env_ids] = torch.where(
            stronger,
            current_depth[env_ids],
            self._data.penetration_depth[env_ids],
        )
        self._data.tactile_normal_force[env_ids] = torch.where(
            stronger,
            current_normal_force[env_ids],
            self._data.tactile_normal_force[env_ids],
        )

    def _set_debug_vis_impl(self, debug_vis: bool) -> None:
        raise NotImplementedError(f"Debug visualization is not implemented for {self.__class__.__name__}.")

    def _debug_vis_callback(self, event: Any) -> None:
        raise NotImplementedError(f"Debug visualization is not implemented for {self.__class__.__name__}.")

    def get_tactile_points(
        self,
        *,
        normalize: bool = False,
        env_origins: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return tactile points as ``[x, y, z, normal_force]``."""
        tactile_data = self.data
        if tactile_data.tactile_points_pos_w is None or tactile_data.tactile_normal_force is None:
            raise RuntimeError("Tactile sensor data is not initialized.")
        positions = tactile_data.tactile_points_pos_w
        if env_origins is not None:
            positions = positions - env_origins.unsqueeze(1)
        normal_force = tactile_data.tactile_normal_force
        if normalize:
            normal_force = normalize_tactile_normal_force(
                normal_force,
                low_force_threshold=self.cfg.normalization_low_force_threshold,
                low_force_scale=self.cfg.normalization_low_force_scale,
            )
        return torch.cat((positions, normal_force.unsqueeze(-1)), dim=-1)


@configclass
class VtRefineTactileSensorCfg(SensorBaseCfg):
    """Configuration for a VT-Refine-style force-field tactile sensor."""

    class_type: type = VtRefineTactileSensor

    contact_prim_paths_expr: tuple[str, ...] = ()
    tactile_array_size: tuple[int, int] = DEFAULT_TABLE_TACTILE_ARRAY_SIZE
    tactile_point_distance: float = DEFAULT_TABLE_TACTILE_POINT_DISTANCE
    pad_size: tuple[float, float, float] = DEFAULT_TABLE_TACTILE_PAD_SIZE
    tactile_pad_bounds: TactilePadBounds | None = None
    tactile_surface_axis: int = 2
    tactile_surface_sign: float = 1.0
    tactile_grid_axes: tuple[int, int] = (0, 1)
    env_prim_path_expr: str = "{ENV_REGEX_NS}"
    normal_contact_stiffness: float = DEFAULT_TABLE_TACTILE_NORMAL_CONTACT_STIFFNESS
    tangential_contact_stiffness: float = DEFAULT_TABLE_TACTILE_TANGENTIAL_CONTACT_STIFFNESS
    friction_coefficient: float = DEFAULT_TABLE_TACTILE_FRICTION_COEFFICIENT
    tactile_normal_axis: tuple[float, float, float] = DEFAULT_TABLE_TACTILE_NORMAL_AXIS
    normalization_low_force_threshold: float = DEFAULT_TABLE_TACTILE_LOW_FORCE_THRESHOLD
    normalization_low_force_scale: float = DEFAULT_TABLE_TACTILE_LOW_FORCE_SCALE


def make_table_tactile_pad_cfg(
    prim_path: str = TABLE_TACTILE_PAD_PRIM_PATH,
    *,
    pos: tuple[float, float, float] = TABLE_TACTILE_PAD_POS,
    rot: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0),
    size: tuple[float, float, float] = DEFAULT_TABLE_TACTILE_PAD_SIZE,
    compliant_contact_stiffness: float = DEFAULT_TABLE_TACTILE_COMPLIANT_CONTACT_STIFFNESS,
    compliant_contact_damping: float = DEFAULT_TABLE_TACTILE_COMPLIANT_CONTACT_DAMPING,
) -> Any:
    """Create the kinematic table tactile pad asset config."""
    _require_isaac_lab()
    return RigidObjectCfg(
        prim_path=prim_path,
        spawn=sim_utils.CuboidCfg(
            size=size,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=True,
                disable_gravity=True,
                max_depenetration_velocity=5.0,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.002, rest_offset=0.0),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.1),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=1.0,
                dynamic_friction=1.0,
                restitution=0.0,
                compliant_contact_stiffness=compliant_contact_stiffness,
                compliant_contact_damping=compliant_contact_damping,
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=pos, rot=rot),
    )


def make_table_tactile_sensor_cfg(
    contact_prim_paths_expr: Sequence[str],
    prim_path: str = TABLE_TACTILE_SENSOR_PRIM_PATH,
    *,
    tactile_array_size: tuple[int, int] = DEFAULT_TABLE_TACTILE_ARRAY_SIZE,
    tactile_point_distance: float = DEFAULT_TABLE_TACTILE_POINT_DISTANCE,
    pad_size: tuple[float, float, float] = DEFAULT_TABLE_TACTILE_PAD_SIZE,
    normal_contact_stiffness: float = DEFAULT_TABLE_TACTILE_NORMAL_CONTACT_STIFFNESS,
    tangential_contact_stiffness: float = DEFAULT_TABLE_TACTILE_TANGENTIAL_CONTACT_STIFFNESS,
    friction_coefficient: float = DEFAULT_TABLE_TACTILE_FRICTION_COEFFICIENT,
    update_period: float = 0.0,
) -> VtRefineTactileSensorCfg:
    """Create the table-mounted tactile sensor config."""
    _require_isaac_lab()
    return VtRefineTactileSensorCfg(
        prim_path=prim_path,
        update_period=update_period,
        contact_prim_paths_expr=tuple(contact_prim_paths_expr),
        tactile_array_size=tactile_array_size,
        tactile_point_distance=tactile_point_distance,
        pad_size=pad_size,
        tactile_surface_axis=2,
        tactile_surface_sign=1.0,
        tactile_grid_axes=(0, 1),
        normal_contact_stiffness=normal_contact_stiffness,
        tangential_contact_stiffness=tangential_contact_stiffness,
        friction_coefficient=friction_coefficient,
        tactile_normal_axis=DEFAULT_TABLE_TACTILE_NORMAL_AXIS,
    )


def make_r1_pro_gripper_tactile_sensor_cfg(
    contact_prim_paths_expr: Sequence[str],
    pad_spec: R1ProGripperTactilePadSpec,
    *,
    tactile_array_size: tuple[int, int] = DEFAULT_TABLE_TACTILE_ARRAY_SIZE,
    tactile_point_distance: float = DEFAULT_TABLE_TACTILE_POINT_DISTANCE,
    normal_contact_stiffness: float = DEFAULT_TABLE_TACTILE_NORMAL_CONTACT_STIFFNESS,
    tangential_contact_stiffness: float = DEFAULT_TABLE_TACTILE_TANGENTIAL_CONTACT_STIFFNESS,
    friction_coefficient: float = DEFAULT_TABLE_TACTILE_FRICTION_COEFFICIENT,
    update_period: float = 0.0,
) -> VtRefineTactileSensorCfg:
    """Create a tactile sensor config attached to one R1 Pro gripper elastomer pad."""
    _require_isaac_lab()
    return VtRefineTactileSensorCfg(
        prim_path=pad_spec.prim_path_expr,
        update_period=update_period,
        contact_prim_paths_expr=tuple(contact_prim_paths_expr),
        tactile_array_size=tactile_array_size,
        tactile_point_distance=tactile_point_distance,
        pad_size=pad_spec.size,
        tactile_pad_bounds=pad_spec.bounds,
        tactile_surface_axis=1,
        tactile_surface_sign=pad_spec.surface_sign,
        tactile_grid_axes=(0, 2),
        normal_contact_stiffness=normal_contact_stiffness,
        tangential_contact_stiffness=tangential_contact_stiffness,
        friction_coefficient=friction_coefficient,
        tactile_normal_axis=pad_spec.normal_axis,
        normalization_low_force_threshold=R1_PRO_GRIPPER_TACTILE_NORMALIZATION_LOW_FORCE_THRESHOLD,
    )


def _contact_prim_paths_from_scene(cfg: Any, contact_part_names: Sequence[str]) -> tuple[str, ...]:
    return tuple(getattr(cfg.scene, part_name).prim_path for part_name in contact_part_names)


def configure_table_tactile_scene_cfg(cfg: Any) -> tuple[str, ...]:
    """Inject table tactile scene entities if the env cfg enables them."""
    if not getattr(cfg, "enable_table_tactile", False):
        return ()
    contact_part_names = resolve_table_tactile_contact_part_names(
        getattr(cfg, "table_tactile_contact_part_names", ()),
        getattr(cfg, "assembly_reset_part_names", ()),
        getattr(cfg, "assembly_part_names", ()),
    )
    contact_prim_paths = _contact_prim_paths_from_scene(cfg, contact_part_names)
    cfg.scene.table_tactile_pad = make_table_tactile_pad_cfg()
    cfg.scene.table_tactile_sensor = make_table_tactile_sensor_cfg(contact_prim_paths)
    return contact_part_names


def configure_r1_pro_gripper_tactile_scene_cfg(cfg: Any) -> tuple[str, ...]:
    """Inject R1 Pro gripper tactile sensors if the env cfg enables them."""
    enabled = getattr(cfg, "enable_r1_pro_gripper_tactile", False)
    append = getattr(cfg, "append_r1_pro_gripper_tactile_to_policy", False)
    if not enabled and not append:
        return ()

    contact_part_names = resolve_tactile_contact_part_names(
        "R1 Pro gripper tactile",
        getattr(cfg, "r1_pro_gripper_tactile_contact_part_names", ()),
        getattr(cfg, "assembly_reset_part_names", ()),
        getattr(cfg, "assembly_part_names", ()),
    )
    contact_prim_paths = _contact_prim_paths_from_scene(cfg, contact_part_names)

    for pad_spec in R1_PRO_GRIPPER_TACTILE_PAD_SPECS:
        setattr(
            cfg.scene,
            pad_spec.sensor_name,
            make_r1_pro_gripper_tactile_sensor_cfg(contact_prim_paths, pad_spec),
        )

    if append and not getattr(cfg, "_r1_pro_gripper_tactile_observation_space_added", False):
        cfg.observation_space += R1_PRO_GRIPPER_TACTILE_OBSERVATION_SIZE
        cfg._r1_pro_gripper_tactile_observation_space_added = True
    return contact_part_names


def get_r1_pro_gripper_tactile_points(
    scene: Any,
    normalize: bool = True,
    env_origins: torch.Tensor | None = None,
) -> torch.Tensor:
    """Return configured R1 Pro gripper tactile pads as ``[x, y, z, normal_force]``."""
    tactile_points = [
        scene[pad_spec.sensor_name].get_tactile_points(normalize=normalize, env_origins=env_origins)
        for pad_spec in R1_PRO_GRIPPER_TACTILE_PAD_SPECS
    ]
    return torch.cat(tactile_points, dim=1)


__all__ = [
    "DEFAULT_TABLE_TACTILE_ARRAY_SIZE",
    "DEFAULT_TABLE_TACTILE_COMPLIANT_CONTACT_DAMPING",
    "DEFAULT_TABLE_TACTILE_COMPLIANT_CONTACT_STIFFNESS",
    "DEFAULT_TABLE_TACTILE_FRICTION_COEFFICIENT",
    "DEFAULT_TABLE_TACTILE_LOW_FORCE_SCALE",
    "DEFAULT_TABLE_TACTILE_LOW_FORCE_THRESHOLD",
    "DEFAULT_TABLE_TACTILE_NORMAL_AXIS",
    "DEFAULT_TABLE_TACTILE_NORMAL_CONTACT_STIFFNESS",
    "DEFAULT_TABLE_TACTILE_PAD_SIZE",
    "DEFAULT_TABLE_TACTILE_POINT_DISTANCE",
    "DEFAULT_TABLE_TACTILE_TANGENTIAL_CONTACT_STIFFNESS",
    "R1ProGripperTactilePadSpec",
    "R1_PRO_GRIPPER_TACTILE_MATERIAL",
    "R1_PRO_GRIPPER_TACTILE_NORMALIZATION_LOW_FORCE_THRESHOLD",
    "R1_PRO_GRIPPER_TACTILE_OBSERVATION_SIZE",
    "R1_PRO_GRIPPER_TACTILE_PAD_SPECS",
    "R1_PRO_GRIPPER_TACTILE_VISUAL_RGBA",
    "TABLE_TACTILE_PAD_POS",
    "TABLE_TACTILE_PAD_PRIM_PATH",
    "TABLE_TACTILE_SENSOR_PRIM_PATH",
    "TactileContactMaterialSpec",
    "TactilePadBounds",
    "VtRefineTactileSensor",
    "VtRefineTactileSensorCfg",
    "VtRefineTactileSensorData",
    "apply_tactile_compliant_material",
    "configure_r1_pro_gripper_tactile_scene_cfg",
    "configure_table_tactile_scene_cfg",
    "generate_table_tactile_points_local",
    "generate_tactile_points_on_rectangular_pad",
    "generate_tactile_points_on_rectangular_pad_from_bounds",
    "get_r1_pro_gripper_tactile_points",
    "make_r1_pro_gripper_tactile_sensor_cfg",
    "make_table_tactile_pad_cfg",
    "make_table_tactile_sensor_cfg",
    "normalize_tactile_normal_force",
    "read_mesh_vertex_bounds",
    "read_stl_vertex_bounds",
    "resolve_table_tactile_contact_part_names",
    "resolve_tactile_contact_part_names",
    "tactile_force_grid",
    "tactile_observation_size",
    "tactile_point_count",
]
