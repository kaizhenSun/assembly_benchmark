# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Assembly and part specifications used by assembly tasks."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

Quat = tuple[float, float, float, float]
Vec3 = tuple[float, float, float]
PartBodyType = Literal["visual", "static", "dynamic"]

ASSEMBLY_ASSET_ROOT = Path(__file__).resolve().parents[1] / "assets" / "furniture"
IDENTITY_QUAT: Quat = (1.0, 0.0, 0.0, 0.0)
DEFAULT_TARGET_INDEX = 0
DEFAULT_POS_THRESHOLD: Vec3 = (0.010, 0.005, 0.010)
DEFAULT_ORI_BOUND = 0.94


@dataclass(frozen=True)
class AssemblyPartSpec:
    """Static metadata for one assembly part asset."""

    scene_key: str
    asset_name: str
    prim_name: str
    urdf_rel_path: Path
    init_pos: Vec3
    init_rot: Quat
    body_type: PartBodyType
    mass: float | None = None
    density: float | None = None
    reset: bool = True
    tag_ids: tuple[int, ...] = ()
    reset_footprint_xy: tuple[float, float] | None = None

    def usd_path(self, asset_root: Path) -> Path:
        """Return the checked-in USD path for this part."""
        return asset_root / "usd" / self.asset_name / f"{self.asset_name}.usd"

    def urdf_path(self, asset_root: Path) -> Path:
        """Return the source URDF path for this part."""
        return asset_root / self.urdf_rel_path


def assembly_asset_root(name: str) -> Path:
    """Return the checked-in asset root for an assembly."""
    return ASSEMBLY_ASSET_ROOT / name


def make_part(
    *,
    scene_key: str,
    asset_name: str,
    prim_name: str,
    urdf_rel_path: str | Path,
    init_pos: Vec3,
    init_rot: Quat,
    body_type: PartBodyType,
    mass: float | None = None,
    density: float | None = None,
    reset: bool = True,
    tag_ids: tuple[int, ...] = (),
    reset_footprint_xy: tuple[float, float] | None = None,
) -> AssemblyPartSpec:
    """Create one assembly part spec."""
    return AssemblyPartSpec(
        scene_key=scene_key,
        asset_name=asset_name,
        prim_name=prim_name,
        urdf_rel_path=Path(urdf_rel_path),
        init_pos=init_pos,
        init_rot=init_rot,
        body_type=body_type,
        mass=mass,
        density=density,
        reset=reset,
        tag_ids=tag_ids,
        reset_footprint_xy=reset_footprint_xy,
    )


def make_visual_part(
    *,
    scene_key: str,
    asset_name: str,
    prim_name: str,
    urdf_rel_path: str | Path,
    init_pos: Vec3,
    init_rot: Quat = IDENTITY_QUAT,
    tag_ids: tuple[int, ...] = (),
) -> AssemblyPartSpec:
    """Create a non-reset visual assembly part."""
    return make_part(
        scene_key=scene_key,
        asset_name=asset_name,
        prim_name=prim_name,
        urdf_rel_path=urdf_rel_path,
        init_pos=init_pos,
        init_rot=init_rot,
        body_type="visual",
        reset=False,
        tag_ids=tag_ids,
    )


def make_dynamic_part(
    *,
    scene_key: str,
    asset_name: str,
    prim_name: str,
    urdf_rel_path: str | Path,
    init_pos: Vec3,
    init_rot: Quat,
    mass: float | None = None,
    density: float | None = None,
    tag_ids: tuple[int, ...] = (),
    reset_footprint_xy: tuple[float, float] | None = None,
) -> AssemblyPartSpec:
    """Create a resettable dynamic assembly part."""
    return make_part(
        scene_key=scene_key,
        asset_name=asset_name,
        prim_name=prim_name,
        urdf_rel_path=urdf_rel_path,
        init_pos=init_pos,
        init_rot=init_rot,
        body_type="dynamic",
        mass=mass,
        density=density,
        tag_ids=tag_ids,
        reset_footprint_xy=reset_footprint_xy,
    )


def make_base_tag_part(init_pos: Vec3, init_rot: Quat = IDENTITY_QUAT) -> AssemblyPartSpec:
    """Create the standard base tag visual asset."""
    return make_visual_part(
        scene_key="base_tag",
        asset_name="base_tag",
        prim_name="BaseTag",
        urdf_rel_path="urdf/base_tag.urdf",
        init_pos=init_pos,
        init_rot=init_rot,
        tag_ids=(0, 1, 2, 3),
    )


@dataclass(frozen=True)
class AssemblyTargetPose:
    """Candidate child pose in the parent part frame."""

    pos: Vec3
    quat: Quat = (1.0, 0.0, 0.0, 0.0)


@dataclass(frozen=True)
class AssemblyRelationSpec:
    """Assembly relation between two parts."""

    parent: str
    child: str
    target_poses: tuple[AssemblyTargetPose, ...]
    default_target_index: int = DEFAULT_TARGET_INDEX
    pos_threshold: Vec3 = DEFAULT_POS_THRESHOLD
    ori_bound: float = DEFAULT_ORI_BOUND

    @property
    def default_target_pose(self) -> AssemblyTargetPose:
        """Return the target pose used by scripted demos."""
        return self.target_poses[self.default_target_index]


def make_relation(
    parent: str,
    child: str,
    target_positions: Iterable[Vec3],
    *,
    target_quats: Iterable[Quat] | None = None,
    default_target_index: int = DEFAULT_TARGET_INDEX,
    pos_threshold: Vec3 = DEFAULT_POS_THRESHOLD,
    ori_bound: float = DEFAULT_ORI_BOUND,
) -> AssemblyRelationSpec:
    """Create an assembly relation from readable target pose tuples."""
    if target_quats is None:
        target_poses = tuple(AssemblyTargetPose(pos=pos) for pos in target_positions)
    else:
        target_poses = tuple(
            AssemblyTargetPose(pos=pos, quat=quat) for pos, quat in zip(target_positions, target_quats, strict=True)
        )

    return AssemblyRelationSpec(
        parent=parent,
        child=child,
        target_poses=target_poses,
        default_target_index=default_target_index,
        pos_threshold=pos_threshold,
        ori_bound=ori_bound,
    )


@dataclass(frozen=True)
class UsdGenerationAssetSpec:
    """URDF-to-USD generation metadata for one part asset."""

    asset_name: str
    urdf_path: Path
    body_type: PartBodyType
    mass: float | None
    density: float | None

    @property
    def is_dynamic(self) -> bool:
        """Whether this asset should be generated as a dynamic rigid body."""
        return self.body_type == "dynamic"


@dataclass(frozen=True)
class AssemblySpec:
    """Collection of part specs and assembly relations for one assembly task."""

    name: str
    asset_root: Path
    parts: tuple[AssemblyPartSpec, ...]
    assembly_relations: tuple[AssemblyRelationSpec, ...]

    def __post_init__(self) -> None:
        """Validate the assembly contract used by Isaac Lab scene generation."""
        part_names = [part.scene_key for part in self.parts]
        duplicate_part_names = sorted({name for name in part_names if part_names.count(name) > 1})
        if duplicate_part_names:
            duplicates = ", ".join(duplicate_part_names)
            raise ValueError(f"Assembly '{self.name}' has duplicate part scene keys: {duplicates}.")

        invalid_scene_keys = [name for name in part_names if not name.isidentifier()]
        if invalid_scene_keys:
            invalid = ", ".join(invalid_scene_keys)
            raise ValueError(
                f"Assembly '{self.name}' has part scene keys that are not valid Python identifiers: {invalid}."
            )

        for part in self.parts:
            if part.body_type == "dynamic":
                if (part.mass is None) == (part.density is None):
                    raise ValueError(
                        f"Assembly '{self.name}' dynamic part '{part.scene_key}' must declare exactly one "
                        "of mass or density."
                    )
            elif part.mass is not None or part.density is not None:
                raise ValueError(
                    f"Assembly '{self.name}' non-dynamic part '{part.scene_key}' must not declare mass or density."
                )

        part_name_set = set(part_names)
        for relation in self.assembly_relations:
            if relation.parent not in part_name_set:
                raise ValueError(f"Assembly '{self.name}' relation parent '{relation.parent}' is not a declared part.")
            if relation.child not in part_name_set:
                raise ValueError(f"Assembly '{self.name}' relation child '{relation.child}' is not a declared part.")
            if not relation.target_poses:
                raise ValueError(
                    f"Assembly '{self.name}' relation '{relation.parent}->{relation.child}' has no target poses."
                )
            if not 0 <= relation.default_target_index < len(relation.target_poses):
                raise ValueError(
                    f"Assembly '{self.name}' relation '{relation.parent}->{relation.child}' has default target "
                    f"index {relation.default_target_index}, but only {len(relation.target_poses)} targets."
                )

    @property
    def part_names(self) -> tuple[str, ...]:
        """Return all part scene keys in declaration order."""
        return tuple(part.scene_key for part in self.parts)

    @property
    def reset_part_names(self) -> tuple[str, ...]:
        """Return rigid parts reset through the runtime environment."""
        return tuple(part.scene_key for part in self.parts if part.reset)

    @property
    def primary_relation(self) -> AssemblyRelationSpec:
        """Return the relation used for sparse success in the current task."""
        if not self.assembly_relations:
            raise ValueError(f"Assembly '{self.name}' has no assembly relations.")
        return self.assembly_relations[0]

    def part(self, scene_key: str) -> AssemblyPartSpec:
        """Look up a part by its scene key."""
        for part in self.parts:
            if part.scene_key == scene_key:
                return part
        raise KeyError(f"Assembly '{self.name}' has no part named '{scene_key}'.")

    def usd_generation_assets(self) -> tuple[UsdGenerationAssetSpec, ...]:
        """Return unique assets needed by the offline USD generation tool."""
        assets: list[UsdGenerationAssetSpec] = []
        seen: set[str] = set()
        for part in self.parts:
            if part.asset_name in seen:
                continue
            seen.add(part.asset_name)
            assets.append(
                UsdGenerationAssetSpec(
                    asset_name=part.asset_name,
                    urdf_path=part.urdf_path(self.asset_root),
                    body_type=part.body_type,
                    mass=part.mass,
                    density=part.density,
                )
            )
        return tuple(assets)
