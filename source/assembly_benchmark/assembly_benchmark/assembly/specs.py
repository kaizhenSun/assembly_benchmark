# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Assembly and part specifications used by assembly tasks."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal


Quat = tuple[float, float, float, float]
Vec3 = tuple[float, float, float]
PartBodyType = Literal["visual", "static", "dynamic"]


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
    observe: bool = False
    reset: bool = True
    tag_ids: tuple[int, ...] = ()
    reset_footprint_xy: tuple[float, float] | None = None

    def usd_path(self, asset_root: Path) -> Path:
        """Return the checked-in USD path for this part."""
        return asset_root / "usd" / self.asset_name / f"{self.asset_name}.usd"

    def urdf_path(self, asset_root: Path) -> Path:
        """Return the source URDF path for this part."""
        return asset_root / self.urdf_rel_path


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
    default_target_index: int = 0
    pos_threshold: Vec3 = (0.010, 0.005, 0.010)
    ori_bound: float = 0.94

    @property
    def default_target_pose(self) -> AssemblyTargetPose:
        """Return the target pose used by scripted demos."""
        return self.target_poses[self.default_target_index]


@dataclass(frozen=True)
class UsdGenerationAssetSpec:
    """URDF-to-USD generation metadata for one part asset."""

    asset_name: str
    urdf_path: Path
    body_type: PartBodyType
    mass: float | None

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

        part_name_set = set(part_names)
        for relation in self.assembly_relations:
            if relation.parent not in part_name_set:
                raise ValueError(
                    f"Assembly '{self.name}' relation parent '{relation.parent}' is not a declared part."
                )
            if relation.child not in part_name_set:
                raise ValueError(
                    f"Assembly '{self.name}' relation child '{relation.child}' is not a declared part."
                )
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
    def observation_part_names(self) -> tuple[str, ...]:
        """Return dynamic parts included in policy observations."""
        return tuple(part.scene_key for part in self.parts if part.observe)

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
                )
            )
        return tuple(assets)
