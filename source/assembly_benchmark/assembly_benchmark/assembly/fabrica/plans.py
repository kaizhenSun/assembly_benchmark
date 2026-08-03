# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Self-contained Fabrica specialist assembly plans."""

from __future__ import annotations

from dataclasses import dataclass

from ..specs import Quat, Vec3
from .beam import BEAM_RELATION_PART_IDS, INSERTION_PATH_LENGTH

JointPosition = tuple[float, ...]
PathPose = tuple[float, float, float, float, float, float, float]

PIPER_FABRICA_GRIPPER_BASE_POS: Vec3 = (0.0, 0.0, -0.0255)
PIPER_FABRICA_GRIPPER_BASE_ROTATION_RPY: Vec3 = (0.0, 0.0, 1.5707963267948966)
PIPER_FABRICA_BASE_POS: Vec3 = (0.31, 0.30, 0.775)

_BEAM_PATH_Z = (
    0.0,
    0.000820000078,
    0.00324000031,
    0.00526000050,
    0.00688000065,
    0.0101000010,
    0.0109200010,
    0.0133400013,
    0.0153600015,
    0.0202000019,
)
BEAM_DISASSEMBLY_PATH: tuple[PathPose, ...] = tuple((0.0, 0.0, z, 1.0, 0.0, 0.0, 0.0) for z in _BEAM_PATH_Z)


@dataclass(frozen=True)
class FabricaRelationPlan:
    """One fixed-plug specialist relation within an assembly."""

    plug_part_id: str
    socket_part_id: str
    disassembly_path: tuple[PathPose, ...]
    piper_gripper_to_plug_pos: Vec3
    piper_gripper_to_plug_quat: Quat
    piper_gripper_opening: float
    piper_preassembly_joint_pos: JointPosition
    source_panda_hand_to_plug_pos: Vec3
    source_panda_hand_to_plug_rpy: Vec3

    @property
    def key(self) -> str:
        """Return the stable relation key used by assets and metrics."""
        return f"{self.plug_part_id}_{self.socket_part_id}"

    @property
    def path_length(self) -> float:
        start = self.disassembly_path[0]
        end = self.disassembly_path[-1]
        return sum((end[index] - start[index]) ** 2 for index in range(3)) ** 0.5


@dataclass(frozen=True)
class FabricaAssemblyPlan:
    """Specialist training data for exactly one Fabrica assembly."""

    assembly_name: str
    relations: tuple[FabricaRelationPlan, ...]

    def __post_init__(self) -> None:
        if not self.relations:
            raise ValueError(f"Fabrica plan '{self.assembly_name}' must contain at least one relation.")
        keys = tuple(relation.key for relation in self.relations)
        if len(keys) != len(set(keys)):
            raise ValueError(f"Fabrica plan '{self.assembly_name}' contains duplicate relations: {keys}.")
        for relation in self.relations:
            if len(relation.disassembly_path) < 2:
                raise ValueError(f"Fabrica relation '{relation.key}' must contain at least two path poses.")
            if len(relation.piper_preassembly_joint_pos) != 6:
                raise ValueError(f"Fabrica relation '{relation.key}' must contain six Piper arm joints.")
            if relation.piper_gripper_opening <= 0.0:
                raise ValueError(f"Fabrica relation '{relation.key}' must use a positive gripper opening.")

    @property
    def relation_keys(self) -> tuple[str, ...]:
        return tuple(relation.key for relation in self.relations)

    def part_scene_key(self, part_id: str) -> str:
        """Return the neutral assembly-spec scene key for a Fabrica part."""
        return f"{self.assembly_name}_part_{part_id}"


def assign_fabrica_relations(num_envs: int, num_relations: int) -> tuple[int, ...]:
    """Assign relations deterministically using the original ``env_id % relation_count`` rule."""
    if num_envs < 0:
        raise ValueError(f"num_envs must be non-negative, got {num_envs}.")
    if num_relations <= 0:
        raise ValueError(f"num_relations must be positive, got {num_relations}.")
    return tuple(env_id % num_relations for env_id in range(num_envs))


BEAM_FABRICA_PLAN = FabricaAssemblyPlan(
    assembly_name="beam",
    relations=(
        FabricaRelationPlan(
            plug_part_id="0",
            socket_part_id="2",
            disassembly_path=BEAM_DISASSEMBLY_PATH,
            piper_gripper_to_plug_pos=(-5.334315211769081e-10, 0.04445007709999993, 0.25219990690000005),
            piper_gripper_to_plug_quat=(0.0, 0.7071067811865476, 0.7071067811865476, 0.0),
            piper_gripper_opening=0.0127,
            piper_preassembly_joint_pos=(
                -4.271296546610338e-09,
                1.2685965310859393,
                -0.8277978161931331,
                -1.34081133119676e-15,
                1.2172640851071919,
                -4.2712957794871606e-09,
            ),
            source_panda_hand_to_plug_pos=(-0.0075073544766095235, 0.04445007741147011, 0.17689691200939883),
            source_panda_hand_to_plug_rpy=(
                3.0644179596970864,
                -2.0854541205039823e-08,
                1.5707959965253213,
            ),
        ),
        FabricaRelationPlan(
            plug_part_id="1",
            socket_part_id="3",
            disassembly_path=BEAM_DISASSEMBLY_PATH,
            piper_gripper_to_plug_pos=(0.01912211586358894, -0.04444991023383182, 0.24726894232464144),
            piper_gripper_to_plug_quat=(
                -0.05765063622328469,
                0.7047527270303521,
                0.7047527155170007,
                -0.05765074030527348,
            ),
            piper_gripper_opening=0.012699999999562,
            piper_preassembly_joint_pos=(
                0.11191567564282157,
                1.5945528117538028,
                -1.1277105201955475,
                -0.1731840908364133,
                1.2145678460070073,
                0.17434303765667045,
            ),
            source_panda_hand_to_plug_pos=(0.032781007149012525, -0.04444992451005167, 0.1686432855376987),
            source_panda_hand_to_plug_rpy=(-2.706459534428822, 1.228743791159559e-07, 1.570796032207707),
        ),
        FabricaRelationPlan(
            plug_part_id="2",
            socket_part_id="6",
            disassembly_path=BEAM_DISASSEMBLY_PATH,
            piper_gripper_to_plug_pos=(-0.05188598874911127, -3.21645931555814e-08, 0.20592751723742492),
            piper_gripper_to_plug_quat=(
                0.084387914348466,
                -3.4149734273e-07,
                0.996432978133401,
                -2.76164643e-09,
            ),
            piper_gripper_opening=0.012272570305757,
            piper_preassembly_joint_pos=(
                3.9858083403439512e-08,
                1.466162813145911,
                -0.8329159849714455,
                -1.0862840531993475e-07,
                1.193792762977204,
                -1.5707968738861926,
            ),
            source_panda_hand_to_plug_pos=(-0.05517548836005579, -1.4864068031972266e-08, 0.14216175090189484),
            source_panda_hand_to_plug_rpy=(-3.141592556120366, 0.24615148504436335, -3.141592286908615),
        ),
        FabricaRelationPlan(
            plug_part_id="3",
            socket_part_id="6",
            disassembly_path=BEAM_DISASSEMBLY_PATH,
            piper_gripper_to_plug_pos=(0.02653035293357381, -0.044451291035876195, 0.21200656004109647),
            piper_gripper_to_plug_quat=(
                -0.164228905092957,
                0.687770715643489,
                0.687771166864897,
                -0.164228899602081,
            ),
            piper_gripper_opening=0.011965549397338,
            piper_preassembly_joint_pos=(
                0.34345033899614524,
                1.7170388431856172,
                -1.0587489223869522,
                -0.4722707579193113,
                1.208759993002123,
                0.5602446158232951,
            ),
            source_panda_hand_to_plug_pos=(0.016314694185361217, -0.04445129952831422, 0.15434261362488108),
            source_panda_hand_to_plug_rpy=(-2.749976999414857, -1.2843730567979605e-07, 1.5707966255624954),
        ),
    ),
)

if tuple((relation.plug_part_id, relation.socket_part_id) for relation in BEAM_FABRICA_PLAN.relations) != (
    BEAM_RELATION_PART_IDS
):
    raise ValueError("Beam Fabrica plan relation order does not match the assembly graph.")
if abs(BEAM_FABRICA_PLAN.relations[0].path_length - INSERTION_PATH_LENGTH) > 1.0e-6:
    raise ValueError("Beam Fabrica path length does not match the assembly insertion path.")

_FABRICA_PLANS = {BEAM_FABRICA_PLAN.assembly_name: BEAM_FABRICA_PLAN}


def available_fabrica_assemblies() -> tuple[str, ...]:
    return tuple(_FABRICA_PLANS)


def load_fabrica_assembly_plan(assembly_name: str) -> FabricaAssemblyPlan:
    """Load one checked-in specialist plan by assembly registry name."""
    try:
        return _FABRICA_PLANS[assembly_name]
    except KeyError as exc:
        available = ", ".join(available_fabrica_assemblies())
        raise KeyError(f"Unknown Fabrica assembly '{assembly_name}'. Available: {available}.") from exc


__all__ = [
    "BEAM_DISASSEMBLY_PATH",
    "BEAM_FABRICA_PLAN",
    "FabricaAssemblyPlan",
    "FabricaRelationPlan",
    "PIPER_FABRICA_BASE_POS",
    "PIPER_FABRICA_GRIPPER_BASE_POS",
    "PIPER_FABRICA_GRIPPER_BASE_ROTATION_RPY",
    "assign_fabrica_relations",
    "available_fabrica_assemblies",
    "load_fabrica_assembly_plan",
]
