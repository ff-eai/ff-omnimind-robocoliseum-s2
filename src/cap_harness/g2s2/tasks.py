# Copyright (c) 2026 FF EAI Robotics Ecosystem Inc. All rights reserved. Review use only; see LICENSE-FF.txt.

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from functools import lru_cache
import json
from types import MappingProxyType
from typing import Any

from cap_harness.g2s2.embodiment import DATA_ROOT, Embodiment, embodiment_for, embodiment_for_cfg

BOARDS = ("finemanip", "wbc", "crsemb", "dexmanip")
BOARD_TASKS = MappingProxyType({"finemanip": 8, "wbc": 8, "crsemb": 24, "dexmanip": 10})
RUBIK_MAX_ROWS = 256


@dataclass(frozen=True, slots=True)
class StageSpec:
    index: int
    checker: str
    step_out: int | None
    B: int | None
    process_points: float
    group: int | None


@dataclass(frozen=True, slots=True)
class Phase:
    index: int
    stages: tuple[StageSpec, ...]

    @property
    def step_out(self) -> int:
        values = [stage.step_out for stage in self.stages if stage.step_out]
        return max(values) if values else 0

    @property
    def B(self) -> int:
        values = [stage.B for stage in self.stages if stage.B]
        return min(values) if values else 0

    @property
    def checkers(self) -> tuple[str, ...]:
        return tuple(stage.checker for stage in self.stages)


@dataclass(frozen=True, slots=True)
class TaskSpec:
    name: str
    board: str
    runtime: str
    robot_cfg: str
    embodiment_key: str
    wire_task_name: str
    num_instances: int
    stages: tuple[StageSpec, ...]
    phases: tuple[Phase, ...]
    episode_step_limit: int
    episode_budget: tuple[int, int] | None

    row_dt_s: float
    max_rows_per_reply: int | None

    @property
    def wbc(self) -> bool:
        return self.board == "wbc"

    @property
    def embodiment(self) -> Embodiment:
        return embodiment_for_cfg(self.robot_cfg)

    @property
    def native(self) -> bool:
        return self.episode_budget is not None

    @property
    def season_weight(self) -> float:
        return 1.0 / (4.0 * BOARD_TASKS[self.board])

    def summary(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "board": self.board,
            "runtime": self.runtime,
            "embodiment": self.embodiment_key,
            "wire_task_name": self.wire_task_name,
            "row_dt_s": self.row_dt_s,
            "episode_step_limit": self.episode_step_limit,
            "episode_budget": self.episode_budget,
            "phases": [
                {
                    "index": phase.index,
                    "checkers": list(phase.checkers),
                    "step_out": phase.step_out,
                    "B": phase.B,
                    "points": [stage.process_points for stage in phase.stages],
                }
                for phase in self.phases
            ],
        }


def _row_dt(name: str, runtime: str) -> float:
    if runtime == "mujoco":
        return 1.0 / 15.0
    if runtime == "libuipc":
        return 1.0 / 30.0
    if runtime == "newton":
        return 0.02
    if "pour_50ml" in name or "write_uppercase_letter" in name:
        return 1.0 / 30.0
    return 1.0 / 60.0


def _build(entry: Mapping[str, Any]) -> TaskSpec:
    variants = entry["variants"]
    if len(variants) != 1:
        raise ValueError(f"{entry['task']}: expected one stage layout, found {len(variants)}")
    variant = variants[0]
    runtime = {"mujoco": "rubik"}.get(entry["runtime"], entry["runtime"])
    stages = tuple(
        StageSpec(
            index=int(stage["step"]),
            checker=str(stage["checker"]),
            step_out=None if stage.get("step_out") is None else int(stage["step_out"]),
            B=None if stage.get("B") is None else int(stage["B"]),
            process_points=float(stage.get("process_points") or 0.0),
            group=stage.get("concurrent_group"),
        )
        for stage in variant["stages"]
    )
    phases: list[Phase] = []
    for stage in stages:
        if phases and stage.group is not None and phases[-1].stages[-1].group == stage.group:
            phases[-1] = Phase(phases[-1].index, phases[-1].stages + (stage,))
        else:
            phases.append(Phase(len(phases), (stage,)))
    episode_budget = None
    if variant.get("episode_budget"):
        episode_budget = (int(variant["step_out"]), int(variant["B"]))
        limit = int(variant["step_out"])
    else:
        limit = int(variant["episode_step_limit"])
    embodiment = embodiment_for_cfg(entry["robot_cfg"])
    return TaskSpec(
        name=str(entry["task"]),
        board=str(entry["board"]),
        runtime=runtime,
        robot_cfg=str(entry["robot_cfg"]),
        embodiment_key=embodiment.key,
        wire_task_name=str(entry["sub_task"]),
        num_instances=int(entry.get("num_instances") or 0),
        stages=stages,
        phases=tuple(phases),
        episode_step_limit=limit,
        episode_budget=episode_budget,
        row_dt_s=_row_dt(str(entry["task"]), runtime),
        max_rows_per_reply=RUBIK_MAX_ROWS if runtime == "rubik" else None,
    )


@lru_cache(maxsize=1)
def season_tasks() -> Mapping[str, TaskSpec]:
    entries = json.loads((DATA_ROOT / "budget_table.json").read_text(encoding="utf-8"))
    return MappingProxyType({task.name: task for task in (_build(entry) for entry in entries)})


def resolve_task(robot_type: str, ee_type: str, wire_task_name: str) -> TaskSpec | None:
    try:
        embodiment = embodiment_for(robot_type, ee_type)
    except KeyError:
        return None
    wire = str(wire_task_name).strip()
    for task in season_tasks().values():
        if task.embodiment_key == embodiment.key and task.wire_task_name == wire:
            return task
    for task in season_tasks().values():
        if task.embodiment_key == embodiment.key and wire and task.wire_task_name.split("_2turn")[0] in wire:
            return task
    return None


def task_by_name(name: str) -> TaskSpec:
    return season_tasks()[name]


__all__ = [
    "BOARDS",
    "BOARD_TASKS",
    "RUBIK_MAX_ROWS",
    "Phase",
    "StageSpec",
    "TaskSpec",
    "resolve_task",
    "season_tasks",
    "task_by_name",
]
