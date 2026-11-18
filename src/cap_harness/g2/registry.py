# Copyright (c) 2026 FF EAI Robotics Ecosystem Inc. All rights reserved. Review use only; see LICENSE-FF.txt.

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

DEFAULT_HORIZON = 450
FAMILY = "g2_geniesim"

BOARD_ALIASES: dict[str, str] = {
    "instruction": "instruction",
    "if": "instruction",
    "spatial": "spatial",
    "manipulation": "manipulation",
    "manip": "manipulation",
}
BOARDS = ("instruction", "spatial", "manipulation")


TASK_NAME_ALIASES: dict[str, str] = {"hold_pot": "bimanual_hold_pot"}


class G2RegistryError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class G2TaskMetadata:
    task_name: str
    board: str
    task_id: int
    description: str
    horizon: int = DEFAULT_HORIZON
    family: str = FAMILY

    @property
    def suite_name(self) -> str:
        return self.board

    @property
    def language(self) -> str:
        return self.description

    @property
    def task_ref(self) -> str:
        return f"{self.board}:{self.task_id}"

    def to_manifest_record(self) -> dict[str, Any]:
        return {
            "board": self.board,
            "description": self.description,
            "family": self.family,
            "horizon": self.horizon,
            "suite_name": self.suite_name,
            "task_id": self.task_id,
            "task_name": self.task_name,
            "task_ref": self.task_ref,
        }


KNOWN_HORIZONS: dict[str, int] = {
    "open_door": 510,
    "bimanual_hold_pot": 1890,
    "pour_workpiece": 960,
    "stock_and_straighten_shelf": 2280,
    "take_wrong_item_shelf": 780,
    "scoop_popcorn": 960,
    "clean_the_desktop": 2880,
    "place_block_into_box": 870,
    "sorting_packages": 3780,
    "sorting_packages_continuous": 6060,
    "pick_object_absolute_position": 450,
    "pick_object_relative_position": 450,
    "place_beverage_to_anothers_position": 990,
    "place_object_relative_position": 990,
    "sort_cubes_by_size": 1260,
    "sort_number": 1260,
    "stack_bowls": 1560,
    "stack_three_building_blocks": 1560,
}

_TASK_TABLE: tuple[tuple[str, str, str], ...] = (
    ("instruction", "pick_billiards_color", "Pick the billiard ball of the named colour."),
    ("instruction", "pick_block_color", "Pick the block of the named colour."),
    ("instruction", "pick_block_number", "Pick the block carrying the named number."),
    ("instruction", "pick_block_shape", "Pick the block of the named shape."),
    ("instruction", "pick_block_size", "Pick the block of the named size."),
    ("instruction", "pick_common_sense", "Pick the object a common-sense cue identifies."),
    ("instruction", "pick_follow_logic_or", "Pick an object satisfying an OR condition."),
    ("instruction", "pick_object_type", "Pick the object of the named type."),
    ("instruction", "pick_specific_object", "Pick the specifically named object."),
    ("instruction", "straighten_object", "Straighten the named object on the table."),
    ("spatial", "pick_object_absolute_position", "Pick the object at an absolute position."),
    ("spatial", "pick_object_relative_position", "Pick the object at a relative position."),
    ("spatial", "place_beverage_to_anothers_position", "Place a beverage where another one was."),
    ("spatial", "place_object_relative_position", "Place an object relative to another."),
    ("spatial", "sort_cubes_by_size", "Sort the cubes by size."),
    ("spatial", "sort_number", "Sort the numbered objects in order."),
    ("spatial", "stack_bowls", "Stack the bowls."),
    ("spatial", "stack_three_building_blocks", "Stack three building blocks."),
    ("manipulation", "open_door", "Open the door."),
    ("manipulation", "bimanual_hold_pot", "Hold the pot with both hands and move it."),
    ("manipulation", "pour_workpiece", "Pour the workpiece out of its container."),
    ("manipulation", "stock_and_straighten_shelf", "Stock the shelf and straighten the items."),
    ("manipulation", "take_wrong_item_shelf", "Take the wrong item off the shelf."),
    ("manipulation", "scoop_popcorn", "Scoop popcorn into the container."),
    ("manipulation", "clean_the_desktop", "Clean the desktop."),
    ("manipulation", "place_block_into_box", "Place the block into the box."),
    ("manipulation", "sorting_packages", "Sort the packages."),
    ("manipulation", "sorting_packages_continuous", "Sort packages continuously as they arrive."),
)


def _build_tasks() -> tuple[G2TaskMetadata, ...]:
    tasks: list[G2TaskMetadata] = []
    counters = dict.fromkeys(BOARDS, 0)
    for board, name, description in _TASK_TABLE:
        tasks.append(
            G2TaskMetadata(
                task_name=name,
                board=board,
                task_id=counters[board],
                description=description,
                horizon=KNOWN_HORIZONS.get(name, DEFAULT_HORIZON),
            )
        )
        counters[board] += 1
    return tuple(tasks)


G2_GENIESIM_TASKS: tuple[G2TaskMetadata, ...] = _build_tasks()
TASKS_BY_BOARD: MappingProxyType[str, tuple[G2TaskMetadata, ...]] = MappingProxyType(
    {board: tuple(task for task in G2_GENIESIM_TASKS if task.board == board) for board in BOARDS}
)


class G2TaskRegistry:
    def __init__(self, tasks: tuple[G2TaskMetadata, ...] = G2_GENIESIM_TASKS) -> None:
        self._by_name = {task.task_name: task for task in tasks}
        if len(self._by_name) != len(tasks):
            raise G2RegistryError("G2 task names must be unique")
        self._by_ref = {(task.board, task.task_id): task for task in tasks}

    @property
    def available_tasks(self) -> tuple[str, ...]:
        return tuple(self._by_name)

    @property
    def boards(self) -> tuple[str, ...]:
        return tuple(sorted({task.board for task in self._by_name.values()}))

    def enumerate_tasks(self, board: str | None = None) -> tuple[G2TaskMetadata, ...]:
        if board is None:
            return tuple(self._by_name.values())
        normalized = self._board(board)
        return tuple(task for task in self._by_name.values() if task.board == normalized)

    @staticmethod
    def _board(name: object) -> str:
        key = str(name).strip().lower()
        try:
            return BOARD_ALIASES[key]
        except KeyError as exc:
            raise G2RegistryError(
                f"unknown G2 benchmark board {name!r}; expected one of {', '.join(BOARDS)}"
            ) from exc

    def resolve(
        self,
        task_ref: str | tuple[str, int] | G2TaskMetadata,
        task_id: int | None = None,
    ) -> G2TaskMetadata:
        if isinstance(task_ref, G2TaskMetadata):
            return task_ref
        if isinstance(task_ref, tuple):
            if len(task_ref) != 2:
                raise G2RegistryError("task tuple must contain (board_or_name, task_id)")
            name, tuple_id = task_ref
            if task_id is not None and int(task_id) != int(tuple_id):
                raise G2RegistryError("conflicting G2 task ids")
            task_id = int(tuple_id)
        else:
            name = str(task_ref)
            if ":" in name:
                name, encoded = name.rsplit(":", 1)
                if task_id is not None and int(task_id) != int(encoded):
                    raise G2RegistryError("conflicting G2 task ids")
                task_id = int(encoded)
        name = str(name).strip()
        name = TASK_NAME_ALIASES.get(name, name)
        if name in self._by_name:
            task = self._by_name[name]
            if task_id is not None and int(task_id) != task.task_id:
                raise G2RegistryError(
                    f"task {name!r} has id {task.task_id} on the {task.board} board, not {task_id}"
                )
            return task
        board = self._board(name)
        if task_id is None:
            raise G2RegistryError(f"a task id is required to select a task on the {board} board")
        try:
            return self._by_ref[(board, int(task_id))]
        except KeyError as exc:
            count = len(TASKS_BY_BOARD[board])
            raise G2RegistryError(
                f"the {board} board has task ids 0..{count - 1}, not {task_id}"
            ) from exc

    def by_task_name(self, task_name: str) -> G2TaskMetadata | None:
        name = str(task_name).strip()
        return self._by_name.get(name) or self._by_name.get(TASK_NAME_ALIASES.get(name, ""))


__all__ = [
    "BOARDS",
    "BOARD_ALIASES",
    "DEFAULT_HORIZON",
    "FAMILY",
    "G2_GENIESIM_TASKS",
    "KNOWN_HORIZONS",
    "TASKS_BY_BOARD",
    "TASK_NAME_ALIASES",
    "G2RegistryError",
    "G2TaskMetadata",
    "G2TaskRegistry",
]
