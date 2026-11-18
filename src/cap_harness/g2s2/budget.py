# Copyright (c) 2026 FF EAI Robotics Ecosystem Inc. All rights reserved. Review use only; see LICENSE-FF.txt.

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from cap_harness.g2s2.tasks import TaskSpec

END_HOLD_MARGIN_ROWS = 60


def stage_factor(n: int | None, b: int) -> float | None:
    if n is None:
        return None
    b = max(1, int(b))
    return max(0.0, min(float(b), 3.0 * b - 2.0 * int(n))) / float(b)


@dataclass(slots=True)
class DispatchRecord:
    index: int
    phase: int
    rows: int
    label: str
    kind: str
    wall_s: float
    reply_bytes: int = 0


@dataclass(slots=True)
class BudgetTracker:
    task: TaskSpec
    phase: int = 0
    dispatches: list[DispatchRecord] = field(default_factory=list)
    phase_dispatches: dict[int, int] = field(default_factory=dict)
    phase_rows: dict[int, int] = field(default_factory=dict)
    total_rows: int = 0
    uncounted_replies: int = 0
    warnings: list[str] = field(default_factory=list)

    @property
    def native(self) -> bool:
        return self.task.native

    @property
    def phase_count(self) -> int:
        return 1 if self.native else len(self.task.phases)

    def advance(self, to_phase: int | None = None) -> int:
        target = self.phase + 1 if to_phase is None else int(to_phase)
        target = max(self.phase, min(target, self.phase_count - 1))
        self.phase = target
        return self.phase

    def baseline(self, phase: int | None = None) -> int:
        if self.native:
            return int(self.task.episode_budget[1])
        return int(self.task.phases[self.phase if phase is None else phase].B)

    def step_out(self, phase: int | None = None) -> int:
        if self.native:
            return int(self.task.episode_budget[0])
        return int(self.task.phases[self.phase if phase is None else phase].step_out)

    def used(self, phase: int | None = None) -> int:
        key = 0 if self.native else (self.phase if phase is None else phase)
        return int(self.phase_dispatches.get(key, 0))

    def rows_used(self, phase: int | None = None) -> int:
        key = 0 if self.native else (self.phase if phase is None else phase)
        return int(self.phase_rows.get(key, 0))

    def dispatches_left(self, phase: int | None = None) -> int:
        return self.baseline(phase) - self.used(phase)

    def rows_left(self, phase: int | None = None) -> int:
        return self.step_out(phase) - self.rows_used(phase)

    def end_hold_rows(self) -> int:
        return max(1, self.task.episode_step_limit - self.total_rows) + END_HOLD_MARGIN_ROWS

    def charge(self, rows: int, *, label: str = "", kind: str = "motion", wall_s: float = 0.0, reply_bytes: int = 0) -> DispatchRecord:
        key = 0 if self.native else self.phase
        record = DispatchRecord(
            index=len(self.dispatches), phase=key, rows=int(rows), label=label, kind=kind,
            wall_s=round(float(wall_s), 3), reply_bytes=int(reply_bytes),
        )
        self.dispatches.append(record)
        self.phase_dispatches[key] = self.phase_dispatches.get(key, 0) + 1
        self.phase_rows[key] = self.phase_rows.get(key, 0) + int(rows)
        self.total_rows += int(rows)
        used, baseline = self.phase_dispatches[key], self.baseline(key)
        if used > baseline:
            self.warnings.append(f"dispatch {record.index} ({label or kind}): phase {key} at {used} > B {baseline}")
        if self.phase_rows[key] > self.step_out(key):
            self.warnings.append(
                f"dispatch {record.index} ({label or kind}): phase {key} rows {self.phase_rows[key]} > StepOut {self.step_out(key)}"
            )
        return record

    def note_uncounted(self) -> None:
        self.uncounted_replies += 1

    def status(self) -> dict[str, Any]:
        phases = []
        for index in range(self.phase_count):
            n = self.phase_dispatches.get(index)
            b = self.baseline(index)
            phases.append(
                {
                    "phase": index,
                    "checkers": ["episode"] if self.native else list(self.task.phases[index].checkers),
                    "N": n,
                    "B": b,
                    "factor_if_scored": stage_factor(n, b),
                    "rows": self.phase_rows.get(index, 0),
                    "step_out": self.step_out(index),
                }
            )
        return {
            "task": self.task.name,
            "believed_phase": self.phase,
            "dispatches": len(self.dispatches),
            "uncounted_replies": self.uncounted_replies,
            "total_rows": self.total_rows,
            "episode_step_limit": self.task.episode_step_limit,
            "phases": phases,
            "warnings": list(self.warnings),
        }


__all__ = ["END_HOLD_MARGIN_ROWS", "BudgetTracker", "DispatchRecord", "stage_factor"]
