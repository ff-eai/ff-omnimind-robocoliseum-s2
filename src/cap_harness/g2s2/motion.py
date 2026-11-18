# Copyright (c) 2026 FF EAI Robotics Ecosystem Inc. All rights reserved. Review use only; see LICENSE-FF.txt.

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import math

import numpy as np

from cap_harness.g2s2.kinematics import S2Kinematics, angle_between, rotation_error


class MotionError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class Limits:
    joint_velocity: float = 1.6

    joint_acceleration: float = 6.0
    linear_velocity: float = 0.3

    angular_velocity: float = 1.2

    min_rows: int = 2


def quintic(s: np.ndarray) -> np.ndarray:
    s = np.clip(s, 0.0, 1.0)
    return s**3 * (10.0 - 15.0 * s + 6.0 * s * s)


def joint_move_rows(delta: np.ndarray, dt: float, limits: Limits) -> int:
    span = float(np.max(np.abs(delta))) if np.size(delta) else 0.0
    if span < 1e-9:
        return limits.min_rows
    duration = max(1.875 * span / limits.joint_velocity, math.sqrt(5.7735 * span / limits.joint_acceleration))
    return max(limits.min_rows, int(math.ceil(duration / dt)))


def interpolate_joints(q0: Sequence[float], q1: Sequence[float], rows: int) -> np.ndarray:
    a = np.asarray(q0, dtype=np.float64)
    b = np.asarray(q1, dtype=np.float64)
    s = quintic(np.arange(1, rows + 1) / float(rows))
    return a[None, :] + (b - a)[None, :] * s[:, None]


def ramp(v0: Sequence[float], v1: Sequence[float], rows: int) -> np.ndarray:
    a = np.asarray(v0, dtype=np.float64)
    b = np.asarray(v1, dtype=np.float64)
    s = np.arange(1, rows + 1) / float(rows)
    return a[None, :] + (b - a)[None, :] * s[:, None]


def slerp_matrix(r0: np.ndarray, r1: np.ndarray, s: float) -> np.ndarray:
    vector = rotation_error(r0, r1)
    angle = float(np.linalg.norm(vector))
    if angle < 1e-12:
        return r0.copy()
    axis = vector / angle
    k = np.array([[0.0, -axis[2], axis[1]], [axis[2], 0.0, -axis[0]], [-axis[1], axis[0], 0.0]])
    theta = angle * s
    rot = np.eye(3) + math.sin(theta) * k + (1.0 - math.cos(theta)) * (k @ k)
    return rot @ r0


def cartesian_line(
    kin: S2Kinematics,
    side: str,
    start: np.ndarray,
    goal: np.ndarray,
    seed: Sequence[float],
    positions: Mapping[str, float],
    *,
    dt: float,
    limits: Limits,
    frame: str = "grasp",
    max_joint_step: float = 0.08,
    free_axis: Sequence[float] | None = None,
    position_tolerance_m: float = 1e-3,
    rotation_tolerance_rad: float = math.radians(0.5),
) -> np.ndarray:
    distance = float(np.linalg.norm(goal[:3, 3] - start[:3, 3]))
    angle = angle_between(start, goal)
    duration = max(
        1.875 * distance / limits.linear_velocity,
        1.875 * angle / limits.angular_velocity,
        limits.min_rows * dt,
    )
    rows = max(limits.min_rows, int(math.ceil(duration / dt)))
    s_values = quintic(np.arange(1, rows + 1) / float(rows))
    q = np.asarray(seed, dtype=np.float64)
    out = np.zeros((rows, q.shape[0]))
    for index, s in enumerate(s_values):
        target = np.eye(4)
        target[:3, 3] = start[:3, 3] + (goal[:3, 3] - start[:3, 3]) * s
        target[:3, :3] = slerp_matrix(start[:3, :3], goal[:3, :3], float(s))
        result = kin.solve(
            side, target, q, positions, frame=frame, restarts=0, max_iterations=100, free_axis=free_axis,
            position_tolerance_m=position_tolerance_m, rotation_tolerance_rad=rotation_tolerance_rad,
        )
        if not result.ok:
            raise MotionError(f"line waypoint {index}/{rows}: {result.message}")
        jump = float(np.max(np.abs(result.joints - q)))
        if jump > max_joint_step:
            raise MotionError(f"line waypoint {index}/{rows}: joint jump {jump:.3f} rad")
        q = result.joints
        out[index] = q
    previous = np.vstack([np.asarray(seed, dtype=np.float64)[None, :], out[:-1]])
    peak = float(np.max(np.abs(out - previous))) / dt
    if peak > limits.joint_velocity * 1.5:
        factor = peak / (limits.joint_velocity * 1.5)
        stretched = max(rows + 1, int(math.ceil(rows * factor)))
        index = np.linspace(0.0, rows - 1, stretched)
        lower = np.floor(index).astype(int)
        upper = np.minimum(lower + 1, rows - 1)
        frac = (index - lower)[:, None]
        out = out[lower] * (1.0 - frac) + out[upper] * frac
    return out


__all__ = [
    "Limits",
    "MotionError",
    "cartesian_line",
    "interpolate_joints",
    "joint_move_rows",
    "quintic",
    "ramp",
    "slerp_matrix",
]
