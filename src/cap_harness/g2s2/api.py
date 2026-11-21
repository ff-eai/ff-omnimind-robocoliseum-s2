# Copyright (c) 2026 FF EAI Robotics Ecosystem Inc. All rights reserved. Review use only; see LICENSE-FF.txt.

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
import math
import time
from typing import Any

import numpy as np

from cap_harness.g2s2.cameras import CameraView
from cap_harness.g2s2.embodiment import ARMS, WUJI_FINGERS, tip_beyond_grasp_90d
from cap_harness.g2s2.episode import Episode, EpisodeEnded
from cap_harness.g2s2.motion import (
    Limits,
    MotionError,
    cartesian_line,
    interpolate_joints,
    joint_move_rows,
    ramp,
)
from cap_harness.g2s2.perception import Detection, Sam3Client, detect, dominant_plane_z, mask_points
from cap_harness.g2s2.wire import Chunk, matrix_to_xyzw


def rotx(angle: float) -> np.ndarray:
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


def roty(angle: float) -> np.ndarray:
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def rotz(angle: float) -> np.ndarray:
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def pose(position: Sequence[float], rotation: Any = None, *, quat_wxyz: Sequence[float] | None = None, rpy: Sequence[float] | None = None) -> np.ndarray:
    out = np.eye(4)
    out[:3, 3] = np.asarray(position, dtype=np.float64).reshape(3)
    if rotation is not None:
        out[:3, :3] = np.asarray(rotation, dtype=np.float64).reshape(3, 3)
    elif quat_wxyz is not None:
        w, x, y, z = (float(v) for v in quat_wxyz)
        n = math.sqrt(w * w + x * x + y * y + z * z)
        w, x, y, z = w / n, x / n, y / n, z / n
        out[:3, :3] = [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    elif rpy is not None:
        r, p, y = (float(v) for v in rpy)
        out[:3, :3] = rotz(y) @ roty(p) @ rotx(r)
    return out


def top_down(position: Sequence[float], yaw: float = 0.0, *, tilt: float = 0.0, tilt_toward: float = 0.0) -> np.ndarray:
    y_axis = np.array([math.cos(yaw), math.sin(yaw), 0.0])
    z_axis = np.array([0.0, 0.0, -1.0])
    x_axis = np.cross(y_axis, z_axis)
    rotation = np.column_stack([x_axis, y_axis, z_axis])
    if abs(tilt) > 1e-9:
        axis = np.array([-math.sin(tilt_toward), math.cos(tilt_toward), 0.0])
        k = np.array([[0.0, -axis[2], axis[1]], [axis[2], 0.0, -axis[0]], [-axis[1], axis[0], 0.0]])
        tilt_rot = np.eye(3) + math.sin(tilt) * k + (1.0 - math.cos(tilt)) * (k @ k)
        rotation = tilt_rot @ rotation
    return pose(position, rotation)


def offset(base_pose: np.ndarray, dx: float = 0.0, dy: float = 0.0, dz: float = 0.0, *, frame: str = "base") -> np.ndarray:
    out = np.asarray(base_pose, dtype=np.float64).copy()
    delta = np.array([dx, dy, dz], dtype=np.float64)
    out[:3, 3] += delta if frame == "base" else out[:3, :3] @ delta
    return out


def _align_free_axis(start: np.ndarray, goal: np.ndarray, axis: Sequence[float]) -> np.ndarray:
    a = np.asarray(axis, dtype=np.float64) / np.linalg.norm(axis)
    have = start[:3, :3] @ a
    want = goal[:3, :3] @ a
    cross = np.cross(have, want)
    sine, cosine = float(np.linalg.norm(cross)), float(np.dot(have, want))
    if sine < 1e-9:
        turn = np.eye(3) if cosine > 0 else -np.eye(3) + 2 * np.outer(np.cross(have, [1, 0, 0]) / max(np.linalg.norm(np.cross(have, [1, 0, 0])), 1e-9), np.cross(have, [1, 0, 0]) / max(np.linalg.norm(np.cross(have, [1, 0, 0])), 1e-9))
    else:
        k = cross / sine
        kx = np.array([[0.0, -k[2], k[1]], [k[2], 0.0, -k[0]], [-k[1], k[0], 0.0]])
        angle = math.atan2(sine, cosine)
        turn = np.eye(3) + math.sin(angle) * kx + (1.0 - math.cos(angle)) * (kx @ kx)
    out = goal.copy()
    out[:3, :3] = turn @ start[:3, :3]
    return out


def _finger_index(finger: str, joint: int) -> int:
    return WUJI_FINGERS.index(finger) * 4 + (joint - 1)


def wuji_open() -> np.ndarray:
    q = np.zeros(20)
    q[_finger_index("thumb", 1)] = 0.06
    return q


def wuji_power(closure: float = 1.0) -> np.ndarray:
    c = min(1.0, max(0.0, float(closure)))
    q = wuji_open()
    for finger in ("index", "middle", "ring", "pinky"):
        q[_finger_index(finger, 1)] = 1.25 * c
        q[_finger_index(finger, 3)] = 1.20 * c
        q[_finger_index(finger, 4)] = 0.90 * c
    q[_finger_index("thumb", 1)] = 0.06 + 1.10 * c
    q[_finger_index("thumb", 2)] = 0.55 * c
    q[_finger_index("thumb", 3)] = 0.55 * c
    q[_finger_index("thumb", 4)] = 0.45 * c
    return q


def wuji_pinch(closure: float = 1.0) -> np.ndarray:
    c = min(1.0, max(0.0, float(closure)))
    q = wuji_open()
    q[_finger_index("index", 1)] = 0.75 * c
    q[_finger_index("index", 3)] = 0.65 * c
    q[_finger_index("index", 4)] = 0.45 * c
    q[_finger_index("thumb", 1)] = 0.06 + 1.05 * c
    q[_finger_index("thumb", 2)] = 0.45 * c
    q[_finger_index("thumb", 3)] = 0.60 * c
    q[_finger_index("thumb", 4)] = 0.50 * c
    for finger in ("middle", "ring", "pinky"):
        q[_finger_index(finger, 1)] = 1.40
        q[_finger_index(finger, 3)] = 1.40
        q[_finger_index(finger, 4)] = 1.10
    return q


@dataclass(slots=True)
class _Segment:
    kind: str
    rows: int
    detail: dict[str, Any] = field(default_factory=dict)


class Motion:
    def __init__(self, api: S2Api) -> None:
        self._api = api
        episode = api.episode
        self._dof = episode.embodiment.arm_dof
        self._width = episode.embodiment.gripper.width
        self.arm_rows: list[np.ndarray] = []
        self.eff_rows: list[np.ndarray] = []
        self.waist_rows: list[np.ndarray] = []
        self.segments: list[_Segment] = []
        self.arm_end = api._start_arm()
        self.eff_end = episode.cmd_effector.copy()
        self.waist_end = None if episode.cmd_waist is None else episode.cmd_waist.copy()
        self.uses_waist = episode.send_waist
        self.wheels_target: np.ndarray | None = None
        self.sent = False

    def _side_q(self, side: str) -> np.ndarray:
        index = ARMS.index(side)
        return self.arm_end[index * self._dof : (index + 1) * self._dof].copy()

    def _set_side(self, row: np.ndarray, side: str, q: np.ndarray) -> np.ndarray:
        index = ARMS.index(side)
        out = row.copy()
        out[index * self._dof : (index + 1) * self._dof] = q
        return out

    def _positions(self) -> dict[str, float]:
        episode = self._api.episode
        positions = episode.positions()
        positions.update(episode.kin.joint_map(arm=self.arm_end))
        if self.waist_end is not None and self.waist_end.size:
            positions.update(episode.kin.joint_map(waist=self.waist_end))
        return positions

    def _append(self, arm: np.ndarray, eff: np.ndarray | None = None, waist: np.ndarray | None = None, *, kind: str, detail: dict[str, Any]) -> None:
        rows = arm.shape[0]
        if eff is None:
            eff = np.tile(self.eff_end, (rows, 1))
        if waist is None and self.waist_end is not None:
            waist = np.tile(self.waist_end, (rows, 1))
        self.arm_rows.extend(arm)
        self.eff_rows.extend(eff)
        if waist is not None:
            self.waist_rows.extend(waist)
        self.arm_end = np.asarray(arm[-1], dtype=np.float64).copy()
        self.eff_end = np.asarray(eff[-1], dtype=np.float64).copy()
        if waist is not None:
            self.waist_end = np.asarray(waist[-1], dtype=np.float64).copy()
        self.segments.append(_Segment(kind, rows, detail))

    def grasp_pose(self, side: str) -> np.ndarray:
        return self._api.episode.kin.grasp_pose(side, self._positions())

    def copy(self) -> Motion:
        other = Motion.__new__(Motion)
        other._api, other._dof, other._width = self._api, self._dof, self._width
        other.arm_rows, other.eff_rows, other.waist_rows = list(self.arm_rows), list(self.eff_rows), list(self.waist_rows)
        other.segments = list(self.segments)
        other.arm_end, other.eff_end = self.arm_end.copy(), self.eff_end.copy()
        other.waist_end = None if self.waist_end is None else self.waist_end.copy()
        other.uses_waist = self.uses_waist
        other.wheels_target = None if self.wheels_target is None else self.wheels_target.copy()
        other.sent = self.sent
        return other

    def adopt(self, other: Motion) -> Motion:
        for name in ("arm_rows", "eff_rows", "waist_rows", "segments", "arm_end", "eff_end", "waist_end", "uses_waist", "wheels_target"):
            setattr(self, name, getattr(other, name))
        return self

    def tcp(
        self,
        targets: Mapping[str, np.ndarray] | str,
        target: np.ndarray | None = None,
        *,
        linear: bool = False,
        frame: str = "grasp",
        free_axis: Sequence[float] | None = None,
        rows: int | None = None,
        limits: Limits | None = None,
        gripper: float | str | None = None,
        planner: str = "ik",
        scene: Any = "head",
        ignore: Sequence[Any] = (),
        slow_down: float = 1.0,
        waist: Sequence[float] | None = None,
        waist_limits: Limits | None = None,
    ) -> Motion:
        api = self._api
        episode = api.episode
        lim = limits or api.limits
        goals = {targets: target} if isinstance(targets, str) else dict(targets)
        for side in goals:
            episode.embodiment.check_active(side)
        positions = self._positions()
        waist_goal = waist_start = None
        if waist is not None:
            if linear or planner == "curobo":
                raise MotionError("tcp(waist=...) takes joint-space IK moves only")
            names = episode.embodiment.waist_joint_names
            if not names or episode.runtime != "isaac":
                raise MotionError(f"{episode.embodiment.key} has no waist channel in this runtime")
            waist_start = self.waist_end if self.waist_end is not None else np.zeros(len(names))
            lo, hi = episode.kin.model.joint_limits(names)
            waist_goal = np.clip(np.asarray(waist, dtype=np.float64).reshape(waist_start.shape), lo, hi)
            positions = dict(positions)
            positions.update(episode.kin.joint_map(waist=waist_goal))
        if linear:
            if len(goals) != 1:
                raise MotionError("linear moves take one arm")
            side, goal = next(iter(goals.items()))
            start = episode.kin.link_pose(episode.embodiment.arm(side).grasp_parent_link, positions) @ episode.embodiment.arm(side).grasp_from_parent if frame == "grasp" else (
                episode.kin.tcp_pose(side, positions) if frame == "tcp" else episode.kin.flange_pose(side, positions)
            )
            goal = np.asarray(goal, dtype=np.float64)
            if free_axis is not None:
                goal = _align_free_axis(start, goal, free_axis)
            q_rows = cartesian_line(
                episode.kin, side, start, goal, self._side_q(side), positions,
                dt=episode.row_dt, limits=lim, frame=frame, free_axis=free_axis,
                position_tolerance_m=api.ik_position_tolerance_m, rotation_tolerance_rad=api.ik_rotation_tolerance_rad,
            )
            if rows is not None and rows > q_rows.shape[0]:
                index = np.linspace(0, q_rows.shape[0] - 1, rows)
                lower = np.floor(index).astype(int)
                upper = np.minimum(lower + 1, q_rows.shape[0] - 1)
                frac = (index - lower)[:, None]
                q_rows = q_rows[lower] * (1 - frac) + q_rows[upper] * frac
            arm = np.array([self._set_side(self.arm_end, side, q) for q in q_rows])
            eff = self._gripper_rows(side, gripper, arm.shape[0])
            self._append(arm, eff, kind="tcp_line", detail={"side": side, "rows": arm.shape[0]})
            return self
        if planner == "curobo":
            if frame != "grasp" or free_axis is not None:
                raise MotionError("cuRobo plans full grasp-frame poses (frame='grasp', no free_axis)")
            from cap_harness.g2s2.planner import PlanningError, resample

            points = api.scene_points(scene, ignore=ignore, start_arm=self.arm_end)
            try:
                trajectory, dt = api.curobo().plan(episode, {s: np.asarray(g, dtype=np.float64) for s, g in goals.items()}, self.arm_end, scene_points=points)
            except PlanningError as error:
                raise MotionError(f"cuRobo: {error}") from error
            arm = resample(trajectory, dt, episode.row_dt, slow_down=slow_down)
            sides = list(goals)
            eff = self._gripper_rows(sides[0], gripper, arm.shape[0]) if gripper is not None and len(sides) == 1 else None
            reached = dict(positions)
            reached.update(episode.kin.joint_map(arm=arm[-1]))
            errors = {s: round(float(np.linalg.norm(episode.kin.grasp_pose(s, reached)[:3, 3] - np.asarray(g)[:3, 3])) * 1000, 2) for s, g in goals.items()}
            self._append(arm, eff, kind="tcp_curobo", detail={"rows": arm.shape[0], "end_error_mm": errors, "scene_points": 0 if points is None else int(len(points))})
            return self
        end = self.arm_end.copy()
        detail: dict[str, Any] = {}
        for side, goal in goals.items():
            result = episode.kin.solve(
                side, np.asarray(goal, dtype=np.float64), self._side_q(side), positions, frame=frame, free_axis=free_axis,
                position_tolerance_m=api.ik_position_tolerance_m, rotation_tolerance_rad=api.ik_rotation_tolerance_rad, restarts=10,
            )
            if not result.ok:
                raise MotionError(f"no IK for the {side} {frame} target: {result.message}")
            end = self._set_side(end, side, result.joints)
            detail[side] = {"ik_mm": round(result.position_error_m * 1000, 3), "ik_deg": round(math.degrees(result.rotation_error_rad), 3)}
        n = rows or joint_move_rows(end - self.arm_end, episode.row_dt, lim)
        waist_rows = None
        if waist_goal is not None:
            if rows is None:
                n = max(n, joint_move_rows(waist_goal - waist_start, episode.row_dt, waist_limits or Limits(joint_velocity=0.5, joint_acceleration=2.0)))
            waist_rows = interpolate_joints(waist_start, waist_goal, n)
            self.uses_waist = True
            detail["waist"] = [round(float(v), 3) for v in waist_goal]
        arm = interpolate_joints(self.arm_end, end, n)
        sides = list(goals)
        eff = self._gripper_rows(sides[0], gripper, n) if gripper is not None and len(sides) == 1 else None
        self._append(arm, eff, waist_rows, kind="tcp", detail={**detail, "rows": n})
        return self

    def joints(self, side: str, q: Sequence[float], *, rows: int | None = None, limits: Limits | None = None, gripper: float | str | None = None) -> Motion:
        api = self._api
        episode = api.episode
        episode.embodiment.check_active(side)
        lo, hi = episode.kin.limits(side)
        q = np.clip(np.asarray(q, dtype=np.float64).reshape(self._dof), lo, hi)
        end = self._set_side(self.arm_end, side, q)
        n = rows or joint_move_rows(end - self.arm_end, episode.row_dt, limits or api.limits)
        arm = interpolate_joints(self.arm_end, end, n)
        self._append(arm, self._gripper_rows(side, gripper, n), kind="joints", detail={"side": side, "rows": n})
        return self

    def _effector_target(self, side: str, value: float | str | Sequence[float]) -> np.ndarray:
        episode = self._api.episode
        gripper = episode.embodiment.gripper
        index = ARMS.index(side)
        target = self.eff_end.copy()
        if gripper.kind == "parallel":
            opening = {"open": 1.0, "close": 0.0, "closed": 0.0}.get(value, value) if isinstance(value, str) else value
            target[index] = gripper.wire_from_opening(float(opening))
        else:
            if isinstance(value, str):
                q = {"open": wuji_open(), "power": wuji_power(1.0), "close": wuji_power(1.0), "pinch": wuji_pinch(1.0)}[value]
            else:
                q = np.asarray(value, dtype=np.float64).reshape(20)
            target[index * 20 : (index + 1) * 20] = gripper.clip_hand(side, q)
        return target

    def _gripper_rows(self, side: str, value: float | str | None, rows: int) -> np.ndarray | None:
        if value is None:
            return None
        return ramp(self.eff_end, self._effector_target(side, value), rows)

    def gripper(self, side: str, value: float | str, *, rows: int | None = None) -> Motion:
        self._api.episode.embodiment.check_active(side)
        n = rows or self._api.gripper_rows
        eff = ramp(self.eff_end, self._effector_target(side, value), n)
        self._append(np.tile(self.arm_end, (n, 1)), eff, kind="gripper", detail={"side": side, "value": value if isinstance(value, (str, float, int)) else "q20", "rows": n})
        return self

    def hand(self, side: str, q: Sequence[float] | str, *, rows: int | None = None) -> Motion:
        return self.gripper(side, q, rows=rows)

    def waist(self, values: Sequence[float], *, rows: int | None = None, limits: Limits | None = None) -> Motion:
        episode = self._api.episode
        if not episode.embodiment.waist_joint_names:
            raise MotionError(f"{episode.embodiment.key} has no waist channel")
        if self._api.episode.runtime != "isaac":
            raise MotionError("only the Isaac runtime takes waist commands")
        start = self.waist_end if self.waist_end is not None else np.zeros(len(episode.embodiment.waist_joint_names))
        goal = np.asarray(values, dtype=np.float64).reshape(start.shape)
        names = episode.embodiment.waist_joint_names
        lo, hi = episode.kin.model.joint_limits(names)
        goal = np.clip(goal, lo, hi)
        n = rows or joint_move_rows(goal - start, episode.row_dt, limits or Limits(joint_velocity=0.5, joint_acceleration=2.0))
        waist = interpolate_joints(start, goal, n)
        self.uses_waist = True
        self._append(np.tile(self.arm_end, (n, 1)), None, waist, kind="waist", detail={"rows": n})
        return self

    def base(self, dx: float = 0.0, dy: float = 0.0, dyaw: float = 0.0, *, rows: int | None = None, speed: float = 0.25, yaw_speed: float = 0.6) -> Motion:
        api = self._api
        if not (api.episode.task is not None and api.episode.task.wbc):
            raise MotionError("base moves are WBC-only (Newton / Rubik / fixed-base tasks reject wheels)")
        from cap_harness.g2s2.episode import compose_planar

        delta = np.array([dx, dy, dyaw], dtype=np.float64)
        self.wheels_target = delta if self.wheels_target is None else compose_planar(self.wheels_target, delta)
        distance = math.hypot(dx, dy)
        n = rows or max(2, int(math.ceil((distance / speed + abs(dyaw) / yaw_speed + 0.5) / api.episode.row_dt)))
        self._append(np.tile(self.arm_end, (n, 1)), kind="base", detail={"delta": [dx, dy, dyaw], "rows": n})
        return self

    def hold(self, rows: int) -> Motion:
        n = max(1, int(rows))
        self._append(np.tile(self.arm_end, (n, 1)), kind="hold", detail={"rows": n})
        return self

    @property
    def rows(self) -> int:
        return len(self.arm_rows)

    def send(self, *, depth_next: bool = False, settle: int | None = None, label: str = "") -> Any:
        if self.sent:
            raise MotionError("this motion was already sent")
        if not self.segments:
            raise MotionError("nothing planned")
        settle_rows = self._api.settle_rows if settle is None else int(settle)
        if settle_rows > 0:
            self.hold(settle_rows)
        api = self._api
        episode = api.episode
        arm = np.asarray(self.arm_rows)
        eff = np.asarray(self.eff_rows)
        waist = np.asarray(self.waist_rows) if (self.uses_waist and len(self.waist_rows) == arm.shape[0]) else None
        wheels = None
        if self.wheels_target is not None:
            wheels = np.tile(self.wheels_target, (arm.shape[0], 1))
        chunk = episode.make_chunk(arm, eff, waist=waist, wheels=wheels)
        self.sent = True
        record = {
            "label": label,
            "rows": arm.shape[0],
            "segments": [{"kind": s.kind, "rows": s.rows, **s.detail} for s in self.segments],
            "depth_next": depth_next,
            "phase": None if episode.budget is None else episode.budget.phase,
            "t": round(time.time() - episode.started_s, 3),
        }
        api.motions.append(record)
        return episode.dispatch(chunk, need_depth=depth_next, label=label or self.segments[0].kind)


class S2Api:
    def __init__(
        self,
        episode: Episode,
        *,
        sam3: Sam3Client | None = None,
        limits: Limits | None = None,
        settle_rows: int = 8,
        gripper_rows: int = 30,
    ) -> None:
        self.episode = episode
        self.sam3 = sam3
        self.limits = limits or Limits()
        self.settle_rows = settle_rows
        self.gripper_rows = gripper_rows
        self.ik_position_tolerance_m = 0.002
        self.ik_rotation_tolerance_rad = math.radians(1.0)
        self.planner: Any = None
        self.motions: list[dict[str, Any]] = []
        self.detections: list[dict[str, Any]] = []
        self.notes: dict[str, Any] = {}
        self.messages: list[str] = []

    def task_info(self) -> dict[str, Any]:
        e = self.episode
        info = {
            "wire_task_name": e.first.task_name,
            "prompt": e.first.prompt,
            "segments": list(e.first.segments),
            "embodiment": e.embodiment.key,
            "robot_type": e.embodiment.robot_type,
            "ee_type": e.embodiment.ee_type,
            "runtime": e.runtime,
            "active_arms": list(e.embodiment.active_arms),
            "row_dt_s": e.row_dt,
        }
        if e.task is not None:
            info.update(e.task.summary())
        return info

    def budget(self) -> dict[str, Any]:
        return {} if self.episode.budget is None else self.episode.budget.status()

    def dispatches_left(self) -> int:
        return 99 if self.episode.budget is None else self.episode.budget.dispatches_left()

    def rows_left(self) -> int:
        return 10**6 if self.episode.budget is None else self.episode.budget.rows_left()

    def reply_elapsed(self) -> float:
        return float(time.time() - getattr(self.episode, "request_at", time.time()))

    def stage_done(self, n: int = 1) -> int:
        if self.episode.budget is None:
            return 0
        return self.episode.budget.advance(self.episode.budget.phase + int(n))

    def log(self, *parts: Any) -> None:
        self.messages.append(" ".join(str(p) for p in parts))

    def note(self, key: str, value: Any) -> None:
        self.notes[str(key)] = value

    def _start_arm(self) -> np.ndarray:
        e = self.episode
        measured = e.request.arm
        if np.max(np.abs(measured - e.cmd_arm)) > 0.05:
            self.log(f"start from measured arm (|measured - commanded| = {np.max(np.abs(measured - e.cmd_arm)):.3f} rad)")
            return measured.copy()
        return e.cmd_arm.copy()

    def robot_state(self) -> dict[str, Any]:
        e = self.episode
        r = e.request
        positions = e.positions()
        state: dict[str, Any] = {
            "arm": {side: r.arm_joints(side) for side in ARMS},
            "effector": {side: r.effector_values(side) for side in ARMS},
            "waist": r.waist.copy(),
            "head": r.head.copy(),
            "flange": {side: e.kin.flange_pose(side, positions) for side in ARMS},
            "tcp": {side: e.kin.tcp_pose(side, positions) for side in ARMS},
            "grasp": {side: e.kin.grasp_pose(side, positions) for side in ARMS},
            "odom": e.odom.copy(),
            "has_depth": r.has_depth,
            "sequence": r.sequence,
        }
        if e.embodiment.gripper.kind == "parallel":
            state["opening"] = {side: r.gripper_opening(side) for side in ARMS}
        return state

    def grasp_pose(self, side: str) -> np.ndarray:
        return self.episode.kin.grasp_pose(side, self.episode.positions())

    def tcp_pose(self, side: str) -> np.ndarray:
        return self.episode.kin.tcp_pose(side, self.episode.positions())

    def flange_pose(self, side: str) -> np.ndarray:
        return self.episode.kin.flange_pose(side, self.episode.positions())

    def base_to_start(self, point: Sequence[float]) -> np.ndarray:
        x, y, yaw = (float(v) for v in self.episode.odom)
        p = np.asarray(point, dtype=np.float64).reshape(3)
        c, s = math.cos(yaw), math.sin(yaw)
        return np.array([x + c * p[0] - s * p[1], y + s * p[0] + c * p[1], p[2]])

    def start_to_base(self, point: Sequence[float]) -> np.ndarray:
        x, y, yaw = (float(v) for v in self.episode.odom)
        p = np.asarray(point, dtype=np.float64).reshape(3)
        dx, dy = p[0] - x, p[1] - y
        c, s = math.cos(yaw), math.sin(yaw)
        return np.array([c * dx + s * dy, -s * dx + c * dy, p[2]])

    def shoulder(self, side: str) -> np.ndarray:
        e = self.episode
        first = e.kin.model.joints[e.embodiment.arm(side).joint_names[0]]
        return e.kin.link_pose(first.child, e.positions())[:3, 3]

    def camera(self, name: str = "head") -> CameraView:
        return self.episode.camera(name)

    def has_depth(self, camera: str | None = None) -> bool:
        depth = self.episode.request.depth_m
        return bool(depth) if camera is None else camera in depth

    def curobo(self) -> Any:
        if self.planner is None:
            from cap_harness.g2s2.planner import CuroboPlanner

            self.planner = CuroboPlanner()
        return self.planner

    def scene_points(
        self,
        scene: Any = "head",
        *,
        ignore: Sequence[Any] = (),
        start_arm: np.ndarray | None = None,
        radius_m: float = 1.6,
        gripper_clearance_m: float = 0.12,
        ignore_margin_m: float = 0.025,
        voxel_m: float = 0.01,
    ) -> np.ndarray | None:
        if scene is None:
            return None
        if isinstance(scene, str):
            view = self.camera(scene)
            if not view.has_depth or view.T_base_optical is None:
                self.log(f"scene_points: no depth on {scene}; planning without a scene")
                return None
            points = view.points_from_mask(None, stride=3)
        else:
            points = np.asarray(scene, dtype=np.float64).reshape(-1, 3)
        points = points[np.linalg.norm(points, axis=1) <= radius_m]
        if len(points) == 0:
            return None
        keys = np.floor(points / voxel_m).astype(np.int64)
        _, first = np.unique(keys, axis=0, return_index=True)
        points = points[np.sort(first)]
        e = self.episode
        positions = e.positions()
        if start_arm is not None:
            positions.update(e.kin.joint_map(arm=start_arm))
        anchors = []
        for side in ARMS:
            anchors.append(e.kin.tcp_pose(side, positions)[:3, 3])
            anchors.append(e.kin.flange_pose(side, positions)[:3, 3])
        keep = np.ones(len(points), dtype=bool)
        for anchor in anchors:
            keep &= np.linalg.norm(points - anchor, axis=1) > gripper_clearance_m
        for item in ignore:
            other = item.points if isinstance(item, Detection) else np.asarray(item, dtype=np.float64).reshape(-1, 3)
            if len(other) == 0:
                continue
            low = other.min(axis=0) - ignore_margin_m
            high = other.max(axis=0) + ignore_margin_m
            keep &= ~np.all((points >= low) & (points <= high), axis=1)
        points = points[keep]
        return points if len(points) else None

    def _sam3(self) -> Sam3Client:
        if self.sam3 is None:
            self.sam3 = Sam3Client()
        return self.sam3

    def segment(self, text: str, camera: str = "head") -> list[dict[str, Any]]:
        return self._sam3().segment(self.camera(camera).rgb, text)

    def perceive(self, text: str, camera: str = "head", *, min_score: float = 0.0, erode: int = 2, max_detections: int = 3) -> list[Detection]:
        view = self.camera(camera)
        if not view.has_depth:
            raise MotionError(f"no depth on the current request for {camera}: send the previous motion with depth_next=True or call observe()")
        found = detect(view, self._sam3(), text, min_score=min_score, erode_px=erode, max_detections=max_detections)
        self.detections.append({"text": text, "camera": camera, "sequence": self.episode.request.sequence, "found": [d.summary() for d in found]})
        return found

    def locate_rgb(
        self,
        text: str,
        cameras: Sequence[str] = ("head", "hand_right", "hand_left"),
        *,
        min_score: float = 0.3,
        max_rms_m: float = 0.02,
    ) -> dict[str, Any] | None:
        from itertools import combinations

        from cap_harness.g2s2.cameras import triangulate

        candidates = []
        for name in cameras:
            view = self.camera(name)
            if view.model is None or view.T_base_optical is None:
                continue
            masks = self._sam3().segment(view.rgb, text)
            if not masks:
                continue
            v, u = np.nonzero(masks[0]["mask"])
            if u.size < 20:
                continue
            candidates.append((masks[0]["score"], name, view, (float(np.mean(u)), float(np.mean(v)))))
        strong = [c for c in candidates if c[0] >= min_score]
        pool = strong if len(strong) >= 2 else sorted(candidates, key=lambda c: c[0], reverse=True)[:3]
        best = None
        for pair in combinations(pool, 2):
            point, rms = triangulate([c[2] for c in pair], [c[3] for c in pair])
            if best is None or rms < best[1]:
                best = (point, rms, list(pair))
        if best is None:
            return None
        point, rms, used = best
        for extra in pool:
            if extra in used:
                continue
            trial_point, trial_rms = triangulate([c[2] for c in used + [extra]], [c[3] for c in used + [extra]])
            if trial_rms <= max_rms_m:
                point, rms, used = trial_point, trial_rms, used + [extra]
        result = {"point": point, "rms_m": rms, "cameras": [c[1] for c in used], "pixels": [c[3] for c in used],
                  "scores": [round(c[0], 3) for c in used]}
        self.detections.append({"text": text, "camera": "+".join(result["cameras"]), "sequence": self.episode.request.sequence,
                                "found": [{"triangulated": [round(float(v), 4) for v in point], "rms_m": round(rms, 4), "scores": result["scores"]}]})
        return result if rms <= max_rms_m else None

    def points(self, camera: str = "head", mask: np.ndarray | None = None, *, stride: int = 2) -> np.ndarray:
        view = self.camera(camera)
        if mask is not None:
            return mask_points(view, mask)
        return view.points_from_mask(None, stride=stride)

    def table_height(self, camera: str = "head", *, box: Sequence[float] | None = None, stride: int = 3) -> float:
        view = self.camera(camera)
        mask = None
        if box is not None:
            x0, y0, x1, y1 = (int(round(v)) for v in box)
            mask = np.zeros(view.depth_m.shape, dtype=bool)
            mask[max(0, y0) : y1, max(0, x0) : x1] = True
        return dominant_plane_z(view.points_from_mask(mask, stride=stride))

    def motion(self) -> Motion:
        return Motion(self)

    def move_tcp(self, side: str, target: np.ndarray, *, linear: bool = False, gripper: float | str | None = None, depth_next: bool = False, label: str = "", **kw: Any) -> Any:
        return self.motion().tcp(side, target, linear=linear, gripper=gripper, **kw).send(depth_next=depth_next, label=label or "move_tcp")

    def move_joints(self, side: str, q: Sequence[float], *, depth_next: bool = False, label: str = "", **kw: Any) -> Any:
        return self.motion().joints(side, q, **kw).send(depth_next=depth_next, label=label or "move_joints")

    def grasp(self, side: str, *, rows: int | None = None, depth_next: bool = False, label: str = "") -> Any:
        return self.motion().gripper(side, "close", rows=rows).send(depth_next=depth_next, label=label or "grasp")

    def release(self, side: str, *, rows: int | None = None, depth_next: bool = False, label: str = "") -> Any:
        return self.motion().gripper(side, "open", rows=rows).send(depth_next=depth_next, label=label or "release")

    def set_gripper(self, side: str, opening: float, *, rows: int | None = None, depth_next: bool = False) -> Any:
        return self.motion().gripper(side, float(opening), rows=rows).send(depth_next=depth_next, label="set_gripper")

    def hand_open(self, side: str, **kw: Any) -> Any:
        return self.motion().hand(side, "open", rows=kw.pop("rows", None)).send(label="hand_open", **kw)

    def hand_power_grasp(self, side: str, closure: float = 1.0, **kw: Any) -> Any:
        return self.motion().hand(side, wuji_power(closure), rows=kw.pop("rows", None)).send(label="hand_power", **kw)

    def hand_pinch(self, side: str, closure: float = 1.0, **kw: Any) -> Any:
        return self.motion().hand(side, wuji_pinch(closure), rows=kw.pop("rows", None)).send(label="hand_pinch", **kw)

    def hand_set(self, side: str, q: Sequence[float], **kw: Any) -> Any:
        return self.motion().hand(side, q, rows=kw.pop("rows", None)).send(label="hand_set", **kw)

    def base_move(self, dx: float = 0.0, dy: float = 0.0, dyaw: float = 0.0, *, rows: int | None = None, depth_next: bool = False) -> Any:
        return self.motion().base(dx, dy, dyaw, rows=rows).send(depth_next=depth_next, label="base_move")

    def waist_move(self, values: Sequence[float], *, rows: int | None = None, depth_next: bool = False) -> Any:
        return self.motion().waist(values, rows=rows).send(depth_next=depth_next, label="waist_move")

    def hold(self, rows: int, *, depth_next: bool = False, label: str = "hold") -> Any:
        return self.motion().hold(rows).send(depth_next=depth_next, settle=0, label=label)

    def choose_arm(self, point: Sequence[float]) -> str:
        active = self.episode.embodiment.active_arms
        if len(active) == 1:
            return active[0]
        return "left" if float(point[1]) >= 0.0 else "right"

    def top_grasp(
        self,
        side: str,
        detection: Detection | None = None,
        *,
        center: Sequence[float] | None = None,
        top_z: float | None = None,
        bottom_z: float | None = None,
        yaw: float | None = None,
        above: float = 0.10,
        lift: float = 0.10,
        depth_into: float = 0.02,
        floor_clearance: float = 0.008,
        close_rows: int | None = None,
        motion: Motion | None = None,
        tilts: Sequence[float] = (20.0, -20.0, 35.0, -35.0, 50.0, -50.0),
        waist_yaws: Sequence[float] = (0.2, 0.35, 0.5),
        settle_before_close: int = 10,
        descend_speed: float = 0.25,
        tip_clearance: float | None = None,
        contact_width: float | None = None,
        limits: Limits | None = None,
        settle_after_close: int = 15,
        corner_yaw: bool = False,
    ) -> Motion:
        if limits is not None:
            saved = self.limits
            self.limits = limits
            try:
                return self.top_grasp(side, detection, center=center, top_z=top_z, bottom_z=bottom_z, yaw=yaw, above=above,
                                      lift=lift, depth_into=depth_into, floor_clearance=floor_clearance,
                                      close_rows=close_rows, motion=motion, tilts=tilts, waist_yaws=waist_yaws,
                                      settle_before_close=settle_before_close, descend_speed=descend_speed,
                                      tip_clearance=tip_clearance, contact_width=contact_width,
                                      settle_after_close=settle_after_close, corner_yaw=corner_yaw)
            finally:
                self.limits = saved
        e = self.episode
        if e.embodiment.gripper.kind != "parallel":
            raise MotionError("top_grasp is for parallel grippers")
        if detection is not None:
            c = detection.center
            top = detection.top_z if top_z is None else top_z
            bottom = detection.bottom_z if bottom_z is None else bottom_z
            if yaw is None:
                yaw, length, width = detection.principal_yaw()
                if not (length > width * 1.15):
                    yaw = None
                    if corner_yaw:
                        cy, ratio = detection.corner_yaw()
                        if ratio > 1.3:
                            yaw = cy
        else:
            if center is None:
                raise MotionError("top_grasp needs a detection or a center")
            c = np.asarray(center, dtype=np.float64)
            top = float(c[2]) if top_z is None else top_z
            bottom = float(c[2]) - 0.02 if bottom_z is None else bottom_z
        base = motion or self.motion()
        current = base.grasp_pose(side)
        current_yaw = math.atan2(float(current[1, 1]), float(current[0, 1]))
        closing = current_yaw if yaw is None else float(yaw) + math.pi / 2.0
        if math.cos(closing - current_yaw) < 0.0:
            closing += math.pi
        grasp_z = max(float(top) - depth_into, float(bottom) + floor_clearance)
        tips = None
        if tip_clearance is not None and e.embodiment.key == "g2_crsb_90d":
            if contact_width is None:
                contact_width = float(detection.principal_yaw()[2]) if detection is not None else 0.02
                if not math.isfinite(contact_width):
                    contact_width = 0.02
            reach = tip_beyond_grasp_90d(contact_width)
            grasp_z = float(bottom) + float(tip_clearance) + reach
            tips = {"contact_width": round(float(contact_width), 4), "tip_reach": round(reach, 4), "tips_z": round(float(bottom) + float(tip_clearance), 4)}
        shoulder = self.shoulder(side)
        toward = math.atan2(float(shoulder[1]) - float(c[1]), float(shoulder[0]) - float(c[0]))
        attempts = [(0.0, closing, toward), (0.0, closing + math.pi / 2.0 if yaw is None else closing, toward)]
        attempts += [(math.radians(abs(t)), closing, toward if t >= 0 else toward + math.pi) for t in tilts]
        errors = []
        torso = []
        names = e.embodiment.waist_joint_names
        if e.runtime == "isaac" and len(names) == 5 and base.waist_end is not None and waist_yaws:
            sign = -1.0 if float(c[1]) < float(shoulder[1]) else 1.0
            for offset_yaw in waist_yaws:
                turned = base.waist_end.copy()
                turned[4] += sign * float(offset_yaw)
                torso.append(turned)
        plans = [(None, attempt) for attempt in attempts] + [(w, attempt) for w in torso for attempt in attempts[:4]]
        for waist_target, (tilt, close_yaw, lean) in plans:
            trial = base.copy()
            if waist_target is not None:
                try:
                    trial.waist(waist_target)
                except MotionError as error:
                    errors.append(f"waist: {error}")
                    continue
                shoulder_now = e.kin.link_pose(e.kin.model.joints[e.embodiment.arm(side).joint_names[0]].child, trial._positions())[:3, 3]
                lean = math.atan2(float(shoulder_now[1]) - float(c[1]), float(shoulder_now[0]) - float(c[0])) + (lean - toward)
            grasp = top_down((float(c[0]), float(c[1]), grasp_z), close_yaw, tilt=tilt, tilt_toward=lean)
            approach = grasp[:3, 2]
            pre = grasp.copy()
            pre[:3, 3] = grasp[:3, 3] - approach * above
            up = offset(grasp, dz=lift)
            try:
                free = (0.0, 0.0, 1.0) if yaw is None else None
                trial.tcp(side, pre, gripper="open", free_axis=free)
                lim = self.limits
                down = Limits(joint_velocity=lim.joint_velocity, joint_acceleration=lim.joint_acceleration,
                              linear_velocity=min(lim.linear_velocity, descend_speed), angular_velocity=lim.angular_velocity)
                trial.tcp(side, grasp, linear=True, free_axis=free, limits=down)
                if settle_before_close > 0:
                    trial.hold(settle_before_close)
                trial.gripper(side, "close", rows=close_rows)
                if settle_after_close > 0:
                    trial.hold(settle_after_close)
                trial.tcp(side, up, linear=True, free_axis=free)
            except MotionError as error:
                errors.append(f"{'waist ' if waist_target is not None else ''}tilt {math.degrees(tilt):.0f} yaw {math.degrees(close_yaw):.0f}: {error}")
                continue
            base.adopt(trial)
            self.note(
                f"top_grasp_{side}",
                {"grasp": [round(float(v), 4) for v in grasp[:3, 3]], "closing_yaw_deg": round(math.degrees(close_yaw), 1),
                 "tilt_deg": round(math.degrees(tilt), 1), "failed_attempts": errors,
                 "waist": None if waist_target is None else [round(float(v), 4) for v in waist_target], "tips": tips},
            )
            return base
        raise MotionError(f"no top grasp for the {side} arm: " + " | ".join(errors))

    def plausible(
        self,
        detections: Sequence[Detection],
        *,
        size: tuple[float, float] = (0.01, 0.30),
        table_z: float | None = None,
        z_tolerance: float = 0.06,
        min_points: int = 20,
        away_from_grippers: float = 0.06,
    ) -> list[Detection]:
        tcps = [self.tcp_pose(side)[:3, 3] for side in ARMS]
        kept = []
        for det in detections:
            if det.points.shape[0] < min_points:
                continue
            extent = det.extent
            horizontal = float(max(extent[0], extent[1]))
            if not (size[0] <= horizontal <= size[1]):
                continue
            if table_z is not None and abs(det.bottom_z - table_z) > z_tolerance:
                continue
            if any(np.linalg.norm(det.center - tcp) < away_from_grippers for tcp in tcps):
                continue
            kept.append(det)
        return sorted(kept, key=lambda d: d.score, reverse=True)

    def on_side(self, detections: Sequence[Detection], side: str | None, *, margin_m: float = 0.0) -> list[Detection]:
        if side not in ("left", "right"):
            return list(detections)
        sign = 1.0 if side == "left" else -1.0
        kept = [d for d in detections if sign * float(d.center[1]) > margin_m]
        return kept or list(detections)

    def instructed_arm(self, text: str | None = None) -> str | None:
        e = self.episode
        sentence = (text if text is not None else " ".join(e.first.segments) or e.first.prompt).lower()
        found = [side for side in ARMS if any(f"{side} {part}" in sentence for part in ("gripper", "hand", "arm"))]
        if len(found) == 1 and found[0] in e.embodiment.active_arms:
            return found[0]
        return None

    def find(self, queries: Sequence[str], camera: str = "head", **filters: Any) -> Detection | None:
        for query in queries:
            found = self.plausible(self.perceive(query, camera, max_detections=5), **filters)
            if found:
                self.note(f"find:{queries[0]}", {"query": query, **found[0].summary()})
                return found[0]
        return None

    def pick_top(self, side: str, detection: Detection, *, depth_next: bool = True, label: str = "pick", **kw: Any) -> Any:
        return self.top_grasp(side, detection, **kw).send(depth_next=depth_next, label=label)

    def holding(self, side: str, *, closed_below: float = 0.03) -> bool | None:
        e = self.episode
        if e.embodiment.gripper.kind != "parallel":
            return None
        commanded = e.embodiment.gripper.opening_from_wire(float(e.cmd_effector[ARMS.index(side)]))
        measured = e.request.gripper_opening(side)
        return bool(commanded < 0.05 and measured > closed_below)

    def hand_lag(self, side: str) -> float | None:
        e = self.episode
        values = e.request.effector_values(side)
        n = values.size
        if n == 0:
            return None
        start = ARMS.index(side) * n if e.cmd_effector.size == 2 * n else 0
        cmd = e.cmd_effector[start : start + n]
        if cmd.size != n:
            return None
        return round(float(np.max(np.abs(values - cmd))), 4)

    def observe(self, *, depth: bool = True, rows: int = 1) -> Any:
        if depth and not self.episode.request.has_depth:
            return self.hold(rows, depth_next=True, label="observe")
        return self.episode.request

    def move_flange_server(self, side: str, target: np.ndarray, *, rows: int = 60, frame: str = "base_link", depth_next: bool = False) -> Any:
        e = self.episode
        if e.runtime != "isaac":
            raise MotionError("EEF_ABS primitive is Isaac-only")
        e.embodiment.check_active(side)
        from cap_harness.g2s2.motion import quintic, slerp_matrix

        positions = e.commanded_positions()
        arm_frame = np.eye(4) if frame == "base_link" else e.kin.arm_frame_pose(positions)
        inv_frame = np.linalg.inv(arm_frame)
        eef: dict[str, np.ndarray] = {}
        for s in ARMS:
            start = inv_frame @ e.kin.flange_pose(s, positions)
            goal = inv_frame @ np.asarray(target, dtype=np.float64) if s == side else start
            values = np.zeros((rows, 7))
            for i, t in enumerate(quintic(np.arange(1, rows + 1) / rows)):
                p = start[:3, 3] + (goal[:3, 3] - start[:3, 3]) * t
                r = slerp_matrix(start[:3, :3], goal[:3, :3], float(t))
                m = np.eye(4)
                m[:3, :3] = r
                values[i, :3], values[i, 3:] = p, matrix_to_xyzw(m)
            eef[s] = values
        chunk = Chunk(arm=np.zeros((rows, e.embodiment.wire_arm_width)), effector=np.tile(e.cmd_effector, (rows, 1)), eef=eef, eef_frame=frame)
        if e.send_waist and e.cmd_waist is not None:
            chunk.waist = np.tile(e.cmd_waist, (rows, 1))
        if e.send_wheels:
            chunk.wheels = np.zeros((rows, 3))
        self.motions.append({"label": "eef_server", "rows": rows, "side": side, "t": round(time.time() - e.started_s, 3)})
        return e.dispatch(chunk, need_depth=depth_next, label="eef_server")

    def segment_text(self, index: int = 0) -> str:
        segments = list(self.episode.first.segments or [])
        if 0 <= index < len(segments):
            return str(segments[index])
        return str(self.episode.first.prompt or "")

    def find_object(
        self,
        queries: Sequence[str],
        *,
        side: str | None = None,
        size: tuple[float, float] = (0.01, 0.30),
        table_z: float | None = None,
        z_tolerance: float = 0.06,
        camera: str = "head",
        exclude: Sequence[Sequence[float]] = (),
        exclude_radius: float = 0.03,
        min_score: float = 0.0,
        max_detections: int = 5,
        region: Callable[[np.ndarray], bool] | None = None,
        rank: str = "score",
        away_from_grippers: float = 0.06,
        clip_z: tuple[float, float] | None = None,
        brightness: tuple[float, float] | None = None,
    ) -> Detection | None:
        for query in queries:
            dets = self.perceive(query, camera, min_score=min_score, max_detections=max_detections)
            if clip_z is not None:
                lo, hi = float(clip_z[0]), float(clip_z[1])
                clipped = []
                for d in dets:
                    keep = (d.points[:, 2] >= lo) & (d.points[:, 2] <= hi)
                    if int(keep.sum()) >= 10:
                        clipped.append(Detection(label=d.label, camera=d.camera, score=d.score, box=d.box, mask=d.mask,
                                                 points=d.points[keep], meta=dict(d.meta, clip_z=[lo, hi])))
                dets = clipped
            dets = self.plausible(dets, size=size, table_z=table_z, z_tolerance=z_tolerance, away_from_grippers=away_from_grippers)
            if brightness is not None:
                img = self.camera(camera).rgb
                lo_b, hi_b = float(brightness[0]), float(brightness[1])
                dets = [d for d in dets if d.mask.any() and lo_b <= float(img[d.mask].mean()) <= hi_b]
            if exclude:
                dets = [d for d in dets if all(np.linalg.norm(d.center[:2] - np.asarray(e, dtype=float)[:2]) > exclude_radius for e in exclude)]
            if region is not None:
                dets = [d for d in dets if region(d.center)]
            if not dets:
                continue
            if rank == "near" and side in ("left", "right"):
                sh = self.shoulder(side)
                dets = sorted(dets, key=lambda d: float(np.linalg.norm(d.center[:2] - sh[:2])))
            elif rank == "low":
                dets = sorted(dets, key=lambda d: d.top_z)
            elif rank == "high":
                dets = sorted(dets, key=lambda d: -d.top_z)
            dets = self.on_side(dets, side) if side in ("left", "right") else dets
            self.note(f"find_object:{queries[0]}", {"query": query, **dets[0].summary()})
            return dets[0]
        self.note(f"find_object:{queries[0]}", None)
        return None

    def footprint_width(self, camera: str, det: Any) -> float | None:
        if not self.has_depth(camera):
            return None
        floor = self.table_height(camera)
        if not math.isfinite(floor):
            return None
        P = self.points(camera, stride=1)
        P = P[np.all(np.isfinite(P), axis=1)]
        reach = 0.5 * float(max(det.extent[0], det.extent[1])) + 0.03
        r = np.hypot(P[:, 0] - float(det.mid[0]), P[:, 1] - float(det.mid[1]))
        F = P[(r < reach) & (P[:, 2] > floor + 0.003) & (P[:, 2] < floor + 0.06)]
        if F.shape[0] < 80:
            return None
        xy = F[:, :2] - np.median(F[:, :2], axis=0)
        w, V = np.linalg.eigh(np.cov(xy.T))
        s = xy @ V[:, int(np.argmin(w))]
        width = float(np.percentile(s, 98) - np.percentile(s, 2))
        return width if 0.005 < width < 0.12 else None

    def pick_object(
        self,
        side: str,
        queries: Sequence[str],
        *,
        retries: int = 1,
        find: Mapping[str, Any] | None = None,
        grasp: Mapping[str, Any] | None = None,
        label: str = "pick",
        verify: bool = True,
        limits: Limits | None = None,
        margin_rows: int = 30,
        clear_view: float = 0.0,
        refine_wrist: float = 0.0,
    ) -> dict[str, Any]:
        find = dict(find or {})
        prefer = find.pop("side", side)
        grasp = dict(grasp or {})
        wrist_floor = bool(grasp.pop("wrist_floor", False)) and refine_wrist > 0.0
        result: dict[str, Any] = {"ok": False, "holding": None, "attempts": 0, "detection": None, "reason": ""}
        saved_limits = self.limits
        for attempt in range(1 + max(0, int(retries))):
            if attempt > 0 and not self.has_depth():
                self.hold(1, depth_next=True, label="observe")
            det = self.find_object(queries, side=prefer, **find)
            if attempt == 0 and clear_view != 0.0:
                tcp = self.grasp_pose(side)[:3, 3]
                if clear_view < 0.0 or det is None or float(np.linalg.norm(tcp[:2] - det.center[:2])) < clear_view:
                    try:
                        up = self.motion()
                        target = self.grasp_pose(side).copy()
                        target[2, 3] += 0.10
                        try:
                            up.tcp(side, target, linear=True)
                        except MotionError:
                            target[1, 3] += 0.08 if side == "left" else -0.08
                            target[2, 3] -= 0.02
                            up = self.motion()
                            up.tcp(side, target)
                        up.send(depth_next=True, label="clear_view")
                        result["cleared"] = True
                        again = self.find_object(queries, side=prefer, **find)
                        det = again if again is not None else det
                    except MotionError as error:
                        result["clear_error"] = str(error)[:200]
            if det is None:
                result["reason"] = "not found"
                return result
            if attempt == 0 and refine_wrist > 0.0 and self.episode.embodiment.gripper.kind == "parallel":
                refined = self._refine_with_wrist(side, det, queries, find, refine_wrist, limits)
                result["refined"] = None if refined is None else refined.summary()
                if refined is not None:
                    det = refined
                    want_floor = wrist_floor or "tip_clearance" in grasp
                    if want_floor and "bottom_z" not in grasp and self.has_depth(f"hand_{side}"):
                        floor = self.table_height(f"hand_{side}")
                        if math.isfinite(floor) and abs(floor - float(det.bottom_z)) < 0.03:
                            grasp["bottom_z"] = floor
                            result["wrist_floor"] = round(floor, 4)
                    if grasp.get("contact_width") == "footprint":
                        width = self.footprint_width(f"hand_{side}", det)
                        result["footprint_width"] = None if width is None else round(width, 4)
                        if width is not None:
                            grasp["contact_width"] = width
            result["detection"] = det.summary()
            result.setdefault("tries", []).append({k: result["detection"][k] for k in ("label", "camera", "score", "mid", "extent", "yaw_deg")})
            try:
                if limits is not None:
                    self.limits = limits
                motion = self.motion()
                if attempt > 0 and self.episode.embodiment.gripper.kind == "parallel":
                    motion.gripper(side, "open", rows=20)
                if grasp.get("contact_width") == "footprint":
                    grasp.pop("contact_width")
                plan = self.top_grasp(side, det, motion=motion, **grasp)
            except MotionError as error:
                result["reason"] = f"plan: {str(error)[:300]}"
                return result
            finally:
                self.limits = saved_limits
            need = plan.rows + self.settle_rows
            if attempt > 0 and need > self.rows_left() - margin_rows:
                result["reason"] = f"missed; no rows left for a retry ({need} > {self.rows_left()} - {margin_rows})"
                return result
            result.setdefault("rows", []).append(int(need))
            note = self.notes.get(f"top_grasp_{side}") or {}
            result["tries"][-1]["grasp"] = note.get("grasp")
            result["tries"][-1]["closing_yaw_deg"] = note.get("closing_yaw_deg")
            plan.send(depth_next=True, label=f"{label}{attempt}" if attempt else label)
            result["attempts"] = attempt + 1
            held = self.holding(side)
            result["holding"] = held
            if held or not verify or held is None:
                result["ok"] = True if (not verify or held is None) else bool(held)
                result["reason"] = "held" if held else "unverified"
                return result
            result["reason"] = "missed"
        return result

    def hand_top_grasp(
        self,
        side: str,
        center: Sequence[float],
        *,
        grasp_z: float,
        closure: Any = None,
        yaw: float | None = None,
        above: float = 0.10,
        lift: float = 0.10,
        close_rows: int = 30,
        motion: Motion | None = None,
    ) -> Motion:
        e = self.episode
        if e.embodiment.gripper.kind == "parallel":
            raise MotionError("hand_top_grasp is for the Wuji hand")
        m = motion or self.motion()
        c = np.asarray(center, dtype=np.float64)
        z_axis = np.array([0.0, 0.0, -1.0])
        if yaw is None:
            y_axis = np.array([0.0, 1.0, 0.0])
        else:
            y_axis = np.array([math.cos(yaw), math.sin(yaw), 0.0])
        x_axis = np.cross(y_axis, z_axis)
        grasp = pose((float(c[0]), float(c[1]), float(grasp_z)), np.column_stack([x_axis, y_axis, z_axis]))
        free = (0.0, 0.0, 1.0) if yaw is None else None
        m.tcp(side, offset(grasp, dz=above), gripper="open", free_axis=free)
        m.tcp(side, grasp, linear=True, free_axis=free)
        m.hand(side, wuji_power(0.8) if closure is None else closure, rows=close_rows)
        if lift > 0:
            m.tcp(side, offset(grasp, dz=lift), linear=True, free_axis=free)
        return m

    def lifted_check(self, queries: Sequence[str], before: Detection, *, min_rise: float = 0.03, radius: float = 0.12,
                     camera: str = "head", **find: Any) -> dict[str, Any]:
        if not self.has_depth(camera):
            return {"lifted": None, "reason": "no depth"}
        near = lambda c: float(np.linalg.norm(c[:2] - before.center[:2])) < radius
        det = self.find_object(queries, camera=camera, region=near, away_from_grippers=0.0, **find)
        if det is None:
            return {"lifted": True, "reason": "not at its place"}
        rise = float(det.top_z - before.top_z)
        return {"lifted": rise >= min_rise, "rise": round(rise, 4), "reason": "seen", "now": det.summary()["mid"]}

    def _refine_with_wrist(self, side: str, det: Detection, queries: Sequence[str], find: Mapping[str, Any], hover: float,
                           limits: Limits | None) -> Detection | None:
        cam = f"hand_{side}"
        c = det.center
        yaw, length, width = det.principal_yaw()
        current = self.grasp_pose(side)
        closing = math.atan2(float(current[1, 1]), float(current[0, 1])) if not (length > width * 1.15) else yaw + math.pi / 2.0
        target = top_down((float(c[0]), float(c[1]), float(det.top_z) + hover), closing)
        saved = self.limits
        try:
            if limits is not None:
                self.limits = limits
            m = self.motion()
            m.tcp(side, target, gripper="open", free_axis=(0.0, 0.0, 1.0))
            m.send(depth_next=True, label="hover")
        except MotionError:
            return None
        finally:
            self.limits = saved
        if not self.has_depth(cam):
            return None
        size = find.get("size", (0.01, 0.30))
        clip = find.get("clip_z")
        for query in queries:
            cands = []
            for d in self.perceive(query, cam, max_detections=4):
                if clip is not None:
                    keep = (d.points[:, 2] >= float(clip[0])) & (d.points[:, 2] <= float(clip[1]))
                    if int(keep.sum()) < 10:
                        continue
                    d = Detection(label=d.label, camera=d.camera, score=d.score, box=d.box, mask=d.mask, points=d.points[keep], meta=d.meta)
                if d.points.shape[0] < 20:
                    continue
                if float(np.linalg.norm(d.center[:2] - c[:2])) > 0.05 or abs(float(d.bottom_z - det.bottom_z)) > 0.04:
                    continue
                horizontal = float(max(d.extent[0], d.extent[1]))
                if not (size[0] <= horizontal <= size[1]):
                    continue
                if any(float(np.linalg.norm(d.center - self.tcp_pose(s)[:3, 3])) < 0.05 for s in ARMS):
                    continue
                cands.append(d)
            if cands:
                best = max(cands, key=lambda d: d.score)
                self.note(f"wrist_refine_{side}", {"query": query, "head_mid": [round(float(v), 4) for v in det.mid], **best.summary()})
                return best
        self.note(f"wrist_refine_{side}", None)
        return None

    def place_held(
        self,
        side: str,
        position: Sequence[float],
        *,
        above: float = 0.08,
        release: bool = True,
        retreat: float = 0.08,
        rotation: np.ndarray | None = None,
        free_axis: Sequence[float] | None = None,
        open_rows: int | None = None,
        motion: Motion | None = None,
        send: bool = True,
        depth_next: bool = False,
        label: str = "place",
        descend_rows: int | None = None,
        limits: Limits | None = None,
    ) -> Motion:
        if limits is not None:
            saved = self.limits
            self.limits = limits
            try:
                return self.place_held(side, position, above=above, release=release, retreat=retreat, rotation=rotation,
                                       free_axis=free_axis, open_rows=open_rows, motion=motion, send=send,
                                       depth_next=depth_next, label=label, descend_rows=descend_rows)
            finally:
                self.limits = saved
        m = motion or self.motion()
        current = m.grasp_pose(side)
        target = np.eye(4)
        target[:3, :3] = current[:3, :3] if rotation is None else np.asarray(rotation, dtype=np.float64)
        target[:3, 3] = np.asarray(position, dtype=np.float64)
        pre = target.copy()
        pre[2, 3] += above
        m.tcp(side, pre, free_axis=free_axis)
        if above > 0:
            m.tcp(side, target, linear=True, free_axis=free_axis, rows=descend_rows)
        if release:
            m.gripper(side, "open", rows=open_rows)
        if retreat > 0:
            m.tcp(side, offset(target, dz=retreat), linear=True, free_axis=free_axis)
        if send:
            m.send(depth_next=depth_next, label=label)
        return m

    def end_episode(self) -> None:
        self.episode.end()

    def summary(self) -> dict[str, Any]:
        return {
            "motions": self.motions,
            "detections": self.detections,
            "notes": self.notes,
            "messages": self.messages[-200:],
            "sam3_calls": None if self.sam3 is None else self.sam3.calls,
            "sam3_s": None if self.sam3 is None else round(self.sam3.seconds, 2),
            "curobo_calls": None if self.planner is None else self.planner.calls,
            "curobo_s": None if self.planner is None else round(self.planner.seconds, 2),
        }


PROGRAM_HELPERS = {
    "pose": pose,
    "top_down": top_down,
    "offset": offset,
    "rotx": rotx,
    "roty": roty,
    "rotz": rotz,
    "wuji_open": wuji_open,
    "wuji_power": wuji_power,
    "wuji_pinch": wuji_pinch,
    "Limits": Limits,
    "MotionError": MotionError,
    "tip_beyond_grasp_90d": tip_beyond_grasp_90d,
    "EpisodeEnded": EpisodeEnded,
}

__all__ = [
    "PROGRAM_HELPERS",
    "Motion",
    "S2Api",
    "offset",
    "pose",
    "rotx",
    "roty",
    "rotz",
    "top_down",
    "wuji_open",
    "wuji_pinch",
    "wuji_power",
]
