# Copyright (c) 2026 FF EAI Robotics Ecosystem Inc. All rights reserved. Review use only; see LICENSE-FF.txt.

from __future__ import annotations

from dataclasses import dataclass
import json
import logging
import math
from pathlib import Path
import time
from typing import Any

import numpy as np

from cap_harness.g2.kinematics import invert
from cap_harness.g2s2.bridge import BridgeEvent, S2Bridge, Superseded
from cap_harness.g2s2.budget import BudgetTracker
from cap_harness.g2s2.cameras import (
    CameraModel,
    CameraView,
    fisheye_model,
    model_from_calibration,
    model_from_spec,
    pose_from_calibration,
)
from cap_harness.g2s2.embodiment import ARMS, CAMERAS, DATA_ROOT, Embodiment
from cap_harness.g2s2.kinematics import S2Kinematics
from cap_harness.g2s2.tasks import TaskSpec
from cap_harness.g2s2.wire import Chunk, S2Request, encode_chunk, terminal_reply

logger = logging.getLogger(__name__)

WAIT_BASE_S = 120.0
WAIT_PER_ROW_S = 0.2


_WUJI_NATIVE = json.loads((DATA_ROOT / "wuji_native_cameras.json").read_text(encoding="utf-8"))["cameras"]
NATIVE_CAMERAS = {
    "rubik": json.loads((DATA_ROOT / "rubik_cameras.json").read_text(encoding="utf-8"))["cameras"],
    "newton": _WUJI_NATIVE,
    "libuipc": _WUJI_NATIVE,
}


class EpisodeEnded(Exception):
    pass


@dataclass(slots=True)
class EpisodeOutcome:
    end_reason: str = ""
    final_request: S2Request | None = None
    carry_over: BridgeEvent | None = None


def compose_planar(pose: np.ndarray, delta: np.ndarray) -> np.ndarray:
    x, y, yaw = (float(v) for v in pose)
    dx, dy, dyaw = (float(v) for v in delta)
    c, s = math.cos(yaw), math.sin(yaw)
    return np.array([x + c * dx - s * dy, y + s * dx + c * dy, math.atan2(math.sin(yaw + dyaw), math.cos(yaw + dyaw))])


class Episode:
    def __init__(
        self,
        bridge: S2Bridge,
        first: S2Request,
        task: TaskSpec | None,
        *,
        log_dir: Path | None = None,
        connection: int = 0,
        hold_waist: bool = True,
    ) -> None:
        self.bridge = bridge
        self.request = first
        self.request_at = time.time()
        self.first = first
        self.connection = connection
        self.embodiment: Embodiment = first.embodiment
        self.task = task
        self.runtime = task.runtime if task is not None else first.runtime
        if task is not None and task.runtime != first.runtime and first.runtime != "isaac":
            logger.warning("runtime from the request (%s) differs from the task's (%s)", first.runtime, task.runtime)
        self.kin = S2Kinematics(self.embodiment)
        self.budget = BudgetTracker(task) if task is not None else None
        self.row_dt = task.row_dt_s if task is not None else 1.0 / 60.0
        self.max_rows = task.max_rows_per_reply if task is not None else None
        self.ended = False
        self.outcome = EpisodeOutcome()
        self.started_s = time.time()
        self.log_dir = log_dir
        self.events: list[dict[str, Any]] = []
        self.tracking: list[dict[str, Any]] = []
        self.cmd_arm = first.arm.copy()
        self.cmd_effector = first.effector.copy()
        self.cmd_waist: np.ndarray | None = first.waist.copy() if first.waist.size else None
        self.hold_waist = bool(hold_waist and self.runtime == "isaac")
        self.send_waist = bool(self.cmd_waist is not None and ((task is not None and task.wbc) or self.hold_waist))
        self.send_wheels = bool(task is not None and task.wbc)
        self.odom = np.zeros(3)
        self._models: dict[str, CameraModel] = {}
        self._mounts: dict[str, np.ndarray] = {}
        self.contract = first.action_contract
        self._learn_cameras(first)
        if bridge is not None:
            bridge.hold_provider = self._keepalive_result

    def positions(self, request: S2Request | None = None) -> dict[str, float]:
        r = request or self.request
        hands = None
        if self.embodiment.is_dexhand:
            hands = {side: r.effector_values(side) for side in ARMS}
        return self.kin.joint_map(arm=r.arm, waist=r.waist if r.waist.size else None, head=r.head if r.head.size else None, hands=hands)

    def commanded_positions(self) -> dict[str, float]:
        positions = self.positions()
        positions.update(self.kin.joint_map(arm=self.cmd_arm))
        if self.cmd_waist is not None and self.cmd_waist.size:
            positions.update(self.kin.joint_map(waist=self.cmd_waist))
        return positions

    def tracking_error(self, request: S2Request) -> dict[str, Any]:
        e = self.embodiment
        dof = e.arm_dof
        arm = {}
        for index, side in enumerate(ARMS):
            if side in e.active_arms:
                diff = request.arm[index * dof : (index + 1) * dof] - self.cmd_arm[index * dof : (index + 1) * dof]
                arm[side] = round(float(np.max(np.abs(diff))), 4)
        effector = round(float(np.max(np.abs(request.effector - self.cmd_effector))), 4) if request.effector.size else 0.0
        out: dict[str, Any] = {"track_arm_rad": arm, "track_effector": effector}
        if self.cmd_waist is not None and request.waist.size == self.cmd_waist.size:
            out["track_waist"] = round(float(np.max(np.abs(request.waist - self.cmd_waist))), 4)
        return out

    def end_pose_check(self, request: S2Request | None = None) -> dict[str, Any] | None:
        r = request or self.request
        if not r.end_pose:
            return None
        positions = self.positions(r)
        out: dict[str, Any] = {}
        arm_frame = self.kin.arm_frame_pose(positions)
        for frame, block in r.end_pose.items():
            for side, reported in block.items():
                fk = self.kin.flange_pose(side, positions)
                if frame == "arm_base_link":
                    fk = invert(arm_frame) @ fk
                err = float(np.linalg.norm(fk[:3, 3] - reported[:3, 3])) * 1000.0
                cosine = (float(np.trace(fk[:3, :3].T @ reported[:3, :3])) - 1.0) / 2.0
                out[f"{frame}.{side}"] = {
                    "pos_mm": round(err, 4),
                    "rot_deg": round(math.degrees(math.acos(max(-1.0, min(1.0, cosine)))), 4),
                }
        return out

    def _learn_cameras(self, request: S2Request) -> None:
        positions = None
        for name, image in request.images.items():
            if not image.calibration_available:
                continue
            try:
                model = model_from_calibration(image.calibration)
                pose = pose_from_calibration(image.calibration)
            except (KeyError, ValueError):
                continue
            self._models[name] = model
            if positions is None:
                positions = self.positions(request)
            link = self.embodiment.camera_links[name]
            self._mounts[name] = invert(self.kin.link_pose(link, positions)) @ pose

    def camera(self, name: str, request: S2Request | None = None) -> CameraView:
        if name not in CAMERAS:
            raise KeyError(f"unknown camera {name!r}")
        r = request or self.request
        image = r.images[name]
        depth = r.depth_m.get(name)
        if image.calibration_available:
            view = CameraView(
                name, image.rgb, model_from_calibration(image.calibration),
                pose_from_calibration(image.calibration), depth, "wire", image.snapshot_id,
            )
            return view
        fisheye = self.embodiment.fisheye_cameras.get(name)
        positions = self.positions(r)
        if fisheye is not None and (image.width, image.height) == (int(fisheye["width"]), int(fisheye["height"])):
            link_pose = self.kin.link_pose(fisheye["parent_link"], positions)
            return CameraView(
                name, image.rgb, fisheye_model(fisheye), link_pose @ np.asarray(fisheye["T_parent_optical"]),
                depth, "usd", image.snapshot_id,
            )
        native = NATIVE_CAMERAS.get(self.runtime, {}).get(name)
        if native is not None and (image.width, image.height) == (int(native["width"]), int(native["height"])):
            link_pose = self.kin.link_pose(native["parent_link"], positions)
            return CameraView(
                name, image.rgb, model_from_spec(native), link_pose @ np.asarray(native["T_parent_optical"]),
                depth, "native_calibration", image.snapshot_id,
            )
        model = self._models.get(name)
        mount = self._mounts.get(name)
        if model is not None and mount is not None and (model.width, model.height) == (image.width, image.height):
            link_pose = self.kin.link_pose(self.embodiment.camera_links[name], positions)
            return CameraView(name, image.rgb, model, link_pose @ mount, depth, "fk", image.snapshot_id)
        return CameraView(name, image.rgb, None, None, depth, "none", image.snapshot_id)

    def make_chunk(self, arm: np.ndarray, effector: np.ndarray, *, waist: np.ndarray | None = None, wheels: np.ndarray | None = None) -> Chunk:
        rows = arm.shape[0]
        if waist is None and self.send_waist and self.cmd_waist is not None:
            waist = np.tile(self.cmd_waist, (rows, 1))
        if wheels is None and self.send_wheels:
            wheels = np.zeros((rows, 3))
        return Chunk(arm=arm, effector=effector, waist=waist, wheels=wheels)

    def hold_chunk(self, rows: int) -> Chunk:
        rows = max(1, int(rows))
        return self.make_chunk(np.tile(self.cmd_arm, (rows, 1)), np.tile(self.cmd_effector, (rows, 1)))

    def dispatch(self, chunk: Chunk, *, need_depth: bool = False, label: str = "", kind: str = "motion") -> S2Request:
        if self.ended:
            raise EpisodeEnded(self.outcome.end_reason)
        if self.max_rows is not None and chunk.rows > self.max_rows:
            pieces = int(math.ceil(chunk.rows / self.max_rows))
            logger.warning("%s: %d rows > Rubik cap %d, split into %d dispatches", label, chunk.rows, self.max_rows, pieces)
            for index in range(pieces):
                part = slice(index * self.max_rows, min(chunk.rows, (index + 1) * self.max_rows))
                sub = Chunk(
                    arm=chunk.arm[part], effector=chunk.effector[part],
                    waist=None if chunk.waist is None else chunk.waist[part],
                    wheels=None if chunk.wheels is None else chunk.wheels[part],
                )
                self.dispatch(sub, need_depth=need_depth and index == pieces - 1, label=f"{label}[{index}]", kind=kind)
            return self.request
        result = encode_chunk(chunk, self.embodiment, self.runtime, need_depth=need_depth, contract=self.contract)
        started = time.time()
        while True:
            try:
                size = self.bridge.reply(self.request, result)
                break
            except Superseded:
                self._absorb_keepalive()
        try:
            if self.budget is not None:
                self.budget.charge(chunk.rows, label=label, kind=kind, reply_bytes=size)
            if chunk.eef is None:
                self.cmd_arm = np.asarray(chunk.arm[-1], dtype=np.float64).copy()
            self.cmd_effector = np.asarray(chunk.effector[-1], dtype=np.float64).copy()
            if chunk.waist is not None:
                self.cmd_waist = np.asarray(chunk.waist[-1], dtype=np.float64).copy()
                self.send_waist = True
            if chunk.wheels is not None:
                self.odom = compose_planar(self.odom, chunk.wheels[-1])
            self._event("dispatch", label=label, kind=kind, rows=chunk.rows, need_depth=need_depth, bytes=size,
                        phase=None if self.budget is None else self.budget.phase)
        except Exception:
            logger.exception("dispatch bookkeeping failed (reply already sent)")
        self._await_next(chunk.rows, started)
        if chunk.eef is not None:
            self.cmd_arm = self.request.arm.copy()
        return self.request

    def _keepalive_result(self, request: S2Request) -> dict[str, Any]:
        return encode_chunk(self.hold_chunk(1), self.embodiment, self.runtime, need_depth=False, contract=self.contract)

    def _absorb_keepalive(self) -> None:
        while True:
            event = self.bridge.next_event(timeout=WAIT_BASE_S)
            if event is None:
                self._finish("timeout")
                raise EpisodeEnded("no request after a keepalive")
            if event.kind == "keepalive":
                if self.budget is not None:
                    self.budget.charge(1, label="keepalive", kind="keepalive")
                self._event("keepalive", seq=event.request.sequence if event.request else None)
                continue
            if event.kind == "request":
                self.request = event.request
                self.request_at = time.time()
                self._learn_cameras(event.request)
                return
            if event.kind == "episode_done":
                self.outcome.final_request = event.request
                self._finish("episode_done")
                raise EpisodeEnded("episode done")
            if event.kind == "disconnect":
                self._finish("disconnect")
                raise EpisodeEnded("client disconnected")

    def _await_next(self, rows: int, started: float) -> None:
        deadline = time.time() + WAIT_BASE_S + WAIT_PER_ROW_S * rows
        while True:
            event = self.bridge.next_event(timeout=max(0.1, deadline - time.time()))
            if event is None:
                if self.bridge.connected and time.time() < deadline + 600.0:
                    continue
                self._finish("timeout")
                raise EpisodeEnded("no request after the reply")
            if event.kind == "request":
                request = event.request
                assert request is not None
                if self._starts_new_episode(request):
                    self.outcome.carry_over = event
                    self._finish("next_episode_started")
                    raise EpisodeEnded("next episode started")
                self.request = request
                self.request_at = time.time()
                self._learn_cameras(request)
                tracking = self.tracking_error(request)
                self.tracking.append(tracking)
                self._event("request", seq=request.sequence, wait_s=round(time.time() - started, 3), depth=request.has_depth, **tracking)
                return
            if event.kind == "keepalive":
                if self.budget is not None:
                    self.budget.charge(1, label="keepalive", kind="keepalive")
                self._event("keepalive", seq=event.request.sequence if event.request else None)
                continue
            if event.kind == "episode_done":
                self.outcome.final_request = event.request
                self._finish("episode_done")
                raise EpisodeEnded("episode done")
            if event.kind == "disconnect":
                if self.runtime in ("newton", "rubik", "libuipc"):
                    self._finish("disconnect")
                    raise EpisodeEnded("client disconnected")
                continue

    def _starts_new_episode(self, request: S2Request) -> bool:
        if request.episode_start is True and request.sequence != self.first.sequence:
            return True
        if request.task_name != self.first.task_name or request.episode_idx != self.first.episode_idx:
            return True
        return False

    def _finish(self, reason: str) -> None:
        if not self.ended:
            self.ended = True
            self.outcome.end_reason = reason
            self._event("ended", reason=reason)

    def end(self, *, wait_s: float = 60.0) -> EpisodeOutcome:
        if self.ended:
            return self.outcome
        if self.runtime in ("isaac", "libuipc"):
            dev_rows = _dev_fast_end_rows() if self.runtime == "isaac" else 0
            if dev_rows:
                return self._dev_fast_end(dev_rows, wait_s=wait_s)
            rows = self.budget.end_hold_rows() if self.budget is not None else 1500
            try:
                self.dispatch(self.hold_chunk(rows), label="end_hold", kind="end_hold")
            except EpisodeEnded:
                return self.outcome
            for _ in range(3):
                try:
                    self.dispatch(self.hold_chunk(rows), label="end_hold_extra", kind="end_hold")
                except EpisodeEnded:
                    return self.outcome
            self._finish("end_hold_exhausted")
            return self.outcome
        result = terminal_reply(self.runtime)
        while True:
            try:
                self.bridge.reply(self.request, result)
                break
            except Superseded:
                try:
                    self._absorb_keepalive()
                except EpisodeEnded:
                    return self.outcome
        if self.budget is not None:
            self.budget.note_uncounted()
        self._event("terminal_reply", runtime=self.runtime)
        deadline = time.time() + wait_s
        while time.time() < deadline:
            event = self.bridge.next_event(timeout=1.0)
            if event is None:
                continue
            if event.kind == "request":
                self.outcome.carry_over = event
                break
            if event.kind in ("episode_done", "disconnect"):
                if event.kind == "episode_done":
                    self.outcome.final_request = event.request
                break
        self._finish("terminal_reply")
        return self.outcome

    def _dev_fast_end(self, rows: int, *, wait_s: float) -> EpisodeOutcome:
        try:
            self.dispatch(self.hold_chunk(rows), label="end_hold_dev", kind="end_hold")
        except EpisodeEnded:
            return self.outcome
        result = {"episode_done": True, "agent2act": {
            "protocol_version": 1, "status": "planning_exhausted", "termination_reason": "no_valid_hold",
            "action_issued": False, "episode_terminal": True,
            "error": "fast episode end after the program finished"}}
        while True:
            try:
                self.bridge.reply(self.request, result)
                break
            except Superseded:
                try:
                    self._absorb_keepalive()
                except EpisodeEnded:
                    return self.outcome
        if self.budget is not None:
            self.budget.note_uncounted()
        self._event("dev_fast_end", rows=rows)
        deadline = time.time() + wait_s
        while time.time() < deadline:
            event = self.bridge.next_event(timeout=1.0)
            if event is None:
                continue
            if event.kind == "request":
                self.outcome.carry_over = event
                break
            if event.kind in ("episode_done", "disconnect"):
                if event.kind == "episode_done":
                    self.outcome.final_request = event.request
                break
        self._finish("dev_fast_end")
        return self.outcome

    def _event(self, event_name: str, **data: Any) -> None:
        record = {"t": round(time.time() - self.started_s, 3), "event": event_name, **data}
        self.events.append(record)
        if self.log_dir is not None:
            try:
                with (self.log_dir / "episode_events.jsonl").open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(record, default=str) + "\n")
            except OSError:
                pass

    def summary(self) -> dict[str, Any]:
        return {
            "task": None if self.task is None else self.task.name,
            "wire_task_name": self.first.task_name,
            "embodiment": self.embodiment.key,
            "runtime": self.runtime,
            "episode_idx": self.first.episode_idx,
            "prompt": self.first.prompt,
            "end_reason": self.outcome.end_reason,
            "wall_s": round(time.time() - self.started_s, 3),
            "budget": None if self.budget is None else self.budget.status(),
            "end_pose_check_first": self.end_pose_check(self.first),
            "odom": [round(float(v), 4) for v in self.odom],
            "tracking": self.tracking,
        }


def _dev_fast_end_rows() -> int:
    import os

    try:
        return max(0, int(os.environ.get("CAP_HARNESS_S2_DEV_FAST_END", "0") or 0))
    except ValueError:
        return 0


__all__ = ["Episode", "EpisodeEnded", "EpisodeOutcome", "compose_planar"]
