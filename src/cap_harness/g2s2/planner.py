# Copyright (c) 2026 FF EAI Robotics Ecosystem Inc. All rights reserved. Review use only; see LICENSE-FF.txt.

from __future__ import annotations

from collections.abc import Mapping
import os
import time
from typing import Any

import numpy as np
import requests

from cap_harness.g2s2.embodiment import ARMS
from cap_harness.g2s2.wire import matrix_to_xyzw
from cap_harness.providers.wire import decode_numpy, encode_numpy

DEFAULT_URL = os.environ.get("CAP_HARNESS_S2_CUROBO_URL", "http://127.0.0.1:8250")


class PlanningError(RuntimeError):
    pass


def _quat_wxyz(matrix: np.ndarray) -> list[float]:
    x, y, z, w = matrix_to_xyzw(matrix)
    return [w, x, y, z]


class CuroboPlanner:
    def __init__(self, url: str | None = None, *, timeout_s: float = 120.0) -> None:
        self.url = (url or DEFAULT_URL).rstrip("/")
        self.timeout_s = timeout_s
        self.session = requests.Session()
        self.calls = 0
        self.seconds = 0.0

    def _payload(self, episode: Any, arm_start: np.ndarray, scene_points: np.ndarray | None, voxel_m: float) -> dict[str, Any]:
        e = episode.embodiment
        dof = e.arm_dof
        joint_positions, joint_names, ee, transforms = {}, {}, {}, {}
        for side in e.active_arms:
            index = ARMS.index(side)
            joint_positions[side] = [float(v) for v in arm_start[index * dof : (index + 1) * dof]]
            joint_names[side] = list(e.arms[side].joint_names)
            arm = e.arms[side]
            ee[side] = arm.grasp_parent_link if e.is_dexhand else arm.tcp_link
            transforms[side] = np.eye(4).tolist()
        locks: dict[str, float] = {}
        request = episode.request
        if request.waist.size:
            locks.update(dict(zip(e.waist_joint_names, (float(v) for v in (episode.cmd_waist if episode.cmd_waist is not None else request.waist)))))
        if request.head.size:
            locks.update(dict(zip(e.head_joint_names, (float(v) for v in request.head))))
        for side in ARMS:
            if side not in e.active_arms:
                index = ARMS.index(side)
                locks.update(dict(zip(e.arms[side].joint_names, (float(v) for v in arm_start[index * dof : (index + 1) * dof]))))
        payload: dict[str, Any] = {
            "base_frame": e.base_frame,
            "model": "g2s2",
            "embodiment": e.key,
            "joint_positions": joint_positions,
            "joint_names": joint_names,
            "base_transforms": transforms,
            "end_effector_links": ee,
            "lock_joints": locks,
            "voxel_size_m": voxel_m,
        }
        if scene_points is not None and len(scene_points):
            payload["scene_points_base64"] = encode_numpy(np.asarray(scene_points, dtype=np.float32))
        return payload

    def plan(
        self,
        episode: Any,
        targets: Mapping[str, np.ndarray],
        arm_start: np.ndarray,
        *,
        scene_points: np.ndarray | None = None,
        voxel_m: float = 0.015,
    ) -> tuple[np.ndarray, float]:
        e = episode.embodiment
        payload = self._payload(episode, arm_start, scene_points, voxel_m)
        goals = {}
        for side, grasp in targets.items():
            e.check_active(side)
            arm = e.arms[side]
            tool = np.asarray(grasp, dtype=np.float64) @ np.linalg.inv(arm.grasp_from_parent)
            goals[side] = {"type": "pose", "position": [float(v) for v in tool[:3, 3]], "quaternion_wxyz": _quat_wxyz(tool), "frame": e.base_frame}
        started = time.time()
        if len(goals) == 1:
            side = next(iter(goals))
            response = self.session.post(f"{self.url}/plan", json={**payload, "arm": side, "target": goals[side]}, timeout=self.timeout_s)
        else:
            response = self.session.post(f"{self.url}/plan_synchronized", json={**payload, "targets": goals}, timeout=self.timeout_s)
        self.calls += 1
        self.seconds += time.time() - started
        if response.status_code != 200:
            raise PlanningError(f"cuRobo {response.status_code}: {str(response.text)[:600]}")
        body = response.json()
        dt = float(body["dt_s"])
        dof = e.arm_dof
        out = None
        if "joint_positions_base64" in body and isinstance(body["joint_positions_base64"], str):
            side = next(iter(goals))
            traj = decode_numpy(body["joint_positions_base64"])
            out = np.tile(arm_start, (traj.shape[0], 1))
            index = ARMS.index(side)
            out[:, index * dof : (index + 1) * dof] = traj
        else:
            parts = {side: decode_numpy(value) for side, value in body["joint_positions_base64"].items()}
            length = max(p.shape[0] for p in parts.values())
            out = np.tile(arm_start, (length, 1))
            for side, traj in parts.items():
                index = ARMS.index(side)
                padded = np.vstack([traj, np.tile(traj[-1], (length - traj.shape[0], 1))]) if traj.shape[0] < length else traj
                out[:, index * dof : (index + 1) * dof] = padded
        return out, dt


def resample(trajectory: np.ndarray, dt: float, row_dt: float, *, slow_down: float = 1.0) -> np.ndarray:
    duration = (trajectory.shape[0] - 1) * dt * slow_down
    rows = max(2, int(np.ceil(duration / row_dt)))
    source = np.linspace(0.0, trajectory.shape[0] - 1, rows + 1)[1:]
    lower = np.floor(source).astype(int)
    upper = np.minimum(lower + 1, trajectory.shape[0] - 1)
    frac = (source - lower)[:, None]
    return trajectory[lower] * (1.0 - frac) + trajectory[upper] * frac


__all__ = ["DEFAULT_URL", "CuroboPlanner", "PlanningError", "resample"]
