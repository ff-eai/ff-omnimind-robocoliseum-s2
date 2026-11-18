# Copyright (c) 2026 FF EAI Robotics Ecosystem Inc. All rights reserved. Review use only; see LICENSE-FF.txt.

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import math
from typing import Any

import numpy as np

from cap_harness.g2.kinematics import UrdfModel, invert, load_urdf
from cap_harness.g2s2.embodiment import ARMS, Embodiment


def rotation_error(current: np.ndarray, target: np.ndarray) -> np.ndarray:
    delta = target @ current.T
    cosine = max(-1.0, min(1.0, (float(np.trace(delta)) - 1.0) / 2.0))
    angle = math.acos(cosine)
    if angle < 1e-9:
        return np.zeros(3)
    if math.pi - angle < 1e-6:
        sym = (delta + np.eye(3)) / 2.0
        axis = np.sqrt(np.clip(np.diag(sym), 0.0, None))
        index = int(np.argmax(axis))
        column = sym[:, index] / max(axis[index], 1e-12)
        return column / np.linalg.norm(column) * angle
    axis = np.array([delta[2, 1] - delta[1, 2], delta[0, 2] - delta[2, 0], delta[1, 0] - delta[0, 1]])
    return axis / (2.0 * math.sin(angle)) * angle


def angle_between(a: np.ndarray, b: np.ndarray) -> float:
    ra, rb = np.asarray(a)[:3, :3], np.asarray(b)[:3, :3]
    return math.acos(max(-1.0, min(1.0, (float(np.trace(ra.T @ rb)) - 1.0) / 2.0)))


@dataclass(frozen=True, slots=True)
class IkResult:
    ok: bool
    joints: np.ndarray
    position_error_m: float
    rotation_error_rad: float
    iterations: int
    restarts: int
    message: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "joints": [round(float(v), 6) for v in self.joints],
            "position_error_mm": round(self.position_error_m * 1000.0, 3),
            "rotation_error_deg": round(math.degrees(self.rotation_error_rad), 3),
            "iterations": self.iterations,
            "restarts": self.restarts,
            "message": self.message,
        }


class _ArmChain:
    def __init__(self, model: UrdfModel, tip_link: str, arm_joints: Sequence[str]) -> None:
        self.model = model
        names = model.chain(tip_link)
        self.joints = [model.joints[name] for name in names]
        self.arm_columns = {name: index for index, name in enumerate(arm_joints)}
        missing = [name for name in arm_joints if name not in names]
        if missing:
            raise ValueError(f"{tip_link} chain lacks arm joints {missing}")

    def forward(self, q_arm: np.ndarray, others: Mapping[str, float]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        pose = np.eye(4)
        axes = np.zeros((len(self.arm_columns), 3))
        origins = np.zeros((len(self.arm_columns), 3))
        for joint in self.joints:
            pose = pose @ joint.origin
            if not joint.moves:
                continue
            column = self.arm_columns.get(joint.name)
            if column is not None:
                value = float(q_arm[column])
                axes[column] = pose[:3, :3] @ (joint.axis / np.linalg.norm(joint.axis))
                origins[column] = pose[:3, 3]
            else:
                value = self.model.joint_value(joint, others)
            pose = pose @ joint.motion(value)
        return pose, axes, origins


class S2Kinematics:
    def __init__(self, embodiment: Embodiment, model: UrdfModel | None = None) -> None:
        self.embodiment = embodiment
        self.model = model or load_urdf(str(embodiment.urdf_path))
        self.base_frame = embodiment.base_frame
        for side, arm in embodiment.arms.items():
            for link in (arm.flange_link, arm.tcp_link, arm.grasp_parent_link):
                if link not in self.model.links:
                    raise ValueError(f"{embodiment.key}: {side} link {link!r} not in {embodiment.urdf}")
        self._chains: dict[tuple[str, str], _ArmChain] = {}
        self._lower: dict[str, np.ndarray] = {}
        self._upper: dict[str, np.ndarray] = {}
        for side, arm in embodiment.arms.items():
            lower, upper = self.model.joint_limits(arm.joint_names)
            self._lower[side], self._upper[side] = lower, upper
        self._root_from_base = self.model.fk(self.base_frame, {})
        self._base_from_root = invert(self._root_from_base)

    def joint_map(
        self,
        *,
        arm: Sequence[float] | None = None,
        waist: Sequence[float] | None = None,
        head: Sequence[float] | None = None,
        hands: Mapping[str, Sequence[float]] | None = None,
    ) -> dict[str, float]:
        e = self.embodiment
        values: dict[str, float] = {}
        if arm is not None:
            vector = np.asarray(arm, dtype=np.float64).reshape(-1)
            if vector.shape[0] != e.wire_arm_width:
                raise ValueError(f"{e.key}: arm vector must have {e.wire_arm_width} values, got {vector.shape[0]}")
            for index, side in enumerate(ARMS):
                names = e.arms[side].joint_names
                values.update(zip(names, vector[index * e.arm_dof : (index + 1) * e.arm_dof].tolist()))
        if waist is not None and len(waist):
            values.update(zip(e.waist_joint_names, [float(v) for v in waist]))
        if head is not None and len(head):
            values.update(zip(e.head_joint_names, [float(v) for v in head]))
        if hands and e.is_dexhand:
            for side, vector in hands.items():
                values.update(zip(e.gripper.joint_names[side], [float(v) for v in vector]))
        return values

    def limits(self, side: str) -> tuple[np.ndarray, np.ndarray]:
        return self._lower[side].copy(), self._upper[side].copy()

    def link_pose(self, link: str, positions: Mapping[str, float]) -> np.ndarray:
        return self._base_from_root @ self.model.fk(link, positions)

    def flange_pose(self, side: str, positions: Mapping[str, float]) -> np.ndarray:
        return self.link_pose(self.embodiment.arm(side).flange_link, positions)

    def tcp_pose(self, side: str, positions: Mapping[str, float]) -> np.ndarray:
        return self.link_pose(self.embodiment.arm(side).tcp_link, positions)

    def grasp_pose(self, side: str, positions: Mapping[str, float]) -> np.ndarray:
        arm = self.embodiment.arm(side)
        return self.link_pose(arm.grasp_parent_link, positions) @ arm.grasp_from_parent

    def flange_from_grasp(self, side: str) -> np.ndarray:
        arm = self.embodiment.arm(side)
        flange = self.model.fk(arm.flange_link, {})
        parent = self.model.fk(arm.grasp_parent_link, {})
        return invert(flange) @ parent @ arm.grasp_from_parent

    def flange_from_tcp(self, side: str) -> np.ndarray:
        arm = self.embodiment.arm(side)
        return invert(self.model.fk(arm.flange_link, {})) @ self.model.fk(arm.tcp_link, {})

    def arm_frame_pose(self, positions: Mapping[str, float]) -> np.ndarray:
        return self.link_pose(self.embodiment.arm_frame, positions)

    def _chain(self, side: str, link: str) -> _ArmChain:
        key = (side, link)
        chain = self._chains.get(key)
        if chain is None:
            chain = _ArmChain(self.model, link, self.embodiment.arm(side).joint_names)
            self._chains[key] = chain
        return chain

    def solve(
        self,
        side: str,
        target: np.ndarray,
        seed: Sequence[float],
        positions: Mapping[str, float],
        *,
        frame: str = "grasp",
        position_tolerance_m: float = 1e-3,
        rotation_tolerance_rad: float = math.radians(0.5),
        orientation_weight: float = 0.3,
        max_iterations: int = 300,
        restarts: int = 6,
        rest_weight: float = 0.02,
        rng_seed: int = 0,
        free_axis: Sequence[float] | None = None,
    ) -> IkResult:
        e = self.embodiment
        arm = e.arm(side)
        if frame == "grasp":
            link, tip_offset = arm.grasp_parent_link, arm.grasp_from_parent
        elif frame == "tcp":
            link, tip_offset = arm.tcp_link, np.eye(4)
        elif frame == "flange":
            link, tip_offset = arm.flange_link, np.eye(4)
        else:
            raise ValueError("frame must be grasp, tcp or flange")
        chain = self._chain(side, link)
        target_root = self._root_from_base @ np.asarray(target, dtype=np.float64)
        lower, upper = self._lower[side], self._upper[side]
        seed_q = np.clip(np.asarray(seed, dtype=np.float64).reshape(e.arm_dof), lower, upper)
        axis_local = None if free_axis is None else np.asarray(free_axis, dtype=np.float64) / np.linalg.norm(free_axis)
        rng = np.random.default_rng(rng_seed)
        best: IkResult | None = None
        total_iterations = 0
        span = np.minimum(upper - lower, 2.0 * math.pi)
        for attempt in range(restarts + 1):
            if attempt == 0:
                q = seed_q.copy()
            elif attempt <= 2:
                q = np.clip(seed_q + rng.normal(0.0, 0.35, size=seed_q.shape) * span / 2.0, lower, upper)
            else:
                q = lower + rng.uniform(0.05, 0.95, size=seed_q.shape) * (upper - lower)
            result = self._dls(
                chain, tip_offset, target_root, q, positions, lower, upper, seed_q,
                position_tolerance_m, rotation_tolerance_rad, orientation_weight, max_iterations,
                rest_weight, axis_local,
            )
            total_iterations += result.iterations
            if best is None or _score(result) < _score(best):
                best = result
            if result.ok:
                break
        assert best is not None
        return IkResult(
            ok=best.ok,
            joints=best.joints,
            position_error_m=best.position_error_m,
            rotation_error_rad=best.rotation_error_rad,
            iterations=total_iterations,
            restarts=attempt,
            message=best.message,
        )

    def _dls(
        self,
        chain: _ArmChain,
        tip_offset: np.ndarray,
        target: np.ndarray,
        q: np.ndarray,
        positions: Mapping[str, float],
        lower: np.ndarray,
        upper: np.ndarray,
        rest: np.ndarray,
        position_tolerance_m: float,
        rotation_tolerance_rad: float,
        orientation_weight: float,
        max_iterations: int,
        rest_weight: float,
        axis_local: np.ndarray | None,
    ) -> IkResult:
        dof = q.shape[0]
        weights = np.array([1.0, 1.0, 1.0, orientation_weight, orientation_weight, orientation_weight])

        def evaluate(q_now: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, float]:
            pose, axes, origins = chain.forward(q_now, positions)
            tip = pose @ tip_offset
            p_err = target[:3, 3] - tip[:3, 3]
            if axis_local is None:
                r_err = rotation_error(tip[:3, :3], target[:3, :3])
            else:
                want = target[:3, :3] @ axis_local
                have = tip[:3, :3] @ axis_local
                cross = np.cross(have, want)
                sine = float(np.linalg.norm(cross))
                angle = math.atan2(sine, float(np.dot(have, want)))
                r_err = np.zeros(3) if sine < 1e-12 else cross / sine * angle
            jac = np.zeros((6, dof))
            for column in range(dof):
                z = axes[column]
                jac[:3, column] = np.cross(z, tip[:3, 3] - origins[column])
                jac[3:, column] = z
            return p_err, r_err, jac, tip, float(np.linalg.norm(p_err)), float(np.linalg.norm(r_err))

        damping = 0.01
        p_err, r_err, jac, _, position_error, rotation_error_norm = evaluate(q)
        cost = float(np.linalg.norm(np.concatenate([p_err, r_err]) * weights))
        eps = 1e-6
        for iteration in range(1, max_iterations + 1):
            if position_error <= position_tolerance_m and rotation_error_norm <= rotation_tolerance_rad:
                return IkResult(True, q, position_error, rotation_error_norm, iteration, 0)
            error = np.concatenate([p_err, r_err]) * weights
            norm = float(np.linalg.norm(error))
            if norm > 0.15:
                error = error * (0.15 / norm)
            jw = jac * weights[:, None]
            free = np.ones(dof, dtype=bool)
            dq = np.zeros(dof)
            for _ in range(3):
                jf = jw[:, free]
                jjt = jf @ jf.T + (damping**2) * np.eye(6)
                step = jf.T @ np.linalg.solve(jjt, error)
                if rest_weight > 0.0:
                    null = np.eye(int(free.sum())) - jf.T @ np.linalg.solve(jjt, jf)
                    step = step + null @ ((rest - q)[free] * rest_weight)
                dq = np.zeros(dof)
                dq[free] = step
                blocked = ((q <= lower + eps) & (dq < 0.0)) | ((q >= upper - eps) & (dq > 0.0))
                if not np.any(blocked & free):
                    break
                free &= ~blocked
                if not free.any():
                    break
            biggest = float(np.max(np.abs(dq))) if dq.size else 0.0
            if biggest > 0.25:
                dq = dq * (0.25 / biggest)
            q_new = np.clip(q + dq, lower, upper)
            p_new, r_new, jac_new, _, pos_new, rot_new = evaluate(q_new)
            cost_new = float(np.linalg.norm(np.concatenate([p_new, r_new]) * weights))
            if cost_new < cost or biggest < 1e-9:
                q, p_err, r_err, jac = q_new, p_new, r_new, jac_new
                position_error, rotation_error_norm, cost = pos_new, rot_new, cost_new
                damping = max(1e-3, damping * 0.5)
            else:
                damping = min(1.0, damping * 4.0)
                if damping >= 1.0 and cost - cost_new < 1e-9:
                    break
        return IkResult(
            False, q, position_error, rotation_error_norm, max_iterations, 0,
            message=f"no convergence: {position_error * 1000:.1f} mm, {math.degrees(rotation_error_norm):.2f} deg",
        )


def _score(result: IkResult) -> float:
    return result.position_error_m + 0.05 * result.rotation_error_rad


__all__ = ["IkResult", "S2Kinematics", "angle_between", "rotation_error"]
