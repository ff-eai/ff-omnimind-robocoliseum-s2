# Copyright (c) 2026 FF EAI Robotics Ecosystem Inc. All rights reserved. Review use only; see LICENSE-FF.txt.

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
import time
from typing import Any

import numpy as np

from cap_harness.g2.codec import decode_jpeg
from cap_harness.g2s2.embodiment import ARMS, CAMERAS, Embodiment, embodiment_for

RUNTIMES = ("isaac", "newton", "rubik", "libuipc")
RUBIK_MAX_ROWS_DEFAULT = 256
DEPTH_KEYS = {camera: f"{camera}_depth" for camera in CAMERAS}


def _vector(value: Any) -> np.ndarray:
    if value is None:
        return np.zeros(0)
    array = np.asarray(list(value), dtype=np.float64).reshape(-1)
    if not np.all(np.isfinite(array)):
        raise ValueError("state vector must be finite")
    return array


def _pose_xyzw(entry: Mapping[str, Any]) -> np.ndarray:
    position = np.asarray(entry["position"], dtype=np.float64).reshape(3)
    x, y, z, w = (float(v) for v in np.asarray(entry["orientation"], dtype=np.float64).reshape(4))
    norm = (x * x + y * y + z * z + w * w) ** 0.5
    x, y, z, w = x / norm, y / norm, z / norm, w / norm
    out = np.eye(4)
    out[:3, :3] = [
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ]
    out[:3, 3] = position
    return out


def matrix_to_xyzw(matrix: np.ndarray) -> list[float]:
    r = np.asarray(matrix, dtype=np.float64)[:3, :3]
    trace = float(np.trace(r))
    if trace > 0.0:
        s = (trace + 1.0) ** 0.5 * 2.0
        w, x, y, z = 0.25 * s, (r[2, 1] - r[1, 2]) / s, (r[0, 2] - r[2, 0]) / s, (r[1, 0] - r[0, 1]) / s
    elif r[0, 0] > r[1, 1] and r[0, 0] > r[2, 2]:
        s = (1.0 + r[0, 0] - r[1, 1] - r[2, 2]) ** 0.5 * 2.0
        w, x, y, z = (r[2, 1] - r[1, 2]) / s, 0.25 * s, (r[0, 1] + r[1, 0]) / s, (r[0, 2] + r[2, 0]) / s
    elif r[1, 1] > r[2, 2]:
        s = (1.0 + r[1, 1] - r[0, 0] - r[2, 2]) ** 0.5 * 2.0
        w, x, y, z = (r[0, 2] - r[2, 0]) / s, (r[0, 1] + r[1, 0]) / s, 0.25 * s, (r[1, 2] + r[2, 1]) / s
    else:
        s = (1.0 + r[2, 2] - r[0, 0] - r[1, 1]) ** 0.5 * 2.0
        w, x, y, z = (r[1, 0] - r[0, 1]) / s, (r[0, 2] + r[2, 0]) / s, (r[1, 2] + r[2, 1]) / s, 0.25 * s
    q = np.array([x, y, z, w])
    q = q / np.linalg.norm(q)
    if q[3] < 0:
        q = -q
    return [float(v) for v in q]


@dataclass(slots=True)
class WireImage:
    camera: str
    jpeg: bytes
    height: int
    width: int
    calibration: Mapping[str, Any] | None
    snapshot_id: str | None
    _rgb: np.ndarray | None = None

    @property
    def rgb(self) -> np.ndarray:
        if self._rgb is None:
            rgb = decode_jpeg(self.jpeg)
            if rgb.shape[:2] != (self.height, self.width):
                raise ValueError(
                    f"{self.camera}: JPEG is {rgb.shape[1]}x{rgb.shape[0]}, record says {self.width}x{self.height}"
                )
            self._rgb = rgb
        return self._rgb

    @property
    def calibration_available(self) -> bool:
        return bool(self.calibration and self.calibration.get("available"))


@dataclass(slots=True)
class S2Request:
    sequence: int
    received_at_s: float
    nbytes: int
    embodiment: Embodiment
    runtime: str
    task_name: str
    prompt: str
    segments: tuple[str, ...]
    episode_idx: int
    episode_done: bool
    episode_start: bool | None
    arm: np.ndarray
    effector: np.ndarray
    waist: np.ndarray
    head: np.ndarray
    chassis: np.ndarray | None
    end_pose: dict[str, dict[str, np.ndarray]] | None
    images: dict[str, WireImage]
    depth_m: dict[str, np.ndarray]
    simulation_time: float | None = None
    action_contract: Mapping[str, Any] | None = None
    task_progress: tuple[Mapping[str, Any], ...] = ()
    raw_keys: tuple[str, ...] = ()
    timestamps: Mapping[str, int] = field(default_factory=dict)

    def arm_joints(self, side: str) -> np.ndarray:
        dof = self.embodiment.arm_dof
        index = ARMS.index(side)
        return self.arm[index * dof : (index + 1) * dof].copy()

    def effector_values(self, side: str) -> np.ndarray:
        width = self.embodiment.gripper.width
        index = ARMS.index(side)
        return self.effector[index * width : (index + 1) * width].copy()

    def gripper_opening(self, side: str) -> float:
        return self.embodiment.gripper.opening_from_wire(float(self.effector_values(side)[0]))

    @property
    def has_depth(self) -> bool:
        return bool(self.depth_m)

    @property
    def summary(self) -> dict[str, Any]:
        return {
            "sequence": self.sequence,
            "runtime": self.runtime,
            "embodiment": self.embodiment.key,
            "task_name": self.task_name,
            "episode_idx": self.episode_idx,
            "episode_done": self.episode_done,
            "episode_start": self.episode_start,
            "bytes": self.nbytes,
            "depth": sorted(self.depth_m),
            "images": {name: f"{image.width}x{image.height}" for name, image in self.images.items()},
            "simulation_time": self.simulation_time,
        }


def detect_runtime(params: Mapping[str, Any], embodiment: Embodiment) -> str:
    if "action_contract" in params:
        return "rubik"
    if "task_progress" in params:
        return "libuipc"
    states = params.get("states") or {}
    images = params.get("images") or {}
    head = images.get("head") or {}
    if embodiment.is_dexhand and states.get("end_pose") is None and "calibration" not in head and "head_depth" not in images:
        return "newton"
    return "isaac"


def decode_request(
    params: Mapping[str, Any],
    *,
    sequence: int = 0,
    nbytes: int = 0,
    runtime: str | None = None,
    decode_depth: bool = True,
) -> S2Request:
    if not isinstance(params, Mapping):
        raise ValueError("infer params must be a mapping")
    embodiment = embodiment_for(str(params.get("robot_type", "")), str(params.get("ee_type", "")))
    states = params.get("states") or {}
    if not isinstance(states, Mapping):
        raise ValueError("states must be a mapping")
    arm = _vector(states.get("arm_joint_states"))
    if arm.shape[0] != embodiment.wire_arm_width:
        raise ValueError(f"{embodiment.key}: arm_joint_states has {arm.shape[0]} values, expected {embodiment.wire_arm_width}")
    effector = _vector(states.get("gripper_states"))
    if effector.shape[0] != embodiment.wire_effector_width:
        raise ValueError(
            f"{embodiment.key}: gripper_states has {effector.shape[0]} values, expected {embodiment.wire_effector_width}"
        )
    waist = _vector(states.get("waist_joint_states"))
    head = _vector(states.get("head_joint_states"))
    if waist.shape[0] not in (0, len(embodiment.waist_joint_names)):
        raise ValueError(f"{embodiment.key}: waist has {waist.shape[0]} values")
    if head.shape[0] not in (0, len(embodiment.head_joint_names)):
        raise ValueError(f"{embodiment.key}: head has {head.shape[0]} values")
    chassis = states.get("chassis_states")
    chassis_array = None if chassis is None else _vector(chassis)

    end_pose = None
    raw_end = states.get("end_pose")
    if isinstance(raw_end, Mapping):
        end_pose = {}
        for frame in ("base_link", "arm_base_link"):
            block = raw_end.get(frame)
            if isinstance(block, Mapping):
                end_pose[frame] = {
                    side: _pose_xyzw(block[f"{side}_arm"]) for side in ARMS if isinstance(block.get(f"{side}_arm"), Mapping)
                }

    raw_images = params.get("images") or {}
    images: dict[str, WireImage] = {}
    depth: dict[str, np.ndarray] = {}
    for camera in CAMERAS:
        record = raw_images.get(camera)
        if not isinstance(record, Mapping):
            raise ValueError(f"images lack the {camera!r} record")
        if str(record.get("encoding", "JPEG")).upper() != "JPEG":
            raise ValueError(f"{camera}: unsupported RGB encoding {record.get('encoding')!r}")
        calibration = record.get("calibration")
        images[camera] = WireImage(
            camera=camera,
            jpeg=bytes(record["image_data"]),
            height=int(record["height"]),
            width=int(record["width"]),
            calibration=dict(calibration) if isinstance(calibration, Mapping) else None,
            snapshot_id=record.get("snapshot_id"),
        )
        depth_record = raw_images.get(DEPTH_KEYS[camera])
        if decode_depth and isinstance(depth_record, Mapping):
            if str(depth_record.get("encoding", "RAW_UINT16")).upper() != "RAW_UINT16":
                raise ValueError(f"{camera}: unsupported depth encoding")
            h, w = int(depth_record["height"]), int(depth_record["width"])
            raw = np.frombuffer(bytes(depth_record["image_data"]), dtype="<u2")
            if raw.size != h * w:
                raise ValueError(f"{camera} depth has {raw.size} values for {w}x{h}")
            depth[camera] = raw.reshape(h, w).astype(np.float64) / 1000.0
    detected = detect_runtime(params, embodiment) if runtime is None else runtime
    progress = params.get("task_progress") or ()
    contract = params.get("action_contract")
    timestamps = params.get("timestamps") or {}
    episode_start = params.get("episode_start")
    return S2Request(
        sequence=int(sequence),
        received_at_s=time.time(),
        nbytes=int(nbytes),
        embodiment=embodiment,
        runtime=detected,
        task_name=str(params.get("task_name", "") or ""),
        prompt=str(params.get("prompt", "") or ""),
        segments=tuple(str(s) for s in (params.get("segments") or ())),
        episode_idx=int(params.get("episode_idx", 0) or 0),
        episode_done=bool(params.get("episode_done", False)),
        episode_start=None if episode_start is None else bool(episode_start),
        arm=arm,
        effector=effector,
        waist=waist,
        head=head,
        chassis=chassis_array,
        end_pose=end_pose,
        images=images,
        depth_m=depth,
        simulation_time=None if params.get("simulation_time") is None else float(params["simulation_time"]),
        action_contract=dict(contract) if isinstance(contract, Mapping) else None,
        task_progress=tuple(dict(item) for item in progress if isinstance(item, Mapping)),
        raw_keys=tuple(sorted(str(k) for k in params)),
        timestamps={str(k): int(v) for k, v in timestamps.items()} if isinstance(timestamps, Mapping) else {},
    )


def _rows(values: np.ndarray) -> list[list[float]]:
    return [[float(v) for v in row] for row in np.asarray(values, dtype=np.float64)]


@dataclass(slots=True)
class Chunk:
    arm: np.ndarray
    effector: np.ndarray
    waist: np.ndarray | None = None
    wheels: np.ndarray | None = None
    eef: dict[str, np.ndarray] | None = None
    eef_frame: str = "base_link"

    @property
    def rows(self) -> int:
        return int(self.effector.shape[0])

    def validate(self, embodiment: Embodiment) -> None:
        h = self.rows
        if h <= 0:
            raise ValueError("chunk must have at least one row")
        if self.eef is None:
            if self.arm.shape != (h, embodiment.wire_arm_width):
                raise ValueError(f"arm chunk {self.arm.shape}, expected ({h}, {embodiment.wire_arm_width})")
        if self.effector.shape != (h, embodiment.wire_effector_width):
            raise ValueError(f"effector chunk {self.effector.shape}, expected ({h}, {embodiment.wire_effector_width})")
        if self.waist is not None and self.waist.shape != (h, len(embodiment.waist_joint_names)):
            raise ValueError(f"waist chunk {self.waist.shape}")
        if self.wheels is not None and self.wheels.shape != (h, 3):
            raise ValueError(f"wheels chunk {self.wheels.shape}")
        for block in (self.arm, self.effector, self.waist, self.wheels):
            if block is not None and not np.all(np.isfinite(block)):
                raise ValueError("chunk values must be finite")


def encode_isaac(chunk: Chunk, embodiment: Embodiment, *, need_depth: bool = False) -> dict[str, Any]:
    chunk.validate(embodiment)
    dof, width = embodiment.arm_dof, embodiment.gripper.width
    result: dict[str, Any] = {}
    if chunk.eef is not None:
        frame = chunk.eef_frame
        if frame not in ("base_link", "arm_base_link"):
            raise ValueError("EEF frame must be base_link or arm_base_link")
        for side in ARMS:
            values = np.asarray(chunk.eef[side], dtype=np.float64)
            if values.shape != (chunk.rows, 7):
                raise ValueError(f"EEF {side} chunk {values.shape}")
            result[f"{side}_arm"] = {"kind": "EEF_ABS", "values": _rows(values), "base_link": frame}
        result["base_link"] = frame
    else:
        result["left_arm"] = {"kind": "JOINT_ABS", "values": _rows(chunk.arm[:, :dof])}
        result["right_arm"] = {"kind": "JOINT_ABS", "values": _rows(chunk.arm[:, dof:])}
    result["left_effector"] = _rows(chunk.effector[:, :width])
    result["right_effector"] = _rows(chunk.effector[:, width:])
    if chunk.waist is not None:
        result["waist"] = {"kind": "JOINT_ABS", "values": _rows(chunk.waist)}
    if chunk.wheels is not None:
        result["wheels"] = {"kind": "POSE_DELTA", "values": _rows(chunk.wheels)}
    result["need_depth"] = bool(need_depth)
    return result


def encode_newton(chunk: Chunk, embodiment: Embodiment) -> dict[str, Any]:
    chunk.validate(embodiment)
    if chunk.wheels is not None:
        raise ValueError("Newton rejects wheels")
    if chunk.eef is not None:
        raise ValueError("only joint targets are sent to Newton")
    dof, width = embodiment.arm_dof, embodiment.gripper.width
    return {
        "left_arm": {"kind": "JOINT_ABS", "values": _rows(chunk.arm[:, :dof])},
        "right_arm": {"kind": "JOINT_ABS", "values": _rows(chunk.arm[:, dof:])},
        "left_effector": _rows(chunk.effector[:, :width]),
        "right_effector": _rows(chunk.effector[:, width:]),
    }


def rubik_bounds(contract: Mapping[str, Any] | None) -> np.ndarray | None:
    if not contract:
        return None
    bounds = np.asarray(contract.get("bounds") or [], dtype=np.float64)
    return bounds if bounds.shape == (54, 2) else None


def encode_rubik(
    chunk: Chunk,
    embodiment: Embodiment,
    contract: Mapping[str, Any] | None,
    *,
    max_rows: int = RUBIK_MAX_ROWS_DEFAULT,
) -> dict[str, Any]:
    chunk.validate(embodiment)
    if chunk.wheels is not None or chunk.waist is not None:
        raise ValueError("Rubik rejects waist / wheels")
    if chunk.eef is not None:
        raise ValueError("only joint targets are sent to Rubik")
    if chunk.rows > max_rows:
        raise ValueError(f"Rubik accepts at most {max_rows} rows per reply, got {chunk.rows}")
    targets = np.concatenate([chunk.arm, chunk.effector], axis=1)
    bounds = rubik_bounds(contract)
    if bounds is not None:
        targets = np.clip(targets, bounds[:, 0], bounds[:, 1])
    dof, width = embodiment.arm_dof, embodiment.gripper.width
    return {
        "left_arm": {"kind": "JOINT_ABS", "values": _rows(targets[:, :dof])},
        "right_arm": {"kind": "JOINT_ABS", "values": _rows(targets[:, dof : 2 * dof])},
        "left_effector": _rows(targets[:, 2 * dof : 2 * dof + width]),
        "right_effector": _rows(targets[:, 2 * dof + width :]),
    }


def encode_chunk(
    chunk: Chunk,
    embodiment: Embodiment,
    runtime: str,
    *,
    need_depth: bool = False,
    contract: Mapping[str, Any] | None = None,
    rubik_max_rows: int = RUBIK_MAX_ROWS_DEFAULT,
) -> dict[str, Any]:
    if runtime == "isaac":
        return encode_isaac(chunk, embodiment, need_depth=need_depth)
    if runtime == "newton":
        return encode_newton(chunk, embodiment)
    if runtime == "rubik":
        return encode_rubik(chunk, embodiment, contract, max_rows=rubik_max_rows)
    if runtime == "libuipc":
        return encode_newton(chunk, embodiment)
    raise ValueError(f"unknown runtime {runtime!r}")


def hold_chunk(request: S2Request, rows: int = 1, *, include_waist: bool = False) -> Chunk:
    rows = max(1, int(rows))
    arm = np.tile(request.arm, (rows, 1))
    effector = np.tile(request.effector, (rows, 1))
    waist = None
    if include_waist and request.waist.size == len(request.embodiment.waist_joint_names) and request.waist.size:
        waist = np.tile(request.waist, (rows, 1))
    return Chunk(arm=arm, effector=effector, waist=waist)


def terminal_reply(runtime: str) -> dict[str, Any]:
    if runtime in ("newton", "rubik"):
        return {"episode_done": True}
    raise ValueError(f"no terminal reply for runtime {runtime!r} (use the long end hold)")


def reply_rows(result: Mapping[str, Any]) -> int:
    if "joint_targets" in result:
        return len(result["joint_targets"])
    left = result.get("left_effector")
    return len(left) if isinstance(left, Sequence) else 0


__all__ = [
    "RUNTIMES",
    "Chunk",
    "S2Request",
    "WireImage",
    "decode_request",
    "detect_runtime",
    "encode_chunk",
    "encode_isaac",
    "encode_newton",
    "encode_rubik",
    "hold_chunk",
    "matrix_to_xyzw",
    "reply_rows",
    "rubik_bounds",
    "terminal_reply",
]
