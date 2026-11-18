# Copyright (c) 2026 FF EAI Robotics Ecosystem Inc. All rights reserved. Review use only; see LICENSE-FF.txt.

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
import io
from types import MappingProxyType
from typing import Any

import numpy as np

from cap_harness.contracts import (
    CameraObservation,
    Observation,
    Pose,
    RobotState,
    TaskContext,
)

EMBODIMENT = "g2"
ARMS = ("left", "right")
ARM_DOF = 7
WAIST_DOF = 5
HEAD_DOF = 3
CAMERAS = ("head", "hand_left", "hand_right")
JPEG_ENCODING = "JPEG"
DEPTH_ENCODING = "RAW_UINT16"
DEPTH_KEYS: Mapping[str, str] = MappingProxyType(
    {"head": "head_depth", "hand_left": "hand_left_depth", "hand_right": "hand_right_depth"}
)
DEPTH_SCALES: Mapping[str, float] = MappingProxyType(
    {"head": 1000.0, "hand_left": 10000.0, "hand_right": 10000.0}
)
CHUNK_ARM_COLUMNS = 2 * ARM_DOF
CHUNK_GRIPPER_COLUMNS = 2
CHUNK_WIDTH = CHUNK_ARM_COLUMNS + CHUNK_GRIPPER_COLUMNS
CHUNK_WIDTH_WITH_WAIST = CHUNK_WIDTH + WAIST_DOF
DEFAULT_RAW_OPEN = -0.785
DEFAULT_RAW_CLOSED = 0.0
DEFAULT_WIRE_SCALE = 1.0

_IDENTITY_QUAT_WXYZ = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)


def quat_xyzw_to_wxyz(quat: Any) -> np.ndarray:
    q = np.asarray(quat, dtype=np.float64).reshape(4)
    return np.array([q[3], q[0], q[1], q[2]], dtype=np.float64)


def quat_wxyz_to_xyzw(quat: Any) -> np.ndarray:
    q = np.asarray(quat, dtype=np.float64).reshape(4)
    return np.array([q[1], q[2], q[3], q[0]], dtype=np.float64)


def rotation_to_wxyz(rotation: Any) -> np.ndarray:
    r = np.asarray(rotation, dtype=np.float64).reshape(3, 3)
    trace = float(np.trace(r))
    if trace > 0.0:
        scale = np.sqrt(trace + 1.0) * 2.0
        quat = [
            0.25 * scale,
            (r[2, 1] - r[1, 2]) / scale,
            (r[0, 2] - r[2, 0]) / scale,
            (r[1, 0] - r[0, 1]) / scale,
        ]
    elif r[0, 0] > r[1, 1] and r[0, 0] > r[2, 2]:
        scale = np.sqrt(1.0 + r[0, 0] - r[1, 1] - r[2, 2]) * 2.0
        quat = [
            (r[2, 1] - r[1, 2]) / scale,
            0.25 * scale,
            (r[0, 1] + r[1, 0]) / scale,
            (r[0, 2] + r[2, 0]) / scale,
        ]
    elif r[1, 1] > r[2, 2]:
        scale = np.sqrt(1.0 + r[1, 1] - r[0, 0] - r[2, 2]) * 2.0
        quat = [
            (r[0, 2] - r[2, 0]) / scale,
            (r[0, 1] + r[1, 0]) / scale,
            0.25 * scale,
            (r[1, 2] + r[2, 1]) / scale,
        ]
    else:
        scale = np.sqrt(1.0 + r[2, 2] - r[0, 0] - r[1, 1]) * 2.0
        quat = [
            (r[1, 0] - r[0, 1]) / scale,
            (r[0, 2] + r[2, 0]) / scale,
            (r[1, 2] + r[2, 1]) / scale,
            0.25 * scale,
        ]
    result = np.array(quat, dtype=np.float64)
    norm = float(np.linalg.norm(result))
    return result / norm if norm > 1e-12 else _IDENTITY_QUAT_WXYZ.copy()


def pose_from_matrix(transform: Any, frame: str) -> Pose:
    t = np.asarray(transform, dtype=np.float64).reshape(4, 4)
    return Pose(position=t[:3, 3].copy(), quaternion_wxyz=rotation_to_wxyz(t[:3, :3]), frame=frame)


def identity_pose(frame: str) -> Pose:
    return Pose(position=np.zeros(3), quaternion_wxyz=_IDENTITY_QUAT_WXYZ.copy(), frame=frame)


def decode_jpeg(image_data: bytes) -> np.ndarray:
    buffer = np.frombuffer(bytes(image_data), dtype=np.uint8)
    try:
        import cv2
    except ImportError:
        cv2 = None
    if cv2 is not None:
        bgr = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
        if bgr is None:
            raise ValueError("image_data is not a decodable JPEG")
        return np.ascontiguousarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), dtype=np.uint8)
    from PIL import Image

    with Image.open(io.BytesIO(bytes(image_data))) as image:
        return np.ascontiguousarray(np.asarray(image.convert("RGB"), dtype=np.uint8))


def encode_jpeg(rgb: np.ndarray, quality: int = 95) -> bytes:
    image = np.ascontiguousarray(np.asarray(rgb, dtype=np.uint8))
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("rgb must have shape (H, W, 3)")
    try:
        import cv2
    except ImportError:
        cv2 = None
    if cv2 is not None:
        ok, buffer = cv2.imencode(
            ".jpg", cv2.cvtColor(image, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, quality]
        )
        if not ok:
            raise ValueError("JPEG encoding failed")
        return buffer.tobytes()
    from PIL import Image

    stream = io.BytesIO()
    Image.fromarray(image).save(stream, format="JPEG", quality=quality)
    return stream.getvalue()


def decode_depth_uint16(image_data: bytes, height: int, width: int, scale: float) -> np.ndarray:
    raw = np.frombuffer(bytes(image_data), dtype=np.uint16)
    expected = int(height) * int(width)
    if raw.size != expected:
        raise ValueError(
            f"depth image_data holds {raw.size} uint16 values, expected {expected} ({height}x{width})"
        )
    if float(scale) <= 0.0:
        raise ValueError("depth scale must be positive")
    return np.ascontiguousarray(
        raw.reshape(int(height), int(width)).astype(np.float64) / float(scale)
    )


def encode_depth_uint16(depth_m: np.ndarray, scale: float) -> bytes:
    depth = np.nan_to_num(np.asarray(depth_m, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    return np.clip(depth * float(scale), 0, np.iinfo(np.uint16).max).astype(np.uint16).tobytes()


def decode_image_field(image: Mapping[str, Any]) -> np.ndarray:
    encoding = str(image.get("encoding", JPEG_ENCODING)).upper()
    if encoding != JPEG_ENCODING:
        raise ValueError(f"unsupported RGB image encoding {encoding!r}; expected {JPEG_ENCODING}")
    rgb = decode_jpeg(image["image_data"])
    height = image.get("height")
    width = image.get("width")
    if height is not None and width is not None and rgb.shape[:2] != (int(height), int(width)):
        raise ValueError(
            f"decoded JPEG is {rgb.shape[1]}x{rgb.shape[0]} but the payload declares "
            f"{int(width)}x{int(height)}"
        )
    return rgb


def decode_depth_field(image: Mapping[str, Any], scale: float) -> np.ndarray:
    encoding = str(image.get("encoding", DEPTH_ENCODING)).upper()
    if encoding != DEPTH_ENCODING:
        raise ValueError(f"unsupported depth encoding {encoding!r}; expected {DEPTH_ENCODING}")
    return decode_depth_uint16(image["image_data"], image["height"], image["width"], scale)


def gripper_norm_to_raw(
    norm: Any, *, raw_open: float = DEFAULT_RAW_OPEN, raw_closed: float = DEFAULT_RAW_CLOSED
) -> np.ndarray:
    value = np.clip(np.asarray(norm, dtype=np.float64), 0.0, 1.0)
    return raw_closed + (raw_open - raw_closed) * value


def gripper_raw_to_norm(
    raw: Any, *, raw_open: float = DEFAULT_RAW_OPEN, raw_closed: float = DEFAULT_RAW_CLOSED
) -> np.ndarray:
    span = raw_open - raw_closed
    if abs(span) < 1e-9:
        raise ValueError("raw_open and raw_closed must differ")
    value = (np.asarray(raw, dtype=np.float64) - raw_closed) / span
    return np.clip(value, 0.0, 1.0)


def gripper_raw_to_wire(raw: Any, *, wire_scale: float = DEFAULT_WIRE_SCALE) -> np.ndarray:
    if float(wire_scale) == 0.0:
        raise ValueError("wire_scale must be non-zero")
    return np.asarray(raw, dtype=np.float64) * float(wire_scale)


def gripper_wire_to_raw(wire: Any, *, wire_scale: float = DEFAULT_WIRE_SCALE) -> np.ndarray:
    if float(wire_scale) == 0.0:
        raise ValueError("wire_scale must be non-zero")
    return np.asarray(wire, dtype=np.float64) / float(wire_scale)


@dataclass(frozen=True, slots=True)
class GripperConvention:
    raw_open: float = DEFAULT_RAW_OPEN
    raw_closed: float = DEFAULT_RAW_CLOSED
    wire_scale: float = DEFAULT_WIRE_SCALE

    def norm_to_wire(self, norm: Any) -> np.ndarray:
        raw = gripper_norm_to_raw(norm, raw_open=self.raw_open, raw_closed=self.raw_closed)
        return gripper_raw_to_wire(raw, wire_scale=self.wire_scale)

    def wire_to_norm(self, wire: Any) -> np.ndarray:
        raw = gripper_wire_to_raw(wire, wire_scale=self.wire_scale)
        return gripper_raw_to_norm(raw, raw_open=self.raw_open, raw_closed=self.raw_closed)


@dataclass(frozen=True, slots=True)
class G2Request:
    images: Mapping[str, np.ndarray]
    depths: Mapping[str, np.ndarray | None]
    arm_joints: np.ndarray
    waist_joints: np.ndarray
    head_joints: np.ndarray
    gripper_wire: np.ndarray
    prompt: str
    task_name: str
    episode_idx: int
    episode_done: bool
    task_progress: tuple[Mapping[str, Any], ...]
    robot_type: str
    timestamps: Mapping[str, int] = field(default_factory=dict)
    sequence: int = 0
    received_at_s: float | None = None
    camera_poses: Mapping[str, np.ndarray] = field(default_factory=dict)

    end_poses: Mapping[str, np.ndarray] = field(default_factory=dict)

    @property
    def depth_complete(self) -> bool:
        return all(self.depths.get(camera) is not None for camera in CAMERAS)

    @property
    def left_joints(self) -> np.ndarray:
        return self.arm_joints[:ARM_DOF]

    @property
    def right_joints(self) -> np.ndarray:
        return self.arm_joints[ARM_DOF:]

    def arm(self, side: str) -> np.ndarray:
        return self.left_joints if side == "left" else self.right_joints

    def gripper_norm(self, side: str, convention: GripperConvention) -> float:
        index = ARMS.index(side)
        return float(convention.wire_to_norm(self.gripper_wire[index]))


def _vector(value: Any, name: str, length: int | None) -> np.ndarray:
    result = np.asarray(list(value) if value is not None else [], dtype=np.float64).reshape(-1)
    if length is not None and result.shape[0] != length:
        raise ValueError(f"{name} must have {length} entries, got {result.shape[0]}")
    if not bool(np.all(np.isfinite(result))):
        raise ValueError(f"{name} must be finite")
    return result


def decode_request(
    params: Mapping[str, Any],
    *,
    decode_images: bool = True,
    depth_scales: Mapping[str, float] = DEPTH_SCALES,
    sequence: int = 0,
    received_at_s: float | None = None,
    camera_poses: bool = True,
) -> G2Request:
    if not isinstance(params, Mapping):
        raise ValueError("infer params must be a mapping")
    raw_images = params.get("images") or {}
    raw_states = params.get("states") or {}
    if not isinstance(raw_images, Mapping) or not isinstance(raw_states, Mapping):
        raise ValueError("infer params must carry 'images' and 'states' mappings")

    images: dict[str, np.ndarray] = {}
    depths: dict[str, np.ndarray | None] = {}
    for camera in CAMERAS:
        entry = raw_images.get(camera)
        if entry is None:
            raise ValueError(f"payload images lack the {camera!r} camera")
        if decode_images:
            images[camera] = decode_image_field(entry)
        depth_entry = raw_images.get(DEPTH_KEYS[camera])
        if depth_entry is not None and decode_images:
            depths[camera] = decode_depth_field(depth_entry, depth_scales[camera])
        else:
            depths[camera] = None

    arm = _vector(raw_states.get("arm_joint_states"), "arm_joint_states", 2 * ARM_DOF)
    waist = _vector(raw_states.get("waist_joint_states"), "waist_joint_states", None)
    head = _vector(raw_states.get("head_joint_states"), "head_joint_states", None)
    grippers = _vector(raw_states.get("gripper_states"), "gripper_states", 2)
    if waist.shape[0] not in (0, WAIST_DOF):
        raise ValueError(f"waist_joint_states must have {WAIST_DOF} entries, got {waist.shape[0]}")
    if waist.shape[0] == 0:
        waist = np.zeros(WAIST_DOF, dtype=np.float64)
    if head.shape[0] > HEAD_DOF:
        raise ValueError(f"head_joint_states must have at most {HEAD_DOF} entries")

    progress = params.get("task_progress") or ()
    if isinstance(progress, Mapping):
        progress = (progress,)
    task_progress = tuple(
        MappingProxyType(dict(item)) for item in progress if isinstance(item, Mapping)
    )
    timestamps = params.get("timestamps") or {}
    reported_poses = decode_camera_poses(params.get("camera_poses")) if camera_poses else {}
    return G2Request(
        images=MappingProxyType(images),
        depths=MappingProxyType(depths),
        arm_joints=arm,
        waist_joints=waist,
        head_joints=head,
        gripper_wire=grippers,
        prompt=str(params.get("prompt", "") or ""),
        task_name=str(params.get("task_name", "") or ""),
        episode_idx=int(params.get("episode_idx", 0) or 0),
        episode_done=bool(params.get("episode_done", False)),
        task_progress=task_progress,
        robot_type=str(params.get("robot_type", "") or ""),
        timestamps=MappingProxyType(
            {str(k): int(v) for k, v in timestamps.items()}
            if isinstance(timestamps, Mapping)
            else {}
        ),
        sequence=int(sequence),
        received_at_s=received_at_s,
        camera_poses=MappingProxyType(reported_poses),
        end_poses=MappingProxyType(decode_end_poses(raw_states.get("end_pose"))),
    )


def decode_end_poses(raw: Any) -> dict[str, np.ndarray]:
    if not isinstance(raw, Mapping):
        return {}
    base = raw.get("base_link")
    if not isinstance(base, Mapping):
        return {}
    poses: dict[str, np.ndarray] = {}
    for side in ARMS:
        entry = base.get(f"{side}_arm")
        if not isinstance(entry, Mapping):
            continue
        try:
            position = np.asarray(entry["position"], dtype=np.float64).reshape(3)
            quat = quat_xyzw_to_wxyz(entry["orientation"])
        except (KeyError, TypeError, ValueError):
            continue
        norm = float(np.linalg.norm(quat))
        if not np.all(np.isfinite(position)) or not np.isfinite(norm) or norm < 1e-9:
            continue
        w, x, y, z = quat / norm
        pose = np.eye(4)
        pose[:3, :3] = [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
        pose[:3, 3] = position
        poses[side] = pose
    return poses


def decode_camera_poses(raw: Any) -> dict[str, np.ndarray]:
    if not isinstance(raw, Mapping):
        return {}
    base = raw.get("base_link")
    try:
        inverse_base = None if base is None else _invert_rigid(_matrix4(base))
    except ValueError:
        return {}
    poses: dict[str, np.ndarray] = {}
    for camera in CAMERAS:
        entry = raw.get(camera)
        if entry is None:
            continue
        try:
            pose = _matrix4(entry)
        except ValueError:
            continue
        poses[camera] = pose if inverse_base is None else inverse_base @ pose
    return poses


def _matrix4(value: Any) -> np.ndarray:
    matrix = np.asarray(value, dtype=np.float64)
    if matrix.shape != (4, 4) or not np.all(np.isfinite(matrix)):
        raise ValueError("camera pose must be a finite 4x4 matrix")
    return matrix


def _invert_rigid(matrix: np.ndarray) -> np.ndarray:
    rotation = matrix[:3, :3].T
    result = np.eye(4)
    result[:3, :3] = rotation
    result[:3, 3] = -rotation @ matrix[:3, 3]
    return result


def robot_state(
    request: G2Request,
    station: Any,
    *,
    fk: Any | None = None,
    timestamp_s: float | None = None,
) -> RobotState:
    base_frame = station.base_frame
    convention = station.gripper.convention
    positions: dict[str, np.ndarray] = {}
    velocities: dict[str, np.ndarray] = {}
    poses: dict[str, Pose] = {}
    grippers: dict[str, float] = {}
    names: dict[str, tuple[str, ...]] = {}
    for side in ARMS:
        joints = np.asarray(request.arm(side), dtype=np.float64).reshape(ARM_DOF)
        positions[side] = joints
        velocities[side] = np.zeros(ARM_DOF, dtype=np.float64)
        if fk is None:
            poses[side] = identity_pose(base_frame)
        else:
            poses[side] = pose_from_matrix(fk(side, joints, request.waist_joints), base_frame)
        grippers[side] = request.gripper_norm(side, convention)
        names[side] = tuple(station.arms[side].joint_names)
    return RobotState(
        joint_positions=positions,
        joint_velocities=velocities,
        end_effector_poses=poses,
        gripper_positions=grippers,
        base_frame=base_frame,
        joint_names=names,
        timestamp_s=timestamp_s,
        embodiment=EMBODIMENT,
    )


def camera_observation(
    request: G2Request,
    station: Any,
    camera: str,
    *,
    camera_fk: Any | None = None,
    timestamp_s: float | None = None,
) -> CameraObservation:
    rgb = request.images[camera]
    depth = request.depths.get(camera)
    if depth is None:
        depth = np.zeros(rgb.shape[:2], dtype=np.float64)
    elif depth.shape != rgb.shape[:2]:
        raise ValueError(
            f"{camera} depth {depth.shape} does not match RGB {rgb.shape[:2]}; "
            "reproject it to the colour camera before building the observation"
        )
    config = station.cameras[camera]
    reported = request.camera_poses.get(camera)
    if reported is not None:
        pose = pose_from_matrix(reported, station.base_frame)
    elif camera_fk is None:
        pose = identity_pose(station.base_frame)
    else:
        pose = pose_from_matrix(camera_fk(camera, request), station.base_frame)
    return CameraObservation(
        rgb=rgb,
        depth_m=depth,
        intrinsics=np.asarray(config.color_K, dtype=np.float64),
        frame=f"{camera}_optical",
        camera_pose=pose,
        timestamp_s=timestamp_s,
    )


def observation(
    request: G2Request,
    station: Any,
    *,
    task_context: TaskContext | None = None,
    fk: Any | None = None,
    camera_fk: Any | None = None,
    timestamp_s: float | None = None,
) -> Observation:
    cameras = {
        camera: camera_observation(
            request, station, camera, camera_fk=camera_fk, timestamp_s=timestamp_s
        )
        for camera in CAMERAS
        if camera in request.images
    }
    return Observation(
        cameras=cameras,
        robot_state=robot_state(request, station, fk=fk, timestamp_s=timestamp_s),
        task_context=task_context,
        timestamp_s=timestamp_s,
    )


def _rows(values: np.ndarray) -> list[list[float]]:
    return [[float(item) for item in row] for row in np.asarray(values, dtype=np.float64)]


def encode_action_chunk(
    chunk: Any,
    *,
    convention: GripperConvention | None = None,
    kind: str = "JOINT_ABS",
) -> dict[str, Any]:
    convention = convention or GripperConvention()
    array = np.asarray(chunk, dtype=np.float64)
    if array.ndim == 1:
        array = array.reshape(1, -1)
    if array.ndim != 2 or array.shape[0] == 0:
        raise ValueError("chunk must be a non-empty (H, 16) or (H, 21) array")
    if array.shape[1] not in (CHUNK_WIDTH, CHUNK_WIDTH_WITH_WAIST):
        raise ValueError(
            f"chunk must have {CHUNK_WIDTH} or {CHUNK_WIDTH_WITH_WAIST} columns, got {array.shape[1]}"
        )
    if not bool(np.all(np.isfinite(array))):
        raise ValueError("chunk must be finite")
    if kind not in ("JOINT_ABS", "EEF_ABS"):
        raise ValueError("kind must be JOINT_ABS or EEF_ABS")
    left = array[:, :ARM_DOF]
    right = array[:, ARM_DOF : 2 * ARM_DOF]
    grippers = convention.norm_to_wire(array[:, CHUNK_ARM_COLUMNS : CHUNK_ARM_COLUMNS + 2])
    result: dict[str, Any] = {
        "left_arm": {"kind": kind, "values": _rows(left)},
        "right_arm": {"kind": kind, "values": _rows(right)},
        "left_effector": _rows(grippers[:, :1]),
        "right_effector": _rows(grippers[:, 1:2]),
    }
    if array.shape[1] == CHUNK_WIDTH_WITH_WAIST:
        result["waist"] = {"kind": "JOINT_ABS", "values": _rows(array[:, CHUNK_WIDTH:])}
    return result


def hold_chunk(
    request: G2Request,
    *,
    length: int = 1,
    convention: GripperConvention | None = None,
    include_waist: bool = False,
) -> np.ndarray:
    convention = convention or GripperConvention()
    if int(length) <= 0:
        raise ValueError("length must be positive")
    grippers = convention.wire_to_norm(request.gripper_wire)
    row = np.concatenate([request.arm_joints, grippers])
    if include_waist:
        row = np.concatenate([row, request.waist_joints])
    return np.tile(row, (int(length), 1))


__all__ = [
    "ARMS",
    "ARM_DOF",
    "CAMERAS",
    "CHUNK_WIDTH",
    "CHUNK_WIDTH_WITH_WAIST",
    "DEFAULT_RAW_CLOSED",
    "DEFAULT_RAW_OPEN",
    "DEFAULT_WIRE_SCALE",
    "DEPTH_ENCODING",
    "DEPTH_KEYS",
    "DEPTH_SCALES",
    "EMBODIMENT",
    "HEAD_DOF",
    "JPEG_ENCODING",
    "WAIST_DOF",
    "G2Request",
    "GripperConvention",
    "camera_observation",
    "decode_depth_field",
    "decode_depth_uint16",
    "decode_end_poses",
    "decode_image_field",
    "decode_jpeg",
    "decode_request",
    "encode_action_chunk",
    "encode_depth_uint16",
    "encode_jpeg",
    "gripper_norm_to_raw",
    "gripper_raw_to_norm",
    "gripper_raw_to_wire",
    "gripper_wire_to_raw",
    "hold_chunk",
    "identity_pose",
    "observation",
    "pose_from_matrix",
    "quat_wxyz_to_xyzw",
    "quat_xyzw_to_wxyz",
    "robot_state",
    "rotation_to_wxyz",
]
