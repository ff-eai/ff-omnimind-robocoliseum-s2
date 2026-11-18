# Copyright (c) 2026 FF EAI Robotics Ecosystem Inc. All rights reserved. Review use only; see LICENSE-FF.txt.

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
import hashlib
from pathlib import Path
import re
from types import MappingProxyType
from typing import Any

import numpy as np
import yaml

from cap_harness.g2.codec import ARM_DOF, ARMS, CAMERAS, HEAD_DOF, WAIST_DOF, GripperConvention

ARM_NAMES = ARMS
BACKENDS = ("geniesim", "gdk")
SCHEMA_VERSION = 1
CAMERA_POSE_SOURCES = ("reported", "fk")
NEED_DEPTH_POLICIES = ("none", "always", "observe_next", "on_demand")
HOLD_TARGETS = ("measured", "last_command")
DEFAULT_FK_URDF: Mapping[str, str] = MappingProxyType(
    {
        "geniesim": "geniesim/G2_omnipicker_sim.urdf",
        "gdk": "real/G2_t2v2_crsB_omnipicker.urdf",
    }
)
FRAME_CONVENTIONS: Mapping[str, np.ndarray] = MappingProxyType(
    {
        "optical": np.eye(4),
        "opengl": np.diag([1.0, -1.0, -1.0, 1.0]),
        "ros": np.array(
            [
                [0.0, 0.0, 1.0, 0.0],
                [-1.0, 0.0, 0.0, 0.0],
                [0.0, -1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        ),
    }
)

_STATION_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]*$")
_DEFAULT_CONFIG_ROOT = Path(__file__).resolve().parents[1] / "configs" / "g2"


def _readonly_array(value: object, name: str, shape: tuple[int, ...]) -> np.ndarray:
    try:
        result = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a numeric array") from exc
    if result.shape != shape or not bool(np.all(np.isfinite(result))):
        raise ValueError(f"{name} must have shape {shape} and contain finite values")
    result = np.array(result, copy=True, order="C")
    result.setflags(write=False)
    return result


def _positive(value: object, name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be positive and finite") from exc
    if not np.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be positive and finite")
    return result


def _non_negative(value: object, name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be non-negative and finite") from exc
    if not np.isfinite(result) or result < 0.0:
        raise ValueError(f"{name} must be non-negative and finite")
    return result


def _port(value: object, name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a TCP port")
    result = int(value)
    if not 0 <= result <= 65535:
        raise ValueError(f"{name} must be a TCP port")
    return result


def _choice(value: object, choices: tuple[str, ...], name: str) -> str:
    result = str(value)
    if result not in choices:
        raise ValueError(f"{name} must be one of {choices}, got {result!r}")
    return result


def _names(value: object, name: str, length: int) -> tuple[str, ...]:
    if isinstance(value, str | bytes) or value is None:
        raise ValueError(f"{name} must be a list of {length} unique names")
    result = tuple(str(item) for item in value)
    if len(result) != length or len(set(result)) != length:
        raise ValueError(f"{name} must contain {length} unique names")
    return result


def _homogeneous(value: object, name: str) -> np.ndarray:
    result = _readonly_array(value, name, (4, 4))
    if not np.allclose(result[3], (0.0, 0.0, 0.0, 1.0), rtol=0.0, atol=1e-9):
        raise ValueError(f"{name} must be homogeneous")
    rotation = result[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), rtol=0.0, atol=1e-5):
        raise ValueError(f"{name} rotation must be orthonormal")
    return result


def _intrinsics(value: object, name: str) -> np.ndarray:
    result = _readonly_array(value, name, (3, 3))
    if result[0, 0] <= 0.0 or result[1, 1] <= 0.0:
        raise ValueError(f"{name} focal lengths must be positive")
    if not np.allclose(result[2], (0.0, 0.0, 1.0), rtol=0.0, atol=1e-6):
        raise ValueError(f"{name} final row must be [0, 0, 1]")
    return result


def _limits(raw: Mapping[str, Any], name: str, length: int) -> tuple[np.ndarray, np.ndarray]:
    if not isinstance(raw, Mapping):
        raise ValueError(f"{name} must map lower and upper")
    lower = _readonly_array(raw.get("lower"), f"{name}.lower", (length,))
    upper = _readonly_array(raw.get("upper"), f"{name}.upper", (length,))
    if np.any(lower >= upper):
        raise ValueError(f"{name} lower bounds must be below upper bounds")
    return lower, upper


def _within(value: np.ndarray, lower: np.ndarray, upper: np.ndarray, name: str) -> None:
    if np.any(value < lower) or np.any(value > upper):
        raise ValueError(f"{name} must lie within its joint limits")


@dataclass(frozen=True, slots=True)
class G2ArmConfig:
    joint_names: tuple[str, ...]

    gdk_joint_names: tuple[str, ...]

    home_joints: np.ndarray

    ready_joints: np.ndarray
    limits_lower: np.ndarray
    limits_upper: np.ndarray
    tool_frame: str

    end_link: str

    tool_offset_m: float


@dataclass(frozen=True, slots=True)
class G2WaistConfig:
    joint_names: tuple[str, ...]
    hold: np.ndarray
    limits_lower: np.ndarray
    limits_upper: np.ndarray


@dataclass(frozen=True, slots=True)
class G2HeadConfig:
    joint_names: tuple[str, ...]
    hold: np.ndarray
    limits_lower: np.ndarray
    limits_upper: np.ndarray
    task_holds: Mapping[str, np.ndarray] = field(default_factory=lambda: MappingProxyType({}))

    def hold_for(self, task_name: str | None) -> np.ndarray:
        return self.task_holds.get(str(task_name or "").strip(), self.hold)


@dataclass(frozen=True, slots=True)
class G2GripperConfig:
    closed: float
    open: float
    raw_open: float
    raw_closed: float
    wire_scale: float
    full_travel_s: float
    stall_delta: float
    stall_steps: int

    @property
    def convention(self) -> GripperConvention:
        return GripperConvention(
            raw_open=self.raw_open, raw_closed=self.raw_closed, wire_scale=self.wire_scale
        )


@dataclass(frozen=True, slots=True)
class G2CameraConfig:
    name: str
    resolution: tuple[int, int]

    color_K: np.ndarray
    depth_K: np.ndarray | None
    T_color_from_depth: np.ndarray | None
    depth_scale: float

    parent_link: str

    T_parent_from_camera: np.ndarray

    frame_convention: str

    optical_correction: np.ndarray

    distortion: tuple[float, ...] | None = None

    @property
    def T_parent_from_optical(self) -> np.ndarray:
        return self.T_parent_from_camera @ self.optical_correction


@dataclass(frozen=True, slots=True)
class G2ControllerConfig:
    control_frequency_hz: float

    servo_hz: float
    ee_pose_hz: float
    settle_s: float
    final_joint_tolerance_rad: float
    max_tracking_error_rad: float
    homing_joint_speed_rad_s: float
    min_homing_s: float
    max_homing_s: float


@dataclass(frozen=True, slots=True)
class G2ModelConfig:
    urdf: str

    curobo_yml: str

    fk_urdf: str

    mjcf: str | None


@dataclass(frozen=True, slots=True)
class G2GenieSimConfig:
    ws_host: str
    ws_port: int
    chunk_len: int

    reply_timeout_s: float

    request_depth: bool
    pad_last_chunk: bool
    keepalive_hold: bool
    reset_timeout_s: float
    step_timeout_s: float

    include_waist: bool
    camera_pose_source: str = "reported"

    need_depth: str = "none"

    keepalive_hold_target: str = "measured"

    drain_chunk_rows: int = 30

    new_connection_new_episode: bool = False

    black_frame_guard: bool = False

    wrist_self_calibration: bool = False

    planner_lock_from_request: bool = False

    planner_lock_quantum_rad: float = 0.002


@dataclass(frozen=True, slots=True)
class G2GdkConfig:
    ws_host: str
    ws_port: int
    robot_host: str


@dataclass(frozen=True, slots=True)
class G2StationConfig:
    schema_version: int
    station: str
    backend: str
    base_frame: str
    default_arm: str
    arm_aliases: Mapping[str, str]
    arms: Mapping[str, G2ArmConfig]
    waist: G2WaistConfig
    head: G2HeadConfig
    gripper: G2GripperConfig
    planning_workspace_lower: np.ndarray
    planning_workspace_upper: np.ndarray
    controller: G2ControllerConfig
    model: G2ModelConfig
    cameras: Mapping[str, G2CameraConfig]
    geniesim: G2GenieSimConfig
    gdk: G2GdkConfig
    config_sha256: str
    source_path: Path

    @property
    def home_joint_positions(self) -> Mapping[str, np.ndarray]:
        return MappingProxyType({name: arm.home_joints for name, arm in self.arms.items()})

    @property
    def joint_limits_lower(self) -> np.ndarray:
        return np.concatenate([self.arms[side].limits_lower for side in ARM_NAMES])

    @property
    def joint_limits_upper(self) -> np.ndarray:
        return np.concatenate([self.arms[side].limits_upper for side in ARM_NAMES])

    @property
    def control_period_s(self) -> float:
        return 1.0 / self.controller.control_frequency_hz

    def resolve_arm(self, arm: str) -> str:
        name = str(arm)
        if name in ARM_NAMES:
            return name
        return str(self.arm_aliases.get(name, name))


def station_config_path(station: str, *, config_root: str | Path | None = None) -> Path:
    if not isinstance(station, str) or _STATION_PATTERN.fullmatch(station) is None:
        raise ValueError("station must be an explicit station identifier")
    root = _DEFAULT_CONFIG_ROOT if config_root is None else Path(config_root).expanduser().resolve()
    resolved_root = root.resolve()

    candidates = [root / f"{station}.yaml"]
    if root.is_dir():
        candidates.extend(sorted(root.glob("*/station.yaml")))

    matches = []
    for candidate in candidates:
        path = candidate.resolve()
        try:
            path.relative_to(resolved_root)
        except ValueError as exc:
            raise ValueError("station profile must remain within config_root") from exc
        if not path.is_file():
            continue
        if path.name == f"{station}.yaml":
            return path
        try:
            declared = yaml.safe_load(path.read_bytes())
        except yaml.YAMLError:
            continue
        if isinstance(declared, dict) and declared.get("station") == station:
            matches.append(path)

    if len(matches) > 1:
        names = ", ".join(str(match.parent.name) for match in matches)
        raise ValueError(f"station {station!r} is declared by more than one profile: {names}")
    if matches:
        return matches[0]
    return (root / f"{station}.yaml").resolve()


def _arm(raw: Mapping[str, Any], name: str) -> G2ArmConfig:
    if not isinstance(raw, Mapping):
        raise ValueError(f"arms.{name} must be a mapping")
    joint_names = _names(raw.get("joint_names"), f"arms.{name}.joint_names", ARM_DOF)
    gdk_names = (
        _names(raw["gdk_joint_names"], f"arms.{name}.gdk_joint_names", ARM_DOF)
        if raw.get("gdk_joint_names") is not None
        else joint_names
    )
    lower, upper = _limits(raw.get("limits"), f"arms.{name}.limits", ARM_DOF)
    home = _readonly_array(raw.get("home_joints"), f"arms.{name}.home_joints", (ARM_DOF,))
    ready = _readonly_array(
        raw.get("ready_joints", raw.get("home_joints")), f"arms.{name}.ready_joints", (ARM_DOF,)
    )
    _within(home, lower, upper, f"arms.{name}.home_joints")
    _within(ready, lower, upper, f"arms.{name}.ready_joints")
    return G2ArmConfig(
        joint_names=joint_names,
        gdk_joint_names=gdk_names,
        home_joints=home,
        ready_joints=ready,
        limits_lower=lower,
        limits_upper=upper,
        tool_frame=str(raw["tool_frame"]),
        end_link=str(raw["end_link"]),
        tool_offset_m=_non_negative(raw.get("tool_offset_m", 0.0), f"arms.{name}.tool_offset_m"),
    )


def _camera(raw: Mapping[str, Any], name: str) -> G2CameraConfig:
    if not isinstance(raw, Mapping):
        raise ValueError(f"cameras.{name} must be a mapping")
    resolution = tuple(int(item) for item in raw["resolution"])
    if len(resolution) != 2 or min(resolution) <= 0:
        raise ValueError(f"cameras.{name}.resolution must be [width, height]")
    depth_K = raw.get("depth_K")
    transform = raw.get("T_color_from_depth")
    correction = raw.get("optical_correction")
    convention = str(raw.get("frame_convention", "optical"))
    if convention not in FRAME_CONVENTIONS:
        raise ValueError(
            f"cameras.{name}.frame_convention must be one of {sorted(FRAME_CONVENTIONS)}, "
            f"got {convention!r}"
        )
    mount = raw.get("T_parent_from_camera")
    distortion = raw.get("distortion")
    if distortion is not None:
        distortion = tuple(float(item) for item in distortion)
        if len(distortion) != 5 or not all(np.isfinite(distortion)):
            raise ValueError(f"cameras.{name}.distortion must be [k1, k2, p1, p2, k3]")
    return G2CameraConfig(
        name=name,
        resolution=(resolution[0], resolution[1]),
        color_K=_intrinsics(raw["color_K"], f"cameras.{name}.color_K"),
        depth_K=None if depth_K is None else _intrinsics(depth_K, f"cameras.{name}.depth_K"),
        T_color_from_depth=(
            None
            if transform is None
            else _homogeneous(transform, f"cameras.{name}.T_color_from_depth")
        ),
        depth_scale=_positive(raw.get("depth_scale", 1000.0), f"cameras.{name}.depth_scale"),
        parent_link=str(raw["parent_link"]),
        T_parent_from_camera=(
            _readonly_array(np.eye(4), "identity", (4, 4))
            if mount is None
            else _mount(mount, f"cameras.{name}.T_parent_from_camera")
        ),
        frame_convention=convention,
        optical_correction=(
            _readonly_array(FRAME_CONVENTIONS[convention], convention, (4, 4))
            if correction is None
            else _homogeneous(correction, f"cameras.{name}.optical_correction")
        ),
        distortion=distortion,
    )


def _mount(value: object, name: str) -> np.ndarray:
    if isinstance(value, Mapping):
        xyz = _readonly_array(value.get("xyz", (0.0, 0.0, 0.0)), f"{name}.xyz", (3,))
        if "quat_wxyz" in value and "quat_xyzw" in value:
            raise ValueError(f"{name} must give quat_wxyz or quat_xyzw, not both")
        if "quat_xyzw" in value:
            q = _readonly_array(value["quat_xyzw"], f"{name}.quat_xyzw", (4,))
            quat = np.array([q[3], q[0], q[1], q[2]])
        else:
            quat = _readonly_array(
                value.get("quat_wxyz", (1.0, 0.0, 0.0, 0.0)), f"{name}.quat_wxyz", (4,)
            )
        norm = float(np.linalg.norm(quat))
        if not np.isfinite(norm) or abs(norm - 1.0) > 1e-3:
            raise ValueError(f"{name} quaternion must be unit length")
        w, x, y, z = quat / norm
        rotation = np.array(
            [
                [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
            ]
        )
        result = np.eye(4)
        result[:3, :3] = rotation
        result[:3, 3] = xyz
        return _readonly_array(result, name, (4, 4))
    return _homogeneous(value, name)


def load_g2_station_config_from_path(path: str | Path) -> G2StationConfig:
    source = Path(path).expanduser().resolve()
    try:
        raw_config = source.read_bytes()
    except FileNotFoundError as exc:
        raise ValueError(f"G2 station profile not found: {source}") from exc
    data = yaml.safe_load(raw_config)
    if not isinstance(data, dict) or data.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"G2 station profile must use schema_version {SCHEMA_VERSION}")
    station = str(data.get("station", "")).strip()
    if _STATION_PATTERN.fullmatch(station) is None:
        raise ValueError("station must be an explicit station identifier")
    backend = str(data.get("backend", "geniesim"))
    if backend not in BACKENDS:
        raise ValueError(f"backend must be one of {BACKENDS}, got {backend!r}")

    default_arm = data.get("default_arm")
    aliases = data.get("arm_aliases")
    if default_arm not in ARM_NAMES:
        raise ValueError(f"default_arm must be one of {ARM_NAMES}, got {default_arm!r}")
    if not isinstance(aliases, dict) or set(aliases) - {"primary", "secondary"}:
        raise ValueError("arm_aliases may only map 'primary' and 'secondary'")
    if set(aliases.values()) - set(ARM_NAMES):
        raise ValueError(f"arm_aliases must point at real arms {ARM_NAMES}")
    if aliases.get("primary") != default_arm:
        raise ValueError("arm_aliases['primary'] must be the default_arm")
    if len(set(aliases.values())) != len(aliases):
        raise ValueError("arm_aliases must not point two roles at the same arm")

    raw_arms = data.get("arms")
    if not isinstance(raw_arms, dict) or set(raw_arms) != set(ARM_NAMES):
        raise ValueError("G2 station must define exactly left and right arms")
    arms = {name: _arm(raw_arms[name], name) for name in ARM_NAMES}

    raw_waist = data.get("waist") or {}
    waist_lower, waist_upper = _limits(raw_waist.get("limits"), "waist.limits", WAIST_DOF)
    waist_hold = _readonly_array(raw_waist.get("hold"), "waist.hold", (WAIST_DOF,))
    _within(waist_hold, waist_lower, waist_upper, "waist.hold")
    waist = G2WaistConfig(
        joint_names=_names(raw_waist.get("joint_names"), "waist.joint_names", WAIST_DOF),
        hold=waist_hold,
        limits_lower=waist_lower,
        limits_upper=waist_upper,
    )

    raw_head = data.get("head") or {}
    head_lower, head_upper = _limits(raw_head.get("limits"), "head.limits", HEAD_DOF)
    head_hold = _readonly_array(raw_head.get("hold"), "head.hold", (HEAD_DOF,))
    _within(head_hold, head_lower, head_upper, "head.hold")
    raw_task_holds = raw_head.get("task_holds") or {}
    if not isinstance(raw_task_holds, Mapping):
        raise ValueError("head.task_holds must map task names to head poses")
    task_holds: dict[str, np.ndarray] = {}
    for task_name, value in raw_task_holds.items():
        task_hold = _readonly_array(value, f"head.task_holds.{task_name}", (HEAD_DOF,))
        _within(task_hold, head_lower, head_upper, f"head.task_holds.{task_name}")
        task_holds[str(task_name)] = task_hold
    head = G2HeadConfig(
        joint_names=_names(raw_head.get("joint_names"), "head.joint_names", HEAD_DOF),
        hold=head_hold,
        limits_lower=head_lower,
        limits_upper=head_upper,
        task_holds=MappingProxyType(task_holds),
    )

    raw_gripper = data.get("gripper")
    if (
        not isinstance(raw_gripper, dict)
        or float(raw_gripper.get("closed", -1)) != 0.0
        or float(raw_gripper.get("open", -1)) != 1.0
    ):
        raise ValueError("gripper must define normalized closed=0.0 and open=1.0 conventions")
    raw_open = float(raw_gripper["raw_open"])
    raw_closed = float(raw_gripper.get("raw_closed", 0.0))
    if not np.isfinite(raw_open) or not np.isfinite(raw_closed) or raw_open == raw_closed:
        raise ValueError("gripper.raw_open and gripper.raw_closed must be distinct finite values")
    wire_scale = float(raw_gripper.get("wire_scale", 1.0))
    if not np.isfinite(wire_scale) or wire_scale == 0.0:
        raise ValueError("gripper.wire_scale must be a non-zero finite factor")
    gripper = G2GripperConfig(
        closed=0.0,
        open=1.0,
        raw_open=raw_open,
        raw_closed=raw_closed,
        wire_scale=wire_scale,
        full_travel_s=_positive(raw_gripper.get("full_travel_s", 1.0), "gripper.full_travel_s"),
        stall_delta=_non_negative(raw_gripper.get("stall_delta", 0.01), "gripper.stall_delta"),
        stall_steps=max(1, int(raw_gripper.get("stall_steps", 5))),
    )

    raw_workspace = data["planning_workspace"]
    workspace_lower = _readonly_array(raw_workspace["lower"], "planning_workspace.lower", (3,))
    workspace_upper = _readonly_array(raw_workspace["upper"], "planning_workspace.upper", (3,))
    if np.any(workspace_lower >= workspace_upper):
        raise ValueError("planning workspace lower bounds must be below upper bounds")
    if np.any(workspace_lower > 0.0) or np.any(workspace_upper < 0.0):
        raise ValueError("planning workspace must contain the base_frame origin")

    raw_controller = data["controller"]
    controller = G2ControllerConfig(
        control_frequency_hz=_positive(raw_controller["control_frequency_hz"], "control frequency"),
        servo_hz=_positive(raw_controller.get("servo_hz", 100.0), "controller.servo_hz"),
        ee_pose_hz=_positive(raw_controller.get("ee_pose_hz", 50.0), "controller.ee_pose_hz"),
        settle_s=_non_negative(raw_controller.get("settle_s", 0.0), "controller.settle_s"),
        final_joint_tolerance_rad=_positive(
            raw_controller["final_joint_tolerance_rad"], "final joint tolerance"
        ),
        max_tracking_error_rad=_positive(
            raw_controller["max_tracking_error_rad"], "maximum tracking error"
        ),
        homing_joint_speed_rad_s=_positive(
            raw_controller.get("homing_joint_speed_rad_s", 0.5), "homing joint speed"
        ),
        min_homing_s=_positive(raw_controller.get("min_homing_s", 1.0), "controller.min_homing_s"),
        max_homing_s=_positive(raw_controller.get("max_homing_s", 8.0), "controller.max_homing_s"),
    )
    if controller.min_homing_s > controller.max_homing_s:
        raise ValueError("controller.min_homing_s must not exceed max_homing_s")

    raw_model = data.get("model") or {}
    if not str(raw_model.get("urdf", "")).strip():
        raise ValueError("model.urdf must name the robot description")
    fk_urdf = str(raw_model.get("fk_urdf") or DEFAULT_FK_URDF[backend])
    model = G2ModelConfig(
        urdf=str(raw_model["urdf"]),
        curobo_yml=str(raw_model.get("curobo_yml", "")),
        fk_urdf=fk_urdf,
        mjcf=None if raw_model.get("mjcf") is None else str(raw_model["mjcf"]),
    )

    raw_cameras = data.get("cameras")
    if not isinstance(raw_cameras, dict) or set(raw_cameras) != set(CAMERAS):
        raise ValueError(f"cameras must define exactly {', '.join(CAMERAS)}")
    cameras = {name: _camera(raw_cameras[name], name) for name in CAMERAS}

    raw_geniesim = data.get("geniesim") or {}
    geniesim = G2GenieSimConfig(
        ws_host=str(raw_geniesim.get("ws_host", "0.0.0.0")),
        ws_port=_port(raw_geniesim.get("ws_port", 8999), "geniesim.ws_port"),
        chunk_len=int(raw_geniesim.get("chunk_len", 30)),
        reply_timeout_s=_positive(raw_geniesim.get("reply_timeout_s", 100.0), "reply timeout"),
        request_depth=bool(raw_geniesim.get("request_depth", False)),
        pad_last_chunk=bool(raw_geniesim.get("pad_last_chunk", True)),
        keepalive_hold=bool(raw_geniesim.get("keepalive_hold", True)),
        reset_timeout_s=_positive(raw_geniesim.get("reset_timeout_s", 600.0), "reset timeout"),
        step_timeout_s=_positive(raw_geniesim.get("step_timeout_s", 300.0), "step timeout"),
        include_waist=bool(raw_geniesim.get("include_waist", False)),
        camera_pose_source=_choice(
            raw_geniesim.get("camera_pose_source", "reported"),
            CAMERA_POSE_SOURCES,
            "geniesim.camera_pose_source",
        ),
        need_depth=_choice(
            raw_geniesim.get("need_depth", "none"), NEED_DEPTH_POLICIES, "geniesim.need_depth"
        ),
        keepalive_hold_target=_choice(
            raw_geniesim.get("keepalive_hold_target", "measured"),
            HOLD_TARGETS,
            "geniesim.keepalive_hold_target",
        ),
        drain_chunk_rows=int(raw_geniesim.get("drain_chunk_rows", 30)),
        new_connection_new_episode=bool(raw_geniesim.get("new_connection_new_episode", False)),
        black_frame_guard=bool(raw_geniesim.get("black_frame_guard", False)),
        wrist_self_calibration=bool(raw_geniesim.get("wrist_self_calibration", False)),
        planner_lock_from_request=bool(raw_geniesim.get("planner_lock_from_request", False)),
        planner_lock_quantum_rad=_positive(
            raw_geniesim.get("planner_lock_quantum_rad", 0.002), "geniesim.planner_lock_quantum_rad"
        ),
    )
    if geniesim.chunk_len <= 0:
        raise ValueError("geniesim.chunk_len must be positive")
    if geniesim.drain_chunk_rows <= 0:
        raise ValueError("geniesim.drain_chunk_rows must be positive")

    raw_gdk = data.get("gdk") or {}
    gdk = G2GdkConfig(
        ws_host=str(raw_gdk.get("ws_host", "0.0.0.0")),
        ws_port=_port(raw_gdk.get("ws_port", 8000), "gdk.ws_port"),
        robot_host=str(raw_gdk.get("robot_host", "")),
    )

    return G2StationConfig(
        schema_version=SCHEMA_VERSION,
        station=station,
        backend=backend,
        base_frame=str(data["base_frame"]),
        default_arm=str(default_arm),
        arm_aliases=MappingProxyType(dict(aliases)),
        arms=MappingProxyType(arms),
        waist=waist,
        head=head,
        gripper=gripper,
        planning_workspace_lower=workspace_lower,
        planning_workspace_upper=workspace_upper,
        controller=controller,
        model=model,
        cameras=MappingProxyType(cameras),
        geniesim=geniesim,
        gdk=gdk,
        config_sha256=hashlib.sha256(raw_config).hexdigest(),
        source_path=source,
    )


def load_g2_station_config(
    station: str = "geniesim-default",
    *,
    config_root: str | Path | None = None,
) -> G2StationConfig:
    path = station_config_path(station, config_root=config_root)
    if not path.is_file():
        raise ValueError(f"unknown G2 station {station!r}")
    config = load_g2_station_config_from_path(path)
    if config.station != station:
        raise ValueError("station profile identity does not match the requested station")
    return config


__all__ = [
    "ARM_NAMES",
    "BACKENDS",
    "CAMERA_POSE_SOURCES",
    "DEFAULT_FK_URDF",
    "FRAME_CONVENTIONS",
    "HOLD_TARGETS",
    "NEED_DEPTH_POLICIES",
    "SCHEMA_VERSION",
    "G2ArmConfig",
    "G2CameraConfig",
    "G2ControllerConfig",
    "G2GdkConfig",
    "G2GenieSimConfig",
    "G2GripperConfig",
    "G2HeadConfig",
    "G2ModelConfig",
    "G2StationConfig",
    "G2WaistConfig",
    "load_g2_station_config",
    "load_g2_station_config_from_path",
    "station_config_path",
]
