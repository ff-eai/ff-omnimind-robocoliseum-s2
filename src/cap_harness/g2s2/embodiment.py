# Copyright (c) 2026 FF EAI Robotics Ecosystem Inc. All rights reserved. Review use only; see LICENSE-FF.txt.

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
import json
import math
from pathlib import Path
from types import MappingProxyType
from typing import Any

import numpy as np

PACKAGE_ROOT = Path(__file__).resolve().parent
DESCRIPTION_ROOT = PACKAGE_ROOT / "description"
DATA_ROOT = PACKAGE_ROOT / "data"

ARMS = ("left", "right")
CAMERAS = ("head", "hand_left", "hand_right")
HAND_CAMERA = MappingProxyType({"left": "hand_left", "right": "hand_right"})

WUJI_FINGERS = ("thumb", "index", "middle", "ring", "pinky")


def _rz(angle: float) -> np.ndarray:
    c, s = math.cos(angle), math.sin(angle)
    out = np.eye(4)
    out[:2, :2] = [[c, -s], [s, c]]
    return out


def _frame(x: Sequence[float], y: Sequence[float], z: Sequence[float], origin: Sequence[float]) -> np.ndarray:
    out = np.eye(4)
    out[:3, 0], out[:3, 1], out[:3, 2], out[:3, 3] = x, y, z, origin
    return out


GRASP_FROM_TCP_IDENTITY = np.eye(4)
GRASP_FROM_TCP_90D = np.eye(4)
GRASP_FROM_TCP_90D[2, 3] = 0.035
_90D_PIVOT_X, _90D_PIVOT_Z = 0.030141, 0.10415
_90D_ARM = math.hypot(0.040359, 0.037365)
_90D_ANGLE_OPEN = math.atan2(0.037365, 0.040359)
_90D_SWING = 0.91


def tip_beyond_grasp_90d(width: float) -> float:
    lo = math.cos(_90D_ANGLE_OPEN + _90D_SWING)
    x_closed = _90D_PIVOT_X + _90D_ARM * lo
    c = (x_closed + max(0.0, float(width)) / 2.0 - _90D_PIVOT_X) / _90D_ARM
    c = min(math.cos(_90D_ANGLE_OPEN), max(lo, c))
    z = _90D_PIVOT_Z + _90D_ARM * math.sin(math.acos(c))
    z_open = _90D_PIVOT_Z + _90D_ARM * math.sin(_90D_ANGLE_OPEN)
    return 0.027 + (z - z_open) - float(GRASP_FROM_TCP_90D[2, 3])
GRASP_FROM_TCP_XCLOSE = _rz(math.pi / 2.0)
GRASP_FROM_TCP_AGILEX = _rz(math.pi / 2.0)
GRASP_FROM_TCP_AGILEX[2, 3] = -0.020
GRASP_FROM_TCP_ARX = _rz(math.pi / 2.0)
GRASP_FROM_TCP_ARX[2, 3] = -0.015
GRASP_FROM_WUJI_BASE = _frame((0.0, 0.0, -1.0), (0.0, 1.0, 0.0), (1.0, 0.0, 0.0), (0.045, 0.0, 0.095))


@dataclass(frozen=True, slots=True)
class GripperSpec:
    kind: str

    width: int
    wire_open: float | None = None
    wire_closed: float | None = None
    joint_names: Mapping[str, tuple[str, ...]] = field(default_factory=dict)
    lower: Mapping[str, tuple[float, ...]] = field(default_factory=dict)
    upper: Mapping[str, tuple[float, ...]] = field(default_factory=dict)

    def wire_from_opening(self, opening: float) -> float:
        if self.kind != "parallel" or self.wire_open is None or self.wire_closed is None:
            raise ValueError("wire_from_opening applies to parallel grippers")
        value = min(1.0, max(0.0, float(opening)))
        return float(self.wire_closed + (self.wire_open - self.wire_closed) * value)

    def opening_from_wire(self, wire: float) -> float:
        if self.kind != "parallel" or self.wire_open is None or self.wire_closed is None:
            raise ValueError("opening_from_wire applies to parallel grippers")
        span = self.wire_open - self.wire_closed
        return float(min(1.0, max(0.0, (float(wire) - self.wire_closed) / span)))

    def clip_hand(self, side: str, values: Sequence[float]) -> np.ndarray:
        array = np.asarray(values, dtype=np.float64).reshape(self.width)
        return np.clip(array, np.asarray(self.lower[side]), np.asarray(self.upper[side]))


@dataclass(frozen=True, slots=True)
class ArmSpec:
    side: str
    joint_names: tuple[str, ...]
    flange_link: str

    tcp_link: str

    grasp_parent_link: str
    grasp_from_parent: np.ndarray

    camera: str


@dataclass(frozen=True, slots=True)
class Embodiment:
    key: str
    robot_type: str
    ee_type: str
    robot_cfgs: tuple[str, ...]
    urdf: str
    arm_dof: int
    arms: Mapping[str, ArmSpec]
    active_arms: tuple[str, ...]
    waist_joint_names: tuple[str, ...]
    head_joint_names: tuple[str, ...]
    gripper: GripperSpec
    camera_links: Mapping[str, str]

    fisheye_cameras: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)

    base_frame: str = "base_link"
    arm_frame: str = "arm_base_link"
    chassis_on_wire: bool = False

    notes: str = ""

    @property
    def urdf_path(self) -> Path:
        return DESCRIPTION_ROOT / self.urdf

    @property
    def wire_arm_width(self) -> int:
        return 2 * self.arm_dof

    @property
    def wire_effector_width(self) -> int:
        return 2 * self.gripper.width

    @property
    def is_dexhand(self) -> bool:
        return self.gripper.kind == "dexhand"

    def arm(self, side: str) -> ArmSpec:
        if side not in self.arms:
            raise KeyError(f"{self.key} has no {side!r} arm")
        return self.arms[side]

    def check_active(self, side: str) -> None:
        if side not in self.active_arms:
            raise ValueError(f"{self.key}: the {side} arm is not commandable (active: {self.active_arms})")

    def summary(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "robot_type": self.robot_type,
            "ee_type": self.ee_type,
            "robot_cfgs": list(self.robot_cfgs),
            "urdf": self.urdf,
            "arm_dof": self.arm_dof,
            "active_arms": list(self.active_arms),
            "waist": list(self.waist_joint_names),
            "head": list(self.head_joint_names),
            "gripper": {
                "kind": self.gripper.kind,
                "width": self.gripper.width,
                "wire_open": self.gripper.wire_open,
                "wire_closed": self.gripper.wire_closed,
            },
            "flange": {side: arm.flange_link for side, arm in self.arms.items()},
            "tcp": {side: arm.tcp_link for side, arm in self.arms.items()},
            "camera_links": dict(self.camera_links),
            "fisheye_cameras": sorted(self.fisheye_cameras),
        }


G2_LEFT = tuple(f"idx2{i}_arm_l_joint{i}" for i in range(1, 8))
G2_RIGHT = tuple(f"idx6{i}_arm_r_joint{i}" for i in range(1, 8))
G2_WAIST = tuple(f"idx0{i}_body_joint{i}" for i in range(1, 6))
G2_HEAD = ("idx11_head_joint1", "idx12_head_joint2", "idx13_head_joint3")
G1_WAIST = ("idx02_body_joint2", "idx01_body_joint1")
G1_HEAD = ("idx11_head_joint1", "idx12_head_joint2")
ARX_LEFT = tuple(f"arm_l_joint{i}" for i in range(1, 7))
ARX_RIGHT = tuple(f"arm_r_joint{i}" for i in range(1, 7))
AGILEX_LEFT = tuple(f"arm_fl_joint{i}" for i in range(1, 7))
AGILEX_RIGHT = tuple(f"arm_fr_joint{i}" for i in range(1, 7))
WUJI_JOINTS = MappingProxyType(
    {
        side: tuple(f"hand_{side[0]}_{finger}_joint{j}" for finger in WUJI_FINGERS for j in range(1, 5))
        for side in ARMS
    }
)


def _wuji_limits() -> tuple[dict[str, tuple[float, ...]], dict[str, tuple[float, ...]]]:
    from cap_harness.g2.kinematics import load_urdf

    model = load_urdf(str(DESCRIPTION_ROOT / "G2_crsB_wuji.urdf"))
    lower: dict[str, tuple[float, ...]] = {}
    upper: dict[str, tuple[float, ...]] = {}
    for side in ARMS:
        lo, hi = model.joint_limits(WUJI_JOINTS[side])
        lower[side] = tuple(float(v) for v in lo)
        upper[side] = tuple(float(v) for v in hi)
    return lower, upper


def _load_fisheye() -> dict[str, dict[str, Any]]:
    data = json.loads((DATA_ROOT / "wuji_wrist_cameras.json").read_text(encoding="utf-8"))
    return {name: dict(spec) for name, spec in data["cameras"].items()}


def _g2_arms(tcp: str, grasp: np.ndarray, *, grasp_parent: str | None = None) -> dict[str, ArmSpec]:
    arms = {}
    for side, joints in (("left", G2_LEFT), ("right", G2_RIGHT)):
        s = side[0]
        tcp_link = tcp.format(s=s)
        arms[side] = ArmSpec(
            side=side,
            joint_names=joints,
            flange_link=f"arm_{s}_end_link",
            tcp_link=tcp_link,
            grasp_parent_link=(grasp_parent or tcp).format(s=s),
            grasp_from_parent=grasp,
            camera=HAND_CAMERA[side],
        )
    return arms


def _build() -> dict[str, Embodiment]:
    wuji_lower, wuji_upper = _wuji_limits()
    fisheye = _load_fisheye()
    table: dict[str, Embodiment] = {}

    table["g2_crsb_90d"] = Embodiment(
        key="g2_crsb_90d",
        robot_type="g2a_sim",
        ee_type="zhiyuan_gripper_ctek90d",
        robot_cfgs=("G2_crsB_90d", "G2_crsB_90d_wbc"),
        urdf="G2_crsB_90D.urdf",
        arm_dof=7,
        arms=MappingProxyType(_g2_arms("gripper_{s}_center_link", GRASP_FROM_TCP_90D)),
        active_arms=ARMS,
        waist_joint_names=G2_WAIST,
        head_joint_names=G2_HEAD,
        gripper=GripperSpec(kind="parallel", width=1, wire_open=0.0, wire_closed=-0.91),
        camera_links=MappingProxyType(
            {"head": "head_link3", "hand_left": "gripper_l_camera_link", "hand_right": "gripper_r_camera_link"}
        ),
        chassis_on_wire=True,
        notes="crsB = the real G2's kinematics; TCP 0.185 m along end_link z (HF-main assets; pinned assets 0.1374)",
    )
    table["g2_crsb_wuji"] = Embodiment(
        key="g2_crsb_wuji",
        robot_type="g2a_sim",
        ee_type="wuji",
        robot_cfgs=("G2_crsB_wuji",),
        urdf="G2_crsB_wuji.urdf",
        arm_dof=7,
        arms=MappingProxyType(
            _g2_arms("hand_{s}_index_link4", GRASP_FROM_WUJI_BASE, grasp_parent="hand_{s}_base_link")
        ),
        active_arms=ARMS,
        waist_joint_names=G2_WAIST,
        head_joint_names=G2_HEAD,
        gripper=GripperSpec(
            kind="dexhand",
            width=20,
            joint_names=WUJI_JOINTS,
            lower=MappingProxyType(wuji_lower),
            upper=MappingProxyType(wuji_upper),
        ),
        camera_links=MappingProxyType(
            {"head": "head_link3", "hand_left": "hand_l_base_link", "hand_right": "hand_r_base_link"}
        ),
        fisheye_cameras=MappingProxyType(fisheye),
        chassis_on_wire=True,
        notes="wrist cameras opencvFisheye 1920x1536 (USD); Newton: end_pose null, no depth; Rubik: action_contract",
    )
    table["g2_omnipicker"] = Embodiment(
        key="g2_omnipicker",
        robot_type="g2a_sim",
        ee_type="zhiyuan_gripper_omnipicker",
        robot_cfgs=("G2_omnipicker",),
        urdf="G2_omnipicker.urdf",
        arm_dof=7,
        arms=MappingProxyType(_g2_arms("gripper_{s}_center_link", GRASP_FROM_TCP_XCLOSE)),
        active_arms=ARMS,
        waist_joint_names=G2_WAIST,
        head_joint_names=G2_HEAD,
        gripper=GripperSpec(kind="parallel", width=1, wire_open=-0.785, wire_closed=0.0),
        camera_links=MappingProxyType(
            {"head": "head_link3", "hand_left": "gripper_l_base_link", "hand_right": "gripper_r_base_link"}
        ),
        chassis_on_wire=True,
        notes="Season I kinematics; G2_crsB_omnipicker (scale -0.785) shares the wire tags but is not a Season II task",
    )
    table["g1_omnipicker"] = Embodiment(
        key="g1_omnipicker",
        robot_type="a2d_sim",
        ee_type="zhiyuan_gripper_omnipicker",
        robot_cfgs=("G1_omnipicker",),
        urdf="G1_omnipicker.urdf",
        arm_dof=7,
        arms=MappingProxyType(_g2_arms("gripper_{s}_center_link", GRASP_FROM_TCP_XCLOSE)),
        active_arms=ARMS,
        waist_joint_names=G1_WAIST,
        head_joint_names=G1_HEAD,
        gripper=GripperSpec(kind="parallel", width=1, wire_open=-0.785, wire_closed=0.0),
        camera_links=MappingProxyType(
            {"head": "head_link2", "hand_left": "gripper_l_base_link", "hand_right": "gripper_r_base_link"}
        ),
        notes="waist [idx02_body_joint2 (rev), idx01_body_joint1 (prismatic lift)], head 2",
    )
    table["arx"] = Embodiment(
        key="arx",
        robot_type="arx_sim",
        ee_type="arx_gripper",
        robot_cfgs=("arx",),
        urdf="arx.urdf",
        arm_dof=6,
        arms=MappingProxyType(
            {
                "left": ArmSpec(
                    "left", ARX_LEFT, "arm_l_end_link", "gripper_l_center", "gripper_l_center",
                    GRASP_FROM_TCP_ARX, "hand_left",
                ),
                "right": ArmSpec(
                    "right", ARX_RIGHT, "arm_r_end_link", "gripper_r_center", "gripper_r_center",
                    GRASP_FROM_TCP_ARX, "hand_right",
                ),
            }
        ),
        active_arms=("right",),
        waist_joint_names=(),
        head_joint_names=(),
        gripper=GripperSpec(kind="parallel", width=1, wire_open=-3.5, wire_closed=0.0),
        camera_links=MappingProxyType(
            {"head": "base_link", "hand_left": "arm_l_end_link", "hand_right": "arm_r_end_link"}
        ),
        arm_frame="base_link",
        notes="6-DoF right arm commanded; left arm decorative; arm_base_link frame on the wire = base_link",
    )
    table["agilex"] = Embodiment(
        key="agilex",
        robot_type="agilex_sim",
        ee_type="agilex_gripper",
        robot_cfgs=("agilex",),
        urdf="agilex.urdf",
        arm_dof=6,
        arms=MappingProxyType(
            {
                "left": ArmSpec(
                    "left", AGILEX_LEFT, "arm_fl_end_link", "gripper_l_center_link", "gripper_l_center_link",
                    GRASP_FROM_TCP_AGILEX, "hand_left",
                ),
                "right": ArmSpec(
                    "right", AGILEX_RIGHT, "arm_fr_end_link", "gripper_r_center_link", "gripper_r_center_link",
                    GRASP_FROM_TCP_AGILEX, "hand_right",
                ),
            }
        ),
        active_arms=("right",),
        waist_joint_names=(),
        head_joint_names=(),
        gripper=GripperSpec(kind="parallel", width=1, wire_open=-3.5, wire_closed=0.0),
        camera_links=MappingProxyType(
            {"head": "base_link", "hand_left": "arm_fl_end_link", "hand_right": "arm_fr_end_link"}
        ),
        notes="6-DoF right arm (arm_fr) commanded; URDF root base_footprint, base_link 0.15 m above",
    )
    return table


EMBODIMENTS: Mapping[str, Embodiment] = MappingProxyType(_build())
_BY_TAGS = {(e.robot_type, e.ee_type): e for e in EMBODIMENTS.values()}
_BY_CFG = {cfg: e for e in EMBODIMENTS.values() for cfg in e.robot_cfgs}


def embodiment_for(robot_type: str, ee_type: str) -> Embodiment:
    key = (str(robot_type), str(ee_type))
    found = _BY_TAGS.get(key)
    if found is None and not key[0].endswith("_sim"):
        found = _BY_TAGS.get((key[0] + "_sim", key[1]))
    if found is None:
        raise KeyError(f"no Season II embodiment for robot_type={robot_type!r} ee_type={ee_type!r}")
    return found


def embodiment_for_cfg(robot_cfg: str) -> Embodiment:
    for cfg, embodiment in _BY_CFG.items():
        if cfg.lower() == str(robot_cfg).lower():
            return embodiment
    raise KeyError(f"no Season II embodiment for robot_cfg={robot_cfg!r}")


__all__ = [
    "ARMS",
    "CAMERAS",
    "EMBODIMENTS",
    "HAND_CAMERA",
    "WUJI_FINGERS",
    "WUJI_JOINTS",
    "ArmSpec",
    "Embodiment",
    "GripperSpec",
    "embodiment_for",
    "embodiment_for_cfg",
]
