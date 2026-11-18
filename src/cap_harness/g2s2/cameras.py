# Copyright (c) 2026 FF EAI Robotics Ecosystem Inc. All rights reserved. Review use only; see LICENSE-FF.txt.

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
import math
from typing import Any

import numpy as np


@dataclass(frozen=True, slots=True)
class CameraModel:
    model: str

    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float
    distortion: tuple[float, ...] = ()

    @property
    def K(self) -> np.ndarray:
        return np.array([[self.fx, 0.0, self.cx], [0.0, self.fy, self.cy], [0.0, 0.0, 1.0]])

    @property
    def fisheye(self) -> bool:
        return self.model == "opencv_fisheye"

    def _coefficients(self) -> np.ndarray:
        values = list(self.distortion) + [0.0] * 8
        return np.asarray(values[:8], dtype=np.float64)

    def undistort_normalized(self, xd: np.ndarray, yd: np.ndarray, iterations: int = 20) -> tuple[np.ndarray, np.ndarray]:
        xd = np.asarray(xd, dtype=np.float64)
        yd = np.asarray(yd, dtype=np.float64)
        if self.model == "pinhole":
            return xd, yd
        if self.fisheye:
            k1, k2, k3, k4 = (list(self.distortion) + [0.0] * 4)[:4]
            theta_d = np.sqrt(xd * xd + yd * yd)
            theta = theta_d.copy()
            for _ in range(iterations):
                t2 = theta * theta
                f = theta * (1 + t2 * (k1 + t2 * (k2 + t2 * (k3 + t2 * k4)))) - theta_d
                df = 1 + t2 * (3 * k1 + t2 * (5 * k2 + t2 * (7 * k3 + 9 * k4 * t2)))
                theta = theta - f / df
            scale = np.where(theta_d > 1e-12, np.tan(theta) / np.maximum(theta_d, 1e-12), 1.0)
            scale = np.where(theta < math.pi / 2 - 1e-6, scale, np.nan)
            return xd * scale, yd * scale
        k1, k2, p1, p2, k3, k4, k5, k6 = self._coefficients()
        x, y = xd.copy(), yd.copy()
        for _ in range(iterations):
            r2 = x * x + y * y
            radial = (1 + r2 * (k1 + r2 * (k2 + r2 * k3))) / (1 + r2 * (k4 + r2 * (k5 + r2 * k6)))
            dx = 2 * p1 * x * y + p2 * (r2 + 2 * x * x)
            dy = p1 * (r2 + 2 * y * y) + 2 * p2 * x * y
            x = (xd - dx) / radial
            y = (yd - dy) / radial
        return x, y

    def distort_normalized(self, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        if self.model == "pinhole":
            return x, y
        if self.fisheye:
            k1, k2, k3, k4 = (list(self.distortion) + [0.0] * 4)[:4]
            r = np.sqrt(x * x + y * y)
            theta = np.arctan(r)
            t2 = theta * theta
            theta_d = theta * (1 + t2 * (k1 + t2 * (k2 + t2 * (k3 + t2 * k4))))
            scale = np.where(r > 1e-12, theta_d / np.maximum(r, 1e-12), 1.0)
            return x * scale, y * scale
        k1, k2, p1, p2, k3, k4, k5, k6 = self._coefficients()
        r2 = x * x + y * y
        radial = (1 + r2 * (k1 + r2 * (k2 + r2 * k3))) / (1 + r2 * (k4 + r2 * (k5 + r2 * k6)))
        xd = x * radial + 2 * p1 * x * y + p2 * (r2 + 2 * x * x)
        yd = y * radial + p1 * (r2 + 2 * y * y) + 2 * p2 * x * y
        return xd, yd

    def pixel_rays(self, u: np.ndarray, v: np.ndarray) -> np.ndarray:
        xd = (np.asarray(u, dtype=np.float64) - self.cx) / self.fx
        yd = (np.asarray(v, dtype=np.float64) - self.cy) / self.fy
        x, y = self.undistort_normalized(xd, yd)
        return np.stack([x, y, np.ones_like(x)], axis=-1)

    def project(self, points_optical: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        p = np.asarray(points_optical, dtype=np.float64).reshape(-1, 3)
        front = p[:, 2] > 1e-6
        z = np.where(front, p[:, 2], 1.0)
        xd, yd = self.distort_normalized(p[:, 0] / z, p[:, 1] / z)
        uv = np.stack([xd * self.fx + self.cx, yd * self.fy + self.cy], axis=-1)
        return uv, front


@dataclass(slots=True)
class CameraView:
    name: str
    rgb: np.ndarray
    model: CameraModel | None
    T_base_optical: np.ndarray | None
    depth_m: np.ndarray | None = None
    pose_source: str = ""

    snapshot_id: str | None = None
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def has_depth(self) -> bool:
        return self.depth_m is not None

    def points_from_mask(
        self,
        mask: np.ndarray | None = None,
        *,
        min_depth_m: float = 0.05,
        max_depth_m: float = 5.0,
        stride: int = 1,
        frame: str = "base",
    ) -> np.ndarray:
        if self.depth_m is None:
            raise ValueError(f"{self.name}: no depth on this request (ask for it with need_depth)")
        if self.model is None or self.T_base_optical is None:
            raise ValueError(f"{self.name}: no camera model / pose for back-projection")
        depth = self.depth_m
        selection = np.ones(depth.shape, dtype=bool) if mask is None else np.asarray(mask, dtype=bool)
        if selection.shape != depth.shape:
            raise ValueError(f"mask {selection.shape} does not match depth {depth.shape}")
        if stride > 1:
            sub = np.zeros_like(selection)
            sub[::stride, ::stride] = True
            selection = selection & sub
        valid = selection & (depth > min_depth_m) & (depth < max_depth_m)
        v, u = np.nonzero(valid)
        if u.size == 0:
            return np.zeros((0, 3))
        rays = self.model.pixel_rays(u.astype(np.float64), v.astype(np.float64))
        points = rays * depth[v, u][:, None]
        points = points[np.all(np.isfinite(points), axis=1)]
        if frame == "optical":
            return points
        return points @ self.T_base_optical[:3, :3].T + self.T_base_optical[:3, 3]

    def project_base_points(self, points_base: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if self.model is None or self.T_base_optical is None:
            raise ValueError(f"{self.name}: no camera model / pose")
        p = np.asarray(points_base, dtype=np.float64).reshape(-1, 3)
        rotation = self.T_base_optical[:3, :3]
        optical = (p - self.T_base_optical[:3, 3]) @ rotation
        uv, front = self.model.project(optical)
        inside = (
            front
            & (uv[:, 0] >= 0) & (uv[:, 0] <= self.model.width - 1)
            & (uv[:, 1] >= 0) & (uv[:, 1] <= self.model.height - 1)
        )
        return uv, inside


def model_from_calibration(calibration: Mapping[str, Any]) -> CameraModel:
    if not calibration.get("available"):
        raise ValueError(f"calibration unavailable: {calibration.get('reason')}")
    model = str(calibration.get("model", "pinhole"))
    if model not in ("pinhole", "opencv_pinhole"):
        raise ValueError(f"unsupported wire camera model {model!r}")
    return CameraModel(
        model=model,
        width=int(calibration["width"]),
        height=int(calibration["height"]),
        fx=float(calibration["fx"]),
        fy=float(calibration["fy"]),
        cx=float(calibration["cx"]),
        cy=float(calibration["cy"]),
        distortion=tuple(float(v) for v in calibration.get("distortion") or ()),
    )


def pose_from_calibration(calibration: Mapping[str, Any]) -> np.ndarray:
    matrix = np.asarray(calibration["optical_to_frame"], dtype=np.float64).reshape(4, 4)
    if not np.all(np.isfinite(matrix)):
        raise ValueError("optical_to_frame must be finite")
    if str(calibration.get("frame", "base_link")) != "base_link":
        raise ValueError(f"unexpected calibration frame {calibration.get('frame')!r}")
    return matrix


def fisheye_model(spec: Mapping[str, Any]) -> CameraModel:
    return CameraModel(
        model="opencv_fisheye",
        width=int(spec["width"]),
        height=int(spec["height"]),
        fx=float(spec["fx"]),
        fy=float(spec["fy"]),
        cx=float(spec["cx"]),
        cy=float(spec["cy"]),
        distortion=tuple(float(v) for v in spec["distortion"]),
    )


def model_from_spec(spec: Mapping[str, Any]) -> CameraModel:
    return CameraModel(
        model=str(spec.get("model", "opencv_pinhole")),
        width=int(spec["width"]),
        height=int(spec["height"]),
        fx=float(spec["fx"]),
        fy=float(spec["fy"]),
        cx=float(spec["cx"]),
        cy=float(spec["cy"]),
        distortion=tuple(float(v) for v in spec.get("distortion") or ()),
    )


def triangulate(views: Sequence[CameraView], pixels: Sequence[Sequence[float]]) -> tuple[np.ndarray, float]:
    a = np.zeros((3, 3))
    b = np.zeros(3)
    rays = []
    for view, (u, v) in zip(views, pixels):
        if view.model is None or view.T_base_optical is None:
            raise ValueError(f"{view.name}: no model / pose")
        ray = view.model.pixel_rays(np.array([float(u)]), np.array([float(v)]))[0]
        direction = view.T_base_optical[:3, :3] @ (ray / np.linalg.norm(ray))
        origin = view.T_base_optical[:3, 3]
        projector = np.eye(3) - np.outer(direction, direction)
        a += projector
        b += projector @ origin
        rays.append((origin, direction))
    point = np.linalg.solve(a, b)
    residuals = [np.linalg.norm((np.eye(3) - np.outer(d, d)) @ (point - o)) for o, d in rays]
    return point, float(np.sqrt(np.mean(np.square(residuals))))


def scale_model(model: CameraModel, width: int, height: int) -> CameraModel:
    sx, sy = width / model.width, height / model.height
    return CameraModel(
        model=model.model,
        width=int(width),
        height=int(height),
        fx=model.fx * sx,
        fy=model.fy * sy,
        cx=model.cx * sx,
        cy=model.cy * sy,
        distortion=model.distortion,
    )


def depth_consistency(
    source: CameraView,
    target: CameraView,
    *,
    stride: int = 4,
    max_points: int = 20000,
    tolerance_m: float = 0.01,
) -> dict[str, Any]:
    points = source.points_from_mask(stride=stride)
    if len(points) > max_points:
        points = points[np.linspace(0, len(points) - 1, max_points).astype(int)]
    if target.model is None or target.T_base_optical is None or target.depth_m is None:
        raise ValueError("target needs a model, pose and depth")
    optical = (points - target.T_base_optical[:3, 3]) @ target.T_base_optical[:3, :3]
    uv, inside = target.project_base_points(points)
    if not inside.any():
        return {"covisible": 0}
    u = np.round(uv[inside, 0]).astype(int)
    v = np.round(uv[inside, 1]).astype(int)
    measured = target.depth_m[v, u]
    expected = optical[inside, 2]
    ok = measured > 0
    difference = np.abs(measured[ok] - expected[ok])
    if difference.size == 0:
        return {"covisible": 0}
    return {
        "covisible": int(difference.size),
        "median_abs_m": float(np.median(difference)),
        "p90_abs_m": float(np.percentile(difference, 90)),
        "within_tolerance": float(np.mean(difference < tolerance_m)),
        "tolerance_m": tolerance_m,
    }


def homogeneous(points: Sequence[Sequence[float]]) -> np.ndarray:
    p = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    return np.concatenate([p, np.ones((p.shape[0], 1))], axis=1)


__all__ = [
    "CameraModel",
    "CameraView",
    "depth_consistency",
    "fisheye_model",
    "homogeneous",
    "model_from_calibration",
    "model_from_spec",
    "pose_from_calibration",
    "scale_model",
    "triangulate",
]
