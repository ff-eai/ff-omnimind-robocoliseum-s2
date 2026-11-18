# Copyright (c) 2026 FF EAI Robotics Ecosystem Inc. All rights reserved. Review use only; see LICENSE-FF.txt.

from __future__ import annotations

import base64
from dataclasses import dataclass, field
import io
import math
import os
from typing import Any

import numpy as np
import requests

from cap_harness.g2s2.cameras import CameraView

DEFAULT_SAM3_URL = os.environ.get("CAP_HARNESS_SAM3_URL", "http://127.0.0.1:8214")


def _png_base64(rgb: np.ndarray) -> str:
    from PIL import Image

    stream = io.BytesIO()
    Image.fromarray(np.asarray(rgb, dtype=np.uint8)).save(stream, format="PNG", compress_level=1)
    return base64.b64encode(stream.getvalue()).decode("ascii")


class Sam3Client:
    def __init__(self, url: str | None = None, *, timeout_s: float = 120.0) -> None:
        self.url = (url or DEFAULT_SAM3_URL).rstrip("/")
        self.timeout_s = timeout_s
        self._session = requests.Session()
        self.calls = 0
        self.seconds = 0.0

    def segment(self, rgb: np.ndarray, text: str) -> list[dict[str, Any]]:
        import time

        started = time.time()
        response = self._session.post(
            f"{self.url}/segment",
            json={"image_base64": _png_base64(rgb), "text_prompt": str(text)},
            timeout=self.timeout_s,
        )
        response.raise_for_status()
        payload = response.json()
        out = []
        for item in payload.get("results") or []:
            shape = tuple(int(v) for v in item["shape"])
            raw = base64.b64decode(item["mask_base64"])
            mask = np.frombuffer(raw, dtype=np.uint8).reshape(shape).astype(bool)
            out.append({"mask": mask, "score": float(item["score"]), "box": [float(v) for v in item["box"]], "label": text})
        self.calls += 1
        self.seconds += time.time() - started
        return out


@dataclass(slots=True)
class Detection:
    label: str
    camera: str
    score: float
    box: list[float]
    mask: np.ndarray
    points: np.ndarray

    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.points.shape[0] >= 10

    @property
    def center(self) -> np.ndarray:
        return np.median(self.points, axis=0) if self.points.size else np.full(3, np.nan)

    @property
    def mid(self) -> np.ndarray:
        if not self.points.size:
            return np.full(3, np.nan)
        return 0.5 * (np.percentile(self.points, 95, axis=0) + np.percentile(self.points, 5, axis=0))

    @property
    def top_z(self) -> float:
        return float(np.percentile(self.points[:, 2], 95)) if self.points.size else float("nan")

    @property
    def bottom_z(self) -> float:
        return float(np.percentile(self.points[:, 2], 3)) if self.points.size else float("nan")

    @property
    def extent(self) -> np.ndarray:
        if not self.points.size:
            return np.full(3, np.nan)
        return np.percentile(self.points, 95, axis=0) - np.percentile(self.points, 5, axis=0)

    def principal_yaw(self) -> tuple[float, float, float]:
        if self.points.shape[0] < 5:
            return 0.0, float("nan"), float("nan")
        xy = self.points[:, :2] - np.median(self.points[:, :2], axis=0)
        cov = np.cov(xy.T)
        values, vectors = np.linalg.eigh(cov)
        major = vectors[:, int(np.argmax(values))]
        yaw = math.atan2(float(major[1]), float(major[0]))
        proj_major = xy @ major
        minor = np.array([-major[1], major[0]])
        proj_minor = xy @ minor
        length = float(np.percentile(proj_major, 95) - np.percentile(proj_major, 5))
        width = float(np.percentile(proj_minor, 95) - np.percentile(proj_minor, 5))
        return yaw, length, width

    def corner_yaw(self, top_band: float = 0.012) -> tuple[float, float]:
        if self.points.shape[0] < 20:
            return 0.0, 1.0
        top = float(np.percentile(self.points[:, 2], 95))
        xy = self.points[self.points[:, 2] > top - top_band][:, :2]
        if xy.shape[0] < 20:
            xy = self.points[:, :2]
        xy = xy - np.median(xy, axis=0)
        areas = []
        steps = np.arange(0.0, 90.0, 0.5)
        for k in steps:
            a = math.radians(float(k))
            u = np.array([math.cos(a), math.sin(a)])
            v = np.array([-math.sin(a), math.cos(a)])
            su, sv = xy @ u, xy @ v
            areas.append((float(np.percentile(su, 99.5)) - float(np.percentile(su, 0.5))) * (float(np.percentile(sv, 99.5)) - float(np.percentile(sv, 0.5))))
        best = int(np.argmin(areas))
        ratio = float(max(areas)) / max(float(min(areas)), 1e-9)
        return math.radians(float(steps[best])), ratio

    def summary(self) -> dict[str, Any]:
        yaw, length, width = self.principal_yaw()
        return {
            "label": self.label,
            "camera": self.camera,
            "score": round(self.score, 3),
            "box": [round(v, 1) for v in self.box],
            "points": int(self.points.shape[0]),
            "center": [round(float(v), 4) for v in self.center],
            "mid": [round(float(v), 4) for v in self.mid],
            "top_z": round(self.top_z, 4),
            "bottom_z": round(self.bottom_z, 4),
            "extent": [round(float(v), 4) for v in self.extent],
            "yaw_deg": round(math.degrees(yaw), 1),
            "length": round(length, 4),
            "width": round(width, 4),
        }


def erode(mask: np.ndarray, pixels: int) -> np.ndarray:
    if pixels <= 0:
        return mask
    try:
        import cv2

        kernel = np.ones((2 * pixels + 1, 2 * pixels + 1), dtype=np.uint8)
        return cv2.erode(mask.astype(np.uint8), kernel).astype(bool)
    except ImportError:
        out = mask.copy()
        for _ in range(pixels):
            out[1:, :] &= out[:-1, :]
            out[:-1, :] &= out[1:, :]
            out[:, 1:] &= out[:, :-1]
            out[:, :-1] &= out[:, 1:]
        return out


def trim_main_mode(points_optical_z: np.ndarray, *, window_m: float = 0.08) -> np.ndarray:
    if points_optical_z.size == 0:
        return np.zeros(0, dtype=bool)
    median = float(np.median(points_optical_z))
    return np.abs(points_optical_z - median) <= max(window_m, 0.15 * median)


def mask_points(view: CameraView, mask: np.ndarray, *, erode_px: int = 2, trim: bool = True, max_points: int = 20000) -> np.ndarray:
    m = erode(np.asarray(mask, dtype=bool), erode_px)
    if m.sum() < 20:
        m = np.asarray(mask, dtype=bool)
    optical = view.points_from_mask(m, frame="optical")
    if optical.shape[0] == 0:
        return np.zeros((0, 3))
    if trim:
        optical = optical[trim_main_mode(optical[:, 2])]
    if optical.shape[0] > max_points:
        optical = optical[np.linspace(0, optical.shape[0] - 1, max_points).astype(int)]
    pose = view.T_base_optical
    assert pose is not None
    return optical @ pose[:3, :3].T + pose[:3, 3]


def detect(
    view: CameraView,
    client: Sam3Client,
    text: str,
    *,
    min_score: float = 0.0,
    erode_px: int = 2,
    max_detections: int = 5,
) -> list[Detection]:
    results = client.segment(view.rgb, text)
    detections = []
    for item in results[:max_detections]:
        if item["score"] < min_score:
            continue
        points = mask_points(view, item["mask"], erode_px=erode_px) if view.has_depth and view.T_base_optical is not None else np.zeros((0, 3))
        detections.append(
            Detection(
                label=text, camera=view.name, score=item["score"], box=item["box"], mask=item["mask"], points=points,
                meta={"mask_pixels": int(item["mask"].sum()), "pose_source": view.pose_source},
            )
        )
    detections.sort(key=lambda d: d.score, reverse=True)
    return detections


def dominant_plane_z(points: np.ndarray, *, bin_m: float = 0.005) -> float:
    if points.shape[0] == 0:
        return float("nan")
    z = points[:, 2]
    bins = np.arange(float(z.min()), float(z.max()) + 2 * bin_m, bin_m)
    if bins.size < 2:
        return float(np.median(z))
    hist, edges = np.histogram(z, bins=bins)
    index = int(np.argmax(hist))
    peak = float((edges[index] + edges[index + 1]) / 2.0)
    near = z[np.abs(z - peak) <= bin_m]
    return float(np.median(near)) if near.size >= 20 else peak


__all__ = ["DEFAULT_SAM3_URL", "Detection", "Sam3Client", "detect", "dominant_plane_z", "erode", "mask_points", "trim_main_mode"]
