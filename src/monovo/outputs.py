"""Serialization of visual-odometry trajectories, metrics, and diagnostics."""

from __future__ import annotations

import csv
import json
from collections.abc import Mapping, Sequence
from dataclasses import fields
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from monovo.odometry import FrameDiagnostics


def save_trajectory(path: str | Path, poses: NDArray[np.floating]) -> Path:
    """Save ``T_world_camera`` poses in KITTI's one-row-per-pose 3x4 format."""
    destination = Path(path)
    array = np.asarray(poses, dtype=np.float64)
    if array.ndim != 3 or array.shape[1:] != (4, 4):
        raise ValueError(f"poses must have shape (N, 4, 4), got {array.shape}")
    if not np.isfinite(array).all():
        raise ValueError("poses must contain only finite values")

    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(destination, array[:, :3, :].reshape(len(array), 12), fmt="%.12e")
    return destination


def save_frame_diagnostics(path: str | Path, diagnostics: Sequence[FrameDiagnostics]) -> Path:
    """Save one CSV row for every frame processed by the odometry loop."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    field_names = [field.name for field in fields(FrameDiagnostics)]

    with destination.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=field_names)
        writer.writeheader()
        for item in diagnostics:
            row = {name: _csv_value(getattr(item, name)) for name in field_names}
            writer.writerow(row)
    return destination


def save_metrics(path: str | Path, metrics: Mapping[str, Any]) -> Path:
    """Write human-readable strict JSON and reject NaN or infinite values."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        serialized = json.dumps(metrics, indent=2, sort_keys=True, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ValueError("metrics must be JSON-serializable and finite") from exc
    destination.write_text(f"{serialized}\n", encoding="utf-8")
    return destination


def _csv_value(value: object) -> object:
    if value is None:
        return ""
    if isinstance(value, tuple):
        return " | ".join(str(item) for item in value)
    if isinstance(value, bool):
        return str(value).lower()
    return value
