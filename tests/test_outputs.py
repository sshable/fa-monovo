"""Tests for persistent VO result artifacts."""

import csv
import json
from pathlib import Path

import numpy as np
import pytest

from monovo.odometry import FrameDiagnostics
from monovo.outputs import save_frame_diagnostics, save_metrics, save_trajectory


def make_diagnostic() -> FrameDiagnostics:
    """Create a diagnostic row containing optional and tuple fields."""
    return FrameDiagnostics(
        frame_index=1,
        status="degraded",
        confidence_score=0.5,
        confidence_reasons=("low coverage", "sudden rotation"),
        failure_reason=None,
        input_feature_count=20,
        tracked_feature_count=10,
        tracking_ratio=0.5,
        ransac_inlier_count=8,
        ransac_inlier_ratio=0.8,
        pose_inlier_count=7,
        pose_inlier_ratio=0.7,
        median_flow_pixels=2.0,
        spatial_coverage=0.25,
        rotation_degrees=3.0,
        translation_direction_change_degrees=4.0,
        output_feature_count=7,
        features_redetected=True,
        processing_time_ms=12.5,
    )


def test_save_trajectory_uses_kitti_3x4_rows(tmp_path: Path) -> None:
    poses = np.repeat(np.eye(4)[None], 2, axis=0)
    poses[1, :3, 3] = [1.0, 2.0, 3.0]

    destination = save_trajectory(tmp_path / "trajectory.txt", poses)
    loaded = np.loadtxt(destination)

    assert loaded.shape == (2, 12)
    np.testing.assert_allclose(loaded[1].reshape(3, 4)[:, 3], [1.0, 2.0, 3.0])


def test_save_frame_diagnostics_preserves_reasons(tmp_path: Path) -> None:
    destination = save_frame_diagnostics(tmp_path / "diagnostics.csv", [make_diagnostic()])

    with destination.open(encoding="utf-8", newline="") as stream:
        row = next(csv.DictReader(stream))

    assert row["confidence_reasons"] == "low coverage | sudden rotation"
    assert row["failure_reason"] == ""
    assert row["features_redetected"] == "true"


def test_save_metrics_is_strict_json(tmp_path: Path) -> None:
    destination = save_metrics(tmp_path / "metrics.json", {"ate_rmse_m": 1.25})

    assert json.loads(destination.read_text(encoding="utf-8")) == {"ate_rmse_m": 1.25}
    with pytest.raises(ValueError, match="finite"):
        save_metrics(tmp_path / "invalid.json", {"bad": float("nan")})
