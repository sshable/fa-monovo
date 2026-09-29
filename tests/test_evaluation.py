"""Tests for post-hoc trajectory evaluation."""

import cv2
import numpy as np
import pytest

from monovo.evaluation import evaluate_trajectory
from monovo.odometry import FrameDiagnostics


def make_diagnostic(frame_index: int, status: str = "good") -> FrameDiagnostics:
    """Create a compact diagnostic record for metric tests."""
    return FrameDiagnostics(
        frame_index=frame_index,
        status=status,
        confidence_score=1.0 if status == "good" else 0.0,
        confidence_reasons=(),
        failure_reason=None if status != "failed" else "synthetic failure",
        input_feature_count=500,
        tracked_feature_count=450,
        tracking_ratio=0.9,
        ransac_inlier_count=400,
        ransac_inlier_ratio=0.8,
        pose_inlier_count=350,
        pose_inlier_ratio=0.7,
        median_flow_pixels=5.0,
        spatial_coverage=0.75,
        rotation_degrees=1.0,
        translation_direction_change_degrees=2.0,
        output_feature_count=350,
        features_redetected=False,
        processing_time_ms=20.0,
    )


def make_similar_trajectories() -> tuple[np.ndarray, np.ndarray]:
    """Create two pose sequences related by one exact Sim(3)."""
    positions = np.asarray([[0.0, 0.0, 0.0], [1.0, 0.1, 0.0], [2.0, 0.3, 0.2], [3.0, 0.6, 0.5]])
    estimated = np.repeat(np.eye(4)[None], len(positions), axis=0)
    estimated[:, :3, 3] = positions
    for index in range(len(positions)):
        estimated[index, :3, :3], _ = cv2.Rodrigues(np.asarray([0.0, 0.02 * index, 0.0]))

    alignment_rotation, _ = cv2.Rodrigues(np.asarray([0.1, -0.05, 0.02]))
    scale = 2.5
    translation = np.asarray([5.0, -1.0, 3.0])
    ground_truth = estimated.copy()
    ground_truth[:, :3, 3] = scale * (alignment_rotation @ positions.T).T + translation
    ground_truth[:, :3, :3] = alignment_rotation[None] @ estimated[:, :3, :3]
    return estimated, ground_truth


def test_evaluation_removes_one_global_similarity() -> None:
    estimated, ground_truth = make_similar_trajectories()
    diagnostics = [make_diagnostic(index) for index in range(len(estimated))]

    result = evaluate_trajectory(estimated, ground_truth, diagnostics)

    assert result.ate_rmse < 1e-12
    assert result.mean_relative_rotation_error_degrees == pytest.approx(0.0, abs=3e-6)
    assert result.mean_translation_direction_error_degrees == pytest.approx(0.0, abs=3e-6)
    assert result.alignment.scale == pytest.approx(2.5)


def test_evaluation_excludes_failed_relative_motion() -> None:
    estimated, ground_truth = make_similar_trajectories()
    diagnostics = [make_diagnostic(index) for index in range(len(estimated))]
    diagnostics[2] = make_diagnostic(2, status="failed")

    result = evaluate_trajectory(estimated, ground_truth, diagnostics)

    assert np.isnan(result.relative_rotation_errors_degrees[2])
    assert np.isnan(result.translation_direction_errors_degrees[2])
    assert result.failed_frame_count == 1
    assert result.to_dict()["relative_rotation_valid_count"] == 2
