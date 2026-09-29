"""Tests for failure-aware confidence classification."""

import numpy as np

from monovo.confidence import (
    ConfidenceConfig,
    ConfidenceSignals,
    assess_confidence,
    spatial_coverage,
    vector_angle_degrees,
)


def make_good_signals(**overrides: object) -> ConfidenceSignals:
    """Create a strong frame and optionally replace selected signals."""
    values: dict[str, object] = {
        "pose_success": True,
        "tracked_feature_count": 700,
        "ransac_inlier_count": 300,
        "ransac_inlier_ratio": 0.75,
        "median_flow_pixels": 5.0,
        "spatial_coverage": 0.75,
        "rotation_degrees": 2.0,
        "translation_direction_change_degrees": 5.0,
    }
    values.update(overrides)
    return ConfidenceSignals(**values)  # type: ignore[arg-type]


def test_strong_geometric_support_is_good() -> None:
    assessment = assess_confidence(make_good_signals(), ConfidenceConfig())

    assert assessment.state == "good"
    assert assessment.score == 1.0
    assert assessment.reasons == ()


def test_sudden_direction_change_is_degraded() -> None:
    assessment = assess_confidence(
        make_good_signals(translation_direction_change_degrees=120.0),
        ConfidenceConfig(),
    )

    assert assessment.state == "degraded"
    assert assessment.score == 0.5
    assert assessment.reasons == ("sudden translation-direction change",)


def test_estimation_failure_is_failed() -> None:
    assessment = assess_confidence(
        make_good_signals(pose_success=False, estimation_failure_reason="no essential matrix"),
        ConfidenceConfig(),
    )

    assert assessment.state == "failed"
    assert assessment.score == 0.0
    assert assessment.reasons == ("no essential matrix",)


def test_low_inlier_ratio_is_failed() -> None:
    assessment = assess_confidence(
        make_good_signals(ransac_inlier_ratio=0.10),
        ConfidenceConfig(),
    )

    assert assessment.state == "failed"
    assert "RANSAC inlier ratio below threshold" in assessment.reasons


def test_spatial_coverage_counts_occupied_grid_cells() -> None:
    points = np.asarray([[10.0, 10.0], [90.0, 10.0], [10.0, 90.0], [90.0, 90.0]])

    coverage = spatial_coverage(points, (100, 100), grid_rows=2, grid_columns=2)

    assert coverage == 1.0
    assert spatial_coverage(points[:1], (100, 100), grid_rows=2, grid_columns=2) == 0.25


def test_vector_angle_is_measured_in_common_frame() -> None:
    assert vector_angle_degrees([1.0, 0.0, 0.0], [0.0, 0.0, 1.0]) == 90.0
