"""Rule-based confidence assessment for monocular visual odometry."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray

ConfidenceState = Literal["good", "degraded", "failed"]


@dataclass(frozen=True)
class ConfidenceConfig:
    """Thresholds and reference values used by the confidence rules."""

    tracked_features_reference: int = 500
    inliers_reference: int = 100
    inlier_ratio_reference: float = 0.60
    spatial_coverage_reference: float = 0.50
    minimum_tracked_features: int = 8
    minimum_inliers: int = 8
    minimum_inlier_ratio: float = 0.15
    minimum_median_flow_pixels: float = 0.30
    minimum_spatial_coverage: float = 0.10
    good_score_threshold: float = 0.65
    maximum_rotation_degrees: float = 20.0
    maximum_direction_change_degrees: float = 90.0
    temporal_penalty: float = 0.50
    coverage_grid_rows: int = 4
    coverage_grid_columns: int = 4

    def __post_init__(self) -> None:
        """Validate confidence thresholds and normalizers."""
        positive_values = {
            "tracked_features_reference": self.tracked_features_reference,
            "inliers_reference": self.inliers_reference,
            "inlier_ratio_reference": self.inlier_ratio_reference,
            "spatial_coverage_reference": self.spatial_coverage_reference,
            "minimum_tracked_features": self.minimum_tracked_features,
            "minimum_inliers": self.minimum_inliers,
            "good_score_threshold": self.good_score_threshold,
            "maximum_rotation_degrees": self.maximum_rotation_degrees,
            "maximum_direction_change_degrees": self.maximum_direction_change_degrees,
            "coverage_grid_rows": self.coverage_grid_rows,
            "coverage_grid_columns": self.coverage_grid_columns,
        }
        for name, value in positive_values.items():
            if value <= 0:
                raise ValueError(f"{name} must be positive")
        for name, value in {
            "minimum_inlier_ratio": self.minimum_inlier_ratio,
            "minimum_spatial_coverage": self.minimum_spatial_coverage,
            "temporal_penalty": self.temporal_penalty,
        }.items():
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must lie in [0, 1]")
        if self.minimum_median_flow_pixels < 0.0:
            raise ValueError("minimum_median_flow_pixels cannot be negative")


@dataclass(frozen=True)
class ConfidenceSignals:
    """Per-frame measurements consumed by the confidence rules."""

    pose_success: bool
    tracked_feature_count: int
    ransac_inlier_count: int
    ransac_inlier_ratio: float
    median_flow_pixels: float | None
    spatial_coverage: float
    rotation_degrees: float | None
    translation_direction_change_degrees: float | None
    estimation_failure_reason: str | None = None


@dataclass(frozen=True)
class ConfidenceAssessment:
    """Confidence score, discrete state, and human-readable reasons."""

    score: float
    state: ConfidenceState
    reasons: tuple[str, ...]


def assess_confidence(signals: ConfidenceSignals, config: ConfidenceConfig) -> ConfidenceAssessment:
    """Classify a frame using geometric support and temporal consistency."""
    if not signals.pose_success:
        reason = signals.estimation_failure_reason or "relative pose estimation failed"
        return ConfidenceAssessment(score=0.0, state="failed", reasons=(reason,))

    components = (
        0.20 * min(signals.tracked_feature_count / config.tracked_features_reference, 1.0)
        + 0.30 * min(signals.ransac_inlier_count / config.inliers_reference, 1.0)
        + 0.30 * min(signals.ransac_inlier_ratio / config.inlier_ratio_reference, 1.0)
        + 0.20 * min(signals.spatial_coverage / config.spatial_coverage_reference, 1.0)
    )
    score = float(np.clip(components, 0.0, 1.0))
    failure_reasons: list[str] = []
    temporal_reasons: list[str] = []

    if signals.tracked_feature_count < config.minimum_tracked_features:
        failure_reasons.append("too few tracked features")
    if signals.ransac_inlier_count < config.minimum_inliers:
        failure_reasons.append("too few RANSAC inliers")
    if signals.ransac_inlier_ratio < config.minimum_inlier_ratio:
        failure_reasons.append("RANSAC inlier ratio below threshold")
    if (
        signals.median_flow_pixels is None
        or signals.median_flow_pixels < config.minimum_median_flow_pixels
    ):
        failure_reasons.append("median optical flow below threshold")
    if signals.spatial_coverage < config.minimum_spatial_coverage:
        failure_reasons.append("inlier spatial coverage below threshold")

    if (
        signals.rotation_degrees is not None
        and signals.rotation_degrees > config.maximum_rotation_degrees
    ):
        temporal_reasons.append("sudden rotation")
    if (
        signals.translation_direction_change_degrees is not None
        and signals.translation_direction_change_degrees > config.maximum_direction_change_degrees
    ):
        temporal_reasons.append("sudden translation-direction change")
    if temporal_reasons:
        score *= config.temporal_penalty

    if failure_reasons:
        return ConfidenceAssessment(
            score=score,
            state="failed",
            reasons=tuple(failure_reasons + temporal_reasons),
        )
    if score < config.good_score_threshold or temporal_reasons:
        reasons = temporal_reasons or ["confidence score below good threshold"]
        return ConfidenceAssessment(score=score, state="degraded", reasons=tuple(reasons))
    return ConfidenceAssessment(score=score, state="good", reasons=())


def spatial_coverage(
    points: NDArray[np.floating],
    image_shape: tuple[int, int],
    grid_rows: int = 4,
    grid_columns: int = 4,
) -> float:
    """Return the fraction of image-grid cells containing at least one point."""
    if grid_rows <= 0 or grid_columns <= 0:
        raise ValueError("Grid dimensions must be positive")
    height, width = image_shape
    if height <= 0 or width <= 0:
        raise ValueError("Image dimensions must be positive")
    array = np.asarray(points, dtype=np.float64)
    if array.ndim != 2 or array.shape[1:] != (2,):
        raise ValueError(f"points must have shape (N, 2), got {array.shape}")
    if len(array) == 0:
        return 0.0

    valid = (
        np.isfinite(array).all(axis=1)
        & (array[:, 0] >= 0.0)
        & (array[:, 0] < width)
        & (array[:, 1] >= 0.0)
        & (array[:, 1] < height)
    )
    valid_points = array[valid]
    if len(valid_points) == 0:
        return 0.0

    columns = np.minimum((valid_points[:, 0] * grid_columns / width).astype(int), grid_columns - 1)
    rows = np.minimum((valid_points[:, 1] * grid_rows / height).astype(int), grid_rows - 1)
    occupied = np.unique(rows * grid_columns + columns).size
    return float(occupied / (grid_rows * grid_columns))


def vector_angle_degrees(first: NDArray[np.floating], second: NDArray[np.floating]) -> float:
    """Return the unsigned angle between two nonzero 3D vectors."""
    first_vector = np.asarray(first, dtype=np.float64).reshape(-1)
    second_vector = np.asarray(second, dtype=np.float64).reshape(-1)
    if first_vector.shape != (3,) or second_vector.shape != (3,):
        raise ValueError("Direction vectors must each contain 3 values")
    first_norm = float(np.linalg.norm(first_vector))
    second_norm = float(np.linalg.norm(second_vector))
    if first_norm <= np.finfo(float).eps or second_norm <= np.finfo(float).eps:
        raise ValueError("Direction vectors must be nonzero")
    cosine = np.clip(
        np.dot(first_vector, second_vector) / (first_norm * second_norm),
        -1.0,
        1.0,
    )
    return float(np.degrees(np.arccos(cosine)))
