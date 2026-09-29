"""Trajectory metrics for scale-ambiguous monocular visual odometry."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from monovo.alignment import SimilarityTransform, umeyama_alignment
from monovo.geometry import rotation_angle_degrees
from monovo.odometry import FrameDiagnostics

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class EvaluationResult:
    """Post-hoc Sim(3) alignment and trajectory-quality measurements."""

    alignment: SimilarityTransform
    aligned_estimated_poses: FloatArray
    position_errors: FloatArray
    relative_rotation_errors_degrees: FloatArray
    translation_direction_errors_degrees: FloatArray
    ate_rmse: float
    mean_relative_rotation_error_degrees: float | None
    mean_translation_direction_error_degrees: float | None
    mean_ransac_inlier_ratio: float
    failed_frame_count: int
    degraded_frame_count: int
    average_processing_time_ms: float
    average_fps: float | None

    def to_dict(self) -> dict[str, Any]:
        """Return strict-JSON-compatible summary metrics."""
        return {
            "alignment_mode": "sim3_post_hoc",
            "sim3_scale": self.alignment.scale,
            "ate_rmse_m": self.ate_rmse,
            "relative_rotation_error_degrees_mean": (self.mean_relative_rotation_error_degrees),
            "relative_rotation_valid_count": int(
                np.count_nonzero(np.isfinite(self.relative_rotation_errors_degrees))
            ),
            "translation_direction_error_degrees_mean": (
                self.mean_translation_direction_error_degrees
            ),
            "translation_direction_valid_count": int(
                np.count_nonzero(np.isfinite(self.translation_direction_errors_degrees))
            ),
            "mean_ransac_inlier_ratio": self.mean_ransac_inlier_ratio,
            "failed_frame_count": self.failed_frame_count,
            "degraded_frame_count": self.degraded_frame_count,
            "average_processing_time_ms": self.average_processing_time_ms,
            "average_fps": self.average_fps,
        }


def evaluate_trajectory(
    estimated_poses: NDArray[np.floating],
    ground_truth_poses: NDArray[np.floating],
    diagnostics: Sequence[FrameDiagnostics],
) -> EvaluationResult:
    """Align a complete estimate with Sim(3) and calculate trajectory metrics."""
    estimated = _validate_poses(estimated_poses, "estimated_poses")
    ground_truth = _validate_poses(ground_truth_poses, "ground_truth_poses")
    if estimated.shape != ground_truth.shape:
        raise ValueError(
            f"Pose-array shapes must match, got {estimated.shape} and {ground_truth.shape}"
        )
    if len(diagnostics) != len(estimated):
        raise ValueError(f"Expected {len(estimated)} diagnostic rows, received {len(diagnostics)}")

    alignment = umeyama_alignment(estimated[:, :3, 3], ground_truth[:, :3, 3])
    aligned = alignment.apply_to_poses(estimated)
    position_errors = np.linalg.norm(aligned[:, :3, 3] - ground_truth[:, :3, 3], axis=1)
    ate_rmse = float(np.sqrt(np.mean(position_errors**2)))

    rotation_errors = np.full(len(estimated), np.nan, dtype=np.float64)
    direction_errors = np.full(len(estimated), np.nan, dtype=np.float64)
    for index in range(1, len(estimated)):
        if diagnostics[index].status == "failed":
            continue

        estimated_relative_rotation = aligned[index, :3, :3].T @ aligned[index - 1, :3, :3]
        truth_relative_rotation = ground_truth[index, :3, :3].T @ ground_truth[index - 1, :3, :3]
        rotation_error = estimated_relative_rotation @ truth_relative_rotation.T
        rotation_errors[index] = rotation_angle_degrees(rotation_error)

        estimated_displacement = aligned[index, :3, 3] - aligned[index - 1, :3, 3]
        truth_displacement = ground_truth[index, :3, 3] - ground_truth[index - 1, :3, 3]
        estimated_norm = float(np.linalg.norm(estimated_displacement))
        truth_norm = float(np.linalg.norm(truth_displacement))
        if estimated_norm > np.finfo(float).eps and truth_norm > np.finfo(float).eps:
            cosine = np.clip(
                np.dot(estimated_displacement, truth_displacement) / (estimated_norm * truth_norm),
                -1.0,
                1.0,
            )
            direction_errors[index] = np.degrees(np.arccos(cosine))

    mean_rotation = _finite_mean_or_none(rotation_errors)
    mean_direction = _finite_mean_or_none(direction_errors)
    processing_times = np.asarray(
        [item.processing_time_ms for item in diagnostics[1:]], dtype=np.float64
    )
    average_processing_time = float(np.mean(processing_times)) if len(processing_times) else 0.0
    average_fps = 1000.0 / average_processing_time if average_processing_time > 0.0 else None
    ransac_ratios = np.asarray(
        [item.ransac_inlier_ratio for item in diagnostics[1:]], dtype=np.float64
    )

    return EvaluationResult(
        alignment=alignment,
        aligned_estimated_poses=aligned,
        position_errors=position_errors,
        relative_rotation_errors_degrees=rotation_errors,
        translation_direction_errors_degrees=direction_errors,
        ate_rmse=ate_rmse,
        mean_relative_rotation_error_degrees=mean_rotation,
        mean_translation_direction_error_degrees=mean_direction,
        mean_ransac_inlier_ratio=float(np.mean(ransac_ratios)) if len(ransac_ratios) else 0.0,
        failed_frame_count=sum(item.status == "failed" for item in diagnostics),
        degraded_frame_count=sum(item.status == "degraded" for item in diagnostics),
        average_processing_time_ms=average_processing_time,
        average_fps=average_fps,
    )


def _validate_poses(poses: NDArray[np.floating], name: str) -> FloatArray:
    array = np.asarray(poses, dtype=np.float64)
    if array.ndim != 3 or array.shape[1:] != (4, 4):
        raise ValueError(f"{name} must have shape (N, 4, 4), got {array.shape}")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain only finite values")
    return array


def _finite_mean_or_none(values: FloatArray) -> float | None:
    finite = values[np.isfinite(values)]
    return float(np.mean(finite)) if len(finite) else None
