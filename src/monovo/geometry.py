"""Calibrated two-view geometry for monocular visual odometry."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class EssentialMatrixConfig:
    """RANSAC and acceptance settings for essential-matrix estimation."""

    ransac_probability: float = 0.999
    ransac_threshold_pixels: float = 1.0
    ransac_max_iterations: int = 1000
    minimum_correspondences: int = 8
    minimum_inliers: int = 8
    random_seed: int = 0

    def __post_init__(self) -> None:
        """Reject invalid geometry settings early."""
        if not 0.0 < self.ransac_probability < 1.0:
            raise ValueError("ransac_probability must lie in (0, 1)")
        if self.ransac_threshold_pixels <= 0.0:
            raise ValueError("ransac_threshold_pixels must be positive")
        if self.ransac_max_iterations <= 0:
            raise ValueError("ransac_max_iterations must be positive")
        if self.minimum_correspondences < 5:
            raise ValueError("minimum_correspondences must be at least 5")
        if self.minimum_inliers < 5:
            raise ValueError("minimum_inliers must be at least 5")


@dataclass(frozen=True)
class RelativePoseEstimate:
    """Result of essential-matrix estimation and pose recovery.

    OpenCV returns the transform from the previous camera coordinates into the
    current camera coordinates:

    ``X_current = R_current_previous @ X_previous + t_current_previous``.

    The translation vector has unit norm and therefore contains direction but
    no metric scale.
    """

    success: bool
    reason: str | None
    essential_matrix: FloatArray | None
    rotation_current_previous: FloatArray | None
    translation_direction_current: FloatArray | None
    ransac_mask: NDArray[np.bool_]
    inlier_mask: NDArray[np.bool_]

    @property
    def ransac_inlier_count(self) -> int:
        """Number of correspondences retained by essential-matrix RANSAC."""
        return int(np.count_nonzero(self.ransac_mask))

    @property
    def inlier_count(self) -> int:
        """Number of RANSAC correspondences that also pass cheirality."""
        return int(np.count_nonzero(self.inlier_mask))

    @property
    def inlier_ratio(self) -> float:
        """Fraction of input correspondences accepted by pose recovery."""
        if len(self.inlier_mask) == 0:
            return 0.0
        return self.inlier_count / len(self.inlier_mask)

    @property
    def transform_current_previous(self) -> FloatArray:
        """Return the scale-ambiguous 4x4 relative transform."""
        if not self.success:
            raise RuntimeError(f"Relative pose is unavailable: {self.reason}")
        assert self.rotation_current_previous is not None
        assert self.translation_direction_current is not None
        transform = np.eye(4, dtype=np.float64)
        transform[:3, :3] = self.rotation_current_previous
        transform[:3, 3] = self.translation_direction_current
        return transform


def estimate_relative_pose(
    previous_points: NDArray[np.floating],
    current_points: NDArray[np.floating],
    intrinsic_matrix: NDArray[np.floating],
    config: EssentialMatrixConfig,
) -> RelativePoseEstimate:
    """Estimate calibrated relative pose from corresponding image points."""
    source, target, intrinsic = _validate_inputs(previous_points, current_points, intrinsic_matrix)
    count = len(source)
    if count < config.minimum_correspondences:
        return _failure(
            count,
            f"Need at least {config.minimum_correspondences} correspondences, got {count}",
        )

    try:
        cv2.setRNGSeed(config.random_seed)
        essential, raw_ransac_mask = cv2.findEssentialMat(
            source,
            target,
            cameraMatrix=intrinsic,
            method=cv2.RANSAC,
            prob=config.ransac_probability,
            threshold=config.ransac_threshold_pixels,
            maxIters=config.ransac_max_iterations,
        )
    except cv2.error as exc:
        return _failure(count, f"Essential-matrix estimation failed: {exc}")

    if essential is None:
        return _failure(count, "Essential-matrix estimation returned no model")

    ransac_mask = _as_boolean_mask(raw_ransac_mask, count)
    if np.count_nonzero(ransac_mask) < config.minimum_inliers:
        return _failure(
            count,
            f"RANSAC found only {np.count_nonzero(ransac_mask)} inliers",
            ransac_mask=ransac_mask,
        )

    candidates = _split_essential_candidates(np.asarray(essential, dtype=np.float64))
    best: tuple[int, FloatArray, FloatArray, FloatArray, NDArray[np.bool_]] | None = None
    for candidate in candidates:
        pose_input_mask = ransac_mask.astype(np.uint8).reshape(-1, 1)
        try:
            recovered_count, rotation, translation, raw_pose_mask = cv2.recoverPose(
                candidate,
                source,
                target,
                cameraMatrix=intrinsic,
                mask=pose_input_mask,
            )
        except cv2.error:
            continue

        pose_mask = _as_boolean_mask(raw_pose_mask, count) & ransac_mask
        pose_count = int(np.count_nonzero(pose_mask))
        if pose_count != recovered_count:
            recovered_count = pose_count
        if best is None or recovered_count > best[0]:
            best = (
                recovered_count,
                candidate,
                np.asarray(rotation, dtype=np.float64),
                np.asarray(translation, dtype=np.float64).reshape(3),
                pose_mask,
            )

    if best is None:
        return _failure(
            count,
            "Pose recovery failed for every essential-matrix candidate",
            ransac_mask=ransac_mask,
        )

    pose_count, selected_essential, rotation, translation, pose_mask = best
    if pose_count < config.minimum_inliers:
        return _failure(
            count,
            f"Pose recovery retained only {pose_count} cheirality inliers",
            essential_matrix=selected_essential,
            ransac_mask=ransac_mask,
            inlier_mask=pose_mask,
        )

    translation_norm = float(np.linalg.norm(translation))
    if not np.isfinite(translation_norm) or translation_norm <= np.finfo(float).eps:
        return _failure(
            count,
            "Pose recovery returned an invalid translation direction",
            essential_matrix=selected_essential,
            ransac_mask=ransac_mask,
            inlier_mask=pose_mask,
        )

    return RelativePoseEstimate(
        success=True,
        reason=None,
        essential_matrix=selected_essential,
        rotation_current_previous=rotation,
        translation_direction_current=translation / translation_norm,
        ransac_mask=ransac_mask,
        inlier_mask=pose_mask,
    )


def rotation_angle_degrees(rotation: NDArray[np.floating]) -> float:
    """Return the magnitude of a 3x3 rotation matrix in degrees."""
    matrix = np.asarray(rotation, dtype=np.float64)
    if matrix.shape != (3, 3):
        raise ValueError(f"Rotation matrix must have shape (3, 3), got {matrix.shape}")
    cosine = np.clip((np.trace(matrix) - 1.0) / 2.0, -1.0, 1.0)
    return float(np.degrees(np.arccos(cosine)))


def _validate_inputs(
    previous_points: NDArray[np.floating],
    current_points: NDArray[np.floating],
    intrinsic_matrix: NDArray[np.floating],
) -> tuple[FloatArray, FloatArray, FloatArray]:
    source = np.asarray(previous_points, dtype=np.float64)
    target = np.asarray(current_points, dtype=np.float64)
    intrinsic = np.asarray(intrinsic_matrix, dtype=np.float64)
    if source.ndim != 2 or source.shape[1:] != (2,):
        raise ValueError(f"previous_points must have shape (N, 2), got {source.shape}")
    if target.shape != source.shape:
        raise ValueError(
            f"current_points shape {target.shape} does not match previous_points {source.shape}"
        )
    if intrinsic.shape != (3, 3):
        raise ValueError(f"intrinsic_matrix must have shape (3, 3), got {intrinsic.shape}")
    if not np.isfinite(source).all() or not np.isfinite(target).all():
        raise ValueError("Point correspondences must contain only finite values")
    if not np.isfinite(intrinsic).all():
        raise ValueError("intrinsic_matrix must contain only finite values")
    if intrinsic[0, 0] <= 0.0 or intrinsic[1, 1] <= 0.0:
        raise ValueError("intrinsic_matrix must contain positive focal lengths")
    return source, target, intrinsic


def _split_essential_candidates(essential: FloatArray) -> list[FloatArray]:
    if essential.shape == (3, 3):
        return [essential]
    if essential.ndim == 2 and essential.shape[1] == 3 and essential.shape[0] % 3 == 0:
        return [essential[index : index + 3] for index in range(0, essential.shape[0], 3)]
    if essential.ndim == 2 and essential.shape[0] == 3 and essential.shape[1] % 3 == 0:
        return [essential[:, index : index + 3] for index in range(0, essential.shape[1], 3)]
    return []


def _as_boolean_mask(mask: NDArray[np.integer] | None, count: int) -> NDArray[np.bool_]:
    if mask is None:
        return np.ones(count, dtype=np.bool_)
    flattened = np.asarray(mask).reshape(-1)
    if len(flattened) != count:
        return np.zeros(count, dtype=np.bool_)
    return flattened.astype(bool)


def _failure(
    count: int,
    reason: str,
    essential_matrix: FloatArray | None = None,
    ransac_mask: NDArray[np.bool_] | None = None,
    inlier_mask: NDArray[np.bool_] | None = None,
) -> RelativePoseEstimate:
    return RelativePoseEstimate(
        success=False,
        reason=reason,
        essential_matrix=essential_matrix,
        rotation_current_previous=None,
        translation_direction_current=None,
        ransac_mask=(
            np.zeros(count, dtype=np.bool_) if ransac_mask is None else ransac_mask.copy()
        ),
        inlier_mask=(
            np.zeros(count, dtype=np.bool_) if inlier_mask is None else inlier_mask.copy()
        ),
    )
