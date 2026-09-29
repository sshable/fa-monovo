"""Similarity alignment for scale-ambiguous monocular trajectories."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class SimilarityTransform:
    """A proper 3D similarity mapping ``y = scale * rotation @ x + translation``."""

    scale: float
    rotation: FloatArray
    translation: FloatArray

    def apply(self, points: NDArray[np.floating]) -> FloatArray:
        """Apply the similarity transform to an ``(N, 3)`` point array."""
        array = _validate_points(points, "points")
        return self.scale * (self.rotation @ array.T).T + self.translation

    def apply_to_poses(self, poses: NDArray[np.floating]) -> FloatArray:
        """Align camera-to-world pose positions and world-frame orientations."""
        array = np.asarray(poses, dtype=np.float64)
        if array.ndim != 3 or array.shape[1:] != (4, 4):
            raise ValueError(f"poses must have shape (N, 4, 4), got {array.shape}")
        aligned = array.copy()
        aligned[:, :3, 3] = self.apply(array[:, :3, 3])
        aligned[:, :3, :3] = self.rotation[None] @ array[:, :3, :3]
        return aligned


def umeyama_alignment(
    source_points: NDArray[np.floating], target_points: NDArray[np.floating]
) -> SimilarityTransform:
    """Estimate the least-squares Sim(3) mapping source points to target points."""
    source = _validate_points(source_points, "source_points")
    target = _validate_points(target_points, "target_points")
    if source.shape != target.shape:
        raise ValueError(f"Point-array shapes must match, got {source.shape} and {target.shape}")
    if len(source) < 3:
        raise ValueError("At least 3 point pairs are required for Sim(3) alignment")

    source_mean = source.mean(axis=0)
    target_mean = target.mean(axis=0)
    centered_source = source - source_mean
    centered_target = target - target_mean
    source_variance = float(np.sum(centered_source**2) / len(source))
    if source_variance <= np.finfo(float).eps:
        raise ValueError("Source trajectory is degenerate and has zero variance")

    covariance = centered_target.T @ centered_source / len(source)
    left, singular_values, right_transpose = np.linalg.svd(covariance)
    correction = np.eye(3)
    if np.linalg.det(left @ right_transpose) < 0.0:
        correction[-1, -1] = -1.0
    rotation = left @ correction @ right_transpose
    scale = float(np.sum(singular_values * np.diag(correction)) / source_variance)
    if not np.isfinite(scale) or scale <= 0.0:
        raise ValueError("Estimated similarity scale is not positive and finite")
    translation = target_mean - scale * (rotation @ source_mean)
    return SimilarityTransform(scale=scale, rotation=rotation, translation=translation)


def _validate_points(points: NDArray[np.floating], name: str) -> FloatArray:
    array = np.asarray(points, dtype=np.float64)
    if array.ndim != 2 or array.shape[1:] != (3,):
        raise ValueError(f"{name} must have shape (N, 3), got {array.shape}")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain only finite values")
    return array
