"""Sparse optical-flow tracking between consecutive grayscale frames."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from numpy.typing import NDArray

FloatPoints = NDArray[np.float32]


@dataclass(frozen=True)
class LucasKanadeConfig:
    """Parameters for pyramidal Lucas–Kanade optical flow."""

    window_size: int = 21
    pyramid_levels: int = 3
    criteria_max_iterations: int = 30
    criteria_epsilon: float = 0.01
    forward_backward_threshold: float = 1.0

    def __post_init__(self) -> None:
        """Reject invalid tracker settings early."""
        if self.window_size < 3 or self.window_size % 2 == 0:
            raise ValueError("window_size must be an odd integer of at least 3")
        if self.pyramid_levels < 0:
            raise ValueError("pyramid_levels cannot be negative")
        if self.criteria_max_iterations <= 0:
            raise ValueError("criteria_max_iterations must be positive")
        if self.criteria_epsilon <= 0.0:
            raise ValueError("criteria_epsilon must be positive")
        if self.forward_backward_threshold < 0.0:
            raise ValueError("forward_backward_threshold cannot be negative")


@dataclass(frozen=True)
class TrackingResult:
    """Raw optical-flow estimates and the mask selecting reliable tracks."""

    source_points: FloatPoints
    target_points: FloatPoints
    valid_mask: NDArray[np.bool_]
    forward_backward_error: FloatPoints

    @property
    def valid_source_points(self) -> FloatPoints:
        """Feature coordinates in the previous frame for valid tracks."""
        return self.source_points[self.valid_mask]

    @property
    def valid_target_points(self) -> FloatPoints:
        """Corresponding feature coordinates in the current frame."""
        return self.target_points[self.valid_mask]

    @property
    def valid_count(self) -> int:
        """Number of tracks that passed every check."""
        return int(np.count_nonzero(self.valid_mask))

    @property
    def inlier_ratio(self) -> float:
        """Fraction of input features retained by the tracker."""
        if len(self.source_points) == 0:
            return 0.0
        return self.valid_count / len(self.source_points)


def track_features(
    previous_image: NDArray[np.uint8],
    current_image: NDArray[np.uint8],
    points: FloatPoints,
    config: LucasKanadeConfig,
) -> TrackingResult:
    """Track points with LK flow and reject unreliable correspondences.

    A point is valid when both forward and backward LK calls succeed, all
    coordinates are finite, the current point remains inside the image, and
    tracking back to the previous frame returns within the configured pixel
    threshold of the original point.
    """
    _validate_images(previous_image, current_image)
    source_points = _validate_points(points)
    if len(source_points) == 0:
        return TrackingResult(
            source_points=source_points,
            target_points=np.empty((0, 2), dtype=np.float32),
            valid_mask=np.empty(0, dtype=np.bool_),
            forward_backward_error=np.empty(0, dtype=np.float32),
        )

    lk_parameters = {
        "winSize": (config.window_size, config.window_size),
        "maxLevel": config.pyramid_levels,
        "criteria": (
            cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
            config.criteria_max_iterations,
            config.criteria_epsilon,
        ),
    }
    cv_points = source_points.reshape(-1, 1, 2)
    target, forward_status, _ = cv2.calcOpticalFlowPyrLK(
        previous_image,
        current_image,
        cv_points,
        None,
        **lk_parameters,
    )
    if target is None or forward_status is None:
        return _failed_result(source_points)

    backward, backward_status, _ = cv2.calcOpticalFlowPyrLK(
        current_image,
        previous_image,
        target,
        None,
        **lk_parameters,
    )
    if backward is None or backward_status is None:
        return _failed_result(source_points, target.reshape(-1, 2))

    target_points = np.ascontiguousarray(target.reshape(-1, 2), dtype=np.float32)
    backward_points = np.ascontiguousarray(backward.reshape(-1, 2), dtype=np.float32)
    forward_backward_error = np.linalg.norm(backward_points - source_points, axis=1).astype(
        np.float32
    )

    height, width = current_image.shape
    inside_image = (
        (target_points[:, 0] >= 0.0)
        & (target_points[:, 0] < width)
        & (target_points[:, 1] >= 0.0)
        & (target_points[:, 1] < height)
    )
    finite = (
        np.isfinite(source_points).all(axis=1)
        & np.isfinite(target_points).all(axis=1)
        & np.isfinite(backward_points).all(axis=1)
        & np.isfinite(forward_backward_error)
    )
    valid_mask = (
        forward_status.reshape(-1).astype(bool)
        & backward_status.reshape(-1).astype(bool)
        & finite
        & inside_image
        & (forward_backward_error <= config.forward_backward_threshold)
    )

    return TrackingResult(
        source_points=source_points,
        target_points=target_points,
        valid_mask=valid_mask,
        forward_backward_error=forward_backward_error,
    )


def _validate_images(previous_image: NDArray[np.uint8], current_image: NDArray[np.uint8]) -> None:
    for name, image in (("previous", previous_image), ("current", current_image)):
        if image.ndim != 2:
            raise ValueError(f"Expected {name} image to be grayscale, got shape {image.shape}")
        if image.dtype != np.uint8:
            raise ValueError(f"Expected {name} image to have uint8 dtype, got {image.dtype}")
    if previous_image.shape != current_image.shape:
        raise ValueError(
            f"Frame shapes must match, got {previous_image.shape} and {current_image.shape}"
        )


def _validate_points(points: FloatPoints) -> FloatPoints:
    array = np.asarray(points)
    if array.ndim != 2 or array.shape[1] != 2:
        raise ValueError(f"Feature points must have shape (N, 2), got {array.shape}")
    if not np.issubdtype(array.dtype, np.floating):
        raise ValueError(f"Feature points must use a floating dtype, got {array.dtype}")
    return np.ascontiguousarray(array, dtype=np.float32)


def _failed_result(
    source_points: FloatPoints, target_points: FloatPoints | None = None
) -> TrackingResult:
    count = len(source_points)
    if target_points is None:
        target_points = np.full((count, 2), np.nan, dtype=np.float32)
    return TrackingResult(
        source_points=source_points,
        target_points=np.ascontiguousarray(target_points, dtype=np.float32),
        valid_mask=np.zeros(count, dtype=np.bool_),
        forward_backward_error=np.full(count, np.inf, dtype=np.float32),
    )
