"""Feature detection for the monocular visual-odometry front end."""

from __future__ import annotations

from dataclasses import dataclass, replace

import cv2
import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class ShiTomasiConfig:
    """Parameters for OpenCV's Shi–Tomasi corner detector."""

    max_features: int = 2000
    quality_level: float = 0.01
    minimum_distance: float = 7.0
    block_size: int = 7

    def __post_init__(self) -> None:
        """Reject invalid detector settings early."""
        if self.max_features <= 0:
            raise ValueError("max_features must be positive")
        if not 0.0 < self.quality_level <= 1.0:
            raise ValueError("quality_level must lie in (0, 1]")
        if self.minimum_distance < 0.0:
            raise ValueError("minimum_distance cannot be negative")
        if self.block_size < 2:
            raise ValueError("block_size must be at least 2")


def detect_shi_tomasi(
    image: NDArray[np.uint8],
    config: ShiTomasiConfig,
    mask: NDArray[np.uint8] | None = None,
) -> NDArray[np.float32]:
    """Detect trackable corners and return their ``(u, v)`` pixel coordinates.

    Shi–Tomasi scores a small image patch using the smaller eigenvalue of its
    gradient matrix. A large score means brightness changes in two directions,
    which makes the point easier to localize and track in the next frame.
    """
    if image.ndim != 2:
        raise ValueError(f"Expected a grayscale image with 2 dimensions, got {image.shape}")
    if image.dtype != np.uint8:
        raise ValueError(f"Expected a uint8 image, got {image.dtype}")

    detector_mask: NDArray[np.uint8] | None = None
    if mask is not None:
        if mask.shape != image.shape:
            raise ValueError(
                f"Feature mask shape {mask.shape} does not match image shape {image.shape}"
            )
        detector_mask = np.where(mask > 0, 255, 0).astype(np.uint8)

    corners = cv2.goodFeaturesToTrack(
        image,
        maxCorners=config.max_features,
        qualityLevel=config.quality_level,
        minDistance=config.minimum_distance,
        mask=detector_mask,
        blockSize=config.block_size,
        useHarrisDetector=False,
    )
    if corners is None:
        return np.empty((0, 2), dtype=np.float32)
    return np.ascontiguousarray(corners.reshape(-1, 2), dtype=np.float32)


def replenish_features(
    image: NDArray[np.uint8],
    existing_points: NDArray[np.floating],
    config: ShiTomasiConfig,
) -> NDArray[np.float32]:
    """Add newly detected corners away from retained feature locations."""
    points = np.asarray(existing_points, dtype=np.float32)
    if points.ndim != 2 or points.shape[1:] != (2,):
        raise ValueError(f"existing_points must have shape (N, 2), got {points.shape}")
    if len(points) >= config.max_features:
        return np.ascontiguousarray(points[: config.max_features])

    mask = np.full(image.shape, 255, dtype=np.uint8)
    exclusion_radius = max(1, int(np.ceil(config.minimum_distance)))
    height, width = image.shape
    for horizontal, vertical in points:
        if 0.0 <= horizontal < width and 0.0 <= vertical < height:
            cv2.circle(
                mask,
                (int(round(float(horizontal))), int(round(float(vertical)))),
                exclusion_radius,
                color=0,
                thickness=-1,
            )

    remaining = config.max_features - len(points)
    additions = detect_shi_tomasi(
        image,
        replace(config, max_features=remaining),
        mask=mask,
    )
    if len(points) == 0:
        return additions
    return np.ascontiguousarray(np.vstack((points, additions)), dtype=np.float32)
