"""Tests for essential-matrix estimation and relative-pose recovery."""

import cv2
import numpy as np
import pytest

from monovo.geometry import (
    EssentialMatrixConfig,
    estimate_relative_pose,
    rotation_angle_degrees,
)


def project(points: np.ndarray, intrinsic: np.ndarray) -> np.ndarray:
    """Project camera-frame 3D points into pixel coordinates."""
    homogeneous = (intrinsic @ points.T).T
    return homogeneous[:, :2] / homogeneous[:, 2, None]


def make_correspondences() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Generate noisy two-view correspondences with deterministic outliers."""
    generator = np.random.default_rng(42)
    intrinsic = np.asarray([[720.0, 0.0, 620.0], [0.0, 720.0, 190.0], [0.0, 0.0, 1.0]])
    previous_3d = np.column_stack(
        (
            generator.uniform(-4.0, 4.0, 400),
            generator.uniform(-1.5, 1.5, 400),
            generator.uniform(8.0, 30.0, 400),
        )
    )
    rotation_vector = np.asarray([0.012, -0.025, 0.008], dtype=np.float64)
    true_rotation, _ = cv2.Rodrigues(rotation_vector)
    # Use a healthy synthetic baseline so translation direction is observable
    # despite pixel noise. KITTI's smaller per-frame baseline is tested by the
    # real-data smoke visualization rather than by this numerical unit test.
    true_translation = np.asarray([1.4, 0.08, 0.48], dtype=np.float64)
    current_3d = (true_rotation @ previous_3d.T).T + true_translation
    previous_pixels = project(previous_3d, intrinsic)
    current_pixels = project(current_3d, intrinsic)
    previous_pixels += generator.normal(0.0, 0.2, previous_pixels.shape)
    current_pixels += generator.normal(0.0, 0.2, current_pixels.shape)

    outlier_indices = generator.choice(len(previous_pixels), size=100, replace=False)
    current_pixels[outlier_indices] = np.column_stack(
        (generator.uniform(0.0, 1240.0, 100), generator.uniform(0.0, 380.0, 100))
    )
    return previous_pixels, current_pixels, intrinsic, true_rotation, true_translation


def test_ransac_and_recover_pose_recover_known_motion() -> None:
    previous, current, intrinsic, true_rotation, true_translation = make_correspondences()

    estimate = estimate_relative_pose(
        previous,
        current,
        intrinsic,
        EssentialMatrixConfig(ransac_threshold_pixels=0.5),
    )

    assert estimate.success, estimate.reason
    assert estimate.inlier_count >= 260
    assert estimate.inlier_count < len(previous)
    assert estimate.rotation_current_previous is not None
    assert estimate.translation_direction_current is not None
    rotation_error = estimate.rotation_current_previous @ true_rotation.T
    assert rotation_angle_degrees(rotation_error) < 0.5
    true_direction = true_translation / np.linalg.norm(true_translation)
    angular_error = np.degrees(
        np.arccos(np.clip(estimate.translation_direction_current @ true_direction, -1.0, 1.0))
    )
    assert angular_error < 3.0
    np.testing.assert_allclose(
        estimate.transform_current_previous[:3, :3],
        estimate.rotation_current_previous,
    )


def test_relative_pose_reports_insufficient_correspondences() -> None:
    points = np.zeros((7, 2), dtype=np.float64)
    intrinsic = np.eye(3, dtype=np.float64)

    estimate = estimate_relative_pose(points, points, intrinsic, EssentialMatrixConfig())

    assert not estimate.success
    assert estimate.reason == "Need at least 8 correspondences, got 7"
    assert estimate.inlier_count == 0


def test_relative_pose_rejects_mismatched_point_arrays() -> None:
    with pytest.raises(ValueError, match="does not match"):
        estimate_relative_pose(
            np.zeros((10, 2)),
            np.zeros((9, 2)),
            np.eye(3),
            EssentialMatrixConfig(),
        )


@pytest.mark.parametrize(
    ("keyword", "value"),
    [
        ("ransac_probability", 1.0),
        ("ransac_threshold_pixels", 0.0),
        ("ransac_max_iterations", 0),
        ("minimum_correspondences", 4),
        ("minimum_inliers", 4),
    ],
)
def test_essential_matrix_rejects_invalid_configuration(keyword: str, value: float) -> None:
    with pytest.raises(ValueError):
        EssentialMatrixConfig(**{keyword: value})
