"""Tests for pyramidal Lucas–Kanade feature tracking."""

import cv2
import numpy as np
import pytest

from monovo.features import ShiTomasiConfig, detect_shi_tomasi
from monovo.tracking import LucasKanadeConfig, track_features


def make_textured_image(seed: int = 7) -> np.ndarray:
    """Create a deterministic image with corners at several spatial scales."""
    generator = np.random.default_rng(seed)
    image = np.zeros((240, 320), dtype=np.uint8)
    for _ in range(120):
        center = tuple(generator.integers([12, 12], [308, 228]).tolist())
        radius = int(generator.integers(2, 6))
        intensity = int(generator.integers(100, 256))
        cv2.circle(image, center, radius, intensity, thickness=-1)
    return cv2.GaussianBlur(image, (3, 3), 0)


def test_lucas_kanade_recovers_known_translation() -> None:
    previous = make_textured_image()
    expected_flow = np.asarray([5.0, 3.0], dtype=np.float32)
    transform = np.asarray([[1.0, 0.0, expected_flow[0]], [0.0, 1.0, expected_flow[1]]])
    current = cv2.warpAffine(previous, transform, (previous.shape[1], previous.shape[0]))
    points = detect_shi_tomasi(
        previous,
        ShiTomasiConfig(max_features=300, minimum_distance=6.0, block_size=5),
    )

    result = track_features(previous, current, points, LucasKanadeConfig())
    measured_flow = result.valid_target_points - result.valid_source_points

    assert result.valid_count >= int(0.9 * len(points))
    np.testing.assert_allclose(np.median(measured_flow, axis=0), expected_flow, atol=0.15)
    assert np.median(result.forward_backward_error[result.valid_mask]) < 0.05


def test_lucas_kanade_rejects_inconsistent_backward_tracks() -> None:
    previous = make_textured_image()
    current = make_textured_image(seed=99)
    points = detect_shi_tomasi(previous, ShiTomasiConfig(max_features=200))

    result = track_features(
        previous,
        current,
        points,
        LucasKanadeConfig(forward_backward_threshold=0.25),
    )

    assert result.inlier_ratio < 0.25


def test_lucas_kanade_handles_empty_feature_array() -> None:
    image = np.zeros((30, 40), dtype=np.uint8)

    result = track_features(
        image,
        image.copy(),
        np.empty((0, 2), dtype=np.float32),
        LucasKanadeConfig(),
    )

    assert result.valid_count == 0
    assert result.inlier_ratio == 0.0
    assert result.valid_target_points.shape == (0, 2)


def test_lucas_kanade_rejects_mismatched_frames() -> None:
    previous = np.zeros((30, 40), dtype=np.uint8)
    current = np.zeros((31, 40), dtype=np.uint8)

    with pytest.raises(ValueError, match="Frame shapes must match"):
        track_features(
            previous,
            current,
            np.asarray([[10.0, 10.0]], dtype=np.float32),
            LucasKanadeConfig(),
        )


@pytest.mark.parametrize(
    ("keyword", "value"),
    [
        ("window_size", 10),
        ("pyramid_levels", -1),
        ("criteria_max_iterations", 0),
        ("criteria_epsilon", 0.0),
        ("forward_backward_threshold", -1.0),
    ],
)
def test_lucas_kanade_rejects_invalid_configuration(keyword: str, value: float) -> None:
    with pytest.raises(ValueError):
        LucasKanadeConfig(**{keyword: value})
