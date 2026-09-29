"""Tests for classical image-feature detection."""

import cv2
import numpy as np
import pytest

from monovo.features import ShiTomasiConfig, detect_shi_tomasi, replenish_features


def test_shi_tomasi_finds_square_corners() -> None:
    image = np.zeros((100, 100), dtype=np.uint8)
    cv2.rectangle(image, (25, 25), (75, 75), color=255, thickness=-1)
    config = ShiTomasiConfig(
        max_features=10,
        quality_level=0.01,
        minimum_distance=5.0,
        block_size=3,
    )

    points = detect_shi_tomasi(image, config)

    assert points.shape == (4, 2)
    expected = np.asarray([[25, 25], [25, 75], [75, 25], [75, 75]], dtype=np.float32)
    for corner in expected:
        assert np.min(np.linalg.norm(points - corner, axis=1)) <= 1.5


def test_shi_tomasi_returns_empty_array_for_textureless_image() -> None:
    image = np.full((50, 60), 127, dtype=np.uint8)

    points = detect_shi_tomasi(image, ShiTomasiConfig())

    assert points.shape == (0, 2)
    assert points.dtype == np.float32


def test_shi_tomasi_respects_detection_mask() -> None:
    image = np.zeros((100, 100), dtype=np.uint8)
    cv2.rectangle(image, (10, 20), (40, 80), color=255, thickness=-1)
    cv2.rectangle(image, (60, 20), (90, 80), color=255, thickness=-1)
    mask = np.zeros_like(image)
    mask[:, :50] = 255

    points = detect_shi_tomasi(image, ShiTomasiConfig(), mask=mask)

    assert len(points) == 4
    assert np.all(points[:, 0] < 50)


def test_replenish_features_preserves_existing_points_and_adds_new_ones() -> None:
    image = np.zeros((100, 100), dtype=np.uint8)
    cv2.rectangle(image, (10, 10), (40, 40), color=255, thickness=-1)
    cv2.rectangle(image, (60, 60), (90, 90), color=255, thickness=-1)
    existing = np.asarray([[10.0, 10.0]], dtype=np.float32)

    replenished = replenish_features(
        image,
        existing,
        ShiTomasiConfig(max_features=20, minimum_distance=8.0, block_size=3),
    )

    np.testing.assert_allclose(replenished[0], existing[0])
    assert len(replenished) > len(existing)
    assert np.all(np.linalg.norm(replenished[1:] - existing[0], axis=1) >= 8.0)


def test_shi_tomasi_requires_grayscale_uint8_input() -> None:
    color_image = np.zeros((20, 30, 3), dtype=np.uint8)
    float_image = np.zeros((20, 30), dtype=np.float32)

    with pytest.raises(ValueError, match="grayscale"):
        detect_shi_tomasi(color_image, ShiTomasiConfig())
    with pytest.raises(ValueError, match="uint8"):
        detect_shi_tomasi(float_image, ShiTomasiConfig())


@pytest.mark.parametrize(
    ("keyword", "value"),
    [
        ("max_features", 0),
        ("quality_level", 0.0),
        ("quality_level", 1.1),
        ("minimum_distance", -1.0),
        ("block_size", 1),
    ],
)
def test_shi_tomasi_rejects_invalid_configuration(keyword: str, value: float) -> None:
    with pytest.raises(ValueError):
        ShiTomasiConfig(**{keyword: value})
