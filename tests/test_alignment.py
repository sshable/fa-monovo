"""Tests for Umeyama similarity alignment."""

import cv2
import numpy as np
import pytest

from monovo.alignment import umeyama_alignment


def test_umeyama_recovers_known_similarity() -> None:
    generator = np.random.default_rng(11)
    source = generator.normal(size=(100, 3))
    rotation, _ = cv2.Rodrigues(np.asarray([0.2, -0.1, 0.05], dtype=np.float64))
    scale = 3.4
    translation = np.asarray([4.0, -2.0, 7.0])
    target = scale * (rotation @ source.T).T + translation

    alignment = umeyama_alignment(source, target)

    assert alignment.scale == pytest.approx(scale, abs=1e-12)
    np.testing.assert_allclose(alignment.rotation, rotation, atol=1e-12)
    np.testing.assert_allclose(alignment.translation, translation, atol=1e-12)
    np.testing.assert_allclose(alignment.apply(source), target, atol=1e-12)


def test_umeyama_returns_proper_rotation_for_reflected_data() -> None:
    generator = np.random.default_rng(5)
    source = generator.normal(size=(50, 3))
    target = source.copy()
    target[:, 0] *= -1.0

    alignment = umeyama_alignment(source, target)

    assert np.linalg.det(alignment.rotation) == pytest.approx(1.0)
    assert not np.allclose(alignment.apply(source), target)


def test_umeyama_rejects_degenerate_trajectory() -> None:
    points = np.ones((5, 3))

    with pytest.raises(ValueError, match="zero variance"):
        umeyama_alignment(points, points)
