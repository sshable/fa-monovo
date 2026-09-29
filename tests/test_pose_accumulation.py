"""Tests for SE(3) conventions and camera-pose accumulation."""

import cv2
import numpy as np
import pytest

from monovo.confidence import ConfidenceConfig
from monovo.features import ShiTomasiConfig
from monovo.geometry import EssentialMatrixConfig
from monovo.odometry import (
    MonocularOdometry,
    MonocularOdometryConfig,
    PoseAccumulator,
    accumulate_camera_pose,
    invert_se3,
    make_transform,
    validate_se3,
)
from monovo.tracking import LucasKanadeConfig


class BlankSequence:
    """Small sequence used to verify graceful estimation failure."""

    def __init__(self, frame_count: int = 4) -> None:
        self.images = [np.zeros((60, 80), dtype=np.uint8) for _ in range(frame_count)]
        self._intrinsic = np.asarray([[100.0, 0.0, 40.0], [0.0, 100.0, 30.0], [0.0, 0.0, 1.0]])

    def __len__(self) -> int:
        return len(self.images)

    @property
    def intrinsic_matrix(self) -> np.ndarray:
        return self._intrinsic.copy()

    def load_image(self, index: int) -> np.ndarray:
        return self.images[index].copy()


def test_rigid_transform_inverse_maps_points_back() -> None:
    rotation, _ = cv2.Rodrigues(np.asarray([0.1, -0.2, 0.05], dtype=np.float64))
    transform = make_transform(rotation, np.asarray([1.0, 2.0, -0.5]))
    inverse = invert_se3(transform)

    np.testing.assert_allclose(transform @ inverse, np.eye(4), atol=1e-12)
    np.testing.assert_allclose(inverse @ transform, np.eye(4), atol=1e-12)


def test_forward_camera_motion_has_positive_world_z_position() -> None:
    world_previous = np.eye(4, dtype=np.float64)
    # A forward-moving camera observes the static scene translated toward -z.
    current_previous = make_transform(np.eye(3), np.asarray([0.0, 0.0, -1.0]))

    world_current = accumulate_camera_pose(world_previous, current_previous)

    np.testing.assert_allclose(world_current[:3, 3], [0.0, 0.0, 1.0])


def test_pose_composition_uses_previous_camera_orientation() -> None:
    rotation_world_previous, _ = cv2.Rodrigues(
        np.asarray([0.0, np.deg2rad(90.0), 0.0], dtype=np.float64)
    )
    world_previous = make_transform(rotation_world_previous, np.asarray([2.0, 0.0, 3.0]))
    current_previous = make_transform(np.eye(3), np.asarray([0.0, 0.0, -1.0]))

    world_current = accumulate_camera_pose(world_previous, current_previous)

    # The previous camera's local +z axis points along world +x after the yaw.
    np.testing.assert_allclose(world_current[:3, 3], [3.0, 0.0, 3.0], atol=1e-12)
    np.testing.assert_allclose(world_current[:3, :3], rotation_world_previous, atol=1e-12)


def test_pose_accumulator_appends_motion_and_holds_failures() -> None:
    accumulator = PoseAccumulator()
    forward_scene_motion = make_transform(np.eye(3), np.asarray([0.0, 0.0, -1.0]))

    accumulator.append_relative_pose(forward_scene_motion)
    accumulator.hold_pose()
    accumulator.append_relative_pose(forward_scene_motion)

    assert len(accumulator) == 4
    np.testing.assert_allclose(
        accumulator.positions,
        [[0.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, 1.0], [0.0, 0.0, 2.0]],
    )


def test_pose_accumulator_returns_copies() -> None:
    accumulator = PoseAccumulator()

    external_pose = accumulator.current_pose
    external_pose[0, 3] = 99.0

    np.testing.assert_allclose(accumulator.current_pose, np.eye(4))


def test_validate_se3_rejects_non_rotation_block() -> None:
    invalid = np.eye(4)
    invalid[0, 0] = 2.0

    with pytest.raises(ValueError, match="orthonormal"):
        validate_se3(invalid)


def make_odometry() -> MonocularOdometry:
    """Construct a small pipeline for loop-control tests."""
    return MonocularOdometry(
        MonocularOdometryConfig(
            features=ShiTomasiConfig(max_features=100),
            tracking=LucasKanadeConfig(),
            geometry=EssentialMatrixConfig(),
            confidence=ConfidenceConfig(),
            minimum_features=20,
        )
    )


def test_odometry_loop_holds_pose_and_continues_on_failure() -> None:
    result = make_odometry().run(BlankSequence())

    assert len(result.poses) == 4
    assert result.failure_count == 3
    assert [item.status for item in result.diagnostics] == [
        "good",
        "failed",
        "failed",
        "failed",
    ]
    assert all(item.features_redetected for item in result.diagnostics)
    np.testing.assert_allclose(result.poses, np.repeat(np.eye(4)[None], 4, axis=0))


def test_odometry_loop_honors_max_frames() -> None:
    result = make_odometry().run(BlankSequence(frame_count=5), max_frames=2)

    assert len(result.poses) == 2
    assert len(result.diagnostics) == 2


def test_odometry_loop_streams_one_observation_per_frame() -> None:
    observations = []

    result = make_odometry().run(
        BlankSequence(frame_count=3),
        frame_observer=observations.append,
    )

    assert len(observations) == len(result.poses) == 3
    assert [item.diagnostic.frame_index for item in observations] == [0, 1, 2]
    assert observations[0].image.shape == (60, 80)
    assert observations[1].inlier_mask.shape == (0,)
