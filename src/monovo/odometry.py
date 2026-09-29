"""SE(3) pose composition for monocular visual odometry."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from time import perf_counter
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

from monovo.confidence import (
    ConfidenceConfig,
    ConfidenceSignals,
    assess_confidence,
    spatial_coverage,
    vector_angle_degrees,
)
from monovo.features import ShiTomasiConfig, detect_shi_tomasi, replenish_features
from monovo.geometry import (
    EssentialMatrixConfig,
    estimate_relative_pose,
    rotation_angle_degrees,
)
from monovo.tracking import LucasKanadeConfig, track_features

FloatArray = NDArray[np.float64]


class ImageSequence(Protocol):
    """Minimum dataset interface required by the odometry loop."""

    def __len__(self) -> int: ...

    @property
    def intrinsic_matrix(self) -> FloatArray: ...

    def load_image(self, index: int) -> NDArray[np.uint8]: ...


@dataclass(frozen=True)
class MonocularOdometryConfig:
    """Configuration groups used by the multi-frame VO loop."""

    features: ShiTomasiConfig
    tracking: LucasKanadeConfig
    geometry: EssentialMatrixConfig
    confidence: ConfidenceConfig
    minimum_features: int = 500

    def __post_init__(self) -> None:
        """Validate cross-component feature-count settings."""
        if self.minimum_features <= 0:
            raise ValueError("minimum_features must be positive")
        if self.minimum_features > self.features.max_features:
            raise ValueError("minimum_features cannot exceed features.max_features")


@dataclass(frozen=True)
class FrameDiagnostics:
    """Measurements and outcome recorded for one processed frame."""

    frame_index: int
    status: str
    confidence_score: float
    confidence_reasons: tuple[str, ...]
    failure_reason: str | None
    input_feature_count: int
    tracked_feature_count: int
    tracking_ratio: float
    ransac_inlier_count: int
    ransac_inlier_ratio: float
    pose_inlier_count: int
    pose_inlier_ratio: float
    median_flow_pixels: float | None
    spatial_coverage: float
    rotation_degrees: float | None
    translation_direction_change_degrees: float | None
    output_feature_count: int
    features_redetected: bool
    processing_time_ms: float


@dataclass(frozen=True)
class OdometryRunResult:
    """Accumulated trajectory and frame-level processing diagnostics."""

    poses: FloatArray
    diagnostics: tuple[FrameDiagnostics, ...]

    @property
    def positions(self) -> FloatArray:
        """Return estimated camera centers in the arbitrary world frame."""
        return self.poses[:, :3, 3]

    @property
    def failure_count(self) -> int:
        """Return the number of frames whose relative pose was held."""
        return sum(item.status == "failed" for item in self.diagnostics)

    @property
    def average_processing_time_ms(self) -> float:
        """Return mean processing time, excluding frame-zero initialization."""
        if len(self.diagnostics) <= 1:
            return 0.0
        return float(np.mean([item.processing_time_ms for item in self.diagnostics[1:]]))


@dataclass(frozen=True)
class FrameObservation:
    """Transient frame data exposed to visualizers without retaining all images."""

    image: NDArray[np.uint8]
    source_points: NDArray[np.float32]
    target_points: NDArray[np.float32]
    inlier_mask: NDArray[np.bool_]
    pose_world_camera: FloatArray
    diagnostic: FrameDiagnostics


class MonocularOdometry:
    """Run feature-based monocular visual odometry over an image sequence."""

    def __init__(self, config: MonocularOdometryConfig) -> None:
        self.config = config

    def run(
        self,
        sequence: ImageSequence,
        max_frames: int | None = None,
        frame_observer: Callable[[FrameObservation], None] | None = None,
    ) -> OdometryRunResult:
        """Process a sequence without accessing ground truth.

        ``frame_observer`` receives transient arrays for streaming visualization.
        The estimator does not retain image frames in the returned result.
        """
        available_frames = len(sequence)
        if available_frames == 0:
            raise ValueError("Cannot run odometry on an empty sequence")
        if max_frames is not None and max_frames <= 0:
            raise ValueError("max_frames must be positive when provided")
        frame_count = available_frames if max_frames is None else min(max_frames, available_frames)

        accumulator = PoseAccumulator()
        previous_image = sequence.load_image(0)
        active_features = detect_shi_tomasi(previous_image, self.config.features)
        previous_world_motion_direction: FloatArray | None = None
        initial_diagnostic = FrameDiagnostics(
            frame_index=0,
            status="good",
            confidence_score=1.0,
            confidence_reasons=(),
            failure_reason=None,
            input_feature_count=0,
            tracked_feature_count=0,
            tracking_ratio=0.0,
            ransac_inlier_count=0,
            ransac_inlier_ratio=0.0,
            pose_inlier_count=0,
            pose_inlier_ratio=0.0,
            median_flow_pixels=None,
            spatial_coverage=spatial_coverage(
                active_features,
                previous_image.shape,
                self.config.confidence.coverage_grid_rows,
                self.config.confidence.coverage_grid_columns,
            ),
            rotation_degrees=None,
            translation_direction_change_degrees=None,
            output_feature_count=len(active_features),
            features_redetected=True,
            processing_time_ms=0.0,
        )
        diagnostics = [initial_diagnostic]
        if frame_observer is not None:
            empty_points = np.empty((0, 2), dtype=np.float32)
            frame_observer(
                FrameObservation(
                    image=previous_image,
                    source_points=empty_points,
                    target_points=empty_points,
                    inlier_mask=np.empty(0, dtype=np.bool_),
                    pose_world_camera=accumulator.current_pose,
                    diagnostic=initial_diagnostic,
                )
            )

        for frame_index in range(1, frame_count):
            start_time = perf_counter()
            current_image = sequence.load_image(frame_index)
            input_feature_count = len(active_features)
            tracks = track_features(
                previous_image,
                current_image,
                active_features,
                self.config.tracking,
            )
            if tracks.valid_count:
                flow = tracks.valid_target_points - tracks.valid_source_points
                median_flow = float(np.median(np.linalg.norm(flow, axis=1)))
            else:
                median_flow = None

            pose = estimate_relative_pose(
                tracks.valid_source_points,
                tracks.valid_target_points,
                sequence.intrinsic_matrix,
                self.config.geometry,
            )
            ransac_inlier_ratio = (
                pose.ransac_inlier_count / tracks.valid_count if tracks.valid_count else 0.0
            )
            inlier_points = (
                tracks.valid_target_points[pose.inlier_mask]
                if pose.success
                else np.empty((0, 2), dtype=np.float32)
            )
            coverage = spatial_coverage(
                inlier_points,
                current_image.shape,
                self.config.confidence.coverage_grid_rows,
                self.config.confidence.coverage_grid_columns,
            )

            rotation_degrees: float | None = None
            direction_change_degrees: float | None = None
            current_world_motion_direction: FloatArray | None = None
            if pose.success:
                assert pose.rotation_current_previous is not None
                assert pose.translation_direction_current is not None
                rotation_degrees = rotation_angle_degrees(pose.rotation_current_previous)
                motion_direction_previous = -(
                    pose.rotation_current_previous.T @ pose.translation_direction_current
                )
                current_world_motion_direction = (
                    accumulator.current_pose[:3, :3] @ motion_direction_previous
                )
                if previous_world_motion_direction is not None:
                    direction_change_degrees = vector_angle_degrees(
                        previous_world_motion_direction,
                        current_world_motion_direction,
                    )

            assessment = assess_confidence(
                ConfidenceSignals(
                    pose_success=pose.success,
                    tracked_feature_count=tracks.valid_count,
                    ransac_inlier_count=pose.ransac_inlier_count,
                    ransac_inlier_ratio=ransac_inlier_ratio,
                    median_flow_pixels=median_flow,
                    spatial_coverage=coverage,
                    rotation_degrees=rotation_degrees,
                    translation_direction_change_degrees=direction_change_degrees,
                    estimation_failure_reason=pose.reason,
                ),
                self.config.confidence,
            )
            pose_accepted = pose.success and assessment.state != "failed"
            if pose_accepted:
                accumulator.append_relative_pose(pose.transform_current_previous)
                retained_features = inlier_points
                failure_reason = None
                previous_world_motion_direction = current_world_motion_direction
            else:
                accumulator.hold_pose()
                retained_features = np.empty((0, 2), dtype=np.float32)
                failure_reason = "; ".join(assessment.reasons)

            features_redetected = (
                not pose_accepted or len(retained_features) < self.config.minimum_features
            )
            if features_redetected:
                active_features = replenish_features(
                    current_image,
                    retained_features,
                    self.config.features,
                )
            else:
                active_features = np.ascontiguousarray(retained_features, dtype=np.float32)

            elapsed_ms = (perf_counter() - start_time) * 1000.0
            diagnostic = FrameDiagnostics(
                frame_index=frame_index,
                status=assessment.state,
                confidence_score=assessment.score,
                confidence_reasons=assessment.reasons,
                failure_reason=failure_reason,
                input_feature_count=input_feature_count,
                tracked_feature_count=tracks.valid_count,
                tracking_ratio=tracks.inlier_ratio,
                ransac_inlier_count=pose.ransac_inlier_count,
                ransac_inlier_ratio=ransac_inlier_ratio,
                pose_inlier_count=pose.inlier_count,
                pose_inlier_ratio=pose.inlier_ratio,
                median_flow_pixels=median_flow,
                spatial_coverage=coverage,
                rotation_degrees=rotation_degrees,
                translation_direction_change_degrees=direction_change_degrees,
                output_feature_count=len(active_features),
                features_redetected=features_redetected,
                processing_time_ms=elapsed_ms,
            )
            diagnostics.append(diagnostic)
            if frame_observer is not None:
                frame_observer(
                    FrameObservation(
                        image=current_image,
                        source_points=tracks.valid_source_points,
                        target_points=tracks.valid_target_points,
                        inlier_mask=pose.inlier_mask,
                        pose_world_camera=accumulator.current_pose,
                        diagnostic=diagnostic,
                    )
                )
            previous_image = current_image

        return OdometryRunResult(
            poses=accumulator.poses,
            diagnostics=tuple(diagnostics),
        )


def make_transform(rotation: NDArray[np.floating], translation: NDArray[np.floating]) -> FloatArray:
    """Construct a 4x4 rigid transform from rotation and translation."""
    rotation_matrix = np.asarray(rotation, dtype=np.float64)
    translation_vector = np.asarray(translation, dtype=np.float64).reshape(-1)
    if rotation_matrix.shape != (3, 3):
        raise ValueError(f"Rotation must have shape (3, 3), got {rotation_matrix.shape}")
    if translation_vector.shape != (3,):
        raise ValueError(f"Translation must contain 3 values, got {translation_vector.shape}")

    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rotation_matrix
    transform[:3, 3] = translation_vector
    validate_se3(transform)
    return transform


def invert_se3(transform: NDArray[np.floating]) -> FloatArray:
    """Invert an SE(3) transform using its rigid-body structure."""
    matrix = np.asarray(transform, dtype=np.float64)
    validate_se3(matrix)
    rotation = matrix[:3, :3]
    translation = matrix[:3, 3]

    inverse = np.eye(4, dtype=np.float64)
    inverse[:3, :3] = rotation.T
    inverse[:3, 3] = -(rotation.T @ translation)
    return inverse


def accumulate_camera_pose(
    transform_world_previous: NDArray[np.floating],
    transform_current_previous: NDArray[np.floating],
) -> FloatArray:
    """Accumulate one relative pose into a camera-to-world trajectory.

    ``recoverPose`` supplies ``T_current_previous``, which maps coordinates from
    the previous camera frame into the current camera frame. The stored global
    pose is ``T_world_camera``, which maps camera coordinates into the world.
    Therefore:

    ``T_world_current = T_world_previous @ inverse(T_current_previous)``.
    """
    world_previous = np.asarray(transform_world_previous, dtype=np.float64)
    current_previous = np.asarray(transform_current_previous, dtype=np.float64)
    validate_se3(world_previous)
    validate_se3(current_previous)
    world_current = world_previous @ invert_se3(current_previous)
    validate_se3(world_current)
    return world_current


def validate_se3(transform: NDArray[np.floating], tolerance: float = 1e-6) -> None:
    """Validate homogeneous shape, rotation orthogonality, and determinant."""
    matrix = np.asarray(transform, dtype=np.float64)
    if matrix.shape != (4, 4):
        raise ValueError(f"SE(3) transform must have shape (4, 4), got {matrix.shape}")
    if not np.isfinite(matrix).all():
        raise ValueError("SE(3) transform must contain only finite values")
    if not np.allclose(matrix[3], [0.0, 0.0, 0.0, 1.0], atol=tolerance):
        raise ValueError("SE(3) transform must end with homogeneous row [0, 0, 0, 1]")

    rotation = matrix[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=tolerance):
        raise ValueError("SE(3) rotation block must be orthonormal")
    if not np.isclose(np.linalg.det(rotation), 1.0, atol=tolerance):
        raise ValueError("SE(3) rotation block must have determinant +1")


class PoseAccumulator:
    """Maintain a sequence of camera-to-world poses.

    The first pose defaults to identity, making the first camera coordinate
    system the arbitrary world frame. In scale-free mode every recovered
    translation has unit length, so positions are expressed in arbitrary VO
    units rather than metres.
    """

    def __init__(self, initial_pose: NDArray[np.floating] | None = None) -> None:
        pose = np.eye(4, dtype=np.float64) if initial_pose is None else np.asarray(initial_pose)
        pose = np.asarray(pose, dtype=np.float64)
        validate_se3(pose)
        self._poses: list[FloatArray] = [pose.copy()]

    def append_relative_pose(self, transform_current_previous: NDArray[np.floating]) -> FloatArray:
        """Accumulate a successful relative pose and return the new global pose."""
        next_pose = accumulate_camera_pose(self._poses[-1], transform_current_previous)
        self._poses.append(next_pose)
        return next_pose.copy()

    def hold_pose(self) -> FloatArray:
        """Append the previous global pose when relative estimation fails."""
        held_pose = self._poses[-1].copy()
        self._poses.append(held_pose)
        return held_pose.copy()

    @property
    def current_pose(self) -> FloatArray:
        """Return the most recent ``T_world_camera`` pose."""
        return self._poses[-1].copy()

    @property
    def poses(self) -> FloatArray:
        """Return all accumulated poses with shape ``(N, 4, 4)``."""
        return np.stack(self._poses)

    @property
    def positions(self) -> FloatArray:
        """Return all camera centers in world coordinates."""
        return self.poses[:, :3, 3]

    def __len__(self) -> int:
        """Return the number of stored frame poses, including frame zero."""
        return len(self._poses)
