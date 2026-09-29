"""Tests for streamed video-frame rendering."""

import numpy as np

from monovo.odometry import FrameDiagnostics, FrameObservation
from monovo.visualization import render_demo_frame


def test_render_demo_frame_combines_camera_and_trajectory_panel() -> None:
    diagnostic = FrameDiagnostics(
        frame_index=4,
        status="good",
        confidence_score=0.9,
        confidence_reasons=(),
        failure_reason=None,
        input_feature_count=20,
        tracked_feature_count=2,
        tracking_ratio=0.1,
        ransac_inlier_count=1,
        ransac_inlier_ratio=0.5,
        pose_inlier_count=1,
        pose_inlier_ratio=0.5,
        median_flow_pixels=3.0,
        spatial_coverage=0.25,
        rotation_degrees=1.0,
        translation_direction_change_degrees=2.0,
        output_feature_count=10,
        features_redetected=True,
        processing_time_ms=5.0,
    )
    observation = FrameObservation(
        image=np.zeros((240, 160), dtype=np.uint8),
        source_points=np.asarray([[20.0, 30.0], [50.0, 60.0]], dtype=np.float32),
        target_points=np.asarray([[24.0, 32.0], [55.0, 61.0]], dtype=np.float32),
        inlier_mask=np.asarray([True, False]),
        pose_world_camera=np.eye(4),
        diagnostic=diagnostic,
    )
    trajectory = np.asarray([[0.0, 0.0, 0.0], [0.1, 0.0, 1.0]])

    frame = render_demo_frame(observation, trajectory, panel_width=240)

    assert frame.shape == (240, 400, 3)
    assert frame.dtype == np.uint8
    assert np.count_nonzero(frame) > 0
