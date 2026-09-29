"""Trajectory plots for scale-free and post-hoc aligned VO results."""

from __future__ import annotations

from pathlib import Path

import cv2
import matplotlib
import numpy as np
from numpy.typing import NDArray

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402

from monovo.evaluation import EvaluationResult
from monovo.odometry import FrameDiagnostics, FrameObservation

STATE_COLORS = {
    "good": (83, 176, 82),
    "degraded": (70, 173, 240),
    "failed": (68, 68, 220),
}


class DemoVideoWriter:
    """Stream annotated odometry observations into an MP4 file."""

    def __init__(
        self,
        path: str | Path,
        fps: float,
        panel_width: int = 420,
        maximum_tracks: int = 300,
    ) -> None:
        if not np.isfinite(fps) or fps <= 0.0:
            raise ValueError("video fps must be positive and finite")
        if panel_width < 240:
            raise ValueError("panel_width must be at least 240 pixels")
        if maximum_tracks <= 0:
            raise ValueError("maximum_tracks must be positive")
        self.path = Path(path)
        self.fps = float(fps)
        self.panel_width = panel_width
        self.maximum_tracks = maximum_tracks
        self._positions: list[NDArray[np.float64]] = []
        self._writer: cv2.VideoWriter | None = None

    def __call__(self, observation: FrameObservation) -> None:
        """Render and append one observation supplied by the VO loop."""
        self._positions.append(observation.pose_world_camera[:3, 3].copy())
        frame = render_demo_frame(
            observation,
            np.asarray(self._positions),
            panel_width=self.panel_width,
            maximum_tracks=self.maximum_tracks,
        )
        if self._writer is None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            height, width = frame.shape[:2]
            writer = cv2.VideoWriter(
                str(self.path),
                cv2.VideoWriter_fourcc(*"mp4v"),
                self.fps,
                (width, height),
            )
            if not writer.isOpened():
                writer.release()
                raise OSError(f"Could not open MP4 video writer: {self.path}")
            self._writer = writer
        self._writer.write(frame)

    def close(self) -> None:
        """Flush and close the video container."""
        if self._writer is not None:
            self._writer.release()
            self._writer = None

    def __enter__(self) -> DemoVideoWriter:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


def render_demo_frame(
    observation: FrameObservation,
    trajectory_positions: NDArray[np.floating],
    panel_width: int = 420,
    maximum_tracks: int = 300,
) -> NDArray[np.uint8]:
    """Compose one feature-track frame and a scale-free trajectory panel."""
    image = np.asarray(observation.image)
    if image.ndim != 2 or image.dtype != np.uint8:
        raise ValueError("observation image must be an 8-bit grayscale frame")
    source = np.asarray(observation.source_points, dtype=np.float32)
    target = np.asarray(observation.target_points, dtype=np.float32)
    inliers = np.asarray(observation.inlier_mask, dtype=np.bool_)
    if source.shape != target.shape or source.ndim != 2 or source.shape[1:] != (2,):
        raise ValueError("source_points and target_points must have matching (N, 2) shapes")
    if inliers.shape != (len(source),):
        raise ValueError("inlier_mask length must match the correspondence count")
    positions = np.asarray(trajectory_positions, dtype=np.float64)
    if positions.ndim != 2 or positions.shape[1:] != (3,) or len(positions) == 0:
        raise ValueError("trajectory_positions must have nonempty shape (N, 3)")

    camera_view = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    _draw_tracks(camera_view, source, target, inliers, maximum_tracks)
    _draw_status_overlay(camera_view, observation.diagnostic)
    trajectory_panel = _draw_trajectory_panel(image.shape[0], panel_width, positions)
    return np.ascontiguousarray(np.hstack((camera_view, trajectory_panel)))


def _draw_tracks(
    canvas: NDArray[np.uint8],
    source: NDArray[np.float32],
    target: NDArray[np.float32],
    inliers: NDArray[np.bool_],
    maximum_tracks: int,
) -> None:
    stride = max(1, int(np.ceil(len(source) / maximum_tracks)))
    for index in range(0, len(source), stride):
        start = tuple(np.rint(source[index]).astype(int))
        end = tuple(np.rint(target[index]).astype(int))
        color = (68, 202, 121) if inliers[index] else (44, 153, 255)
        cv2.line(canvas, start, end, color, 1, cv2.LINE_AA)
        cv2.circle(canvas, end, 2, color, -1, cv2.LINE_AA)


def _draw_status_overlay(canvas: NDArray[np.uint8], diagnostic: FrameDiagnostics) -> None:
    overlay = canvas.copy()
    cv2.rectangle(overlay, (0, 0), (canvas.shape[1], 82), (15, 23, 37), -1)
    cv2.addWeighted(overlay, 0.82, canvas, 0.18, 0.0, canvas)
    color = STATE_COLORS[diagnostic.status]
    cv2.putText(
        canvas,
        f"FRAME {diagnostic.frame_index:06d}",
        (18, 29),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (245, 247, 250),
        2,
        cv2.LINE_AA,
    )
    cv2.rectangle(canvas, (210, 10), (330, 36), color, -1)
    cv2.putText(
        canvas,
        diagnostic.status.upper(),
        (222, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        canvas,
        f"confidence {diagnostic.confidence_score:.2f}   "
        f"tracked {diagnostic.tracked_feature_count}   "
        f"pose inliers {diagnostic.pose_inlier_count}   "
        f"RANSAC ratio {diagnostic.ransac_inlier_ratio:.1%}",
        (18, 63),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        (232, 237, 244),
        1,
        cv2.LINE_AA,
    )


def _draw_trajectory_panel(
    height: int,
    width: int,
    positions: NDArray[np.float64],
) -> NDArray[np.uint8]:
    panel = np.full((height, width, 3), (27, 34, 48), dtype=np.uint8)
    cv2.putText(
        panel,
        "SCALE-FREE TRAJECTORY",
        (24, 34),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.62,
        (242, 245, 249),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        panel,
        "top view: world x / z (VO units)",
        (24, 58),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42,
        (157, 169, 188),
        1,
        cv2.LINE_AA,
    )
    left, right, top, bottom = 28, width - 28, 82, height - 42
    cv2.rectangle(panel, (left, top), (right, bottom), (53, 64, 82), 1)

    path = positions[:, [0, 2]]
    minimum = path.min(axis=0)
    maximum = path.max(axis=0)
    extent = np.maximum(maximum - minimum, 1.0)
    padding = np.maximum(extent * 0.12, 0.5)
    minimum -= padding
    maximum += padding
    extent = maximum - minimum
    scale = min((right - left) / extent[0], (bottom - top) / extent[1])
    center = (minimum + maximum) / 2.0
    pixel_center = np.asarray([(left + right) / 2.0, (top + bottom) / 2.0])
    pixels = (path - center) * scale
    pixels[:, 1] *= -1.0
    pixels += pixel_center
    pixel_path = np.rint(pixels).astype(np.int32)
    if len(pixel_path) > 1:
        cv2.polylines(panel, [pixel_path], False, (181, 134, 45), 2, cv2.LINE_AA)
    cv2.circle(panel, tuple(pixel_path[0]), 5, (83, 176, 82), -1, cv2.LINE_AA)
    cv2.circle(panel, tuple(pixel_path[-1]), 5, (68, 68, 220), -1, cv2.LINE_AA)
    accepted_steps = int(
        np.count_nonzero(np.linalg.norm(np.diff(positions, axis=0), axis=1) > 1e-9)
    )
    cv2.putText(
        panel,
        f"accepted pose updates: {accepted_steps}",
        (24, height - 14),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42,
        (157, 169, 188),
        1,
        cv2.LINE_AA,
    )
    return panel


def save_trajectory_plot(
    path: str | Path,
    estimated_poses: NDArray[np.floating],
    diagnostics: tuple[FrameDiagnostics, ...],
    sequence: str,
    evaluation: EvaluationResult | None = None,
    ground_truth_poses: NDArray[np.floating] | None = None,
) -> Path:
    """Save a top-down trajectory and per-frame confidence overview."""
    raw_poses = np.asarray(estimated_poses, dtype=np.float64)
    if raw_poses.ndim != 3 or raw_poses.shape[1:] != (4, 4):
        raise ValueError(f"estimated_poses must have shape (N, 4, 4), got {raw_poses.shape}")
    if len(diagnostics) != len(raw_poses):
        raise ValueError("diagnostics and estimated_poses must have equal lengths")

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    plot_poses = evaluation.aligned_estimated_poses if evaluation is not None else raw_poses
    positions = plot_poses[:, :3, 3]
    frames = np.arange(len(raw_poses))
    scores = np.asarray([item.confidence_score for item in diagnostics])

    figure, axes = plt.subplots(1, 2, figsize=(14, 6), facecolor="#f8f9fb")
    mode_label = "post-hoc Sim(3) aligned" if evaluation is not None else "scale-free"
    figure.suptitle(
        f"KITTI sequence {sequence} — {mode_label} monocular VO",
        x=0.06,
        y=0.965,
        ha="left",
        fontsize=17,
        fontweight="bold",
        color="#172033",
    )

    axes[0].plot(positions[:, 0], positions[:, 2], color="#15869b", linewidth=2.0)
    axes[0].scatter(positions[0, 0], positions[0, 2], color="#198754", s=45, label="start")
    axes[0].scatter(positions[-1, 0], positions[-1, 2], color="#dc3545", s=45, label="end")
    if evaluation is not None and ground_truth_poses is not None:
        truth = np.asarray(ground_truth_poses, dtype=np.float64)[:, :3, 3]
        axes[0].plot(
            truth[:, 0],
            truth[:, 2],
            color="#172033",
            linewidth=1.8,
            linestyle="--",
            label="ground truth",
        )
    unit = "metres" if evaluation is not None else "arbitrary VO units"
    axes[0].set_xlabel(f"world x ({unit})")
    axes[0].set_ylabel(f"world z ({unit})")
    axes[0].set_title("Camera path", loc="left")
    axes[0].set_aspect("equal", adjustable="datalim")
    axes[0].grid(color="#dce1e8")
    axes[0].legend(frameon=False)

    colors = {"good": "#198754", "degraded": "#f0ad4e", "failed": "#dc3545"}
    axes[1].plot(frames, scores, color="#8090a5", linewidth=1.0, alpha=0.55)
    for state, color in colors.items():
        mask = np.asarray([item.status == state for item in diagnostics])
        if np.any(mask):
            axes[1].scatter(frames[mask], scores[mask], color=color, s=16, label=state)
    axes[1].axhline(0.65, color="#a8b1bf", linewidth=1.0, linestyle="--")
    axes[1].set_ylim(-0.03, 1.03)
    axes[1].set_xlabel("frame index")
    axes[1].set_ylabel("rule-based confidence score")
    axes[1].set_title("Failure awareness", loc="left")
    axes[1].grid(color="#dce1e8")
    axes[1].legend(frameon=False)

    if evaluation is None:
        note = "Each accepted translation has unit length; metric scale is unknown."
    else:
        note = (
            f"Ground truth used only after inference • ATE RMSE {evaluation.ate_rmse:.2f} m • "
            f"Sim(3) scale {evaluation.alignment.scale:.3f} m/VO-unit"
        )
    figure.text(0.5, 0.02, note, ha="center", color="#5c667a", fontsize=9)
    figure.subplots_adjust(left=0.08, right=0.97, top=0.87, bottom=0.14, wspace=0.24)
    figure.savefig(destination, dpi=170, facecolor="#f8f9fb")
    plt.close(figure)
    return destination
