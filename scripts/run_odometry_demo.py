"""Run the scale-free VO loop on a KITTI prefix and plot diagnostics."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import numpy as np
from tqdm import tqdm

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402

from monovo.config import build_odometry_config, load_yaml_config
from monovo.datasets import KittiSequence
from monovo.odometry import MonocularOdometry


def parse_args() -> argparse.Namespace:
    """Parse demo arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/kitti.yaml"))
    parser.add_argument("--max-frames", type=int, default=100)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("docs/assets/scale-free-trajectory-first-100.png"),
    )
    return parser.parse_args()


def main() -> None:
    """Run VO and create a trajectory/diagnostics figure."""
    args = parse_args()
    settings = load_yaml_config(args.config)
    dataset = settings["dataset"]
    sequence = KittiSequence(
        root=dataset["root"],
        sequence=dataset["sequence"],
        image_directory=dataset["image_directory"],
        calibration_key=dataset["calibration_key"],
        poses_directory=dataset["poses_directory"],
    )
    odometry = MonocularOdometry(build_odometry_config(settings))

    # Keep progress reporting outside the reusable odometry implementation.
    with tqdm(total=1, desc=f"Processing {args.max_frames} frames", unit="run") as progress:
        result = odometry.run(sequence, max_frames=args.max_frames)
        progress.update()

    positions = result.positions
    frame_indices = np.arange(len(result.diagnostics))
    tracked_counts = np.asarray([item.tracked_feature_count for item in result.diagnostics])
    pose_inliers = np.asarray([item.pose_inlier_count for item in result.diagnostics])
    output_features = np.asarray([item.output_feature_count for item in result.diagnostics])
    confidence_scores = np.asarray([item.confidence_score for item in result.diagnostics])
    coverage = np.asarray([item.spatial_coverage for item in result.diagnostics])
    good = np.asarray([item.status == "good" for item in result.diagnostics])
    degraded = np.asarray([item.status == "degraded" for item in result.diagnostics])
    failed = np.asarray([item.status == "failed" for item in result.diagnostics])

    fig = plt.figure(figsize=(15, 8.5), facecolor="#f8f9fb")
    grid = fig.add_gridspec(2, 2, width_ratios=(1.05, 1.0), hspace=0.34, wspace=0.25)
    trajectory_axis = fig.add_subplot(grid[:, 0])
    feature_axis = fig.add_subplot(grid[0, 1])
    confidence_axis = fig.add_subplot(grid[1, 1])
    fig.suptitle(
        f"Scale-free monocular VO — first {len(positions)} frames",
        x=0.055,
        y=0.965,
        ha="left",
        fontsize=18,
        fontweight="bold",
        color="#172033",
    )
    fig.text(
        0.055,
        0.92,
        f"{np.count_nonzero(good)} good • {np.count_nonzero(degraded)} degraded • "
        f"{result.failure_count} failed/held • {result.average_processing_time_ms:.1f} ms/frame",
        color="#5c667a",
        fontsize=10,
    )

    trajectory_axis.plot(positions[:, 0], positions[:, 2], color="#15869b", linewidth=2.0)
    trajectory_axis.scatter(positions[0, 0], positions[0, 2], color="#15869b", s=55, label="start")
    if np.any(degraded):
        trajectory_axis.scatter(
            positions[degraded, 0],
            positions[degraded, 2],
            color="#d89a20",
            s=22,
            label="degraded",
            zorder=3,
        )
    if np.any(failed):
        trajectory_axis.scatter(
            positions[failed, 0],
            positions[failed, 2],
            color="#e45b3c",
            s=28,
            label="held pose",
            zorder=3,
        )
    trajectory_axis.set_aspect("equal", adjustable="datalim")
    trajectory_axis.set_xlabel("world x (arbitrary VO units)")
    trajectory_axis.set_ylabel("world z (arbitrary VO units)")
    trajectory_axis.set_title("Accumulated camera path", loc="left")
    trajectory_axis.grid(color="#dce1e8")
    trajectory_axis.legend(frameon=False)

    feature_axis.plot(
        frame_indices, output_features, label="features carried forward", color="#172033"
    )
    feature_axis.plot(frame_indices, tracked_counts, label="LK tracks", color="#15869b")
    feature_axis.plot(frame_indices, pose_inliers, label="pose inliers", color="#e45b3c")
    feature_axis.axhline(
        settings["features"]["minimum_features"],
        color="#8a93a3",
        linestyle="--",
        linewidth=1,
        label="re-detection threshold",
    )
    feature_axis.set_xlabel("frame index")
    feature_axis.set_ylabel("feature count")
    feature_axis.set_title("Tracking and re-detection", loc="left")
    feature_axis.grid(color="#dce1e8")
    feature_axis.legend(frameon=False, fontsize=8, ncol=2)

    confidence_axis.plot(
        frame_indices, confidence_scores, color="#172033", linewidth=1.5, label="confidence"
    )
    confidence_axis.plot(
        frame_indices, coverage, color="#15869b", linewidth=1.2, label="spatial coverage"
    )
    confidence_axis.axhline(
        settings["confidence"]["good_score_threshold"],
        color="#8a93a3",
        linestyle="--",
        linewidth=1,
        label="good threshold",
    )
    confidence_axis.scatter(
        frame_indices[degraded],
        confidence_scores[degraded],
        color="#d89a20",
        s=22,
        zorder=3,
        label="degraded frame",
    )
    confidence_axis.scatter(
        frame_indices[failed],
        confidence_scores[failed],
        color="#e45b3c",
        s=25,
        zorder=3,
        label="failed frame",
    )
    confidence_axis.set_ylim(-0.03, 1.05)
    confidence_axis.set_xlabel("frame index")
    confidence_axis.set_ylabel("score / image fraction")
    confidence_axis.set_title("Failure awareness", loc="left")
    confidence_axis.grid(color="#dce1e8")
    confidence_axis.legend(frameon=False, fontsize=8, ncol=2)

    fig.text(
        0.5,
        0.02,
        "Raw trajectory: no ground-truth scale or alignment is used.",
        ha="center",
        color="#5c667a",
        fontsize=9,
    )
    fig.subplots_adjust(left=0.07, right=0.97, top=0.86, bottom=0.09)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=180, facecolor="#f8f9fb")
    plt.close(fig)

    print(f"Processed frames: {len(result.poses)}")
    print(f"Good frames: {np.count_nonzero(good)}")
    print(f"Degraded frames: {np.count_nonzero(degraded)}")
    print(f"Failed/held poses: {result.failure_count}")
    print(f"Average processing time: {result.average_processing_time_ms:.2f} ms/frame")
    print(f"Saved visualization to {args.output}")


if __name__ == "__main__":
    main()
