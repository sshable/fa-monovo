"""Run a KITTI prefix and visualize post-hoc Sim(3) trajectory evaluation."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402

from monovo.config import build_odometry_config, load_yaml_config
from monovo.datasets import KittiSequence
from monovo.evaluation import evaluate_trajectory
from monovo.odometry import MonocularOdometry


def parse_args() -> argparse.Namespace:
    """Parse evaluation-demo arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/kitti.yaml"))
    parser.add_argument("--max-frames", type=int, default=100)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("docs/assets/sim3-evaluation-first-100.png"),
    )
    return parser.parse_args()


def main() -> None:
    """Estimate, align, evaluate, and plot one KITTI sequence prefix."""
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
    run = odometry.run(sequence, max_frames=args.max_frames)
    ground_truth = sequence.selected_camera_ground_truth_poses()[: len(run.poses)]
    evaluation = evaluate_trajectory(run.poses, ground_truth, run.diagnostics)

    aligned_positions = evaluation.aligned_estimated_poses[:, :3, 3]
    truth_positions = ground_truth[:, :3, 3]
    frame_indices = np.arange(len(run.poses))

    fig, axes = plt.subplots(1, 2, figsize=(15, 6.5), facecolor="#f8f9fb")
    fig.suptitle(
        f"Post-hoc Sim(3) evaluation — first {len(run.poses)} frames",
        x=0.055,
        y=0.965,
        ha="left",
        fontsize=18,
        fontweight="bold",
        color="#172033",
    )
    direction_mean = evaluation.mean_translation_direction_error_degrees
    rotation_mean = evaluation.mean_relative_rotation_error_degrees
    fig.text(
        0.055,
        0.92,
        f"ATE RMSE {evaluation.ate_rmse:.2f} m • rotation error {rotation_mean:.2f}° • "
        f"translation-direction error {direction_mean:.2f}° • scale "
        f"{evaluation.alignment.scale:.3f} m/VO-unit",
        color="#5c667a",
        fontsize=10,
    )

    axes[0].plot(
        truth_positions[:, 0],
        truth_positions[:, 2],
        color="#172033",
        linewidth=2.2,
        label="KITTI ground truth",
    )
    axes[0].plot(
        aligned_positions[:, 0],
        aligned_positions[:, 2],
        color="#15869b",
        linewidth=1.8,
        label="VO after one Sim(3)",
    )
    axes[0].set_aspect("equal", adjustable="datalim")
    axes[0].set_xlabel("KITTI world x (m)")
    axes[0].set_ylabel("KITTI world z (m)")
    axes[0].set_title("Aligned trajectory", loc="left")
    axes[0].grid(color="#dce1e8")
    axes[0].legend(frameon=False)

    axes[1].plot(
        frame_indices,
        evaluation.translation_direction_errors_degrees,
        color="#e45b3c",
        linewidth=1.3,
        label="translation direction",
    )
    axes[1].plot(
        frame_indices,
        evaluation.relative_rotation_errors_degrees,
        color="#15869b",
        linewidth=1.3,
        label="relative rotation",
    )
    axes[1].set_xlabel("frame index")
    axes[1].set_ylabel("angular error (degrees)")
    axes[1].set_title("Adjacent-frame motion errors", loc="left")
    axes[1].grid(color="#dce1e8")
    axes[1].legend(frameon=False)

    fig.text(
        0.5,
        0.02,
        "Ground truth is used only here, after inference; one global similarity "
        "cannot remove drift.",
        ha="center",
        color="#5c667a",
        fontsize=9,
    )
    fig.subplots_adjust(left=0.07, right=0.97, top=0.86, bottom=0.12, wspace=0.25)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=180, facecolor="#f8f9fb")
    plt.close(fig)

    print(f"Processed frames: {len(run.poses)}")
    print(f"Sim(3) scale: {evaluation.alignment.scale:.6f} m/VO-unit")
    print(f"ATE RMSE: {evaluation.ate_rmse:.6f} m")
    print(f"Mean relative rotation error: {rotation_mean:.6f} deg")
    print(f"Mean translation-direction error: {direction_mean:.6f} deg")
    print(f"Mean RANSAC inlier ratio: {evaluation.mean_ransac_inlier_ratio:.3%}")
    print(f"Saved visualization to {args.output}")


if __name__ == "__main__":
    main()
