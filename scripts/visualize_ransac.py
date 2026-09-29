"""Visualize essential-matrix RANSAC on a pair of KITTI frames."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import numpy as np
import yaml

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402

from monovo.datasets import KittiSequence
from monovo.features import ShiTomasiConfig, detect_shi_tomasi
from monovo.geometry import EssentialMatrixConfig, estimate_relative_pose, rotation_angle_degrees
from monovo.tracking import LucasKanadeConfig, track_features


def parse_args() -> argparse.Namespace:
    """Parse visualization arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/kitti.yaml"))
    parser.add_argument("--frame", type=int, default=0)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("docs/assets/essential-ransac-inliers-000000-000001.png"),
    )
    return parser.parse_args()


def draw_correspondences(
    axis: plt.Axes,
    image: np.ndarray,
    source: np.ndarray,
    target: np.ndarray,
    mask: np.ndarray,
    line_color: str,
    point_color: str,
    title: str,
) -> None:
    """Draw selected two-frame correspondences over the current image."""
    axis.imshow(image, cmap="gray", vmin=0, vmax=255)
    selected_source = source[mask]
    selected_target = target[mask]
    for old, new in zip(selected_source, selected_target, strict=True):
        axis.plot(
            [old[0], new[0]],
            [old[1], new[1]],
            color=line_color,
            linewidth=0.55,
            alpha=0.65,
        )
    axis.plot(
        selected_target[:, 0],
        selected_target[:, 1],
        ".",
        color=point_color,
        markersize=2.6,
    )
    axis.set_title(title, loc="left", fontsize=11, color="#172033")
    axis.set_axis_off()


def main() -> None:
    """Run tracking, robust geometry, and visualization for one frame pair."""
    args = parse_args()
    settings = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    dataset = settings["dataset"]
    feature_settings = settings["features"]
    tracking_settings = settings["tracking"]
    geometry_settings = settings["geometry"]
    sequence = KittiSequence(
        root=dataset["root"],
        sequence=dataset["sequence"],
        image_directory=dataset["image_directory"],
        calibration_key=dataset["calibration_key"],
        poses_directory=dataset["poses_directory"],
    )
    previous = sequence.load_image(args.frame)
    current = sequence.load_image(args.frame + 1)
    features = detect_shi_tomasi(
        previous,
        ShiTomasiConfig(
            max_features=feature_settings["max_features"],
            quality_level=feature_settings["quality_level"],
            minimum_distance=feature_settings["minimum_distance"],
            block_size=feature_settings["block_size"],
        ),
    )
    tracks = track_features(
        previous,
        current,
        features,
        LucasKanadeConfig(
            window_size=tracking_settings["window_size"],
            pyramid_levels=tracking_settings["pyramid_levels"],
            criteria_max_iterations=tracking_settings["criteria_max_iterations"],
            criteria_epsilon=tracking_settings["criteria_epsilon"],
            forward_backward_threshold=tracking_settings["forward_backward_threshold"],
        ),
    )
    pose = estimate_relative_pose(
        tracks.valid_source_points,
        tracks.valid_target_points,
        sequence.intrinsic_matrix,
        EssentialMatrixConfig(
            ransac_probability=geometry_settings["ransac_probability"],
            ransac_threshold_pixels=geometry_settings["ransac_threshold_pixels"],
            ransac_max_iterations=geometry_settings["ransac_max_iterations"],
            minimum_correspondences=geometry_settings["minimum_correspondences"],
            minimum_inliers=geometry_settings["minimum_inliers"],
        ),
    )
    if not pose.success:
        raise RuntimeError(pose.reason)
    assert pose.rotation_current_previous is not None
    assert pose.translation_direction_current is not None

    fig, axes = plt.subplots(2, 1, figsize=(14, 8.8), facecolor="#f8f9fb")
    fig.suptitle(
        f"Essential-matrix RANSAC — frames {args.frame:06d} → {args.frame + 1:06d}",
        x=0.045,
        y=0.975,
        ha="left",
        fontsize=18,
        fontweight="bold",
        color="#172033",
    )
    translation = pose.translation_direction_current
    fig.text(
        0.045,
        0.938,
        f"{pose.inlier_count:,} / {tracks.valid_count:,} pose inliers "
        f"({pose.inlier_ratio:.1%}) • rotation "
        f"{rotation_angle_degrees(pose.rotation_current_previous):.2f}° • "
        f"translation direction [{translation[0]:+.2f}, {translation[1]:+.2f}, "
        f"{translation[2]:+.2f}]",
        color="#5c667a",
        fontsize=10,
    )

    rejected_mask = ~pose.inlier_mask
    draw_correspondences(
        axes[0],
        current,
        tracks.valid_source_points,
        tracks.valid_target_points,
        rejected_mask,
        line_color="#e45b3c",
        point_color="#e45b3c",
        title=f"Rejected by RANSAC or cheirality — {np.count_nonzero(rejected_mask):,}",
    )
    draw_correspondences(
        axes[1],
        current,
        tracks.valid_source_points,
        tracks.valid_target_points,
        pose.inlier_mask,
        line_color="#15869b",
        point_color="#ff6b45",
        title=f"Rigid-scene pose inliers — {pose.inlier_count:,}",
    )
    fig.text(
        0.5,
        0.018,
        "RANSAC keeps the largest correspondence set consistent with one calibrated camera motion.",
        ha="center",
        color="#5c667a",
        fontsize=9,
    )
    fig.subplots_adjust(left=0.04, right=0.985, top=0.9, bottom=0.06, hspace=0.19)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=180, facecolor="#f8f9fb")
    plt.close(fig)

    print(f"Tracked correspondences: {tracks.valid_count}")
    print(f"RANSAC inliers: {pose.ransac_inlier_count}")
    print(f"Pose inliers: {pose.inlier_count} ({pose.inlier_ratio:.1%})")
    print(f"Rotation magnitude: {rotation_angle_degrees(pose.rotation_current_previous):.3f} deg")
    print(f"Translation direction: {pose.translation_direction_current}")
    print(f"Saved visualization to {args.output}")


if __name__ == "__main__":
    main()
