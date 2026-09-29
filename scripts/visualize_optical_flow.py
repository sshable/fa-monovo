"""Track KITTI features across two frames and visualize sparse optical flow."""

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
from monovo.tracking import LucasKanadeConfig, track_features


def parse_args() -> argparse.Namespace:
    """Parse visualization arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/kitti.yaml"))
    parser.add_argument("--frame", type=int, default=0)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("docs/assets/lucas-kanade-tracks-000000-000001.png"),
    )
    return parser.parse_args()


def main() -> None:
    """Detect features, track them to the next frame, and render vectors."""
    args = parse_args()
    settings = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    dataset = settings["dataset"]
    feature_settings = settings["features"]
    tracking_settings = settings["tracking"]
    sequence = KittiSequence(
        root=dataset["root"],
        sequence=dataset["sequence"],
        image_directory=dataset["image_directory"],
        calibration_key=dataset["calibration_key"],
        poses_directory=dataset["poses_directory"],
    )
    if args.frame + 1 >= len(sequence):
        raise ValueError(f"Frame {args.frame} has no following frame")

    detector_config = ShiTomasiConfig(
        max_features=feature_settings["max_features"],
        quality_level=feature_settings["quality_level"],
        minimum_distance=feature_settings["minimum_distance"],
        block_size=feature_settings["block_size"],
    )
    tracker_config = LucasKanadeConfig(
        window_size=tracking_settings["window_size"],
        pyramid_levels=tracking_settings["pyramid_levels"],
        criteria_max_iterations=tracking_settings["criteria_max_iterations"],
        criteria_epsilon=tracking_settings["criteria_epsilon"],
        forward_backward_threshold=tracking_settings["forward_backward_threshold"],
    )
    previous = sequence.load_image(args.frame)
    current = sequence.load_image(args.frame + 1)
    features = detect_shi_tomasi(previous, detector_config)
    tracks = track_features(previous, current, features, tracker_config)
    source = tracks.valid_source_points
    target = tracks.valid_target_points
    flow = target - source
    magnitudes = np.linalg.norm(flow, axis=1)

    fig, axes = plt.subplots(2, 1, figsize=(14, 8.8), facecolor="#f8f9fb")
    fig.suptitle(
        f"Pyramidal Lucas–Kanade optical flow — frames {args.frame:06d} → {args.frame + 1:06d}",
        x=0.045,
        y=0.975,
        ha="left",
        fontsize=18,
        fontweight="bold",
        color="#172033",
    )
    fig.text(
        0.045,
        0.938,
        f"{tracks.valid_count:,} / {len(features):,} tracks retained "
        f"({tracks.inlier_ratio:.1%}) • median flow {np.median(magnitudes):.2f} px",
        color="#5c667a",
        fontsize=10,
    )

    axes[0].imshow(current, cmap="gray", vmin=0, vmax=255)
    axes[0].plot(target[:, 0], target[:, 1], ".", color="#ff6b45", markersize=2.4)
    for old, new in zip(source, target, strict=True):
        axes[0].plot(
            [old[0], new[0]],
            [old[1], new[1]],
            color="#22a7b5",
            linewidth=0.45,
            alpha=0.65,
        )
    axes[0].set_title("All valid tracks at their true pixel displacement", loc="left", fontsize=11)
    axes[0].set_axis_off()

    center_x, center_y = current.shape[1] // 2, current.shape[0] // 2
    half_width, half_height = 240, 125
    x_min, x_max = center_x - half_width, center_x + half_width
    y_min, y_max = center_y - half_height, center_y + half_height
    in_crop = (
        (source[:, 0] >= x_min)
        & (source[:, 0] < x_max)
        & (source[:, 1] >= y_min)
        & (source[:, 1] < y_max)
    )
    axes[1].imshow(current, cmap="gray", vmin=0, vmax=255)
    axes[1].quiver(
        source[in_crop, 0],
        source[in_crop, 1],
        flow[in_crop, 0],
        flow[in_crop, 1],
        angles="xy",
        scale_units="xy",
        scale=1,
        color="#ff6b45",
        width=0.0035,
    )
    axes[1].set_xlim(x_min, x_max)
    axes[1].set_ylim(y_max, y_min)
    axes[1].set_title("Central crop — arrows show true displacement", loc="left", fontsize=11)
    axes[1].set_axis_off()

    fig.text(
        0.5,
        0.018,
        "Tracks failing LK, forward–backward consistency, finite-value, "
        "or boundary checks are removed.",
        ha="center",
        color="#5c667a",
        fontsize=9,
    )
    fig.subplots_adjust(left=0.04, right=0.985, top=0.9, bottom=0.06, hspace=0.19)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=180, facecolor="#f8f9fb")
    plt.close(fig)

    print(f"Detected features: {len(features)}")
    print(f"Valid tracks: {tracks.valid_count} ({tracks.inlier_ratio:.1%})")
    print(f"Median optical-flow magnitude: {np.median(magnitudes):.3f} pixels")
    print(f"Saved visualization to {args.output}")


if __name__ == "__main__":
    main()
