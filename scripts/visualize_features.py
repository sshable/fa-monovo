"""Detect Shi–Tomasi features in one KITTI frame and save an annotated figure."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import yaml

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402

from monovo.datasets import KittiSequence
from monovo.features import ShiTomasiConfig, detect_shi_tomasi


def parse_args() -> argparse.Namespace:
    """Parse visualization arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/kitti.yaml"))
    parser.add_argument("--frame", type=int, default=0)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("docs/assets/shi-tomasi-features-frame-000000.png"),
    )
    return parser.parse_args()


def main() -> None:
    """Run detection on the selected frame and render the result."""
    args = parse_args()
    settings = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    dataset = settings["dataset"]
    feature_settings = settings["features"]

    sequence = KittiSequence(
        root=dataset["root"],
        sequence=dataset["sequence"],
        image_directory=dataset["image_directory"],
        calibration_key=dataset["calibration_key"],
        poses_directory=dataset["poses_directory"],
    )
    detector_config = ShiTomasiConfig(
        max_features=feature_settings["max_features"],
        quality_level=feature_settings["quality_level"],
        minimum_distance=feature_settings["minimum_distance"],
        block_size=feature_settings["block_size"],
    )
    image = sequence.load_image(args.frame)
    points = detect_shi_tomasi(image, detector_config)

    fig, axis = plt.subplots(figsize=(14, 5.4), facecolor="#f8f9fb")
    axis.imshow(image, cmap="gray", vmin=0, vmax=255)
    axis.scatter(
        points[:, 0],
        points[:, 1],
        s=10,
        facecolors="none",
        edgecolors="#ff6b45",
        linewidths=0.65,
    )
    axis.set_title(
        f"Shi–Tomasi features — frame {args.frame:06d}",
        loc="left",
        fontsize=18,
        fontweight="bold",
        color="#172033",
        pad=18,
    )
    axis.text(
        0,
        1.015,
        f"{len(points):,} corners • circles show detected pixel coordinates (u, v)",
        transform=axis.transAxes,
        color="#5c667a",
        fontsize=10,
    )
    axis.set_axis_off()
    fig.text(
        0.5,
        0.025,
        "Corners are patches with strong brightness change in two directions.",
        ha="center",
        color="#5c667a",
        fontsize=9,
    )
    fig.subplots_adjust(left=0.035, right=0.985, top=0.86, bottom=0.09)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=180, facecolor="#f8f9fb")
    plt.close(fig)
    print(f"Detected {len(points)} features in frame {args.frame:06d}")
    print(f"Saved visualization to {args.output}")


if __name__ == "__main__":
    main()
