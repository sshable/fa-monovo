"""Generate reusable documentation figures for a KITTI odometry sequence."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import matplotlib
import numpy as np

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.collections import LineCollection  # noqa: E402

from monovo.datasets import KittiSequence

INK = "#172033"
MUTED = "#5c667a"
ACCENT = "#e45b3c"
SECONDARY = "#167d9a"
GRID = "#dce1e8"
BACKGROUND = "#f8f9fb"


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=Path("data/kitti"))
    parser.add_argument("--sequence", default="00")
    parser.add_argument("--image-directory", default="image_2")
    parser.add_argument("--calibration-key", default="P2")
    parser.add_argument("--output-directory", type=Path, default=Path("docs/assets"))
    return parser.parse_args()


def load_rgb(sequence: KittiSequence, index: int) -> np.ndarray:
    """Load one sequence frame in RGB order for Matplotlib."""
    image = cv2.imread(str(sequence.image_paths[index]), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"Could not decode {sequence.image_paths[index]}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def add_footer(fig: plt.Figure, text: str) -> None:
    """Add consistent source metadata to a figure."""
    fig.text(0.5, 0.018, text, ha="center", va="bottom", color=MUTED, fontsize=9)


def create_frame_grid(sequence: KittiSequence, output_path: Path) -> None:
    """Show representative frames sampled uniformly over the drive."""
    indices = np.linspace(0, len(sequence) - 1, 12, dtype=int)
    fig, axes = plt.subplots(3, 4, figsize=(16, 8.5), facecolor=BACKGROUND)
    fig.suptitle(
        "KITTI Odometry Sequence 00 — representative frames",
        x=0.05,
        y=0.975,
        ha="left",
        color=INK,
        fontsize=19,
        fontweight="bold",
    )
    fig.text(
        0.05,
        0.935,
        "Uniform samples across 470.6 seconds of driving",
        ha="left",
        color=MUTED,
        fontsize=11,
    )

    for axis, index in zip(axes.flat, indices, strict=True):
        axis.imshow(load_rgb(sequence, int(index)))
        axis.set_title(
            f"frame {index:06d}  ·  {sequence.timestamps[index]:.1f} s",
            loc="left",
            color=INK,
            fontsize=10,
            pad=6,
        )
        axis.set_axis_off()

    fig.subplots_adjust(left=0.045, right=0.98, top=0.89, bottom=0.065, wspace=0.025, hspace=0.18)
    add_footer(
        fig,
        "Left rectified color camera (image_2) • 1241 × 376 px • approximately 10 Hz",
    )
    fig.savefig(output_path, dpi=180, facecolor=BACKGROUND)
    plt.close(fig)


def create_frame_anatomy(sequence: KittiSequence, output_path: Path) -> None:
    """Explain the relationship between a color frame, grayscale input, and K."""
    index = 0
    rgb = load_rgb(sequence, index)
    grayscale = sequence.load_image(index)
    intrinsic = sequence.intrinsic_matrix
    cx, cy = intrinsic[0, 2], intrinsic[1, 2]

    fig = plt.figure(figsize=(16, 8.8), facecolor=BACKGROUND)
    grid = fig.add_gridspec(
        2,
        2,
        height_ratios=(3.2, 1.25),
        left=0.05,
        right=0.97,
        top=0.88,
        bottom=0.08,
        wspace=0.05,
        hspace=0.16,
    )
    color_axis = fig.add_subplot(grid[0, 0])
    gray_axis = fig.add_subplot(grid[0, 1])
    matrix_axis = fig.add_subplot(grid[1, :])

    fig.suptitle(
        "What one KITTI frame contains",
        x=0.05,
        y=0.965,
        ha="left",
        color=INK,
        fontsize=20,
        fontweight="bold",
    )
    fig.text(
        0.05,
        0.92,
        "Frame 000000 from sequence 00, captured at t = 0.0 s",
        color=MUTED,
        fontsize=11,
    )

    color_axis.imshow(rgb)
    color_axis.set_title(
        "Stored data: rectified left color image",
        loc="left",
        color=INK,
        fontsize=12,
    )
    color_axis.set_axis_off()

    gray_axis.imshow(grayscale, cmap="gray", vmin=0, vmax=255)
    gray_axis.axvline(cx, color=ACCENT, linewidth=1.4, linestyle="--")
    gray_axis.axhline(cy, color=ACCENT, linewidth=1.4, linestyle="--")
    gray_axis.scatter([cx], [cy], s=55, color=ACCENT, edgecolor="white", linewidth=1.2)
    gray_axis.annotate(
        f"principal point\n(cx, cy) = ({cx:.1f}, {cy:.1f}) px",
        xy=(cx, cy),
        xytext=(cx + 120, cy - 105),
        color=INK,
        fontsize=10,
        arrowprops={"arrowstyle": "->", "color": ACCENT, "linewidth": 1.5},
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "edgecolor": GRID},
    )
    gray_axis.annotate(
        "u / columns",
        xy=(grayscale.shape[1] - 20, grayscale.shape[0] - 18),
        xytext=(grayscale.shape[1] - 230, grayscale.shape[0] - 18),
        color="white",
        fontsize=9,
        arrowprops={"arrowstyle": "->", "color": "white"},
    )
    gray_axis.annotate(
        "v / rows",
        xy=(20, grayscale.shape[0] - 18),
        xytext=(20, grayscale.shape[0] - 110),
        color="white",
        fontsize=9,
        arrowprops={"arrowstyle": "->", "color": "white"},
    )
    gray_axis.set_title(
        "VO input: the same frame converted to grayscale",
        loc="left",
        color=INK,
        fontsize=12,
    )
    gray_axis.set_axis_off()

    matrix_axis.set_axis_off()
    matrix_text = (
        "Camera intrinsics from P2\n"
        f"K =  [ {intrinsic[0, 0]:7.3f}       0.000    {intrinsic[0, 2]:7.3f} ]\n"
        f"     [     0.000   {intrinsic[1, 1]:7.3f}    {intrinsic[1, 2]:7.3f} ]\n"
        "     [     0.000       0.000        1.000 ]"
    )
    matrix_axis.text(
        0.02,
        0.53,
        matrix_text,
        transform=matrix_axis.transAxes,
        va="center",
        color=INK,
        fontsize=12,
        family="monospace",
    )
    explanation = (
        "A frame is one 2D image captured at one instant.\n"
        "Each pixel stores color; the VO pipeline uses brightness.\n"
        "fx, fy map camera rays to pixel distances.\n"
        "cx, cy locate the optical axis on the image plane."
    )
    matrix_axis.text(
        0.58,
        0.53,
        explanation,
        transform=matrix_axis.transAxes,
        va="center",
        color=INK,
        fontsize=11,
        linespacing=1.5,
    )

    add_footer(fig, "The image is rectified: lens distortion has already been removed by KITTI.")
    fig.savefig(output_path, dpi=180, facecolor=BACKGROUND)
    plt.close(fig)


def create_trajectory(sequence: KittiSequence, output_path: Path) -> None:
    """Plot the provided ground-truth camera path from above."""
    if sequence.poses is None:
        raise RuntimeError("Ground-truth poses are required for the trajectory figure")

    positions = sequence.poses[:, :3, 3]
    x_values = positions[:, 0]
    z_values = positions[:, 2]
    points = np.column_stack((x_values, z_values))
    segments = np.stack((points[:-1], points[1:]), axis=1)

    fig, axis = plt.subplots(figsize=(11, 8), facecolor=BACKGROUND)
    axis.set_facecolor("white")
    colors = plt.Normalize(sequence.timestamps[0], sequence.timestamps[-1])
    trajectory = LineCollection(segments, cmap="viridis", norm=colors, linewidth=2.2)
    trajectory.set_array(sequence.timestamps[:-1])
    axis.add_collection(trajectory)
    axis.scatter(x_values[0], z_values[0], s=90, color=SECONDARY, label="start", zorder=3)
    axis.scatter(x_values[-1], z_values[-1], s=90, color=ACCENT, label="end", zorder=3)
    axis.annotate("start", (x_values[0], z_values[0]), xytext=(8, 8), textcoords="offset points")
    axis.annotate("end", (x_values[-1], z_values[-1]), xytext=(8, 8), textcoords="offset points")
    axis.autoscale()
    axis.set_aspect("equal", adjustable="datalim")
    axis.set_xlabel("KITTI world x (m)", color=INK)
    axis.set_ylabel("KITTI world z (m)", color=INK)
    axis.set_title(
        "Sequence 00 ground-truth camera path",
        loc="left",
        color=INK,
        fontsize=18,
        fontweight="bold",
        pad=18,
    )
    axis.text(
        0,
        1.015,
        "Top-down view • color shows elapsed time",
        transform=axis.transAxes,
        color=MUTED,
        fontsize=10,
    )
    axis.grid(color=GRID, linewidth=0.8)
    axis.spines[["top", "right"]].set_visible(False)
    colorbar = fig.colorbar(trajectory, ax=axis, pad=0.02, shrink=0.82)
    colorbar.set_label("elapsed time (s)", color=INK)
    add_footer(
        fig,
        "Provided KITTI poses (T_world_camera0) • used only for evaluation, not motion estimation",
    )
    fig.subplots_adjust(left=0.11, right=0.9, top=0.88, bottom=0.12)
    fig.savefig(output_path, dpi=180, facecolor=BACKGROUND)
    plt.close(fig)


def main() -> None:
    """Generate all dataset documentation figures."""
    args = parse_args()
    sequence = KittiSequence(
        root=args.dataset_root,
        sequence=args.sequence,
        image_directory=args.image_directory,
        calibration_key=args.calibration_key,
    )
    args.output_directory.mkdir(parents=True, exist_ok=True)

    create_frame_grid(sequence, args.output_directory / "kitti-sequence-00-frame-grid.png")
    create_frame_anatomy(sequence, args.output_directory / "kitti-frame-anatomy.png")
    create_trajectory(sequence, args.output_directory / "kitti-sequence-00-trajectory.png")


if __name__ == "__main__":
    main()
