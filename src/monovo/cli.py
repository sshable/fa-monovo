"""Command-line interface for running failure-aware monocular VO."""

from __future__ import annotations

import argparse
import copy
import logging
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

from monovo import __version__
from monovo.config import build_odometry_config, load_yaml_config
from monovo.datasets import KittiDataError, KittiSequence
from monovo.evaluation import EvaluationResult, evaluate_trajectory
from monovo.odometry import MonocularOdometry, OdometryRunResult
from monovo.outputs import save_frame_diagnostics, save_metrics, save_trajectory
from monovo.visualization import DemoVideoWriter, save_trajectory_plot

LOGGER = logging.getLogger("monovo")
SCALE_MODES = ("scale_free", "sim3_aligned")


def build_parser() -> argparse.ArgumentParser:
    """Create the top-level command-line parser."""
    parser = argparse.ArgumentParser(
        prog="monovo",
        description="Failure-aware monocular visual odometry",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="Run VO on a KITTI odometry sequence")
    run_parser.add_argument("--config", type=Path, default=Path("configs/kitti.yaml"))
    run_parser.add_argument("--dataset-root", type=Path, help="Override dataset.root")
    run_parser.add_argument("--sequence", help="Two-digit KITTI sequence, for example 00")
    run_parser.add_argument("--max-frames", type=_positive_int, help="Maximum frames to process")
    run_parser.add_argument(
        "--feature-detector",
        choices=("shi_tomasi",),
        help="Feature detector override",
    )
    run_parser.add_argument(
        "--minimum-features",
        type=_positive_int,
        help="Re-detect features below this count",
    )
    run_parser.add_argument(
        "--ransac-threshold",
        type=_positive_float,
        metavar="PIXELS",
        help="Essential-matrix RANSAC threshold",
    )
    run_parser.add_argument("--scale-mode", choices=SCALE_MODES, help="Evaluation mode")
    run_parser.add_argument("--output-directory", type=Path, help="Artifact directory")
    run_parser.add_argument(
        "--visualization",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Save trajectory.png (default comes from config)",
    )
    run_parser.add_argument(
        "--save-video",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Save annotated demo.mp4 (default comes from config)",
    )
    run_parser.add_argument(
        "--log-level",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        default="INFO",
    )
    run_parser.set_defaults(handler=_run_command)
    return parser


def main() -> None:
    """Parse arguments, execute a command, and report actionable errors."""
    parser = build_parser()
    args = parser.parse_args()
    logging.basicConfig(level=args.log_level, format="%(levelname)s %(message)s")
    try:
        args.handler(args)
    except (KittiDataError, OSError, ValueError) as exc:
        parser.error(str(exc))


def _run_command(args: argparse.Namespace) -> None:
    settings = copy.deepcopy(load_yaml_config(args.config))
    dataset_settings = _mapping(settings, "dataset")
    run_settings = _mapping(settings, "run")
    evaluation_settings = _mapping(settings, "evaluation")
    visualization_settings = _mapping(settings, "visualization")

    configured_sequence = str(dataset_settings.get("sequence", "00"))
    sequence_name = args.sequence or configured_sequence
    _validate_sequence_name(sequence_name)
    dataset_root = args.dataset_root or Path(str(dataset_settings["root"]))
    max_frames = args.max_frames or int(run_settings["max_frames"])
    scale_mode = args.scale_mode or str(evaluation_settings["scale_mode"])
    if scale_mode not in SCALE_MODES:
        raise ValueError(f"scale_mode must be one of {', '.join(SCALE_MODES)}")
    if scale_mode == "sim3_aligned" and max_frames < 3:
        raise ValueError("sim3_aligned evaluation requires at least 3 frames")

    configured_detector = str(_mapping(settings, "features").get("detector", "shi_tomasi"))
    detector = args.feature_detector or configured_detector
    if detector != "shi_tomasi":
        raise ValueError("Only the shi_tomasi feature detector is implemented")

    pipeline_config = build_odometry_config(settings)
    if args.minimum_features is not None:
        pipeline_config = replace(pipeline_config, minimum_features=args.minimum_features)
    if args.ransac_threshold is not None:
        pipeline_config = replace(
            pipeline_config,
            geometry=replace(
                pipeline_config.geometry,
                ransac_threshold_pixels=args.ransac_threshold,
            ),
        )

    sequence = KittiSequence(
        root=dataset_root,
        sequence=sequence_name,
        image_directory=str(dataset_settings["image_directory"]),
        calibration_key=str(dataset_settings["calibration_key"]),
        poses_directory=str(dataset_settings["poses_directory"]),
    )
    output_directory = _output_directory(
        args.output_directory,
        Path(str(run_settings["output_directory"])),
        sequence_name,
        configured_sequence,
    )
    visualization_enabled = (
        bool(visualization_settings.get("enabled", True))
        if args.visualization is None
        else args.visualization
    )
    video_enabled = (
        bool(visualization_settings.get("save_video", False))
        if args.save_video is None
        else args.save_video
    )
    output_directory.mkdir(parents=True, exist_ok=True)
    video_path = output_directory / "demo.mp4"
    video_writer = (
        DemoVideoWriter(video_path, fps=_sequence_fps(sequence)) if video_enabled else None
    )

    LOGGER.info(
        "Running sequence %s: %d/%d frames, camera %s, mode %s",
        sequence_name,
        min(max_frames, len(sequence)),
        len(sequence),
        sequence.calibration.key,
        scale_mode,
    )
    try:
        result = MonocularOdometry(pipeline_config).run(
            sequence,
            max_frames=max_frames,
            frame_observer=video_writer,
        )
    finally:
        if video_writer is not None:
            video_writer.close()
    ground_truth, evaluation = _evaluate_if_requested(scale_mode, sequence, result)

    trajectory_path = save_trajectory(output_directory / "trajectory_estimated.txt", result.poses)
    diagnostics_path = save_frame_diagnostics(
        output_directory / "frame_diagnostics.csv", result.diagnostics
    )
    metrics = _build_metrics(
        result=result,
        evaluation=evaluation,
        sequence=sequence,
        scale_mode=scale_mode,
        max_frames=max_frames,
    )
    metrics_path = save_metrics(output_directory / "metrics.json", metrics)

    plot_path: Path | None = None
    if visualization_enabled:
        plot_path = save_trajectory_plot(
            output_directory / "trajectory.png",
            result.poses,
            result.diagnostics,
            sequence_name,
            evaluation=evaluation,
            ground_truth_poses=ground_truth,
        )

    LOGGER.info("Processed %d frames", len(result.poses))
    LOGGER.info("Trajectory: %s", trajectory_path)
    LOGGER.info("Diagnostics: %s", diagnostics_path)
    LOGGER.info("Metrics: %s", metrics_path)
    if plot_path is not None:
        LOGGER.info("Visualization: %s", plot_path)
    if video_enabled:
        LOGGER.info("Video: %s", video_path)
    if evaluation is not None:
        LOGGER.info(
            "ATE RMSE %.3f m after one post-hoc Sim(3); scale %.3f m/VO-unit",
            evaluation.ate_rmse,
            evaluation.alignment.scale,
        )


def _evaluate_if_requested(
    scale_mode: str,
    sequence: KittiSequence,
    result: OdometryRunResult,
) -> tuple[np.ndarray | None, EvaluationResult | None]:
    if scale_mode == "scale_free":
        return None, None
    if not sequence.has_ground_truth:
        raise ValueError(
            f"scale mode {scale_mode!r} requires ground-truth poses for sequence "
            f"{sequence.sequence}"
        )
    ground_truth = sequence.selected_camera_ground_truth_poses()[: len(result.poses)]
    return ground_truth, evaluate_trajectory(result.poses, ground_truth, result.diagnostics)


def _build_metrics(
    result: OdometryRunResult,
    evaluation: EvaluationResult | None,
    sequence: KittiSequence,
    scale_mode: str,
    max_frames: int,
) -> dict[str, Any]:
    diagnostics = result.diagnostics[1:]
    average_time = result.average_processing_time_ms
    metrics: dict[str, Any] = {
        "sequence": sequence.sequence,
        "processed_frame_count": len(result.poses),
        "requested_max_frames": max_frames,
        "scale_mode": scale_mode,
        "trajectory_file": "trajectory_estimated.txt",
        "trajectory_pose_convention": "T_world_camera",
        "trajectory_units": "arbitrary_vo_units",
        "ground_truth_used_during_estimation": False,
        "failed_frame_count": result.failure_count,
        "degraded_frame_count": sum(item.status == "degraded" for item in diagnostics),
        "mean_ransac_inlier_ratio": (
            float(np.mean([item.ransac_inlier_ratio for item in diagnostics]))
            if diagnostics
            else 0.0
        ),
        "average_processing_time_ms": average_time,
        "average_fps": 1000.0 / average_time if average_time > 0.0 else None,
    }
    if evaluation is None:
        metrics.update(
            {
                "ate_rmse_m": None,
                "ate_limitation": (
                    "ATE in metres is unavailable without post-hoc scale alignment."
                ),
                "translation_magnitude_error": None,
                "translation_magnitude_limitation": (
                    "Metric translation magnitude is unobservable from monocular "
                    "correspondences alone."
                ),
            }
        )
        return metrics

    metrics.update(evaluation.to_dict())
    metrics.update(
        {
            "sim3_rotation": evaluation.alignment.rotation.tolist(),
            "sim3_translation_m": evaluation.alignment.translation.tolist(),
            "translation_magnitude_error": None,
            "translation_magnitude_limitation": (
                "Per-frame metric translation remains unobservable; Sim(3) supplies "
                "one trajectory-level scale only."
            ),
        }
    )
    return metrics


def _mapping(settings: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = settings.get(key)
    if not isinstance(value, Mapping):
        raise ValueError(f"Configuration section {key!r} must be a mapping")
    return value


def _output_directory(
    override: Path | None,
    configured: Path,
    sequence: str,
    configured_sequence: str,
) -> Path:
    if override is not None:
        return override
    if sequence != configured_sequence:
        return configured.parent / f"sequence_{sequence}"
    return configured


def _validate_sequence_name(value: str) -> None:
    if len(value) != 2 or not value.isdigit():
        raise ValueError(f"KITTI sequence must be two digits, received {value!r}")


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def _positive_float(value: str) -> float:
    parsed = float(value)
    if not np.isfinite(parsed) or parsed <= 0.0:
        raise argparse.ArgumentTypeError("must be a positive finite number")
    return parsed


def _sequence_fps(sequence: KittiSequence) -> float:
    if len(sequence.timestamps) < 2:
        return 10.0
    median_interval = float(np.median(np.diff(sequence.timestamps)))
    if not np.isfinite(median_interval) or median_interval <= 0.0:
        raise ValueError("KITTI timestamps cannot determine a valid video frame rate")
    return 1.0 / median_interval


if __name__ == "__main__":
    main()
