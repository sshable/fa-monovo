"""YAML loading and typed construction of pipeline configuration."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml

from monovo.confidence import ConfidenceConfig
from monovo.features import ShiTomasiConfig
from monovo.geometry import EssentialMatrixConfig
from monovo.odometry import MonocularOdometryConfig
from monovo.tracking import LucasKanadeConfig


def load_yaml_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML mapping and reject missing or invalid documents."""
    config_path = Path(path)
    if not config_path.is_file():
        raise ValueError(f"Configuration file does not exist: {config_path}")
    document = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(document, dict):
        raise ValueError(f"Configuration root must be a mapping: {config_path}")
    return document


def build_odometry_config(settings: Mapping[str, Any]) -> MonocularOdometryConfig:
    """Build validated feature, tracking, geometry, and confidence settings."""
    try:
        features = settings["features"]
        tracking = settings["tracking"]
        geometry = settings["geometry"]
        confidence = settings["confidence"]
        return MonocularOdometryConfig(
            features=ShiTomasiConfig(
                max_features=features["max_features"],
                quality_level=features["quality_level"],
                minimum_distance=features["minimum_distance"],
                block_size=features["block_size"],
            ),
            tracking=LucasKanadeConfig(
                window_size=tracking["window_size"],
                pyramid_levels=tracking["pyramid_levels"],
                criteria_max_iterations=tracking["criteria_max_iterations"],
                criteria_epsilon=tracking["criteria_epsilon"],
                forward_backward_threshold=tracking["forward_backward_threshold"],
            ),
            geometry=EssentialMatrixConfig(
                ransac_probability=geometry["ransac_probability"],
                ransac_threshold_pixels=geometry["ransac_threshold_pixels"],
                ransac_max_iterations=geometry["ransac_max_iterations"],
                minimum_correspondences=geometry["minimum_correspondences"],
                minimum_inliers=geometry["minimum_inliers"],
                random_seed=geometry["random_seed"],
            ),
            confidence=ConfidenceConfig(
                tracked_features_reference=confidence["tracked_features_reference"],
                inliers_reference=confidence["inliers_reference"],
                inlier_ratio_reference=confidence["inlier_ratio_reference"],
                spatial_coverage_reference=confidence["spatial_coverage_reference"],
                minimum_tracked_features=confidence["minimum_tracked_features"],
                minimum_inliers=confidence["minimum_inliers"],
                minimum_inlier_ratio=confidence["minimum_inlier_ratio"],
                minimum_median_flow_pixels=confidence["minimum_median_flow_pixels"],
                minimum_spatial_coverage=confidence["minimum_spatial_coverage"],
                good_score_threshold=confidence["good_score_threshold"],
                maximum_rotation_degrees=confidence["maximum_rotation_degrees"],
                maximum_direction_change_degrees=confidence["maximum_direction_change_degrees"],
                temporal_penalty=confidence["temporal_penalty"],
                coverage_grid_rows=confidence["coverage_grid_rows"],
                coverage_grid_columns=confidence["coverage_grid_columns"],
            ),
            minimum_features=features["minimum_features"],
        )
    except (KeyError, TypeError) as exc:
        raise ValueError(f"Missing or invalid configuration value: {exc}") from exc
