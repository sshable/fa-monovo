"""Tests for KITTI odometry parsing and validation."""

from pathlib import Path

import cv2
import numpy as np
import pytest

from monovo.datasets.kitti import KittiDataError, KittiSequence, load_calibration, load_poses


@pytest.fixture
def kitti_root(tmp_path: Path) -> Path:
    """Create a tiny, structurally valid KITTI sequence."""
    root = tmp_path / "kitti"
    sequence = root / "sequences" / "00"
    image_directory = sequence / "image_2"
    pose_directory = root / "poses"
    image_directory.mkdir(parents=True)
    pose_directory.mkdir()

    projection = (
        "P2: 7.000000e+02 0.0 6.000000e+02 4.0e+01 "
        "0.0 7.010000e+02 1.800000e+02 0.0 "
        "0.0 0.0 1.0 0.0\n"
    )
    (sequence / "calib.txt").write_text(projection, encoding="utf-8")
    (sequence / "times.txt").write_text("0.0\n0.1\n", encoding="utf-8")

    pose_0 = "1 0 0 0 0 1 0 0 0 0 1 0"
    pose_1 = "1 0 0 1 0 1 0 2 0 0 1 3"
    (pose_directory / "00.txt").write_text(f"{pose_0}\n{pose_1}\n", encoding="utf-8")

    first = np.zeros((6, 8), dtype=np.uint8)
    second = np.full((6, 8), 127, dtype=np.uint8)
    assert cv2.imwrite(str(image_directory / "000000.png"), first)
    assert cv2.imwrite(str(image_directory / "000001.png"), second)
    return root


def test_sequence_loads_images_calibration_timestamps_and_poses(kitti_root: Path) -> None:
    sequence = KittiSequence(kitti_root)

    assert len(sequence) == 2
    assert sequence.has_ground_truth
    assert sequence.load_image(1).shape == (6, 8)
    assert sequence.load_image(1).dtype == np.uint8
    np.testing.assert_allclose(sequence.timestamps, [0.0, 0.1])
    np.testing.assert_allclose(
        sequence.intrinsic_matrix,
        [[700.0, 0.0, 600.0], [0.0, 701.0, 180.0], [0.0, 0.0, 1.0]],
    )
    np.testing.assert_allclose(sequence.ground_truth_pose(1)[:3, 3], [1.0, 2.0, 3.0])
    np.testing.assert_allclose(sequence.ground_truth_pose(1)[3], [0.0, 0.0, 0.0, 1.0])
    np.testing.assert_allclose(
        sequence.calibration.camera_center_in_reference,
        [-40.0 / 700.0, 0.0, 0.0],
    )
    np.testing.assert_allclose(
        sequence.selected_camera_ground_truth_poses()[1, :3, 3],
        [1.0 - 40.0 / 700.0, 2.0, 3.0],
    )


def test_calibration_requires_requested_camera_key(kitti_root: Path) -> None:
    with pytest.raises(KittiDataError, match="Calibration key 'P0' is missing"):
        load_calibration(kitti_root / "sequences" / "00" / "calib.txt", "P0")


def test_pose_parser_rejects_malformed_rows(tmp_path: Path) -> None:
    path = tmp_path / "poses.txt"
    path.write_text("1 0 0\n", encoding="utf-8")

    with pytest.raises(KittiDataError, match="expected 12 values, found 3"):
        load_poses(path)


def test_sequence_rejects_frame_timestamp_count_mismatch(kitti_root: Path) -> None:
    times_path = kitti_root / "sequences" / "00" / "times.txt"
    times_path.write_text("0.0\n", encoding="utf-8")

    with pytest.raises(KittiDataError, match="Frame/timestamp count mismatch"):
        KittiSequence(kitti_root)


def test_frame_index_must_be_in_range(kitti_root: Path) -> None:
    sequence = KittiSequence(kitti_root)

    with pytest.raises(IndexError, match="outside"):
        sequence.load_image(-1)
