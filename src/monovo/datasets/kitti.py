"""Loading and validation for the KITTI odometry benchmark."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


class KittiDataError(RuntimeError):
    """Raised when a KITTI sequence is missing or malformed."""


@dataclass(frozen=True)
class KittiCalibration:
    """Projection and intrinsic matrices for one rectified KITTI camera."""

    key: str
    projection: FloatArray
    intrinsic: FloatArray
    camera_center_in_reference: FloatArray


def load_calibration(path: Path, key: str) -> KittiCalibration:
    """Read one 3x4 projection matrix and its 3x3 intrinsic matrix."""
    if not path.is_file():
        raise KittiDataError(f"Calibration file does not exist: {path}")

    entries: dict[str, list[str]] = {}
    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw_line.strip()
        if not line:
            continue
        if ":" not in line:
            raise KittiDataError(f"Invalid calibration line {line_number} in {path}")
        name, raw_values = line.split(":", maxsplit=1)
        if name in entries:
            raise KittiDataError(f"Duplicate calibration key {name!r} in {path}")
        entries[name] = raw_values.split()

    if key not in entries:
        available = ", ".join(sorted(entries)) or "none"
        raise KittiDataError(
            f"Calibration key {key!r} is missing from {path}; available keys: {available}"
        )

    try:
        values = np.asarray([float(value) for value in entries[key]], dtype=np.float64)
    except ValueError as exc:
        raise KittiDataError(f"Calibration key {key!r} contains a non-numeric value") from exc

    if values.size != 12:
        raise KittiDataError(f"Calibration key {key!r} must contain 12 values, found {values.size}")
    if not np.isfinite(values).all():
        raise KittiDataError(f"Calibration key {key!r} contains non-finite values")

    projection = values.reshape(3, 4)
    intrinsic = projection[:, :3].copy()
    if intrinsic[0, 0] <= 0 or intrinsic[1, 1] <= 0:
        raise KittiDataError(f"Calibration key {key!r} has a non-positive focal length")
    if not np.isclose(intrinsic[2, 2], 1.0):
        raise KittiDataError(f"Calibration key {key!r} has an invalid homogeneous scale")

    # KITTI rectified projection matrices have the form P = K [I | t].
    # The selected camera center expressed in camera-0 coordinates is C = -K^-1 t_pixel.
    camera_center = -np.linalg.solve(intrinsic, projection[:, 3])

    return KittiCalibration(
        key=key,
        projection=projection,
        intrinsic=intrinsic,
        camera_center_in_reference=camera_center,
    )


def load_poses(path: Path) -> FloatArray:
    """Load KITTI camera-to-world poses as an array with shape (N, 4, 4)."""
    rows = _load_numeric_rows(path, columns=12, description="pose")
    poses = np.repeat(np.eye(4, dtype=np.float64)[None, :, :], len(rows), axis=0)
    poses[:, :3, :] = rows.reshape(-1, 3, 4)
    return poses


def load_timestamps(path: Path) -> FloatArray:
    """Load KITTI timestamps in seconds."""
    timestamps = _load_numeric_rows(path, columns=1, description="timestamp").reshape(-1)
    if len(timestamps) > 1 and np.any(np.diff(timestamps) <= 0):
        raise KittiDataError(f"Timestamps must be strictly increasing: {path}")
    return timestamps


class KittiSequence:
    """Validated access to one KITTI odometry image sequence.

    Ground-truth matrices are returned as ``T_world_camera0`` because KITTI's
    pose files describe the reference grayscale camera. When ``image_2`` is
    selected, conversion to the color camera frame belongs in evaluation code.
    """

    def __init__(
        self,
        root: str | Path,
        sequence: str = "00",
        image_directory: str = "image_2",
        calibration_key: str = "P2",
        poses_directory: str = "poses",
    ) -> None:
        if len(sequence) != 2 or not sequence.isdigit():
            raise ValueError(f"KITTI sequence must be two digits, received {sequence!r}")

        self.root = Path(root)
        self.sequence = sequence
        self.sequence_directory = self.root / "sequences" / sequence
        self.image_directory = self.sequence_directory / image_directory

        if not self.root.is_dir():
            raise KittiDataError(f"KITTI dataset root does not exist: {self.root}")
        if not self.sequence_directory.is_dir():
            raise KittiDataError(
                f"KITTI sequence directory does not exist: {self.sequence_directory}"
            )
        if not self.image_directory.is_dir():
            raise KittiDataError(f"KITTI image directory does not exist: {self.image_directory}")

        self.image_paths = self._find_image_paths()
        self.calibration = load_calibration(self.sequence_directory / "calib.txt", calibration_key)
        self.timestamps = load_timestamps(self.sequence_directory / "times.txt")

        pose_path = self.root / poses_directory / f"{sequence}.txt"
        self.poses = load_poses(pose_path) if pose_path.is_file() else None
        self._validate_counts()

    def __len__(self) -> int:
        """Return the number of frames in the sequence."""
        return len(self.image_paths)

    @property
    def has_ground_truth(self) -> bool:
        """Whether ground-truth poses are available for this sequence."""
        return self.poses is not None

    @property
    def intrinsic_matrix(self) -> FloatArray:
        """Return a copy of the selected camera's 3x3 intrinsic matrix."""
        return self.calibration.intrinsic.copy()

    def load_image(self, index: int) -> NDArray[np.uint8]:
        """Load one rectified frame as an 8-bit grayscale image."""
        self._validate_index(index)
        path = self.image_paths[index]
        image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise KittiDataError(f"OpenCV could not decode image: {path}")
        return image

    def ground_truth_pose(self, index: int) -> FloatArray:
        """Return one ``T_world_camera0`` ground-truth pose."""
        self._validate_index(index)
        if self.poses is None:
            raise KittiDataError(f"Ground-truth poses are unavailable for sequence {self.sequence}")
        return self.poses[index].copy()

    def selected_camera_ground_truth_poses(self) -> FloatArray:
        """Return ground-truth ``T_world_camera`` poses for the selected camera.

        KITTI pose files describe rectified camera 0. The projection matrix
        supplies the fixed center offset of the selected rectified camera.
        """
        if self.poses is None:
            raise KittiDataError(f"Ground-truth poses are unavailable for sequence {self.sequence}")
        transform_reference_selected = np.eye(4, dtype=np.float64)
        transform_reference_selected[:3, 3] = self.calibration.camera_center_in_reference
        return self.poses @ transform_reference_selected

    def _find_image_paths(self) -> tuple[Path, ...]:
        paths = list(self.image_directory.glob("*.png"))
        if not paths:
            raise KittiDataError(f"No PNG images found in {self.image_directory}")
        if any(not path.stem.isdigit() for path in paths):
            raise KittiDataError(f"Image names must be numeric in {self.image_directory}")

        paths.sort(key=lambda path: int(path.stem))
        frame_numbers = [int(path.stem) for path in paths]
        expected = list(range(len(paths)))
        if frame_numbers != expected:
            raise KittiDataError(
                f"Image sequence must be contiguous from frame 0 in {self.image_directory}"
            )
        return tuple(paths)

    def _validate_counts(self) -> None:
        frame_count = len(self.image_paths)
        if len(self.timestamps) != frame_count:
            raise KittiDataError(
                f"Frame/timestamp count mismatch: {frame_count} images and "
                f"{len(self.timestamps)} timestamps"
            )
        if self.poses is not None and len(self.poses) != frame_count:
            raise KittiDataError(
                f"Frame/pose count mismatch: {frame_count} images and {len(self.poses)} poses"
            )

    def _validate_index(self, index: int) -> None:
        if not 0 <= index < len(self):
            raise IndexError(f"Frame index {index} is outside [0, {len(self) - 1}]")


def _load_numeric_rows(path: Path, columns: int, description: str) -> FloatArray:
    if not path.is_file():
        raise KittiDataError(f"{description.capitalize()} file does not exist: {path}")

    rows: list[list[float]] = []
    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        values = raw_line.split()
        if not values:
            continue
        if len(values) != columns:
            raise KittiDataError(
                f"Invalid {description} row {line_number} in {path}: "
                f"expected {columns} values, found {len(values)}"
            )
        try:
            row = [float(value) for value in values]
        except ValueError as exc:
            raise KittiDataError(
                f"Invalid numeric value in {description} row {line_number} of {path}"
            ) from exc
        rows.append(row)

    if not rows:
        raise KittiDataError(f"{description.capitalize()} file is empty: {path}")

    array = np.asarray(rows, dtype=np.float64)
    if not np.isfinite(array).all():
        raise KittiDataError(f"{description.capitalize()} file contains non-finite values: {path}")
    return array
