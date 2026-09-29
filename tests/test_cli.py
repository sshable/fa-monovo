"""Tests for command-line validation and override parsing."""

import pytest

from monovo.cli import build_parser


def test_run_parser_accepts_pipeline_overrides() -> None:
    args = build_parser().parse_args(
        [
            "run",
            "--sequence",
            "05",
            "--max-frames",
            "100",
            "--minimum-features",
            "300",
            "--ransac-threshold",
            "1.5",
            "--scale-mode",
            "scale_free",
            "--no-visualization",
            "--save-video",
        ]
    )

    assert args.sequence == "05"
    assert args.max_frames == 100
    assert args.minimum_features == 300
    assert args.ransac_threshold == pytest.approx(1.5)
    assert args.scale_mode == "scale_free"
    assert args.visualization is False
    assert args.save_video is True


@pytest.mark.parametrize(
    ("option", "value"),
    [("--max-frames", "0"), ("--minimum-features", "-1"), ("--ransac-threshold", "nan")],
)
def test_run_parser_rejects_nonpositive_numbers(option: str, value: str) -> None:
    with pytest.raises(SystemExit):
        build_parser().parse_args(["run", option, value])
