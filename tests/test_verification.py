import json

import pytest

from road_damage import verification
from road_damage.detection import Detection
from road_damage.verification import Tolerances, compare_detections, summarize_reports


def det(cls=0, conf=0.8, box=(10, 10, 50, 50)):
    return Detection(cls, f"class{cls}", conf, box)


def test_order_and_confidence_rank_do_not_define_correspondence():
    a, b = det(), det(box=(100, 100, 150, 150))
    report = compare_detections([a, b], [b, a])
    assert report["passed"]
    assert [m["candidate_index"] for m in report["matches"]] == [1, 0]


def test_matching_recovers_from_ambiguous_first_choice():
    report = compare_detections([det(conf=0.5), det(conf=0.6)],
                                [det(conf=0.55), det(conf=0.45)],
                                Tolerances(max_confidence_delta=0.051))
    assert report["passed"]
    assert [m["candidate_index"] for m in report["matches"]] == [1, 0]


@pytest.mark.parametrize("candidate", [[], [det(cls=1)], [det(conf=0.7)],
                                       [det(box=(11, 10, 51, 50))], [det(), det()]])
def test_count_class_confidence_and_coordinate_mismatches_fail(candidate):
    assert not compare_detections([det()], candidate)["passed"]


def test_each_candidate_can_only_match_one_reference():
    assert not compare_detections([det(), det()],
                                  [det(), det(box=(100, 100, 150, 150))])["passed"]


@pytest.mark.parametrize("bad", [det(conf=float("nan")), det(box=(0, 0, 0, 5)),
                                 det(conf=1.1), det(box=(0, 0, float("inf"), 5))])
def test_invalid_numeric_detections_fail(bad):
    assert compare_detections([bad], [bad])["reason"] == "invalid_or_nonfinite_detection"


def test_empty_only_run_is_inconclusive():
    summary = summarize_reports([compare_detections([], [])])
    assert not summary["passed"]
    assert summary["status"] == "inconclusive"
    assert not summarize_reports([])["passed"]


def test_nonempty_and_background_cases_can_pass_together():
    reports = [compare_detections([det()], [det()]), compare_detections([], [])]
    assert summarize_reports(reports)["passed"]
    assert not summarize_reports(reports, min_nonempty=2)["passed"]


@pytest.mark.parametrize("kwargs", [{"min_iou": 0}, {"min_iou": 1.1},
                                    {"max_box_delta_px": -1},
                                    {"max_confidence_delta": float("nan")}])
def test_invalid_tolerances_are_rejected(kwargs):
    with pytest.raises(ValueError):
        Tolerances(**kwargs)


@pytest.mark.parametrize("passed,expected", [(False, 1), (True, 0)])
def test_cli_returns_failure_or_success_and_writes_report(tmp_path, monkeypatch, passed, expected):
    config = tmp_path / "config.yaml"
    config.write_text("{}")
    output = tmp_path / "report.json"
    report = {"summary": {"passed": passed, "status": "passed" if passed else "failed"}}
    monkeypatch.setattr(verification, "verify", lambda *a, **k: report)
    result = verification.main(["--pt", "pt", "--onnx", "onnx", "--image", "image.jpg",
                                "--config", str(config), "--output", str(output)])
    assert result == expected
    assert json.loads(output.read_text()) == report


def test_empty_input_folder_has_error_exit(tmp_path):
    with pytest.raises(SystemExit) as exc:
        verification.main(["--pt", "pt", "--onnx", "onnx", "--folder", str(tmp_path)])
    assert exc.value.code == 2
