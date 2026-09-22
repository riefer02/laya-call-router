"""Reporting statistics: the interval is what makes an 81-case result readable."""

from __future__ import annotations

from jev_classifier import evalharness as H


def test_wilson_interval_brackets_the_point_estimate():
    ci = H.wilson_ci(53, 81)
    assert ci["lo"] < 53 / 81 < ci["hi"]


def test_wilson_interval_narrows_with_more_data():
    small = H.wilson_ci(53, 81)["half_width"]
    large = H.wilson_ci(530, 810)["half_width"]
    assert large < small


def test_wilson_interval_stays_in_range_at_the_extremes():
    for successes, n in ((0, 81), (81, 81), (0, 1), (1, 1)):
        ci = H.wilson_ci(successes, n)
        assert 0.0 <= ci["lo"] <= ci["hi"] <= 1.0


def test_wilson_interval_handles_empty():
    assert H.wilson_ci(0, 0) == {"lo": 0.0, "hi": 0.0, "half_width": 0.0}


def test_one_case_on_81_is_about_1_2_points():
    """The reason we report intervals: on this test set a single case is worth real percentage."""
    a = H.wilson_ci(50, 81)["lo"]
    b = H.wilson_ci(51, 81)["lo"]
    assert abs((b - a) - 0.0123) < 0.004


def test_confusion_groups_by_expected_and_counts_got():
    conf = H._confusion([("service", "service"), ("service", "body_shop"), ("parts", "parts")])
    assert conf["service"] == {"service": 1, "body_shop": 1}
    assert conf["parts"] == {"parts": 1}


def test_agreement_defaults_to_destination():
    runs = [
        [{"id": "a", "destination": "service"}, {"id": "b", "destination": "parts"}],
        [{"id": "a", "destination": "service"}, {"id": "b", "destination": "body_shop"}],
    ]
    assert H.agreement(runs) == 0.5
