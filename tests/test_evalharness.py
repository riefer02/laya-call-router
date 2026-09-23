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


# --------------------------------------------------------------------------- calibration
def test_calibration_bands_report_accuracy_per_confidence():
    cal = H._calibration([(0.1, False), (0.15, False), (0.9, True), (0.95, True)])
    assert cal["0.0-0.2"]["accuracy"] == 0.0
    assert cal["0.8-1.0"]["accuracy"] == 1.0


def test_calibration_exposes_a_flat_confident_curve():
    """The fine-tune's actual failure: confident on everything, so one band holds every case and
    no threshold can separate the errors. This is why the gate flagged 0 of 81."""
    cal = H._calibration([(0.95, True)] * 9 + [(0.95, False)])
    assert cal == {"0.8-1.0": {"n": 10, "accuracy": 0.9}}


def test_calibration_ignores_cases_with_no_confidence():
    assert H._calibration([(None, True), (None, False)]) == {}


def test_calibration_counts_the_top_band_inclusively():
    """A confidence of exactly 1.0 must not fall outside every band."""
    cal = H._calibration([(1.0, True)])
    assert cal["0.8-1.0"] == {"n": 1, "accuracy": 1.0}


# --------------------------------------------------------------------------- queue accuracy
def _cases(*pairs):
    return [H.RoutingCase(id=f"c{i}", text="x", destination=d, subqueue=s) for i, (d, s) in enumerate(pairs)]


def _results(*pairs):
    return [
        {"id": f"c{i}", "destination": d, "subqueue": s, "latency_ms": 1.0, "valid": True}
        for i, (d, s) in enumerate(pairs)
    ]


def test_queue_accuracy_credits_a_label_miss_that_routes_the_same():
    """`front_desk` and `non_customer` both route to Front Desk.

    A label miss there is not a routing miss. Queue accuracy is reported *alongside* destination
    accuracy, never instead of it, so a taxonomy disagreement stays visible.
    """
    cases = _cases(("front_desk", "general_question"))
    results = _results(("non_customer", "wrong_number"))
    score = H.score_routing(cases, results)
    assert score["destination_accuracy"] == 0.0  # the label is wrong, and that is still shown
    assert score["queue_accuracy"] == 1.0  # but it lands on the same queue


def test_queue_accuracy_still_punishes_a_real_misroute():
    cases = _cases(("service", "tires"))
    results = _results(("sales", "new_vehicle"))
    score = H.score_routing(cases, results)
    assert score["queue_accuracy"] == 0.0


def test_queue_confusion_is_reported():
    cases = _cases(("service", "tires"), ("service", "tires"))
    results = _results(("service", "tires"), ("service", "roadside_assistance"))
    score = H.score_routing(cases, results)
    assert score["queue_accuracy"] == 0.5
    assert score["queue_confusion"]["Tire Bay"]["Roadside / Towing"] == 1


# ------------------------------------------------------------------- precision at a deployment rate
def test_precision_at_the_eval_rate_round_trips_to_the_printed_precision():
    """The 40% entry IS the severity set's own rate, so Bayes must return what the set measured.

    When it does not, either the set changed or the conversion is wrong - and the number everyone
    reads (precision on an enriched set) has quietly stopped describing anything real.
    """
    from jev_classifier.evalharness import precision_at_base_rate

    # Exact counts, not the rounded 0.704/1.000 printed in the report - the round trip is the point.
    tp, fp, tn, fn = 18, 8, 19, 0
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)
    base_rate = (tp + fn) / (tp + fp + tn + fn)
    measured = tp / (tp + fp)
    assert abs(precision_at_base_rate(sensitivity, specificity, base_rate) - measured) < 1e-12


def test_precision_collapses_as_the_base_rate_falls():
    """The whole point: precision is not a property of the classifier.

    The trained safety question measures 0.692 precision on a 40%-positive set and 0.064 at 2%,
    because it traded specificity (0.889 -> 0.704) for sensitivity. A reviewer reading only the set's
    precision would conclude the dispatch was roughly three-quarters right.
    """
    from jev_classifier.evalharness import DEPLOYMENT_BASE_RATES, precision_by_base_rate

    by_rate = precision_by_base_rate(sensitivity=1.0, specificity=0.704)
    values = [by_rate[f"{r:.0%}"] for r in DEPLOYMENT_BASE_RATES]
    assert values == sorted(values), "precision must fall as the class gets rarer"
    assert by_rate["2%"] < 0.10
    assert by_rate["40%"] > 0.60

    # And the untrained model, which is LESS sensitive but more specific, wins at low base rates.
    untrained = precision_by_base_rate(sensitivity=1.0, specificity=0.889)
    assert untrained["2%"] > by_rate["2%"]
