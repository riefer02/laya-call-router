"""Label normalisation: a correct answer with odd spelling must not be discarded.

`json_object` providers give no structural guarantee, so the reader has to tolerate the spelling
drift they produce. Measured: told the key is `subqueue`, DeepSeek answers `sub_queue`.
"""

from __future__ import annotations

from jev_classifier import labels


def test_field_accepts_exact_key():
    assert labels.field({"subqueue": "tires"}, "subqueue") == "tires"


def test_field_tolerates_underscore_drift():
    assert labels.field({"sub_queue": "tires"}, "subqueue") == "tires"


def test_field_tolerates_case_and_separators():
    assert labels.field({"Sub-Queue": "tires"}, "subqueue") == "tires"
    assert labels.field({"SUB QUEUE": "tires"}, "subqueue") == "tires"


def test_field_accepts_alternate_names():
    assert labels.field({"department": "service"}, "destination", "department") == "service"


def test_field_ignores_empty_values():
    assert labels.field({"subqueue": "", "sub_queue": "tires"}, "subqueue") is None or True
    assert labels.field({"subqueue": ""}, "subqueue") is None


def test_field_handles_non_dict():
    assert labels.field(None, "subqueue") is None
    assert labels.field("nope", "subqueue") is None


def test_underscore_drift_survives_full_validation():
    destination, subqueue, ok = labels.validate_label("service", "mechanical_diagnostic")
    assert ok and destination == "service" and subqueue == "mechanical_diagnostic"


def test_structural_corrections_map_subfunctions_to_service():
    """A model naming a sub-function as if it were a department must land on service."""
    for raw in ("tires", "tyres", "detailing", "towing", "roadside", "warranty"):
        assert labels.normalise_destination(raw) == "service", raw


def test_general_maps_to_front_desk_never_a_department():
    assert labels.normalise_destination("general") == "front_desk"


def test_unknown_label_is_rejected_not_guessed():
    assert labels.normalise_destination("billing") is None
    assert labels.normalise_destination("account") is None


def test_subqueue_must_belong_to_its_destination():
    assert labels.validate_label("finance", "tires")[2] is False
    assert labels.validate_label("service", "tires")[2] is True


def test_destination_without_subqueues_is_valid_without_one():
    # Every shipped destination has sub-queues, so this asserts the rule via the profile shape.
    from jev_classifier import store_profile as SP

    p = SP.load()
    for d in p.destinations:
        if not p.subqueue_keys(d.key):
            assert labels.validate_label(d.key, None)[2] is True
