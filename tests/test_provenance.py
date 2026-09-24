"""Tests for immutable experiment manifests."""

from __future__ import annotations

import json

from jev_classifier.provenance import (
    build_run_manifest,
    sha256_file,
    verify_manifest,
    write_manifest,
)


def test_manifest_hashes_inputs_and_verifies(tmp_path):
    source = tmp_path / "data.jsonl"
    source.write_text('{"id":"one"}\n')
    manifest_path = tmp_path / "manifest.json"
    manifest = build_run_manifest(
        experiment_id="phase-a-fixture",
        inputs={"data": source},
        config={"epochs": 8},
        model_id="fixture/model",
        model_revision="abc123",
        seed=7,
        root=tmp_path,
    )
    write_manifest(manifest_path, manifest)
    checked = verify_manifest(manifest_path)
    assert checked["inputs"]["data"]["sha256"] == sha256_file(source)
    assert checked["model"]["revision"] == "abc123"


def test_manifest_detects_changed_input(tmp_path):
    source = tmp_path / "data.jsonl"
    source.write_text('{"id":"one"}\n')
    manifest_path = tmp_path / "manifest.json"
    write_manifest(
        manifest_path,
        build_run_manifest(
            experiment_id="phase-a-fixture",
            inputs={"data": source},
            config={},
            model_id="fixture/model",
            root=tmp_path,
        ),
    )
    source.write_text('{"id":"changed"}\n')
    try:
        verify_manifest(manifest_path)
    except ValueError as exc:
        assert "changed" in str(exc)
    else:
        raise AssertionError("changed input was not rejected")


def test_manifest_does_not_capture_environment_secrets(tmp_path):
    source = tmp_path / "data.jsonl"
    source.write_text("{}\n")
    manifest = build_run_manifest(
        experiment_id="phase-a-fixture",
        inputs={"data": source},
        config={"debug": True},
        model_id="fixture/model",
        extra={"note": "no secrets"},
        root=tmp_path,
    )
    text = json.dumps(manifest)
    assert "OPENAI_API_KEY" not in text
    assert "DEEPSEEK_API_KEY" not in text
