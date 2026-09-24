"""Provenance records for data, training, and evaluation artefacts.

A result is only auditable when the inputs and runtime that produced it travel with it. These
helpers deliberately do not know about secrets: callers pass paths and metadata, never auth
headers or environment dumps.
"""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional

ROOT = Path(__file__).resolve().parents[2]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_json(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def git_revision(root: Path = ROOT) -> Optional[str]:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def package_versions(names: Iterable[str] = ()) -> Dict[str, Optional[str]]:
    """Return installed versions without importing optional packages."""
    from importlib.metadata import PackageNotFoundError, version

    out: Dict[str, Optional[str]] = {}
    for name in names:
        try:
            out[name] = version(name)
        except PackageNotFoundError:
            out[name] = None
    return out


def file_manifest(paths: Mapping[str, Path], root: Path = ROOT) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for label, path in paths.items():
        resolved = path if path.is_absolute() else root / path
        if not resolved.is_file():
            raise FileNotFoundError(f"manifest input {label!r} does not exist: {resolved}")
        out[label] = {
            "path": str(resolved.relative_to(root) if resolved.is_relative_to(root) else resolved),
            "sha256": sha256_file(resolved),
            "bytes": resolved.stat().st_size,
        }
    return out


def build_run_manifest(
    *,
    experiment_id: str,
    inputs: Mapping[str, Path],
    config: Mapping[str, Any],
    model_id: str,
    model_revision: Optional[str] = None,
    seed: Optional[int] = None,
    packages: Iterable[str] = (),
    extra: Optional[Mapping[str, Any]] = None,
    root: Path = ROOT,
) -> Dict[str, Any]:
    """Build a JSON-serialisable manifest with no environment or credential capture."""
    if not experiment_id.strip():
        raise ValueError("experiment_id must not be empty")
    manifest: Dict[str, Any] = {
        "schema_version": 1,
        "experiment_id": experiment_id,
        "root": str(root),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "git_revision": git_revision(root),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "model": {"id": model_id, "revision": model_revision},
        "seed": seed,
        "inputs": file_manifest(inputs, root),
        "config": json.loads(json.dumps(dict(config), sort_keys=True)),
        "packages": package_versions(packages),
    }
    if extra:
        manifest["extra"] = json.loads(json.dumps(dict(extra), sort_keys=True))
    manifest["manifest_sha256"] = sha256_json(manifest)
    return manifest


def write_manifest(path: Path, manifest: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def verify_manifest(path: Path) -> Dict[str, Any]:
    """Read a manifest and verify its self-hash and referenced local files."""
    manifest = json.loads(path.read_text())
    expected = manifest.get("manifest_sha256")
    actual = sha256_json({k: v for k, v in manifest.items() if k != "manifest_sha256"})
    if expected != actual:
        raise ValueError(f"manifest self-hash mismatch: {path}")
    manifest_root = Path(manifest.get("root", str(ROOT)))
    for label, record in (manifest.get("inputs") or {}).items():
        source = Path(record["path"])
        if not source.is_absolute():
            source = manifest_root / source
        if not source.is_file():
            raise FileNotFoundError(f"manifest input {label!r} is missing: {source}")
        if sha256_file(source) != record["sha256"]:
            raise ValueError(f"manifest input {label!r} changed: {source}")
    return manifest
