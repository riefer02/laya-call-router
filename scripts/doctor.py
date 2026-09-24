"""Check whether a checkout is ready to run the local Laya Call Router demo.

This is intentionally dependency-light. It does not start the model, call hosted APIs, or read
credential values. Run it after installing dependencies and building the frontend:

    uv run python scripts/doctor.py
"""

from __future__ import annotations

import importlib.util
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ACTIVE = ROOT / "models" / "active"


def version(command: list[str]) -> str | None:
    exe = shutil.which(command[0])
    if not exe:
        return None
    try:
        result = subprocess.run(
            [exe, *command[1:]], capture_output=True, text=True, timeout=10, check=False
        )
    except (OSError, subprocess.SubprocessError):
        return None
    output = (result.stdout or result.stderr).strip().splitlines()
    return output[0] if output else None


def main() -> int:
    failures = 0
    warnings = 0

    def ok(message: str) -> None:
        print(f"[ok]   {message}")

    def warn(message: str) -> None:
        nonlocal warnings
        warnings += 1
        print(f"[warn] {message}")

    def fail(message: str) -> None:
        nonlocal failures
        failures += 1
        print(f"[FAIL] {message}")

    print("Laya Call Router setup check")
    print(f"repository: {ROOT}")

    if sys.version_info[:2] == (3, 12):
        ok(f"Python {platform.python_version()}")
    else:
        fail(f"Python 3.12 is required; found {platform.python_version()}")

    for command, label in (("uv", "uv"), ("node", "Node.js"), ("npm", "npm")):
        found = version([command, "--version"])
        if found:
            ok(f"{label}: {found}")
        else:
            fail(f"{label} is not installed or not on PATH")

    if importlib.util.find_spec("laya_mlx") is not None:
        ok("laya-mlx is installed")
    else:
        fail("laya-mlx is not installed; run `uv sync`")

    if ACTIVE.is_symlink():
        target = ACTIVE.resolve(strict=False)
        ok(f"models/active -> {target.relative_to(ROOT) if target.is_relative_to(ROOT) else target}")
        weights = target / "model.safetensors"
        if weights.is_file() and weights.stat().st_size > 1_000_000:
            ok(f"active checkpoint weights present ({weights.stat().st_size:,} bytes)")
        elif weights.is_file():
            fail("active checkpoint looks like a Git LFS pointer; run `git lfs pull`")
            fail(f"pointer file: {weights}")
        else:
            fail(f"active checkpoint weights are missing: {weights}")
    else:
        fail("models/active is missing or is not a symlink")

    lfs = shutil.which("git-lfs")
    if lfs:
        ok(f"Git LFS: {version(['git', 'lfs', 'version']) or 'installed'}")
    else:
        warn("Git LFS is not installed; fresh clones may contain a model pointer")

    if (ROOT / ".env.example").is_file():
        ok(".env.example is present")
    else:
        fail(".env.example is missing")

    if (ROOT / "web" / "dist" / "index.html").is_file():
        ok("frontend build exists at web/dist/")
    else:
        warn("frontend build is missing; run `cd web && npm run build`")

    if platform.system() == "Darwin" and platform.machine() in {"arm64", "aarch64"}:
        ok("platform matches the documented Apple Silicon runner")
    else:
        warn(
            f"documented runner is macOS on Apple Silicon; found {platform.system()} {platform.machine()}"
        )

    if shutil.which("kaggle") or (Path.home() / ".local" / "bin" / "kaggle").is_file():
        ok("Kaggle CLI is available (optional)")
    else:
        warn("Kaggle CLI is not installed (only needed for training)")

    kaggle_auth = (
        bool(os.environ.get("KAGGLE_API_TOKEN"))
        or (Path.home() / ".kaggle" / "access_token").is_file()
        or (Path.home() / ".kaggle" / "kaggle.json").is_file()
    )
    if kaggle_auth:
        ok("Kaggle authentication is configured (value not displayed)")
    else:
        warn("Kaggle authentication is not configured; run `kaggle auth login` if training")

    if os.environ.get("KAGGLE_USERNAME"):
        ok("KAGGLE_USERNAME is set (value not displayed)")
    else:
        warn("KAGGLE_USERNAME is unset; pass --owner to training/make_kaggle_dataset.py")

    print()
    if failures:
        print(f"Setup is not ready: {failures} failure(s), {warnings} warning(s).")
        return 1
    print(f"Setup looks ready: {warnings} warning(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
