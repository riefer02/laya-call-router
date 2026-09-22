"""Submit the fine-tune to Kaggle and (optionally) wait for it and fetch the result.

Turns a multi-step manual handoff into one command by driving the Kaggle CLI:

    uv run python training/kaggle_run.py submit             # create/version dataset + push kernel
    uv run python training/kaggle_run.py watch              # poll until done, download the model
    uv run python training/kaggle_run.py submit --watch     # both

Requires the Kaggle CLI (`uv tool install kaggle`) and a token at `~/.kaggle/access_token`.
Nothing here reads or prints the token.

Pushing a kernel **runs** it on Kaggle's GPUs and consumes your quota — that is the point, but it
is not free, so `submit` prints what it is about to do first.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
KAGGLE = Path.home() / ".local" / "bin" / "kaggle"
STAGE = ROOT / "kaggle"

DATASET_SLUG = "jev-dealership-routing-data"
KERNEL_SLUG = "jev-laya-dealership-fine-tune"
# Kaggle derives the URL slug from the title and refuses to resolve the id if they disagree,
# which shows up as a confusing "Permission 'kernels.get' was denied".
NOTEBOOK = "laya_finetune_dealership_kaggle.ipynb"


def kaggle(*args: str, check: bool = True) -> subprocess.CompletedProcess:
    exe = str(KAGGLE) if KAGGLE.exists() else "kaggle"
    return subprocess.run([exe, *args], capture_output=True, text=True, check=check)


def _kaggle_python() -> str:
    """The uv-tool venv has the `kaggle` package; the project venv does not."""
    tool = Path.home() / ".local" / "share" / "uv" / "tools" / "kaggle" / "bin" / "python"
    return str(tool) if tool.exists() else sys.executable


def username() -> str:
    """Read the account name. `kaggle config view` would print the token, so use the client API."""
    code = (
        "from kaggle.api.kaggle_api_extended import KaggleApi;"
        "a=KaggleApi();a.authenticate();print(a.config_values.get('username'))"
    )
    out = subprocess.run([_kaggle_python(), "-c", code], capture_output=True, text=True)
    name = (out.stdout or "").strip()
    if not name:
        raise SystemExit(
            "could not determine the Kaggle username. Is ~/.kaggle/access_token present, and is "
            "the CLI installed (`uv tool install kaggle`)?\n"
            f"{out.stderr[:300]}"
        )
    return name


def submit_dataset(owner: str) -> None:
    print("packaging dataset ...")
    subprocess.run(
        [sys.executable, str(ROOT / "training" / "make_kaggle_dataset.py"), "--owner", owner],
        check=True,
        capture_output=True,
        text=True,
    )
    folder = STAGE / "jev-dealership-data"
    print(f"creating/versioning dataset {owner}/{DATASET_SLUG} ...")
    created = kaggle("datasets", "create", "-p", str(folder), "--dir-mode", "zip", check=False)
    if created.returncode == 0:
        print("  created")
        return
    if "already exists" in (created.stderr + created.stdout).lower() or "409" in created.stderr:
        versioned = kaggle(
            "datasets", "version", "-p", str(folder), "-m", "regenerate", "--dir-mode", "zip",
            check=False,
        )
        print("  versioned" if versioned.returncode == 0 else f"  version failed: {versioned.stderr[:200]}")
        return
    print(f"  dataset create failed: {(created.stderr or created.stdout)[:400]}")


def submit_kernel(owner: str) -> None:
    stage = STAGE / KERNEL_SLUG
    stage.mkdir(parents=True, exist_ok=True)
    shutil.copy(ROOT / "notebooks" / NOTEBOOK, stage / NOTEBOOK)
    meta = {
        "id": f"{owner}/{KERNEL_SLUG}",
        "title": "JEV Laya dealership fine-tune",
        "code_file": NOTEBOOK,
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": True,
        "enable_internet": True,
        "dataset_sources": [f"{owner}/{DATASET_SLUG}"],
        "competition_sources": [],
        "kernel_sources": [],
    }
    (stage / "kernel-metadata.json").write_text(json.dumps(meta, indent=2) + "\n")

    print("\nabout to push a GPU kernel (consumes Kaggle GPU quota):")
    print(f"  kernel    {owner}/{KERNEL_SLUG}")
    print(f"  notebook  {NOTEBOOK}")
    print(f"  data      {owner}/{DATASET_SLUG}")
    print("  gpu       T4 x2, internet on")
    result = kaggle("kernels", "push", "-p", str(stage), check=False)
    if result.returncode != 0:
        print(f"\npush failed:\n{(result.stderr or result.stdout)[:800]}")
        return
    print(f"\n{result.stdout.strip()}")
    print(f"  https://www.kaggle.com/code/{owner}/{KERNEL_SLUG}")


def watch(owner: str, timeout_min: int, out_dir: Path) -> None:
    ref = f"{owner}/{KERNEL_SLUG}"
    print(f"\nwaiting for the kernel to finish (timeout {timeout_min} min) ...")
    deadline = time.time() + timeout_min * 60
    last = ""
    status = ""
    while time.time() < deadline:
        res = kaggle("kernels", "status", ref, check=False)
        text = (res.stdout or res.stderr).strip()
        status = text.lower()
        if text and text != last:
            print(f"  {text}")
            last = text
        if "complete" in status or "error" in status or "cancelled" in status:
            break
        time.sleep(30)
    else:
        print("  timed out waiting; check the Kaggle URL")
        return

    if "error" in status:
        diagnose(ref, out_dir)
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"\ndownloading output to {out_dir} ...")
    res = kaggle("kernels", "output", ref, "-p", str(out_dir), check=False)
    if res.returncode != 0:
        print(f"  output download failed: {(res.stderr or res.stdout)[:300]}")
        return
    for p in sorted(out_dir.rglob("*")):
        if p.is_file():
            print(f"  {p.relative_to(out_dir)}  {p.stat().st_size:,} bytes")


def diagnose(ref: str, out_dir: Path) -> None:
    """On failure, fetch the log and say what to do about the common causes."""
    out_dir.mkdir(parents=True, exist_ok=True)
    kaggle("kernels", "output", ref, "-p", str(out_dir), check=False)
    logs = list(out_dir.glob("*.log"))
    if not logs:
        print(f"  kernel errored; no log downloaded. See https://www.kaggle.com/code/{ref}")
        return
    raw = logs[0].read_text()
    try:
        rows = json.loads(raw)
    except Exception:
        rows = [json.loads(line) for line in raw.splitlines() if line.strip()]
    text = "".join(r.get("data", "") for r in rows)
    print("\nkernel failed. last output:")
    for line in [l for l in text.splitlines() if l.strip()][-12:]:
        print(f"    {line}")

    if "No GPU is attached" in text or "visible GPUs: 0" in text:
        print(
            "\n  >> Kaggle accepted enable_gpu but attached no accelerator.\n"
            "     The usual cause is that the account is not phone-verified, which Kaggle\n"
            "     requires for GPUs. Verify at https://www.kaggle.com/settings (Phone\n"
            "     Verification), then re-run:  uv run python training/kaggle_run.py submit\n"
            "     Your quota is fine, so this is the only blocker."
        )
    else:
        print(f"\n  full log: {logs[0]}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("action", choices=["submit", "watch"])
    ap.add_argument("--watch", action="store_true", help="with submit: also wait and download")
    ap.add_argument("--timeout", type=int, default=90, help="minutes to wait")
    ap.add_argument("--out", default="models", help="where to download kernel output")
    args = ap.parse_args()

    owner = username()
    print(f"kaggle account: {owner}\n")

    if args.action == "submit":
        submit_dataset(owner)
        submit_kernel(owner)
        if args.watch:
            watch(owner, args.timeout, ROOT / args.out)
    else:
        watch(owner, args.timeout, ROOT / args.out)


if __name__ == "__main__":
    main()
