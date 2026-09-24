"""Submit the fine-tune to Kaggle and (optionally) wait for it and fetch the result.

Turns a multi-step manual handoff into one command by driving the Kaggle CLI:

    uv run python training/kaggle_run.py submit             # create/version dataset + push kernel
    uv run python training/kaggle_run.py watch              # poll until done, download the model
    uv run python training/kaggle_run.py submit --watch     # both

Requires the Kaggle CLI (`uv tool install kaggle`) and an authenticated Kaggle account. The CLI
supports `kaggle auth login`, `KAGGLE_API_TOKEN`, `~/.kaggle/access_token`, and legacy
`~/.kaggle/kaggle.json`. Nothing here reads or prints the token. Official documentation:
https://github.com/Kaggle/kaggle-cli/blob/main/docs/README.md#authentication (checked 2026-09-24,
main branch).

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
            "could not determine the Kaggle username. Run `kaggle auth login`, set "
            "KAGGLE_API_TOKEN, or place a token at ~/.kaggle/access_token, and ensure the CLI is "
            "installed (`uv tool install kaggle`).\n"
            f"{out.stderr[:300]}"
        )
    return name


def _shipped_files():
    """The dataset manifest, read from its one source so the two cannot drift apart."""
    sys.path.insert(0, str(ROOT / "training"))
    import make_kaggle_dataset

    return make_kaggle_dataset.FILES


def submit_dataset(owner: str) -> None:
    print("packaging dataset ...")
    subprocess.run(
        [sys.executable, str(ROOT / "training" / "make_kaggle_dataset.py"), "--owner", owner],
        check=True,
        capture_output=True,
        text=True,
    )
    folder = STAGE / "jev-dealership-data"
    ref = f"{owner}/{DATASET_SLUG}"

    # `datasets create` reports success even when the dataset already exists, and silently does
    # nothing — which trained a kernel on stale data once. So decide explicitly.
    exists = kaggle("datasets", "files", ref, check=False).returncode == 0
    if exists:
        print(f"versioning existing dataset {ref} ...")
        res = kaggle(
            "datasets", "version", "-p", str(folder), "-m", "regenerated training data",
            "--dir-mode", "zip", check=False,
        )
        print("  versioned" if res.returncode == 0 else f"  version failed: {(res.stderr or res.stdout)[:300]}")
    else:
        print(f"creating dataset {ref} ...")
        res = kaggle("datasets", "create", "-p", str(folder), "--dir-mode", "zip", check=False)
        print("  created" if res.returncode == 0 else f"  create failed: {(res.stderr or res.stdout)[:300]}")

    # Verify the bytes actually landed, rather than trusting the exit code. Kaggle processes a new
    # version asynchronously, so poll briefly before declaring a mismatch.
    #
    # **Every** shipped file, not just the first. This checked only `synthetic.jsonl`, so a round
    # where the changed files were `severity_train.jsonl` and `acceptance_train.jsonl` would have
    # reported OK while the kernel trained on the previous copy of the data that mattered - the same
    # "the check is pointed at the wrong thing" failure as the held-out guard.
    def remote_size(name: str) -> str:
        listing = kaggle("datasets", "files", ref, check=False)
        line = next((l for l in (listing.stdout or "").splitlines() if name in l), "")
        return line.split()[1] if len(line.split()) > 1 else "?"

    names = [dst for _, dst in _shipped_files()]
    sizes = {name: (folder / name).stat().st_size for name in names}
    remote = {name: "?" for name in names}
    for _ in range(10):
        remote = {name: remote_size(name) for name in names}
        if all(str(sizes[n]) == remote[n] for n in names):
            break
        time.sleep(15)

    mismatched = [n for n in names if str(sizes[n]) != remote[n]]
    for name in names:
        flag = "OK" if name not in mismatched else "MISMATCH"
        print(f"  {name:26s} local={sizes[name]} remote={remote[name]}  [{flag}]")
    if mismatched:
        print(f"  !! {', '.join(mismatched)} did not update; the kernel would train on stale data")


def submit_kernel(owner: str) -> None:
    stage = STAGE / KERNEL_SLUG
    stage.mkdir(parents=True, exist_ok=True)
    shutil.copy(ROOT / "notebooks" / NOTEBOOK, stage / NOTEBOOK)
    meta = {
        "id": f"{owner}/{KERNEL_SLUG}",
        "title": "Laya dealership fine-tune",
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

    audit_snapshot(out_dir)


def audit_snapshot(out_dir: Path) -> None:
    """Record what the checkpoint was trained on, before anyone quotes a number from it.

    A checkpoint is scored on cases its own snapshot may contain, and the snapshot on disk beside
    the weights is what it trained on - not whatever `data/calls/` holds today. v6 was published at
    0.963 destination while its snapshot carried `gen-01` verbatim inside a training row and
    `sev-04` six times. That was found by a second reader, by hand, after the numbers were quoted.

    So the audit runs at download and writes a manifest next to the weights. Its verdict travels
    with the checkpoint instead of depending on someone remembering.
    """
    sys.path.insert(0, str(ROOT / "src"))
    from jev_classifier import snapshots

    report = snapshots.audit_snapshot(out_dir)
    if not report["files"]:
        return
    (out_dir / snapshots.MANIFEST).write_text(json.dumps(report, indent=2) + "\n")
    if report["held_out_clean"]:
        print("\n  held out: no evaluated case appears in the packaged training data")
    else:
        print(
            f"\n  WARNING: {len(report['leaks'])} evaluated case(s) appear in this checkpoint's "
            "packaged training data. Its scores are not fully held out:"
        )
        for leak in report["leaks"][:6]:
            print(f"    {leak['eval_id']} ({leak['eval_file']}) in {leak['train_file']}")
        print(f"  recorded in {snapshots.MANIFEST} - quote the numbers with this caveat.")


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
