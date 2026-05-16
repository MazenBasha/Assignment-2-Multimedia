"""Trigger a Kaggle Notebook run from your laptop, headlessly.

You stay in VSCode. This script:
  1. Generates a temporary `kernel-metadata.json` describing the run
     (GPU on, internet on, MIMIC-CXR dataset attached).
  2. Pushes `kaggle/notebook.ipynb` as a Kaggle kernel via the CLI.
  3. Polls until the run finishes (or fails).
  4. Downloads everything the notebook wrote under `/kaggle/working/outputs/`
     back into your local `outputs/` directory.

Prereqs (one time):
  * `pip install kaggle`
  * Put your kaggle.json in `~/.kaggle/kaggle.json` (Windows:
    `%USERPROFILE%\.kaggle\kaggle.json`) -- get it from
    kaggle.com -> Account -> Create New Token.
  * Set up `HF_TOKEN` and `ANTHROPIC_API_KEY` as Kaggle Secrets
    **once**, in any Kaggle notebook UI. Secrets are user-scoped and
    persist across kernels -- you do this just once.

Usage:
  python scripts/run_on_kaggle.py                       # push, watch, download
  python scripts/run_on_kaggle.py --no-wait             # fire and forget
  python scripts/run_on_kaggle.py --status              # check last run
  python scripts/run_on_kaggle.py --pull                # download outputs only
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path


DEFAULT_SLUG_BASE = "cxr-rag-run"
DEFAULT_DATASET = "simhadrisadaram/mimic-cxr-dataset"
NOTEBOOK_PATH = Path("kaggle/notebook.ipynb")
LOCAL_OUTPUT_DIR = Path("outputs")
META_VERSION_NOTES = "DSAI 413 Assignment 2 -- auto-pushed from laptop"


def _check_kaggle_cli() -> None:
    try:
        subprocess.run(["kaggle", "--version"], check=True, capture_output=True, text=True)
    except (FileNotFoundError, subprocess.CalledProcessError):
        print("ERROR: `kaggle` CLI not found. Run `pip install kaggle` and put "
              "kaggle.json in ~/.kaggle/.", file=sys.stderr)
        sys.exit(1)


def _kaggle_username() -> str:
    """Read kaggle.json to extract the username."""
    candidates = [
        Path.home() / ".kaggle" / "kaggle.json",
        Path(os.environ.get("KAGGLE_CONFIG_DIR", "")) / "kaggle.json",
    ]
    for p in candidates:
        if p.exists():
            try:
                d = json.loads(p.read_text(encoding="utf-8"))
                if "username" in d:
                    return d["username"]
            except Exception:
                pass
    print("ERROR: cannot find kaggle.json. Place it at ~/.kaggle/kaggle.json.",
          file=sys.stderr)
    sys.exit(1)


def _make_kernel_metadata(
    username: str,
    slug: str,
    *,
    enable_gpu: bool = True,
    enable_internet: bool = True,
    dataset: str = DEFAULT_DATASET,
) -> dict:
    return {
        "id":                  f"{username}/{slug}",
        "title":               "DSAI 413 -- ColPali-RAG vs MedGemma",
        "code_file":           "notebook.ipynb",
        "language":            "python",
        "kernel_type":         "notebook",
        "is_private":          "true",
        "enable_gpu":          "true" if enable_gpu else "false",
        "enable_internet":     "true" if enable_internet else "false",
        "dataset_sources":     [dataset],
        "kernel_sources":      [],
        "competition_sources": [],
    }


def push(slug: str) -> str:
    """Stage the notebook + metadata in a temp dir and `kaggle kernels push`.

    Returns the fully-qualified kernel id (user/slug).
    """
    _check_kaggle_cli()
    username = _kaggle_username()
    kernel_id = f"{username}/{slug}"

    if not NOTEBOOK_PATH.exists():
        print(f"ERROR: notebook not found at {NOTEBOOK_PATH}", file=sys.stderr)
        sys.exit(1)

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        shutil.copy(NOTEBOOK_PATH, tmp_path / "notebook.ipynb")
        meta = _make_kernel_metadata(username, slug)
        (tmp_path / "kernel-metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

        print(f"Pushing kernel {kernel_id} ...")
        subprocess.run(["kaggle", "kernels", "push", "-p", str(tmp_path)], check=True)
        print(f"Pushed.   View: https://www.kaggle.com/code/{kernel_id}")
    return kernel_id


def wait(kernel_id: str, poll_seconds: int = 30) -> str:
    """Block until the kernel finishes. Returns the final status string."""
    print(f"Polling {kernel_id} every {poll_seconds}s ...")
    while True:
        r = subprocess.run(
            ["kaggle", "kernels", "status", kernel_id],
            capture_output=True, text=True,
        )
        out = (r.stdout or "") + (r.stderr or "")
        lower = out.lower()
        ts = time.strftime("%H:%M:%S")
        line = out.strip().splitlines()[-1] if out.strip() else "(no status)"
        print(f"[{ts}] {line}")

        if "complete" in lower or "kernelversionstatus.complete" in lower:
            return "complete"
        if any(k in lower for k in ("error", "cancel", "kernelworkererror", "failed")):
            return "failed"
        time.sleep(poll_seconds)


def pull(kernel_id: str, dest: Path = LOCAL_OUTPUT_DIR) -> None:
    """Download everything the kernel wrote into /kaggle/working/."""
    dest.mkdir(parents=True, exist_ok=True)
    print(f"Downloading outputs to {dest}/ ...")
    subprocess.run(
        ["kaggle", "kernels", "output", kernel_id, "-p", str(dest)],
        check=True,
    )
    interesting = list(dest.glob("**/metrics_mode_*.json")) + list(dest.glob("**/mode_*.jsonl"))
    print(f"Got {sum(1 for _ in dest.iterdir())} top-level entries; "
          f"{len(interesting)} look like our metric/output files.")


def cmd_status(slug: str) -> None:
    username = _kaggle_username()
    kernel_id = f"{username}/{slug}"
    subprocess.run(["kaggle", "kernels", "status", kernel_id], check=False)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--slug", default=DEFAULT_SLUG_BASE,
                    help=f"Kernel slug (default: {DEFAULT_SLUG_BASE}). "
                         "Pushes overwrite the existing kernel of the same slug.")
    ap.add_argument("--no-wait",   action="store_true", help="Push and exit; don't poll.")
    ap.add_argument("--no-pull",   action="store_true", help="Skip downloading outputs after completion.")
    ap.add_argument("--status",    action="store_true", help="Just check the kernel's current status.")
    ap.add_argument("--pull",      action="store_true", help="Skip push; only download outputs.")
    ap.add_argument("--poll-secs", type=int, default=30)
    args = ap.parse_args()

    _check_kaggle_cli()
    username = _kaggle_username()
    kernel_id = f"{username}/{args.slug}"

    if args.status:
        cmd_status(args.slug)
        return

    if args.pull:
        pull(kernel_id)
        return

    push(args.slug)
    if args.no_wait:
        print("Exiting without waiting. Use --status to check later.")
        return
    status = wait(kernel_id, poll_seconds=args.poll_secs)
    print(f"Final status: {status}")
    if status == "complete" and not args.no_pull:
        pull(kernel_id)


if __name__ == "__main__":
    main()
