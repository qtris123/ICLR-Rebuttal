#!/usr/bin/env python3
"""Automated research experiment tracker & runner.

Ensures 100% reproducibility across both Repo A and Repo B by:
1. Creating a structured run directory with timestamp & unique hash.
2. Capturing git commit, branch, and exact uncommitted `patch.diff`.
3. Logging environment details (PyTorch, CUDA, GPU model, package versions).
4. Capturing the full resolved command line and configuration.
5. Streaming logs to console and `run.log`.
6. Appending summary metadata to a master `experiments_registry.tsv`.

Usage:
    python3 launch_experiment.py --repo a --name "tune_ridge_1e-4" -- python run_continual.py ...
    python3 launch_experiment.py --repo b --name "farid_mono_baseline" -- python run_continual.py ...
"""

import argparse
import csv
import datetime
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path


def get_git_info(repo_dir: Path) -> dict:
    """Collect git commit, branch, status, and diff."""
    info = {
        "commit": None,
        "branch": None,
        "is_dirty": False,
        "diff": "",
    }
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repo_dir, text=True
        ).strip()
        info["commit"] = commit

        branch = subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=repo_dir, text=True
        ).strip()
        info["branch"] = branch

        status = subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=repo_dir, text=True
        ).strip()
        info["is_dirty"] = bool(status)

        diff = subprocess.check_output(
            ["git", "diff", "HEAD"], cwd=repo_dir, text=True
        )
        info["diff"] = diff
    except Exception as e:
        info["error"] = str(e)
    return info


def get_env_info() -> dict:
    """Collect hardware and software environment metadata."""
    info = {
        "hostname": platform.node(),
        "os": platform.platform(),
        "python_version": platform.python_version(),
    }
    try:
        import torch

        info["torch_version"] = torch.__version__
        info["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            info["cuda_version"] = torch.version.cuda
            info["gpu_count"] = torch.cuda.device_count()
            info["gpu_name"] = torch.cuda.get_device_name(0)
    except ImportError:
        info["torch_version"] = "not_installed"

    try:
        smi = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=driver_version,name,memory.total",
                "--format=csv,noheader",
            ],
            text=True,
        ).strip()
        info["nvidia_smi"] = smi
    except Exception:
        pass

    return info


def append_registry(registry_path: Path, row: dict):
    """Append a run row to the TSV registry."""
    file_exists = registry_path.exists()
    keys = list(row.keys())

    with open(registry_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys, delimiter="\t")
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def main():
    base_dir = Path(__file__).resolve().parent
    default_runs_dir = base_dir / "experiments" / "runs"
    default_registry = base_dir / "experiments" / "experiments_registry.tsv"

    parser = argparse.ArgumentParser(
        description="Launch and log reproducible ML experiments."
    )
    parser.add_argument(
        "--repo",
        choices=["a", "b", "other"],
        default="b",
        help="Target repo: 'a' (Tri / gated-continual-cartridges) or 'b' (Farid).",
    )
    parser.add_argument(
        "--name",
        type=str,
        required=True,
        help="Brief mnemonic name for the experiment (e.g. mono_16k_ridge0).",
    )
    parser.add_argument(
        "--runs-dir",
        type=str,
        default=str(default_runs_dir),
        help="Root directory to store individual experiment runs.",
    )
    parser.add_argument(
        "--registry",
        type=str,
        default=str(default_registry),
        help="Path to master experiments registry TSV file.",
    )
    parser.add_argument(
        "command",
        nargs=argparse.REMAINDER,
        help="Command to run, starting after '--'.",
    )

    args = parser.parse_args()

    cmd = args.command
    if cmd and cmd[0] == "--":
        cmd = cmd[1:]
    if not cmd:
        parser.error("No command specified to execute.")

    # Determine workspace directory based on repo choice
    if args.repo == "a":
        cand = base_dir / "gated-continual-cartridges"
        repo_dir = cand if cand.exists() else Path("/home/voquangtri2025/gated-continual-cartridges")
    elif args.repo == "b":
        cand = base_dir / "farid" / "gated-continual-cartridges"
        repo_dir = cand if cand.exists() else Path("/home/voquangtri2025/farid/gated-continual-cartridges")
    else:
        repo_dir = Path.cwd()

    now = datetime.datetime.now(datetime.timezone.utc)
    date_str = now.strftime("%Y%m%d_%H%M%S")
    cmd_hash = hashlib.sha1(" ".join(cmd).encode()).hexdigest()[:6]
    clean_name = args.name.strip().replace(" ", "_").replace("/", "_")
    run_id = f"{date_str}_{args.repo}_{clean_name}_{cmd_hash}"

    run_dir = Path(args.runs_dir) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    # Auto-create output directory if --out or --generate-out is in command
    out_file_path = None
    for i, arg in enumerate(cmd):
        if arg in ("--out", "--generate-out") and i + 1 < len(cmd):
            target_p = Path(cmd[i + 1])
            if not target_p.is_absolute():
                target_p = repo_dir / target_p
            target_p.parent.mkdir(parents=True, exist_ok=True)
            if arg == "--out":
                out_file_path = target_p

    print("=" * 70)
    print(f"🚀 LAUNCHING EXPERIMENT: {run_id}")
    print(f"📁 Run directory:       {run_dir}")
    print(f"📂 Repo directory:      {repo_dir}")
    print(f"⚙️  Command:             {' '.join(cmd)}")
    print("=" * 70)

    # 1. Capture Git Metadata & Diff
    git_info = get_git_info(repo_dir)
    diff_text = git_info.pop("diff", "")
    with open(run_dir / "git_info.json", "w") as f:
        json.dump(git_info, f, indent=2)

    with open(run_dir / "patch.diff", "w") as f:
        f.write(diff_text)

    # 2. Capture Environment Metadata
    env_info = get_env_info()
    with open(run_dir / "env.json", "w") as f:
        json.dump(env_info, f, indent=2)

    # 3. Capture Command
    with open(run_dir / "run_command.sh", "w") as f:
        f.write("#!/bin/bash\n")
        f.write(f"cd {repo_dir}\n")
        f.write(" ".join(cmd) + "\n")

    log_path = run_dir / "run.log"

    # 4. Execute with log teeing
    start_time = time.time()
    with open(log_path, "w") as log_file:
        proc = subprocess.Popen(
            cmd,
            cwd=repo_dir,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        for line in proc.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            log_file.write(line)
            log_file.flush()

        proc.wait()
        ret_code = proc.returncode

    duration = time.time() - start_time
    status = "SUCCESS" if ret_code == 0 else f"FAILED({ret_code})"
    print("=" * 70)
    print(f"🏁 EXPERIMENT FINISHED: {status} in {duration:.1f}s")
    print(f"📄 Full log saved to: {log_path}")
    print("=" * 70)

    # 5. Extract metrics if output JSON exists
    metrics_summary = {}
    if out_file_path and out_file_path.exists():
        try:
            with open(out_file_path, "r") as f:
                res_data = json.load(f)
            if "final" in res_data:
                metrics_summary["final_acc"] = res_data["final"].get("acc")
                metrics_summary["final_loss"] = res_data["final"].get("loss")
            if "forget" in res_data:
                metrics_summary["forgetting"] = res_data["forget"].get("mean")
            # Copy r.json to run_dir as metrics.json
            with open(run_dir / "metrics.json", "w") as f:
                json.dump(res_data, f, indent=2)
        except Exception as e:
            metrics_summary["metrics_read_error"] = str(e)

    # 6. Append to Master Registry
    registry_path = Path(args.registry)
    registry_path.parent.mkdir(parents=True, exist_ok=True)

    summary_row = {
        "timestamp": now.isoformat(),
        "run_id": run_id,
        "repo": args.repo,
        "name": args.name,
        "status": status,
        "duration_s": f"{duration:.1f}",
        "branch": git_info.get("branch", ""),
        "commit": (git_info.get("commit") or "")[:8],
        "is_dirty": git_info.get("is_dirty", False),
        **metrics_summary,
        "command": " ".join(cmd),
        "run_dir": str(run_dir),
    }

    append_registry(registry_path, summary_row)
    print(f"📊 Run logged to registry: {registry_path}")


if __name__ == "__main__":
    main()

