"""Run fresh baseline/candidate pilot pairs, then the existing transfer gate.

Local only, bounded, four workers at most. The known difficult Fairbanks pair
starts first. Completed legacy instrumented summaries are never accepted.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import fcntl


def read(path):
    try:
        return json.loads(path.read_text()) if path.exists() else None
    except json.JSONDecodeError:
        return None


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--candidate", choices=["adaptive_heating", "lag_aware_allocator", "lag_aware_pid", "regularized_heating"], required=True)
    parser.add_argument("--baseline", choices=["pid_water", "adaptive_heating"], default="pid_water")
    parser.add_argument("--reference-root", type=Path)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--hours", type=float, default=4)
    args = parser.parse_args()
    if not 1 <= args.workers <= 4 or not 0 < args.hours <= 12:
        parser.error("Use 1–4 workers and a budget in (0,12] hours")
    args.root.mkdir(parents=True, exist_ok=True)
    lock = (args.root / "queue.lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    state_path = args.root / "queue.json"
    previous = read(state_path) or {}
    if previous and (previous.get("baseline", args.baseline) != args.baseline
                     or previous.get("reference_root") != (str(args.reference_root) if args.reference_root else None)):
        raise ValueError("Reference changed: start a new campaign directory")
    paths = sorted(p for p in Path("smart_buildings/baselines").glob("*.py")
                   if not p.name.endswith("_test.py") and not p.name.startswith("test_"))
    paths.append(Path("smart_buildings/diagnose_policy.py"))
    hashes = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    if previous.get("sources") and previous["sources"] != hashes:
        raise ValueError("Source changed: start a new campaign directory")
    state = {"stage": "fresh_full_rollouts", "sources": hashes, "candidate": args.candidate,
             "baseline": args.baseline, "reference_root": str(args.reference_root) if args.reference_root else None,
             "evaluation_protocol": "non_mutating_v2", "supervisor_pid": os.getpid(), "running": {},
             "authorization": "Owner authorized grounded autonomous validation and promising submissions"}
    def save():
        state["checked_at_unix"] = time.time()
        tmp = state_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(state, indent=2));tmp.replace(state_path)
    deadline = time.monotonic() + args.hours * 3600
    # Do not duplicate children orphaned by an interrupted supervisor.
    while previous.get("running"):
        live = {}
        for key, pid in previous["running"].items():
            try: command = Path(f"/proc/{pid}/cmdline").read_bytes().decode().split("\0")
            except OSError: command = []
            if "smart_buildings.diagnose_policy" in command and key in command:
                live[key] = pid
            elif not (Path(key) / "summary.json").exists():
                raise RuntimeError(f"Previous child ended without a complete result: {key}")
        previous["running"] = live
        state.update(stage="waiting_existing_evaluations", running=live);save()
        if not live: break
        if time.monotonic() >= deadline: return
        time.sleep(20)
    scenarios = ["building_train_2/fairbanks_2", "building_train_1/fairbanks_1",
                 "building_train_8/phoenix_4", "building_train_5/singapore_4"]
    tasks = [(s, p) for s in scenarios for p in [args.baseline, args.candidate]
             if p == args.candidate or args.reference_root is None]
    running = {}
    try:
        while time.monotonic() < deadline:
            if any(hashlib.sha256(p.read_bytes()).hexdigest() != hashes[str(p)] for p in paths):
                raise RuntimeError("Source changed during campaign")
            for key, (process, stream) in list(running.items()):
                if process.poll() is not None:
                    stream.close();del running[key]
                    if process.returncode: raise RuntimeError(f"Evaluation failed: {key}")
            for scenario, policy in tasks:
                output = args.root / scenario / policy
                key = str(output)
                complete = read(output / "summary.json")
                if complete:
                    if complete.get("evaluation_protocol") != "non_mutating_v2" or complete["steps"] != 8640:
                        raise ValueError(f"Incomplete or legacy evaluation: {output}")
                    if any(complete["source_sha256"].get(k) != v for k, v in hashes.items()):
                        raise ValueError(f"Evaluation source differs from frozen campaign: {output}")
                    continue
                if key in running: continue
                if len(running) >= args.workers: break
                output.mkdir(parents=True, exist_ok=True)
                stream = (output / "run.log").open("a")
                command = [sys.executable, "-u", "-m", "smart_buildings.diagnose_policy",
                           "--scenario", "smart_buildings/training_scenarios/" + scenario,
                           "--policy", policy, "--steps", "8640", "--output", key]
                environment = os.environ.copy()
                environment.update(OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1",
                                   MPLCONFIGDIR="/tmp/matplotlib-smart-buildings")
                running[key] = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT, env=environment), stream
            state.update(stage="fresh_full_rollouts", running={k: p.pid for k, (p, _) in running.items()});save()
            if not running and all((args.root / s / p / "summary.json").exists() for s, p in tasks):
                state.update(stage="delegated_to_transfer_gate");save()
                remaining = (deadline - time.monotonic()) / 3600
                if remaining <= 0: break
                command = [sys.executable, "-u", "-m", "smart_buildings.run_allocator_campaign",
                           "--root", str(args.root), "--baseline", args.baseline,
                           "--candidate", args.candidate, "--skip-ablation",
                           "--workers", str(args.workers), "--hours", str(remaining)]
                if args.reference_root is not None:
                    command += ["--reference-root", str(args.reference_root)]
                os.execv(sys.executable, command)
            time.sleep(10)
        state.update(stage="time_budget_reached")
    except Exception as error:
        state.update(stage="failed", error=str(error));raise
    finally:
        for process, stream in running.values():
            if process.poll() is None: process.terminate()
            stream.close()
        state["running"] = {};save()


if __name__ == "__main__":
    main()
