"""Bounded, resumable local validation for the first allocator hypothesis.

Waits for existing pilot runs, evaluates a predeclared gate, then runs three
additional validation pairs and one sensor-band ablation. Makes no network
requests or competition submissions. Every child uses only shipped training data.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time


PILOT = ["building_train_1/fairbanks_1", "building_train_5/singapore_4", "building_train_8/phoenix_4"]
VALIDATION = ["building_train_2/fairbanks_2", "building_train_4/atlanta_5", "building_train_7/denver_5"]
POLICIES = ["pid_water", "thermal_allocator"]


def load(path):
    try:
        return json.loads(path.read_text()) if path.exists() else None
    except json.JSONDecodeError:
        # A child may still be writing its completion file.
        return None


def write(path, data):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(data, indent=2))
    temporary.replace(path)


def compare(root, scenarios, policies=POLICIES, reference_root=None):
    rows = []
    for scenario in scenarios:
        roots = [reference_root or root, root]
        pair = [load(directory / scenario / policy / "summary.json")
                for directory, policy in zip(roots, policies)]
        if any(row is None for row in pair):
            return None
        base, candidate = pair
        if base["steps"] != 8640 or candidate["steps"] != 8640:
            raise ValueError(f"Incomplete episode: {scenario}")
        rows.append({"scenario": scenario, "baseline_loss": base["mean_loss"],
                     "candidate_loss": candidate["mean_loss"],
                     "relative_improvement": 1 - candidate["mean_loss"] / max(base["mean_loss"], 1e-12),
                     "baseline_components": base["mean_components"],
                     "candidate_components": candidate["mean_components"],
                     "candidate_policy_ms": candidate["policy_ms_mean"]})
    baseline_mean = sum(r["baseline_loss"] for r in rows) / len(rows)
    candidate_mean = sum(r["candidate_loss"] for r in rows) / len(rows)
    return {"rows": rows, "baseline_mean": baseline_mean,
            "candidate_mean": candidate_mean,
            "relative_improvement": 1 - candidate_mean / max(baseline_mean, 1e-12),
            "worst_relative_improvement": min(r["relative_improvement"] for r in rows)}


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--root", type=Path, default=Path("results/allocator-v1"))
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--hours", type=float, default=4)
    parser.add_argument("--baseline", choices=["pid_water", "thermal_allocator"], default="pid_water")
    parser.add_argument("--candidate", choices=["thermal_allocator", "lag_aware_allocator", "lag_aware_pid", "adaptive_heating"], default="thermal_allocator")
    parser.add_argument("--reference-root", type=Path,
                        help="Reuse completed matched reference runs; never relaunch reference jobs")
    parser.add_argument("--skip-ablation", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.workers <= 4 or not 0 < args.hours <= 12:
        parser.error("Use 1–4 workers and a positive budget of at most 12 hours")
    args.root.mkdir(parents=True, exist_ok=True)
    lock = (args.root / "campaign.lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    state_path = args.root / "campaign.json"
    previous = load(state_path)
    previous_running = dict(previous.get("running", {})) if previous else {}
    if previous and previous.get("stage") in ("qualified_for_kaggle_validation", "rejected", "failed"):
        print(json.dumps(previous), flush=True)
        return
    policies = [args.baseline, args.candidate]
    source_path = Path("smart_buildings/baselines") / (args.candidate + ".py")
    source_hash = hashlib.sha256(source_path.read_bytes()).hexdigest()
    if previous and previous["controller_sha256"] != source_hash:
        raise SystemExit("Controller changed: use a new campaign directory")
    if previous and previous.get("policies", policies) != policies:
        raise SystemExit("Policies changed: use a new campaign directory")
    state = previous or {"stage": "waiting_for_pilot", "controller_sha256": source_hash,
                         "pilot_scenarios": PILOT, "validation_scenarios": VALIDATION,
                         "policies": policies,
                         "reference_root": str(args.reference_root) if args.reference_root else None,
                         "authorization": "User authorized autonomous grounded experiments and validation",
                         "gate": "Pilot mean improves >=5%; no scenario regresses >10%; then validate three additional pairs",
                         "network_actions": "None; Kaggle packaging/submission requires a separate validated artifact"}
    state["supervisor_pid"] = os.getpid()
    deadline = time.monotonic() + args.hours * 3600
    running = {}
    try:
        while time.monotonic() < deadline:
            state["checked_at_unix"] = time.time()
            if hashlib.sha256(source_path.read_bytes()).hexdigest() != source_hash:
                state.update(stage="failed", error="Controller source changed during campaign")
                break
            if previous_running:
                live_previous = {}
                for key, pid in previous_running.items():
                    try:
                        command = Path(f"/proc/{pid}/cmdline").read_bytes().decode().split("\0")
                    except (FileNotFoundError, PermissionError):
                        command = []
                    if "smart_buildings.diagnose_policy" in command and key in command:
                        live_previous[key] = pid
                    elif load(Path(key) / "summary.json") is None:
                        state.update(stage="failed", error=f"Previous evaluation ended without a complete result: {key}")
                        raise RuntimeError(f"Previous evaluation ended without a complete result: {key}")
                previous_running = live_previous
                if live_previous:
                    state.update(stage="waiting_for_previous_children", running=live_previous)
                    write(state_path, state)
                    time.sleep(30)
                    continue
                state["running"] = {}
            pilot = compare(args.root, PILOT, policies, args.reference_root)
            if pilot is None:
                write(state_path, state)
                time.sleep(30)
                continue
            state["pilot"] = pilot
            for scenario in PILOT:
                candidate = load(args.root / scenario / args.candidate / "summary.json")
                recorded = candidate["source_sha256"][str(source_path)]
                if recorded != source_hash:
                    raise ValueError(f"Candidate source differs from campaign: {scenario}")
            if pilot["relative_improvement"] < .05 or pilot["worst_relative_improvement"] < -.10:
                state.update(stage="rejected", reason="Predeclared pilot improvement/regression gate failed")
                break
            state["stage"] = "validating_transfer"
            tasks = [(scenario, policy, False) for scenario in VALIDATION for policy in policies
                     if policy == args.candidate or args.reference_root is None]
            if not args.skip_ablation:
                # This is the slow case; overlap it with the validation pairs.
                tasks.insert(0, (PILOT[1], args.baseline, True))
            for key, (process, stream) in list(running.items()):
                if process.poll() is not None:
                    stream.close()
                    del running[key]
                    if process.returncode:
                        state.update(stage="failed", error=f"Evaluation failed: {key}, code {process.returncode}")
                        write(state_path, state)
                        return
            for scenario, policy, ablation in tasks:
                name = policy + ("_sensor_bands" if ablation else "")
                output = args.root / scenario / name
                key = str(output)
                if (output / "summary.json").exists() or key in running:
                    continue
                if len(running) >= args.workers:
                    break
                output.mkdir(parents=True, exist_ok=True)
                stream = (output / "run.log").open("a")
                cmd = [".venv/bin/python", "-u", "-m", "smart_buildings.diagnose_policy",
                       "--scenario", "smart_buildings/training_scenarios/" + scenario,
                       "--policy", policy, "--steps", "8640", "--output", str(output)]
                if ablation:
                    cmd.append("--sensor-bands")
                env = os.environ.copy()
                env.update(OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1",
                           MPLCONFIGDIR="/tmp/matplotlib-smart-buildings")
                process = subprocess.Popen(cmd, stdout=stream, stderr=subprocess.STDOUT, env=env)
                running[key] = process, stream
            state["running"] = {key: process.pid for key, (process, _) in running.items()}
            validation = compare(args.root, VALIDATION, policies, args.reference_root)
            if validation is not None:
                combined = compare(args.root, PILOT + VALIDATION, policies, args.reference_root)
                state["validation"] = validation
                state["combined"] = combined
                ablation = load(args.root / PILOT[1] / (args.baseline + "_sensor_bands") / "summary.json")
                if ablation is not None:
                    state["sensor_band_ablation"] = {k: ablation[k] for k in
                                                    ("scenario", "mean_loss", "mean_components", "steps")}
                if not running and (args.skip_ablation or ablation is not None):
                    passed = (combined["relative_improvement"] >= .05
                              and validation["relative_improvement"] >= 0
                              and combined["worst_relative_improvement"] >= -.10
                              and all(r["candidate_policy_ms"] < 10 for r in combined["rows"]))
                    state["stage"] = "qualified_for_kaggle_validation" if passed else "rejected"
                    state["reason"] = "Full-episode transfer/latency gate passed" if passed else "Transfer/latency gate failed"
                    break
            write(state_path, state)
            time.sleep(30)
        else:
            state.update(stage="time_budget_reached", reason="Bounded local campaign ended")
    finally:
        for process, stream in running.values():
            if process.poll() is None:
                process.terminate()
            stream.close()
        state["running"] = {}
        write(state_path, state)
        print(json.dumps(state), flush=True)


if __name__ == "__main__":
    main()
