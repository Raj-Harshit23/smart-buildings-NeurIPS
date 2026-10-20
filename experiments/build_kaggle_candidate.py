"""Export a frozen controller and prove equivalence on two complete rollouts.

This performs no network actions. The base artifact supplies the already tested
official gateway adapter and pinned Kaggle image metadata. Local qualification
and official gateway completion are separately required before scored submission.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
from pathlib import Path
import subprocess
import sys


POLICIES = {
    "lag_aware_allocator": ("lag_aware_allocator.py", "LagAwareAllocatorPolicy"),
    "lag_aware_pid": ("lag_aware_pid.py", "LagAwarePIDPolicy"),
    "adaptive_heating": ("adaptive_heating.py", "AdaptiveHeatingPolicy"),
}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--tests", type=Path, required=True)
    parser.add_argument("--base-artifact", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--kernel-ref", required=True)
    parser.add_argument("--policy", choices=POLICIES, required=True)
    parser.add_argument("--scenario", action="append", required=True)
    args = parser.parse_args()
    if len(set(args.scenario)) < 2:
        parser.error("Provide at least two distinct complete rollout scenarios")
    source = args.source_root.resolve()
    output = args.output.resolve()
    # An artifact with a publishing/submission marker is immutable.
    if any((output / name).exists() for name in ("launch-state.json", "leaderboard-status.json")):
        raise ValueError("Use a new artifact directory; existing publication state is immutable")
    names = ["pid.py", "pid_plus.py", "thermal_allocator.py", "lag_aware_allocator.py"]
    leaf, class_name = POLICIES[args.policy]
    if args.policy == "adaptive_heating":
        names.append("lag_aware_pid.py")
    if leaf not in names:
        names.append(leaf)
    paths = [source / "smart_buildings/baselines" / name for name in names]
    clean = subprocess.check_output(
        ["git", "-C", str(source), "status", "--porcelain", "--", *map(str, paths)], text=True
    )
    if clean:
        raise ValueError("Controller source must be committed and clean")
    revision = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
    hashes = {str(path.relative_to(source)): digest(path) for path in paths}
    tests = json.loads(args.tests.read_text())
    if tests["passed"] < 87 or any(tests["sources"].get(k) != v for k, v in hashes.items()):
        raise ValueError("Test receipt must identify every exported controller source")
    parts = []
    for path in paths:
        lines = path.read_text().splitlines(keepends=True)
        parts.append("".join(line for line in lines
                             if line != "from __future__ import annotations\n"
                             and not line.startswith("from smart_buildings.baselines.")))
    core = ("# SPDX-License-Identifier: Apache-2.0\n"
            "# Source: https://github.com/Raj-Harshit23/smart-buildings-NeurIPS\n"
            f"# Frozen source revision: {revision}\n\n" + "\n\n".join(parts)
            + f"\n\nclass BasicPolicy({class_name}):\n    pass\n")
    scope = {}
    exec(compile(core, "controller.py", "exec"), scope)
    # Import from the selected snapshot, independent of editable-install cwd.
    sys.path.insert(0, str(source))
    import numpy as np
    from smart_buildings.evaluate import create_building_meta_data, load_hybrid_environment

    module = importlib.import_module("smart_buildings.baselines." + args.policy)
    reference_type = getattr(module, class_name)
    if Path(module.__file__).resolve() != paths[-1].resolve():
        raise ValueError("Imported controller is not from the selected source snapshot")
    rows = []
    for scenario in args.scenario:
        root = args.results_root / scenario / args.policy
        summary = json.loads((root / "summary.json").read_text())
        if summary["steps"] != 8640 or any(summary["source_sha256"].get(k) != v for k, v in hashes.items()):
            raise ValueError(f"Rollout does not match complete frozen source: {scenario}")
        path = source / "smart_buildings/training_scenarios" / scenario
        env = load_hybrid_environment(path / "config.gin")
        metadata = create_building_meta_data(env, path)
        reference = reference_type(metadata, env.action_spec())
        exported = scope["BasicPolicy"](metadata, env.action_spec())
        with np.load(root / "trajectory.npz") as archive:
            observations = archive["observations"]
            continuous = archive["continuous_actions"]
            binary = archive["discrete_actions"]
        if len(observations) != 8640:
            raise ValueError("Incomplete observation trajectory")
        for step, observation in enumerate(observations):
            a, b = reference.get_action(observation), exported.get_action(observation)
            for key, recorded in (("continuous_action", continuous[step]), ("discrete_action", binary[step])):
                spec, value = env.action_spec()[key], b[key]
                if (not np.array_equal(a[key], value) or not np.array_equal(value, recorded)
                        or value.dtype != spec.dtype or value.shape != spec.shape
                        or not np.isfinite(value).all()
                        or not ((value >= spec.low) & (value <= spec.high)).all()):
                    raise ValueError(f"Export/recorded action mismatch: {scenario}, step {step}, {key}")
        rows.append({"scenario": scenario, "steps": 8640, "exact_baseline_equivalence": True})
        print(json.dumps(rows[-1]), flush=True)
    template = json.loads((args.base_artifact / "submission.ipynb").read_text())
    metadata = json.loads((args.base_artifact / "kernel-metadata.json").read_text())
    metadata.update(id=args.kernel_ref, title="Smart Buildings " + args.policy.replace("_", " ").title(),
                    is_private=True, enable_internet=False, enable_gpu=False, enable_tpu=False)
    controller_hash = hashlib.sha256(core.encode()).hexdigest()
    note = (f"# {args.policy}\n\nCausal controller from revision `{revision}`.\n\n"
            f"Controller SHA256: `{controller_hash}`.\n\n"
            "See the repository mathematical framework and experiment record. "
            "No hidden scenario lookup or future disturbance access.\n\n"
            + (source / "NOTICE").read_text() + "\n\n```text\n" + (source / "LICENSE").read_text() + "\n```")
    for cell, text in zip(template["cells"], [note, "".join(template["cells"][1]["source"]), core,
                                            "".join(template["cells"][3]["source"]).replace(
                                                "V0: shipped pid_water controller", args.policy)]):
        cell["source"] = text
        if cell["cell_type"] == "code":
            cell.update(execution_count=None, outputs=[])
    output.mkdir(parents=True, exist_ok=True)
    manifest = {"kernel_ref": args.kernel_ref, "frozen_revision": revision, "sources": hashes,
                "policy_sha256": hashes["smart_buildings/baselines/" + leaf],
                "controller_sha256": controller_hash, "private": True, "internet": False,
                "policy": args.policy,
                "authorization": "Owner authorized grounded experiments and promising Kaggle submissions"}
    report = {"scenarios": rows, "tests_passed": tests["passed"], "sources": hashes,
              "scope": "Exact exported/source/recorded-action equivalence on complete local trajectories"}
    for name, value in [("manifest.json", manifest), ("local-validation.json", report),
                        ("kernel-metadata.json", metadata), ("submission.ipynb", template)]:
        (output / name).write_text(json.dumps(value, indent=2))
    (output / "controller.py").write_text(core)
    for name in ["NOTICE", "LICENSE"]:
        (output / name).write_text((source / name).read_text())
    print(json.dumps(manifest), flush=True)


if __name__ == "__main__":
    main()
