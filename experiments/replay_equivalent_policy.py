"""Reuse a complete local rollout only after exact causal action equivalence.

For an unchanged seeded simulator and initial state, the identical action
sequence produces the same state/reward path. This is not a new simulator run;
the resulting receipt explicitly identifies its source. Any action mismatch
rejects reuse, without retaining a candidate performance summary.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path
import subprocess
import time

import numpy as np

from smart_buildings.evaluate import POLICY_MAP, create_building_meta_data, load_hybrid_environment


def verify_dynamics(revision, summary):
    source = "smart_buildings/baselines/" + summary["policy"] + ".py"
    frozen = subprocess.check_output(["git", "show", revision + ":" + source])
    if hashlib.sha256(frozen).hexdigest() != summary["source_sha256"][source]:
        raise ValueError("Reference revision does not identify the recorded policy")
    paths = ["smart_buildings/smart_control", "smart_buildings/weather",
             "smart_buildings/training_scenarios", "smart_buildings/diagnose_policy.py"]
    subprocess.run(["git", "diff", "--exit-code", revision, "--", *paths],
                   check=True, stdout=subprocess.PIPE)
    interface = "smart_buildings/evaluate.py"
    def environment_ast(text):
        tree = ast.parse(text)
        def dispatch(node):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target] if isinstance(node, ast.AnnAssign) else []
            return any(isinstance(t, ast.Name) and t.id == "POLICY_MAP" for t in targets)
        tree.body = [n for n in tree.body if not dispatch(n) and not (
            isinstance(n, ast.ImportFrom) and n.module and n.module.startswith("smart_buildings.baselines."))]
        return ast.dump(tree, include_attributes=False)
    original = subprocess.check_output(["git", "show", revision + ":" + interface], text=True)
    if environment_ast(original) != environment_ast(Path(interface).read_text()):
        raise ValueError("Environment interface changed beyond controller dispatch")
    return {"reference_revision": revision, "physics_weather_scenarios_runner_unchanged": True,
            "environment_ast_unchanged_except_controller_dispatch": True}


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--reference-revision", required=True)
    parser.add_argument("--policy", choices=POLICY_MAP, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if (args.output / "summary.json").exists():
        raise ValueError("Existing evaluation must not be overwritten")
    summary = json.loads((args.reference / "summary.json").read_text())
    if summary.get("evaluation_protocol") != "non_mutating_v2":
        raise ValueError("Legacy instrumented trajectory cannot qualify for reuse")
    if summary["steps"] != 8640:
        raise ValueError("Reference must be a complete rollout")
    provenance = verify_dynamics(args.reference_revision, summary)
    scenario = Path(summary["scenario"])
    env = load_hybrid_environment(scenario / "config.gin")
    metadata = create_building_meta_data(env, scenario)
    policy = POLICY_MAP[args.policy](metadata, env.action_spec())
    paths = sorted(p for p in Path("smart_buildings/baselines").glob("*.py")
                   if not p.name.endswith("_test.py") and not p.name.startswith("test_"))
    paths.append(Path("smart_buildings/diagnose_policy.py"))
    hashes = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    with np.load(args.reference / "trajectory.npz") as archive:
        arrays = {k: archive[k] for k in archive.files}
    if len(arrays["observations"]) != 8640:
        raise ValueError("Incomplete reference observations")
    timing = []
    started = time.monotonic()
    for step, observation in enumerate(arrays["observations"]):
        before = time.perf_counter()
        action = policy.get_action(observation)
        timing.append((time.perf_counter() - before) * 1000)
        for key, recorded in (("continuous_action", arrays["continuous_actions"][step]),
                              ("discrete_action", arrays["discrete_actions"][step])):
            value, spec = action[key], env.action_spec()[key]
            if (not np.array_equal(value, recorded) or value.dtype != spec.dtype
                    or value.shape != spec.shape or not np.isfinite(value).all()
                    or not ((value >= spec.low) & (value <= spec.high)).all()):
                raise ValueError(f"Cannot reuse trajectory: action differs at step {step}, {key}")
        if (step + 1) % 2048 == 0:
            print(json.dumps({"replayed_steps": step + 1}), flush=True)
    if any(hashlib.sha256(p.read_bytes()).hexdigest() != hashes[str(p)] for p in paths):
        raise ValueError("Controller source changed during replay")
    elapsed = time.monotonic() - started
    arrays["policy_ms"] = np.asarray(timing, dtype=np.float32)
    result = dict(summary)
    result.update(policy=args.policy, sensor_band_ablation=False,
                  evaluation_mode="exact_action_equivalent_replay",
                  equivalence_provenance=provenance,
                  reference_rollout=str(args.reference.resolve()),
                  source_rollout_elapsed_seconds=summary["elapsed_seconds"],
                  elapsed_seconds=elapsed, policy_ms_mean=float(np.mean(timing)),
                  policy_ms_p99=float(np.quantile(timing, .99)), source_sha256=hashes,
                  interpretation="Exact action-equivalent replay of a complete local seeded rollout")
    args.output.mkdir(parents=True, exist_ok=True)
    source_dir = args.output / "source"
    source_dir.mkdir(exist_ok=True)
    for path in paths:
        (source_dir / path.name).write_bytes(path.read_bytes())
    np.savez_compressed(args.output / "trajectory.npz", **arrays)
    (args.output / "summary.json").write_text(json.dumps(result, indent=2))
    print(json.dumps({k: result[k] for k in ["policy", "scenario", "steps", "mean_loss",
                                           "evaluation_mode", "policy_ms_mean"]}), flush=True)


if __name__ == "__main__":
    main()
