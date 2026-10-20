"""Retrospective dispatch diagnostics; future rows supply offline state labels.

References: MATHEMATICAL_FRAMEWORK.md Sections 1, 3, 8–9. Associations with loss
are not counterfactual causal gains. Nothing here is used as a deployed input.
"""
from pathlib import Path
import hashlib
import json
import sys

import numpy as np

from smart_buildings.evaluate import POLICY_MAP, create_building_meta_data, load_hybrid_environment


def main():
    root, output = map(Path, sys.argv[1:3])
    summary = json.loads((root / "summary.json").read_text())
    assert summary.get("evaluation_protocol") == "non_mutating_v2" and summary["steps"] == 8640
    for relative, digest in summary["source_sha256"].items():
        assert hashlib.sha256(Path(relative).read_bytes()).hexdigest() == digest, relative
    scenario = Path(summary["scenario"])
    env = load_hybrid_environment(scenario / "config.gin")
    metadata = create_building_meta_data(env, scenario)
    policy = POLICY_MAP[summary["policy"]](metadata, env.action_spec())
    with np.load(root / "trajectory.npz") as archive:
        obs = archive["observations"]
        binary = archive["discrete_actions"]
        comfort = archive["comfort"][:-1]
        loss = -archive["rewards"][:-1]
    names = metadata["observation_field_names"]
    # obs[t+1] labels physical T[t] and the schedule band at decision t. These
    # future rows are offline evaluation labels, never controller inputs.
    temperature = (obs[1:, policy._indices] * policy.zone_air_temperature_sensor_std
                   + policy.zone_air_temperature_sensor_mean)
    lower = obs[1:, [names.index(z + "_zone_air_heating_temperature_setpoint_sensor") for z in policy._ids]]
    upper = obs[1:, [names.index(z + "_zone_air_cooling_temperature_setpoint_sensor") for z in policy._ids]]
    any_off_hot = np.zeros(len(temperature), dtype=bool)
    any_pure_off_hot = np.zeros_like(any_off_hot)
    rows = []
    for ctrl, indices in zip(policy.ahu_controllers, policy._groups):
        values = temperature[:, indices]
        hot = np.any(values > upper[:, indices] + .5, axis=1)
        pure = np.all(values >= lower[:, indices], axis=1)
        off = binary[:-1, ctrl["run_discrete_idx"]] == 0
        event = off & hot
        any_off_hot |= event
        any_pure_off_hot |= event & pure
        rows.append({"ahu": ctrl["ahu_id"], "zone_count": len(indices),
                     "off_fraction": float(off.mean()), "off_while_zone_hot_fraction": float(event.mean()),
                     "off_pure_cooling_fraction": float((event & pure).mean())})
    flag = obs[:-1, names.index("comfort_mode_now")] > 0
    work = []
    for observation in obs[:-1]:
        hour, day = policy._get_hour_and_day(observation)
        work.append(day < 5 and policy.morning_start_hour <= hour < policy.evening_start_hour)
    events = {}
    for name, event in [("any_off_hot_ahu", any_off_hot), ("pure_cooling_off_hot_ahu", any_pure_off_hot)]:
        events[name] = {"fraction_of_steps": float(event.mean()),
                        "fraction_of_total_comfort_loss_on_these_steps": float(comfort[event].sum() / max(float(comfort.sum()), 1e-12)),
                        "fraction_of_total_loss_on_these_steps": float(loss[event].sum() / max(float(loss.sum()), 1e-12))}
    report = {"scenario": summary["scenario"], "policy": summary["policy"], "labeled_decisions": len(temperature),
              "scope": "Offline labels and loss associations only; not a causal performance estimate",
              "hot_threshold_above_band_k": .5,
              "actual_parent_clock_vs_comfort_flag_mismatch_fraction": float((np.asarray(work) != flag).mean()),
              "events": events, "ahu_rows": rows}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
