"""Run reproducible policy comparisons with strict actions and physical logs.

All simulator internals used here are evaluation diagnostics. Controllers receive
only the public metadata, action specification, and observation vector.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np

from smart_buildings.evaluate import (
    POLICY_MAP,
    create_building_meta_data,
    load_hybrid_environment,
)


def run(scenario: Path, policy_key: str, steps: int, output: Path, sensor_bands: bool = False) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    env = load_hybrid_environment(scenario / "config.gin")
    env.metrics_path = None
    env._occupancy_normalization_constant = 125.0
    metadata = create_building_meta_data(env, scenario)
    policy = POLICY_MAP[policy_key](metadata, env.action_spec())
    if sensor_bands:
        if not hasattr(policy, "USE_SENSOR_BAND"):
            raise ValueError("Selected controller does not support sensor-band ablation")
        policy.USE_SENSOR_BAND = True
    run_name = policy_key + ("_sensor_bands" if sensor_bands else "")
    source_files = sorted(Path("smart_buildings/baselines").glob("*.py"))
    source_hashes = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in source_files}
    source_dir = output / "source"
    source_dir.mkdir(exist_ok=True)
    for path in source_files:
        (source_dir / path.name).write_bytes(path.read_bytes())
    obs, _ = env.reset()
    simulator = env.building.simulator
    vavs = list(simulator._hvac.vavs.values())
    ahus = list(simulator._hvac.air_handler.ahus)
    water = simulator._hvac.hot_water_system
    logs: dict[str, list] = {key: [] for key in (
        "observations", "continuous_actions", "discrete_actions", "rewards",
        "comfort", "energy", "carbon", "water_temperature", "water_flow",
        "zone_temperature", "zone_airflow", "zone_reheat_w", "zone_net_heat_w",
        "ahu_supply_temperature", "ahu_pressure", "outside_temperature",
        "water_limited_heating_zones", "policy_ms",
    )}
    start = time.monotonic()
    for step in range(steps):
        logs["observations"].append(np.asarray(obs, dtype=np.float32).copy())
        before = time.perf_counter()
        action = policy.get_action(obs)
        policy_ms = (time.perf_counter() - before) * 1000
        for key, dtype in (("continuous_action", np.float32), ("discrete_action", np.int32)):
            value = action[key]
            spec = env.action_spec()[key]
            if value.dtype != dtype or value.shape != spec.shape:
                raise ValueError(f"Wrong action dtype/shape at step {step}: {key}")
            if not np.isfinite(value).all() or not ((value >= spec.low) & (value <= spec.high)).all():
                raise ValueError(f"Nonfinite/out-of-bounds action at step {step}: {key}")
        obs, reward, terminated, truncated, _ = env.step(action)
        _, response = env.get_reward_info_and_response()
        average = simulator.building.get_zone_average_temps()
        supply = {}
        for ahu in ahus:
            rooms = simulator._hvac.air_handler.ahu_zones_map[ahu]
            areas = [len(simulator.building.room_dict[z]) for z in rooms]
            return_temp = float(np.average([average[z] for z in rooms], weights=areas))
            supply[ahu.device_id()] = ahu.get_supply_air_temp(
                return_temp, ahu.outside_air_temperature_sensor
            )
        water_temp = water.supply_water_temperature_sensor
        inlet = [supply[v.air_handler.device_id()] for v in vavs]
        temps = [v.zone_air_temperature for v in vavs]
        reheat = [v.compute_reheat_energy_rate(t, water_temp) for v, t in zip(vavs, inlet)]
        net_heat = [v.compute_energy_applied_to_zone(tz, ta, water_temp)
                    for v, tz, ta in zip(vavs, temps, inlet)]
        limited = 0
        for v, ta, tz in zip(vavs, inlet, temps):
            if ta < tz and water_temp > ta and int(v.air_handler.run_command):
                capacity_air = 1.2 * 1006 * v.flow_rate_demand
                capacity_water = 1000 * 4180 * v._reheat_flow_rate()
                effectiveness = capacity_water / max(capacity_air, 1e-12)
                if 0 < effectiveness < 1 and ta + effectiveness * (water_temp - ta) < 308.15:
                    limited += 1
        values = {
            "continuous_actions": action["continuous_action"].copy(),
            "discrete_actions": action["discrete_action"].copy(),
            "rewards": float(reward),
            "comfort": -response.normalized_productivity_regret,
            "energy": response.normalized_energy_cost,
            "carbon": response.normalized_carbon_emission,
            "water_temperature": water_temp,
            "water_flow": water.total_flow_rate,
            "zone_temperature": temps,
            "zone_airflow": [v.flow_rate_demand for v in vavs],
            "zone_reheat_w": reheat,
            "zone_net_heat_w": net_heat,
            "ahu_supply_temperature": list(supply.values()),
            "ahu_pressure": [a.supply_air_static_pressure_sensor for a in ahus],
            "outside_temperature": ahus[0].outside_air_temperature_sensor,
            "water_limited_heating_zones": limited,
            "policy_ms": policy_ms,
        }
        for key, value in values.items():
            logs[key].append(value)
        if (step + 1) % 288 == 0 or step + 1 == steps or terminated or truncated:
            progress = {"policy": run_name, "scenario": str(scenario),
                        "steps": step + 1, "return": float(np.sum(logs["rewards"])),
                        "seconds": time.monotonic() - start}
            (output / "progress.json").write_text(json.dumps(progress, indent=2))
            print(json.dumps(progress), flush=True)
        if terminated or truncated:
            break
    arrays = {k: np.asarray(v, dtype=np.float32) for k, v in logs.items()}
    count = len(logs["rewards"])
    component_loss = {key: float(np.mean(arrays[key])) for key in ("comfort", "energy", "carbon")}
    weighted = .6 * component_loss["comfort"] + .2 * component_loss["energy"] + .2 * component_loss["carbon"]
    mean_loss = -float(np.mean(logs["rewards"]))
    if not np.isclose(weighted, mean_loss, atol=2e-6):
        raise ValueError(f"Component/reward mismatch: {weighted}, {mean_loss}")
    summary = {
        "policy": run_name, "scenario": str(scenario), "steps": count,
        "sensor_band_ablation": sensor_bands,
        "total_reward": float(np.sum(logs["rewards"])), "mean_loss": mean_loss,
        "mean_components": component_loss,
        "elapsed_seconds": time.monotonic() - start,
        "policy_ms_mean": float(np.mean(logs["policy_ms"])),
        "policy_ms_p99": float(np.quantile(logs["policy_ms"], .99)),
        "mean_water_limited_heating_zones": float(np.mean(logs["water_limited_heating_zones"])),
        "source_sha256": source_hashes,
        "observation_fields": metadata["observation_field_names"],
        "zone_ids": [v.device_id() for v in vavs],
        "interpretation": "Local full episode" if count == 8640 else "Short diagnostic; not a leaderboard estimate",
        "physical_probe_timing": "Device heat/flow probes recomputed after the step; reward components are the scored interval values",
    }
    np.savez_compressed(output / "trajectory.npz", **arrays)
    (output / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: v for k, v in summary.items()
                      if k not in ("observation_fields", "source_sha256", "zone_ids")}), flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--scenario", type=Path, required=True)
    parser.add_argument("--policy", choices=sorted(POLICY_MAP), required=True)
    parser.add_argument("--steps", type=int, default=8640)
    parser.add_argument("--sensor-bands", action="store_true",
                        help="Ablation: use observed bands in a compatible reference controller")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.steps <= 0:
        parser.error("--steps must be positive")
    if (args.output / "summary.json").exists():
        parser.error("Output already contains a completed run; choose a new directory")
    run(args.scenario, args.policy, args.steps, args.output, args.sensor_bands)


if __name__ == "__main__":
    main()
