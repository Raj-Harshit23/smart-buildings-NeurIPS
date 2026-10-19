"""Physical regression and metadata-equivariance checks for the new controller."""

import copy
from pathlib import Path

import numpy as np
import pytest

from smart_buildings.baselines.pid_plus import PIDWaterResetPolicy
from smart_buildings.baselines.thermal_allocator import ThermalAllocatorPolicy
from smart_buildings.evaluate import create_building_meta_data, load_hybrid_environment


@pytest.fixture(scope="module")
def environment():
    scenario = Path("smart_buildings/training_scenarios/building_train_1/fairbanks_1")
    env = load_hybrid_environment(scenario / "config.gin")
    return env, create_building_meta_data(env, scenario)


def observation(env, metadata, temperature, outside):
    obs, _ = env.reset()
    obs = obs.copy()
    normalizer = metadata["observation_normalization_metadata"]["zone_air_temperature_sensor"]
    for i, name in enumerate(metadata["observation_field_names"]):
        if name.endswith("zone_air_temperature_sensor"):
            obs[i] = (temperature - normalizer["sample_mean"]) / normalizer["sample_std"]
        elif name.endswith("outside_air_temperature_sensor"):
            obs[i] = outside
        elif name.endswith("zone_air_heating_temperature_setpoint_sensor"):
            obs[i] = 294.0
        elif name.endswith("zone_air_cooling_temperature_setpoint_sensor"):
            obs[i] = 297.0
        elif name.endswith("supply_water_temperature_sensor"):
            obs[i] = 340.0
    names = metadata["observation_field_names"]
    for name, value in {"hod_sin_000": 0, "hod_cos_000": -1,
                        "dow_sin_000": 0, "dow_cos_000": 1}.items():
        obs[names.index(name)] = value
    return obs


def sensor_baseline(metadata, spec):
    policy = PIDWaterResetPolicy(metadata, spec)
    policy.USE_SENSOR_BAND = True
    return policy


def test_cold_airflow_change_improves_actual_water_limited_coil(environment):
    """Use simulator device methods to test the derived cold-air failure mode."""
    env, metadata = environment
    obs = observation(env, metadata, 290.0, 260.0)
    baseline = sensor_baseline(metadata, env.action_spec()).get_action(obs)
    candidate = ThermalAllocatorPolicy(metadata, env.action_spec()).get_action(obs)
    vav = env.building.simulator._hvac.vavs["VAV_1"]
    water = vav.hot_water_system
    indices = metadata["zone_action_indices"]["VAV_1"]
    def delivered_heat(action):
        cont = action["continuous_action"]
        vav.air_handler.supply_air_static_pressure_setpoint = 1000 * (1 + cont[metadata["pressure_indices"][0]])
        vav.damper_setting = .5 * (1 + cont[indices["supply_air_damper_percentage_command"]])
        vav.reheat_valve_setting = .5 * (1 + cont[indices["reheat_valve_percentage_command"]])
        water.differential_pressure_setpoint = 20.0
        water.reset_demand()
        water.add_demand(vav.reheat_flow_factor)
        return vav.compute_energy_applied_to_zone(290.0, 281.0, 340.0)
    baseline_heat = delivered_heat(baseline)
    candidate_heat = delivered_heat(candidate)
    assert candidate_heat > baseline_heat + 1000
    assert candidate["continuous_action"][metadata["pressure_indices"][0]] < baseline["continuous_action"][metadata["pressure_indices"][0]]


def test_observation_and_action_reindexing_preserves_physical_commands(environment):
    env, metadata = environment
    obs = observation(env, metadata, 290.0, 260.0)
    expected = ThermalAllocatorPolicy(metadata, env.action_spec()).get_action(obs)
    m = copy.deepcopy(metadata)
    rng = np.random.default_rng(42)
    obs_order = rng.permutation(len(obs))
    inverse_obs = np.argsort(obs_order)
    m["observation_field_names"] = [metadata["observation_field_names"][i] for i in obs_order]
    m["zone_temp_indices"] = [int(inverse_obs[i]) for i in metadata["zone_temp_indices"]]
    action_order = rng.permutation(metadata["n_continuous_action"])
    inverse_action = np.argsort(action_order)
    for key in ("temp_indices", "pressure_indices", "water_indices", "recirculation_indices"):
        m[key] = [int(inverse_action[i]) for i in metadata[key]]
    for fields in m["zone_action_indices"].values():
        for key, index in fields.items():
            fields[key] = int(inverse_action[index])
    actual = ThermalAllocatorPolicy(m, env.action_spec()).get_action(obs[obs_order])
    np.testing.assert_allclose(actual["continuous_action"][inverse_action], expected["continuous_action"], atol=1e-6)
    np.testing.assert_array_equal(actual["discrete_action"], expected["discrete_action"])


def test_cooling_control_remains_comparable_to_sensor_baseline(environment):
    env, metadata = environment
    obs = observation(env, metadata, 300.0, 310.0)
    expected = sensor_baseline(metadata, env.action_spec()).get_action(obs)
    actual = ThermalAllocatorPolicy(metadata, env.action_spec()).get_action(obs)
    for key in expected:
        np.testing.assert_array_equal(actual[key], expected[key])


def test_reset_clears_history_and_preserves_valid_actions(environment):
    env, metadata = environment
    obs = observation(env, metadata, 290.0, 260.0)
    policy = ThermalAllocatorPolicy(metadata, env.action_spec())
    first = policy.get_action(obs)
    policy.get_action(obs + np.float32(.001))
    policy.reset()
    repeated = policy.get_action(obs)
    for key, spec in env.action_spec().items():
        value = repeated[key]
        assert value.dtype == spec.dtype and value.shape == spec.shape
        assert np.isfinite(value).all()
        assert ((value >= spec.low) & (value <= spec.high)).all()
        np.testing.assert_array_equal(value, first[key])
