"""Permanent information, prediction Jacobian, reset and large corrections."""
from pathlib import Path

import numpy as np
import pytest

from smart_buildings.baselines.adaptive_heating import AdaptiveHeatingPolicy
from smart_buildings.baselines.regularized_heating import RegularizedHeatingPolicy
from smart_buildings.evaluate import create_building_meta_data, load_hybrid_environment


@pytest.fixture(scope="module")
def data():
    scenario = Path("smart_buildings/training_scenarios/building_train_8/phoenix_4")
    env = load_hybrid_environment(scenario / "config.gin")
    metadata = create_building_meta_data(env, scenario)
    observation, _ = env.reset()
    return metadata, env.action_spec(), observation


def test_unexcited_heat_gain_retains_prior_information(data):
    metadata, spec, obs = data
    policy = RegularizedHeatingPolicy(metadata, spec)
    count = len(policy._ids)
    start = np.full(count, 294.)
    policy._mass_proxy = start.copy()
    policy._equipment = lambda temperature, record: (
        np.zeros(count), start.copy(), np.zeros(count), start.copy())
    prior = policy.prior.copy()
    for _ in range(1024):
        policy._learn_delayed_transition(start, start, {})
    np.testing.assert_allclose(policy._theta[:, 2], prior[:, 2], atol=1e-10)
    np.testing.assert_allclose(policy._covariance[:, 2, 2], 100., atol=1e-8)
    assert np.linalg.eigvalsh(policy.information - policy.precision).min() >= -1e-8


def test_temperature_prediction_jacobian_matches_finite_difference(data):
    metadata, spec, obs = data
    policy = RegularizedHeatingPolicy(metadata, spec)
    start = np.asarray([policy._denormalize_zone_temp(obs[i]) for i in policy._indices])
    policy._mass_proxy = start.copy()
    continuous = np.zeros(metadata["n_continuous_action"], dtype=np.float32)
    continuous[metadata["temp_indices"]] = -1
    binary = np.ones(metadata["n_discrete_action"], dtype=np.int32);binary[-1] = 0
    record = policy._record(obs, {"continuous_action": continuous, "discrete_action": binary})
    equipment = policy._equipment(start, record)
    analytic, _, _ = policy.features_and_prediction(start, start, record)
    original = policy._theta.copy()
    for j in range(4):
        epsilon = 1e-4 if j == 2 else 1e-5
        policy._theta = original.copy();policy._theta[:, j] += epsilon
        plus = policy._predict(start, equipment, policy._mass_proxy)
        policy._theta = original.copy();policy._theta[:, j] -= epsilon
        minus = policy._predict(start, equipment, policy._mass_proxy)
        np.testing.assert_allclose((plus - minus) / (2 * epsilon), analytic[:, j], atol=1e-5)


def test_confident_large_correction_reaches_control_and_actions_remain_finite(data):
    metadata, spec, obs = data
    policies = [kind(metadata, spec) for kind in [AdaptiveHeatingPolicy, RegularizedHeatingPolicy]]
    for policy in policies:
        policy.get_action(obs)
        policy._updates = 400
        policy._residual_mae[:] = .1
        policy._predict = lambda start, equipment, mass: start + 5.
        action = policy.get_action(obs)
        for key, value in action.items():
            assert np.isfinite(value).all() and value.dtype == spec[key].dtype
            assert value.shape == spec[key].shape
            assert ((value >= spec[key].low) & (value <= spec[key].high)).all()
    np.testing.assert_allclose(policies[1]._previous_temps - policies[0]._previous_temps, 3.5, atol=1e-4)


def test_reset_rebuilds_information_and_prior(data):
    metadata, spec, obs = data
    policy = RegularizedHeatingPolicy(metadata, spec)
    prior = policy.prior.copy()
    first = policy.get_action(obs)
    policy.get_action(obs);policy.get_action(obs)
    assert policy._updates == 1
    policy.reset()
    np.testing.assert_array_equal(policy._theta, prior)
    np.testing.assert_array_equal(policy.information, policy.precision)
    assert not policy._commands and policy._updates == 0
    repeated = policy.get_action(obs)
    for key in first: np.testing.assert_array_equal(first[key], repeated[key])


@pytest.mark.parametrize("scenario", ["building_train_2/fairbanks_2", "building_train_8/phoenix_4"])
@pytest.mark.parametrize("temperature,outside", [(288., 260.), (300., 310.)])
def test_mature_actions_adapt_to_real_building_sizes(scenario, temperature, outside):
    path = Path("smart_buildings/training_scenarios") / scenario
    env = load_hybrid_environment(path / "config.gin")
    metadata = create_building_meta_data(env, path)
    obs, _ = env.reset();policy = RegularizedHeatingPolicy(metadata, env.action_spec())
    obs[metadata["zone_temp_indices"]] = (temperature - policy.zone_air_temperature_sensor_mean) / policy.zone_air_temperature_sensor_std
    for i, name in enumerate(metadata["observation_field_names"]):
        if name.endswith("outside_air_temperature_sensor"): obs[i] = outside
        elif name.endswith("supply_water_temperature_sensor"): obs[i] = 340.
    policy._updates = 1000;policy._residual_mae[:] = .1
    for _ in range(4):
        action = policy.get_action(obs)
        for key, value in action.items():
            spec = env.action_spec()[key]
            assert value.dtype == spec.dtype and value.shape == spec.shape
            assert np.isfinite(value).all() and ((value >= spec.low) & (value <= spec.high)).all()
        np.testing.assert_array_equal(policy._commands[-1]["continuous"], action["continuous_action"])
