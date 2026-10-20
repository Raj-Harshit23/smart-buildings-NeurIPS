"""Tests for delayed labels, reset, and unexcited-estimator numerical behavior."""

from pathlib import Path

import numpy as np
import pytest

from smart_buildings.baselines.lag_aware_allocator import LagAwareAllocatorPolicy
from smart_buildings.baselines.lag_aware_pid import LagAwarePIDPolicy
from smart_buildings.baselines.adaptive_heating import AdaptiveHeatingPolicy
from smart_buildings.evaluate import create_building_meta_data, load_hybrid_environment


@pytest.fixture(scope="module")
def data():
    scenario = Path("smart_buildings/training_scenarios/building_train_8/phoenix_4")
    env = load_hybrid_environment(scenario / "config.gin")
    metadata = create_building_meta_data(env, scenario)
    obs, _ = env.reset()
    return metadata, env.action_spec(), obs


@pytest.mark.parametrize("policy_type", [LagAwareAllocatorPolicy, LagAwarePIDPolicy, AdaptiveHeatingPolicy])
def test_identification_uses_action_two_calls_before_observed_label(data, policy_type):
    metadata, spec, original = data
    policy = policy_type(metadata, spec)
    obs = original.copy()
    for i, name in enumerate(metadata["observation_field_names"]):
        if name.endswith("outside_air_temperature_sensor"):
            obs[i] = 260.0
    first = policy.get_action(obs)
    captured = []
    learn = policy._learn_delayed_transition
    def record(start, end, command):
        captured.append(command["binary"].copy())
        learn(start, end, command)
    policy._learn_delayed_transition = record
    obs[metadata["zone_temp_indices"]] = -2.0
    second = policy.get_action(obs)
    assert not captured
    assert not np.array_equal(first["discrete_action"], second["discrete_action"])
    policy.get_action(obs)
    np.testing.assert_array_equal(captured[0], first["discrete_action"])
    policy.get_action(obs)
    np.testing.assert_array_equal(captured[1], second["discrete_action"])


@pytest.mark.parametrize("policy_type", [LagAwareAllocatorPolicy, LagAwarePIDPolicy, AdaptiveHeatingPolicy])
def test_unexcited_estimation_stays_finite_and_covariance_bounded(data, policy_type):
    metadata, spec, obs = data
    policy = policy_type(metadata, spec)
    for _ in range(2048):
        result = policy.get_action(obs)
        for key, value in result.items():
            assert np.isfinite(value).all()
            assert ((value >= spec[key].low) & (value <= spec[key].high)).all()
    eigenvalues = np.linalg.eigvalsh(policy._covariance)
    assert eigenvalues.min() >= -1e-8
    # Between spectral projections, forgetting can expand uncertainty at most
    # sixteen times; observed directions can only reduce that upper bound.
    assert eigenvalues.max() <= 1e4 / policy.FORGETTING**16 + 1e-4
    assert np.isfinite(policy._theta).all()


@pytest.mark.parametrize("policy_type", [LagAwareAllocatorPolicy, LagAwarePIDPolicy, AdaptiveHeatingPolicy])
def test_reset_clears_delayed_command_queue(data, policy_type):
    metadata, spec, obs = data
    policy = policy_type(metadata, spec)
    first = policy.get_action(obs)
    policy.get_action(obs)
    policy.get_action(obs)
    assert policy._updates == 1
    policy.reset()
    assert not policy._commands and policy._updates == 0
    repeated = policy.get_action(obs)
    for key in first:
        np.testing.assert_array_equal(first[key], repeated[key])
