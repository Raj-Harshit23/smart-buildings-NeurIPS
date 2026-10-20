"""Verify that this ablation changes control dispatch, not observer timing."""

from pathlib import Path

import numpy as np
import pytest

from smart_buildings.baselines.lag_aware_allocator import LagAwareAllocatorPolicy
from smart_buildings.baselines.lag_aware_pid import LagAwarePIDPolicy
from smart_buildings.baselines.pid_plus import PIDWaterResetPolicy
from smart_buildings.baselines.thermal_allocator import ThermalAllocatorPolicy
from smart_buildings.evaluate import create_building_meta_data, load_hybrid_environment


@pytest.fixture(scope="module")
def data():
    scenario = Path("smart_buildings/training_scenarios/building_train_8/phoenix_4")
    env = load_hybrid_environment(scenario / "config.gin")
    metadata = create_building_meta_data(env, scenario)
    observation, _ = env.reset()
    return metadata, env.action_spec(), observation


def test_same_actuator_law_preserves_frozen_observer_exactly(data):
    metadata, spec, original = data
    reference = LagAwareAllocatorPolicy(metadata, spec)
    ablation = LagAwarePIDPolicy(metadata, spec)
    ablation._control_from_estimate = lambda obs: ThermalAllocatorPolicy.get_action(ablation, obs)
    # Exercise warmup, adaptation, clipping and covariance projections under a
    # changing deterministic signal. Identical selected actions align histories.
    for step in range(420):
        obs = original.copy()
        obs[metadata["zone_temp_indices"]] = -.4 + .7 * np.sin(step / 17)
        a, b = reference.get_action(obs), ablation.get_action(obs)
        for key in a:
            np.testing.assert_array_equal(a[key], b[key])
        np.testing.assert_array_equal(reference._theta, ablation._theta)
        np.testing.assert_array_equal(reference._covariance, ablation._covariance)


def test_pid_receives_estimate_and_observer_records_selected_action(data):
    metadata, spec, obs = data
    policy = LagAwarePIDPolicy(metadata, spec)
    pid = PIDWaterResetPolicy(metadata, spec)
    pid.USE_SENSOR_BAND = True
    # During initial observer warmup the state estimate equals the raw sensor.
    a, b = policy.get_action(obs), pid.get_action(obs)
    for key in a:
        np.testing.assert_array_equal(a[key], b[key])
    np.testing.assert_array_equal(policy._commands[-1]["continuous"], a["continuous_action"])
    # A nonzero correction must reach PID rather than only the diagnostic log.
    captured = []
    control = policy._control_from_estimate
    policy._control_from_estimate = lambda estimate: (captured.append(estimate.copy()) or control(estimate))
    policy._updates = 400
    policy._residual_mae[:] = 0
    policy._predict = lambda start, equipment, mass: start + 1
    policy.get_action(obs)
    expected = obs[policy._indices] + 1 / policy.zone_air_temperature_sensor_std
    np.testing.assert_allclose(captured[0][policy._indices], expected, atol=1e-6)
