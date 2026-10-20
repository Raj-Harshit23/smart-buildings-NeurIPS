"""Physical inverse, authority guards, and isolated cooling-control tests."""

from pathlib import Path

import numpy as np
import pytest

from smart_buildings.baselines.adaptive_cooling import AdaptiveCoolingPolicy
from smart_buildings.baselines.adaptive_heating import AdaptiveHeatingPolicy
from smart_buildings.evaluate import create_building_meta_data, load_hybrid_environment


@pytest.fixture(scope="module", params=["building_train_8/phoenix_4", "building_train_2/fairbanks_2"])
def data(request):
    scenario = Path("smart_buildings/training_scenarios") / request.param
    env = load_hybrid_environment(scenario / "config.gin")
    metadata = create_building_meta_data(env, scenario)
    observation, _ = env.reset()
    return metadata, env.action_spec(), observation


def test_inverse_flow_reaches_buffered_target_in_forward_model(data):
    metadata, spec, obs = data
    policy = AdaptiveCoolingPolicy(metadata, spec)
    count = len(policy._ids)
    indices = np.arange(count)
    policy._theta[:] = [.05, .8, 8., .25]
    policy._mass_proxy = np.full(count, 300.)
    policy._updates = 1000
    policy._residual_mae[:] = .1
    temperature, inlet, outside = [np.full(count, x) for x in [300., 285., 310.]]
    flow, target, weight = policy._cooling_flow(indices, temperature, np.full(count, 294.),
                                               np.full(count, 297.), inlet, outside, np.ones(count))
    predicted = policy._predict(temperature, (policy.AIR_CAPACITY * flow, inlet,
                                             np.zeros(count), outside), policy._mass_proxy)
    np.testing.assert_allclose(predicted, target, atol=1e-10)
    assert (weight == 1).all() and (flow > 0).all()


def test_missing_cooling_authority_preserves_existing_flow(data):
    metadata, spec, obs = data
    policy = AdaptiveCoolingPolicy(metadata, spec)
    count = len(policy._ids)
    policy._mass_proxy = np.full(count, 300.)
    policy._updates = 1000
    policy._residual_mae[:] = 0
    current = np.linspace(.1, 1, count)
    flow, _, weight = policy._cooling_flow(np.arange(count), np.full(count, 300.),
        np.full(count, 294.), np.full(count, 297.), np.full(count, 298.), np.full(count, 310.), current)
    np.testing.assert_array_equal(flow, current)
    assert (weight == 0).all()


@pytest.mark.parametrize("temperature,outside", [(288., 260.), (300., 310.)])
def test_actual_commands_and_history_on_two_buildings(data, temperature, outside):
    metadata, spec, original = data
    baseline, candidate = [kind(metadata, spec) for kind in [AdaptiveHeatingPolicy, AdaptiveCoolingPolicy]]
    observation = original.copy()
    for policy in [baseline, candidate]:
        policy._updates = 1000
        policy._residual_mae[:] = 0
    observation[metadata["zone_temp_indices"]] = ((temperature - candidate.zone_air_temperature_sensor_mean)
                                                 / candidate.zone_air_temperature_sensor_std)
    for index, name in enumerate(metadata["observation_field_names"]):
        if name.endswith("outside_air_temperature_sensor"): observation[index] = outside
        elif name.endswith("supply_water_temperature_sensor"): observation[index] = 340.
    a, b = baseline.get_action(observation), candidate.get_action(observation)
    for key, value in b.items():
        assert value.dtype == spec[key].dtype and value.shape == spec[key].shape
        assert np.isfinite(value).all() and ((value >= spec[key].low) & (value <= spec[key].high)).all()
    np.testing.assert_array_equal(candidate._commands[-1]["continuous"], b["continuous_action"])
    if temperature < 294:
        for key in a: np.testing.assert_array_equal(a[key], b[key])
    else:
        indices = metadata["pressure_indices"][:-1]
        assert np.any(b["continuous_action"][indices] < a["continuous_action"][indices])
