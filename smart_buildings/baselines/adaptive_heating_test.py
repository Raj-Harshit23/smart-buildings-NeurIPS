"""Check inverse thermal control, physical bounds and selected-action history."""

from pathlib import Path

import numpy as np
import pytest

from smart_buildings.baselines.adaptive_heating import AdaptiveHeatingPolicy
from smart_buildings.evaluate import create_building_meta_data, load_hybrid_environment


@pytest.fixture(scope="module", params=["building_train_2/fairbanks_2", "building_train_8/phoenix_4"])
def data(request):
    scenario = Path("smart_buildings/training_scenarios") / request.param
    env = load_hybrid_environment(scenario / "config.gin")
    metadata = create_building_meta_data(env, scenario)
    observation, _ = env.reset()
    return metadata, env.action_spec(), observation


def test_inverse_request_reaches_target_in_forward_model(data):
    metadata, spec, obs = data
    policy = AdaptiveHeatingPolicy(metadata, spec)
    count = len(policy._ids)
    indices = np.arange(count)
    temperature = np.full(count, 290.)
    heating = np.full(count, 4.)
    policy._mass_proxy = np.full(count, 289.)
    policy._theta[:] = [.02, .4, 8., .15]
    policy._updates = 1000
    policy._residual_mae[:] = 0
    heat, target, weight = policy._heat_request(indices, temperature, heating, 270., np.zeros(count))
    air = np.full(count, 300.)
    inlet = np.full(count, 280.)
    reheat = heat - air * (inlet - target)
    predicted = policy._predict(temperature, (air, inlet, reheat, np.full(count, 270.)), policy._mass_proxy)
    np.testing.assert_allclose(predicted, target, atol=1e-10)
    assert (weight == 1).all() and (heat > 0).all()
    policy._residual_mae[:] = 1
    guarded, _, weight = policy._heat_request(indices, temperature, heating, 270., np.zeros(count))
    np.testing.assert_allclose(guarded, heating / policy.RECOVERY_STEPS / policy.HEAT_GAIN)
    assert (weight == 0).all()


def test_mature_actions_adapt_to_building_dimensions_and_record_applied_command(data):
    metadata, spec, original = data
    for temperature, outside in [(288., 260.), (300., 310.), (294.5, 290.)]:
        policy = AdaptiveHeatingPolicy(metadata, spec)
        observation = original.copy()
        observation[metadata["zone_temp_indices"]] = ((temperature - policy.zone_air_temperature_sensor_mean)
                                                     / policy.zone_air_temperature_sensor_std)
        for index, name in enumerate(metadata["observation_field_names"]):
            if name.endswith("outside_air_temperature_sensor"):
                observation[index] = outside
            elif name.endswith("supply_water_temperature_sensor"):
                observation[index] = 340.
        policy._updates = 1000
        policy._residual_mae[:] = 0
        action = policy.get_action(observation)
        for key, value in action.items():
            assert value.dtype == spec[key].dtype and value.shape == spec[key].shape
            assert np.isfinite(value).all()
            assert ((value >= spec[key].low) & (value <= spec[key].high)).all()
        np.testing.assert_array_equal(policy._commands[-1]["continuous"], action["continuous_action"])
        np.testing.assert_array_equal(policy._commands[-1]["binary"], action["discrete_action"])
