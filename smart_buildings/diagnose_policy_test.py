"""Regression: diagnostics must reproduce an ordinary uninstrumented rollout."""

from pathlib import Path

import numpy as np
import pandas as pd

from smart_buildings.baselines.pid_plus import PIDWaterResetPolicy
from smart_buildings.diagnose_policy import run
from smart_buildings.evaluate import create_building_meta_data, load_hybrid_environment
from smart_buildings.smart_control.simulator.boiler import Boiler


def test_boiler_getter_is_not_a_safe_diagnostic_read():
    boiler = Boiler(330., heating_rate=.5, cooling_rate=.1)
    start = pd.Timestamp("2023-01-01T00:00:00Z")
    boiler.set_action("supply_water_temperature_setpoint", 350., start)
    first = boiler.get_observation("supply_water_temperature_sensor", start + pd.Timedelta(minutes=5))
    second = boiler.get_observation("supply_water_temperature_sensor", start + pd.Timedelta(minutes=5))
    assert first == 332.5 and second == 335.


def test_diagnostic_rollout_matches_uninstrumented_actions_observations_and_rewards(tmp_path):
    scenario = Path("smart_buildings/training_scenarios/building_train_8/phoenix_4")
    env = load_hybrid_environment(scenario / "config.gin")
    env.metrics_path = None
    env._occupancy_normalization_constant = 125.
    metadata = create_building_meta_data(env, scenario)
    policy = PIDWaterResetPolicy(metadata, env.action_spec())
    observation, _ = env.reset()
    observations, actions, rewards = [], [], []
    for _ in range(24):
        observations.append(observation.copy())
        action = policy.get_action(observation)
        actions.append(action["continuous_action"].copy())
        observation, reward, _, _, _ = env.step(action)
        rewards.append(reward)
    summary = run(scenario, "pid_water", 24, tmp_path)
    with np.load(tmp_path / "trajectory.npz") as archive:
        np.testing.assert_array_equal(archive["observations"], np.asarray(observations, dtype=np.float32))
        np.testing.assert_array_equal(archive["continuous_actions"], np.asarray(actions, dtype=np.float32))
        np.testing.assert_array_equal(archive["rewards"], np.asarray(rewards, dtype=np.float32))
    assert summary["evaluation_protocol"] == "non_mutating_v2"
