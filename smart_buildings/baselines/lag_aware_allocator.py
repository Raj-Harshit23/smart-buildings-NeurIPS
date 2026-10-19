"""Delay-aware online thermal estimation around the first physical allocator.

See MATHEMATICAL_FRAMEWORK.md's verified sensor timing and ALGORITHM_RESEARCH.md
Sections 5 and 8. The estimator uses only public observations and past commands.
This is an experimental observer/controller, not a solved economic MPC policy.
"""

from __future__ import annotations

from collections import deque

import numpy as np

from smart_buildings.baselines.thermal_allocator import ThermalAllocatorPolicy


class LagAwareAllocatorPolicy(ThermalAllocatorPolicy):
    """Identify a positive thermal core with correctly delayed action labels."""

    FORGETTING = .997
    MASS_RETAINED = float(np.exp(-300 / (4 * 3600)))
    WARMUP_STEPS = 96
    RAMP_STEPS = 192
    MAX_CORRECTION_K = 1.5

    def __init__(self, building_meta_data, action_spec):
        super().__init__(building_meta_data, action_spec)
        self._ids = [z[0] for z in self._zone_order]
        self._indices = np.asarray([z[1] for z in self._zone_order])
        self._lookup = {z: i for i, z in enumerate(self._ids)}
        self._areas = self._geometry_areas(building_meta_data)
        self._groups = [np.asarray([self._lookup[z[0]] for z in c["zones"]])
                        for c in self.ahu_controllers]
        self._design_flow_indices = [self.field_names.index(z + "_supply_air_flowrate_setpoint")
                                     if z + "_supply_air_flowrate_setpoint" in self.field_names else None
                                     for z in self._ids]
        self._damper_indices = np.asarray([self.zone_action_indices[z]["supply_air_damper_percentage_command"]
                                          for z in self._ids])
        self._valve_indices = np.asarray([self.zone_action_indices[z]["reheat_valve_percentage_command"]
                                         for z in self._ids])
        self._norms = self.metadata["action_normalization_metadata"]
        self._initialize_observer()

    @property
    def name(self):
        return "lag_aware_allocator"

    def reset(self):
        super().reset()
        if hasattr(self, "_ids"):
            self._initialize_observer()

    def _geometry_areas(self, metadata):
        """Relative zone area from the same public four-connected grid encoding."""
        result = np.full(len(self._ids), 100.0)
        grid = metadata.get("zone_map")
        if grid is None or np.asarray(grid).ndim != 2:
            return result
        import cv2

        mask = (np.asarray(grid) == 0).astype(np.uint8)
        _, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=4)
        for i, zone in enumerate(self._ids):
            try:
                number = int(zone.split("_")[-1])
            except ValueError:
                continue
            if 0 < number < len(stats):
                # 0.2 m cells are a public-model prior, not hidden metadata.
                result[i] = max(1.0, float(stats[number, cv2.CC_STAT_AREA]) * .04)
        return result

    def _initialize_observer(self):
        count = len(self._ids)
        # Shared priors informed by a public Phoenix training-trajectory fit;
        # heat response scales with geometry rather than a building/zone lookup.
        heat_gain = np.clip(.8 * 300 / (1.2 * 1006 * 3 * self._areas), 1e-5, .005)
        self._theta = np.column_stack([np.full(count, .05), np.full(count, .8),
                                       heat_gain * 10000, np.full(count, .25)])
        self._covariance = np.tile(np.diag([.1, .5, 100.0, .2]), (count, 1, 1))
        self._commands = deque(maxlen=2)
        self._raw_previous = None
        self._mass_proxy = None
        self._residual_mae = np.full(count, .5)
        self._updates = 0

    def _to_native(self, command, field, indices):
        n = self._norms[field]
        return n["min_native_value"] + (
            np.asarray(command)[indices] - n["min_normalized_value"]
        ) / (n["max_normalized_value"] - n["min_normalized_value"]) * (
            n["max_native_value"] - n["min_native_value"]
        )

    def _record(self, obs, action):
        return {
            "continuous": action["continuous_action"].astype(float).copy(),
            "binary": action["discrete_action"].copy(),
            "outside": np.asarray([float(obs[c["oat_idx"]]) if c.get("oat_idx") is not None else 295.0
                                    for c in self.ahu_controllers]),
            "water": float(obs[self._water_sensor]) if self._water_sensor is not None else 330.0,
            "design": np.asarray([float(obs[i]) if i is not None else 2.0
                                   for i in self._design_flow_indices]),
        }

    def _equipment(self, start_temperature, record):
        """Reconstruct device terms from legal state estimates and past actions."""
        command, binary = record["continuous"], record["binary"]
        dampers = np.maximum(.1, self._to_native(command, "supply_air_damper_percentage_command", self._damper_indices))
        valves = np.clip(self._to_native(command, "reheat_valve_percentage_command", self._valve_indices), 0, 1)
        count = len(start_temperature)
        flow, inlet, outside = np.zeros(count), np.zeros(count), np.zeros(count)
        for j, (ctrl, indices) in enumerate(zip(self.ahu_controllers, self._groups)):
            if not len(indices):
                continue
            ahu = ctrl["ahu_id"]
            pressure = float(self._to_native(command, ahu + "_supply_air_static_pressure_setpoint", ctrl["pressure_idx"]))
            pressure = max(pressure, 200.0) if binary[ctrl["run_discrete_idx"]] else 0.0
            recirc = float(self._to_native(command, ahu + "_recirculation_percentage_command", ctrl["recirc_idx"]))
            recirc = float(np.clip(recirc, 0, .7))
            target = float(self._to_native(command, ahu + "_supply_air_temperature_setpoint", ctrl["temp_idx"]))
            return_temp = float(np.average(start_temperature[indices], weights=self._areas[indices]))
            mixed = recirc * return_temp + (1 - recirc) * record["outside"][j]
            supplied = min(mixed, target) if binary[ctrl["run_discrete_idx"]] else mixed
            delivered = np.maximum(dampers[indices] * np.maximum(record["design"][indices], .01)
                                   * np.sqrt(pressure / 2000), 1e-5)
            delivered *= min(1.0, 20.0 / max(float(delivered.sum()), 1e-12))
            flow[indices], inlet[indices], outside[indices] = delivered, supplied, record["outside"][j]
        pressure = float(self._to_native(command, "differential_pressure_setpoint", self.water_pressure_idx))
        pressure = max(0.0, pressure) if binary[self.water_discrete_idx] else 0.0
        water_flow = valves * .108 / 3600 * np.sqrt(pressure)
        air = self.AIR_CAPACITY * flow
        gap = record["water"] - inlet
        effectiveness = np.minimum(1.0, self.WATER_CAPACITY * water_flow / np.maximum(air, 1e-12))
        effectiveness = np.where(gap > 0, np.minimum(effectiveness,
            np.maximum(0.0, self.DISCHARGE_CAP - inlet) / np.maximum(gap, 1e-12)), effectiveness)
        reheat = air * effectiveness * gap
        return air, inlet, reheat, outside

    def _predict(self, start, equipment, mass):
        air, inlet, reheat, outside = equipment
        a, b, k, c = self._theta.T
        k = k / 10000
        return (start + a * outside + b * mass + k * (air * inlet + reheat) + c) / (
            1 + a + b + k * air)

    def _learn_delayed_transition(self, start, end, record):
        equipment = self._equipment(start, record)
        air, inlet, reheat, outside = equipment
        prediction = self._predict(start, equipment, self._mass_proxy)
        self._residual_mae = .97 * self._residual_mae + .03 * np.abs(prediction - end)
        heat = air * (inlet - end) + reheat
        features = np.column_stack([outside - end, self._mass_proxy - end, heat / 10000, np.ones(len(start))])
        target = end - start
        px = np.einsum("vij,vj->vi", self._covariance, features)
        denominator = self.FORGETTING + np.sum(features * px, axis=1)
        gain = px / np.maximum(denominator[:, None], 1e-9)
        residual = target - np.sum(features * self._theta, axis=1)
        self._theta += gain * np.clip(residual, -10.0, 10.0)[:, None]
        self._theta = np.clip(self._theta, [0.0, 0.0, .001, -2.0], [5.0, 5.0, 100.0, 2.0])
        self._covariance = (self._covariance - np.einsum("vi,vj->vij", gain, px)) / self.FORGETTING
        self._covariance = .5 * (self._covariance + self._covariance.transpose(0, 2, 1))
        self._updates += 1
        if self._updates % 16 == 0:
            # Unexcited directions otherwise grow as forgetting**(-steps).
            eigenvalues, vectors = np.linalg.eigh(self._covariance)
            self._covariance = (vectors * np.clip(eigenvalues, 1e-8, 1e4)[:, None, :]) @ vectors.transpose(0, 2, 1)

    def get_action(self, obs):
        raw = np.asarray([self._denormalize_zone_temp(obs[i]) for i in self._indices])
        if self._mass_proxy is None:
            self._mass_proxy = raw.copy()
        if len(self._commands) == 2 and self._raw_previous is not None:
            # obs[t] reports physical T[t-1], so this label belongs to action[t-2].
            self._learn_delayed_transition(self._raw_previous, raw, self._commands[0])
            self._mass_proxy = self.MASS_RETAINED * self._mass_proxy + (1 - self.MASS_RETAINED) * self._raw_previous
        correction = np.zeros_like(raw)
        trusted = np.zeros_like(raw)
        if self._commands:
            prediction = self._predict(raw, self._equipment(raw, self._commands[-1]), self._mass_proxy)
            warmup = float(np.clip((self._updates - self.WARMUP_STEPS) / self.RAMP_STEPS, 0, 1))
            trusted = warmup * np.clip((.5 - self._residual_mae) / .3, 0, 1)
            correction = trusted * np.clip(prediction - raw, -self.MAX_CORRECTION_K, self.MAX_CORRECTION_K)
        estimated_obs = np.asarray(obs).copy()
        estimated_obs[self._indices] = (raw + correction - self.zone_air_temperature_sensor_mean) / self.zone_air_temperature_sensor_std
        action = super().get_action(estimated_obs)
        self._commands.append(self._record(obs, action))
        self._raw_previous = raw.copy()
        self.last_diagnostics.update(observer_updates=self._updates,
                                     observer_mae_k=float(np.mean(self._residual_mae)),
                                     observer_weight=float(np.mean(trusted)),
                                     mean_correction_k=float(np.mean(np.abs(correction))))
        return action
