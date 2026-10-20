"""Guarded thermal inverse for pure cooling; see ADAPTIVE_COOLING.md."""

from __future__ import annotations

import numpy as np

from smart_buildings.baselines.adaptive_heating import AdaptiveHeatingPolicy


class AdaptiveCoolingPolicy(AdaptiveHeatingPolicy):
    @property
    def name(self):
        return "adaptive_cooling"

    def _cooling_flow(self, indices, temperatures, lower, upper, inlet, outside, current):
        a, b, scaled_k, c = self._theta[indices].T
        margin = np.clip(2 * self._residual_mae[indices], .1, .5)
        target = np.maximum(lower + .1, upper - margin)
        gap = target - inlet
        numerator = (temperatures + a * outside + b * self._mass_proxy[indices] + c
                     - (1 + a + b) * target)
        model = np.maximum(0., numerator / (np.maximum(scaled_k / 10000, 1e-9)
                                            * self.AIR_CAPACITY * np.maximum(gap, .25)))
        warmup = float(np.clip((self._updates - self.WARMUP_STEPS) / self.RAMP_STEPS, 0, 1))
        weight = warmup * np.clip((.5 - self._residual_mae[indices]) / .3, 0, 1)
        weight = np.where(gap > .25, weight, 0.)
        return (1 - weight) * current + weight * model, target, weight

    def _control_from_estimate(self, obs):
        action = super()._control_from_estimate(obs)
        continuous, binary = action["continuous_action"], action["discrete_action"]
        temperature = np.asarray([self._denormalize_zone_temp(obs[i]) for i in self._indices])
        record = self._record(obs, action)
        air, inlet, _, outside = self._equipment(temperature, record)
        valves = self._to_native(continuous, "reheat_valve_percentage_command", self._valve_indices)
        hour, day = self._get_hour_and_day(obs)
        work = day < 5 and self.morning_start_hour <= hour < self.evening_start_hour
        low, high = self.comfort_temp_window if work else self.eco_temp_window
        for ctrl, indices in zip(self.ahu_controllers, self._groups):
            if (not len(indices) or not binary[ctrl["run_discrete_idx"]]
                    or continuous[ctrl["temp_idx"]] >= 0 or ctrl["pressure_idx"] is None):
                continue
            bands = [self._zone_band(obs, zone, low, high) for zone, _, _ in ctrl["zones"]]
            lower, upper = np.asarray(bands).T
            values = temperature[indices]
            if (np.any(values < lower) or np.any(valves[indices] > 1e-6)
                    or not np.any(values > upper)):
                continue
            flow, _, weight = self._cooling_flow(indices, values, lower, upper, inlet[indices],
                                                  outside[indices], air[indices] / self.AIR_CAPACITY)
            if not np.any(weight > 0):
                continue
            designs = np.maximum(record["design"][indices], .01)
            floor = self.MIN_DAMPER * designs * np.sqrt(self.MIN_PRESSURE / self.DESIGN_PRESSURE)
            flow = np.clip(flow, floor, designs)
            pressure = max(self.MIN_PRESSURE, float(np.max(self.DESIGN_PRESSURE * (flow / designs)**2)))
            dampers = np.clip(flow / (designs * np.sqrt(pressure / self.DESIGN_PRESSURE)), self.MIN_DAMPER, 1)
            ahu = ctrl["ahu_id"]
            continuous[ctrl["pressure_idx"]] = self._native_command(ahu + "_supply_air_static_pressure_setpoint", pressure)
            for (zone, _, mapping), damper in zip(ctrl["zones"], dampers):
                index = mapping.get("supply_air_damper_percentage_command")
                if index is not None:
                    continuous[index] = self._native_command(zone + "_supply_air_damper_percentage_command", damper)
        return action
