"""Adaptive inverse heat demand; see ADAPTIVE_HEATING.md for assumptions."""

from __future__ import annotations

import numpy as np

from smart_buildings.baselines.lag_aware_pid import LagAwarePIDPolicy
from smart_buildings.baselines.pid_plus import PIDWaterResetPolicy


class AdaptiveHeatingPolicy(LagAwarePIDPolicy):
    """Retain the causal observer and replace the fixed-gain heating request."""

    @property
    def name(self):
        return "adaptive_heating"

    def _heat_request(self, indices, temperatures, heating, outside, trend):
        target = temperatures + heating / self.RECOVERY_STEPS
        a, b, scaled_k, c = self._theta[indices].T
        model = np.maximum(0.0, (target - temperatures - a * (outside - target)
                                - b * (self._mass_proxy[indices] - target) - c)
                           / np.maximum(scaled_k / 10000, 1e-9))
        fixed = (heating / self.RECOVERY_STEPS + np.maximum(0.0, -trend)) / self.HEAT_GAIN
        warmup = float(np.clip((self._updates - self.WARMUP_STEPS) / self.RAMP_STEPS, 0, 1))
        weight = warmup * np.clip((.5 - self._residual_mae[indices]) / .3, 0, 1)
        return (1 - weight) * fixed + weight * model, target, weight

    def _control_from_estimate(self, obs):
        action = PIDWaterResetPolicy.get_action(self, obs)
        continuous, binary = action["continuous_action"], action["discrete_action"]
        temperatures = np.asarray([self._denormalize_zone_temp(obs[i]) for i in self._indices])
        if not np.isfinite(temperatures).all():
            return action
        if self._previous_temps is not None:
            self._temperature_trend = (.8 * self._temperature_trend
                                       + .2 * np.clip(temperatures - self._previous_temps, -1, 1))
        self._previous_temps = temperatures.copy()
        hour, day = self._get_hour_and_day(obs)
        work = day < 5 and self.morning_start_hour <= hour < self.evening_start_hour
        low, high = self.comfort_temp_window if work else self.eco_temp_window
        water = float(obs[self._water_sensor]) if self._water_sensor is not None else np.nan
        if not np.isfinite(water) or not 250 <= water <= 400:
            return action
        changed = 0
        for ctrl, indices in zip(self.ahu_controllers, self._groups):
            if not len(indices) or ctrl.get("oat_idx") is None or ctrl["pressure_idx"] is None:
                continue
            values = temperatures[indices]
            bands = [self._zone_band(obs, zone, low, high) for zone, _, _ in ctrl["zones"]]
            heating = np.asarray([max(0., lo - t) for t, (lo, _) in zip(values, bands)])
            cooling = np.asarray([max(0., t - hi) for t, (_, hi) in zip(values, bands)])
            outside = float(obs[ctrl["oat_idx"]])
            return_temperature = float(np.average(values, weights=self._areas[indices]))
            mixed = .7 * return_temperature + .3 * outside
            if (not np.isfinite(mixed) or mixed >= return_temperature
                    or heating.max(initial=0) < .05 or heating.sum() <= cooling.sum()):
                continue
            inlet = min(mixed, 305.)
            if water <= inlet + 1:
                continue
            requested, target, _ = self._heat_request(
                indices, values, heating, outside, self._temperature_trend[indices])
            designs = np.asarray([self._flow_design(obs, zone) for zone, _, _ in ctrl["zones"]])
            floor = self.MIN_DAMPER * designs * np.sqrt(self.MIN_PRESSURE / self.DESIGN_PRESSURE)
            flow = np.clip(requested / (self.AIR_CAPACITY * np.maximum(self.DISCHARGE_CAP - target, 1)),
                           floor, designs)
            max_reheat = self.WATER_CAPACITY * self.VALVE_KV / 3600 * np.sqrt(20) * (water - inlet)
            cap_flow = max_reheat / max(self.AIR_CAPACITY * (self.DISCHARGE_CAP - inlet), 1)
            flow = np.maximum(floor, np.minimum(flow, cap_flow))
            pressure = max(self.MIN_PRESSURE, float(np.max(self.DESIGN_PRESSURE * (flow / designs)**2)))
            flow = np.maximum(flow, self.MIN_DAMPER * designs * np.sqrt(pressure / self.DESIGN_PRESSURE))
            dampers = np.clip(flow / (designs * np.sqrt(pressure / self.DESIGN_PRESSURE)), self.MIN_DAMPER, 1)
            flow *= min(1., 20. / max(float(flow.sum()), 1e-12))
            reheat = np.maximum(0., requested - self.AIR_CAPACITY * flow * (inlet - target))
            comfortable = heating <= 0
            reheat[comfortable] = np.maximum(0., self.AIR_CAPACITY * flow[comfortable]
                                             * (values[comfortable] - inlet))
            valves = np.clip(reheat / max(max_reheat, 1), 0, 1)
            ahu = ctrl["ahu_id"]
            continuous[ctrl["temp_idx"]] = self._native_command(ahu + "_supply_air_temperature_setpoint", 305)
            continuous[ctrl["pressure_idx"]] = self._native_command(ahu + "_supply_air_static_pressure_setpoint", pressure)
            if ctrl.get("recirc_idx") is not None:
                continuous[ctrl["recirc_idx"]] = self._native_command(ahu + "_recirculation_percentage_command", .7)
            binary[ctrl["run_discrete_idx"]] = 1
            for (zone, _, mapping), damper, valve in zip(ctrl["zones"], dampers, valves):
                for field, value in [("supply_air_damper_percentage_command", damper),
                                     ("reheat_valve_percentage_command", valve)]:
                    if field in mapping:
                        continuous[mapping[field]] = self._native_command(zone + "_" + field, value)
            changed += 1
        if changed:
            if self.water_temp_idx is not None:
                continuous[self.water_temp_idx] = self._native_command("supply_water_temperature_setpoint", 350)
            if self.water_pressure_idx is not None:
                continuous[self.water_pressure_idx] = self._native_command("differential_pressure_setpoint", 20)
            binary[self.water_discrete_idx] = 1
        return action
