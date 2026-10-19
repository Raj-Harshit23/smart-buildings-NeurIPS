"""Heating-aware physical allocation, the first increment toward adaptive MPC.

References: MATHEMATICAL_FRAMEWORK.md Sections 1, 3, 6, 7; ALGORITHM_RESEARCH.md
Sections 6.1, 6.3, 6.4. Uses only participant metadata and observation history.
The shipped PID cooling behavior remains a control for this first experiment.
Equipment constants below are public-model priors, not hidden-building lookups.
"""

from __future__ import annotations

import numpy as np

from smart_buildings.baselines.pid_plus import PIDWaterResetPolicy


class ThermalAllocatorPolicy(PIDWaterResetPolicy):
    """Allocate heat through airflow and water, avoiding cold-air amplification.

    The local predictive target asks for a fraction of the temperature deficit
    per step, compensates an observed cooling trend, and respects the discharge
    temperature cap. It is a feedback heuristic, not a solved global MPC problem.
    """

    USE_SENSOR_BAND = True
    AIR_CAPACITY = 1.2 * 1006.0
    WATER_CAPACITY = 1000.0 * 4180.0
    VALVE_KV = 0.108  # m3/hour per sqrt(bar) at full opening
    MIN_PRESSURE = 200.0
    DESIGN_PRESSURE = 2000.0
    MIN_DAMPER = 0.1
    DISCHARGE_CAP = 308.15
    HEAT_GAIN = 2e-5  # prior K per W over a five-minute step
    RECOVERY_STEPS = 6.0

    def __init__(self, building_meta_data, action_spec):
        super().__init__(building_meta_data, action_spec)
        names = list(self.field_names)
        self._water_sensor = next(
            (i for i, n in enumerate(names) if n.endswith("supply_water_temperature_sensor")), None
        )
        self._zone_order = [zone for ctrl in self.ahu_controllers for zone in ctrl["zones"]]
        self._previous_temps = None
        self._temperature_trend = np.zeros(len(self._zone_order))
        self.last_diagnostics = {}

    @property
    def name(self):
        return "thermal_allocator"

    def reset(self):
        super().reset()
        self._water_integral = 0.0
        self._previous_temps = None
        if hasattr(self, "_zone_order"):
            self._temperature_trend = np.zeros(len(self._zone_order))
        self.last_diagnostics = {}

    def _native_command(self, field, value):
        """Apply the exact action normalizer supplied in public metadata."""
        normalizer = self.metadata["action_normalization_metadata"].get(field)
        if normalizer is None and field.startswith("VAV_"):
            normalizer = self.metadata["action_normalization_metadata"].get(field.split("_", 2)[2])
        if normalizer is None:
            raise ValueError(f"Missing public action normalizer: {field}")
        low, high = normalizer["min_native_value"], normalizer["max_native_value"]
        if high <= low:
            raise ValueError(f"Degenerate public action range: {field}")
        normalized = (float(value) - low) / (high - low)
        result = normalizer["min_normalized_value"] + normalized * (
            normalizer["max_normalized_value"] - normalizer["min_normalized_value"]
        )
        return np.float32(np.clip(result, -1.0, 1.0))

    def _flow_design(self, obs, vav_id):
        field = f"{vav_id}_supply_air_flowrate_setpoint"
        if field not in self.field_names:
            return 2.0
        # This measurement is the raw design flow in the public implementation.
        value = float(obs[self.field_names.index(field)])
        return value if np.isfinite(value) and value > 0 else 2.0

    def get_action(self, obs):
        action = super().get_action(obs)
        cont, binary = action["continuous_action"], action["discrete_action"]
        temperatures = np.asarray([
            self._denormalize_zone_temp(obs[index]) for _, index, _ in self._zone_order
        ])
        if not np.isfinite(temperatures).all():
            return action
        if self._previous_temps is not None:
            difference = np.clip(temperatures - self._previous_temps, -1.0, 1.0)
            self._temperature_trend = .8 * self._temperature_trend + .2 * difference
        self._previous_temps = temperatures.copy()
        position = {vav: i for i, (vav, _, _) in enumerate(self._zone_order)}
        hour, day = self._get_hour_and_day(obs)
        work = day < 5 and self.morning_start_hour <= hour < self.evening_start_hour
        low, high = self.comfort_temp_window if work else self.eco_temp_window
        water_temp = float(obs[self._water_sensor]) if self._water_sensor is not None else np.nan
        if not np.isfinite(water_temp) or not 250 <= water_temp <= 400:
            return action
        changed = 0
        desired_flows = []
        for ctrl in self.ahu_controllers:
            zones = ctrl["zones"]
            if not zones or ctrl.get("oat_idx") is None or ctrl["pressure_idx"] is None:
                continue
            values = np.asarray([temperatures[position[z[0]]] for z in zones])
            bands = [self._zone_band(obs, vav, low, high) for vav, _, _ in zones]
            heating = np.asarray([max(0.0, lo - t) for t, (lo, _) in zip(values, bands)])
            cooling = np.asarray([max(0.0, t - hi) for t, (_, hi) in zip(values, bands)])
            outside = float(obs[ctrl["oat_idx"]])  # raw kelvin for AHU child fields
            mixed = .7 * float(np.mean(values)) + .3 * outside
            # Keep the baseline as the first experiment's cooling controller.
            if (not np.isfinite(mixed) or mixed >= float(np.mean(values))
                    or heating.max(initial=0.0) < .05 or heating.sum() <= cooling.sum()):
                continue
            inlet = min(mixed, 305.0)
            if water_temp <= inlet + 1.0:
                continue
            designs = np.asarray([self._flow_design(obs, vav) for vav, _, _ in zones])
            trend = self._temperature_trend[[position[z[0]] for z in zones]]
            # Compensate only the adverse trend, avoiding an unobserved-state oracle.
            requested_heat = np.maximum(0.0, heating / self.RECOVERY_STEPS
                                         + np.maximum(0.0, -trend)) / self.HEAT_GAIN
            headroom = np.maximum(self.DISCHARGE_CAP - values, 1.0)
            flow = requested_heat / (self.AIR_CAPACITY * headroom)
            floor_flow = self.MIN_DAMPER * designs * np.sqrt(self.MIN_PRESSURE / self.DESIGN_PRESSURE)
            flow = np.clip(flow, floor_flow, designs)
            # Water-limited coils lose net heat if airflow is increased too far.
            max_water_flow = self.VALVE_KV / 3600.0 * np.sqrt(20.0)
            max_reheat = self.WATER_CAPACITY * max_water_flow * (water_temp - inlet)
            max_flow_at_cap = max_reheat / max(self.AIR_CAPACITY * (self.DISCHARGE_CAP - inlet), 1.0)
            flow = np.maximum(floor_flow, np.minimum(flow, max_flow_at_cap))
            pressure = max(self.MIN_PRESSURE, float(np.max(self.DESIGN_PRESSURE * (flow / designs) ** 2)))
            # Raising pressure can make a different zone's ventilation floor bind.
            flow = np.maximum(flow, self.MIN_DAMPER * designs * np.sqrt(pressure / self.DESIGN_PRESSURE))
            dampers = np.clip(flow / (designs * np.sqrt(pressure / self.DESIGN_PRESSURE)), self.MIN_DAMPER, 1.0)
            # A 20 m3/s AHU cap is a public prior; use scaled delivered flows.
            flow *= min(1.0, 20.0 / max(float(flow.sum()), 1e-12))
            required_reheat = np.maximum(0.0, requested_heat - self.AIR_CAPACITY * flow * (inlet - values))
            # Avoid heating comfortable zones beyond their lower-band neighborhood.
            near_band = heating <= 0
            neutral_heat = np.maximum(0.0, self.AIR_CAPACITY * flow * (values - inlet))
            required_reheat[near_band] = neutral_heat[near_band]
            valve = np.clip(required_reheat / max(max_reheat, 1.0), 0.0, 1.0)
            ahu = ctrl["ahu_id"]
            cont[ctrl["temp_idx"]] = self._native_command(f"{ahu}_supply_air_temperature_setpoint", 305)
            cont[ctrl["pressure_idx"]] = self._native_command(f"{ahu}_supply_air_static_pressure_setpoint", pressure)
            if ctrl.get("recirc_idx") is not None:
                cont[ctrl["recirc_idx"]] = self._native_command(f"{ahu}_recirculation_percentage_command", .7)
            binary[ctrl["run_discrete_idx"]] = 1
            for (vav, _, indices), damper, opening in zip(zones, dampers, valve):
                if "supply_air_damper_percentage_command" in indices:
                    cont[indices["supply_air_damper_percentage_command"]] = self._native_command(
                        f"{vav}_supply_air_damper_percentage_command", damper)
                if "reheat_valve_percentage_command" in indices:
                    cont[indices["reheat_valve_percentage_command"]] = self._native_command(
                        f"{vav}_reheat_valve_percentage_command", opening)
            changed += 1
            desired_flows.extend(flow.tolist())
        if changed:
            if self.water_temp_idx is not None:
                cont[self.water_temp_idx] = self._native_command("supply_water_temperature_setpoint", 350)
            if self.water_pressure_idx is not None:
                cont[self.water_pressure_idx] = self._native_command("differential_pressure_setpoint", 20)
            binary[self.water_discrete_idx] = 1
        self.last_diagnostics = {"heating_ahus": changed, "allocated_flow": sum(desired_flows)}
        return action
