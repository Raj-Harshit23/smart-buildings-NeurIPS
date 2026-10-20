"""Controlled ablation: delayed-state observer with the original PID actuator law.

See MATHEMATICAL_FRAMEWORK.md Sections 1, 3, 6–9 and LAG_AWARE_PID.md.
Observer equations and priors match the frozen second experiment. Only the
control law receiving the estimated temperature changes; no future labels enter.
"""

from __future__ import annotations

import numpy as np

from smart_buildings.baselines.lag_aware_allocator import LagAwareAllocatorPolicy
from smart_buildings.baselines.pid_plus import PIDWaterResetPolicy


class LagAwarePIDPolicy(LagAwareAllocatorPolicy):
    """Separate the benefit of causal state estimation from heat allocation."""

    @property
    def name(self):
        return "lag_aware_pid"

    def _control_from_estimate(self, observation):
        return PIDWaterResetPolicy.get_action(self, observation)

    def get_action(self, obs):
        # Keep the frozen V2 observer unchanged, including its delayed-action
        # indexing and confidence guards. This explicit ablation dispatches to
        # PID instead of ThermalAllocatorPolicy after constructing the estimate.
        raw = np.asarray([self._denormalize_zone_temp(obs[i]) for i in self._indices])
        if self._mass_proxy is None:
            self._mass_proxy = raw.copy()
        if len(self._commands) == 2 and self._raw_previous is not None:
            self._learn_delayed_transition(self._raw_previous, raw, self._commands[0])
            self._mass_proxy = (self.MASS_RETAINED * self._mass_proxy
                                + (1 - self.MASS_RETAINED) * self._raw_previous)
        correction = np.zeros_like(raw)
        trusted = np.zeros_like(raw)
        if self._commands:
            prediction = self._predict(raw, self._equipment(raw, self._commands[-1]), self._mass_proxy)
            warmup = float(np.clip((self._updates - self.WARMUP_STEPS) / self.RAMP_STEPS, 0, 1))
            trusted = warmup * np.clip((.5 - self._residual_mae) / .3, 0, 1)
            correction = trusted * np.clip(prediction - raw, -self.MAX_CORRECTION_K, self.MAX_CORRECTION_K)
        estimated_obs = np.asarray(obs).copy()
        estimated_obs[self._indices] = ((raw + correction - self.zone_air_temperature_sensor_mean)
                                        / self.zone_air_temperature_sensor_std)
        action = self._control_from_estimate(estimated_obs)
        self._commands.append(self._record(obs, action))
        self._raw_previous = raw.copy()
        self.last_diagnostics = {
            "observer_updates": self._updates,
            "observer_mae_k": float(np.mean(self._residual_mae)),
            "observer_weight": float(np.mean(trusted)),
            "mean_correction_k": float(np.mean(np.abs(correction))),
        }
        return action
