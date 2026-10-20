"""Permanent-prior prediction-error observer around adaptive heating control.

References: MATHEMATICAL_FRAMEWORK.md Sections 1, 7–9 and
REGULARIZED_OBSERVER_STUDY.md. Only metadata, observations and past actions enter.
"""
from __future__ import annotations

import numpy as np

from smart_buildings.baselines.adaptive_heating import AdaptiveHeatingPolicy


class RegularizedHeatingPolicy(AdaptiveHeatingPolicy):
    """Retain physical prior information in unexcited parameter directions."""

    extended = True
    # Confidence still blends prediction with the measured register. The
    # positive thermal model bounds prediction by its input temperatures plus
    # the bounded offset; a separate 1.5 K correction cap is removed here.
    MAX_CORRECTION_K = float("inf")

    @property
    def name(self):
        return "regularized_heating"

    def _initialize_observer(self):
        super()._initialize_observer()
        self.prior = self._theta.copy()
        self.precision = np.linalg.inv(self._covariance)
        self.information = self.precision.copy()
        self.prior_rhs = np.einsum("vij,vj->vi", self.precision, self.prior)
        self.rhs = self.prior_rhs.copy()

    def features_and_prediction(self, start, end, record):
        equipment = self._equipment(start, record)
        air, inlet, reheat, outside = equipment
        prediction = self._predict(start, equipment, self._mass_proxy)
        if self.extended:
            denominator = 1 + self._theta[:, 0] + self._theta[:, 1] + self._theta[:, 2] / 10000 * air
            features = np.column_stack([outside - prediction, self._mass_proxy - prediction,
                (air * (inlet - prediction) + reheat) / 10000, np.ones(len(start))]) / denominator[:, None]
            residual = end - prediction
        else:
            # Retained solely for the controlled implicit/extended forecast
            # comparison; the registered policy uses the extended branch.
            features = np.column_stack([outside - end, self._mass_proxy - end,
                (air * (inlet - end) + reheat) / 10000, np.ones(len(start))])
            residual = end - start - np.sum(features * self._theta, axis=1)
        return features, prediction, residual

    def _learn_delayed_transition(self, start, end, record):
        features, prediction, residual = self.features_and_prediction(start, end, record)
        self._residual_mae = .97 * self._residual_mae + .03 * abs(prediction - end)
        pseudo_target = np.sum(features * self._theta, axis=1) + np.clip(residual, -10, 10)
        keep = self.FORGETTING
        self.information = (keep * self.information + np.einsum("vi,vj->vij", features, features)
                            + (1 - keep) * self.precision)
        self.rhs = keep * self.rhs + features * pseudo_target[:, None] + (1 - keep) * self.prior_rhs
        estimate = np.linalg.solve(self.information, self.rhs[..., None])[..., 0]
        self._theta = np.clip(estimate, [0, 0, .001, -2], [5, 5, 100, 2])
        # This is inverse regularized information, not a calibrated posterior.
        covariance = np.linalg.inv(self.information)
        self._covariance = .5 * (covariance + covariance.transpose(0, 2, 1))
        self._updates += 1
