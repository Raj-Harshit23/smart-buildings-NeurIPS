"""Prequential forecast comparison using actual causal V4 command histories."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import numpy as np

from smart_buildings.baselines.lag_aware_allocator import LagAwareAllocatorPolicy
from smart_buildings.evaluate import create_building_meta_data, load_hybrid_environment


class PermanentPriorObserver(LagAwareAllocatorPolicy):
    def __init__(self, metadata, spec, extended=False):
        super().__init__(metadata, spec)
        self.extended = extended
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
        self._updates += 1


def check_gradient(observer, start, record):
    equipment = observer._equipment(start, record)
    observer._mass_proxy = start.copy()
    analytic, _, _ = observer.features_and_prediction(start, start, record)
    theta = observer._theta.copy()
    errors = []
    for j in range(4):
        epsilon = 1e-4 if j == 2 else 1e-5
        observer._theta = theta.copy();observer._theta[:, j] += epsilon
        plus = observer._predict(start, equipment, observer._mass_proxy)
        observer._theta = theta.copy();observer._theta[:, j] -= epsilon
        minus = observer._predict(start, equipment, observer._mass_proxy)
        errors.append(float(np.max(abs((plus - minus) / (2 * epsilon) - analytic[:, j]))))
    observer._theta = theta
    assert max(errors) < 1e-5, errors
    return max(errors)


def replay(observer, observations, continuous, binary):
    raw_trace, predictions, confidence = [], [], []
    for t, obs in enumerate(observations):
        raw = np.asarray([observer._denormalize_zone_temp(obs[i]) for i in observer._indices])
        if observer._mass_proxy is None: observer._mass_proxy = raw.copy()
        if len(observer._commands) == 2:
            observer._learn_delayed_transition(observer._raw_previous, raw, observer._commands[0])
            observer._mass_proxy = (observer.MASS_RETAINED * observer._mass_proxy
                                   + (1 - observer.MASS_RETAINED) * observer._raw_previous)
        prediction = observer._predict(raw, observer._equipment(raw, observer._commands[-1]), observer._mass_proxy) if observer._commands else raw.copy()
        warmup = float(np.clip((observer._updates - observer.WARMUP_STEPS) / observer.RAMP_STEPS, 0, 1))
        weight = warmup * np.clip((.5 - observer._residual_mae) / .3, 0, 1)
        assert np.isfinite(prediction).all() and np.isfinite(observer._theta).all()
        raw_trace.append(raw);predictions.append(prediction);confidence.append(weight)
        actual = {"continuous_action": continuous[t], "discrete_action": binary[t].astype(np.int32)}
        observer._commands.append(observer._record(obs, actual));observer._raw_previous = raw.copy()
    return np.asarray(raw_trace), np.asarray(predictions), np.asarray(confidence)


def metrics(raw, prediction, confidence, truth):
    methods = {"unguarded_model": prediction, "confidence_weighted_unclipped": raw + confidence * (prediction - raw),
               "confidence_weighted_clipped": raw + confidence * np.clip(prediction - raw, -1.5, 1.5)}
    result = {}
    for name, estimate in methods.items():
        result[name] = {}
        for label, start in [("whole_episode", 0), ("after_one_day", 288)]:
            error = abs(estimate[start:] - truth[start:])
            result[name][label] = {"mae_k": float(error.mean()), "p99_k": float(np.quantile(error, .99))}
    return result


def main():
    root, output = map(Path, sys.argv[1:3]);output.mkdir(parents=True, exist_ok=True)
    scenarios = ["building_train_2/fairbanks_2", "building_train_1/fairbanks_1",
                 "building_train_8/phoenix_4", "building_train_5/singapore_4"]
    reports = []
    for scenario in scenarios:
        directory = root / scenario / "adaptive_heating"
        summary = json.loads((directory / "summary.json").read_text())
        assert summary.get("evaluation_protocol") == "non_mutating_v2"
        for relative, digest in summary["source_sha256"].items():
            assert hashlib.sha256(Path(relative).read_bytes()).hexdigest() == digest
        path = Path("smart_buildings/training_scenarios") / scenario
        env = load_hybrid_environment(path / "config.gin");metadata = create_building_meta_data(env, path)
        with np.load(directory / "trajectory.npz") as archive:
            observations = archive["observations"];continuous = archive["continuous_actions"];binary = archive["discrete_actions"]
        row = {"scenario": scenario, "models": {}, "scope": "Actual V4 commands; causal forecasts; no alternative control-performance claim"}
        for name in ["original_rls", "permanent_implicit", "permanent_extended"]:
            observer = (LagAwareAllocatorPolicy(metadata, env.action_spec()) if name == "original_rls"
                        else PermanentPriorObserver(metadata, env.action_spec(), extended=name == "permanent_extended"))
            if name == "permanent_extended":
                start = np.asarray([observer._denormalize_zone_temp(observations[0, i]) for i in observer._indices])
                record = observer._record(observations[0], {"continuous_action": continuous[0], "discrete_action": binary[0].astype(np.int32)})
                row["jacobian_maximum_finite_difference_error"] = check_gradient(observer, start, record)
                observer._mass_proxy = None
            raw, prediction, confidence = replay(observer, observations, continuous, binary)
            truth = raw[1:]
            raw, prediction, confidence = raw[:-1], prediction[:-1], confidence[:-1]
            if scenario == scenarios[0] and name == "original_rls":
                with np.load(root / "observer-fairbanks-2/estimates.npz") as saved:
                    np.testing.assert_array_equal(prediction, saved["prediction"])
                    np.testing.assert_array_equal(observer._theta, saved["final_theta"])
                row["baseline_causal_prediction_trace_exact"] = True
            row["models"][name] = metrics(raw, prediction, confidence, truth)
            print(json.dumps({"scenario": scenario, "model": name, "metrics": row["models"][name]}), flush=True)
        reports.append(row)
        (output / (scenario.replace("/", "-") + ".json")).write_text(json.dumps(row, indent=2))
    report = {"scope": "Prequential model comparison only; no control or leaderboard claim", "scenarios": reports}
    (output / "comparison.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__": main()
