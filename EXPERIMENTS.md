# Experiment record

## First scored baseline

- Kaggle notebook: `harshitraj2123221/smart-buildings-basic-v0-pid-water`, version 1.
- Submission: `57407757`, submitted 2026-10-19 at 19:55 UTC.
- Latest check during this implementation: pending, public score absent, no error.
- Official training validation: 8641 reported steps, return -215.2204509578005.
  This is not a leaderboard score.

## Heating-aware allocator

Design: [THERMAL_ALLOCATOR.md](THERMAL_ALLOCATOR.md).
Validation plan and autonomous scope: [AUTONOMOUS_WORK.md](AUTONOMOUS_WORK.md).

The full existing suite plus new allocator regression tests passed:
**84 tests passed**, with 14 existing deprecation warnings, on 2026-10-19.
The new tests cover an actual water-limited coil, metadata reindexing, a cooling
control comparison, and controller memory reset. Real simulator smoke and
second-building cold/mixed/hot action checks also passed.

Fresh 8640-step pilot comparisons are running. The first completed V0 case,
building 8/Phoenix, returned -1462.7918917707952 (mean loss 0.1693046171).
Its comfort/energy/carbon means were 0.1620191187 / 0.2509862483 / 0.1094794646.
Fresh results differ from stored `baseline_returns.json`; compare candidates
against the fresh matched runs rather than attributing that difference to an
unverified cause.

No candidate leaderboard improvement is claimed before scored evaluation.
Completed campaign comparisons will be added here after qualification.

## Next modeling finding: sensor timing

A real three-step probe verified a one-step lag in local zone-temperature
observations. The [probe record](experiments/observation_delay.json) and
[framework](MATHEMATICAL_FRAMEWORK.md) now document it. This motivates a
delay-aware state estimator and correct action/temperature alignment before
fitting a predictive thermal model. It does not establish identical hidden-SDK
timing, or prove that a simple extrapolation will improve control.
