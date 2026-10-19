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

Fresh 8640-step pilot comparisons are complete:

| Scenario | V0 mean loss | Allocator mean loss | Improvement |
| --- | ---: | ---: | ---: |
| building 1 / Fairbanks | 0.21387482 | 0.13391418 | 37.4% |
| building 5 / Singapore | 0.10944237 | 0.10904646 | 0.4% |
| building 8 / Phoenix | 0.16930462 | 0.15549259 | 8.2% |

Mean pilot loss improved by about 19.1%. All three components improved in
Fairbanks; Singapore barely changed. Three additional validation pairs and the
sensor-band ablation are running from frozen commit `5f53c59`. Fresh results
differ from stored `baseline_returns.json`; compare candidates against fresh
matched runs rather than attributing that difference to an unverified cause.

No candidate leaderboard improvement is claimed before scored evaluation.
Completed campaign comparisons will be added here after qualification.

## Next modeling finding: sensor timing

A real three-step probe verified a one-step lag in local zone-temperature
observations. The [probe record](experiments/observation_delay.json) and
[framework](MATHEMATICAL_FRAMEWORK.md) now document it. This motivates a
delay-aware state estimator and correct action/temperature alignment before
fitting a predictive thermal model. It does not establish identical hidden-SDK
timing, or prove that a simple extrapolation will improve control.

## Delay-aware observer/controller

Design and identification evidence: [DELAY_AWARE_ALLOCATOR.md](DELAY_AWARE_ALLOCATOR.md).
The full suite now passes **87 tests**. Correct action/label alignment reduced
conditional one-step thermal-core MAE from 1.8432 K to 0.1222 K on a withheld time
segment of the public Phoenix baseline trajectory. This is dynamics evidence,
not a deployable forecast-performance claim.

The first full 8640-step local Phoenix/building-8 run returned -341.0682588817434,
mean loss **0.03947549**. That is a 76.7% reduction from fresh V0 and 74.6% from
the first allocator. Comfort/energy/carbon means were 0.03676543 / 0.07069865 /
0.01638252. Mean controller time was 2.46 ms/action in this run. Fairbanks and
Singapore comparisons are running; transfer performance is not yet established.
