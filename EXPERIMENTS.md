# Experiment record

**Evaluation correction, 2026-10-20:** the original diagnostic runner made an
extra read of the boiler temperature getter, which advances tank state. All
allocator/observer results below from `results/allocator-v1`, `lag-aware-v2`,
`lag-aware-pid-v3` and `adaptive-heating-v4` describe this instrumented system;
they do **not** qualify as normal local evaluation or justify a submission.
Their campaign decisions are invalidated. No candidate was submitted from them.
The official Kaggle baseline score is unaffected. Fresh comparisons use protocol
`non_mutating_v2`, which passed exact observation/action/reward equivalence with
an ordinary 24-step rollout. Both the exporter and campaign reject legacy runs.
See [side-effect proof](experiments/diagnostic_side_effect.json).

The historical occupancy breakdown also treated a positive normalized occupancy
signal as occupied. With this runner's normalizer, that threshold means more
than 125 occupants; it is not the zero-occupancy threshold. Historical power
figures described as occupied-time below are therefore high-occupancy diagnostics,
not an exact occupied/unoccupied partition. Policy code does not use that split.

## First scored baseline

- Kaggle notebook: `harshitraj2123221/smart-buildings-basic-v0-pid-water`, version 1.
- Submission: `57407757`, submitted 2026-10-19 at 19:55 UTC.
- Confirmed 2026-10-20 09:25 UTC: **COMPLETE, public score 0.06404**.
- [Scored baseline record](experiments/kaggle_baseline.json).
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
sensor-band ablation completed from frozen commit `5f53c59`. Fresh results
differ from stored `baseline_returns.json`; compare candidates against fresh
matched runs rather than attributing that difference to an unverified cause.

V1 failed the transfer gate: building 2 / Fairbanks regressed 72.5%, Atlanta
0.6% and Denver 1.9%. Mean transfer loss regressed 40.4%, despite a 5.6% mean
improvement across all six cases. The launcher therefore published no V1
notebook and submitted no V1 entry. The Singapore sensor-band-only ablation
exactly reproduced V1's mean components and loss; its small gain is not evidence
for the heating allocation mechanism.

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
0.01638252. Mean controller time was 2.46 ms/action in this run.

All six 8640-step comparisons are now complete:

| Building / climate | V0 loss | V1 loss | V2 loss | V2 change from V0 |
| --- | ---: | ---: | ---: | ---: |
| 1 / Fairbanks | 0.21387482 | 0.13391418 | 0.10043117 | 53.0% better |
| 5 / Singapore | 0.10944237 | 0.10904646 | 0.02603479 | 76.2% better |
| 8 / Phoenix | 0.16930462 | 0.15549259 | 0.03947549 | 76.7% better |
| 2 / Fairbanks | 0.07945105 | 0.13705198 | 0.09006283 | 13.4% worse |
| 4 / Atlanta | 0.02893918 | 0.02911322 | 0.03057654 | 5.7% worse |
| 7 / Denver | 0.03627489 | 0.03696271 | 0.02592785 | 28.5% better |
| **Mean** | **0.10621449** | **0.10026352** | **0.05208478** | **51.0% better** |

V2 passed the supervisor's comparison against V1, improving the six-case mean
48.1%. However, an additional comparison against the original V0 rejects a
scored submission: the worst regression exceeds 10% and mean transfer loss
regresses 1.3%. Beating the failed V1 controller is insufficient qualification.
The official objective is the scenario mean; this regression threshold is our
predeclared experiment gate, not a competition constraint.

[Machine-readable full comparison](experiments/allocator_comparison.json).
These local means must not be compared numerically with V0's 0.06404 public
score as if they came from the same evaluation distribution.

## Observer measurement and next ablation

Replaying every action exactly on completed V2 trajectories yields:

| Scenario | Stale sensor MAE | Causal observer MAE | Stale sensor p99 | Observer p99 |
| --- | ---: | ---: | ---: | ---: |
| 8 / Phoenix | 0.593 K | 0.179 K | 3.693 K | 2.205 K |
| 2 / Fairbanks | 1.085 K | 0.909 K | 10.187 K | 9.889 K |

The next observed temperature is used solely as an offline current-state label
under the verified local sensor delay. It never enters the action calculation.
The small mean correction cap and weak model confidence leave substantial
Fairbanks tail error. These results are distinct from the conditional offline
thermal-core fit, which assumes known physical starting state.

[Observer evaluation record](experiments/observer_evaluation.json).
The conditional pump-pressure saving estimate is only 0.84 W in Phoenix and
183 W in Fairbanks, so pump symmetry is not the next priority.

V2's Fairbanks comfort component improves from 0.10061 to 0.08439, but its
energy/carbon components approximately double. The next controlled ablation
retains the observer and replaces heating allocation with the original PID
actuator law. See [design and unchanged qualification gate](LAG_AWARE_PID.md).
Its full-rollout results are pending; no new candidate public score is claimed.
The expanded suite passed **92 tests** (14 existing warnings), including exact
observer equivalence when both variants use the same actuator law, correct
two-call action/label alignment, reset, bounded unexcited covariance, and delivery
of the corrected state to PID. Test receipt and source hashes are retained in
`results/lag-aware-pid-v3/tests.json`.

## Adaptive inverse heat demand

[Design and mathematical limitations](ADAPTIVE_HEATING.md). This retains the
delay observer and heating airflow/water coordination, replacing the shared
fixed-gain heat request with guarded inversion of the online thermal model.
Both Fairbanks full episodes are running; partial returns are not final results.

The expanded suite passed **99 tests**, with 14 existing warnings. New tests
check the inverse request against the forward implicit model, confidence
fallback, mature actions on two differently sized real buildings, exact selected
action history, delayed identification, reset and covariance stability.

The complete Phoenix and Singapore actions exactly match V2 on all 8640 steps
each. Their losses are therefore 0.03947549 and 0.02603479. These are explicitly
recorded as **action-equivalent replay** of completed local simulator rollouts,
not new simulator runs. The physics, weather, scenarios, diagnostic runner and
environment AST (apart from controller dispatch) were checked against `e13512e`.
This equivalence says nothing about hidden evaluator parity. A mismatching
action prevents trajectory reuse. No adaptive-controller public score exists yet.
