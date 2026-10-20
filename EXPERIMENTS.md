# Experiment record

## Actual Kaggle baseline

Submission **57407757**, notebook
`harshitraj2123221/smart-buildings-basic-v0-pid-water`, version 1, is complete:
**public score 0.06404**, confirmed 2026-10-20. Lower is better.
[API record](experiments/kaggle_baseline.json).

Its official Atlanta training validation returned -215.2204509578005 over 8641
reported steps. That training return is distinct from its hidden public score.
The [public SDK verification](OFFICIAL_SDK_VERIFICATION.md) explains the terminal
count, output schema, observation units/timing and request timeouts.

## Fresh adaptive heating evaluation

Source is frozen at `cb2256a`; all comparisons use **8640 matching actions** and
verified **non_mutating_v2** diagnostics. The prior full suite passed 99 tests,
and the two added harness regressions passed separately. The test receipt records
both runs and all source hashes; it does not claim a single 101-test invocation.

| Scenario | V0 loss | Adaptive heating loss | Improvement |
| --- | ---: | ---: | ---: |
| Building 1 / Fairbanks | 0.21392442 | 0.08664817 | 59.5% |
| Building 5 / Singapore | 0.10990292 | 0.02603477 | 76.3% |
| Building 8 / Phoenix | 0.16020952 | 0.03947548 | 75.4% |
| Building 2 / Fairbanks | 0.07929014 | 0.06932532 | 12.6% |
| Building 4 / Atlanta | 0.02893980 | 0.03057651 | **5.7% worse** |
| Building 7 / Denver | 0.03687855 | 0.02603332 | 29.4% |

The predeclared three-case pilot mean improved **68.6%**. All six cases are now
complete: mean loss fell from **0.10485756 to 0.04634893 (55.8%)**. Mean transfer
loss improved **13.2%**; the worst regression is Atlanta at 5.7%, within the
predeclared 10% limit. The complete local gate passed. [Full precision completed results](experiments/fresh_adaptive_heating_results.json).
No candidate leaderboard score is claimed.

All three components improve on Fairbanks-1. On the previously problematic
Fairbanks-2, comfort improves from 0.09979904 to 0.07910492 and energy from
0.03720329 to 0.03657675; carbon worsens from 0.05985028 to 0.07273506 (**21.5%**).
The exact weighted loss still improves 12.6%, so carbon is a remaining tradeoff.

The frozen notebook export exactly reproduced **17280 recorded actions** across
complete Fairbanks-1 and Phoenix trajectories. The private offline artifact is
prepared. Private notebook `harshitraj2123221/smart-buildings-adaptive-heating`,
version 1, completed official validation. Kaggle derived its slug from its title;
tracking was reconciled using exact source/adapter equivalence without relaunching.
The tracker requires official
full-horizon gateway completion and at most 10% regression on V0's matching
Atlanta training case before requesting one scored entry. Attempt markers and
allowance checks prevent duplicate entries. Submission **57435234**, created at 11:16 UTC on 2026-10-20, is pending scoring.
Its official 8641-step Atlanta validation improved loss 11.5% (return
-190.38652838796406). [API record](experiments/kaggle_adaptive_v4.json).

## Controlled cooling-inverse study

[Design](ADAPTIVE_COOLING.md), frozen source `0db1fe8`.
The prior 101 checks plus **8 new tests** passed in separate recorded runs.
Tests check inverse/forward consistency, loss of cooling authority, real-building
bounds, unchanged heating actions and actual selected-command history.

Private offline notebook:
`harshitraj2123221/smart-buildings-v5-cooling-inverse-study`, version 1.
It compares V4 and the cooling variant on the same full Phoenix and Singapore
training cases, on one Kaggle image. It creates no scored entry. The exported
implementations matched their source on 8192 policy/input checks across two
building trajectories; these prefix checks validate serialization and do not
estimate V5 control performance. The paired study completed: Phoenix loss 0.03947548 → 0.03908260 (1.0% better),
Singapore 0.02603477 → 0.02507432 (3.7% better). Mean improvement is **2.1%**.
This modest standalone gain is held from submission; no wider V5 campaign is
being launched. The V4 reference losses reproduce local results on Kaggle
Python 3.13.15 / NumPy 2.1.3. [Research output](experiments/cooling_inverse_study.json).

## Permanent-prior observer validation

[Design and causal forecast evidence](REGULARIZED_OBSERVER_STUDY.md).
The production implementation matches all four complete forecast comparisons
exactly; the full suite passed **120 tests**. A fresh frozen V6 control campaign
at `9633fda` is running, with the same six public scenarios and V4 as reference.
The independent absolute V0 regression gate remains in force. No V6 return or
leaderboard gain is claimed before those runs complete.
[Source, tests, equivalence and bounded pipeline receipt](experiments/regularized_heating_validation.json).

The next [joint cooling derivation](JOINT_COOLING_DESIGN.md) identifies when colder
supply can reduce both fan and chiller electricity at fixed room targets and
why minimum airflow prevents applying that rule everywhere. Its conditional
algebra was checked on 10000 generated feasible cases and a floor-induced
counterexample; this is not a control-performance result.

## Measured remaining limitations

[Fresh diagnostics](experiments/fresh_control_diagnostics.json):

- Fairbanks-2 causal state estimation MAE is 0.348 K versus 0.384 K from the stale
  sensor, with p99 error still 7.342 K. The fixed 1.5 K correction cap implies
  even a perfect bounded predictor cannot beat p99 **5.873 K** on these labels.
  Forecast ablations found only 3.5% MAE improvement by removing the cap, while
  removing confidence weighting worsens MAE to 3.123 K. No guard-only candidate
  is being promoted. The next study retains physical priors permanently and
  fits actual prediction error; no closed-loop gain is inferred from forecasts.
- No final heat-gain, outside-coupling or mass-coupling upper bound is active in
  this case. Region area is at least 52 square metres. Thus a proposed tiny-zone
  heat-gain-cap explanation is unsupported and is not being implemented.
- In Phoenix, an AHU is off while a zone is over its current upper band by 0.5 K
  on 14.7% of decisions. Such steps contain 32.1% of the total comfort loss.
  The pure-cooling subset accounts for 31.3%. These are associations, not gains
  obtainable merely by turning equipment on. The actual parent clock matches
  the comfort flag, rejecting a broad timezone-mismatch explanation there.

The mathematical reference now distinguishes physical reward-zone temperatures
from VAV registers, past rounded occupancy observations from forward interval
reward occupancy, and possible sensor/actuator/physical-zone count differences.

## Invalidated historical measurements

The original runner made an extra read of the boiler temperature getter, which
advances tank state. Two reads at the same timestamp returned 332.5 K then 335 K.
[Reproduction](experiments/diagnostic_side_effect.json). The repaired runner reuses
returned observations and captures the response already produced inside
`env.step`; its observations, actions and rewards matched an ordinary 24-step
rollout exactly. Campaign and export code reject legacy summaries.

All earlier performance records under `allocator-v1`, `lag-aware-v2`,
`lag-aware-pid-v3` and `adaptive-heating-v4` are instrumented-system history and
cannot qualify a normal-evaluation submission. No candidate was submitted from
those results. The actual Kaggle V0 score is unaffected. Historical comparisons
remain [archived with an explicit invalidation](experiments/allocator_comparison.json).

Those experiments motivated the observer/PID ablation and thermal inversion,
but their quantitative gains, conditional thermal fits and observer-error metrics
must not be presented as normal local or leaderboard performance. Unit-test
results remain valid within their stated scope. The earlier claim that a positive
occupancy signal meant more than 125 occupants was also corrected: the private
normalizer assignment is unused, and the actual constructor setting is zero.

## Next closed-loop candidate: permanent-prior observer

The [regularized observer study](REGULARIZED_OBSERVER_STUDY.md) improves causal
state forecasts on all four completed cases. Its production implementation
exactly reproduces the prototype's forecast metrics across four complete
recorded trajectories. This is forecast evidence, not a new control score.

The full suite passes **120 tests**, including permanent information in an
unexcited direction, an independently checked prediction Jacobian, large trusted
corrections reaching the controller, reset, delayed labels, and real-building
action bounds. The next full campaign compares `regularized_heating` with V4
using its completed matched references; V0's absolute regression gate is retained.
