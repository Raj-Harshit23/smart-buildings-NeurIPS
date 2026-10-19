# Delay-aware thermal observer: second controlled experiment

References: [MATHEMATICAL_FRAMEWORK.md](MATHEMATICAL_FRAMEWORK.md), verified sensor
timing and Sections 1, 6, 7; [ALGORITHM_RESEARCH.md](ALGORITHM_RESEARCH.md), Sections
5 and 8. Registered controller: `lag_aware_allocator`.

## Hypothesis and information

The public local simulator returns zone temperature one physical step late:

$$
o_t^T=\mathcal N(T_{t-1}),\qquad T_{t+1}=F_T(x_t,u_t,w_t).
$$

Feedback from $o_t$ can therefore keep cooling or heating after the actual zone
has already crossed its target. Estimate the current physical temperature from
the delayed sensor and the most recent applied action, then feed that estimate
to the unchanged first allocator. No reward, actual grid temperature, future
weather, or scenario identifier enters the controller.

## Model and online identification

The compact per-zone model is

$$
T^+=\frac{T+aT_o+bM+k(c_afT_{ahu}+h)+c}
               {1+a+b+kc_af},\quad
a,b\geq0,\quad k>0.
$$

$M$ is a causal four-hour filtered temperature proxy; $c_af$ is air heat-capacity
rate; $h$ is reheat power reconstructed from past commands and public equipment
priors. This approximates the spatial thermal field, internal heat gains and
shared equipment. It is not an exact wall-state reconstruction.

For recursive identification, rearrange the model as

$$
T^+-T=
a(T_o-T^+)+b(M-T^+)+kQ^{HVAC}+c.
$$

At call $t$, the observed transition $o_{t-1}^T\to o_t^T$ is paired with
$u_{t-2}$, not $u_t$ or $u_{t-1}$. The observer then predicts current physical
temperature using $u_{t-1}$. Delayed labels are available from the policy's own
history; no future observation is used to choose an action.

Projected recursive least squares uses forgetting factor 0.997 and physical
coefficient bounds. Covariance eigenvalues are bounded periodically to avoid
unlimited growth in unexcited directions. This is an approximate adaptive model;
its covariance is not a calibrated Bayesian posterior.

Shared priors are informed by a public Phoenix training fit. The initial heat
gain scales with zone area recovered from the public zone map using the public
four-connected component convention. It assumes 0.2 m cells and 3 m floor height
as model priors, not hidden-building facts. No fitted building/zone parameter
lookup is deployed.

After 96 identification steps, correction weight ramps over 192 steps and is
reduced when recent prediction residuals are large. Applied temperature correction
is capped at 1.5 K. These are explicit heuristics, not stability or improvement
guarantees. In particular, small residuals alone do not prove identification in
unexcited operating regimes. Cross-building validation must test that risk.

The zone thermal gain used by the first allocator's heat-demand feedback is
unchanged. This experiment isolates the added current-state estimation rather
than simultaneously retuning the allocator. It does not yet solve a long-horizon
economic MPC problem.

## Grounding and reproduction

A conditional thermal-core identification experiment fitted the first 70% of a
public Phoenix/building-8 baseline trajectory and tested the remaining 30%:

| Alignment | One-step MAE | 99th-percentile absolute error |
| --- | ---: | ---: |
| Naive action/observation pairing | 1.8432 K | 11.3278 K |
| Correct delayed pairing | 0.1222 K | 0.4792 K |

This experiment supplies physical state/input conditions to assess the thermal
core; it is not an end-to-end causal forecast or control result. The deployed
observer instead reconstructs current state from legal delayed observations.
The [fit record](experiments/thermal_core_identification.json) and
[reproduction script](experiments/fit_delayed_thermal_core.py) preserve the exact
scope and fitted parameters. Those per-zone fits are diagnostic records, not a
lookup used in the controller.

```bash
python experiments/fit_delayed_thermal_core.py \
  results/allocator-v1/building_train_8/phoenix_4/pid_water \
  results/thermal-core-reproduction

python -m smart_buildings.diagnose_policy \
  --scenario smart_buildings/training_scenarios/building_train_8/phoenix_4 \
  --policy lag_aware_allocator --steps 8640 \
  --output results/lag-aware-v2/building_train_8/phoenix_4/lag_aware_allocator
```

The suite passes 87 tests, including correct delayed-label pairing, estimator
behavior without excitation, and resetting the delayed command queue. The first
full local Phoenix/building-8 run achieved mean loss 0.03947549, compared with
0.15549259 for the first allocator and 0.16930462 for fresh V0. Other climates and
buildings still require full validation. No hidden-leaderboard result is implied.
