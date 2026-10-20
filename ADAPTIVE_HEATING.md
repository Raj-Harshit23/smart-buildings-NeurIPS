# Adaptive inverse heat demand

References: [MATHEMATICAL_FRAMEWORK.md](MATHEMATICAL_FRAMEWORK.md), Sections 1,
6–9; [ALGORITHM_RESEARCH.md](ALGORITHM_RESEARCH.md), Sections 5 and 6.3;
[DELAY_AWARE_ALLOCATOR.md](DELAY_AWARE_ALLOCATOR.md).

## Mechanism and evidence

V2 improves mean loss but uses approximately twice the baseline occupied-time
reheat power on building 2 / Fairbanks. The observer/PID ablation is still running;
matched prefixes retain cooling gains but lose the allocation benefit in the
colder building 1 / Fairbanks case. These partial results motivate retaining
physical airflow/water coordination and adapting its heat request. No full
ablation outcome is assumed here.

The first allocator converts a desired temperature increment to heat using the
same fixed $2\times10^{-5}$ K/W gain in every zone. The observer already learns
a causal implicit thermal model:

$$
T^+-T=a(T_o-T^+)+b(M-T^+)+\kappa Q+c,\qquad
Q=c_af(T_{ahu}-T^+)+h.
$$

For the unchanged six-step recovery target $T^*=T+(L-T)_+/6$, inversion gives

$$
Q_{\mathrm{model}}=
\left[\frac{T^*-T-a(T_o-T^*)-b(M-T^*)-c}{\kappa}\right]_+.
$$

The deployed request blends this with V1's request using the unchanged observer
warmup/residual weight $w\in[0,1]$:

$$
Q^*=(1-w)Q_{\mathrm{fixed}}+wQ_{\mathrm{model}},\qquad
Q_{\mathrm{fixed}}=
\frac{(L-T)_+/6+[-\widehat{\Delta T}]_+}{2\times10^{-5}}.
$$

This weight is an empirical guard, not a calibrated probability. In a badly
identified zone the policy approaches the earlier heuristic rather than trusting
arbitrary learned coefficients. Identifiability, boundary/interzone heat exchange,
the mass proxy and equipment priors remain limitations.

Inversion amplifies parameter error: at fixed numerator,
$\partial Q_{\mathrm{model}}/\partial\kappa=-Q_{\mathrm{model}}/\kappa$.
Small fitted residuals do not identify the correct heat gain when the regressor
matrix has deficient rank. Moreover, the implicit regression contains $T^+$ in
its regressors; model error can correlate with those regressors, violating the
usual $\mathbb E[X^T\varepsilon]=0$ least-squares condition. Finite covariance,
positive coefficients and blending control numerical risk but do not prove
unbiased identification or accurate counterfactual predictions. Full closed-loop
validation is therefore necessary even when one-step fit error looks small.

At a chosen AHU inlet, airflow is requested from the discharge cap:

$$
f^*=\frac{Q^*}{c_a(308.15-T^*)},\qquad
h^*=\left[Q^*-c_af(T_{ahu}-T^*)\right]_+.
$$

Apply the same ventilation floor, design-flow bounds, water-limited coil cap,
shared minimum pressure and AHU capacity prior as V1. Command reheat through the
current measured water temperature. Return-air mixing uses area weights derived
from public geometry, consistently with the observer's equipment reconstruction.
The existing cooling law remains unchanged. Current physical temperature is
estimated causally; future observations, reward and scenario IDs are unavailable.

This is adaptive inverse feedback with a recovery surrogate, not an economic MPC
solution or a guarantee of better reward. It minimizes neither an exact comfort
term nor the full long-horizon weighted objective directly. Evaluation retains
the exact 0.6/0.2/0.2 comfort/energy/carbon score and 8640-step horizons.

## Validation plan

First verify inversion against the forward implicit thermal model and all native
action constraints on differently sized public buildings. Then run the same six
full scenarios. Qualification uses V0 as the absolute reference: at least 5%
combined mean improvement, nonnegative transfer mean improvement, no scenario
regression above 10%, and mean controller latency below 10 ms. Also compare with
V2 and the observer/PID ablation. Keep source/test/export hashes pinned and require
official gateway completion before one authorized scored entry.

In a climate where the heating branch never changes an action, a full simulator
rerun can be replaced by exact causal replay of an existing complete trajectory.
Require all 8640 continuous and binary actions to match exactly, with the same
unchanged seeded simulator and initial-state definition. Record the reference
rollout and evaluation mode explicitly; any mismatch requires a fresh simulation.
This is an equivalence argument for one realized trajectory, not an assumption
that policies are universally identical or that hidden dynamics are the same.
