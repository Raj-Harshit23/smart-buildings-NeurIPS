# Causal forecast study: permanent priors and prediction-error fitting

Basis: [MATHEMATICAL_FRAMEWORK.md](MATHEMATICAL_FRAMEWORK.md), Sections 1, 7–9;
[ALGORITHM_RESEARCH.md](ALGORITHM_RESEARCH.md), Sections 5 and 8.
This is offline prequential model evaluation on actual recorded commands. It
does not change the deployed V4 policy or establish a control/leaderboard gain.

## Evidence and rejected shortcuts

On corrected Fairbanks-2 data, stale/current-observer MAE is 0.384/0.348 K.
Removing only the correction cap yields 0.336 K; using the ungated model yields
3.123 K and p99 error 23.674 K. Therefore the model is not accurate enough to
remove confidence weighting. No heat-gain upper bound is active at the final
state, and the smallest physical region is 52 square metres; a tiny-region
heat-gain-cap explanation is unsupported.

The 1.5 K cap is nevertheless a structural restriction:

$$
|\widehat T-T|\geq (|T-T_{stale}|-1.5)_+.
$$

Its idealized p99 error lower bound is 5.873 K on these labels. Better dynamics
and uncertainty estimates are needed before enlarging correction authority.

## Permanent physical prior

Ignoring clipping/projection safeguards, current exponentially weighted RLS has
information matrix

$$
S_t=\lambda^t\Lambda+\sum_{j=1}^t\lambda^{t-j}\phi_j\phi_j^T,
\qquad\lambda=0.997.
$$

If a direction is not excited, its prior precision decays toward zero. A global
numerical covariance cap prevents overflow but does not restore a meaningful
physical prior. The first comparison retains exactly the existing prior mean,
prior covariance, forgetting factor and parameter bounds, while maintaining

$$
\begin{aligned}
S_t&=\lambda S_{t-1}+\phi_t\phi_t^T+(1-\lambda)\Lambda,\\
b_t&=\lambda b_{t-1}+\phi_t z_t+(1-\lambda)\Lambda\theta_0.
\end{aligned}
$$

With $S_0=\Lambda$, this gives
$S_t=\Lambda+\sum_j\lambda^{t-j}\phi_j\phi_j^T\succeq\Lambda>0$.
The unconstrained estimate is $S_t^{-1}b_t$. Apply the same componentwise physical
projection as V4; that projection is not the exact box-constrained ridge optimum.
Use the existing residual clipping in constructing each pseudo-observation
$z_t=\phi_t^T\theta_{t-1}+\operatorname{clip}(y_t-\phi_t^T\theta_{t-1},-10,10)$.
This is a guarded recursive fit, not a proof of unbiased physical identification.

## Actual prediction error versus implicit regression

The model is

$$
f_\theta=\frac{T+aT_o+bM+\kappa(AF T_a+h)+c}
                   {D},\quad D=1+a+b+\kappa AF,
$$

where $AF=c_af$ is air heat-capacity rate. Its implicit regression residual is

$$
e_{implicit}=D(T^+-f_\theta).
$$

Thus its squared error weights temperature errors by $D^2$, which changes with
airflow and fitted parameters. Its regressors also contain the observed $T^+$,
allowing structural/process error to correlate with the regressors.

The second comparison uses one extended least-squares linearization of the
actual temperature predictor per newly available delayed label. For the scaled
parameter $k_s=10000\kappa$:

$$
g=\nabla_{(a,b,k_s,c)}f_\theta
=\frac1D\left(T_o-f_\theta,\ M-f_\theta,
\frac{AF(T_a-f_\theta)+h}{10000},\ 1\right).
$$

Use the same permanent-prior updates with $\phi_t=g_t$ and

$$
z_t=g_t^T\theta_{t-1}
 +\operatorname{clip}(T_t^+-f_{\theta_{t-1}},-10,10).
$$

Past Jacobians remain their original linearizations, as in extended recursive
least squares. This approximates regularized nonlinear prediction-error fitting;
it does not solve a global nonlinear problem or eliminate unobserved mass and
interzone coupling. Forecast confidence remains the existing empirical residual
weight, not a calibrated posterior.

## Causal comparison

Reproduce the baseline observer on the actual V4 observation/action sequence and
require equality with its saved causal prediction trace. At decision $t$, learn
from the observed transition using actual $u_{t-2}$ and predict current physical
temperature using actual $u_{t-1}$. The next observation is an offline validation
label only. Do not substitute counterfactual controller commands into the model's
history. Validate the analytic Jacobian against finite differences.

Compare stale, original RLS, permanent-prior implicit fit, and permanent-prior
extended fit on all four completed public scenarios, reporting both guarded and
unguarded forecast metrics. A model that improves prediction still needs fresh
closed-loop transfer tests before any scored submission. Hidden-scenario labels,
future weather and private physical state are not model inputs.
