# Controlled cooling-inverse experiment

Basis: [MATHEMATICAL_FRAMEWORK.md](MATHEMATICAL_FRAMEWORK.md), Sections 1, 3,
6–9; [ADAPTIVE_HEATING.md](ADAPTIVE_HEATING.md);
[PUBLIC_CONTROL_REVIEW.md](PUBLIC_CONTROL_REVIEW.md).

The first corrected Fairbanks-1 pair supports adaptive heating allocation. The
heating controller retains the original cooling law, so cooling inverse control
is a separate next hypothesis. Public zone-PI examples also motivate reducing
fixed-gain cooling bursts; their claimed performance is not our evidence.

For a running cooling AHU, fixed inlet $T_a$, zero reheat, and the same learned
implicit model, solve for the flow reaching a chosen upper-band target:

$$
f^*=\left[
\frac{T+aT_o+bM+c-(1+a+b)T^*}
{\kappa c_a(T^*-T_a)}
\right]_+,\qquad T^*-T_a>0.25\ \mathrm K.
$$

The experiment changes only airflow in **pure cooling** groups: the AHU is
already running with a cooling command, no zone requests heat, and every reheat
valve is closed. Heating, mixed groups, AHU on/off decisions, supply temperature,
recirculation and shared water commands retain V4 behavior. All actual selected
commands feed the unchanged causal observer's delayed history.

Use $T^*=\max(L+0.1,U-m)$ with
$m=\operatorname{clip}(2\widehat{\mathrm{MAE}},0.1,0.5)$ kelvin. This is an
empirical uncertainty buffer, not a calibrated chance constraint. For a symmetric
Laplace error with scale $b$, the expected upper-band excess at margin $m$ is
$\mathbb E[(\varepsilon-m)_+]=(b/2)e^{-m/b}$; a two-MAE buffer is a simple
bounded surrogate motivated by that tradeoff. Actual residuals need not be
Laplace, and extra cooling increases energy/carbon cost. The exact weighted
score, not this surrogate, decides whether it helps.

Blend inverse flow with the existing delivered flow using the unchanged observer
warmup/residual weight. Apply design bounds, damper floors, minimum shared fan
pressure and the AHU capacity map. Inlet gaps without cooling authority fall
back to existing flow. Learned parameters, geometry and equipment priors remain
approximate; saturation can make the target unreachable.

First test inversion against forward dynamics and mature real-building action
bounds. Then compare V4 and this variant in matching complete public Phoenix and
Singapore runs on the same Kaggle image, keeping the repaired evaluation protocol
and seeds fixed. This research notebook produces no scored entry. A promising
result requires wider fresh transfer tests before any scored submission.
