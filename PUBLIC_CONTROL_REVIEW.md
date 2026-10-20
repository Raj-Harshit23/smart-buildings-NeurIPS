# Public zone-control review and an adaptive PI derivation

References: [MATHEMATICAL_FRAMEWORK.md](MATHEMATICAL_FRAMEWORK.md), Sections 1,
3, 6–9; [ALGORITHM_RESEARCH.md](ALGORITHM_RESEARCH.md), Sections 5–8.
This is a proposed next direction, not a new submitted controller.

## Sources actually inspected

Public Kaggle notebooks read on 2026-10-20:

1. [Zone PI K10, Danil Malyshev](https://www.kaggle.com/code/malyshevdanil/smart-buildings-probe-zone-pi-k10).
   Uses per-zone PI control toward the comfort-band centre, conditional
   integration, scheduled prestart, and an AHU cooling-run hold.
2. [Zone PI hold, mdmahfujulkarim](https://www.kaggle.com/code/mdmahfujulkarim/smart-buildings-zone-pi-hold).
   Uses a related PI/hold architecture, with attribution to the earlier work.
3. [Cross-building adaptive thermodynamic HVAC control, Avik Das](https://www.kaggle.com/code/avikdas567/cross-building-adaptive-thermodynamic-hvac-control).
   Includes economizer/pressure arguments and a soft worst-zone controller.

Their source was downloaded for inspection, not executed or copied into our
policy. Notebook metadata returned no license field. Preserve source attribution
and establish reuse terms before copying implementation. The mathematical ideas
below are derived independently.

The first notebook reports low local comfort cost on selected scenarios; that
is not a verified public leaderboard result. Some figures in the third notebook
are generated from synthetic random curves or hardcoded benchmark values, so
they are not runtime validation. Its narrative also describes a freezing lockout
that is not established by our local source: the verified mechanism is a
recirculation cap and an AHU that cannot heat its mixed inlet.

The first notebook decodes an AHU weather field using generic weather-normalizer
statistics. Our verified local AHU child field is raw kelvin, so that detail
cannot be transferred without checking the official SDK. Public-SDK compatibility
is being checked separately; hidden evaluator parity remains a distinct question.

## Why the controller can oscillate

Linearize a cooling-controlled zone around an operating point:

$$
x_{t+1}=\alpha x_t-\beta u_t+d_t,
\qquad 0\leq\alpha<1,\quad\beta>0.
$$

$x$ is temperature error, $u$ is an actuator coordinate increasing cooling, and
$d$ collects disturbance/model error. The verified local observation is delayed:
$y_t=x_{t-1}$. For proportional feedback $u_t=K_p y_t$:

$$
p(\lambda)=\lambda^2-\alpha\lambda+\beta K_p.
$$

For these signs, Schur stability requires $\beta K_p<1$. A controller using an
accurate current-state estimate instead has
$p(\lambda)=\lambda-\alpha+\beta K_p$ and requires
$\beta K_p<1+\alpha$. Compensation removes one delay-induced restriction,
but a single fixed gain can still be too aggressive in a small fast zone.
Saturation bounds commands while leaving persistent temperature cycling possible.

For damper command $d$ at fixed pressure and inlet, the implicit model gives

$$
\beta_d=-\frac{\partial T^+}{\partial d}
=\frac{\kappa c_a f_{design}\sqrt{p/p_{design}}(T^+-T_{ahu})}
       {1+a+b+\kappa c_af}.
$$

Thus sensitivity changes with geometry, learned thermal parameters, pressure,
inlet temperature and damper position. A universally fixed PI gain does not
account for these changes. During water-limited heating the sign of airflow's
thermal effect can reverse; a cooling damper law cannot simply be sign-flipped.

## Combining PI with causal estimation and physical allocation

Let the observer provide $\widehat x_t$, and update the integrator before issuing
the unsaturated command:

$$
I_{t+1}=I_t+\widehat x_t,\qquad
u_t=K_p\widehat x_t+K_iI_{t+1}.
$$

With exact estimation and the linearized plant:

$$
p(\lambda)=\lambda^2-
[1+\alpha-\beta(K_p+K_i)]\lambda+\alpha-\beta K_p.
$$

For desired real poles $r_1,r_2\in(0,1)$:

$$
K_p=\frac{\alpha-r_1r_2}{\beta},\qquad
K_i=\frac{(1-r_1)(1-r_2)}{\beta}.
$$

Choose $r_1r_2\leq\alpha$ if nonnegative proportional gains are required.
The point is to scale gains with identified response, rather than transplanting
the public notebook's numerical settings.

Jury conditions reduce, for nonnegative gains and $0\leq\alpha<1$, to

$$
K_i>0,\qquad \beta(2K_p+K_i)<2(1+\alpha).
$$

For credible bounds $\alpha\geq\alpha_{min}$ and $0<\beta\leq\beta_{max}$,
enforce the conservative condition

$$
\beta_{max}(2K_p+K_i)<2(1+\alpha_{min}).
$$

Our RLS covariance is not a calibrated confidence set, so it does not yet supply
such credible bounds. The inequality is a local design constraint, not a proof
of stability for the uncertain saturated hybrid building.

With the one-step measurement delay left uncompensated, the same PI convention
instead produces

$$
p_{delay}(\lambda)=\lambda^3-(1+\alpha)\lambda^2+
[\alpha+\beta(K_p+K_i)]\lambda-\beta K_p,
$$

so tuning a delay-free PI controller directly against stale observations changes
the characteristic equation. Approximate observation compensation must still
be checked under residual and timing errors.

## The actuator coordinate matters

Use signed requested **net thermal power** as the supervisory coordinate,
then allocate airflow/reheat using the physical equations, rather than treating
damper magnitude as interchangeable cooling and heating authority. The adaptive
inverse currently implements the heating half. For cooling without reheat,
at target $T^*$ and inlet $T_{ahu}<T^*$:

$$
f^*=\left[
\frac{T+aT_o+bM+c-(1+a+b)T^*}
{\kappa c_a(T^*-T_{ahu})}
\right]_+.
$$

Apply flow/design/ventilation limits, shared fan pressure and AHU capacity.
If $T_{ahu}\geq T^*$, this formula has no positive cooling authority and must
not be used with a clipped near-zero denominator. A different supply candidate
or off action is needed. Shared inlet and water constraints couple zone requests.

Conditional integration or back-calculation should use reconstructed achievable
power, so an unreachable request cannot accumulate unlimited integral demand.
Reconstruction uses public observations, metadata, past commands and explicit
equipment priors; actual private simulator heat cannot enter the policy.

## What must be measured next

Finish fresh `non_mutating_v2` baseline/adaptive-heating comparisons and official
gateway validation first. Diagnose remaining occupied-zone errors, switching,
power and transfer failures using correctly decoded occupancy. Test the smallest
change supported by those results: adaptive cooling inversion or zone integral
feedback, with the observer held fixed. Keep scheduled cooling holds or night
preconditioning separate until full-horizon comfort gains exceed their extra
energy/carbon cost. A 24-hour hold and a band-centre target are heuristics, not
solutions of the exact weighted long-horizon objective.
