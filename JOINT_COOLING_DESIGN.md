# Joint supply-temperature, airflow and dispatch design

Basis: [MATHEMATICAL_FRAMEWORK.md](MATHEMATICAL_FRAMEWORK.md), Sections 1–4,
6–9, and the verified public AHU/VAV implementation. This is a derived next
research direction, not a submitted controller or a measured improvement.
The [V5 study](ADAPTIVE_COOLING.md) changed airflow while retaining PID supply
temperature and dispatch; its two-case mean gain was only 2.1%.

## Why the remaining coordinates matter

The inherited PID assigns supply temperature and pressure from the same scalar
effort, and switches an AHU off when its mean temperature lies in band.
In corrected Phoenix data, off/pure-cooling/hot-zone decisions contain 31.3%
of comfort loss. This is an association, not the benefit of switching on.
Changing airflow alone leaves both mechanisms untouched. Better estimation in
V6 may already change these decisions, so measure V6 before launching another
full campaign.

## Conditional supply-temperature dominance

For one AHU, no reheat and fixed current model state, write

$$
D_i=1+a_i+b_i,\quad
T_i^0=\frac{T_i+a_iT_o+b_iM_i+c_i}{D_i},\quad
\gamma_i=\frac{\kappa_i c_a}{D_i},\quad
T_i^+(f_i,s)=\frac{T_i^0+\gamma_i f_i s}{1+\gamma_i f_i}.
$$

Here $s$ is delivered supply temperature and $c_a=\rho_{air}c_{p,air}$;
the actual supply is $\min(T_{mix},\tau)$, rather than always the command $\tau$.
For a fixed target $T_i^*<T_i^0$ and $s<T_i^*$, the required flow is

$$
f_i(s)=\frac{q_i}{c_a(T_i^*-s)},\qquad
q_i=\frac{D_i(T_i^0-T_i^*)}{\kappa_i}>0.
$$

Suppose these flows are feasible, minimum damper floors do not alter them,
the total-flow cap is inactive, recirculation $\rho$ is fixed, and
$T_{mix}\geq\max_i T_i^*$. With public-model design flows $\bar f_i$:

$$
p(s)=\max\left(p_{min},p_{design}\max_i(f_i(s)/\bar f_i)^2\right),\quad
F(s)=\sum_i f_i(s),
$$

$$
P_{fan}(s)=\frac{(2-\rho)p(s)F(s)}{\eta},\qquad
P_{cool}(s)=\frac{c_aF(s)(T_{mix}-s)}{COP}.
$$

Both $f_i(s)$ and $p(s)$ are nondecreasing in $s$. Also

$$
P_{cool}(s)=\sum_i\frac{q_i}{COP}
\frac{T_{mix}-s}{T_i^*-s},\qquad
\frac{dP_{cool}}{ds}=\sum_i\frac{q_i}{COP}
\frac{T_{mix}-T_i^*}{(T_i^*-s)^2}\geq0.
$$

Therefore colder supply weakly reduces **both** fan and cooling electricity
while reaching exactly the same targets under these assumptions. It also
reduces both electricity cost and carbon when their coefficients are positive.
This result depends on the simulator's constant COP, sensible-only cooling,
flow-dependent outside-air intake and fixed mixing temperature. It is not a
general recommendation for real HVAC equipment or an episode-level proof.

## The exact obstruction to commanding the coldest supply everywhere

Running equipment has

$$
f_i\geq0.1\bar f_i\sqrt{p/p_{design}},\qquad p\geq200\ \mathrm{Pa}.
$$

A room with no cooling demand still receives this flow. If $T_i^0$ is near its
lower band, colder supply can create a new heating violation. Once the pressure
floor binds, lowering $s$ no longer reduces that unwanted airflow. Shared
pressure, total-flow saturation, simultaneous heating, and changes in return
temperature also break the unconstrained fixed-target derivation. If
$T_{mix}<T_i^*$, the cooling-electricity derivative can reverse sign; the
economizer branch must be evaluated separately.

Thus the next useful optimization couples supply temperature with actual
delivered flows and binary dispatch. It cannot be justified by merely changing
the supply command to its minimum or extending the V5 inverse to every state.

## A structured, bounded optimization to investigate

For fixed $p,s,\rho$ and a running AHU, ignore cap rescaling initially and use

$$
f_i\in[0.1\bar f_i\sqrt{p/p_{design}},\bar f_i\sqrt{p/p_{design}}],\quad
z_i=\frac1{1+\gamma_i f_i},\quad
T_i^+=s+(T_i^0-s)z_i.
$$

For an electrical shadow price $\lambda_e\geq0$, define

$$
A=\lambda_e\left[\frac{(2-\rho)p}{\eta}
          +\frac{c_a(T_{mix}-s)_+}{COP}\right],\quad
L_i(z_i)=0.6w_i\phi(d_i(T_i^+))
          +\frac{A}{\gamma_i}(z_i^{-1}-1).
$$

$d_i$ denotes the band distance in framework equation (5); $\bar f_i$ is
design airflow. Equal $w_i$ or area-based $w_i$ would be
explicit occupancy surrogates: the legal aggregate occupancy signal does not
identify current per-room reward weights. Likewise geometry does not prove
one actuator corresponds to one scored room. If the scenario normalization or
current tariff is unavailable to the policy, $\lambda_e$ is also a surrogate
coefficient. No exact reward claim follows.

On each of the five temperature intervals separated by $L-5,L,U,U+5$, comfort
is a convex quadratic, affine or zero function of $z_i$; $z_i^{-1}$ is convex
for $z_i>0$. Each zone subproblem is therefore convex **within a branch**.
Enumerating its valid branch minima and endpoints finds its scalar global
minimum for fixed AHU coordinates when the total-flow cap is inactive. The
comfort derivative drop at a 5 K miss prevents claiming global convexity across
branches. A shared flow cap couples the choices; a Lagrange multiplier can
help, but a zero duality gap is not guaranteed for their nonconvex union.

An eventual controller can compare a small physically chosen set of delivered
supply, pressure and economizer cases, their optimized zone actions, the
inherited action and the off action. This is bounded candidate optimization,
not a global mixed-integer MPC solution. Include inherited actions as a fallback,
use the delayed causal observer, model uncertainty and actual actuator floors,
and record candidate gain versus uncertainty before giving it control authority.

This one-step objective still omits future occupancy, thermal mass consequences
and switching value. A short causal rollout with persistence disturbances can
estimate those effects, but is another approximation to the undiscounted
episode objective. Fresh paired full-horizon control tests, transfer limits,
runtime checks and official validation remain necessary.
