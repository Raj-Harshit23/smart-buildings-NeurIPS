# Observer and heating-allocation ablation

**Evidence correction:** the original diagnostic runner added a state-mutating
boiler read. Historical figures below describe an instrumented system and cannot
qualify a normal-evaluation submission. Fresh validation requires protocol
`non_mutating_v2`; see [experiment correction](EXPERIMENTS.md).

Basis: [MATHEMATICAL_FRAMEWORK.md](MATHEMATICAL_FRAMEWORK.md), Sections 1,
3, 6–9; [DELAY_AWARE_ALLOCATOR.md](DELAY_AWARE_ALLOCATOR.md).

The six complete V2 local evaluations reduce mean loss by about 51% from V0,
but building 2 / Fairbanks remains 13.4% worse. Its comfort component improves
from 0.10061 to 0.08439 while energy rises from 0.03663 to 0.07204 and carbon
from 0.05879 to 0.12510. During intervals whose current occupancy observation is
positive, diagnostic reheat power rises from approximately 205 kW to 430 kW.
These device probes are post-step diagnostics, not integrated metered energy.

The first allocator uses a shared fixed thermal gain, forces maximum water
temperature when active, and adds ventilation while recovering heat deficits.
This can trade a modest comfort improvement for excess heating. A more accurate
temperature observer does not by itself correct that actuator decision.

## Controlled hypothesis

Retain V2's causal delayed RLS state observer, including all priors, warmup,
covariance bounds, residual weighting and correction clipping. Change only the
actuator law applied to that estimate:

$$
\widehat T_t=\mathcal O(o_{0:t},u_{0:t-1}),\qquad
u_t=\pi_{\mathrm{PIDWaterReset}}(o_t[T\leftarrow\widehat T_t],z_t).
$$

Use the observed zone setpoint bands, as in V2. This is a feedback heuristic,
not an optimizer of the exact weighted reward. The unchanged evaluation target
is $J=\operatorname{mean}_s(-R_s/8640)$ with comfort/energy/carbon weights
0.6/0.2/0.2. All observation normalization, action dimensions, native units and
actuator limits remain governed by metadata and the framework.

No building ID lookup, future weather, future temperature label, reward or
private simulator state enters the policy. The observer records the action
actually selected by PID for subsequent estimation. State is reset per rollout.

## Evaluation

Run the same six complete 8640-step scenarios as V0 and V2. Start the problematic
Fairbanks case alongside the three original pilot scenarios, then the remaining
two transfer cases. Compare every result with both V0 and V2. Require at least
5% mean improvement from V0, nonnegative mean improvement on the three transfer
cases, no case more than 10% worse than V0, and mean action latency below 10 ms
before a scored submission. Never select a policy using a hidden scenario label.

If removing allocation sacrifices important cold-climate performance, use the
component and trajectory evidence to motivate an adaptive heat-demand law.
Do not blend models merely to fit these six scenario names.
