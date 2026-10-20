# Thermal allocator: first controlled experiment

Earlier performance measurements used a diagnostic runner with an extra
state-mutating boiler read and are historical only. The physical derivations
and unit tests retain their stated scope. Current qualification must use
`non_mutating_v2`; see [experiment record](EXPERIMENTS.md).

This implements the first increment in [ALGORITHM_RESEARCH.md](ALGORITHM_RESEARCH.md),
using [MATHEMATICAL_FRAMEWORK.md](MATHEMATICAL_FRAMEWORK.md) Sections 1, 3, 6, and 7.
The controller is registered as `thermal_allocator`.

## Hypothesis and algorithm

For water-limited reheat, fixed water flow and temperatures give

$$
Q_i^{HVAC}=\rho_ac_af_i(T_{AHU}-T_i)
+\rho_wc_wq_i^w(T_w-T_{AHU}),\qquad
\frac{\partial Q_i^{HVAC}}{\partial f_i}<0\quad(T_{AHU}<T_i).
$$

The shipped controller can increase fan pressure and damper opening as heating
error grows. This experiment instead requests useful net heating, selects
airflow within estimated water/discharge capacity, and uses minimum pressure
for that airflow. Reheat valves compensate the incoming cold-air load as well
as the requested net heat.

Controller memory holds an exponentially filtered temperature trend. For zone
shortfall $e_i$, the first target is

$$
Q_i^{request}=\frac{e_i/6+\max(0,-\widehat{\Delta T_i})}{2\times10^{-5}}.
$$

The gain is an explicit physical-model prior, not a fitted hidden-building
parameter. The target aims to recover a deficit over roughly six control steps
while compensating an observed adverse trend. It is a surrogate feedback target;
this version does not solve the economic MPC optimization or estimate hidden
wall temperatures. Those remain later increments gated by measured results.

Heating allocation runs when heating shortfall dominates cooling shortfall and
incoming mixed air is colder than the zones. Cooling retains the inherited PID
control logic. The candidate also reads the actual zone setpoint sensors, while
V0 reconstructs bands from time features. `diagnose_policy --sensor-bands`
supports a baseline ablation to separate that difference from allocation.

## Interface, constraints and assumptions

- Information: public metadata, current observation, and past observed temperatures.
  No reward, simulator object, future weather, scenario identity, or private state
  is passed to the policy.
- Action indexing and affine normalization come from metadata. Pressure uses Pa
  for AHUs and bar for the water loop; zone temperatures are decoded from their
  normalization constants. AHU outside temperature and water temperature sensor
  values are raw kelvin in the current public observation contract.
- Public equipment priors: damper floor 0.1, running pressure floor 200 Pa,
  recirculation cap 0.7, design pressure 2000 Pa, AHU capacity 20 m3/s, discharge
  cap 308.15 K, full-valve coefficient 0.108 m3/hour/sqrt(bar), zero header
  resistance. Hidden equipment variations may invalidate these predictions.
- The first experiment commands a 350 K water setpoint and 20 bar pressure when
  its heating allocation is active. This favors capacity while the valves meter
  water. Energy/carbon tradeoffs are judged by actual evaluation, not assumed.
- Shared-water interactions, temperature lags, unweighted return-air estimates,
  sensor-band changes, and fixed response gain are explicit approximation risks.

## Validation and reproduction

The diagnostic runner checks action shape, dtype, finite values and bounds
before every simulator step. It records full observations/actions, scored
comfort/energy/carbon components, controller latency, and post-step physical
device probes. The probes are diagnostic recomputations, not exact integrated
interval energy measurements. They never enter controller observations.

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python -m smart_buildings.diagnose_policy \
  --scenario smart_buildings/training_scenarios/building_train_1/fairbanks_1 \
  --policy thermal_allocator --steps 8640 \
  --output results/allocator-v1/building_train_1/fairbanks_1/thermal_allocator
```

Use a new output directory for each completed run. Compare full episodes using
$J_s=-R_s/H_s$ and the same scenario/horizon. Short runs are contract/diagnostic
checks, not leaderboard estimates. Completed results and deployment decisions
will be recorded in an experiment report after the runs finish.
