# Public competition SDK verification

Evidence: private offline Kaggle diagnostic
`harshitraj2123221/smart-buildings-sdk-parity-probe`, versions 2 and 3,
2026-10-20. [Machine-readable record](experiments/public_sdk_probe.json).
The diagnostic reads attached public data; it is not a scored submission and
does not establish the hidden scenario distribution.

Five SDK files match local SHA256 hashes byte-for-byte: boiler, VAV, flexible
floor-plan simulator, finite-difference solver and weighted reward. Base/hybrid
environment differences are comments only; their full Python ASTs match. The
evaluator AST matches after excluding policy imports/dispatch, where we added
controllers. The public gateway reuses this evaluator's loader/metadata builder.
It documents the private occupancy-normalizer assignment as a retained typo:
the actual public normalizer stays zero, making occupancy a raw count here.

| Decision/step | Observed zone temperature | Current physical temperature |
| --- | ---: | ---: |
| Reset | 294.0000 K | 294.0000 K |
| After first cooling action | 294.0000 K | 289.4473 K |
| After second cooling action | 289.4473 K | 288.3456 K |

AHU outside temperature was raw 275.65 K and water temperature raw 340 K.
Zone normalization was mean 295 K, standard deviation 5 K. Native air/water
pressure ranges were 0–2000 Pa and 0–20 bar, confirming the unit assumptions.

Public `kaggle_evaluation/gateway.py` defines one output row per scenario:

$$
\texttt{submission.parquet}:
\quad\texttt{row\_id}:\mathrm{String},\quad
\texttt{reward}:\mathrm{Float32},\qquad
\texttt{reward}_s=R_s/H_s.
$$

$H_s$ is the actual executed count. The policy supplies actions and cannot choose
these tabular values directly. The gateway discovers `evaluation_scenarios`,
with no public-training fallback. The first construction/turn allowance is
900 seconds; subsequent turn allowance is 60 seconds. These request timeouts
do not establish an overall notebook/competition compute limit.

The environment checks termination before incrementing its zero-based counter.
Configured length 8640 therefore terminates on action 8641; the driver counts
and divides by 8641. This explains the V0 log. Controlled local comparisons
explicitly use the same 8640 actions for both policies and report that horizon.

Gateway comments claim rewards lie in [-1,0], but matching reward code has an
unbounded comfort penalty and no such clipping. The implemented equations in
[MATHEMATICAL_FRAMEWORK.md](MATHEMATICAL_FRAMEWORK.md) take precedence.

The matching boiler has the state-mutating sensor getter. Repaired diagnostics
reuse the returned observation and capture the response from `env.step`.
Historical extra-read trajectories cannot qualify an export. Public compatibility
and gateway training validation do not reveal hidden geometries, climates,
occupancy allocations, or test revisions. Public scores must be measured.

A scored candidate requires the fresh six-case gate, frozen source/test/export
proof, full official gateway completion, and no greater than 10% regression on
the same official Atlanta training case used by V0.
