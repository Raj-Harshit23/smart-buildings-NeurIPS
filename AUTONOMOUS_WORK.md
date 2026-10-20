# Authorized autonomous experiment scope

The repository owner authorized continued grounded research, implementation,
testing, validation, promising Kaggle submissions, and iteration while away.
Network use is limited to research, Kaggle and previously authorized GitHub work.
Commits use Harshit Raj's configured identity without AI co-author attribution.
Every model change must refer to [MATHEMATICAL_FRAMEWORK.md](MATHEMATICAL_FRAMEWORK.md)
and state its mechanism, available information, surrogate and relation to the
actual comfort/energy/carbon objective. Avoid arbitrary parameter sweeps and
submissions without supporting evidence.

## Actual competition status

- V0 entry **57407757** completed with public score **0.06404**; lower is better.
- V4 entry **57435234** was submitted on 2026-10-20 at 11:16 UTC and its public
  score is pending. Private notebook
  `harshitraj2123221/smart-buildings-adaptive-heating`, version 1, passed the
  official 8641-action Atlanta training validation with loss 0.02203293,
  11.5% better than V0 on that same training case.
- The corrected six-case V4 local comparison improved mean loss by 55.8% over
  V0, with 13.2% transfer improvement and worst regression 5.7%. These local
  numbers are distinct from a hidden leaderboard score.
- V5's paired cooling study improved the two hot-case mean by only 2.1%; it is
  held from submission. No broader V5 campaign is running.

[Experiment record](EXPERIMENTS.md) contains the evidence and limitations.
The bounded V4 score tracker is
`/workspace/smart-buildings-research/track_clean_v4.py`; its artifact/status is
`/workspace/smart-buildings-research/kaggle/clean-adaptive-v4/`. Its persisted
entry ID prevents a duplicate submission. The notebook slug was reconciled
against exact source after Kaggle derived the initial slug from its title.

## Current V6 experiment

The [permanent-prior prediction-error observer](REGULARIZED_OBSERVER_STUDY.md)
addresses lost prior information in unexcited model directions. Its actual
production forecasts exactly reproduce the four-case causal study. The full
suite passed **120 tests**. Forecast improvement does not establish control gain.

Controller source is frozen at **9633fda516c55dc1f3d082a6d1f1560828ac8825** in
`/workspace/smart-buildings-research/frozen-regularized-heating-v6`.
The four-hour local campaign runs at most four simultaneous full 8640-action
candidate episodes under `results/regularized-heating-v6/`. Completed V4
reference episodes are reused from `results/clean-adaptive-v4/`; they are not
rerun. The same six public scenarios are used. Require:

1. At least 5% pilot improvement over V4 and no pilot regression above 10%.
2. If the pilot qualifies, complete the transfer cases; require at least 5%
   combined improvement, nonnegative transfer improvement, no scenario more
   than 10% worse than V4 and mean action latency below 10 ms.
3. Before publishing a validation notebook, also require at least 5% combined
   improvement over V0, nonnegative transfer improvement and no scenario more
   than 10% worse than V0.
4. Require matching test/source/runner hashes, strict legal actions and exact
   source/export/recorded-action equality on two complete distinct trajectories.
5. Publish at most one new private offline notebook, then require complete
   official gateway validation, the required output file and at most 10%
   regression against V0's matched official Atlanta training loss. Check quota
   and persist an attempt marker before requesting one scored entry.

The separate eight-hour preparation/publishing/tracking pipeline starts at
`/workspace/smart-buildings-research/prepare_regularized_v6.py`, with subsequent
`launch_regularized_v6.py` and `track_regularized_v6.py`. Its artifact directory is
`/workspace/smart-buildings-research/kaggle/regularized-heating-v6/`.
A failed gate stops publication. Ambiguous publication/submission responses are
recorded and never blindly retried. Requested title/slug agreement and the
actual push response ref are checked before tracking. Local source snapshots
remain immutable while these jobs run.

## Evaluation validity

The original diagnostic runner read a boiler getter that advances tank
state on each read. All four older campaigns (`allocator-v1`, `lag-aware-v2`,
`lag-aware-pid-v3`, `adaptive-heating-v4`) are invalid for normal-evaluation or
submission qualification. Their records remain historical only; their
supervisors are stopped. No scored candidate used them. Any earlier claimed
transfer failure or improvement from those instrumented runs is superseded.

Only protocol **non_mutating_v2** can qualify a candidate. It reuses the water
observation and captures the reward response inside `env.step`. A direct
24-step comparison matched normal observations, actions and rewards exactly;
two meaningful regression tests guard this behavior. Summaries, exports and
campaign comparisons reject legacy results and record runner hashes.

The public SDK probe verified physical/reward source hashes, observation delay,
units and gateway output semantics. See
[verification](OFFICIAL_SDK_VERIFICATION.md). Complete hidden-evaluator parity and
the hidden scenario distribution remain unverified. Direct downloads remain
proxy-blocked; the private offline probe supplied public SDK source through logs.

These are finite processes that can continue while their execution environment
exists. They do not guarantee ongoing model reasoning or new experiment design
after the cloud session terminates.
