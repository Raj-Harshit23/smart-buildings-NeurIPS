# Authorized autonomous experiment scope

**Current evaluation requirement:** the original diagnostic runner was found to
advance boiler temperature through an extra getter read. All four historical
campaigns are invalidated for normal-evaluation/submission purposes. Keep their
records, but rerun qualification using protocol `non_mutating_v2`. A direct
24-step comparison passed exact observations, actions and rewards against an
ordinary rollout. The official V0 score remains 0.06404. New campaign comparisons
and exports reject legacy summaries. No candidate has been submitted.

The fresh campaign is `results/clean-adaptive-v4/`, dispatched by
`experiments/run_clean_validation.py`. It runs both Fairbanks baseline/candidate
pairs first alongside the other pilot pairs, at most four concurrent full
episodes, then delegates the remaining transfer pairs to the existing gate.
Its total local budget is four hours. Each receipt identifies the controller
and diagnostic-runner hashes. The prior 99-test suite passed, and the two added
harness regression tests passed separately; the combined receipt distinguishes
these test runs. Only fresh `non_mutating_v2` results can qualify a Kaggle export.

Fresh source is frozen at `cb2256a` in
`/workspace/smart-buildings-research/frozen-clean-adaptive-v4`. The eight-hour
pipeline `/workspace/smart-buildings-research/prepare_clean_v4.py` waits for
complete Fairbanks-1 and Phoenix candidate trajectories, then proves frozen
source/export/recorded-action equivalence. Its artifact is
`/workspace/smart-buildings-research/kaggle/clean-adaptive-v4`.
`launch_clean_v4.py` waits for the fresh absolute V0 gate before publishing one
new private offline notebook. `track_clean_v4.py` requires the output artifact,
at least 8640 validation steps, and at most 10% regression from V0's matched
official Atlanta validation loss before one scored entry. It checks allowance
and persists attempt markers before mutations; ambiguous responses are not
blindly retried. No candidate entry exists yet.

The separate public-SDK probe completed. Core physical/reward files match
exactly; environment ASTs match apart from controller registration. Runtime
delay and units are independently confirmed. See
[verification](OFFICIAL_SDK_VERIFICATION.md). Direct downloads remain
proxy-blocked; the authorized private Kaggle probe supplied source hashes and
the public interface source through its logs.

The repository owner authorized continued research, implementation, testing,
validation, promising Kaggle submissions, and iteration while they are away.
Network use is limited to research, Kaggle, and previously authorized GitHub work.
Commits use Harshit Raj's configured identity with no AI co-author attribution.

Every experiment must have a written physical or mathematical hypothesis and
refer to [MATHEMATICAL_FRAMEWORK.md](MATHEMATICAL_FRAMEWORK.md). Avoid arbitrary
parameter sweeps, repeated weak ideas, and submissions without supporting evidence.

Current bounded campaign:

1. Check V0 submission `57407757`; preserve its notebook/version and score record.
2. Compare the first allocator with fresh V0 runs at 8640 steps on three pilot
   scenarios. Qualify for wider validation only if mean loss improves at least 5%
   and no individual scenario regresses more than 10%.
3. If qualified, run three additional building/weather validation pairs and one
   sensor-band-only ablation. Require overall improvement, validation improvement,
   bounded regressions and acceptable controller latency before Kaggle validation.
4. Require passing tests, strict action checks, a pinned source artifact, and
   successful official gateway validation before a scored candidate submission.
   Check current Kaggle allowance. Persist a submission-attempt marker before
   requesting it; reconcile ambiguous responses instead of blindly resubmitting.
5. Record actual leaderboard scores separately from local results. If scoring
   remains pending, continue grounded local work without inventing leaderboard
   feedback or repeatedly submitting the same candidate.

`smart_buildings.run_allocator_campaign` is a resumable local supervisor with a
four-hour default budget and at most three concurrent child evaluations. It does
not access the network or submit entries. Logs, progress, trajectories, source
snapshots and its qualification decision are under `results/allocator-v1/`.
The existing V0 status tracker has its own bounded lifetime.

V1 validation is pinned to detached worktree
`/workspace/smart-buildings-research/frozen-allocator-v1` at commit `5f53c59`.
The exported notebook matched 17280 recorded actions across two complete
different-building trajectories. A separate eight-hour launcher at
`/workspace/smart-buildings-research/launch_validated_v1.py` waits for the full
local gate before creating one private, offline validation notebook. It then
requires official gateway completion before requesting at most one scored entry.
Attempt markers prevent blind retries after ambiguous network responses.

The delay-aware observer experiment runs separately in the working checkout;
it cannot silently change the frozen V1 validation or notebook source.

On 2026-10-20 the first scored entry completed at **0.06404**. Both bounded
allocator campaigns finished. V1 failed transfer. V2 qualified relative to V1,
but an absolute V0 comparison found a 13.4% regression on building 2 / Fairbanks
and a 1.3% regression in mean transfer loss; no scored V2 entry was requested.
See [completed comparisons](EXPERIMENTS.md).

The next controlled [observer/PID ablation](LAG_AWARE_PID.md) uses the same
six scenarios and the original V0 regression threshold. Its supervisor adopts
four initial full-rollout jobs, then evaluates the remaining transfer cases if
the pilot qualifies. The full suite passes 92 tests. No source changes to the
frozen V1/V2 models are included. The reusable offline exporter
`experiments/build_kaggle_candidate.py` requires a clean committed source,
matching test/source hashes, and exact exported/source/recorded-action equality
on two distinct complete trajectories. It has no network side effects.

The next [adaptive heat-demand experiment](ADAPTIVE_HEATING.md) uses the learned
implicit thermal inverse in place of the fixed heat gain. Its full suite passes
99 tests. Both Fairbanks cases are fresh simulator runs; cooling trajectories
are reused only after exact complete causal action replay and verified unchanged
local dynamics. `experiments/replay_equivalent_policy.py` records this provenance
explicitly. Its separate four-hour supervisor compares directly with V0 and
enforces the same transfer gate. Local results remain distinct from public scores.

These processes can continue while their execution environment exists. They do
not guarantee that model reasoning or new experiment design continues after the
cloud session terminates.
