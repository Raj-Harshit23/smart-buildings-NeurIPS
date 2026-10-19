# Authorized autonomous experiment scope

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

These processes can continue while their execution environment exists. They do
not guarantee that model reasoning or new experiment design continues after the
cloud session terminates.
