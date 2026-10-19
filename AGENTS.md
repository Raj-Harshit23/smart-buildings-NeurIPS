# Repository instructions

## Required reference for model work

Before designing or changing any policy, controller, model architecture, training
objective, or evaluation procedure, read [MATHEMATICAL_FRAMEWORK.md](MATHEMATICAL_FRAMEWORK.md).
Use it as the mathematical basis for the work and cite the relevant sections in
model design notes and change descriptions.

- State the proposed mechanism, information available to the policy, and relation
  between any training surrogate and the actual comfort/energy/carbon objective.
- Respect causality, metadata-dependent dimensions and indexing, field-specific
  observation normalization, native units, and effective actuator constraints.
- Compare policies using matching horizons and scenarios. Distinguish short
  validation, local evaluation, and actual Kaggle leaderboard results.
- Follow the baseline-first experiment direction: obtain a scored basic policy,
  diagnose failures, and use evidence to choose the next model change.
- Treat the framework's external-verification limits as unresolved until checked.
  If source code or verified competition requirements change, reconcile and update
  the framework with evidence before relying on a changed assumption.
- Validate action shapes, dtypes, finite values, bounds, and adaptation to different
  buildings when changing policy code. Preserve the scoring definition during
  comparisons; identify any surrogate loss explicitly.

## Commit attribution

Use the repository owner's configured Git identity: Harshit Raj
(`harshitraj969@gmail.com`). Do not add Codex, OpenAI, or other AI assistant
co-author trailers or signoffs to commits. Preserve existing upstream attribution,
licenses, and notices.
