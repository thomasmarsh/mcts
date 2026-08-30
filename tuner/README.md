# MCTS game tuner

`tuner` is a foreground, reproducible strategy tuner for executables that
implement the game-host `describe`, `compare validate`, and `compare eval`
protocol. It freezes an explicit deployment objective before creating an
artifact, then runs as many complete retained-elite cohorts as the declared
evaluation budget funds. The first cohort uses the bootstrap/SMAC/random-reserve
schedule; at each completed-cohort boundary the top `--finalists` candidates are
retained as elites and the next challenger cohort (filled from the frozen
challenger schedule) starts only when the remaining tuning budget can fund all
of its planned new pairs. When another whole cohort does not fit, the latest
cohort's finalists receive held-out validation.

An objective is strict JSON containing the schema default and one or more raw,
inline historical opponent configurations. The checked-in Druid deployment
objective is `tuner/objectives/druid-reference-v1.json`; Python never resolves
named Rust presets at runtime.

```bash
uv run --project tuner tuner \
  --game-binary target/release/game-druid \
  --objective-file tuner/objectives/druid-reference-v1.json \
  --run-dir /tmp/mcts-tuner-druid \
  --seed 7 --task-seed 11 --cohort-size 8 --finalists 2 \
  --bootstrap-candidates 3 --random-reserve-candidates 2 \
  --tuning-pairs 6 --tuning-pair-budget 132 --validation-pair-budget 12 \
  --production-validation-pairs 12 \
  --tuning-max-iterations 16 --validation-max-iterations 32 \
  --production-max-iterations 64
```

Panel weights produce a deterministic weighted-fair task order. Every task
names the exact panel opponent, canonical configuration fingerprint, seed, and
start stratum it uses. `--seed` controls proposal streams only;
`--task-seed` controls the disjoint tuning and held-out validation corpora.
All configured task counts must be complete panel weight cycles. `--tuning-pairs`
is the maximum tuning prefix: the tuner evaluates every accepted candidate on
each cumulative complete-cycle prefix before deepening the full cohort, with no
elimination. `--finalists` is both the retained-elite count and the final
shortlist count. The selected validation corpus is always a leading prefix of the
frozen production validation corpus.

`--tuning-pair-budget` and `--validation-pair-budget` are required total
budgets over pair attempts, frozen in the manifest under the
`safe-boundary-pair-attempts-v1` policy. The initial cohort always runs; a later
challenger cohort is admitted at a completed-cohort boundary only when the
remaining tuning budget (counting every started pair attempt, including ones
that later fail or are interrupted) covers all of its planned new pairs. The
validation budget divides evenly across the finalists and derives one common
held-out prefix (`budget / finalists` pairs each, at least one complete panel
cycle, never longer than `--production-validation-pairs`). The number of
cohorts is an output of the budget, not a configuration.

The budgets are soft caps at scientifically safe boundaries: the tuner never
stops between the two seats of a pair or inside an admitted cohort, so a retry
after a failure or interruption may push actual attempts over the declared cap.
The report's `compute` section accounts for this truthfully — per-phase pair
attempts, completed pairs, failed and censored attempts, physical games,
actual search iterations, recorded game wall time, and unspent/overrun pair
attempts relative to the frozen budgets.

The run directory contains three version-4 artifacts:

- `manifest.json` freezes the resolved objective/panel, full corpora and
  selected prefixes, fidelity axes, mixed proposal schedule, model dependency
  versions, total compute budgets, and derived objective epoch.
- `evidence.jsonl` records append-only proposal, pair-atomic, contextual
  observation, selection, and completion evidence.
- `report.json` is a replaceable projection with proposal-search provenance, weighted held-out marginals,
  per-opponent matchup rows, matched finalist differences, unresolved ties, and
  the evidence-derived compute ledger.

Validation is `production` only when both axes reach their declared target:
the selected held-out validation prefix is the complete production corpus and
its search effort equals `--production-max-iterations`. Every other result is
`mechanics_smoke`, with the lower axis or axes named in the report.

Resume uses the same scientific options and objective file:

```bash
uv run --project tuner tuner \
  --game-binary target/release/game-druid \
  --objective-file tuner/objectives/druid-reference-v1.json \
  --run-dir /tmp/mcts-tuner-druid --resume \
  --seed 7 --task-seed 11 --cohort-size 8 --finalists 2 \
  --bootstrap-candidates 3 --random-reserve-candidates 2 \
  --tuning-pairs 6 --tuning-pair-budget 132 --validation-pair-budget 12 \
  --production-validation-pairs 12 \
  --tuning-max-iterations 16 --validation-max-iterations 32 \
  --production-max-iterations 64
```

Resume validates the manifest and complete evidence log before append. It
rejects changed objective content/order/weights/configurations, task corpora,
prefixes, efforts, budgets, or epoch before evaluating another game. The objective and
binary paths may move when their resolved scientific identity is unchanged;
`--pair-timeout-seconds` remains operational. Resuming a completed run only
rebuilds `report.json`. A resumed run reproduces the uninterrupted run's
scientific projection, selection, and validation exactly; only the compute
ledger truthfully records the extra censored or retried attempts, including any
budget overrun they cause.
