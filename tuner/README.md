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
  --exclude-family meta_mcts \
  --tuning-pairs 6 --tuning-pair-budget 132 --validation-pair-budget 12 \
  --production-validation-pairs 12 \
  --tuning-max-iterations 16 --validation-max-iterations 32 \
  --production-max-iterations 64
```

`--evaluator-workers` is an operational setting and defaults to `1`. Each
evaluator runs one search thread, so the worker count cannot exceed the
available logical CPUs. Values above one execute an allocator-ordered batch of
seat-swapped pair subprocesses concurrently; starts may be batched, but terminal
evidence is committed in the same canonical order as sequential execution.
Worker count is not frozen in the manifest, so a run may resume with a different
count. Tuning pairs retry automatically after one recorded failure. After two
started, incomplete attempts for the same tuning pair, the frozen
`terminal-candidate-refill-v1` policy records `candidate_failed`, preserves the
candidate's factual work, removes it from the live cohort, and refills the
vacancy through the ordinary scheduled proposal source (or `random_reserve`
after the schedule is exhausted). Validation failures still require explicit
resume and never trigger finalist replacement. Interrupting a run cancels active
children, leaving any uncommitted starts censored; those starts count toward the
same two-attempt tuning limit on resume.

Panel weights produce a deterministic weighted-fair task order. Every task
names the exact panel opponent, canonical configuration fingerprint, seed, and
start stratum it uses. `--seed` controls proposal streams only;
`--task-seed` controls the disjoint tuning and held-out validation corpora.
`--exclude-family FAMILY` may be repeated to remove named families from candidate
proposals only. Its normalized set is frozen for resume; it does not change
schema-default or inline objective opponents.
All configured task counts must be complete panel weight cycles. `--tuning-pairs`
is the maximum tuning prefix: the tuner evaluates every accepted candidate on
each cumulative complete-cycle prefix before deepening the full cohort, with no
elimination. `--finalists` is both the retained-elite count and the final
shortlist count. The selected validation corpus is always a leading prefix of the
frozen production validation corpus.

At every complete non-final tuning prefix, the tuner records a deterministic
paired, stratum-aware `shadow_race_decided` screening disposition. Its practical
margin and nominal elimination threshold are frozen by `--shadow-practical-margin`
and `--shadow-elimination-threshold`. This is evidence only: every candidate
still reaches the maximum tuning prefix, and the nominal threshold has not earned
an active-pruning safety claim.

`report.json` includes a `candidate_lifecycle` projection of this policy,
terminal failures, and replacement lineage, plus a `shadow_elimination` audit
of those frozen decisions.
It labels each candidate against the same cohort's maximum tuning-prefix top
set, while calibration and stratum reversals compare against the exact early
boundary candidate recorded in the decision. `top_set_false_elimination_rate`
uses eligible unprotected top-set paths as its denominator; `trash_precision`
uses counterfactual eliminations and calls only candidates outside that same
tuning top set “trash.” Avoided work is factual suffix work after the first
unprotected elimination, including retries and partial failed attempts. The
section also reports fixed probability-bin calibration and Brier score only for
looks an active path would reach. It never uses held-out validation, is not an
anytime-valid safety guarantee, and does not enable active pruning.

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
  the evidence-derived compute ledger and shadow-elimination audit.

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
  --exclude-family meta_mcts \
  --tuning-pairs 6 --tuning-pair-budget 132 --validation-pair-budget 12 \
  --production-validation-pairs 12 \
  --tuning-max-iterations 16 --validation-max-iterations 32 \
  --production-max-iterations 64
```

Resume validates the manifest and complete evidence log before append. It
rejects changed objective content/order/weights/configurations, task corpora,
prefixes, efforts, budgets, or epoch before evaluating another game. The objective and
binary paths may move when their resolved scientific identity is unchanged;
`--pair-timeout-seconds` and `--evaluator-workers` remain operational. Resuming a completed run only
rebuilds `report.json`. A resumed run reproduces the uninterrupted run's
scientific projection, selection, and validation exactly; only the compute
ledger truthfully records the extra censored or retried attempts, including any
budget overrun they cause.
