# MCTS game tuner

`tuner` is a foreground, reproducible strategy tuner for executables that
implement the game-host `describe`, `compare validate`, and `compare eval`
protocol. It freezes an explicit deployment objective before creating an
artifact, then evaluates a fixed bootstrap/SMAC/random-reserve cohort on common task corpora.

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
  --tuning-pairs 6 --validation-pairs 6 --production-validation-pairs 12 \
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
elimination. The selected validation corpus is always a leading prefix of the
frozen production validation corpus.

The run directory contains three version-4 artifacts:

- `manifest.json` freezes the resolved objective/panel, full corpora and
  selected prefixes, fidelity axes, mixed proposal schedule, model dependency
  versions, and derived objective epoch.
- `evidence.jsonl` records append-only proposal, pair-atomic, contextual
  observation, selection, and completion evidence.
- `report.json` is a replaceable projection with proposal-search provenance, weighted held-out marginals,
  per-opponent matchup rows, matched finalist differences, and unresolved ties.

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
  --tuning-pairs 6 --validation-pairs 6 --production-validation-pairs 12 \
  --tuning-max-iterations 16 --validation-max-iterations 32 \
  --production-max-iterations 64
```

Resume validates the manifest and complete evidence log before append. It
rejects changed objective content/order/weights/configurations, task corpora,
prefixes, efforts, or epoch before evaluating another game. The objective and
binary paths may move when their resolved scientific identity is unchanged;
`--pair-timeout-seconds` remains operational. Resuming a completed run only
rebuilds `report.json`.
