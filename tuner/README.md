# MCTS game tuner

`tuner` is a foreground, reproducible strategy tuner for any executable that
implements the game-host `describe`, `compare validate`, and `compare eval`
protocol. It discovers and strictly validates the game's declared tuning
metadata, evaluates seeded ConfigSpace candidates against the schema default,
selects finalists on common tuning tasks, and validates them on a fresh common
task block.

Pass the target explicitly; the command never infers or searches for a game
binary:

```bash
uv run --project tuner tuner \
  --game-binary target/release/game-druid \
  --run-dir /tmp/mcts-tuner-druid \
  --seed 7 --cohort-size 3 --finalists 2 --tuning-pairs 1 \
  --validation-pairs 1 --tuning-max-iterations 16 \
  --validation-max-iterations 32 --production-max-iterations 10000
```

The same command shape works for Tic-Tac-Toe:

```bash
uv run --project tuner tuner \
  --game-binary target/release/game-ttt \
  --run-dir /tmp/mcts-tuner-ttt \
  --seed 7 --cohort-size 3 --finalists 2 --tuning-pairs 1 \
  --validation-pairs 1 --tuning-max-iterations 16 \
  --validation-max-iterations 32 --production-max-iterations 10000
```

The run directory must not already exist unless `--resume` is supplied. It contains:

- `manifest.json`: frozen executable, game description, tuning schema, default opponent,
  budgets, and disjoint task blocks;
- `evidence.jsonl`: append-only proposal, pair-atomic game, selection, and validation evidence;
- `report.json`: a replaceable read model rebuilt solely from the first two files.

`report.json` calls validation `production` only when
`--validation-max-iterations` exactly equals `--production-max-iterations`.
Otherwise it is a `mechanics_smoke`, regardless of pair count or elapsed time.

To continue an interrupted or recoverably failed comparison, repeat the same
scientific options with `--resume`:

```bash
uv run --project tuner tuner --game-binary target/release/game-druid \
  --run-dir /tmp/mcts-tuner-druid --resume \
  --seed 7 --cohort-size 3 --finalists 2 --tuning-pairs 1 \
  --validation-pairs 1 --tuning-max-iterations 16 \
  --validation-max-iterations 32 --production-max-iterations 10000
```

Resume verifies the manifest and full evidence log before appending, then
checks the selected executable bytes, `describe` response, game/schema
fingerprints, ConfigSpace version, frozen scientific options, and task plan.
The executable may move paths if its bytes and discovery metadata remain
identical. `--pair-timeout-seconds` is operational and may change.

Each completed comparison is one evidence line containing both seat-swapped
raw game records. A timeout, malformed result, non-zero comparison exit, or
operator interruption records no scientific result for that pair; an explicit
resume reruns it from both seats. Configuration failures are terminal, while
pair failures and interruptions are resumable. `report.json` is disposable:
resuming a completed run rebuilds it without evaluating a game.

This command does not support multiple opponents, non-default starts,
automatic retries, time budgets, or concurrent execution.
