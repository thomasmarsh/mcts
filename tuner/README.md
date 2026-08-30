# MCTS Druid tuner

`tuner` is a foreground, Druid-only strategy tuner. It asks the built
`target/release/game-druid` binary for its schema, evaluates ConfigSpace's
seeded default/random candidates against its frozen default strategy, selects
finalists using common tuning tasks, and validates those finalists on a fresh
common task block.

```bash
uv run --project tuner tuner --run-dir /tmp/mcts-tuner-druid \
  --seed 7 --cohort-size 3 --finalists 2 --tuning-pairs 1 \
  --validation-pairs 1 --tuning-max-iterations 16 \
  --validation-max-iterations 32 --production-max-iterations 10000
```

The run directory must not already exist. It contains:

- `manifest.json`: frozen binary/schema/opponent, budgets, and disjoint task blocks;
- `evidence.jsonl`: append-only proposal, game, selection, and validation evidence;
- `report.json`: a replaceable read model rebuilt solely from the first two files.

`report.json` calls validation `production` only when
`--validation-max-iterations` exactly equals `--production-max-iterations`.
Otherwise it is a `mechanics_smoke`, regardless of pair count or elapsed time.

This command does not support other games, multiple opponents, non-default
starts, resume, retries, time budgets, or concurrent execution.
