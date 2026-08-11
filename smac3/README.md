# smac3 — SMAC3 hyperparameter optimisation for MCTS

Uses [SMAC3](https://github.com/automl/SMAC3) (Bayesian optimisation + racing) to
find strong MCTS search strategies. Each trial invokes the game binary's
`tune eval` subcommand to play a match between a candidate and a fixed
baseline, and reports the loss rate. The search space has two levels: a
top-level `family` choice (which `Strategy<G>` -- select/simulate/backprop/
final-action composition -- to run, e.g. `ucb1`, `ucb1_tuned`, `amaf_mast`,
`rave`, ...; see `mcts-tune`'s crate doc comment for the full 14-entry
catalog) and, within the chosen family, that family's own hyperparameters
(RAVE's schedule/`c`/epsilon, the UCB families' exploration constant, etc).
12 of the workspace's 16 games support this out of the box via the shared
`mcts-tune` crate (the other 4 -- `null`, `unit`, `shibumi`, `count` -- have
no real 2-player search to tune).

---

## Quick start

```bash
# 1. Build the game binary (one-time)
cargo build --release -p game-traffic-lights

# 2. Run from the project root (cwd must be where target/release/ lives)
uv run --project smac3/ smac3
```

This runs a 1000-trial optimisation over the full multi-family search space
(default `target.binary` is `game-traffic-lights`; point it at any of the
other 11 supported games' binaries with `--override target.binary=...`, or
use `bench smac3 --game <name>` from the Rust side, which sets it for you).
Results appear in `smac3_output/`.

> **Why `--project smac3/`?** The Rust project is managed by Cargo in the root;
> the Python/SMAC tooling lives in `smac3/` as its own uv project. The game
> binary path in the config is relative to the *current working directory*, so
> running from the project root makes `target/release/game-traffic-lights`
> resolve correctly.

---

## Customising a run

### Short test runs

Use `--override` to shrink the budget for quick smoke tests:

```bash
uv run --project smac3/ smac3 \
    --override optimizer.n_trials=10 \
    --override optimizer.deterministic=True \
    --override optimizer.n_workers=1
```

### All override keys

| Key | Type | Default | Description |
|---|---|---|---|
| `optimizer.n_trials` | int | 1000 | Number of configurations to evaluate |
| `optimizer.deterministic` | bool | false | Use one seed per trial (vs. multiple) |
| `optimizer.n_workers` | int | cpu//2 | Parallel workers |
| `optimizer.seed` | int | 42 | Random seed |
| `target.binary` | str | `target/release/game-traffic-lights` | Path to the game binary (relative to CWD) |
| `target.rounds` | int | 20 | Self-play rounds per trial, passed as `tune eval --rounds` |

### Full config file

Edit `smac3/config/default.yaml` or create your own:

```bash
uv run --project smac3/ smac3 --config my-search.yaml --verbose
```

The config file defines:

- **`parameters`** — the search space (float, int, categorical, constant)
- **`conditions`** — conditional activation (e.g. `c` is only active when
  `rave_ucb` is `ucb1` or `tuned`)
- **`optimizer`** — SMAC settings (budget, parallelism, seed)
- **`target`** — the game binary path (relative to CWD) and rounds per trial

---

## Adding a new parameter

1. Add an entry to `parameters:` in the YAML config.
2. If it should only be active when another param has a specific value, add a
   `conditions:` entry.
3. Make sure the game's `tuner()`/`tune_eval` (e.g.
   `games/traffic-lights/src/tuner.rs`) has a matching field on its params
   struct — active parameters are passed as keys in the `tune eval --config`
   JSON object, named exactly as in the YAML.

---

## Adding SMAC3 to another game

12 games (atarigo, bid_ttt, breakthrough, druid, gonnect, knightthrough, nim,
othello, tak, tanbo, traffic-lights, ttt) already support this via the shared
`mcts-tune` crate, which does the actual candidate-building/self-play work
generically over `G: Game` -- a game's `GameAdapter` impl only needs a few
lines forwarding to it:

```rust
fn tuner(&self) -> Option<TunerInfo> {
    Some(mcts_tune::strategy_tuner_info("strong", TUNE_EVAL_ROUNDS))
}

fn tune_eval(&self, params: Value, rounds: u32, seed: Option<u64>) -> Result<Value, HostError> {
    let outcome = mcts_tune::strategy_tune_eval(&params, rounds, seed, use_transpositions, build_strong)?;
    Ok(serde_json::json!({"cost": outcome.cost, "wins": outcome.wins, "losses": outcome.losses, "draws": outcome.draws}))
}
```

(`use_transpositions` should only be `true` for a game with a real
`Game::zobrist_hash` override -- see `mcts-tune`'s doc comment on
`strategy_tune_eval`.) `null`/`unit`/`shibumi` have no real search to tune at
all; `count` has a real search but is a 1-player puzzle
(`num_players() == 1`) wearing a `GameAdapter`, so a "candidate vs baseline"
self-play comparison doesn't make sense for it either -- see the note on
`CountAdapter` in `games/count/src/main.rs`. For a genuinely new game (a
17th game crate, or one
whose baseline preset doesn't fit the `impl Fn() -> Box<dyn Search<G>>`
shape `strategy_tune_eval`'s `baseline_build` parameter wants), see
`games/nim/src/main.rs` for the smallest reference wiring; `tuner()` and
`tune_eval()` on `GameAdapter` default to "unsupported" otherwise.

The YAML search space (`parameters:`/`conditions:`) is shared across every
game, not per-game -- it doesn't need editing to add a new game, only when
the family catalog or a family's own parameters change (see "Adding a new
parameter" above). `bench smac3 --game <name> ...` and the Bench UI's SMAC3
launch form work unchanged once `/api/bench/smac3/kinds` reflects the new
game's metadata.

## Output

SMAC writes results to `smac3_output/<run-name>/<seed>/`. The final incumbent
is printed at the end:

```
============================================================
Best config:  {'epsilon': 0.4, 'c': 0.32, 'rave': 1118, ...}
Best cost:    0.375000
Default cost: 0.425000
============================================================
```

Cost is the loss rate of the candidate vs. a fixed baseline over 20 rounds
(0.0 = always wins, 1.0 = always loses).

---

## Dependencies

All Python dependencies are managed by uv via `pyproject.toml`. Key packages:

| Package | Version pin | Why |
|---|---|---|
| `smac` | >=2.4.0 | Bayesian optimisation engine |
| `scikit-learn` | >=1.6.1, <1.9.0 | RF surrogate model (pinned below 1.9 which removed `DTYPE`) |
| `pyyaml` | >=6.0 | Config file parsing |
| `ConfigSpace` | >=1.0.0 | Hyperparameter search spaces |
