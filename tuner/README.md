# tuner — Optuna + OpenSkill hyperparameter optimisation for MCTS

Uses [Optuna](https://optuna.org/) (TPE sampler) with OpenSkill-based
matchmaking (ladder-of-trash, **Thurstone-Mosteller Partial** model) to find
strong MCTS search strategies.
Each trial invokes the game binary's `tune eval` subcommand to play a
match between a candidate and a dynamically selected opponent, and
reports the loss rate. The search space has two levels: a top-level
`family` choice (which `Strategy<G>` -- select/simulate/backprop/
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
uv run --project tuner/ tuner
```

This runs a 1000-trial optimisation over the full multi-family search space
(default `target.binary` is `game-traffic-lights`; point it at any of the
other 11 supported games' binaries with `--override target.binary=...`, or
use `bench tuner --game <name>` from the Rust side, which sets it for you).
Results appear in `tuner_output/`.

> **Why `--project tuner/`?** The Rust project is managed by Cargo in the root;
> the Python/tuner tooling lives in `tuner/` as its own uv project. The game
> binary path in the config is relative to the *current working directory*, so
> running from the project root makes `target/release/game-traffic-lights`
> resolve correctly.

---

## Customising a run

### Short test runs

Use `--override` to shrink the budget for quick smoke tests:

```bash
uv run --project tuner/ tuner \
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

Edit `tuner/config/default.yaml` or create your own:

```bash
uv run --project tuner/ tuner --config my-search.yaml --verbose
```

The config file defines:

- **`optimizer`** — Optuna settings (budget, parallelism, seed)
- **`target`** — the game binary path (relative to CWD) and rounds per trial

The search space itself (`parameters`/`conditions` — float, int, categorical,
constant; conditional activation like `c` only being active when `rave_ucb`
is `ucb1` or `tuned`) is **not** in the YAML. It's queried at launch time from
`target.binary`'s own `tune describe` subcommand (`SearchConfig.
parameters_from_binary`), so it can never drift out of sync with what the
binary actually accepts.

---

## Adding a new parameter

1. Add the field to the game's `tuner()`/`tune_eval` (e.g.
   `games/traffic-lights/src/tuner.rs`) — this is the only place the search
   space is declared. If it should only be active when another param has a
   specific value, add it to the `TunerInfo.conditions` that same `tuner()`
   builds.
2. `<binary> tune describe` picks it up automatically; no YAML edit needed.

---

## Adding tuner support to another game

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
parameter" above). `bench tuner --game <name> ...` and the Bench UI's tuner
launch form work unchanged once `/api/bench/tuner/kinds` reflects the new
game's metadata.

## Output

The tuner writes results to `tuner_output/<run-name>/<seed>/`. The final
incumbent is printed at the end:

```
============================================================
Best config:  {'epsilon': 0.4, 'c': 0.32, 'rave': 1118, ...}
Best score:   5.123
============================================================
```

Score is the OpenSkill mu - 3*sigma (Thurstone-Mosteller Partial model)
estimate of the candidate against the opponent pool (higher is better).

---

## Dependencies

All Python dependencies are managed by uv via `pyproject.toml`. Key packages:

| Package | Version pin | Why |
|---|---|---|
| `optuna` | >=4.9.0 | Bayesian optimisation engine |
| `openskill` | >=6.2.0 | OpenSkill rating (Thurstone-Mosteller Partial) for matchmaking |
| `pyyaml` | >=6.0 | Config file parsing |