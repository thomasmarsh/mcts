# smac3 — SMAC3 hyperparameter optimisation for MCTS

Uses [SMAC3](https://github.com/automl/SMAC3) (Bayesian optimisation + racing) to
find strong hyperparameters for MCTS search strategies. Each trial invokes the
game binary's `tune eval` subcommand to play a match between the candidate
strategy and a fixed baseline, and reports the loss rate.

---

## Quick start

```bash
# 1. Build the game binary (one-time)
cargo build --release -p game-traffic-lights

# 2. Run from the project root (cwd must be where target/release/ lives)
uv run --project smac3/ smac3
```

This runs a 1000-trial optimisation using the default search space (RAVE/GRAVE
knobs for the TrafficLights game). Results appear in `smac3_output/`.

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
