# Bench and tuner Tuning Architecture

> **Superseded.** This document describes the legacy Optuna/OpenSkill tuning
> stack (`/api/bench/tuner/sessions/*`, the DuckDB `tuning_*` tables, the
> `POST /api/bench/launch` tuner branch). That stack has been removed. The
> version-4 foreground tuner reconnects to the bench server through
> `/api/bench/tuner/runs` (detached launch/stop) with its run directory as the
> sole scientific authority; a rebuildable query projection and read-only API
> replace the DuckDB read path. This file is kept only for historical context
> until the new architecture is written up.

This document describes the system built around tuner: how a tuning request moves through the TypeScript UI, Rust server and launcher, Python optimizer, game binary, append-only logs, DuckDB ingestion, and back to the UI. It intentionally does not explain Bayesian optimization or the tuner's internal algorithms.

The most important architectural distinction is this:

> A tuning session is one logical run, but it may be implemented by several physical processes and storage rows.

A baseline ladder exposes that distinction. Each rung needs a new tuner `Scenario`, output directory, process, log, and `runs` row, yet the operator should see one continuous experiment with one trial budget, one history, and visible baseline boundaries. Most serious regressions have come from allowing the physical representation to leak through that logical abstraction.

## Design philosophy

The surrounding system follows a few principles.

### Rust owns orchestration and durable operational state

The server launches and stops processes, assigns run IDs, records launch configuration, ingests logs, serves the API, and advances ladders. It is the only process that opens `bench-runs/bench.duckdb` read-write. Neither Python nor the browser writes the database.

Rust does not attempt to reconstruct the tuner's optimizer state. Resumption uses the tuner's saved runhistory, and incumbent selection comes from tuner itself.

### Python is an adapter around tuner

The `tuner` package translates repository configuration into a tuner `Scenario`, obtains the search space from a game binary, invokes that binary for evaluations, and emits stable JSONL events. It does not own run discovery, UI state, ladder policy, or the DuckDB schema.

### Game binaries own legal tuning configurations and evaluation

The search space is reported by `<game-binary> tune describe`. The Python YAML is not a second declaration of the space. A trial is executed through `<game-binary> tune eval`, which builds the candidate and baseline searches and plays the games. This keeps parameter validity and search construction next to the Rust implementations they configure.

### Logs are the process boundary

Detached jobs communicate results by appending JSONL. The launcher redirects structured stdout to `log.jsonl`, diagnostics to `stdout.log`, and optional per-ply observations to `moves.jsonl`. The ingest loop converts those append-only records into queryable tables using byte cursors.

This makes process execution recoverable and inspectable. The database is a projection of durable files, not the only copy of a run's output.

### The UI is a projection, not an orchestrator

The browser calls typed server APIs and holds transient presentation state. Reducers own polling and effects; components render state and dispatch actions. Components must not reproduce server policy or infer ladder transitions from chart data.

## The end-to-end path

```text
Launch form
    |
    v
POST /api/bench/launch
    |
    v
Rust command builder and detached launcher
    |
    +--> registry.log          process lifecycle
    +--> log.jsonl             trials and incumbents
    +--> stdout.log            diagnostics
    +--> moves.jsonl           optional game traces
    |
    v
bench tuner -> Python tuner_cli -> game binary tune describe/eval
    |
    v
Rust ingest loop -> DuckDB
    |
    v
/api/bench/* -> typed TS client -> reducer polling -> UI
```

### 1. Launch configuration

The launch form produces a free-form JSON configuration stored on the `runs` row. For tuner this normally includes:

- ordered `overrides`, such as `optimizer.n_trials=100` and `target.baselines=['flat_mc']`;
- optional `game_config`;
- optional `baseline_configs`, keyed raw strategy configurations;
- `baseline_settings`, which records the resolved baseline parameters for display and comparison.

Overrides are deliberately ordered. Python parses them into a dictionary, so the final occurrence of a dotted key wins.

The server generates the physical `run_id` before spawning. The same ID is passed to Python, stored in DuckDB, written to the registry, and used as the `bench-runs/<run_id>/` directory. Modern continuation is owned by the logical tuning session and its command journal.

### 2. Process launch and lifecycle

`mcts-bench/src/launch.rs` creates a run directory and spawns the command in its own Unix process group. Stopping a run signals the entire group so Python, workers, and game subprocesses are not orphaned.

The master `registry.log` contains start and stop events. A background reaper waits for each child and records its real exit code. The ingest loop also reconciles liveness after server restarts, but that is a fallback; it cannot recover a precise exit code when the original reaper disappeared.

Run states have operational meaning:

- `running`: the physical process is expected to be alive and its logs are still ingestible;
- `completed`: the process exited naturally; a nonzero exit code still disqualifies ladder advancement;
- `stopped`: an operator or ladder transition deliberately ended it;
- `crashed`: launch or liveness reconciliation determined abnormal termination.

An automatic transition may stop a healthy parent. That physical row being `stopped` does not mean the logical ladder stopped: its newest child determines the logical run's current status.

### 3. Python scenario construction

`tuner/src/tuner_cli/__main__.py` loads YAML defaults, applies ordered command-line overrides, and asks the target game binary for its parameters, conditions, and advertised named baselines through `tune describe`.

A run must explicitly select at least one baseline. Scenario instances are the union of:

- `target.baselines`: named game presets;
- `target.baseline_configs`: IDs backed by raw discovered strategy parameters.

`target.py` decides how each instance reaches `tune eval`:

- discovered configurations and the repository's `flat_mc`/`random` floor baselines use `--baseline-config <json>`;
- actual game presets use `--baseline <name>`.

That distinction is semantic. Passing a floor baseline as though it were a named game preset turns evaluation errors into apparent cost `1.0` results.

The Python target function invokes one game-binary subprocess per tuner evaluation and returns the trailing JSON `cost`. Timeouts, nonzero exits, and missing cost output are scored as `1.0` and diagnosed in `stdout.log`.

### 4. Rust evaluation

The game adapter exposes `tuner()` metadata and `tune_eval()`. Shared construction and self-play live in `mcts-tune`.

Each evaluation builds a candidate from the sampled configuration, builds a fresh baseline search for every game, and plays both move orders for every round. The emitted cost is the candidate's losses divided by `2 * rounds`. The bench system treats this cost as an opaque optimizer metric except when applying a configured ladder saturation threshold.

Game setup belongs to `game_config`; optimizer configuration belongs to the sampled parameter object. They must remain separate so every trial in a run uses the same board/rules while tuner changes only the search strategy.

### 5. Structured output and ingestion

Python callbacks emit two important records:

- `trial`: one completed tuner evaluation, including configuration, seed, cost, and baseline instance;
- `incumbent`: the tuner's current incumbent configuration and aggregated cost.

The incumbent must come from `smbo.intensifier.get_incumbent()` and `runhistory.get_cost()`. It must never be reconstructed as the minimum raw trial cost: trials against different instances or seeds are not directly interchangeable, while the intensifier owns their aggregation.

The ingest loop runs periodically and processes:

1. `registry.log` into `runs`;
2. each running run's `log.jsonl` into `trials`, `match_results`, and `incumbents`;
3. `moves.jsonl` into `game_moves`;
4. process liveness reconciliation.

`_ingest_cursor` stores a byte offset per file. Inserts are idempotent by their natural keys. Incumbents are upserted so there is one current incumbent per physical run.

## Resume semantics

the tuner's normal continuation path requires an identical scenario and may prompt interactively when settings change. The repository therefore creates a fresh scenario and explicitly loads the parent's `runhistory.json` through `tuner/src/tuner_cli/resume.py`.

For an ordinary resume with the same baseline instances, the saved runhistory is merged into the fresh facade. A ladder transition changes the objective: costs measured against the old opponent are neither valid training data for the new opponent nor legal runhistory entries for a scenario whose instance set contains only the new baseline. A new rung therefore starts with an empty runhistory and reduces its physical `optimizer.n_trials` by the number of completed trials in the parent history. In-flight entries saved as `RUNNING` during the transition do not consume the logical budget because they produced no sample.

`optimizer.n_trials` is a total logical budget, not a per-process or per-rung allocation. If a 100-trial run changes baseline after 28 completed trials, the launch configuration still carries the logical total of 100, while the Python adapter gives the fresh child scenario a physical budget of 72. Increasing the logical value at each rung silently expands the experiment.

An explicit ordinary Resume action is different: it intentionally accepts a new larger total budget from the operator. Baseline advancement, manual or automatic, preserves the existing total unless the caller explicitly supplies a replacement.

The parent must be stopped and reaped before the child loads its runhistory. Reading while Python is still flushing risks a torn or incomplete file.

## Ladder semantics

A ladder changes the opponent when the current incumbent has become sufficiently strong against the current baseline.

The durable ladder policy lives in the launch configuration:

```json
{
  "ladder": {
    "max_rungs": 5,
    "saturation_threshold": 0.15
  },
  "ladder_root": "<root run id>"
}
```

The background driver periodically reads physical runs, trial counts, and incumbents. A rung is eligible when:

- it is running or naturally completed;
- it has a valid ladder block and root;
- it has no child already;
- the chain has not reached `max_rungs`;
- it has an incumbent;
- incumbent cost is at or below `saturation_threshold`.

The threshold is an immediate transition condition. It is not deferred until the physical run exhausts its budget. For a running rung the driver stops and reaps the process, then launches a child resumed from that parent.

The child faces only the promoted incumbent. `baseline_configs` is replaced, not accumulated, and a trailing `target.baselines=[]` clears any inherited named floor baseline. Accumulating old opponents changes the meaning of cost and defeats the curriculum's “always face the current incumbent” model.

Each child stores the promoted configuration in both `baseline_configs` for execution and `baseline_settings` for faithful UI comparison. The chain endpoint is the only active physical process.

Manual “Use best as new baseline” uses the same stop, resume, and baseline-replacement semantics. It may create a ladder chain retroactively for a plain tuner run, but does not add automatic ladder policy unless the original run opted into it.

## Physical rows and logical sessions

DuckDB retains one `runs` row per physical process because lifecycle, logs, PIDs, and output directories are physical. Modern tuner attempts link to their logical session through `tuning_attempts`; the session owns continuation and analysis, while physical detail retains logs, stderr, and traces.

## UI architecture

The bench UI has three layers:

1. `api-client.ts` is the only bench file allowed to call `fetch`.
2. `BenchEnv` lifts API methods into `Effect` values.
3. `benchReducer` owns asynchronous workflows and polling; Solid components render state.

Tests mock `BenchEnv`. Component and reducer tests must not require a live server or browser.

Opening a run starts a generation-tagged polling loop. Each tick fetches the current physical detail and log, its chain, and every rung's trials. Generation checks prevent late responses from a previously open run mutating the new panel.

When automatic laddering creates a child, the chain's newest ID differs from the currently open physical ID. The reducer must open that newest rung before treating the stopped parent as the end of observation. Manual advancement follows the same rule through its success action.

The tuner chart concatenates trials in chain order. Trial IDs may repeat between physical runs, so identity and ordering use `(rungIndex, trial_id)`, never `trial_id` alone. Baseline grouping must include the baseline instance so observations from different opponents are not pooled.

A rung boundary is a state transition, not a data point. The “new baseline” marker must appear as soon as the chain contains the new rung. If the new rung has not scored a trial yet, the marker is pinned to the previous rung's final point; once both sides have points it is placed between them.

The list, graph, trial table, progress display, and status badges should all answer questions about the same logical run. Whenever adding UI derived state, explicitly decide whether it is physical-rung or logical-chain state and name it accordingly.

## Architectural invariants

Changes to this system should preserve these contracts:

1. One ladder is one logical run.
2. A baseline transition does not increase the total trial budget.
3. Crossing the saturation threshold may transition a running rung immediately.
4. Only the tuner's tracked incumbent may be promoted.
5. A promoted rung faces only that incumbent unless a different curriculum is deliberately designed.
6. A parent process is fully stopped before its runhistory is loaded.
7. Physical logs, PIDs, rows, and output directories remain distinct and inspectable.
8. Logical collection views collapse ladder rows and use the newest rung's status.
9. An open logical run follows its newest physical rung automatically.
10. Chain visualizations retain all earlier trials and show a boundary immediately.
11. Search-space declarations come from the game binary, not duplicated Python or UI schemas.
12. The server is the sole DuckDB writer; Python communicates through files.

## Required regression coverage

Architectural changes are incomplete without tests at the layer that owns the behavior.

Rust server tests should cover:

- command construction and ordered override precedence;
- run ID propagation and resume configuration;
- incumbent ingestion and upsert behavior;
- immediate running-rung advancement at the threshold;
- rejection of stopped, crashed, child-bearing, exhausted, or unsaturated rungs;
- replacement rather than accumulation of baseline instances;
- preservation of the total trial budget for automatic and manual transitions;
- logical run-list collapse, aggregate counts, latest status, filtering, and limits;
- chain discovery from root, middle, and latest rungs;
- stop-before-resume process behavior.

Python tests should cover:

- override parsing and last-value-wins behavior;
- explicit baseline requirements;
- named versus raw baseline routing;
- runhistory loading into a fresh scenario;
- trial and incumbent JSONL contracts;
- propagation of game config, seed, iteration budget, and trace path.

Reducer tests should cover:

- polling and stale-generation rejection;
- concatenation and rung tagging of chain trials;
- automatic following of a newly created latest rung;
- terminal behavior only at the logical chain endpoint;
- manual advancement following the same path.

Component tests should cover:

- one logical run row per ladder;
- continuous cross-rung graphs and tables;
- a boundary marker before the new rung's first score;
- comparisons against the latest rung's actual recorded baseline;
- logical status, counts, and progress.

Tests should use in-memory DuckDB, mocked environments, and fake game data. Slow real self-play belongs in stress tests or reusable examples, not the fast unit suite.

## Guidance for future changes

Before changing laddering, resume, run lists, or charts, write down whether each affected value belongs to:

- the logical tuning session;
- one physical tuner invocation;
- one baseline rung;
- one tuner trial;
- one game inside a trial.

Trial budget and the user-facing experiment belong to the logical session. PID, log cursor, exit code, and output directory belong to a physical invocation. Baseline settings and promoted incumbent belong to a rung. Candidate configuration, seed, cost, and instance belong to a trial. Move traces belong to games.

If a proposed field or API mixes those levels, introduce an explicit projection instead. The chain model is not incidental bookkeeping; it is the boundary that lets physical restarts remain operationally honest while the operator experiences one coherent optimization.
