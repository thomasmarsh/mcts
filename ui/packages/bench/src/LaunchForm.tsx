// LaunchForm.tsx — Data-driven launch form for bench runs.
//
// Populated from `GET /api/bench/kinds` metadata (available kinds, games,
// strategies) rather than hardcoding one form per run kind.  Mirrors how
// `GameAdapter::default_config` drives the existing new-game form in
// `GameShell.tsx`.

import { createMemo, createSignal, For, Show, type Component } from "solid-js";
import type { Store } from "@mcts/core";
import type { BenchAction, BenchState } from "./index.js";
import { isEmptyGameConfig, TunerLaunchFields } from "./TunerLaunchFields.js";

const tuner_KIND = "tuner";
const DEFAULT_TUNER_N_TRIALS = 100;
const DEFAULT_TUNER_SEED = 42;
const DEFAULT_TUNER_MAX_RUNGS = 5;
const DEFAULT_tuner_SATURATION_THRESHOLD = 0.0;

/** Build the `--override key=value` argv (as `config.overrides`, per
 * `build_command`'s `"tuner"` arm in `server/src/bench/mod.rs`) from the
 * budget fields. `n_workers` is omitted entirely when left blank ("auto"),
 * matching `tuner/config/default.yaml`'s `null -> cpu_count // 2` default.
 * `rounds` is likewise omitted when it matches the tuner's own
 * `eval_rounds` default, so a run launched without touching the field
 * produces the same argv as before this field existed. `startingBaselines`
 * is always forwarded as `target.baselines=[...]` (one entry per selected
 * panel member -- tuner evaluates every trial against all of them and
 * averages cost across instances); tuner metadata lists available presets
 * but deliberately supplies no runtime default.
 * `tuner_cli`'s `_apply_overrides`
 * parses the value with `ast.literal_eval`, so a Python list literal
 * round-trips as-is (see `tuner/src/tuner_cli/__main__.py`). */
function buildTunerOverrides(opts: {
  nTrials: number;
  nWorkers: string;
  deterministic: boolean;
  seed: number;
  rounds: number;
  defaultRounds: number | null;
  startingBaselines: Set<string>;
  /** Empty string means "unset" -- see `TunerLaunchFields.tsx`'s
   * `maxIterations` prop doc comment. */
  maxIterations: string;
  /** Empty string means "unset" -- see `TunerLaunchFields.tsx`'s
   * `maxTimeMs` prop doc comment. Mutually exclusive with `maxIterations`;
   * the form itself keeps only one of the two fields enabled at a time, so
   * both being non-empty here shouldn't happen, but this function doesn't
   * re-validate that -- the server-side `target.py` rejects it either way. */
  maxTimeMs: string;
}): string[] {
  const overrides = [
    `optimizer.n_trials=${opts.nTrials}`,
    `optimizer.deterministic=${opts.deterministic ? "True" : "False"}`,
    `optimizer.seed=${opts.seed}`,
  ];
  const workers = opts.nWorkers.trim();
  if (workers !== "") overrides.push(`optimizer.n_workers=${workers}`);
  if (opts.defaultRounds !== null && opts.rounds !== opts.defaultRounds) {
    overrides.push(`target.rounds=${opts.rounds}`);
  }
  const maxIterations = opts.maxIterations.trim();
  if (maxIterations !== "") overrides.push(`target.max_iterations=${maxIterations}`);
  const maxTimeMs = opts.maxTimeMs.trim();
  if (maxTimeMs !== "") overrides.push(`target.max_time_ms=${maxTimeMs}`);
  if (opts.startingBaselines.size > 0) {
    const items = Array.from(opts.startingBaselines)
      .map((b) => `'${b}'`)
      .join(", ");
    overrides.push(`target.baselines=[${items}]`);
  }
  return overrides;
}

export const LaunchForm: Component<{
  store: Store<BenchState, BenchAction>;
}> = (props) => {
  const state = props.store.getState();
  const dispatch = props.store.dispatch;

  const kinds = createMemo(() => {
    const k = state().kinds;
    return k.status === "done" ? (k.result ?? []) : [];
  });
  const tunerGames = createMemo(() => {
    const k = state().tunerKinds;
    return k.status === "done" ? (k.result ?? []) : [];
  });
  const tunerGamesLoading = createMemo(() => state().tunerKinds.status === "pending");

  // Starts collapsed if a run is already open when this mounts (e.g. after
  // a page reload) -- the detail panel is usually what the operator wants
  // the vertical space for at that point. Only an initial heuristic: it
  // doesn't auto-collapse on a later `openRun` since the operator may be
  // mid-edit on a new launch; the toggle is one click either way.
  const [expanded, setExpanded] = createSignal(state().openRun === null);

  // Form state.
  const [selectedKind, setSelectedKind] = createSignal("");
  const [selectedGame, setSelectedGame] = createSignal("");
  const [selectedStrategies, setSelectedStrategies] = createSignal<Set<string>>(new Set<string>());
  const [rounds, setRounds] = createSignal(1);

  // tuner-only budget fields (see TunerLaunchFields.tsx / buildTunerOverrides).
  const [tunerNTrials, setTunerNTrials] = createSignal(DEFAULT_TUNER_N_TRIALS);
  const [tunerNWorkers, setTunerNWorkers] = createSignal("");
  const [tunerDeterministic, setTunerDeterministic] = createSignal(false);
  const [tunerSeed, setTunerSeed] = createSignal(DEFAULT_TUNER_SEED);
  const [tunerRounds, setTunerRounds] = createSignal(1);
  // Per-run MCTS iteration ceiling override -- "" means unset/auto, same
  // convention as `tunerNWorkers`. See TunerLaunchFields.tsx's
  // `maxIterations` prop doc comment.
  const [tunerMaxIterations, setTunerMaxIterations] = createSignal("");
  // Per-run wall-clock search budget override -- "" means unset, mutually
  // exclusive with `tunerMaxIterations`. See TunerLaunchFields.tsx's
  // `maxTimeMs` prop doc comment.
  const [tunerMaxTimeMs, setTunerMaxTimeMs] = createSignal("");
  // Raw JSON text for the "Game config" field -- only meaningful (and only
  // rendered by TunerLaunchFields) when the selected game's tuner reports a
  // non-empty `game_config`.
  const [tunerGameConfig, setTunerGameConfig] = createSignal("");
  // Which opponent panel a fresh run's root rung starts against, and
  // whether it opts into the automated ladder driver -- see
  // TunerLaunchFields.tsx's prop doc comments and buildTunerOverrides
  // above. Defaults to every named preset the selected game's tuner
  // reports (see `defaultStartingBaselines` below), matching the old
  // single-select behavior when a game has exactly one preset.
  const [tunerStartingBaselines, setTunerStartingBaselines] = createSignal<Set<string>>(new Set<string>());
  const [tunerLadderEnabled, setTunerLadderEnabled] = createSignal(false);
  const [tunerMaxRungs, setTunerMaxRungs] = createSignal(DEFAULT_TUNER_MAX_RUNGS);
  const [tunerSaturationThreshold, setTunerSaturationThreshold] = createSignal(
    DEFAULT_tuner_SATURATION_THRESHOLD,
  );

  // Derived from kinds metadata.
  const currentKind = createMemo(() => kinds().find((k) => k.kind === selectedKind()));
  const currentGame = createMemo(() => currentKind()?.games.find((g) => g.game === selectedGame()));
  const isTuner = createMemo(() => selectedKind() === tuner_KIND);
  const currentTunerTuner = createMemo(() => tunerGames().find((g) => g.game === selectedGame())?.tuner ?? null);

  const launchStatus = createMemo(() => state().launch.status);
  const launchError = createMemo(() => (state().launch.status === "error" ? state().launch.error : null));
  // If the launch HTTP call succeeded but the child process died immediately
  // (e.g. bad CLI args), the response carries the stderr in `launch_error`.
  const launchResponseError = createMemo(() => {
    const r = state().launch.result;
    return r?.launch_error ?? null;
  });

  const busy = createMemo(() => launchStatus() === "pending");

  // Pre-filled "Game config" textarea text for a tuner -- "" when the game
  // has nothing configurable, matching how the field itself is hidden then.
  function gameConfigTextFor(tuner: { game_config: unknown } | undefined): string {
    if (!tuner || isEmptyGameConfig(tuner.game_config)) return "";
    return JSON.stringify(tuner.game_config, null, 2);
  }

  // Default starting-baseline panel for a tuner: every named preset it
  // reports, so a game shipping only "strong" still launches against that
  // one instance (old single-select behavior) while a game with several
  // presets (the common case -- see `games/*/presets.json`) starts with
  // the full bracket already selected.
  function defaultStartingBaselines(tuner: { baselines: string[] } | undefined): Set<string> {
    return new Set(tuner?.baselines ?? []);
  }

  // Reset game and strategies when kind changes.
  function onKindChange(kind: string): void {
    setSelectedKind(kind);
    setSelectedGame("");
    setSelectedStrategies(new Set<string>());
    if (kind === tuner_KIND) {
      // Pre-select the first tunable game, if the metadata has loaded, and
      // default rounds/trial to that game's tuner-declared eval_rounds.
      if (tunerGames().length > 0) {
        const first = tunerGames()[0]!;
        setSelectedGame(first.game);
        setTunerRounds(first.tuner.eval_rounds);
        setTunerGameConfig(gameConfigTextFor(first.tuner));
        setTunerStartingBaselines(defaultStartingBaselines(first.tuner));
      }
      return;
    }
    // Pre-select first game if available.
    const k = kinds().find((k: { kind: string }) => k.kind === kind);
    if (k && k.games.length > 0) {
      setSelectedGame(k.games[0]!.game);
    }
  }

  // Reset strategies when game changes.
  function onGameChange(game: string): void {
    setSelectedGame(game);
    setSelectedStrategies(new Set<string>());
  }

  // Reset rounds/trial and the game-config field to the newly selected
  // game's tuner defaults.
  function onTunerGameChange(game: string): void {
    setSelectedGame(game);
    const tuner = tunerGames().find((g) => g.game === game)?.tuner;
    if (tuner) setTunerRounds(tuner.eval_rounds);
    setTunerGameConfig(gameConfigTextFor(tuner));
    setTunerStartingBaselines(defaultStartingBaselines(tuner));
  }

  function toggleStartingBaseline(id: string): void {
    setTunerStartingBaselines((prev) => {
      const next = new Set(prev);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });
  }

  function toggleStrategy(id: string): void {
    setSelectedStrategies((prev) => {
      const next = new Set(prev);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });
  }

  // `null` when the field is hidden (nothing to configure) or valid; an
  // error message when shown and its contents don't parse as JSON.
  const tunerGameConfigError = createMemo(() => {
    const tuner = currentTunerTuner();
    if (!tuner || isEmptyGameConfig(tuner.game_config)) return null;
    try {
      JSON.parse(tunerGameConfig());
      return null;
    } catch (e) {
      return e instanceof Error ? e.message : "invalid JSON";
    }
  });

  function canLaunch(): boolean {
    if (busy() || selectedKind() === "" || selectedGame() === "") return false;
    if (isTuner()) {
      return tunerNTrials() >= 1 && tunerGameConfigError() === null && tunerStartingBaselines().size > 0;
    }
    return selectedStrategies().size >= 2 && rounds() >= 1;
  }

  function onSubmit(e: Event): void {
    e.preventDefault();
    if (!canLaunch()) return;
    const tuner = currentTunerTuner();
    const config = isTuner()
      ? {
          overrides: buildTunerOverrides({
            nTrials: tunerNTrials(),
            nWorkers: tunerNWorkers(),
            deterministic: tunerDeterministic(),
            seed: tunerSeed(),
            rounds: tunerRounds(),
            defaultRounds: tuner?.eval_rounds ?? null,
            startingBaselines: tunerStartingBaselines(),
            maxIterations: tunerMaxIterations(),
            maxTimeMs: tunerMaxTimeMs(),
          }),
          ...(tuner && !isEmptyGameConfig(tuner.game_config)
            ? { game_config: JSON.parse(tunerGameConfig()) }
            : {}),
          // Consumed by `inject_ladder_root_if_new_ladder`
          // (`server/src/bench/mod.rs`) -- this launch becomes a new
          // ladder's root rung, and `plan_ladder_advances`'s background
          // poll loop takes it from there once it saturates.
          ...(tunerLadderEnabled()
            ? {
                ladder: {
                  max_rungs: tunerMaxRungs(),
                  saturation_threshold: tunerSaturationThreshold(),
                },
              }
            : {}),
        }
      : {
          strategies: Array.from(selectedStrategies()),
          rounds: rounds(),
        };
    dispatch({
      tag: "launch",
      action: {
        tag: "request",
        kind: selectedKind(),
        game: selectedGame(),
        config,
      },
    });
  }

  return (
    <form id="launch-form" onSubmit={onSubmit}>
      <button
        type="button"
        id="launch-form-toggle"
        onClick={() => setExpanded((e) => !e)}
        aria-expanded={expanded()}
      >
        <span classList={{ "launch-form-chevron": true, "launch-form-chevron-collapsed": !expanded() }}>▾</span>
        Launch New Run
      </button>

      <Show when={expanded()}>
        <Show when={launchStatus() === "done" && !launchResponseError()}>
          <div class="launch-success">
            Run launched: <code>{state().launch.result?.run_id}</code>
          </div>
          <Show when={launchResponseError()}>
            <div class="launch-error launch-error-response">
              <strong>Launch error:</strong>
              <pre>{launchResponseError()}</pre>
            </div>
          </Show>
        </Show>

        <Show when={launchError()}>
          <div class="launch-error">{launchError()}</div>
        </Show>

        <Show when={state().kinds.status === "done" && kinds().length === 0}>
          <div class="launch-empty">No run kinds available.</div>
        </Show>

        <Show when={state().kinds.status === "error"}>
          <div class="launch-error">{state().kinds.error}</div>
        </Show>

        <Show when={kinds().length > 0} fallback={<div class="loading-bench">Loading run kinds…</div>}>
          <label>
            Run Kind
            <select value={selectedKind()} onChange={(e) => onKindChange(e.currentTarget.value)} disabled={busy()}>
              <option value="">— Select —</option>
              <For each={kinds()}>
                {(k) => <option value={k.kind}>{k.label}</option>}
              </For>
            </select>
          </label>

          <Show when={isTuner()} fallback={
            <>
              <Show when={currentKind()}>
                <label>
                  Game
                  <select value={selectedGame()} onChange={(e) => onGameChange(e.currentTarget.value)} disabled={busy()}>
                    <option value="">— Select —</option>
                    <For each={currentKind()!.games}>
                      {(g) => <option value={g.game}>{g.game}</option>}
                    </For>
                  </select>
                </label>
              </Show>

              <Show when={currentGame()}>
                <fieldset id="strategy-picker">
                  <legend>Strategies (select at least 2)</legend>
                  <For each={currentGame()!.strategies}>
                    {(s) => (
                      <label class="strategy-option">
                        <input
                          type="checkbox"
                          checked={selectedStrategies().has(s.id)}
                          onChange={() => toggleStrategy(s.id)}
                          disabled={busy()}
                        />
                        <span class="strategy-label">{s.label}</span>
                        <span class="strategy-desc">{s.description}</span>
                      </label>
                    )}
                  </For>
                </fieldset>
              </Show>

              <Show when={selectedGame()}>
                <label>
                  Rounds
                  <input
                    type="number"
                    min={1}
                    value={rounds()}
                    onInput={(e) => setRounds(Math.max(1, parseInt(e.currentTarget.value) || 1))}
                    disabled={busy()}
                  />
                </label>
              </Show>
            </>
          }>
            <TunerLaunchFields
              games={tunerGames()}
              gamesLoading={tunerGamesLoading()}
              game={selectedGame()}
              onGameChange={onTunerGameChange}
              nTrials={tunerNTrials()}
              onNTrialsChange={setTunerNTrials}
              nWorkers={tunerNWorkers()}
              onNWorkersChange={setTunerNWorkers}
              deterministic={tunerDeterministic()}
              onDeterministicChange={setTunerDeterministic}
              seed={tunerSeed()}
              onSeedChange={setTunerSeed}
              rounds={tunerRounds()}
              onRoundsChange={setTunerRounds}
              maxIterations={tunerMaxIterations()}
              onMaxIterationsChange={setTunerMaxIterations}
              maxTimeMs={tunerMaxTimeMs()}
              onMaxTimeMsChange={setTunerMaxTimeMs}
              gameConfig={tunerGameConfig()}
              onGameConfigChange={setTunerGameConfig}
              gameConfigError={tunerGameConfigError()}
              startingBaselines={tunerStartingBaselines()}
              onToggleStartingBaseline={toggleStartingBaseline}
              ladderEnabled={tunerLadderEnabled()}
              onLadderEnabledChange={setTunerLadderEnabled}
              maxRungs={tunerMaxRungs()}
              onMaxRungsChange={setTunerMaxRungs}
              saturationThreshold={tunerSaturationThreshold()}
              onSaturationThresholdChange={setTunerSaturationThreshold}
              disabled={busy()}
            />
          </Show>

          <button type="submit" id="launch-button" disabled={!canLaunch()}>
            {busy() ? "Launching…" : "Launch"}
          </button>
        </Show>
      </Show>
    </form>
  );
};
