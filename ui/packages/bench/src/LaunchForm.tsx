// LaunchForm.tsx — Data-driven launch form for bench runs.
//
// Populated from `GET /api/bench/kinds` metadata (available kinds, games,
// strategies) rather than hardcoding one form per run kind.  Mirrors how
// `GameAdapter::default_config` drives the existing new-game form in
// `GameShell.tsx`.

import { createMemo, createSignal, For, Show, type Component } from "solid-js";
import type { Store } from "@mcts/core";
import type { BenchAction, BenchState } from "./index.js";
import { FLOOR_BASELINES, isEmptyGameConfig, Smac3LaunchFields } from "./Smac3LaunchFields.js";

const SMAC3_KIND = "smac3";
const DEFAULT_SMAC3_N_TRIALS = 100;
const DEFAULT_SMAC3_SEED = 42;
const DEFAULT_SMAC3_STARTING_BASELINE = FLOOR_BASELINES[0]; // "flat_mc"
const DEFAULT_SMAC3_MAX_RUNGS = 5;
const DEFAULT_SMAC3_SATURATION_THRESHOLD = 0.0;

/** Build the `--override key=value` argv (as `config.overrides`, per
 * `build_command`'s `"smac3"` arm in `server/src/bench/mod.rs`) from the
 * budget fields. `n_workers` is omitted entirely when left blank ("auto"),
 * matching `smac3/config/default.yaml`'s `null -> cpu_count // 2` default.
 * `rounds` is likewise omitted when it matches the tuner's own
 * `eval_rounds` default, so a run launched without touching the field
 * produces the same argv as before this field existed. `startingBaseline`
 * is forwarded as `target.baselines=[...]` whenever it differs from the
 * tuner's own default baseline list -- `smac3_cli`'s `_apply_overrides`
 * parses the value with `ast.literal_eval`, so a Python list literal
 * round-trips as-is (see `smac3/src/smac3_cli/__main__.py`). */
function buildSmac3Overrides(opts: {
  nTrials: number;
  nWorkers: string;
  deterministic: boolean;
  seed: number;
  rounds: number;
  defaultRounds: number | null;
  startingBaseline: string;
  defaultBaselines: string[];
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
  const isDefaultBaseline =
    opts.defaultBaselines.length === 1 && opts.defaultBaselines[0] === opts.startingBaseline;
  if (opts.startingBaseline !== "" && !isDefaultBaseline) {
    overrides.push(`target.baselines=['${opts.startingBaseline}']`);
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
  const smac3Games = createMemo(() => {
    const k = state().smac3Kinds;
    return k.status === "done" ? (k.result ?? []) : [];
  });
  const smac3GamesLoading = createMemo(() => state().smac3Kinds.status === "pending");

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

  // SMAC3-only budget fields (see Smac3LaunchFields.tsx / buildSmac3Overrides).
  const [smac3NTrials, setSmac3NTrials] = createSignal(DEFAULT_SMAC3_N_TRIALS);
  const [smac3NWorkers, setSmac3NWorkers] = createSignal("");
  const [smac3Deterministic, setSmac3Deterministic] = createSignal(false);
  const [smac3Seed, setSmac3Seed] = createSignal(DEFAULT_SMAC3_SEED);
  const [smac3Rounds, setSmac3Rounds] = createSignal(1);
  // Raw JSON text for the "Game config" field -- only meaningful (and only
  // rendered by Smac3LaunchFields) when the selected game's tuner reports a
  // non-empty `game_config`.
  const [smac3GameConfig, setSmac3GameConfig] = createSignal("");
  // Which opponent a fresh run's root rung starts against, and whether it
  // opts into the automated ladder driver -- see Smac3LaunchFields.tsx's
  // prop doc comments and buildSmac3Overrides above.
  const [smac3StartingBaseline, setSmac3StartingBaseline] = createSignal(DEFAULT_SMAC3_STARTING_BASELINE);
  const [smac3LadderEnabled, setSmac3LadderEnabled] = createSignal(false);
  const [smac3MaxRungs, setSmac3MaxRungs] = createSignal(DEFAULT_SMAC3_MAX_RUNGS);
  const [smac3SaturationThreshold, setSmac3SaturationThreshold] = createSignal(
    DEFAULT_SMAC3_SATURATION_THRESHOLD,
  );

  // Derived from kinds metadata.
  const currentKind = createMemo(() => kinds().find((k) => k.kind === selectedKind()));
  const currentGame = createMemo(() => currentKind()?.games.find((g) => g.game === selectedGame()));
  const isSmac3 = createMemo(() => selectedKind() === SMAC3_KIND);
  const currentSmac3Tuner = createMemo(() => smac3Games().find((g) => g.game === selectedGame())?.tuner ?? null);

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

  // Reset game and strategies when kind changes.
  function onKindChange(kind: string): void {
    setSelectedKind(kind);
    setSelectedGame("");
    setSelectedStrategies(new Set<string>());
    if (kind === SMAC3_KIND) {
      // Pre-select the first tunable game, if the metadata has loaded, and
      // default rounds/trial to that game's tuner-declared eval_rounds.
      if (smac3Games().length > 0) {
        const first = smac3Games()[0]!;
        setSelectedGame(first.game);
        setSmac3Rounds(first.tuner.eval_rounds);
        setSmac3GameConfig(gameConfigTextFor(first.tuner));
        setSmac3StartingBaseline(DEFAULT_SMAC3_STARTING_BASELINE);
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
  function onSmac3GameChange(game: string): void {
    setSelectedGame(game);
    const tuner = smac3Games().find((g) => g.game === game)?.tuner;
    if (tuner) setSmac3Rounds(tuner.eval_rounds);
    setSmac3GameConfig(gameConfigTextFor(tuner));
    setSmac3StartingBaseline(DEFAULT_SMAC3_STARTING_BASELINE);
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
  const smac3GameConfigError = createMemo(() => {
    const tuner = currentSmac3Tuner();
    if (!tuner || isEmptyGameConfig(tuner.game_config)) return null;
    try {
      JSON.parse(smac3GameConfig());
      return null;
    } catch (e) {
      return e instanceof Error ? e.message : "invalid JSON";
    }
  });

  function canLaunch(): boolean {
    if (busy() || selectedKind() === "" || selectedGame() === "") return false;
    if (isSmac3()) return smac3NTrials() >= 1 && smac3GameConfigError() === null;
    return selectedStrategies().size >= 2 && rounds() >= 1;
  }

  function onSubmit(e: Event): void {
    e.preventDefault();
    if (!canLaunch()) return;
    const tuner = currentSmac3Tuner();
    const config = isSmac3()
      ? {
          overrides: buildSmac3Overrides({
            nTrials: smac3NTrials(),
            nWorkers: smac3NWorkers(),
            deterministic: smac3Deterministic(),
            seed: smac3Seed(),
            rounds: smac3Rounds(),
            defaultRounds: tuner?.eval_rounds ?? null,
            startingBaseline: smac3StartingBaseline(),
            defaultBaselines: tuner?.baselines ?? [],
          }),
          ...(tuner && !isEmptyGameConfig(tuner.game_config)
            ? { game_config: JSON.parse(smac3GameConfig()) }
            : {}),
          // Consumed by `inject_ladder_root_if_new_ladder`
          // (`server/src/bench/mod.rs`) -- this launch becomes a new
          // ladder's root rung, and `plan_ladder_advances`'s background
          // poll loop takes it from there once it saturates.
          ...(smac3LadderEnabled()
            ? {
                ladder: {
                  max_rungs: smac3MaxRungs(),
                  saturation_threshold: smac3SaturationThreshold(),
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

          <Show when={isSmac3()} fallback={
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
            <Smac3LaunchFields
              games={smac3Games()}
              gamesLoading={smac3GamesLoading()}
              game={selectedGame()}
              onGameChange={onSmac3GameChange}
              nTrials={smac3NTrials()}
              onNTrialsChange={setSmac3NTrials}
              nWorkers={smac3NWorkers()}
              onNWorkersChange={setSmac3NWorkers}
              deterministic={smac3Deterministic()}
              onDeterministicChange={setSmac3Deterministic}
              seed={smac3Seed()}
              onSeedChange={setSmac3Seed}
              rounds={smac3Rounds()}
              onRoundsChange={setSmac3Rounds}
              gameConfig={smac3GameConfig()}
              onGameConfigChange={setSmac3GameConfig}
              gameConfigError={smac3GameConfigError()}
              startingBaseline={smac3StartingBaseline()}
              onStartingBaselineChange={setSmac3StartingBaseline}
              ladderEnabled={smac3LadderEnabled()}
              onLadderEnabledChange={setSmac3LadderEnabled}
              maxRungs={smac3MaxRungs()}
              onMaxRungsChange={setSmac3MaxRungs}
              saturationThreshold={smac3SaturationThreshold()}
              onSaturationThresholdChange={setSmac3SaturationThreshold}
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