// LaunchForm.tsx — Data-driven launch form for bench runs.
//
// Populated from `GET /api/bench/kinds` metadata (available kinds, games,
// strategies) rather than hardcoding one form per run kind.  Mirrors how
// `GameAdapter::default_config` drives the existing new-game form in
// `GameShell.tsx`.
//
// Lives in the main pane (not the sidebar) — the sidebar has only the run
// list with a compact "New Run" toggling this open.

import { createMemo, createSignal, For, Show, type Component } from "solid-js";
import type { Store } from "@mcts/core";
import type { BenchAction, BenchState } from "./index.js";
import { isEmptyGameConfig, TunerLaunchFields } from "./TunerLaunchFields.js";
import { buildTunerOverrides, validateTunerLaunchPolicy } from "./tuner-launch-policy.js";

const DEFAULT_TUNER_N_TRIALS = 100;
const DEFAULT_TUNER_SEED = 42;
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
  const [tunerMaxIterations, setTunerMaxIterations] = createSignal("");
  const [tunerMaxTimeMs, setTunerMaxTimeMs] = createSignal("");
  const [tunerGameConfig, setTunerGameConfig] = createSignal("");
  const [tunerMinPairs, setTunerMinPairs] = createSignal(2);
  const [tunerMaxPairs, setTunerMaxPairs] = createSignal(6);
  const [tunerPruningEnabled, setTunerPruningEnabled] = createSignal(false);
  const [tunerReductionFactor, setTunerReductionFactor] = createSignal(3);
  const [tunerPruningStartupTerminalTrials, setTunerPruningStartupTerminalTrials] = createSignal(5);
  const [tunerSigmaStop, setTunerSigmaStop] = createSignal("");
  const [tunerTpeStartupTrials, setTunerTpeStartupTrials] = createSignal(3);

  // Derived from kinds metadata.
  const currentKind = createMemo(() => kinds().find((k) => k.kind === selectedKind()));
  const currentGame = createMemo(() => currentKind()?.games.find((g) => g.game === selectedGame()));
  const isTuner = createMemo(() => selectedKind() === "tuner");
  const currentTunerTuner = createMemo(() => tunerGames().find((g) => g.game === selectedGame())?.tuner ?? null);

  const launchStatus = createMemo(() => state().launch.status);
  const launchError = createMemo(() => (state().launch.status === "error" ? state().launch.error : null));
  const launchResponseError = createMemo(() => {
    const r = state().launch.result;
    return r?.launch_error ?? null;
  });

  const busy = createMemo(() => launchStatus() === "pending");

  // Reset game and strategies when kind changes.
  function onKindChange(kind: string): void {
    setSelectedKind(kind);
    setSelectedGame("");
    setSelectedStrategies(new Set<string>());
    if (kind === "tuner") {
      if (tunerGames().length > 0) {
        const first = tunerGames()[0]!;
        setSelectedGame(first.game);
        setTunerGameConfig(gameConfigTextFor(first.tuner));
      }
      return;
    }
    const k = kinds().find((k: { kind: string }) => k.kind === kind);
    if (k && k.games.length > 0) {
      setSelectedGame(k.games[0]!.game);
    }
  }

  function onGameChange(game: string): void {
    setSelectedGame(game);
    setSelectedStrategies(new Set<string>());
  }

  function onTunerGameChange(game: string): void {
    setSelectedGame(game);
    setTunerGameConfig(gameConfigTextFor(tunerGames().find((g) => g.game === game)?.tuner));
  }

  function toggleStrategy(id: string): void {
    setSelectedStrategies((prev) => {
      const next = new Set(prev);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });
  }

  function gameConfigTextFor(tuner: { game_config: unknown } | undefined): string {
    if (!tuner || isEmptyGameConfig(tuner.game_config)) return "";
    return JSON.stringify(tuner.game_config, null, 2);
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
      return tunerGameConfigError() === null && validateTunerLaunchPolicy(tunerLaunchPolicy()) === null;
    }
    return selectedStrategies().size >= 2 && rounds() >= 1;
  }

  function onSubmit(e: Event): void {
    e.preventDefault();
    if (!canLaunch()) return;
    const config = isTuner()
      ? {
          overrides: buildTunerOverrides(tunerLaunchPolicy()),
          ...(currentTunerTuner() && !isEmptyGameConfig(currentTunerTuner()!.game_config)
            ? { game_config: JSON.parse(tunerGameConfig()) }
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

  function tunerLaunchPolicy() {
    return {
      nTrials: tunerNTrials(), nWorkers: tunerNWorkers(), deterministic: tunerDeterministic(), seed: tunerSeed(),
      minPairs: tunerMinPairs(), maxPairs: tunerMaxPairs(), pruningEnabled: tunerPruningEnabled(),
      reductionFactor: tunerReductionFactor(), pruningStartupTerminalTrials: tunerPruningStartupTerminalTrials(),
      sigmaStop: tunerSigmaStop(), tpeStartupTrials: tunerTpeStartupTrials(),
      maxIterations: tunerMaxIterations(), maxTimeMs: tunerMaxTimeMs(),
    };
  }

  return (
    <form id="launch-form" onSubmit={onSubmit}>
      <div id="launch-form-header">
        <h3>Launch New Run</h3>
      </div>

      <Show when={launchStatus() === "done" && !launchResponseError()}>
        <div class="launch-success">
          Run launched: <code>{state().launch.result?.run_id}</code>
        </div>
      </Show>

      <Show when={launchStatus() === "done" && launchResponseError()}>
        <div class="launch-error launch-error-response">
          <strong>Launch error:</strong>
          <pre>{launchResponseError()}</pre>
        </div>
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
            maxIterations={tunerMaxIterations()}
            onMaxIterationsChange={setTunerMaxIterations}
            maxTimeMs={tunerMaxTimeMs()}
            onMaxTimeMsChange={setTunerMaxTimeMs}
            gameConfig={tunerGameConfig()}
            onGameConfigChange={setTunerGameConfig}
            gameConfigError={tunerGameConfigError()}
            minPairs={tunerMinPairs()}
            onMinPairsChange={setTunerMinPairs}
            maxPairs={tunerMaxPairs()}
            onMaxPairsChange={setTunerMaxPairs}
            pruningEnabled={tunerPruningEnabled()}
            onPruningEnabledChange={setTunerPruningEnabled}
            reductionFactor={tunerReductionFactor()}
            onReductionFactorChange={setTunerReductionFactor}
            pruningStartupTerminalTrials={tunerPruningStartupTerminalTrials()}
            onPruningStartupTerminalTrialsChange={setTunerPruningStartupTerminalTrials}
            sigmaStop={tunerSigmaStop()}
            onSigmaStopChange={setTunerSigmaStop}
            tpeStartupTrials={tunerTpeStartupTrials()}
            onTpeStartupTrialsChange={setTunerTpeStartupTrials}
            validationError={validateTunerLaunchPolicy(tunerLaunchPolicy())}
            disabled={busy()}
          />
        </Show>

        <button type="submit" id="launch-button" disabled={!canLaunch()}>
          {busy() ? "Launching…" : "Launch"}
        </button>
      </Show>
    </form>
  );
};
