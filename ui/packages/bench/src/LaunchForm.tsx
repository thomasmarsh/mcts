// LaunchForm.tsx — launch form for bench Tuning runs.
//
// Lives in the main pane (not the sidebar) — the sidebar has only the run
// list with a compact "New Run" toggling this open.

import { createEffect, createMemo, createSignal, Show, type Component } from "solid-js";
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

  const tunerGames = createMemo(() => {
    const k = state().tunerKinds;
    return k.status === "done" ? (k.result ?? []) : [];
  });
  const tunerGamesLoading = createMemo(() => state().tunerKinds.status === "pending");

  // Form state.
  const [selectedGame, setSelectedGame] = createSignal("");

  // tuner budget fields (see TunerLaunchFields.tsx / buildTunerOverrides).
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
  const [tunerPruningStartupTrials, setTunerPruningStartupTrials] = createSignal(5);
  const [tunerSigmaStop, setTunerSigmaStop] = createSignal("");
  const [tunerTpeStartupTrials, setTunerTpeStartupTrials] = createSignal(3);

  const currentTunerTuner = createMemo(
    () => tunerGames().find((g) => g.game === selectedGame())?.tuner ?? null,
  );

  // Auto-select the first tunable game once the tuner metadata loads.
  createEffect(() => {
    if (selectedGame() === "" && tunerGames().length > 0) {
      onTunerGameChange(tunerGames()[0]!.game);
    }
  });

  const launchStatus = createMemo(() => state().launch.status);
  const launchError = createMemo(() =>
    state().launch.status === "error" ? state().launch.error : null,
  );
  const launchResponseError = createMemo(() => {
    const r = state().launch.result;
    return r?.launch_error ?? null;
  });

  const busy = createMemo(() => launchStatus() === "pending");

  function onTunerGameChange(game: string): void {
    setSelectedGame(game);
    setTunerGameConfig(gameConfigTextFor(tunerGames().find((g) => g.game === game)?.tuner));
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
    if (busy() || selectedGame() === "") return false;
    return (
      tunerGameConfigError() === null && validateTunerLaunchPolicy(tunerLaunchPolicy()) === null
    );
  }

  function onSubmit(e: Event): void {
    e.preventDefault();
    if (!canLaunch()) return;
    const config = {
      overrides: buildTunerOverrides(tunerLaunchPolicy()),
      ...(currentTunerTuner() && !isEmptyGameConfig(currentTunerTuner()!.game_config)
        ? { game_config: JSON.parse(tunerGameConfig()) }
        : {}),
    };
    dispatch({
      tag: "launch",
      action: {
        tag: "request",
        kind: "tuner",
        game: selectedGame(),
        config,
      },
    });
  }

  function tunerLaunchPolicy() {
    return {
      nTrials: tunerNTrials(),
      nWorkers: tunerNWorkers(),
      deterministic: tunerDeterministic(),
      seed: tunerSeed(),
      minPairs: tunerMinPairs(),
      maxPairs: tunerMaxPairs(),
      pruningEnabled: tunerPruningEnabled(),
      reductionFactor: tunerReductionFactor(),
      pruningStartupTrials: tunerPruningStartupTrials(),
      sigmaStop: tunerSigmaStop(),
      tpeStartupTrials: tunerTpeStartupTrials(),
      maxIterations: tunerMaxIterations(),
      maxTimeMs: tunerMaxTimeMs(),
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
        pruningStartupTrials={tunerPruningStartupTrials()}
        onPruningStartupTrialsChange={setTunerPruningStartupTrials}
        sigmaStop={tunerSigmaStop()}
        onSigmaStopChange={setTunerSigmaStop}
        tpeStartupTrials={tunerTpeStartupTrials()}
        onTpeStartupTrialsChange={setTunerTpeStartupTrials}
        validationError={validateTunerLaunchPolicy(tunerLaunchPolicy())}
        disabled={busy()}
      />

      <button type="submit" id="launch-button" disabled={!canLaunch()}>
        {busy() ? "Launching…" : "Launch"}
      </button>
    </form>
  );
};
