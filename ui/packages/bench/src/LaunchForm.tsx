// LaunchForm.tsx — Data-driven launch form for bench runs.
//
// Populated from `GET /api/bench/kinds` metadata (available kinds, games,
// strategies) rather than hardcoding one form per run kind.  Mirrors how
// `GameAdapter::default_config` drives the existing new-game form in
// `GameShell.tsx`.

import { createMemo, createSignal, For, Show, type Component } from "solid-js";
import type { Store } from "@mcts/core";
import type { BenchAction, BenchState } from "./index.js";
import { Smac3LaunchFields } from "./Smac3LaunchFields.js";

const SMAC3_KIND = "smac3";
const DEFAULT_SMAC3_N_TRIALS = 100;
const DEFAULT_SMAC3_SEED = 42;

/** Build the `--override key=value` argv (as `config.overrides`, per
 * `build_command`'s `"smac3"` arm in `server/src/bench/mod.rs`) from the
 * budget fields. `n_workers` is omitted entirely when left blank ("auto"),
 * matching `smac3/config/default.yaml`'s `null -> cpu_count // 2` default. */
function buildSmac3Overrides(opts: {
  nTrials: number;
  nWorkers: string;
  deterministic: boolean;
  seed: number;
}): string[] {
  const overrides = [
    `optimizer.n_trials=${opts.nTrials}`,
    `optimizer.deterministic=${opts.deterministic ? "True" : "False"}`,
    `optimizer.seed=${opts.seed}`,
  ];
  const workers = opts.nWorkers.trim();
  if (workers !== "") overrides.push(`optimizer.n_workers=${workers}`);
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

  // Derived from kinds metadata.
  const currentKind = createMemo(() => kinds().find((k) => k.kind === selectedKind()));
  const currentGame = createMemo(() => currentKind()?.games.find((g) => g.game === selectedGame()));
  const isSmac3 = createMemo(() => selectedKind() === SMAC3_KIND);

  const launchStatus = createMemo(() => state().launch.status);
  const launchError = createMemo(() => (state().launch.status === "error" ? state().launch.error : null));
  // If the launch HTTP call succeeded but the child process died immediately
  // (e.g. bad CLI args), the response carries the stderr in `launch_error`.
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
    if (kind === SMAC3_KIND) {
      // Pre-select the first tunable game, if the metadata has loaded.
      if (smac3Games().length > 0) setSelectedGame(smac3Games()[0]!.game);
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

  function toggleStrategy(id: string): void {
    setSelectedStrategies((prev) => {
      const next = new Set(prev);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });
  }

  function canLaunch(): boolean {
    if (busy() || selectedKind() === "" || selectedGame() === "") return false;
    if (isSmac3()) return smac3NTrials() >= 1;
    return selectedStrategies().size >= 2 && rounds() >= 1;
  }

  function onSubmit(e: Event): void {
    e.preventDefault();
    if (!canLaunch()) return;
    const config = isSmac3()
      ? {
          overrides: buildSmac3Overrides({
            nTrials: smac3NTrials(),
            nWorkers: smac3NWorkers(),
            deterministic: smac3Deterministic(),
            seed: smac3Seed(),
          }),
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
      <h3>Launch New Run</h3>

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
            onGameChange={setSelectedGame}
            nTrials={smac3NTrials()}
            onNTrialsChange={setSmac3NTrials}
            nWorkers={smac3NWorkers()}
            onNWorkersChange={setSmac3NWorkers}
            deterministic={smac3Deterministic()}
            onDeterministicChange={setSmac3Deterministic}
            seed={smac3Seed()}
            onSeedChange={setSmac3Seed}
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