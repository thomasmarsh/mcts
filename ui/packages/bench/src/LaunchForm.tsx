// LaunchForm.tsx — Data-driven launch form for bench runs.
//
// Populated from `GET /api/bench/kinds` metadata (available kinds, games,
// strategies) rather than hardcoding one form per run kind.  Mirrors how
// `GameAdapter::default_config` drives the existing new-game form in
// `GameShell.tsx`.

import { createMemo, createSignal, For, Show, type Component } from "solid-js";
import type { Store } from "@mcts/core";
import type { BenchAction, BenchState } from "./index.js";

export const LaunchForm: Component<{
  store: Store<BenchState, BenchAction>;
}> = (props) => {
  const state = props.store.getState();
  const dispatch = props.store.dispatch;

  const kinds = createMemo(() => {
    const k = state().kinds;
    return k.status === "done" ? (k.result ?? []) : [];
  });

  // Form state.
  const [selectedKind, setSelectedKind] = createSignal("");
  const [selectedGame, setSelectedGame] = createSignal("");
  const [selectedStrategies, setSelectedStrategies] = createSignal<Set<string>>(new Set<string>());
  const [rounds, setRounds] = createSignal(1);

  // Derived from kinds metadata.
  const currentKind = createMemo(() => kinds().find((k) => k.kind === selectedKind()));
  const currentGame = createMemo(() => currentKind()?.games.find((g) => g.game === selectedGame()));

  const launchStatus = createMemo(() => state().launch.status);
  const launchError = createMemo(() => (state().launch.status === "error" ? state().launch.error : null));

  const busy = createMemo(() => launchStatus() === "pending");

  // Reset game and strategies when kind changes.
  function onKindChange(kind: string): void {
    setSelectedKind(kind);
    setSelectedGame("");
    setSelectedStrategies(new Set<string>());
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
    return (
      selectedKind() !== "" &&
      selectedGame() !== "" &&
      selectedStrategies().size >= 2 &&
      rounds() >= 1 &&
      !busy()
    );
  }

  function onSubmit(e: Event): void {
    e.preventDefault();
    if (!canLaunch()) return;
    dispatch({
      tag: "launch",
      action: {
        tag: "request",
        kind: selectedKind(),
        game: selectedGame(),
        config: {
          strategies: Array.from(selectedStrategies()),
          rounds: rounds(),
        },
      },
    });
  }

  return (
    <form id="launch-form" onSubmit={onSubmit}>
      <h3>Launch New Run</h3>

      <Show when={launchStatus() === "done"}>
        <div class="launch-success">
          Run launched: <code>{state().launch.result?.run_id}</code>
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

        <button type="submit" id="launch-button" disabled={!canLaunch()}>
          {busy() ? "Launching…" : "Launch"}
        </button>
      </Show>
    </form>
  );
};