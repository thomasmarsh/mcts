// CommitComparison.tsx — Two-commit comparison view showing per-strategy
// win-rate deltas, as the concrete answer to "find regressions."
//
// Reads the `commitTrends` slice of BenchState (populated by
// `fetchCommitTrends`) and lets the user pick two git commits to compare.
// Shows a table of strategies with their win rates at each commit and the
// delta between them, sorted by absolute delta descending so the biggest
// changes appear first.

import { createMemo, createSignal, For, Show, type Component } from "solid-js";
import type { Store } from "@mcts/core";
import type { BenchAction, BenchState, LeaderboardEntry } from "./index.js";

function fmtRate(v: number): string {
  return (v * 100).toFixed(1) + "%";
}

function shortSha(sha: string): string {
  return sha.length > 7 ? sha.slice(0, 7) : sha;
}

interface ComparisonRow {
  strategy: string;
  a: LeaderboardEntry | null;
  b: LeaderboardEntry | null;
  winRateDelta: number; // b - a
  absDelta: number;
}

export const CommitComparison: Component<{
  store: Store<BenchState, BenchAction>;
}> = (props) => {
  const state = props.store.getState();
  const dispatch = props.store.dispatch;

  const trends = createMemo(() => state().commitTrends);
  const runsState = createMemo(() => state().runs);
  const runs = createMemo(() => (runsState().status === "done" ? runsState().result ?? [] : []));

  // Available games from runs.
  const availableGames = createMemo(() => {
    const set = new Set<string>();
    for (const r of runs()) {
      if (r.game) set.add(r.game);
    }
    return Array.from(set).sort();
  });

  // Available commits from trend data.
  const availableCommits = createMemo(() => trends().shas);

  // Local UI state.
  const [selectedGame, setSelectedGame] = createSignal("");
  const [shaA, setShaA] = createSignal("");
  const [shaB, setShaB] = createSignal("");

  // Fetch trends when game changes.
  function onGameChange(game: string): void {
    setSelectedGame(game);
    setShaA("");
    setShaB("");
    dispatch({ tag: "fetchCommitTrends", game: game || null });
  }

  // Auto-select newest and oldest commits when data loads.
  const autoSelected = createMemo(() => {
    const shas = availableCommits();
    if (shas.length >= 2 && !shaA() && !shaB()) {
      return { a: shas[shas.length - 1]!, b: shas[0]! };
    }
    return null;
  });

  // Build comparison rows.
  const rows = createMemo(() => {
    const aSha = shaA() || autoSelected()?.a || "";
    const bSha = shaB() || autoSelected()?.b || "";
    if (!aSha || !bSha || aSha === bSha) return [];

    const data = trends().data;
    const aEntries = data[aSha] ?? [];
    const bEntries = data[bSha] ?? [];

    // Collect all strategies present in either commit.
    const strategySet = new Set<string>();
    for (const e of aEntries) strategySet.add(e.strategy);
    for (const e of bEntries) strategySet.add(e.strategy);

    const aMap = new Map<string, LeaderboardEntry>();
    for (const e of aEntries) aMap.set(e.strategy, e);
    const bMap = new Map<string, LeaderboardEntry>();
    for (const e of bEntries) bMap.set(e.strategy, e);

    const result: ComparisonRow[] = [];
    for (const strategy of strategySet) {
      const a = aMap.get(strategy) ?? null;
      const b = bMap.get(strategy) ?? null;
      const wrA = a ? a.win_rate : 0.5;
      const wrB = b ? b.win_rate : 0.5;
      const delta = wrB - wrA;
      result.push({ strategy, a, b, winRateDelta: delta, absDelta: Math.abs(delta) });
    }

    result.sort((x, y) => y.absDelta - x.absDelta);
    return result;
  });

  const aSha = createMemo(() => shaA() || autoSelected()?.a || "");
  const bSha = createMemo(() => shaB() || autoSelected()?.b || "");

  return (
    <div id="comparison-panel">
      <div id="comparison-header">
        <h3>Commit Comparison</h3>
        <div id="comparison-controls">
          <Show when={availableGames().length > 0}>
            <select
              value={selectedGame()}
              onChange={(e) => onGameChange(e.currentTarget.value)}
            >
              <option value="">All games</option>
              <For each={availableGames()}>
                {(g) => <option value={g}>{g}</option>}
              </For>
            </select>
          </Show>
        </div>
      </div>

      <Show when={availableCommits().length >= 2}>
        <div id="comparison-pickers">
          <div class="picker-group">
            <label>Older</label>
            <select value={aSha()} onChange={(e) => setShaA(e.currentTarget.value)}>
              <option value="">— Select —</option>
              <For each={availableCommits()}>
                {(sha) => (
                  <option value={sha} disabled={sha === bSha()}>
                    {shortSha(sha)}
                  </option>
                )}
              </For>
            </select>
          </div>
          <div class="picker-group">
            <label>Newer</label>
            <select value={bSha()} onChange={(e) => setShaB(e.currentTarget.value)}>
              <option value="">— Select —</option>
              <For each={availableCommits()}>
                {(sha) => (
                  <option value={sha} disabled={sha === aSha()}>
                    {shortSha(sha)}
                  </option>
                )}
              </For>
            </select>
          </div>
        </div>
      </Show>

      <Show when={trends().status === "loading"}>
        <div class="loading-bench">Loading commit data…</div>
      </Show>

      <Show when={trends().status === "error"}>
        <div class="lb-error">{trends().error}</div>
      </Show>

      <Show
        when={availableCommits().length < 2 && trends().status !== "loading" && trends().status !== "error"}
      >
        <div class="lb-empty">Need at least 2 commits with data to compare. Select a game above.</div>
      </Show>

      <Show when={rows().length > 0}>
        <div id="comparison-table-wrapper">
          <table id="comparison-table">
            <thead>
              <tr>
                <th>Strategy</th>
                <th class="col-older">{shortSha(aSha())}</th>
                <th class="col-newer">{shortSha(bSha())}</th>
                <th class="col-delta">Delta</th>
              </tr>
            </thead>
            <tbody>
              <For each={rows()}>
                {(row) => {
                  const wrA = row.a ? fmtRate(row.a.win_rate) : "—";
                  const wrB = row.b ? fmtRate(row.b.win_rate) : "—";
                  const deltaLabel = row.winRateDelta >= 0 ? `+${fmtRate(row.winRateDelta)}` : fmtRate(row.winRateDelta);
                  const deltaClass = row.winRateDelta > 0.02
                    ? "delta-up"
                    : row.winRateDelta < -0.02
                      ? "delta-down"
                      : "delta-flat";
                  return (
                    <tr class="comp-row">
                      <td class="comp-strategy">{row.strategy}</td>
                      <td class="comp-older">
                        {wrA}
                        <Show when={row.a}>
                          <span class="comp-ci">
                            {fmtRate(row.a!.ci_lower)}–{fmtRate(row.a!.ci_upper)}
                          </span>
                        </Show>
                      </td>
                      <td class="comp-newer">
                        {wrB}
                        <Show when={row.b}>
                          <span class="comp-ci">
                            {fmtRate(row.b!.ci_lower)}–{fmtRate(row.b!.ci_upper)}
                          </span>
                        </Show>
                      </td>
                      <td class={`comp-delta ${deltaClass}`}>{deltaLabel}</td>
                    </tr>
                  );
                }}
              </For>
            </tbody>
          </table>
        </div>
      </Show>
    </div>
  );
};