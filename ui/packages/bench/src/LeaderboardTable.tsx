// LeaderboardTable.tsx — Win-rate leaderboard with Wilson CI,
// sortable/filterable by game and git commit SHA.
//
// Reads the `leaderboard` / `leaderboardFilters` slices of BenchState
// and dispatches `setLeaderboardFilters` / `leaderboard/request` through
// the store. No direct API calls — the hard rule is enforced by the
// fetch-ban eslint rule.

import { createMemo, createSignal, For, Show, type Component } from "solid-js";
import type { Store } from "@mcts/core";
import type { BenchAction, BenchState } from "./index.js";
import { formatInterval, formatLeaderboardResult, formatRate, formatWld } from "./result-format.js";

export const LeaderboardTable: Component<{
  store: Store<BenchState, BenchAction>;
}> = (props) => {
  const state = props.store.getState();
  const dispatch = props.store.dispatch;

  const status = createMemo(() => state().leaderboard.status);
  const entries = createMemo(() => (status() === "done" ? state().leaderboard.result ?? [] : []));
  const filters = createMemo(() => state().leaderboardFilters);

  // Local filter state committed on "Apply".
  const [filterGame, setFilterGame] = createSignal(filters().game ?? "");
  const [filterSha, setFilterSha] = createSignal(filters().gitSha ?? "");
  const [filterSince, setFilterSince] = createSignal(filters().since ?? "");

  // Sort: by win_rate desc by default, toggles per column.
  const [sortKey, setSortKey] = createSignal<"win_rate" | "strategy" | "total" | "wins">("win_rate");
  const [sortAsc, setSortAsc] = createSignal(false);

  const sorted = createMemo(() => {
    const list = [...entries()];
    const key = sortKey();
    const asc = sortAsc();
    list.sort((a, b) => {
      let cmp: number;
      switch (key) {
        case "strategy":
          cmp = a.strategy.localeCompare(b.strategy);
          break;
        case "total":
          cmp = a.total - b.total;
          break;
        case "wins":
          cmp = a.wins - b.wins;
          break;
        default:
          cmp = a.win_rate - b.win_rate;
      }
      return asc ? cmp : -cmp;
    });
    return list;
  });

  function toggleSort(key: string): void {
    if (sortKey() === key) {
      setSortAsc((a) => !a);
    } else {
      setSortKey(key as "win_rate" | "strategy" | "total" | "wins");
      setSortAsc(false);
    }
  }

  function sortArrow(col: string): string {
    if (sortKey() !== col) return "";
    return sortAsc() ? " ▲" : " ▼";
  }

  function applyFilters(): void {
    dispatch({
      tag: "setLeaderboardFilters",
      game: filterGame() || null,
      gitSha: filterSha() || null,
      since: filterSince() || null,
    });
  }

  function refresh(): void {
    dispatch({ tag: "leaderboard", action: { tag: "request" } });
  }

  return (
    <div id="leaderboard-panel">
      <div id="leaderboard-header">
        <h3>Leaderboard</h3>
        <button id="refresh-leaderboard" onClick={refresh} disabled={status() === "pending"} title="Refresh">
          &#x21bb;
        </button>
      </div>

      <div id="leaderboard-filters">
        <input
          type="text"
          placeholder="Game…"
          value={filterGame()}
          onInput={(e) => setFilterGame(e.currentTarget.value)}
        />
        <input
          type="text"
          placeholder="Git SHA…"
          value={filterSha()}
          onInput={(e) => setFilterSha(e.currentTarget.value)}
        />
        <input
          type="text"
          placeholder="Since (ISO date)…"
          value={filterSince()}
          onInput={(e) => setFilterSince(e.currentTarget.value)}
        />
        <button id="lb-apply-filters" onClick={applyFilters}>
          Apply
        </button>
      </div>

      <Show
        when={status() === "done"}
        fallback={
          <Show when={status() === "pending"} fallback={<div class="lb-empty">No data yet.</div>}>
            <div class="loading-bench">Loading…</div>
          </Show>
        }
      >
        <Show
          when={sorted().length > 0}
          fallback={<div class="lb-empty">No leaderboard entries match the filters.</div>}
        >
          <div id="leaderboard-scroll">
            <table id="leaderboard-table">
              <thead>
                <tr>
                  <th onClick={() => toggleSort("strategy")} class="sortable">
                    Strategy{sortArrow("strategy")}
                  </th>
                  <th onClick={() => toggleSort("total")} class="sortable">
                    Games{sortArrow("total")}
                  </th>
                  <th onClick={() => toggleSort("wins")} class="sortable">
                    W/L/D{sortArrow("wins")}
                  </th>
                  <th onClick={() => toggleSort("win_rate")} class="sortable">
                    Win Rate{sortArrow("win_rate")}
                  </th>
                  <th>Wilson CI (95%)</th>
                </tr>
              </thead>
              <tbody>
                <For each={sorted()}>
                  {(entry) => (
                    <tr class="lb-row">
                      <td class="lb-strategy">{entry.strategy}</td>
                      <td class="lb-total">{entry.total}</td>
                      <td class="lb-wld">
                        {formatWld(entry)}
                      </td>
                      <td class="lb-rate">
                        <div class="lb-bar-bg">
                          <div
                            class="lb-bar-fill"
                            style={{ width: `${entry.total === 0 ? 0 : entry.win_rate * 100}%` }}
                          />
                        </div>
                        <span class="lb-rate-text">{entry.total === 0 ? formatLeaderboardResult(entry) : formatRate(entry.win_rate)}</span>
                      </td>
                      <td class="lb-ci">{entry.total === 0 ? formatLeaderboardResult(entry) : formatInterval(entry.ci_lower, entry.ci_upper)}</td>
                    </tr>
                  )}
                </For>
              </tbody>
            </table>
          </div>
        </Show>
      </Show>
    </div>
  );
};
