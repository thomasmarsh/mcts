// RunList.tsx — Round-robin run list with status badges, filters, row
// selection, and a "New Run" button in the header.
//
// Displays the runs fetched from the bench store, with clickable rows that
// dispatch `openRun` to view the detail/log-tail panel. Filter controls
// (status, game) dispatch `setRunFilters` which triggers a refetch. The
// version-4 tuner has its own surface (`TunerApp`); this list is the
// round-robin bench only.

import { createMemo, createSignal, For, Show, type Component } from "solid-js";
import type { Store } from "@mcts/core";
import type { BenchAction } from "./reducer.js";
import type { BenchState } from "./state.js";
import type { RunSummary, RunStatus } from "./types.js";

function statusBadgeClass(status: RunStatus): string {
  if (status === "running") return "badge-running";
  if (status === "completed") return "badge-completed";
  if (status === "crashed") return "badge-crashed";
  if (status === "stopped") return "badge-stopped";
  return "";
}

function statusLabel(status: RunStatus): string {
  if (status === "running") return "Running";
  if (status === "completed") return "Completed";
  if (status === "crashed") return "Crashed";
  if (status === "stopped") return "Stopped";
  return status;
}

export const RunList: Component<{
  store: Store<BenchState, BenchAction>;
  onNewRun: () => void;
}> = (props) => {
  const state = props.store.getState();
  const dispatch = props.store.dispatch;

  const runsStatus = createMemo(() => state().runs.status);
  const runs = createMemo(() =>
    state().runs.status === "done" ? (state().runs.result ?? []) : [],
  );
  const openRunId = createMemo(() => state().openRun?.runId ?? null);
  const showLaunchForm = createMemo(() => state().showLaunchForm);
  const runFilters = createMemo(() => state().runFilters);
  const busy = createMemo(() => runsStatus() === "pending");

  // Local filter state committed on apply.
  const [filterStatus, setFilterStatus] = createSignal(runFilters().status ?? "");
  const [filterGame, setFilterGame] = createSignal(runFilters().game ?? "");

  function applyFilters(): void {
    dispatch({
      tag: "setRunFilters",
      status: filterStatus() || null,
      game: filterGame() || null,
    });
  }

  function refresh(): void {
    dispatch({ tag: "runs", action: { tag: "request" } });
  }

  return (
    <div id="run-list-panel">
      <div id="run-list-header">
        <h3>Runs</h3>
        <div id="run-list-header-actions">
          <button
            id="new-run-btn"
            classList={{ "new-run-active": showLaunchForm() }}
            onClick={props.onNewRun}
          >
            {showLaunchForm() ? "Close" : "New Run"}
          </button>
          <button id="refresh-runs" onClick={refresh} disabled={busy()} title="Refresh">
            &#x21bb;
          </button>
        </div>
      </div>

      <div id="run-filters">
        <select value={filterStatus()} onChange={(e) => setFilterStatus(e.currentTarget.value)}>
          <option value="">All statuses</option>
          <option value="running">Running</option>
          <option value="completed">Completed</option>
          <option value="crashed">Crashed</option>
          <option value="stopped">Stopped</option>
        </select>
        <input
          type="text"
          placeholder="Game filter…"
          value={filterGame()}
          onInput={(e) => setFilterGame(e.currentTarget.value)}
        />
        <button id="apply-filters" onClick={applyFilters}>
          Apply
        </button>
      </div>

      <Show
        when={runs().length > 0}
        fallback={
          <Show
            when={busy()}
            fallback={<div class="run-list-empty">No runs yet. Click "New Run" above!</div>}
          >
            <div class="loading-bench">Loading runs…</div>
          </Show>
        }
      >
        <div id="run-list-scroll">
          <For each={runs()}>
            {(run) => (
              <RunRow
                run={run}
                isOpen={openRunId() === run.run_id}
                onClick={() => dispatch({ tag: "openRun", runId: run.run_id })}
              />
            )}
          </For>
        </div>
      </Show>
    </div>
  );
};

/** A single run row in the list. */
const RunRow: Component<{
  run: RunSummary;
  isOpen: boolean;
  onClick: () => void;
}> = (props) => {
  const started = createMemo(() => {
    const raw = props.run.started_at;
    try {
      return new Date(raw).toLocaleString();
    } catch {
      return raw;
    }
  });

  return (
    <div class="run-row" classList={{ "run-row-open": props.isOpen }} onClick={props.onClick}>
      <span class={`status-badge ${statusBadgeClass(props.run.status)}`}>
        {statusLabel(props.run.status)}
      </span>
      <span class="run-row-game">{props.run.game}</span>
      <span class="run-row-matches">{props.run.match_count} matches</span>
      <span class="run-row-time">{started()}</span>
      <span class="run-row-id">{props.run.run_id.slice(0, 30)}</span>
    </div>
  );
};
