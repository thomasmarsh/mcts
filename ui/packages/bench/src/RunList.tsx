// RunList.tsx — Run list with status badges, filters, row selection, and a
// "New Run" button in the header.
//
// Displays the runs fetched from the bench store, with clickable rows that
// dispatch `openRun` to view the detail/log-tail panel.  Filter controls
// (status, game) dispatch `setRunFilters` which triggers a refetch.

import { createMemo, createSignal, For, Show, type Component } from "solid-js";
import type { Store } from "@mcts/core";
import type { BenchAction } from "./reducer.js";
import type { BenchState } from "./state.js";
import type { RunSummary, RunStatus, TuningSessionListItem } from "./types.js";
import { formatTimestamp, sessionLabel, terminalProgress } from "./tuning/tuning-view-model.js";

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
  const sessions = createMemo(() => state().tuningNavigation.list.snapshot?.sessions ?? []);
  const sessionsStatus = createMemo(() => state().tuningNavigation.list.status);
  const sessionsError = createMemo(() => state().tuningNavigation.list.error);
  const selectedSessionId = createMemo(() => state().tuningNavigation.selection.sessionId);
  const projectedTunerRunIds = createMemo(
    () =>
      new Set(
        sessions().flatMap((session) =>
          session.attempts.flatMap((attempt) =>
            attempt.bench_run_id ? [attempt.bench_run_id] : [],
          ),
        ),
      ),
  );
  const visibleRuns = createMemo(() =>
    runs().filter((run) => run.kind !== "tuner" || !projectedTunerRunIds().has(run.run_id)),
  );
  const showLaunchForm = createMemo(() => state().showLaunchForm);
  const runFilters = createMemo(() => state().runFilters);
  const busy = createMemo(() => runsStatus() === "pending" || sessionsStatus() === "loading");

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

      <Show when={sessionsError()}>
        {(error) => (
          <div class="tuning-list-error" role="alert">
            Tuning sessions could not refresh: {error()}
          </div>
        )}
      </Show>

      <Show
        when={sessions().length > 0 || visibleRuns().length > 0}
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
          <For each={sessions()}>
            {(session) => (
              <TuningSessionRow
                store={props.store}
                session={session}
                selected={selectedSessionId() === session.session_id}
              />
            )}
          </For>
          <For each={visibleRuns()}>
            {(run) => (
              <RunRow
                run={run}
                isOpen={openRunId() === run.run_id}
                legacyTuner={run.kind === "tuner" && run.tuning_session_id == null}
                modernTuner={run.kind === "tuner" && run.tuning_session_id != null}
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
  legacyTuner: boolean;
  modernTuner: boolean;
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
      <Show
        when={props.legacyTuner}
        fallback={
          <Show
            when={props.modernTuner}
            fallback={<span class="run-row-matches">{props.run.match_count} matches</span>}
          >
            <span class="modern-tuner-label">Tuning attempt</span>
          </Show>
        }
      >
        <span class="legacy-tuner-label">Legacy tuner run</span>
      </Show>
      <span class="run-row-time">{started()}</span>
      <span class="run-row-id">{props.run.run_id.slice(0, 30)}</span>
    </div>
  );
};

const TuningSessionRow: Component<{
  store: Store<BenchState, BenchAction>;
  session: TuningSessionListItem;
  selected: boolean;
}> = (props) => {
  const state = props.store.getState();
  const expansionId = () => `session:${props.session.session_id}`;
  const expanded = () => state().tuningNavigation.expandedIds.includes(expansionId());

  function selectSession(): void {
    props.store.dispatch({ tag: "closeRun" });
    props.store.dispatch({
      tag: "tuningNavigation",
      action: { tag: "selectSession", sessionId: props.session.session_id },
    });
  }

  function selectAttempt(attemptId: string): void {
    if (state().tuningNavigation.selection.sessionId !== props.session.session_id) selectSession();
    props.store.dispatch({ tag: "tuningNavigation", action: { tag: "selectAttempt", attemptId } });
  }

  return (
    <section class="tuning-session-row" classList={{ "tuning-session-row-open": props.selected }}>
      <div class="tuning-session-main">
        <button
          class="tuning-session-disclosure"
          aria-label={`${expanded() ? "Collapse" : "Expand"} attempts for ${sessionLabel(props.session)}`}
          aria-expanded={expanded()}
          onClick={() =>
            props.store.dispatch({
              tag: "tuningNavigation",
              action: { tag: "toggleExpanded", id: expansionId() },
            })
          }
        >
          {expanded() ? "−" : "+"}
        </button>
        <button
          class="tuning-session-select"
          aria-current={props.selected ? "page" : undefined}
          onClick={selectSession}
        >
          <span class="tuning-session-label">{sessionLabel(props.session)}</span>
          <span class="tuning-session-progress">{terminalProgress(props.session)}</span>
        </button>
        <span
          class={`status-badge ${props.session.status === "active" ? "badge-running" : "badge-stopped"}`}
        >
          {props.session.status}
        </span>
        <span class="tuning-session-counts">
          {props.session.counts.running} active · {props.session.counts.completed} complete ·{" "}
          {props.session.counts.failed} failed · {props.session.counts.pruned} pruned
        </span>
        <span class="tuning-session-time">
          Updated {formatTimestamp(props.session.last_activity_at)}
        </span>
      </div>
      <Show when={expanded()}>
        <TuningAttemptList session={props.session} onSelect={selectAttempt} />
      </Show>
    </section>
  );
};

const TuningAttemptList: Component<{
  session: TuningSessionListItem;
  onSelect: (attemptId: string) => void;
}> = (props) => (
  <ul class="tuning-attempt-list" aria-label={`${sessionLabel(props.session)} attempts`}>
    <For each={props.session.attempts}>
      {(attempt) => (
        <li>
          <button class="tuning-attempt-row" onClick={() => props.onSelect(attempt.attempt_id)}>
            <span>{attempt.status}</span>
            <span>Attempt {attempt.attempt_id.slice(0, 12)}</span>
            <span class="tuning-attempt-meta">{formatTimestamp(attempt.started_at)}</span>
          </button>
        </li>
      )}
    </For>
  </ul>
);
