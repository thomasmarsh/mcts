import { createMemo, Show, type Component } from "solid-js";
import type { Store } from "@mcts/core";
import type { BenchAction } from "../reducer.js";
import type { BenchState } from "../state.js";
import type { BenchSpectatorProps, TuningSessionDetail } from "../types.js";
import { TuningEvidenceDetail } from "./TuningEvidenceDetail.js";
import { TuningHierarchy } from "./TuningHierarchy.js";
import { TuningTrialsView } from "./TuningTrialsView.js";
import { TuningProgressView } from "./TuningProgressView.js";
import { TuningPruningView } from "./TuningPruningView.js";
import { sessionLabel } from "./tuning-view-model.js";

export const TuningSessionWorkbench: Component<{
  store: Store<BenchState, BenchAction>;
  Spectator?: Component<BenchSpectatorProps>;
}> = (props) => {
  const state = props.store.getState();
  const navigation = () => state().tuningNavigation;
  const detail = () => navigation().detail.snapshot;
  const session = createMemo(() => navigation().list.snapshot?.sessions.find((row) => row.session_id === navigation().selection.sessionId) ?? null);
  const title = () => session() ? sessionLabel(session()!) : `Tuning ${navigation().selection.sessionId ?? "session"}`;
  return (
    <main id="tuning-session-workbench">
      <header class="tuning-workbench-header">
        <div>
          <h3>{title()}</h3>
          <Show when={detail()}>{(value) => <><div class="tuning-summary-status">Session status: {value().summary.status}</div><div class="tuning-summary-counts">queued {value().summary.counts.queued} · running {value().summary.counts.running} · complete {value().summary.counts.completed} · failed {value().summary.counts.failed} · pruned {value().summary.counts.pruned} · cancelled {value().summary.counts.cancelled}</div></>}</Show>
          <Show when={navigation().unavailable}><div class="tuning-unavailable" role="status">{navigation().unavailable}</div></Show>
          <Show when={navigation().detail.error}><div class="tuning-load-error" role="alert">{navigation().detail.error}</div></Show>
        </div>
        <Show when={detail()}>{(value) => <SessionProgress detail={value()} />}</Show>
        <button onClick={() => props.store.dispatch({ tag: "tuningNavigation", action: { tag: "clearSession" } })}>Close</button>
      </header>
      <div class="tuning-workbench-tabs" role="tablist" aria-label="Tuning session views">
        <button role="tab" aria-selected={navigation().tab === "progress"} onClick={() => props.store.dispatch({ tag: "tuningNavigation", action: { tag: "setAnalysisTab", tab: "progress" } })}>Progress</button>
        <button role="tab" aria-selected={navigation().tab === "pruning"} onClick={() => props.store.dispatch({ tag: "tuningNavigation", action: { tag: "setAnalysisTab", tab: "pruning" } })}>Pruning</button>
        <button role="tab" aria-selected={navigation().tab === "trials"} onClick={() => props.store.dispatch({ tag: "tuningNavigation", action: { tag: "setAnalysisTab", tab: "trials" } })}>Trials</button>
        <button role="tab" aria-selected={navigation().tab === "game"} onClick={() => props.store.dispatch({ tag: "tuningNavigation", action: { tag: "setAnalysisTab", tab: "game" } })}>Game evidence</button>
      </div>
      <Show when={navigation().tab === "progress"} fallback={<Show when={navigation().tab === "pruning"} fallback={<Show when={navigation().tab === "trials"} fallback={<Show when={detail()} fallback={<div class="loading-bench">Loading tuning session…</div>}>{(value) => <div class="tuning-workbench-grid"><TuningHierarchy store={props.store} detail={value()} /><TuningEvidenceDetail store={props.store} detail={value()} session={session()} Spectator={props.Spectator} /></div>}</Show>}><TuningTrialsView store={props.store} /></Show>}><TuningPruningView store={props.store} /></Show>}>
        <TuningProgressView store={props.store} />
      </Show>
    </main>
  );
};

const SessionProgress: Component<{ detail: TuningSessionDetail }> = (props) => {
  const target = () => props.detail.summary.target_trial_count;
  const terminal = () => props.detail.summary.counts.terminal;
  return (
    <div class="tuning-summary-progress">
      <span>{terminal()} {target() === null ? "terminal trials" : `/ ${target()} terminal trials`}</span>
      <Show when={target() !== null && target()! > 0}>
        <progress value={Math.min(terminal(), target()!)} max={target()!} aria-label={`${terminal()} of ${target()} terminal trials`} />
      </Show>
    </div>
  );
};
