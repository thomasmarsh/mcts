import { createMemo, createSignal, For, Show, type Component } from "solid-js";
import type { Store } from "@mcts/core";
import type { BenchAction } from "../reducer.js";
import type { BenchState } from "../state.js";
import type { BenchSpectatorProps, TuningAllowedCommand, TuningSessionCommandKind, TuningSessionControl, TuningSessionDetail } from "../types.js";
import { TuningEvidenceDetail } from "./TuningEvidenceDetail.js";
import { TuningHierarchy } from "./TuningHierarchy.js";
import { TuningTrialsView } from "./TuningTrialsView.js";
import { TuningProgressView } from "./TuningProgressView.js";
import { TuningPruningView } from "./TuningPruningView.js";
import { TuningLadderView } from "./TuningLadderView.js";
import { TuningGameEvidence } from "./TuningGameEvidence.js";
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
          <Show when={detail()} fallback={<Show when={session()}>{(value) => <><div class="tuning-summary-status">Session status: {value().status}</div><div class="tuning-summary-counts">queued {value().counts.queued} · running {value().counts.running} · complete {value().counts.completed} · failed {value().counts.failed} · pruned {value().counts.pruned} · cancelled {value().counts.cancelled}</div></>}</Show>}>
            {(value) => <><div class="tuning-summary-status">Session status: {value().summary.status}</div><div class="tuning-summary-counts">queued {value().summary.counts.queued} · running {value().summary.counts.running} · complete {value().summary.counts.completed} · failed {value().summary.counts.failed} · pruned {value().summary.counts.pruned} · cancelled {value().summary.counts.cancelled}</div></>}
          </Show>
          <Show when={navigation().unavailable}><div class="tuning-unavailable" role="status">{navigation().unavailable}</div></Show>
          <Show when={navigation().detail.error}><div class="tuning-load-error" role="alert">{navigation().detail.error}</div></Show>
        </div>
        <Show when={detail()} fallback={<Show when={session()}>{(value) => <SessionProgress summary={value()} />}</Show>}>{(value) => <SessionProgress summary={value().summary} />}</Show>
        <Show when={session()?.control ?? detail()?.control}>{(control) => <SessionControls store={props.store} sessionId={navigation().selection.sessionId!} control={control()} command={navigation().commands[navigation().selection.sessionId!]} />}</Show>
        <button onClick={() => props.store.dispatch({ tag: "tuningNavigation", action: { tag: "clearSession" } })}>Close</button>
      </header>
      <div class="tuning-workbench-tabs" role="tablist" aria-label="Tuning session views">
        <button role="tab" aria-selected={navigation().tab === "progress"} onClick={() => props.store.dispatch({ tag: "tuningNavigation", action: { tag: "setAnalysisTab", tab: "progress" } })}>Progress</button>
        <button role="tab" aria-selected={navigation().tab === "pruning"} onClick={() => props.store.dispatch({ tag: "tuningNavigation", action: { tag: "setAnalysisTab", tab: "pruning" } })}>Pruning</button>
        <button role="tab" aria-selected={navigation().tab === "ladder"} onClick={() => props.store.dispatch({ tag: "tuningNavigation", action: { tag: "setAnalysisTab", tab: "ladder" } })}>Ladder</button>
        <button role="tab" aria-selected={navigation().tab === "trials"} onClick={() => props.store.dispatch({ tag: "tuningNavigation", action: { tag: "setAnalysisTab", tab: "trials" } })}>Trials</button>
        <button role="tab" aria-selected={navigation().tab === "game"} onClick={() => props.store.dispatch({ tag: "tuningNavigation", action: { tag: "setAnalysisTab", tab: "game" } })}>Game</button>
      </div>
      <Show when={navigation().tab === "progress"} fallback={<Show when={navigation().tab === "pruning"} fallback={<Show when={navigation().tab === "ladder"} fallback={<Show when={navigation().tab === "trials"} fallback={<Show when={session()?.capabilities.has_lifecycle} fallback={<Show when={detail()} fallback={<div class="loading-bench">Loading legacy tuning session…</div>}>{(value) => <div class="tuning-workbench-grid"><TuningHierarchy store={props.store} detail={value()} /><TuningEvidenceDetail store={props.store} detail={value()} session={session()} Spectator={props.Spectator} /></div>}</Show>}><TuningGameEvidence store={props.store} session={session()} Spectator={props.Spectator} /></Show>}><TuningTrialsView store={props.store} /></Show>}><TuningLadderView store={props.store} /></Show>}><TuningPruningView store={props.store} /></Show>}>
        <TuningProgressView store={props.store} />
      </Show>
    </main>
  );
};

let fallbackCommandSequence = 0;

/** A command id is created for one user submission and then retained by reducer state for retries. */
export function newTuningCommandId(): string {
  const value = globalThis.crypto?.randomUUID?.()
    ?? `${Date.now().toString(36)}-${(++fallbackCommandSequence).toString(36)}-${Math.random().toString(36).slice(2)}`;
  return `tuning-ui-${value}`;
}

const denialText = (reason: string | null): string => reason ? reason.replaceAll("_", " ") : "not allowed by the server";
const commandLabel = (kind: TuningSessionCommandKind): string => kind === "add_budget" ? "Add N trials" : kind[0]!.toUpperCase() + kind.slice(1);

const SessionControls: Component<{
  store: Store<BenchState, BenchAction>;
  sessionId: string;
  control: TuningSessionControl;
  command: import("../tuning-navigation.js").TuningSessionCommandState | undefined;
}> = (props) => {
  const [delta, setDelta] = createSignal("1");
  const [start, setStart] = createSignal(false);
  const [workers, setWorkers] = createSignal("");
  const pending = () => props.command?.status === "pending";
  const active = () => props.control.continuation.active_attempt_id !== null;
  const parsedDelta = () => /^\d+$/.test(delta()) ? Number(delta()) : Number.NaN;
  const parsedWorkers = () => workers() === "" ? undefined : (/^\d+$/.test(workers()) ? Number(workers()) : Number.NaN);
  const validation = () => {
    const amount = parsedDelta();
    const target = props.control.continuation.target_trial_count;
    if (!Number.isSafeInteger(amount) || amount < 1) return "Enter a positive whole number of trials.";
    if (target === null) return "The server did not project a current trial target.";
    if (!Number.isSafeInteger(target + amount)) return "The resulting target must be a safe integer.";
    if (start()) {
      const count = parsedWorkers();
      if (count !== undefined && (!Number.isSafeInteger(count) || count < 1 || count > 1024)) return "Workers must be a whole number from 1 to 1024.";
    }
    return null;
  };
  const submit = (kind: TuningSessionCommandKind) => {
    const error = kind === "add_budget" ? validation() : null;
    if (error || pending()) return;
    props.store.dispatch({
      tag: "tuningNavigation",
      action: {
        tag: "sessionCommandSubmit", sessionId: props.sessionId, kind, commandId: newTuningCommandId(), expectedVersion: props.control.version,
        ...(kind === "add_budget" ? { delta: parsedDelta(), start: start(), nWorkers: start() ? parsedWorkers() : undefined } : {}),
      },
    });
  };
  const announcement = () => {
    const command = props.command;
    if (!command) return null;
    const label = commandLabel(command.kind);
    if (command.status === "pending") return `${label} pending.`;
    if (command.status === "failed") return `${label} failed: ${command.error ?? "request failed"}`;
    return `${label} succeeded${command.response?.replay ? " (replayed request)" : ""}.`;
  };
  return (
    <section class="tuning-session-controls" aria-label="Session controls">
      <For each={props.control.allowed_commands}>{(allowed) => <ProjectedCommand
        allowed={allowed}
        pending={pending()}
        active={active()}
        validation={validation()}
        target={props.control.continuation.target_trial_count}
        delta={delta()}
        start={start()}
        workers={workers()}
        onDelta={setDelta}
        onStart={setStart}
        onWorkers={setWorkers}
        onSubmit={submit}
      />}</For>
      <Show when={announcement()}>{(message) => <div class="tuning-command-announcement" role={props.command?.status === "failed" ? "alert" : "status"} aria-live={props.command?.status === "failed" ? "assertive" : "polite"}>{message()}</div>}</Show>
      <Show when={props.command?.status === "succeeded" ? props.command.response?.attempt_id ?? null : null}>{(attemptId) => <div class="tuning-command-attempt">New attempt {attemptId()} is available. <button type="button" onClick={() => props.store.dispatch({ tag: "tuningNavigation", action: { tag: "openCommandAttempt", sessionId: props.sessionId, attemptId: attemptId() } })}>Open attempt</button></div>}</Show>
      <Show when={props.command?.status === "failed" && props.command.retriable}><button type="button" onClick={() => props.store.dispatch({ tag: "tuningNavigation", action: { tag: "sessionCommandRetry", sessionId: props.sessionId } })}>Retry {commandLabel(props.command!.kind)}</button></Show>
    </section>
  );
};

const ProjectedCommand: Component<{
  allowed: TuningAllowedCommand;
  pending: boolean;
  active: boolean;
  validation: string | null;
  target: number | null;
  delta: string;
  start: boolean;
  workers: string;
  onDelta: (value: string) => void;
  onStart: (value: boolean) => void;
  onWorkers: (value: string) => void;
  onSubmit: (kind: TuningSessionCommandKind) => void;
}> = (props) => {
  const unavailable = () => !props.allowed.allowed;
  if (props.allowed.command !== "add_budget") return (
    <div class="tuning-command-control">
      <button type="button" disabled={unavailable() || props.pending} onClick={() => props.onSubmit(props.allowed.command)}>{commandLabel(props.allowed.command)}</button>
      <Show when={unavailable()}><p class="tuning-command-denial">{commandLabel(props.allowed.command)} unavailable: {denialText(props.allowed.denial_reason)}</p></Show>
    </div>
  );
  const preview = () => {
    const amount = /^\d+$/.test(props.delta) ? Number(props.delta) : Number.NaN;
    return props.target !== null && Number.isSafeInteger(amount) ? `${props.target} + ${amount} = ${props.target + amount}` : "Enter a positive delta to preview the new target.";
  };
  return (
    <fieldset class="tuning-command-control tuning-budget-control" disabled={props.pending || unavailable()}>
      <legend>Add N trials</legend>
      <label>Trials to add <input name="trial-delta" inputmode="numeric" value={props.delta} onInput={(event) => props.onDelta(event.currentTarget.value)} aria-describedby="tuning-budget-preview tuning-budget-validation" /></label>
      <output id="tuning-budget-preview" class="tuning-budget-preview">{preview()}</output>
      <Show when={!props.active}><label class="tuning-start-attempt"><input type="checkbox" checked={props.start} onChange={(event) => props.onStart(event.currentTarget.checked)} /> Start a new attempt</label></Show>
      <Show when={props.start && !props.active}><label>Workers (optional) <input name="workers" inputmode="numeric" value={props.workers} onInput={(event) => props.onWorkers(event.currentTarget.value)} /></label></Show>
      <Show when={props.validation}><p id="tuning-budget-validation" class="tuning-command-validation" role="alert">{props.validation}</p></Show>
      <button type="button" disabled={Boolean(props.validation)} onClick={() => props.onSubmit("add_budget")}>{props.start && !props.active ? "Add N trials and start" : "Add N trials"}</button>
      <Show when={unavailable()}><p class="tuning-command-denial">Add N trials unavailable: {denialText(props.allowed.denial_reason)}</p></Show>
    </fieldset>
  );
};

const SessionProgress: Component<{ summary: TuningSessionDetail["summary"] }> = (props) => {
  const target = () => props.summary.target_trial_count;
  const terminal = () => props.summary.counts.terminal;
  return (
    <div class="tuning-summary-progress">
      <span>{terminal()} {target() === null ? "terminal trials" : `/ ${target()} terminal trials`}</span>
      <Show when={target() !== null && target()! > 0}>
        <progress value={Math.min(terminal(), target()!)} max={target()!} aria-label={`${terminal()} of ${target()} terminal trials`} />
      </Show>
    </div>
  );
};
