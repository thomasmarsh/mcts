// RunDetailPanel.tsx — Open run's detail summary and live log tail.
//
// The log tail polls via the bench reducer's self-scheduling loop
// (tailTick/tailed/tailFailed actions).  This component only reads state
// and dispatches close/stop — it never manages timers itself.

import { createMemo, createSignal, For, Show, type Component } from "solid-js";
import type { Store } from "@mcts/core";
import { createBenchApiClient } from "./api-client.js";
import type { BenchAction, BenchState } from "./index.js";
import type { BenchSpectatorProps } from "./types.js";

const MAX_VISIBLE_LINES = 500;

function configOverride(config: unknown, key: string): number | null {
  const overrides = (config as { overrides?: unknown } | null)?.overrides;
  if (!Array.isArray(overrides)) return null;
  for (const override of overrides) {
    if (typeof override !== "string" || !override.startsWith(`${key}=`)) continue;
    const value = Number(override.slice(key.length + 1));
    if (Number.isFinite(value) && value > 0) return value;
  }
  return null;
}

function progress(detail: NonNullable<BenchState["openRun"]>["detail"]): {
  completed: number;
  total: number | null;
  workers: number | null;
} {
  if (!detail) return { completed: 0, total: null, workers: null };
  if (detail.kind === "tuner") {
    return {
      completed: detail.trial_count,
      total: configOverride(detail.config, "optimizer.n_trials"),
      workers: configOverride(detail.config, "optimizer.n_workers"),
    };
  }
  const config = detail.config as { strategies?: unknown; rounds?: unknown } | null;
  const strategies = Array.isArray(config?.strategies) ? config.strategies.length : 0;
  const rounds = typeof config?.rounds === "number" && config.rounds > 0 ? config.rounds : 1;
  return {
    completed: detail.match_count,
    total: strategies > 1 ? strategies * (strategies - 1) * rounds : null,
    workers: null,
  };
}

export const RunDetailPanel: Component<{
  store: Store<BenchState, BenchAction>;
  /** App-owned board panel: the bench package deliberately does not depend
   * on individual game renderers. */
  Spectator?: Component<BenchSpectatorProps>;
}> = (props) => {
  const Spectator = props.Spectator;
  const state = props.store.getState();
  const dispatch = props.store.dispatch;

  const openRun = createMemo(() => state().openRun);
  const detail = createMemo(() => openRun()?.detail ?? null);
  const tail = createMemo(() => openRun()?.tail ?? null);
  const stopError = createMemo(() => state().stopError);
  const deleteError = createMemo(() => state().deleteError);
  const [spectatorVisible, setSpectatorVisible] = createSignal(false);
  const [deleteArmed, setDeleteArmed] = createSignal(false);

  const isTuner = createMemo(() => detail()?.kind === "tuner");
  const tuningSessionId = createMemo(() => detail()?.tuning_session_id ?? null);
  const isModernTuningAttempt = createMemo(() => tuningSessionId() !== null);
  const runProgress = createMemo(() => progress(detail()));
  const progressPercent = createMemo(() => {
    const { completed, total } = runProgress();
    return total === null ? null : Math.min(100, Math.floor((completed / total) * 100));
  });

  // One-shot fetch for the raw stdout.log (stderr output).  Not part of
  // the reducer — this is a debug view fetched on demand.
  const [stdoutContent, setStdoutContent] = createSignal<string | null>(null);
  const [stdoutError, setStdoutError] = createSignal<string | null>(null);
  const [stdoutLoading, setStdoutLoading] = createSignal(false);
  const [stdoutVisible, setStdoutVisible] = createSignal(false);

  async function fetchStdout(): Promise<void> {
    const run = openRun();
    if (!run) return;
    setStdoutLoading(true);
    setStdoutError(null);
    try {
      const api = createBenchApiClient();
      const content = await api.getRunStdout(run.runId);
      setStdoutContent(content);
      setStdoutVisible(true);
    } catch (e: unknown) {
      setStdoutError(String(e));
    } finally {
      setStdoutLoading(false);
    }
  }

  // Auto-scroll to bottom when new lines arrive.
  let logEndRef: HTMLDivElement | undefined;
  const [userScrolledUp, setUserScrolledUp] = createSignal(false);
  let logContainer: HTMLDivElement | undefined;

  function onLogScroll(): void {
    if (!logContainer) return;
    const el = logContainer;
    const threshold = 60;
    setUserScrolledUp(el.scrollHeight - el.scrollTop - el.clientHeight > threshold);
  }

  // Scroll to bottom when lines grow, unless the user scrolled up.
  createMemo(() => {
    const lines = tail()?.lines ?? [];
    const _len = lines.length;
    if (!userScrolledUp() && logEndRef) {
      queueMicrotask(() => logEndRef?.scrollIntoView({ behavior: "auto" }));
    }
    return _len;
  });

  return (
    <Show when={openRun()} fallback={null}>
      <div id="run-detail-panel">
        <div id="run-detail-header">
          <h3>{isModernTuningAttempt() ? "Tuning attempt" : "Run Detail"}</h3>
          <div id="run-detail-actions">
            <Show when={detail()?.status === "running" && !isModernTuningAttempt()}>
              <button
                id="stop-run-btn"
                onClick={() => dispatch({ tag: "stopRun", runId: openRun()!.runId })}
              >
                Stop
              </button>
            </Show>
            <button id="close-run-btn" onClick={() => dispatch({ tag: "closeRun" })}>
              Close
            </button>
            <Show when={props.Spectator && !isModernTuningAttempt()}>
              <button id="watch-games-btn" onClick={() => setSpectatorVisible(!spectatorVisible())}>
                {spectatorVisible() ? "Hide games" : "Browse games"}
              </button>
            </Show>
            <Show when={detail()?.status !== "running" && !isModernTuningAttempt()}>
              <button
                id="delete-run-btn"
                onClick={() => {
                  if (deleteArmed()) dispatch({ tag: "deleteRun", runId: openRun()!.runId });
                  else setDeleteArmed(true);
                }}
              >
                {deleteArmed() ? "Confirm delete" : "Delete"}
              </button>
            </Show>
          </div>
        </div>

        <Show when={stopError()}>
          <div class="launch-error">{stopError()}</div>
        </Show>
        <Show when={deleteError()}>
          <div class="launch-error">{deleteError()}</div>
        </Show>

        <div id="run-detail-summary">
          <Show when={detail() && !isModernTuningAttempt()}>
            <div id="run-detail-meta">
              <div class="meta-row">
                <span class="meta-label">Run ID</span>
                <span class="meta-value">
                  <code>{detail()!.run_id}</code>
                </span>
              </div>
              <div class="meta-row">
                <span class="meta-label">Status</span>
                <span class="meta-value">{detail()!.status}</span>
              </div>
              <div class="meta-row">
                <span class="meta-label">Game</span>
                <span class="meta-value">{detail()!.game}</span>
              </div>
              <div class="meta-row">
                <span class="meta-label">Matches</span>
                <span class="meta-value">{detail()!.match_count}</span>
              </div>
              <div class="meta-row">
                <span class="meta-label">Trials</span>
                <span class="meta-value">{detail()!.trial_count}</span>
              </div>
              <div class="meta-row progress-row">
                <span class="meta-label">Progress</span>
                <span class="meta-value">
                  {runProgress().completed}
                  {runProgress().total !== null
                    ? ` / ${runProgress().total} (${progressPercent()}%) complete`
                    : " completed"}
                  <Show when={isTuner() && detail()!.status === "running"}>
                    {` · ${runProgress().workers ?? "auto"} workers`}
                  </Show>
                  <Show when={runProgress().total !== null}>
                    <progress
                      class="run-progress-bar"
                      value={runProgress().completed}
                      max={runProgress().total!}
                      aria-label="Run progress"
                    />
                  </Show>
                </span>
              </div>
              <div class="meta-row">
                <span class="meta-label">Git SHA</span>
                <span class="meta-value">
                  <code>{detail()!.git_sha.slice(0, 12)}</code>
                </span>
              </div>
              <Show when={detail()!.pid}>
                <div class="meta-row">
                  <span class="meta-label">PID</span>
                  <span class="meta-value">{detail()!.pid}</span>
                </div>
              </Show>
              <Show when={detail()!.ended_at}>
                <div class="meta-row">
                  <span class="meta-label">Ended</span>
                  <span class="meta-value">{detail()!.ended_at}</span>
                </div>
              </Show>
            </div>
          </Show>

          <Show when={isTuner() && !isModernTuningAttempt()}>
            <p class="tuning-run-diagnostics">
              This physical run keeps its log and diagnostics here. This legacy run has no logical
              session.
            </p>
          </Show>
          <Show when={isModernTuningAttempt() && detail()}>
            <section class="tuning-attempt-summary" aria-label="Tuning attempt">
              <div>
                <strong>{detail()!.status}</strong> tuning attempt
              </div>
              <p>Continue and analyze this work from its logical tuning session.</p>
              <button
                id="open-tuning-session-btn"
                type="button"
                onClick={() =>
                  dispatch({
                    tag: "tuningNavigation",
                    action: { tag: "selectSession", sessionId: tuningSessionId()! },
                  })
                }
              >
                Open tuning session
              </button>
              <details id="tuning-attempt-diagnostics">
                <summary>Attempt diagnostics</summary>
                <dl>
                  <div>
                    <dt>Run ID</dt>
                    <dd>
                      <code>{detail()!.run_id}</code>
                    </dd>
                  </div>
                  <Show when={detail()!.ended_at}>
                    <div>
                      <dt>Ended</dt>
                      <dd>{detail()!.ended_at}</dd>
                    </div>
                  </Show>
                  <Show when={detail()!.exit_code !== null}>
                    <div>
                      <dt>Exit code</dt>
                      <dd>{detail()!.exit_code}</dd>
                    </div>
                  </Show>
                </dl>
                <button id="show-stdout-btn" onClick={fetchStdout} disabled={stdoutLoading()}>
                  {stdoutLoading()
                    ? "Loading…"
                    : stdoutContent() !== null
                      ? "Refresh error output"
                      : "Show error output"}
                </button>
                <Show when={stdoutVisible() && stdoutContent() !== null}>
                  <pre id="stdout-content">{stdoutContent()}</pre>
                </Show>
                <Show when={stdoutError()}>
                  <div class="log-error">Error output fetch failed: {stdoutError()}</div>
                </Show>
              </details>
            </section>
          </Show>
        </div>

        <Show when={!isModernTuningAttempt() && spectatorVisible() && Spectator && detail()}>
          {Spectator ? (
            <Spectator
              runId={openRun()!.runId}
              game={detail()!.game ?? ""}
              kind={detail()!.kind}
              live={detail()!.status === "running"}
            />
          ) : null}
        </Show>

        <Show when={!isModernTuningAttempt()}>
          <div id="log-panel">
            <div id="log-header">
              <span>Log Tail</span>
              <Show when={tail()}>
                <span class="log-status">
                  {tail()!.active ? (
                    <>
                      Polling (<code>{tail()!.offset}</code> bytes)…
                    </>
                  ) : (
                    <>Complete ({tail()!.lines.length} lines)</>
                  )}
                </span>
              </Show>
            </div>
            <Show when={tail()?.error}>
              <div class="log-error">Error: {tail()!.error}</div>
            </Show>
            <div id="log-content" ref={logContainer} onScroll={onLogScroll}>
              <Show
                when={tail() && tail()!.lines.length > 0}
                fallback={<div class="log-empty">Waiting for log output…</div>}
              >
                <For each={tail()!.lines.slice(-MAX_VISIBLE_LINES)}>
                  {(line) => <div class="log-line">{line}</div>}
                </For>
                <div ref={logEndRef} />
              </Show>
            </div>
          </div>
        </Show>

        {/* Stdout button + content shown for every run kind — the raw
            stderr output (clap errors, panic traces) is the primary
            diagnostic for a crashed run, regardless of kind.
            TODO: rename the backend's file to avoid "stdout" for stderr output. */}
        <Show when={!isModernTuningAttempt()}>
          <div id="run-stdout-section">
            <button
              id="show-stdout-btn"
              onClick={fetchStdout}
              disabled={stdoutLoading()}
              title="Fetch the raw stdout.log (stderr output from the run process)"
            >
              {stdoutLoading()
                ? "Loading…"
                : stdoutContent() !== null
                  ? "Refresh Stderr Log"
                  : "Show Stderr Log"}
            </button>

            <Show when={stdoutVisible() && stdoutContent() !== null}>
              <div id="stdout-panel">
                <div id="stdout-header">
                  <span>Stdout Log (stderr output)</span>
                  <button onClick={() => setStdoutVisible(false)}>Hide</button>
                </div>
                <Show when={stdoutContent() && stdoutContent()!.length > 0}>
                  <pre id="stdout-content">{stdoutContent()}</pre>
                </Show>
                <Show when={stdoutContent() !== null && stdoutContent()!.length === 0}>
                  <div class="log-empty">stdout.log is empty</div>
                </Show>
              </div>
            </Show>

            <Show when={stdoutError()}>
              <div class="log-error">Stdout fetch error: {stdoutError()}</div>
            </Show>
          </div>
        </Show>
      </div>
    </Show>
  );
};
