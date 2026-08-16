// RunDetailPanel.tsx — Open run's detail summary and live log tail.
//
// The log tail polls via the bench reducer's self-scheduling loop
// (tailTick/tailed/tailFailed actions).  This component only reads state
// and dispatches close/stop — it never manages timers itself.

import { createEffect, createMemo, createSignal, For, Show, type Component } from "solid-js";
import type { Store } from "@mcts/core";
import { createBenchApiClient } from "./api-client.js";
import type { BenchAction, BenchState } from "./index.js";
import { Smac3RunDetail } from "./Smac3RunDetail.js";

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

function progress(detail: NonNullable<BenchState["openRun"]>["detail"]): { completed: number; total: number | null; workers: number | null } {
  if (!detail) return { completed: 0, total: null, workers: null };
  if (detail.kind === "smac3") {
    return {
      completed: detail.trial_count,
      total: configOverride(detail.config, "optimizer.n_trials"),
      workers: configOverride(detail.config, "optimizer.n_workers"),
    };
  }
  const config = detail.config as { strategies?: unknown; rounds?: unknown } | null;
  const strategies = Array.isArray(config?.strategies) ? config.strategies.length : 0;
  const rounds = typeof config?.rounds === "number" && config.rounds > 0 ? config.rounds : 1;
  return { completed: detail.match_count, total: strategies > 1 ? strategies * (strategies - 1) * rounds : null, workers: null };
}

export const RunDetailPanel: Component<{
  store: Store<BenchState, BenchAction>;
  /** App-owned board panel: the bench package deliberately does not depend
   * on individual game renderers. */
  Spectator?: Component<{ runId: string; game: string; kind: string; live: boolean }>;
}> = (props) => {
  const Spectator = props.Spectator;
  const state = props.store.getState();
  const dispatch = props.store.dispatch;

  const openRun = createMemo(() => state().openRun);
  const detail = createMemo(() => openRun()?.detail ?? null);
  const tail = createMemo(() => openRun()?.tail ?? null);
  const stopError = createMemo(() => state().stopError);
  const resumeError = createMemo(() => state().resumeError);
  const advanceBaselineError = createMemo(() => state().advanceBaselineError);
  const deleteError = createMemo(() => state().deleteError);
  const [spectatorVisible, setSpectatorVisible] = createSignal(false);
  const [deleteArmed, setDeleteArmed] = createSignal(false);

  const isSmac3 = createMemo(() => detail()?.kind === "smac3");
  const runProgress = createMemo(() => progress(detail()));
  const progressPercent = createMemo(() => {
    const { completed, total } = runProgress();
    return total === null ? null : Math.min(100, Math.floor((completed / total) * 100));
  });
  // A run can only be resumed once it's stopped producing new trials --
  // resuming a still-running one would launch a second process racing the
  // first over the same prior state.
  const canResume = createMemo(() => isSmac3() && detail() !== null && detail()!.status !== "running");
  // A reasonable starting point for "how many more trials" -- the operator
  // can always change it before clicking Resume.
  const resumeDefaultTrials = createMemo(() => (detail()?.trial_count ?? 0) + 200);
  const [resumeTrials, setResumeTrials] = createSignal<number | null>(null);
  // Unlike Resume, this is available while the run is still `running` --
  // the advance-baseline route stops it first itself (see server/bench's
  // `advance_baseline` doc comment), so the operator doesn't have to click
  // Stop and wait before promoting the current incumbent. The only real
  // precondition is having an incumbent to promote at all.
  const canAdvanceBaseline = createMemo(() => isSmac3() && detail()?.incumbent != null);
  const [advancing, setAdvancing] = createSignal(false);
  // Clears the optimistic "in flight" flag once the request settles --
  // success replaces `openRun` with the new rung (see reducer.ts's
  // `advanceBaselineFinished`), failure surfaces `advanceBaselineError`.
  createEffect(() => {
    const _error = advanceBaselineError();
    const _runId = openRun()?.runId;
    void _error;
    void _runId;
    setAdvancing(false);
  });
  const smac3Tuner = createMemo(() => {
    const d = detail();
    const kinds = state().smac3Kinds;
    if (!d || kinds.status !== "done") return null;
    return kinds.result?.find((g) => g.game === d.game)?.tuner ?? null;
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
          <h3>Run Detail</h3>
          <div id="run-detail-actions">
            <Show when={detail()?.status === "running"}>
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
            <Show when={props.Spectator}>
              <button id="watch-games-btn" onClick={() => setSpectatorVisible(!spectatorVisible())}>
                {spectatorVisible() ? "Hide games" : "Browse games"}
              </button>
            </Show>
            <Show when={detail()?.status !== "running"}>
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

        <Show when={canResume()}>
          <div id="resume-run-row">
            <label for="resume-n-trials">Resume with n_trials</label>
            <input
              id="resume-n-trials"
              type="number"
              min="1"
              value={resumeTrials() ?? resumeDefaultTrials()}
              onInput={(e) => setResumeTrials(Number(e.currentTarget.value))}
            />
            <button
              id="resume-run-btn"
              onClick={() =>
                dispatch({
                  tag: "resumeRun",
                  runId: openRun()!.runId,
                  nTrials: resumeTrials() ?? resumeDefaultTrials(),
                })
              }
            >
              Resume
            </button>
          </div>
          <Show when={resumeError()}>
            <div class="launch-error">{resumeError()}</div>
          </Show>
        </Show>

        <Show when={canAdvanceBaseline()}>
          <div id="advance-baseline-row">
            <button
              id="advance-baseline-btn"
              disabled={advancing()}
              title="Promote this run's current incumbent to a new baseline instance and continue tuning against it -- stops the run first if it's still running."
              onClick={() => {
                setAdvancing(true);
                dispatch({ tag: "advanceBaseline", runId: openRun()!.runId });
              }}
            >
              {advancing() ? "Advancing…" : "Use best as new baseline"}
            </button>
          </div>
          <Show when={advanceBaselineError()}>
            <div class="launch-error">{advanceBaselineError()}</div>
          </Show>
        </Show>

        <div id="run-detail-summary">
          <Show when={detail()}>
            <div id="run-detail-meta">
              <div class="meta-row">
                <span class="meta-label">Run ID</span>
                <span class="meta-value"><code>{detail()!.run_id}</code></span>
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
                  {runProgress().completed}{runProgress().total !== null ? ` / ${runProgress().total} (${progressPercent()}%) complete` : " completed"}
                  <Show when={isSmac3() && detail()!.status === "running"}>
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
                <span class="meta-value"><code>{detail()!.git_sha.slice(0, 12)}</code></span>
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

          <Show when={isSmac3()}>
            <Smac3RunDetail
              trials={openRun()?.trials ?? []}
              chain={openRun()?.chain ?? []}
              chainedTrials={openRun()?.chainedTrials ?? []}
              tuner={smac3Tuner()}
              launchConfig={detail()?.config ?? null}
              incumbent={detail()?.incumbent ?? null}
            />
          </Show>
        </div>

        <Show when={spectatorVisible() && Spectator && detail()}>
          {Spectator ? <Spectator runId={openRun()!.runId} game={detail()!.game ?? ""} kind={detail()!.kind} live={detail()!.status === "running"} /> : null}
        </Show>

        <div id="log-panel">
          <div id="log-header">
            <span>Log Tail</span>
            <Show when={tail()}>
              <span class="log-status">
                {tail()!.active ? (
                  <>Polling (<code>{tail()!.offset}</code> bytes)…</>
                ) : (
                  <>Complete ({tail()!.lines.length} lines)</>
                )}
              </span>
            </Show>
            <button
              id="show-stdout-btn"
              onClick={fetchStdout}
              disabled={stdoutLoading()}
              title="Fetch the raw stdout.log (stderr output from the run process)"
            >
              {stdoutLoading() ? "Loading…" : stdoutContent() !== null ? "Refresh Stdout" : "Show Stdout Log"}
            </button>
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
  );
};
