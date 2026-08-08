// RunDetailPanel.tsx — Open run's detail summary and live log tail.
//
// The log tail polls via the bench reducer's self-scheduling loop
// (tailTick/tailed/tailFailed actions).  This component only reads state
// and dispatches close/stop — it never manages timers itself.

import { createMemo, createSignal, For, Show, type Component } from "solid-js";
import type { Store } from "@mcts/core";
import type { BenchAction, BenchState } from "./index.js";

const MAX_VISIBLE_LINES = 500;

export const RunDetailPanel: Component<{
  store: Store<BenchState, BenchAction>;
}> = (props) => {
  const state = props.store.getState();
  const dispatch = props.store.dispatch;

  const openRun = createMemo(() => state().openRun);
  const detail = createMemo(() => openRun()?.detail ?? null);
  const tail = createMemo(() => openRun()?.tail ?? null);
  const stopError = createMemo(() => state().stopError);

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
          </div>
        </div>

        <Show when={stopError()}>
          <div class="launch-error">{stopError()}</div>
        </Show>

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
      </div>
    </Show>
  );
};