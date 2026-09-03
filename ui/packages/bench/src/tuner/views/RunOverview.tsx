// RunOverview — a run's landing view and the ship decision. Header + live
// progress rail + launch-log tail (the sub-second feedback the old UI
// lacked), then, once the projection has the run, the ship verdict and the
// ranked validation table. Cohort race strips and the full science /
// evidence views are their own slices.

import { createMemo, For, Show, type Component } from "solid-js";
import type { Store } from "@mcts/core";
import { peek } from "../remote-data.js";
import type { TunerAction, TunerState } from "../tuner-reducer.js";
import type { TunerRoute } from "../tuner-routes.js";
import { deriveProgress } from "../models/progress-model.js";
import { deriveVerdict } from "../models/verdict-model.js";
import { schemaDefaults } from "../models/config-diff-model.js";
import { foldEvidence, tickerLines } from "../models/evidence-fold.js";
import { deriveConvergence } from "../models/science-models.js";
import { RunStatusBadge } from "../primitives/RunStatusBadge.js";
import { ProgressRail } from "../primitives/ProgressRail.js";
import { EventTicker } from "../primitives/EventTicker.js";
import { StepLine } from "../primitives/StepLine.js";
import { ShipVerdict } from "../primitives/ShipVerdict.js";
import { Forest } from "../primitives/Forest.js";
import { DataTable } from "../primitives/DataTable.js";
import type { VerdictCandidate } from "../models/verdict-model.js";

export const RunOverview: Component<{
  store: Store<TunerState, TunerAction>;
  runId: string;
  navigate: (route: TunerRoute) => void;
}> = (props) => {
  const state = props.store.getState();
  const dispatch = props.store.dispatch;

  const journalRow = createMemo(() =>
    (peek(state().runs) ?? []).find((r) => r.run_id === props.runId),
  );
  const projectionRow = createMemo(() =>
    (peek(state().projectionRuns) ?? []).find((r) => r.run_id === props.runId),
  );
  const detail = createMemo(() => peek(state().projectionDetail));
  const status = createMemo(() => journalRow()?.status ?? (projectionRow() ? "exited" : null));
  const live = createMemo(() => status() === "live");

  const evidenceRing = createMemo(() => state().evidence.ring);
  const liveProgress = createMemo(() => (live() ? foldEvidence(evidenceRing()) : null));
  const ticker = createMemo(() => tickerLines(evidenceRing(), 200));

  // A compact live "is it improving" signal without opening Science: one step
  // per cohort, from the projection candidate + observation rows.
  const cohorts = createMemo(() => {
    const byIndex = new Map<number, string[]>();
    for (const c of peek(state().candidates) ?? []) {
      const list = byIndex.get(c.cohort_index) ?? [];
      list.push(c.candidate_id);
      byIndex.set(c.cohort_index, list);
    }
    return [...byIndex.entries()]
      .sort(([a], [b]) => a - b)
      .map(([cohort_index, candidate_ids]) => ({
        cohort_index,
        candidate_ids,
        retained_candidate_ids: [],
      }));
  });
  const convergence = createMemo(() =>
    deriveConvergence(peek(state().report), cohorts(), peek(state().observations) ?? []),
  );

  const progress = createMemo(() =>
    deriveProgress({
      status: status(),
      startedAt: journalRow()?.started_at ?? null,
      nowMs: Date.now(),
      compute: detail()?.compute,
    }),
  );

  const gameKind = createMemo(
    () => detail()?.manifest?.game_kind ?? projectionRow()?.game_kind ?? null,
  );
  const baseConfig = createMemo(() => {
    const info = (peek(state().kinds) ?? []).find((k) => k.game === gameKind());
    return info ? schemaDefaults(info.tuner.parameters) : {};
  });
  const verdict = createMemo(() =>
    deriveVerdict({
      validation: peek(state().validation),
      candidates: peek(state().candidates),
      report: peek(state().report),
    }),
  );

  const openCandidate = (candidateId: string): void =>
    props.navigate({ view: "run", runId: props.runId, tab: "overview", candidate: candidateId });

  return (
    <div class="tuner-run-overview" data-testid="tuner-run-overview">
      <div class="tuner-run-overview-header">
        <button class="tuner-back" onClick={() => props.navigate({ view: "fleet" })}>
          ← Fleet
        </button>
        <h2>{props.runId}</h2>
        <RunStatusBadge
          status={status()}
          terminalOutcome={journalRow()?.terminal_outcome}
          reportStatus={projectionRow()?.report_status}
        />
        <Show when={live()}>
          <button onClick={() => dispatch({ tag: "stopRun", runId: props.runId })}>Stop</button>
        </Show>
        <button
          onClick={() => dispatch({ tag: "refreshProjection" })}
          disabled={state().refreshing}
        >
          {state().refreshing ? "Refreshing…" : "Refresh science"}
        </button>
      </div>

      <Show when={journalRow()?.run_dir}>
        <p class="tuner-run-dir">{journalRow()!.run_dir}</p>
      </Show>
      <Show when={state().stopError}>
        <div class="launch-error" role="alert">
          {state().stopError}
        </div>
      </Show>
      <Show when={state().refreshError}>
        <div class="launch-error" role="alert">
          {state().refreshError}
        </div>
      </Show>

      <ProgressRail
        status={status()}
        startedAt={journalRow()?.started_at ?? null}
        nowMs={Date.now()}
        compute={detail()?.compute}
        live={liveProgress()}
      />

      <Show when={convergence().present && convergence().steps.length > 1}>
        <section class="tuner-overview-sparkline" data-testid="overview-convergence">
          <h3>Convergence</h3>
          <StepLine
            points={convergence().steps.map((s) => ({ x: s.x, y: s.bestMargin, label: s.label }))}
            domain={convergence().domain}
          />
        </section>
      </Show>

      <Show when={live() || ticker().length > 0}>
        <section class="tuner-live-feed">
          <h3>Live events</h3>
          <EventTicker
            lines={ticker()}
            emptyLabel={
              live()
                ? "Waiting for the tuner's first event…"
                : "No events streamed for this run."
            }
          />
        </section>
      </Show>

      <Show when={progress().phase !== "starting" && !live()}>
        <p class="tuner-run-compute-summary">
          {progress().pairs.completed} of {progress().pairs.attempted} pair attempts completed
          {progress().pairs.failed > 0 ? `, ${progress().pairs.failed} failed` : ""}
        </p>
      </Show>

      <Show when={verdict().ranked.length > 0 || verdict().finalist}>
        <ShipVerdict
          verdict={verdict()}
          gameKind={gameKind()}
          baseConfig={baseConfig()}
          onOpenCandidate={openCandidate}
        />

        <section class="tuner-validation-table">
          <h3>Validation ranking</h3>
          <Forest
            domain={verdict().domain}
            reference={0}
            rows={verdict().ranked.map((r) => ({
              key: r.candidateId,
              label: `#${r.rank} ${r.shortId}`,
              mean: r.estimate,
              lower: r.lower,
              upper: r.upper,
              highlight: r.rank === 1,
              onClick: () => openCandidate(r.candidateId),
            }))}
          />
          <DataTable<VerdictCandidate>
            testid="validation-wdl"
            rows={verdict().ranked}
            rowKey={(r) => r.candidateId}
            onRowClick={(r) => openCandidate(r.candidateId)}
            columns={[
              { key: "rank", header: "#", render: (r) => r.rank },
              { key: "id", header: "Candidate", render: (r) => r.shortId },
              {
                key: "est",
                header: "Estimate",
                align: "right",
                render: (r) => r.estimate.toFixed(3),
              },
              {
                key: "wdl",
                header: "W / D / L",
                align: "right",
                render: (r) => `${r.wins} / ${r.draws} / ${r.losses}`,
              },
              {
                key: "tie",
                header: "",
                render: (r) =>
                  verdict().ties.some((t) => t.left === r.candidateId || t.right === r.candidateId)
                    ? "tie"
                    : "",
              },
            ]}
          />
        </section>
      </Show>

      <div class="tuner-run-links">
        <button onClick={() => props.navigate({ view: "run", runId: props.runId, tab: "science" })}>
          Full science →
        </button>
        <button
          onClick={() => props.navigate({ view: "run", runId: props.runId, tab: "evidence" })}
        >
          Raw evidence →
        </button>
      </div>

      <Show when={live() || state().log.lines.length > 0}>
        <details class="tuner-process-diag">
          <summary>Process diagnostics</summary>
          <section class="tuner-log-tail">
          <Show when={state().log.error}>
            <div class="launch-error" role="alert">
              {state().log.error}
            </div>
          </Show>
          <pre class="tuner-log-lines" data-testid="tuner-log-lines">
            <For each={state().log.lines}>
              {(line) => (
                <>
                  {line}
                  {"\n"}
                </>
              )}
            </For>
          </pre>
          <Show when={state().log.errLines.length > 0}>
            <pre class="tuner-log-err" data-testid="tuner-log-err">
              <For each={state().log.errLines}>
                {(line) => (
                  <>
                    {line}
                    {"\n"}
                  </>
                )}
              </For>
            </pre>
          </Show>
          </section>
        </details>
      </Show>
    </div>
  );
};
