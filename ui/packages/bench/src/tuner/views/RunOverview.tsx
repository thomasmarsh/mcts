// RunOverview — a run's landing view and the ship decision. Header + live
// progress rail + launch-log tail (the sub-second feedback the old UI
// lacked), then, once the projection has the run, the ship verdict and the
// ranked validation table. Cohort race strips and the full science /
// evidence views are their own slices.

import { createEffect, createMemo, createSignal, For, Show, type Component } from "solid-js";
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

  // A run is relaunchable (plain resume, or a budget-raising extend) once
  // its process has exited cleanly — a tuning-budget-exhaustion freeze, a
  // normal completion, or the server's own reaper concluding its pid died
  // with no exit ever observed (`lost`: this server, or the whole machine,
  // restarted mid-run — see `mcts_bench::tuner_launch::reap_lost`). A run
  // that never wrote a manifest (`status: "failed"`) or was deliberately
  // killed (`signalled` / `spawn_failed`) is not: there is nothing coherent
  // to continue.
  const relaunchable = createMemo(() => {
    if (status() !== "exited") return false;
    const outcome = journalRow()?.terminal_outcome;
    return outcome == null || outcome === "exited" || outcome === "lost";
  });

  const [extendTuning, setExtendTuning] = createSignal("");
  const [extendValidation, setExtendValidation] = createSignal("");
  const [extendDiagnostic, setExtendDiagnostic] = createSignal("");
  const [extendReason, setExtendReason] = createSignal("");

  // A non-negative integer, or null if the field is malformed. Empty is 0.
  const parseDelta = (raw: string): number | null => {
    const t = raw.trim();
    if (t === "") return 0;
    if (!/^\d+$/.test(t)) return null;
    return Number(t);
  };
  const extendDeltas = createMemo(() => ({
    tuning: parseDelta(extendTuning()),
    validation: parseDelta(extendValidation()),
    diagnostic: parseDelta(extendDiagnostic()),
  }));
  const extendValid = createMemo(() => {
    const d = extendDeltas();
    if (d.tuning === null || d.validation === null || d.diagnostic === null) return false;
    if (d.tuning + d.validation + d.diagnostic <= 0) return false;
    return extendReason().trim().length > 0;
  });

  // Reset the fields once an extend has succeeded — keyed off the reducer's
  // `extendSeq` rather than an `extendBusy` edge, which the async valtio→Solid
  // snapshot bridge can coalesce away on a synchronous success. A rejection
  // leaves the values in place so the operator can adjust and retry.
  let seenExtendSeq = state().extendSeq;
  createEffect(() => {
    const seq = state().extendSeq;
    if (seq !== seenExtendSeq) {
      seenExtendSeq = seq;
      setExtendTuning("");
      setExtendValidation("");
      setExtendDiagnostic("");
      setExtendReason("");
    }
  });

  const submitExtend = (): void => {
    const d = extendDeltas();
    if (!extendValid() || d.tuning === null || d.validation === null || d.diagnostic === null) {
      return;
    }
    dispatch({
      tag: "extendRun",
      runId: props.runId,
      extension: {
        tuning_pair_attempts_delta: d.tuning,
        validation_pair_attempts_delta: d.validation,
        diagnostic_pair_attempts_delta: d.diagnostic,
        reason: extendReason().trim(),
      },
    });
  };

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

  // True only until the run's own detail has loaded for the first time (a
  // page reload or a fresh navigation, not a background refresh — `peek`
  // means a slice that has already loaded once, even if a refetch is now in
  // flight, no longer counts as "initial"). Before that, `deriveProgress`
  // has no compute ledger to work from and reports a literal "0 / 0 pairs",
  // which reads as real (if unremarkable) data rather than "still loading" —
  // show a skeleton instead so a run that's actually running does not look
  // like one that stalled at zero.
  const initialLoad = createMemo(() => detail() === undefined);

  const gameKind = createMemo(
    () => detail()?.manifest?.game_kind ?? projectionRow()?.game_kind ?? null,
  );
  const baseConfig = createMemo(() => {
    const info = (peek(state().tunableGames) ?? []).find((k) => k.game === gameKind());
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
        <Show when={!live() && relaunchable()}>
          <button
            data-testid="resume-run"
            disabled={state().resumeBusy}
            onClick={() => dispatch({ tag: "resumeRun", runId: props.runId })}
          >
            {state().resumeBusy ? "Resuming…" : "Resume"}
          </button>
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
      <Show when={state().resumeError}>
        <div class="launch-error" role="alert" data-testid="resume-error">
          {state().resumeError}
        </div>
      </Show>

      <Show when={!live() && relaunchable()}>
        <section class="tuner-extend-budget" data-testid="extend-budget-form">
          <h3>Extend budget</h3>
          <p class="tuner-extend-hint">
            Raise this run's pair-attempt budgets and resume it. At least one
            delta must be positive; the run re-opens as <code>live</code>.
          </p>
          <div class="tuner-extend-fields">
            <label>
              Tuning pairs
              <input
                type="number"
                min="0"
                step="1"
                data-testid="extend-tuning-delta"
                value={extendTuning()}
                onInput={(e) => setExtendTuning(e.currentTarget.value)}
              />
            </label>
            <label>
              Validation pairs
              <input
                type="number"
                min="0"
                step="1"
                data-testid="extend-validation-delta"
                value={extendValidation()}
                onInput={(e) => setExtendValidation(e.currentTarget.value)}
              />
            </label>
            <label>
              Diagnostic pairs
              <input
                type="number"
                min="0"
                step="1"
                data-testid="extend-diagnostic-delta"
                value={extendDiagnostic()}
                onInput={(e) => setExtendDiagnostic(e.currentTarget.value)}
              />
            </label>
          </div>
          <label class="tuner-extend-reason">
            Reason
            <input
              type="text"
              data-testid="extend-reason"
              value={extendReason()}
              onInput={(e) => setExtendReason(e.currentTarget.value)}
            />
          </label>
          <button
            data-testid="extend-submit"
            disabled={!extendValid() || state().extendBusy}
            onClick={submitExtend}
          >
            {state().extendBusy ? "Extending…" : "Extend budget"}
          </button>
          <Show when={state().extendError}>
            <div class="launch-error" role="alert" data-testid="extend-error">
              {state().extendError}
            </div>
          </Show>
        </section>
      </Show>

      <Show
        when={!initialLoad()}
        fallback={
          <div class="tuner-run-skeleton" data-testid="run-overview-skeleton">
            Loading run…
            <div class="tuner-run-skeleton-bars">
              <div class="tuner-run-skeleton-bar" />
              <div class="tuner-run-skeleton-bar tuner-run-skeleton-bar--short" />
            </div>
          </div>
        }
      >
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
