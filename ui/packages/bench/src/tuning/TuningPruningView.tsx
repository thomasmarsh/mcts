import { createMemo, createUniqueId, For, Show, type Component } from "solid-js";
import type { Store } from "@mcts/core";
import type { BenchAction } from "../reducer.js";
import type { BenchState } from "../state.js";
import type { TuningNavigationAction } from "../tuning-navigation.js";
import type { TuningTrialDetailView } from "../types.js";
import {
  analysisSampleMetadata,
  exactPlotRows,
  pruningFunnelRows,
  rungFunnelRows,
  trialTrajectoriesFromRows,
  type AnalysisPlotRow,
  type TrialTrajectory,
} from "./analysis-models.js";

const WIDTH = 420;
const HEIGHT = 200;
const LEFT = 42;
const RIGHT = 16;
const TOP = 18;
const BOTTOM = 34;

type Metric = "score" | "mu" | "sigma";

function send(store: Store<BenchState, BenchAction>, action: TuningNavigationAction): void {
  store.dispatch({ tag: "tuningNavigation", action });
}

function value(row: AnalysisPlotRow, metric: Metric): number {
  return metric === "score" ? row.score : metric === "mu" ? row.rating.mu : row.rating.sigma;
}

function metricLabel(metric: Metric): string {
  return metric === "score" ? "Conservative score" : metric === "mu" ? "Rating μ" : "Rating σ";
}

function numberText(value: number): string {
  return Number.isInteger(value) ? String(value) : String(value);
}

function domain(values: readonly number[]): [number, number] {
  if (values.length === 0) return [0, 1];
  const low = Math.min(...values);
  const high = Math.max(...values);
  if (low === high) return [low - 1, high + 1];
  const padding = (high - low) * 0.06;
  return [low - padding, high + padding];
}

function rowsFromDetail(trial: TuningTrialDetailView): AnalysisPlotRow[] {
  return trial.reports.map((report) => ({
    trial_id: trial.trial_id,
    trial_number: trial.trial_number,
    trial_status: trial.state,
    resource: report.completed_pairs,
    rating: report.rating,
    score: report.score,
    outcome: report.decision.outcome,
    reason: report.decision.reason,
    pruning_exempt: report.decision.pruning_exempt,
    bracket_id: report.decision.bracket_id,
    rung_resource: report.decision.rung_resource,
    best: false,
    selected: true,
  }));
}

function terminal(path: TrialTrajectory): AnalysisPlotRow & { terminal: boolean } {
  return path.points.at(-1)!;
}

const TrajectoryPlot: Component<{
  metric: Metric;
  paths: TrialTrajectory[];
  selectedTrialId: string | null;
  onSelect: (trialId: string) => void;
}> = (props) => {
  const id = createUniqueId();
  const resources = createMemo(() => [...new Set(props.paths.flatMap((path) => path.points.map((point) => point.resource)))].sort((a, b) => a - b));
  const rungs = createMemo(() => [...new Set(props.paths.flatMap((path) => path.points.map((point) => point.rung_resource).filter((rung): rung is number => rung !== null)))].sort((a, b) => a - b));
  const yDomain = createMemo(() => domain(props.paths.flatMap((path) => path.points.map((point) => value(point, props.metric)))));
  const x = (resource: number) => {
    const values = resources();
    if (values.length <= 1) return LEFT + (WIDTH - LEFT - RIGHT) / 2;
    return LEFT + ((resource - values[0]!) / (values.at(-1)! - values[0]!)) * (WIDTH - LEFT - RIGHT);
  };
  const y = (metricValue: number) => TOP + (1 - (metricValue - yDomain()[0]) / (yDomain()[1] - yDomain()[0])) * (HEIGHT - TOP - BOTTOM);
  const pathData = (path: TrialTrajectory) => path.points.map((point, index) => `${index === 0 ? "M" : "L"}${x(point.resource)},${y(value(point, props.metric))}`).join(" ");
  const pathLabel = (path: TrialTrajectory) => {
    const end = terminal(path);
    return `Select trial ${path.trialNumber} ${metricLabel(props.metric)} trajectory. It ends after ${end.resource} completed pairs: ${end.outcome}, ${end.reason}; observed rung ${end.rung_resource ?? "Not recorded"}.`;
  };
  const keyboard = (event: KeyboardEvent, trialId: string) => {
    if (event.key === "Enter" || event.key === " ") { event.preventDefault(); props.onSelect(trialId); }
  };
  return (
    <section class="tuning-pruning-trajectory" aria-labelledby={`${id}-heading`}>
      <h5 id={`${id}-heading`}>{metricLabel(props.metric)}</h5>
      <svg class="tuning-pruning-plot" viewBox={`0 0 ${WIDTH} ${HEIGHT}`} role="img" aria-labelledby={`${id}-title ${id}-description`}>
        <title id={`${id}-title`}>{metricLabel(props.metric)} candidate trajectories</title>
        <desc id={`${id}-description`}>Each path is one candidate and ends at its last recorded report. Vertical guides are observed rungs, not inferred cutoffs.</desc>
        <line class="tuning-progress-axis" x1={LEFT} y1={HEIGHT - BOTTOM} x2={WIDTH - RIGHT} y2={HEIGHT - BOTTOM} />
        <line class="tuning-progress-axis" x1={LEFT} y1={TOP} x2={LEFT} y2={HEIGHT - BOTTOM} />
        <For each={rungs()}>{(rung) => <><line class="tuning-pruning-rung" x1={x(rung)} y1={TOP} x2={x(rung)} y2={HEIGHT - BOTTOM} /><text class="tuning-progress-x-label" x={x(rung)} y={TOP + 10} text-anchor="middle">rung {rung}</text></>}</For>
        <For each={resources()}>{(resource) => <text class="tuning-progress-x-label" x={x(resource)} y={HEIGHT - 16} text-anchor="middle">{resource}</text>}</For>
        <text class="tuning-progress-y-label" x={LEFT - 5} y={TOP + 4} text-anchor="end">{numberText(yDomain()[1])}</text>
        <text class="tuning-progress-y-label" x={LEFT - 5} y={HEIGHT - BOTTOM} text-anchor="end">{numberText(yDomain()[0])}</text>
        <text class="tuning-progress-axis-label" x={(LEFT + WIDTH - RIGHT) / 2} y={HEIGHT - 3} text-anchor="middle">completed pairs</text>
        <For each={props.paths}>{(path) => <g classList={{ "tuning-pruning-path": true, "tuning-pruning-selected-path": path.trialId === props.selectedTrialId }} data-testid="pruning-trajectory" data-trial-id={path.trialId} role="button" tabindex="0" aria-label={pathLabel(path)} onClick={() => props.onSelect(path.trialId)} onKeyDown={(event) => keyboard(event, path.trialId)}>
          <title>{pathLabel(path)}</title>
          <path d={pathData(path)} />
          <circle class="tuning-pruning-terminal" cx={x(terminal(path).resource)} cy={y(value(terminal(path), props.metric))} r="4" />
        </g>}</For>
      </svg>
    </section>
  );
};

export const TuningPruningView: Component<{ store: Store<BenchState, BenchAction> }> = (props) => {
  const state = props.store.getState();
  const navigation = () => state().tuningNavigation;
  const overview = () => navigation().overview.snapshot;
  const selectedDetail = () => {
    const trialId = navigation().selection.trialId;
    return trialId ? navigation().trialDetails[trialId]?.snapshot?.trial ?? null : null;
  };
  const rows = createMemo(() => {
    const snapshot = overview();
    if (!snapshot) return [] as AnalysisPlotRow[];
    const selectedId = navigation().selection.trialId;
    const sampled = exactPlotRows(snapshot, selectedId);
    const detail = selectedDetail();
    if (!detail || detail.trial_id !== selectedId) return sampled;
    return [...sampled.filter((row) => row.trial_id !== selectedId), ...rowsFromDetail(detail)]
      .sort((a, b) => a.trial_number - b.trial_number || a.resource - b.resource || a.trial_id.localeCompare(b.trial_id));
  });
  const paths = createMemo(() => trialTrajectoriesFromRows(rows()).filter((path) => {
    const end = terminal(path);
    const { filters, selection } = navigation();
    if (path.trialId === selection.trialId) return true;
    if (filters.reason !== null && end.reason !== filters.reason) return false;
    if (filters.state !== null && path.trialStatus !== filters.state) return false;
    if (filters.bracket !== null && (filters.bracket === "unassigned" ? end.bracket_id !== null : end.bracket_id !== filters.bracket)) return false;
    return true;
  }));
  const selectTrial = (trialId: string) => send(props.store, { tag: "selectTrial", trialId });
  const openExactTrials = (reason: string | null, bracket: string | null = navigation().filters.bracket) => {
    send(props.store, { tag: "setTrialFilters", filters: { reason, bracket, state: null } });
    send(props.store, { tag: "setAnalysisTab", tab: "trials" });
  };
  return (
    <section class="tuning-pruning" aria-labelledby="tuning-pruning-heading">
      <header class="tuning-trials-heading"><div><h4 id="tuning-pruning-heading">Pruning</h4><p>Exact lifecycle decisions and bounded candidate trajectories.</p></div></header>
      <Show when={navigation().overview.status === "error" && !overview()}><div class="tuning-load-error" role="alert">Could not load pruning evidence: {navigation().overview.error}</div></Show>
      <Show when={overview()} fallback={<p class="tuning-empty">Loading pruning evidence…</p>}>{(snapshot) => <>
        <Show when={navigation().overview.status === "loading"}><p class="tuning-page-refresh" role="status">Refreshing pruning evidence…</p></Show>
        <Show when={snapshot().coverage.reports > 0} fallback={<section class="tuning-progress-legacy" aria-label="Reduced pruning capability"><p>Pruning evidence was not retained for this session. Exact trial summaries may still be available.</p><button type="button" onClick={() => send(props.store, { tag: "setAnalysisTab", tab: "trials" })}>Open Trials</button></section>}>
          <Show when={snapshot().policy?.pruning.enabled === false}><p class="tuning-not-recorded">Pruning was disabled by the recorded policy; continued reports are shown without claiming a pruning decision.</p></Show>
          <p class="tuning-pruning-cutoff">Pruning cutoff / threshold: <strong>Not recorded</strong>. The retained evidence records outcomes and observed rungs, not a cutoff value.</p>
          <div class="tuning-pruning-grid">
            <section class="tuning-pruning-funnel" aria-labelledby="tuning-pruning-funnel-heading">
              <h5 id="tuning-pruning-funnel-heading">Full-population decision funnel</h5>
              <p>{snapshot().coverage.reports} exact reports. Failure {snapshot().coverage.trials.failed} and cancellation {snapshot().coverage.trials.cancelled} remain separate terminal trial totals.</p>
              <div class="tuning-pruning-segments" role="list" aria-label="Pruning decision segments">
                <div role="listitem"><button type="button" class="tuning-pruning-segment" onClick={() => openExactTrials(null)} aria-label={`Show all ${snapshot().coverage.reports} reported trials`}><strong>Full-population reported</strong><span>{snapshot().coverage.reports}</span></button></div>
                <For each={pruningFunnelRows(snapshot())}>{(row) => <div role="listitem"><button type="button" class="tuning-pruning-segment" onClick={() => openExactTrials(row.reason)} aria-label={`Show ${row.reports} trials with ${row.label.toLowerCase()} reason`}><strong>{row.label}</strong><span>{row.reports}</span></button></div>}</For>
              </div>
              <div class="tuning-pruning-table-wrap"><table aria-label="Exact pruning decision counts"><thead><tr><th>Decision</th><th>Reports</th><th>Meaning</th><th>Exact trials</th></tr></thead><tbody><For each={pruningFunnelRows(snapshot())}>{(row) => <tr><td>{row.label}</td><td>{row.reports}</td><td>{row.description}</td><td><button type="button" onClick={() => openExactTrials(row.reason)}>Filter Trials</button></td></tr>}</For></tbody></table></div>
            </section>
            <section class="tuning-pruning-rungs" aria-labelledby="tuning-pruning-rungs-heading">
              <h5 id="tuning-pruning-rungs-heading">Observed bracket and resource coverage</h5>
              <p>Rung membership is exact. Decision totals are retained session-wide, so this compact read does not assign a decision bucket to a specific rung.</p>
              <div class="tuning-pruning-table-wrap"><table aria-label="Exact observed bracket resources"><thead><tr><th>Bracket</th><th>Completed pairs</th><th>Observed rung</th><th>Reports</th><th>Trials</th><th>Exact trials</th></tr></thead><tbody><For each={rungFunnelRows(snapshot())}>{(row) => <tr><td>{row.bracket}</td><td>{row.resource}</td><td>{row.rung_resource ?? "Not recorded"}</td><td>{row.reports}</td><td>{row.trials}</td><td><button type="button" onClick={() => openExactTrials(null, row.bracket_id ?? "unassigned")}>Filter bracket</button></td></tr>}</For></tbody></table></div>
            </section>
          </div>
          <section class="tuning-pruning-reasons" aria-labelledby="tuning-pruning-reasons-heading"><h5 id="tuning-pruning-reasons-heading">Recorded reason guide</h5><dl><For each={pruningFunnelRows(snapshot())}>{(row) => <><dt>{row.label}</dt><dd>{row.description}</dd></>}</For></dl></section>
          <section class="tuning-pruning-trajectories" aria-labelledby="tuning-pruning-trajectories-heading"><h5 id="tuning-pruning-trajectories-heading">Candidate trajectories</h5><p>{analysisSampleMetadata(snapshot()).sampled ? `Showing ${analysisSampleMetadata(snapshot()).returned} sampled reports of ${analysisSampleMetadata(snapshot()).total} observed; selected trial detail is shown in full when retained.` : `Showing all ${analysisSampleMetadata(snapshot()).total} observed reports.`} Paths end at the last recorded report; terminal outcome, reason, and observed rung appear in the exact table.</p>
            <Show when={paths().length > 0} fallback={<p class="tuning-empty">No sampled candidate trajectories match the current filters.</p>}><div class="tuning-pruning-plots"><For each={["score", "mu", "sigma"] as Metric[]}>{(metric) => <TrajectoryPlot metric={metric} paths={paths()} selectedTrialId={navigation().selection.trialId} onSelect={selectTrial} />}</For></div>
              <div class="tuning-pruning-table-wrap"><table aria-label="Exact candidate trajectory endpoints"><thead><tr><th>Trial</th><th>Reports</th><th>Last pairs</th><th>Score</th><th>μ</th><th>σ</th><th>Outcome / reason</th><th>Observed rung</th><th>Select</th></tr></thead><tbody><For each={paths()}>{(path) => <tr classList={{ "tuning-progress-selected-row": path.trialId === navigation().selection.trialId }}><td>#{path.trialNumber}</td><td>{path.reportCount}</td><td>{terminal(path).resource}</td><td>{numberText(terminal(path).score)}</td><td>{numberText(terminal(path).rating.mu)}</td><td>{numberText(terminal(path).rating.sigma)}</td><td>{terminal(path).outcome} / {terminal(path).reason}</td><td>{terminal(path).rung_resource ?? "Not recorded"}</td><td><button type="button" onClick={() => selectTrial(path.trialId)} aria-label={`Select trial ${path.trialNumber}`}>Select</button></td></tr>}</For></tbody></table></div>
            </Show>
          </section>
        </Show>
      </>}</Show>
    </section>
  );
};
