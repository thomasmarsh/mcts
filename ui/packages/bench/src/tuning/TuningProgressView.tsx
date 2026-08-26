import { createMemo, createUniqueId, For, Show, type Component } from "solid-js";
import type { Store } from "@mcts/core";
import type { BenchAction } from "../reducer.js";
import type { BenchState } from "../state.js";
import type {
  TuningNavigationAction,
  TuningProgressMetric,
  TuningProgressScale,
} from "../tuning-navigation.js";
import type { TuningTrialDetailView } from "../types.js";
import {
  UNASSIGNED_BRACKET,
  bracketFacets,
  exactPlotRows,
  stateSymbol,
  type AnalysisPlotRow,
  type BracketFacet,
} from "./analysis-models.js";

const WIDTH = 360;
const HEIGHT = 220;
const LEFT = 42;
const RIGHT = 12;
const TOP = 20;
const BOTTOM = 34;

type ProgressRow = AnalysisPlotRow;

function send(store: Store<BenchState, BenchAction>, action: TuningNavigationAction): void {
  store.dispatch({ tag: "tuningNavigation", action });
}

function numberText(value: number): string {
  return Number.isInteger(value) ? String(value) : String(value);
}

function metricLabel(metric: TuningProgressMetric): string {
  return metric === "score" ? "Conservative score" : metric === "mu" ? "Rating μ" : "Rating σ";
}

function metricValue(row: ProgressRow, metric: TuningProgressMetric): number {
  return metric === "score" ? row.score : metric === "mu" ? row.rating.mu : row.rating.sigma;
}

function pointFromReport(trial: TuningTrialDetailView): ProgressRow[] {
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

function domain(values: number[]): [number, number] {
  if (values.length === 0) return [0, 1];
  const low = Math.min(...values);
  const high = Math.max(...values);
  if (low === high) return [low - 1, high + 1];
  const padding = (high - low) * 0.06;
  return [low - padding, high + padding];
}

function markerKind(
  state: string,
): "complete" | "pruned" | "running" | "queued" | "failed" | "cancelled" | "other" {
  const normalized = state.toLowerCase();
  if (normalized === "complete" || normalized === "completed") return "complete";
  if (normalized === "pruned" || normalized === "prune") return "pruned";
  if (normalized === "running") return "running";
  if (normalized === "queued") return "queued";
  if (normalized === "failed") return "failed";
  if (normalized === "cancelled") return "cancelled";
  return "other";
}

const Marker: Component<{
  row: ProgressRow;
  x: number;
  y: number;
  selected: boolean;
  onSelect: () => void;
}> = (props) => {
  const kind = () => markerKind(props.row.trial_status);
  const label = () =>
    `Select trial ${props.row.trial_number}: ${stateSymbol(props.row.trial_status).label}, ${props.row.resource} completed pairs, score ${numberText(props.row.score)}, μ ${numberText(props.row.rating.mu)}, σ ${numberText(props.row.rating.sigma)}`;
  const keyboard = (event: KeyboardEvent) => {
    if (event.key === "Enter" || event.key === " ") {
      event.preventDefault();
      props.onSelect();
    }
  };
  return (
    <g
      classList={{ "tuning-progress-mark": true, "tuning-progress-selected-mark": props.selected }}
      data-testid="progress-mark"
      data-trial-id={props.row.trial_id}
      role="button"
      tabindex="0"
      aria-label={label()}
      onClick={props.onSelect}
      onKeyDown={keyboard}
    >
      <title>{label()}</title>
      <Show when={kind() === "complete"}>
        <polygon
          points={`${props.x},${props.y - 5} ${props.x + 5},${props.y} ${props.x},${props.y + 5} ${props.x - 5},${props.y}`}
        />
      </Show>
      <Show when={kind() === "pruned"}>
        <polygon
          points={`${props.x},${props.y - 5} ${props.x + 5},${props.y + 4} ${props.x - 5},${props.y + 4}`}
        />
      </Show>
      <Show when={kind() === "running"}>
        <rect x={props.x - 4} y={props.y - 4} width="8" height="8" />
      </Show>
      <Show when={kind() === "queued" || kind() === "other"}>
        <circle cx={props.x} cy={props.y} r="4" />
      </Show>
      <Show when={kind() === "failed"}>
        <>
          <line x1={props.x - 4} y1={props.y - 4} x2={props.x + 4} y2={props.y + 4} />
          <line x1={props.x - 4} y1={props.y + 4} x2={props.x + 4} y2={props.y - 4} />
        </>
      </Show>
      <Show when={kind() === "cancelled"}>
        <line x1={props.x - 5} y1={props.y} x2={props.x + 5} y2={props.y} />
      </Show>
      <Show when={props.row.best && kind() === "complete"}>
        <circle class="tuning-progress-best-ring" cx={props.x} cy={props.y} r="8" />
      </Show>
    </g>
  );
};

const ProgressPlot: Component<{
  facet: BracketFacet;
  rows: ProgressRow[];
  metric: TuningProgressMetric;
  scale: TuningProgressScale;
  sharedDomain: [number, number];
  selectedTrialId: string | null;
  onSelect: (trialId: string) => void;
}> = (props) => {
  const id = createUniqueId();
  const resources = createMemo(() =>
    [...new Set([...props.facet.resources, ...props.rows.map((row) => row.resource)])].sort(
      (a, b) => a - b,
    ),
  );
  const values = createMemo(() => props.rows.map((row) => metricValue(row, props.metric)));
  const yDomain = createMemo(() =>
    props.scale === "shared" ? props.sharedDomain : domain(values()),
  );
  const x = (resource: number) => {
    const values = resources();
    if (values.length <= 1) return LEFT + (WIDTH - LEFT - RIGHT) / 2;
    return (
      LEFT + ((resource - values[0]!) / (values.at(-1)! - values[0]!)) * (WIDTH - LEFT - RIGHT)
    );
  };
  const y = (value: number) =>
    TOP + (1 - (value - yDomain()[0]) / (yDomain()[1] - yDomain()[0])) * (HEIGHT - TOP - BOTTOM);
  const selectedRows = createMemo(() =>
    props.rows
      .filter((row) => row.trial_id === props.selectedTrialId)
      .sort((a, b) => a.resource - b.resource),
  );
  const selectedPath = createMemo(() =>
    selectedRows().length < 2
      ? ""
      : selectedRows()
          .map(
            (row, index) =>
              `${index === 0 ? "M" : "L"}${x(row.resource)},${y(metricValue(row, props.metric))}`,
          )
          .join(" "),
  );
  const description = () =>
    `${props.facet.label} bracket. X axis is completed pairs. Y axis is ${metricLabel(props.metric)} using ${props.scale} scale. ${props.rows.length} report marks.`;
  return (
    <section class="tuning-progress-facet" aria-labelledby={`${id}-heading`}>
      <header>
        <h5 id={`${id}-heading`}>{props.facet.label}</h5>
        <span>
          {props.rows.length} sampled reports · {props.facet.trials} exact trials
        </span>
      </header>
      <svg
        class="tuning-progress-plot"
        viewBox={`0 0 ${WIDTH} ${HEIGHT}`}
        role="img"
        aria-labelledby={`${id}-title ${id}-description`}
      >
        <title id={`${id}-title`}>{props.facet.label} progress plot</title>
        <desc id={`${id}-description`}>{description()}</desc>
        <line
          class="tuning-progress-axis"
          x1={LEFT}
          y1={HEIGHT - BOTTOM}
          x2={WIDTH - RIGHT}
          y2={HEIGHT - BOTTOM}
        />
        <line class="tuning-progress-axis" x1={LEFT} y1={TOP} x2={LEFT} y2={HEIGHT - BOTTOM} />
        <For each={resources()}>
          {(resource) => (
            <>
              <line
                class="tuning-progress-rung"
                x1={x(resource)}
                y1={TOP}
                x2={x(resource)}
                y2={HEIGHT - BOTTOM}
              />
              <text
                class="tuning-progress-x-label"
                x={x(resource)}
                y={HEIGHT - 16}
                text-anchor="middle"
              >
                {resource}
              </text>
            </>
          )}
        </For>
        <text class="tuning-progress-y-label" x={LEFT - 5} y={TOP + 4} text-anchor="end">
          {numberText(yDomain()[1])}
        </text>
        <text class="tuning-progress-y-label" x={LEFT - 5} y={HEIGHT - BOTTOM} text-anchor="end">
          {numberText(yDomain()[0])}
        </text>
        <text
          class="tuning-progress-axis-label"
          x={(LEFT + WIDTH - RIGHT) / 2}
          y={HEIGHT - 3}
          text-anchor="middle"
        >
          completed pairs
        </text>
        <Show when={selectedPath()}>
          <path
            class="tuning-progress-selected-path"
            data-testid="progress-selected-path"
            d={selectedPath()}
          />
        </Show>
        <For each={props.rows}>
          {(row) => (
            <Marker
              row={row}
              x={x(row.resource)}
              y={y(metricValue(row, props.metric))}
              selected={row.trial_id === props.selectedTrialId}
              onSelect={() => props.onSelect(row.trial_id)}
            />
          )}
        </For>
      </svg>
      <div class="tuning-progress-legend" aria-label="Plot legend">
        <span>
          <i class="tuning-progress-complete" /> complete
        </span>
        <span>
          <i class="tuning-progress-running" /> running
        </span>
        <span>
          <i class="tuning-progress-pruned" /> pruned
        </span>
        <span>
          <i class="tuning-progress-queued" /> queued
        </span>
        <span>
          <i class="tuning-progress-failed" /> failed
        </span>
        <span>
          <i class="tuning-progress-cancelled" /> cancelled
        </span>
        <span>
          <i class="tuning-progress-best" /> best complete
        </span>
      </div>
      <div class="tuning-progress-table-wrap">
        <table aria-label={`Exact values for ${props.facet.label} progress`}>
          <thead>
            <tr>
              <th>Trial</th>
              <th>Pairs</th>
              <th>{metricLabel(props.metric)}</th>
              <th>μ</th>
              <th>σ</th>
              <th>State / reason</th>
              <th>Select</th>
            </tr>
          </thead>
          <tbody>
            <For each={props.rows}>
              {(row) => (
                <tr
                  classList={{
                    "tuning-progress-selected-row": row.trial_id === props.selectedTrialId,
                  }}
                >
                  <td>
                    #{row.trial_number}
                    {row.best ? " · best complete" : ""}
                  </td>
                  <td>{row.resource}</td>
                  <td>{numberText(metricValue(row, props.metric))}</td>
                  <td>{numberText(row.rating.mu)}</td>
                  <td>{numberText(row.rating.sigma)}</td>
                  <td>
                    {stateSymbol(row.trial_status).symbol} {row.trial_status} / {row.reason}
                  </td>
                  <td>
                    <button
                      type="button"
                      aria-label={`Select trial ${row.trial_number}`}
                      onClick={() => props.onSelect(row.trial_id)}
                    >
                      Select
                    </button>
                  </td>
                </tr>
              )}
            </For>
          </tbody>
        </table>
      </div>
      <Show when={props.rows.length === 0}>
        <p class="tuning-empty">No sampled reports match this facet and filter.</p>
      </Show>
    </section>
  );
};

export const TuningProgressView: Component<{ store: Store<BenchState, BenchAction> }> = (props) => {
  const state = props.store.getState();
  const navigation = () => state().tuningNavigation;
  const overview = () => navigation().overview.snapshot;
  const selectedDetail = () => {
    const trialId = navigation().selection.trialId;
    return trialId ? (navigation().trialDetails[trialId]?.snapshot?.trial ?? null) : null;
  };
  const rows = createMemo(() => {
    const snapshot = overview();
    if (!snapshot) return [] as ProgressRow[];
    const selectedId = navigation().selection.trialId;
    const fromOverview = exactPlotRows(snapshot, selectedId);
    const detail = selectedDetail();
    if (!detail || detail.trial_id !== selectedId) return fromOverview;
    const bestIds = new Set(snapshot.best?.trial_ids ?? []);
    return [
      ...fromOverview.filter((row) => row.trial_id !== selectedId),
      ...pointFromReport(detail).map((row) => ({
        ...row,
        best: bestIds.has(row.trial_id) && row.trial_status === "complete",
      })),
    ].sort(
      (a, b) =>
        a.trial_number - b.trial_number ||
        a.resource - b.resource ||
        a.trial_id.localeCompare(b.trial_id),
    );
  });
  const visibleRows = createMemo(() =>
    rows().filter((row) => {
      const filters = navigation().filters;
      const bracket = filters.bracket;
      if (
        bracket !== null &&
        (bracket === UNASSIGNED_BRACKET ? row.bracket_id !== null : row.bracket_id !== bracket)
      )
        return false;
      return filters.state === null || row.trial_status === filters.state;
    }),
  );
  const selectedFacetIds = createMemo(() =>
    navigation().filters.bracket === null
      ? []
      : [navigation().filters.bracket === UNASSIGNED_BRACKET ? null : navigation().filters.bracket],
  );
  const facets = createMemo(() => {
    const snapshot = overview();
    return snapshot
      ? bracketFacets(snapshot, selectedFacetIds()).filter(
          (facet) =>
            navigation().filters.bracket === null || facet.key === navigation().filters.bracket,
        )
      : [];
  });
  const sharedDomain = createMemo(() =>
    domain(visibleRows().map((row) => metricValue(row, navigation().progressMetric))),
  );
  const states = createMemo(() => [...new Set(rows().map((row) => row.trial_status))].sort());
  const selectTrial = (trialId: string) => send(props.store, { tag: "selectTrial", trialId });
  const setFilter = (field: "state" | "bracket" | "family", value: string) =>
    send(props.store, { tag: "setTrialFilters", filters: { [field]: value || null } });
  return (
    <section class="tuning-progress" aria-labelledby="tuning-progress-heading">
      <header class="tuning-trials-heading">
        <div>
          <h4 id="tuning-progress-heading">Progress</h4>
          <p>Bracket-separated reports, plotted by completed pairs.</p>
        </div>
      </header>
      <Show when={navigation().overview.status === "error" && !overview()}>
        <div class="tuning-load-error" role="alert">
          Could not load progress: {navigation().overview.error}
        </div>
      </Show>
      <Show when={overview()} fallback={<p class="tuning-empty">Loading progress evidence…</p>}>
        {(snapshot) => (
          <>
            <Show when={navigation().overview.status === "loading"}>
              <p class="tuning-page-refresh" role="status">
                Refreshing progress evidence…
              </p>
            </Show>
            <Show
              when={snapshot().coverage.reports > 0}
              fallback={
                <section class="tuning-progress-legacy" aria-label="Reduced progress capability">
                  <p>
                    Progress evidence was not retained for this legacy session. Use Trials for
                    recorded trial summaries or Game evidence for its retained games.
                  </p>
                  <button
                    type="button"
                    onClick={() => send(props.store, { tag: "setAnalysisTab", tab: "trials" })}
                  >
                    Open Trials
                  </button>
                  <button
                    type="button"
                    onClick={() => send(props.store, { tag: "setAnalysisTab", tab: "game" })}
                  >
                    Open Game evidence
                  </button>
                </section>
              }
            >
              <fieldset class="tuning-progress-controls">
                <legend>Progress controls</legend>
                <label>
                  Bracket{" "}
                  <select
                    aria-label="Progress bracket"
                    value={navigation().filters.bracket ?? ""}
                    onChange={(event) => setFilter("bracket", event.currentTarget.value)}
                  >
                    <option value="">All brackets</option>
                    <For each={bracketFacets(snapshot(), selectedFacetIds())}>
                      {(facet) => <option value={facet.key}>{facet.label}</option>}
                    </For>
                  </select>
                </label>
                <label>
                  Metric{" "}
                  <select
                    aria-label="Progress metric"
                    value={navigation().progressMetric}
                    onChange={(event) =>
                      send(props.store, {
                        tag: "setProgressMetric",
                        metric: event.currentTarget.value as TuningProgressMetric,
                      })
                    }
                  >
                    <option value="score">Conservative score</option>
                    <option value="mu">Rating μ</option>
                    <option value="sigma">Rating σ</option>
                  </select>
                </label>
                <label>
                  Y scale{" "}
                  <select
                    aria-label="Progress Y scale"
                    value={navigation().progressScale}
                    onChange={(event) =>
                      send(props.store, {
                        tag: "setProgressScale",
                        scale: event.currentTarget.value as TuningProgressScale,
                      })
                    }
                  >
                    <option value="shared">Shared across brackets</option>
                    <option value="local">Local to each bracket</option>
                  </select>
                </label>
                <label>
                  State{" "}
                  <select
                    aria-label="Progress state"
                    value={navigation().filters.state ?? ""}
                    onChange={(event) => setFilter("state", event.currentTarget.value)}
                  >
                    <option value="">All states</option>
                    <For each={states()}>{(state) => <option value={state}>{state}</option>}</For>
                  </select>
                </label>
                <label>
                  Family{" "}
                  <input
                    aria-label="Progress family"
                    value={navigation().filters.family ?? ""}
                    onInput={(event) => setFilter("family", event.currentTarget.value.trim())}
                  />
                </label>
              </fieldset>
              <p class="tuning-progress-disclosure">
                {snapshot().coverage.points.sampled
                  ? `Showing ${snapshot().coverage.points.returned} sampled reports of ${snapshot().coverage.points.total} observed.`
                  : `Showing all ${snapshot().coverage.points.total} observed reports.`}{" "}
                Exact full-population counts: {snapshot().coverage.trials.total} trials and{" "}
                {snapshot().coverage.reports} reports. Family names are retained on the paged Trials
                read, not compact progress reports.
              </p>
              <p class="tuning-progress-scale-label">
                Y scale:{" "}
                {navigation().progressScale === "shared"
                  ? "shared across all displayed brackets"
                  : "local to each bracket"}{" "}
                · metric: {metricLabel(navigation().progressMetric)}.
              </p>
              <div class="tuning-progress-facets">
                <For each={facets()}>
                  {(facet) => (
                    <ProgressPlot
                      facet={facet}
                      rows={visibleRows().filter((row) => row.bracket_id === facet.id)}
                      metric={navigation().progressMetric}
                      scale={navigation().progressScale}
                      sharedDomain={sharedDomain()}
                      selectedTrialId={navigation().selection.trialId}
                      onSelect={selectTrial}
                    />
                  )}
                </For>
              </div>
            </Show>
          </>
        )}
      </Show>
    </section>
  );
};
