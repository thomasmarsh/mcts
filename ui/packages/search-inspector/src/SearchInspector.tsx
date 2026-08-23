import { createMemo, createSignal, createUniqueId, For, Show, type JSX } from "solid-js";
import { moveEquals, type SearchReport, type SearchWarning } from "@mcts/game";

export interface SearchInspectorPoint<M> {
  ply: number;
  player: string;
  move: M;
  report: SearchReport<M> | null | undefined;
}

export type SearchMetric = "iterations" | "elapsed" | "nodes" | "meanDepth" | "maxDepth" | "ttHitRatio";

export interface SearchInspectorProps<S, M> {
  report: SearchReport<M> | null | undefined;
  points?: SearchInspectorPoint<M>[];
  before: S;
  formatMove?: (move: M, before: S) => string;
}

interface MetricDefinition<M> {
  key: SearchMetric;
  label: string;
  value: (report: SearchReport<M> | null | undefined) => number | null;
  format: (value: number) => string;
}

const warningDetail: Record<SearchWarning, string> = {
  actions_truncated: "The action list was truncated before every root action could be retained.",
  principal_variation_truncated: "The principal variation was truncated before its natural end.",
  structural_diagnostics_omitted: "Tree and graph diagnostics were not retained for this search.",
  root_parallel_pv_single_tree: "The principal variation represents one root-parallel tree, not the aggregate search.",
};

const reportReason: Record<NonNullable<SearchReport<unknown>["reason"]>, string> = {
  strategy_unsupported: "This strategy does not expose final-search evidence.",
  search_not_run: "No search was run for this result.",
  root_parallel_pv_single_tree: "Only a single root-parallel tree could provide a principal variation.",
};

function number(value: number): string {
  return new Intl.NumberFormat(undefined, { maximumFractionDigits: 3 }).format(value);
}

function seconds(value: number | null): string {
  return value === null ? "Unavailable" : `${number(value)} s`;
}

function nullableNumber(value: number | null): string {
  return value === null ? "Unavailable" : number(value);
}

function percent(value: number): string {
  return `${(value * 100).toFixed(1)}%`;
}

function reasonDetail(reason: SearchReport<unknown>["reason"], fallback: string): string {
  return reason === null ? fallback : reportReason[reason];
}

function throughput(value: number | null): string {
  return value === null ? "Unavailable" : `${number(value)} iterations/s`;
}

function hitRatio(value: number | null): string {
  return value === null ? "Unavailable" : percent(value);
}

function rawMoveLabel<M>(move: M): string {
  try {
    return JSON.stringify(move) ?? String(move);
  } catch {
    return String(move);
  }
}

function reportMetric<M>(report: SearchReport<M> | null | undefined, key: SearchMetric): number | null {
  if (!report || report.status === "unavailable") return null;
  if (key === "iterations") return report.completed_iterations;
  if (key === "elapsed") return report.elapsed_seconds;
  if (key === "nodes") return report.warnings.includes("structural_diagnostics_omitted") ? null : report.tree_nodes;
  if (key === "meanDepth") return report.mean_depth;
  if (key === "maxDepth") return report.max_depth;
  return report.tt_hit_ratio;
}

const metrics: MetricDefinition<unknown>[] = [
  { key: "iterations", label: "Iterations", value: (r) => reportMetric(r, "iterations"), format: number },
  { key: "elapsed", label: "Elapsed", value: (r) => reportMetric(r, "elapsed"), format: (v) => `${number(v)} s` },
  { key: "nodes", label: "Nodes", value: (r) => reportMetric(r, "nodes"), format: number },
  { key: "meanDepth", label: "Mean depth", value: (r) => reportMetric(r, "meanDepth"), format: number },
  { key: "maxDepth", label: "Max depth", value: (r) => reportMetric(r, "maxDepth"), format: number },
  { key: "ttHitRatio", label: "TT hit ratio", value: (r) => reportMetric(r, "ttHitRatio"), format: percent },
];

function TrendChart<M>(props: { points: SearchInspectorPoint<M>[]; metric: MetricDefinition<M> }): JSX.Element {
  const width = 440;
  const height = 150;
  const left = 36;
  const right = 12;
  const top = 16;
  const bottom = 24;
  const plotWidth = width - left - right;
  const plotHeight = height - top - bottom;
  const values = props.points.map((point) => props.metric.value(point.report));
  const present = values.filter((value): value is number => value !== null);
  const hasValues = present.length > 0;
  const lower = hasValues ? Math.min(...present) : 0;
  const upper = hasValues ? Math.max(...present) : 0;
  const span = upper - lower || 1;
  const x = (index: number) => left + (props.points.length === 1 ? plotWidth / 2 : (index / (props.points.length - 1)) * plotWidth);
  const y = (value: number) => top + ((upper - value) / span) * plotHeight;
  const path = values.reduce((d, value, index) => value === null ? "" : `${d}${index === 0 || values[index - 1] === null ? "M" : "L"}${x(index)},${y(value)} `, "");

  return (
    <svg
      width={width}
      height={height}
      viewBox={`0 0 ${width} ${height}`}
      role="img"
      aria-label={`Search metric trend: ${props.metric.label}`}
    >
      <line x1={left} y1={top + plotHeight} x2={width - right} y2={top + plotHeight} stroke="currentColor" stroke-opacity="0.35" />
      <line x1={left} y1={top} x2={left} y2={top + plotHeight} stroke="currentColor" stroke-opacity="0.35" />
      <path d={path} fill="none" stroke="currentColor" stroke-width="2" />
      <For each={values}>
        {(value, index) => <Show when={value !== null}><circle cx={x(index())} cy={y(value!)} r="3" fill="currentColor" /></Show>}
      </For>
      <Show when={hasValues} fallback={<text x={left + 6} y={top + plotHeight / 2} font-size="11">No values available</text>}>
        <><text x={left - 4} y={top + 4} text-anchor="end" font-size="10">{props.metric.format(upper)}</text><text x={left - 4} y={top + plotHeight} text-anchor="end" font-size="10">{props.metric.format(lower)}</text></>
      </Show>
    </svg>
  );
}

export const SearchInspector = <S, M>(props: SearchInspectorProps<S, M>) => {
  const id = createUniqueId();
  const warningHeadingId = `${id}-warnings`;
  const summaryHeadingId = `${id}-summary`;
  const actionsHeadingId = `${id}-actions`;
  const pvHeadingId = `${id}-pv`;
  const trendHeadingId = `${id}-trend`;
  const [metricKey, setMetricKey] = createSignal<SearchMetric>("iterations");
  const trendPoints = createMemo(() => props.points ?? []);
  const metric = createMemo(() => metrics.find((entry) => entry.key === metricKey()) as MetricDefinition<M>);
  const trendRows = createMemo(() => {
    const currentMetric = metric();
    return trendPoints().map((point) => ({ point, value: currentMetric.value(point.report), format: currentMetric.format }));
  });
  const actionLabel = (move: M) => props.formatMove?.(move, props.before) ?? rawMoveLabel(move);
  const pvLabel = (move: M, index: number) => index === 0 ? actionLabel(move) : rawMoveLabel(move);

  return (
    <section aria-label="Final search inspector">
      <Show when={props.report} fallback={<p role="status">No final-search report is available from this legacy result.</p>}>
        {(report) => (
          <>
            <Show when={report().status === "unavailable"}>
              <p role="status">Final-search evidence unavailable. {reasonDetail(report().reason, "No further reason was supplied.")}</p>
            </Show>
            <Show when={report().status === "partial"}>
              <p role="status">Final-search evidence is partial. {reasonDetail(report().reason, "Review the warnings below.")}</p>
            </Show>

            <Show when={report().warnings.length > 0}>
              <section aria-labelledby={warningHeadingId}>
                <h3 id={warningHeadingId}>Warnings</h3>
                <ul>
                  <For each={report().warnings}>{(warning) => <li>{warningDetail[warning]}</li>}</For>
                </ul>
              </section>
            </Show>

            <section aria-labelledby={summaryHeadingId}>
              <h3 id={summaryHeadingId}>Search summary</h3>
              <dl>
                <dt>Iteration limit</dt><dd>{nullableNumber(report().iteration_limit)}</dd>
                <dt>Time limit</dt><dd>{seconds(report().time_limit_seconds)}</dd>
                <dt>Completed iterations</dt><dd>{number(report().completed_iterations)}</dd>
                <dt>Termination</dt><dd>{report().termination ?? "Unavailable"}</dd>
                <dt>Elapsed</dt><dd>{seconds(report().elapsed_seconds)}</dd>
                <dt>Throughput</dt><dd>{throughput(report().iterations_per_second)}</dd>
                <dt>Root visits</dt><dd>{number(report().root_visits)}</dd>
                <dt>Tree nodes</dt><dd>{number(report().tree_nodes)}</dd>
                <dt>Mean depth</dt><dd>{nullableNumber(report().mean_depth)}</dd>
                <dt>Max depth</dt><dd>{nullableNumber(report().max_depth)}</dd>
                <dt>Graph</dt><dd>{report().graph_mode ?? "Unavailable"}</dd>
                <dt>TT reads</dt><dd>{number(report().tt_reads)}</dd>
                <dt>TT writes</dt><dd>{number(report().tt_writes)}</dd>
                <dt>TT hits</dt><dd>{number(report().tt_hits)}</dd>
                <dt>TT hit ratio</dt><dd>{hitRatio(report().tt_hit_ratio)}</dd>
              </dl>
            </section>

            <section aria-labelledby={actionsHeadingId}>
              <h3 id={actionsHeadingId}>Root actions</h3>
              <table>
                <thead><tr><th scope="col">Action</th><th scope="col">Visits</th><th scope="col">Share</th><th scope="col">Mean</th><th scope="col">Outcome proven</th><th scope="col">Selected</th></tr></thead>
                <tbody>
                  <For each={report().actions}>
                    {(action) => <tr><td>{actionLabel(action.action)}</td><td>{number(action.visits)}</td><td>{percent(action.share)}</td><td>{number(action.mean_value)}</td><td>{action.is_proven ? "Yes" : "No"}</td><td>{report().selected_action !== null && moveEquals(action.action, report().selected_action) ? "Selected" : ""}</td></tr>}
                  </For>
                </tbody>
              </table>
            </section>

            <Show when={report().principal_variation.length > 0}>
              <section aria-labelledby={pvHeadingId}>
                <h3 id={pvHeadingId}>Principal variation</h3>
                <p>{report().principal_variation.map(pvLabel).join(" → ")}</p>
              </section>
            </Show>
          </>
        )}
      </Show>

      <Show when={trendPoints().length >= 2}>
        <section aria-labelledby={trendHeadingId}>
          <h3 id={trendHeadingId}>Per-ply search trend</h3>
          <label>
            Metric
            <select value={metricKey()} onChange={(event) => setMetricKey(event.currentTarget.value as SearchMetric)}>
              <For each={metrics}>{(entry) => <option value={entry.key}>{entry.label}</option>}</For>
            </select>
          </label>
          <TrendChart points={trendPoints()} metric={metric()} />
          <table aria-label={`Exact per-ply values for ${metric().label}`}>
            <thead><tr><th scope="col">Ply</th><th scope="col">Player</th><th scope="col">Move</th><th scope="col">{metric().label}</th></tr></thead>
            <tbody>
              <For each={trendRows()}>
                {(row) => <tr><td>{row.point.ply}</td><td>{row.point.player}</td><td>{rawMoveLabel(row.point.move)}</td><td>{row.value === null ? "Unavailable" : row.format(row.value)}</td></tr>}
              </For>
            </tbody>
          </table>
        </section>
      </Show>
    </section>
  );
};
