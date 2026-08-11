// Smac3RunDetail.tsx — Trial history for an open `kind: "smac3"` run:
// stats, a cost-over-trials chart (per-trial cost + running best-so-far),
// and a best-trial-vs-default parameter table.
//
// Pure presentational component: `trials`/`tuner` are read from BenchState
// by RunDetailPanel.tsx (which owns the tail loop that keeps `trials`
// current — see reducer.ts's `tailTick`) and passed down, same convention
// as WinRateChart/LeaderboardTable reading their slice of BenchState
// directly, except this one takes props since it's nested inside another
// store-reading component rather than mounted as its own tab.

import { createMemo, For, Show, type Component } from "solid-js";
import type { TrialRow, TunerInfo } from "./index.js";

const CHART_W = 480;
const CHART_H = 160;
const PAD_LEFT = 40;
const PAD_RIGHT = 12;
const PAD_TOP = 12;
const PAD_BOTTOM = 20;
const PLOT_W = CHART_W - PAD_LEFT - PAD_RIGHT;
const PLOT_H = CHART_H - PAD_TOP - PAD_BOTTOM;

function fmtCost(cost: number | null): string {
  return cost === null ? "—" : (cost * 100).toFixed(1) + "%";
}

interface ChartPoint {
  x: number;
  trial: TrialRow;
  bestSoFar: number;
}

export const Smac3RunDetail: Component<{
  trials: TrialRow[];
  tuner: TunerInfo | null;
}> = (props) => {
  // Trials arrive in trial_id order already (the `trials` route's
  // `ORDER BY trial_id ASC`), but sort defensively -- nothing here assumes
  // the caller preserved that.
  const sorted = createMemo(() => [...props.trials].sort((a, b) => a.trial_id - b.trial_id));
  const scored = createMemo(() => sorted().filter((t) => t.cost !== null));

  const bestTrial = createMemo((): TrialRow | null => {
    let best: TrialRow | null = null;
    for (const t of scored()) {
      if (best === null || (t.cost as number) < (best.cost as number)) best = t;
    }
    return best;
  });

  const chartPoints = createMemo((): ChartPoint[] => {
    const ts = scored();
    if (ts.length === 0) return [];
    let best = Infinity;
    return ts.map((trial, i) => {
      best = Math.min(best, trial.cost as number);
      const x = PAD_LEFT + (ts.length > 1 ? (i / (ts.length - 1)) * PLOT_W : PLOT_W / 2);
      return { x, trial, bestSoFar: best };
    });
  });

  const yMax = createMemo(() => {
    const pts = chartPoints();
    if (pts.length === 0) return 1;
    const max = Math.max(...pts.map((p) => p.trial.cost as number));
    return Math.max(0.05, max * 1.1);
  });

  const yScale = createMemo(() => {
    const max = yMax();
    return (v: number) => PAD_TOP + PLOT_H - (v / max) * PLOT_H;
  });

  const bestPathD = createMemo(() => {
    const pts = chartPoints();
    if (pts.length === 0) return "";
    const scale = yScale();
    return pts.map((p, i) => `${i === 0 ? "M" : "L"}${p.x.toFixed(1)},${scale(p.bestSoFar).toFixed(1)}`).join(" ");
  });

  const yTicks = createMemo(() => {
    const max = yMax();
    const step = max / 4;
    return [0, step, step * 2, step * 3, max];
  });

  // Active parameters of the best trial, compared against each parameter's
  // declared default (or `value`, for a `constant`-type param) from the
  // tuner metadata. Only the best trial's *active* params are shown --
  // inactive ones are simply absent from its `config` (see TrialRow's
  // doc comment in types.ts).
  const bestVsDefault = createMemo(() => {
    const best = bestTrial();
    const tuner = props.tuner;
    if (!best || !tuner) return [];
    return Object.entries(best.config).map(([name, value]) => {
      const spec = tuner.parameters.find((p) => p.name === name);
      const def = spec ? (spec.type === "constant" ? spec.value : spec.default) : undefined;
      const changed = def !== undefined && JSON.stringify(def) !== JSON.stringify(value);
      return { name, value, def, changed };
    });
  });

  return (
    <div id="smac3-run-detail">
      <Show
        when={scored().length > 0}
        fallback={<div class="log-empty">No scored trials yet.</div>}
      >
        <div id="smac3-stats-row">
          <div class="smac3-stat">
            <span class="smac3-stat-value">{sorted().length}</span>
            <span class="smac3-stat-label">Trials</span>
          </div>
          <div class="smac3-stat">
            <span class="smac3-stat-value">{fmtCost(bestTrial()?.cost ?? null)}</span>
            <span class="smac3-stat-label">Best cost (loss rate)</span>
          </div>
          <div class="smac3-stat">
            <span class="smac3-stat-value">#{bestTrial()?.trial_id ?? "—"}</span>
            <span class="smac3-stat-label">Best trial</span>
          </div>
        </div>

        <div id="smac3-chart-wrapper">
          <svg width={CHART_W} height={CHART_H} viewBox={`0 0 ${CHART_W} ${CHART_H}`} id="smac3-cost-chart">
            <For each={yTicks()}>
              {(tick) => {
                const y = yScale()(tick);
                return (
                  <>
                    <line x1={PAD_LEFT} y1={y} x2={CHART_W - PAD_RIGHT} y2={y} stroke="rgba(255,255,255,0.06)" stroke-width="1" />
                    <text x={PAD_LEFT - 6} y={y + 3} text-anchor="end" fill="#8a8b96" font-size="9">
                      {(tick * 100).toFixed(0)}%
                    </text>
                  </>
                );
              }}
            </For>

            {/* Running best-so-far step line */}
            <path d={bestPathD()} fill="none" stroke="#4caf7a" stroke-width="2" stroke-linejoin="round" />

            {/* Per-trial cost dots */}
            <For each={chartPoints()}>
              {(p) => (
                <circle
                  cx={p.x}
                  cy={yScale()(p.trial.cost as number)}
                  r={p.trial.trial_id === bestTrial()?.trial_id ? 4 : 2.5}
                  fill={p.trial.trial_id === bestTrial()?.trial_id ? "#4caf7a" : "#5b7fd6"}
                >
                  <title>
                    Trial #{p.trial.trial_id}: cost {fmtCost(p.trial.cost)}
                  </title>
                </circle>
              )}
            </For>
          </svg>
          <div class="smac3-chart-legend">
            <span><i class="legend-swatch legend-swatch-trial" /> trial cost</span>
            <span><i class="legend-swatch legend-swatch-best" /> best so far</span>
          </div>
        </div>

        <Show when={bestVsDefault().length > 0}>
          <table id="smac3-diff-table">
            <caption>Best trial (#{bestTrial()!.trial_id}) vs. search-space default</caption>
            <thead>
              <tr>
                <th>Parameter</th>
                <th>Best</th>
                <th>Default</th>
              </tr>
            </thead>
            <tbody>
              <For each={bestVsDefault()}>
                {(row) => (
                  <tr classList={{ "smac3-diff-changed": row.changed }}>
                    <td class="smac3-param-name">{row.name}</td>
                    <td>{String(row.value)}</td>
                    <td>{row.def === undefined ? "—" : String(row.def)}</td>
                  </tr>
                )}
              </For>
            </tbody>
          </table>
        </Show>
      </Show>

      <div id="smac3-trials-scroll">
        <table id="smac3-trials-table">
          <thead>
            <tr>
              <th>#</th>
              <th>Family</th>
              <th>Cost</th>
              <th>Seed</th>
              <th>Time</th>
            </tr>
          </thead>
          <tbody>
            <For each={sorted().slice().reverse()}>
              {(t) => (
                <tr classList={{ "smac3-trial-best": t.trial_id === bestTrial()?.trial_id }} title={JSON.stringify(t.config)}>
                  <td>{t.trial_id}</td>
                  <td class="smac3-trial-family">{typeof t.config.family === "string" ? t.config.family : "—"}</td>
                  <td>{fmtCost(t.cost)}</td>
                  <td>{t.seed ?? "—"}</td>
                  <td>{t.ts}</td>
                </tr>
              )}
            </For>
          </tbody>
        </table>
      </div>
    </div>
  );
};
