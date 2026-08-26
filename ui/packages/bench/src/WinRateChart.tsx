// WinRateChart.tsx — SVG line chart showing a selected strategy's win rate
// (with Wilson CI band) across git commits.
//
// Reads the `commitTrends` slice of BenchState (populated by
// `fetchCommitTrends` in the reducer) and lets the user pick a strategy and
// game to filter by. No direct API calls — the fetch-ban eslint rule is
// enforced for the whole package outside api-client.ts.

import { createMemo, createSignal, For, Show, type Component } from "solid-js";
import type { Store } from "@mcts/core";
import type { BenchAction, BenchState, LeaderboardEntry } from "./index.js";

const CHART_W = 500;
const CHART_H = 220;
const PAD_LEFT = 48;
const PAD_RIGHT = 16;
const PAD_TOP = 16;
const PAD_BOTTOM = 36;
const PLOT_W = CHART_W - PAD_LEFT - PAD_RIGHT;
const PLOT_H = CHART_H - PAD_TOP - PAD_BOTTOM;

function shortSha(sha: string): string {
  return sha.length > 7 ? sha.slice(0, 7) : sha;
}

interface PlotPoint {
  x: number;
  y: number;
  entry: LeaderboardEntry;
  sha: string;
  shortSha: string;
}

interface PlotData {
  points: PlotPoint[];
}

export const WinRateChart: Component<{
  store: Store<BenchState, BenchAction>;
}> = (props) => {
  const state = props.store.getState();
  const dispatch = props.store.dispatch;

  const trends = createMemo(() => state().commitTrends);
  const runsState = createMemo(() => state().runs);
  const runs = createMemo(() => (runsState().status === "done" ? (runsState().result ?? []) : []));

  // Derive available strategies and games from the loaded trend data.
  const availableStrategies = createMemo(() => {
    const set = new Set<string>();
    const d = trends().data;
    for (const sha of trends().shas) {
      for (const e of d[sha] ?? []) set.add(e.strategy);
    }
    return Array.from(set).sort();
  });

  const availableGames = createMemo(() => {
    const set = new Set<string>();
    for (const r of runs()) {
      if (r.game) set.add(r.game);
    }
    return Array.from(set).sort();
  });

  // Local selection state.
  const [selectedStrategy, setSelectedStrategy] = createSignal("");
  const [selectedGame, setSelectedGame] = createSignal("");
  const [hoveredPoint, setHoveredPoint] = createSignal<{
    sha: string;
    entry: LeaderboardEntry;
  } | null>(null);

  // Fetch trends when the game filter changes.
  function applyGame(game: string): void {
    setSelectedGame(game);
    setHoveredPoint(null);
    dispatch({ tag: "fetchCommitTrends", game: game || null });
  }

  // Compute plot data: for the selected strategy, collect one point per SHA.
  const plotData = createMemo((): PlotData | null => {
    const strategy = selectedStrategy();
    if (!strategy) return null;
    const d = trends().data;
    const shas = trends().shas;
    if (shas.length === 0) return null;

    const points: PlotPoint[] = [];
    for (let i = 0; i < shas.length; i++) {
      const sha = shas[i]!;
      const entries = d[sha];
      if (!entries) continue;
      const entry = entries.find((e) => e.strategy === strategy);
      if (!entry) continue;

      const x = PAD_LEFT + (shas.length > 1 ? (i / (shas.length - 1)) * PLOT_W : PLOT_W / 2);
      points.push({ x, y: 0, entry, sha, shortSha: shortSha(sha) });
    }
    if (points.length < 2) return null;
    return { points };
  });

  // Find win-rate range across all points for Y-axis scaling.
  const yRange = createMemo(() => {
    const pts = plotData();
    if (!pts) return { min: 0, max: 1 };
    let min = Infinity;
    let max = -Infinity;
    for (const p of pts.points) {
      const lo = p.entry.ci_lower;
      const hi = p.entry.ci_upper;
      if (lo < min) min = lo;
      if (hi > max) max = hi;
    }
    // Add 10% padding.
    const pad = (max - min) * 0.1 || 0.05;
    return { min: Math.max(0, min - pad), max: Math.min(1, max + pad) };
  });

  const yScale = createMemo(() => {
    const { min, max } = yRange();
    const range = max - min || 0.5;
    return (v: number) => PAD_TOP + PLOT_H - ((v - min) / range) * PLOT_H;
  });

  // Build line path.
  const pathD = createMemo(() => {
    const pts = plotData();
    if (!pts) return "";
    const scale = yScale();
    return pts.points
      .map((p, i) => {
        const y = scale(p.entry.win_rate);
        return `${i === 0 ? "M" : "L"}${p.x.toFixed(1)},${y.toFixed(1)}`;
      })
      .join(" ");
  });

  // CI band area path
  const ciPathD = createMemo(() => {
    const pts = plotData();
    if (!pts) return "";
    const scale = yScale();
    // Upper edge (left to right)
    const upper = pts.points
      .map(
        (p, i) => `${i === 0 ? "M" : "L"}${p.x.toFixed(1)},${scale(p.entry.ci_upper).toFixed(1)}`,
      )
      .join(" ");
    // Lower edge (right to left)
    const lower = pts.points
      .slice()
      .reverse()
      .map((p: PlotPoint) => `L${p.x.toFixed(1)},${scale(p.entry.ci_lower).toFixed(1)}`)
      .join(" ");
    return `${upper} ${lower} Z`;
  });

  // Y-axis tick marks.
  const yTicks = createMemo(() => {
    const { min, max } = yRange();
    const ticks: number[] = [];
    const step = Math.max(0.05, Math.round(((max - min) / 4) * 20) / 20);
    let t = Math.ceil(min / step) * step;
    while (t <= max) {
      ticks.push(t);
      t += step;
    }
    return ticks;
  });

  return (
    <div id="chart-panel">
      <div id="chart-header">
        <h3>Win Rate Over Commits</h3>
        <div id="chart-controls">
          <Show when={availableGames().length > 0}>
            <select value={selectedGame()} onChange={(e) => applyGame(e.currentTarget.value)}>
              <option value="">All games</option>
              <For each={availableGames()}>{(g) => <option value={g}>{g}</option>}</For>
            </select>
          </Show>
          <Show when={trends().shas.length > 0 && availableStrategies().length > 0}>
            <select
              value={selectedStrategy()}
              onChange={(e) => setSelectedStrategy(e.currentTarget.value)}
            >
              <option value="">— Select strategy —</option>
              <For each={availableStrategies()}>{(s) => <option value={s}>{s}</option>}</For>
            </select>
          </Show>
        </div>
      </div>

      <Show when={trends().status === "loading"}>
        <div class="loading-bench">Loading commit trends…</div>
      </Show>
      <Show when={trends().status === "error"}>
        <div class="lb-error">{trends().error}</div>
      </Show>

      <Show
        when={
          trends().shas.length === 0 && trends().status !== "loading" && trends().status !== "error"
        }
      >
        <div class="lb-empty">No commit data yet. Select a game above and ensure runs exist.</div>
      </Show>

      <Show
        when={
          selectedStrategy() &&
          !plotData() &&
          trends().status !== "loading" &&
          trends().shas.length > 0
        }
      >
        <div class="lb-empty">
          Need at least 2 data points for strategy &quot;{selectedStrategy()}&quot; to draw a trend
          line.
        </div>
      </Show>

      <Show when={plotData()}>
        <div id="chart-svg-wrapper">
          <svg
            width={CHART_W}
            height={CHART_H}
            viewBox={`0 0 ${CHART_W} ${CHART_H}`}
            id="winrate-chart"
          >
            {/* Grid lines */}
            <For each={yTicks()}>
              {(tick) => {
                const y = yScale()(tick);
                return (
                  <>
                    <line
                      x1={PAD_LEFT}
                      y1={y}
                      x2={CHART_W - PAD_RIGHT}
                      y2={y}
                      stroke="rgba(255,255,255,0.06)"
                      stroke-width="1"
                    />
                    <text
                      x={PAD_LEFT - 6}
                      y={y + 4}
                      text-anchor="end"
                      fill="#8a8b96"
                      font-size="10"
                    >
                      {(tick * 100).toFixed(0)}%
                    </text>
                  </>
                );
              }}
            </For>

            {/* CI band */}
            <path d={ciPathD()} fill="rgba(91,127,214,0.12)" stroke="none" />

            {/* Line */}
            <path
              d={pathD()}
              fill="none"
              stroke="#5b7fd6"
              stroke-width="2"
              stroke-linejoin="round"
            />

            {/* Dots with hover */}
            <For each={plotData()?.points ?? []}>
              {(p) => {
                const y = yScale()(p.entry.win_rate);
                const isHovered = hoveredPoint()?.sha === p.sha;
                return (
                  <g
                    class="chart-dot"
                    onMouseEnter={() => setHoveredPoint({ sha: p.sha, entry: p.entry })}
                    onMouseLeave={() => setHoveredPoint(null)}
                  >
                    <circle
                      cx={p.x}
                      cy={y}
                      r={isHovered ? 6 : 4}
                      fill={isHovered ? "#8aaff0" : "#5b7fd6"}
                      stroke="#0d0e12"
                      stroke-width="1.5"
                    />
                  </g>
                );
              }}
            </For>

            {/* X-axis labels */}
            <For each={plotData()?.points ?? []}>
              {(p, i) => (
                <text
                  x={p.x}
                  y={CHART_H - 8}
                  text-anchor={
                    i() === 0 ? "start" : i() === plotData()!.points.length - 1 ? "end" : "middle"
                  }
                  fill="#6b6e78"
                  font-size="9"
                  transform={`rotate(-25, ${p.x}, ${CHART_H - 8})`}
                >
                  {p.shortSha}
                </text>
              )}
            </For>
          </svg>
        </div>
      </Show>

      {/* Hover tooltip */}
      <Show when={hoveredPoint()}>
        <div id="chart-tooltip">
          <div class="tooltip-sha">{hoveredPoint()!.sha}</div>
          <div class="tooltip-row">
            <span class="tooltip-label">Win rate</span>
            <span class="tooltip-value">{(hoveredPoint()!.entry.win_rate * 100).toFixed(1)}%</span>
          </div>
          <div class="tooltip-row">
            <span class="tooltip-label">95% CI</span>
            <span class="tooltip-value">
              {(hoveredPoint()!.entry.ci_lower * 100).toFixed(1)}% –{" "}
              {(hoveredPoint()!.entry.ci_upper * 100).toFixed(1)}%
            </span>
          </div>
          <div class="tooltip-row">
            <span class="tooltip-label">Games</span>
            <span class="tooltip-value">
              {hoveredPoint()!.entry.wins}W / {hoveredPoint()!.entry.losses}L /{" "}
              {hoveredPoint()!.entry.draws}D
            </span>
          </div>
        </div>
      </Show>
    </div>
  );
};
