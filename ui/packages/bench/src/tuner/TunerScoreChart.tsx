/** TunerScoreChart — SVG chart of a tuner run's trial scores over time.
 *
 * The Y axis shows `mu - 3*sigma` — higher is better. Every trial is
 * scored via the optuna pipeline's OpenSkill-based matchmaking.
 *
 * Semantics (see the help popover for the full explanation):
 *   - Dots   — one per scored trial
 *   - Solid  — bold confirmed-lower-bound line (running max of lower CI)
 *   - Dashed — faint best-so-far line (raw max score so far)
 *   - Bar    — per-point 95% confidence whisker (mu ± 2σ)
 *   - Flags  — rung-boundary markers at baseline promotions
 *
 * Chart dimensions are fixed; the SVG viewBox scales naturally with CSS
 * `max-width: 100%`.
 */

import { createMemo, createSignal, For, Show, type Component } from "solid-js";
import type { ChainedTrial, ChainRung, TrialRow } from "../index.js";
import {
  buildGroups,
  fmtScore,
  groupKey,
  trialScore,
} from "./tuner-helpers.js";

// ── Layout constants ──────────────────────────────────────────────

const CHART_W = 480;
const CHART_H = 160;
const PAD_LEFT = 40;
const PAD_RIGHT = 12;
const PAD_TOP = 12;
const PAD_BOTTOM = 20;
const PLOT_W = CHART_W - PAD_LEFT - PAD_RIGHT;
const PLOT_H = CHART_H - PAD_TOP - PAD_BOTTOM;

// ── Helpers ───────────────────────────────────────────────────────

function chainRungLabel(rung: ChainRung, index: number): string {
  return index === 0
    ? `Root (${rung.run_id})`
    : `Rung ${index + 1} (${rung.run_id})`;
}

// ── Component ─────────────────────────────────────────────────────

export const TunerScoreChart: Component<{
  scoredEntries: ChainedTrial[];
  scored: TrialRow[];
  chain: ChainRung[];
  bestTrial: TrialRow | null;
}> = (props) => {
  const groups = createMemo(() => buildGroups(props.scored));
  const groupFor = (t: TrialRow) => groups().get(groupKey(t))!;

  // ── Chart points ──────────────────────────────────────────────

  const chartPoints = createMemo(() => {
    const ts = props.scoredEntries;
    if (ts.length === 0) return [] as (ChainedTrial & { x: number; bestSoFar: number; confirmedFloor: number })[];

    let best = -Infinity;
    let confirmedFloor = -Infinity;

    return ts.map((entry, i) => {
      const score = trialScore(entry.trial) ?? -Infinity;
      best = Math.max(best, score);
      const g = groupFor(entry.trial);
      confirmedFloor = Math.max(confirmedFloor, g.ci.lower);

      const x = PAD_LEFT + (ts.length > 1 ? (i / (ts.length - 1)) * PLOT_W : PLOT_W / 2);
      return { ...entry, x, bestSoFar: best, confirmedFloor };
    });
  });

  // ── Y axis ────────────────────────────────────────────────────

  const yMin = createMemo(() => {
    const pts = chartPoints();
    if (pts.length === 0) return -1;
    const min = Math.min(
      ...pts.map((p) => {
        const g = groupFor(p.trial);
        return Math.min(trialScore(p.trial) ?? 0, g.ci.lower);
      }),
    );
    return Math.min(0, min * 1.05 - 0.5);
  });

  const yMax = createMemo(() => {
    const pts = chartPoints();
    if (pts.length === 0) return 1;
    const max = Math.max(
      ...pts.map((p) => {
        const g = groupFor(p.trial);
        return Math.max(trialScore(p.trial) ?? -Infinity, g.ci.upper);
      }),
    );
    return Math.max(0.05, max * 1.1 || 1);
  });

  const yScale = createMemo(() => {
    const range = yMax() - yMin();
    return (v: number) => PAD_TOP + PLOT_H - ((v - yMin()) / range) * PLOT_H;
  });

  // ── Lines ─────────────────────────────────────────────────────

  const bestPathD = createMemo(() => {
    const pts = chartPoints();
    if (pts.length === 0) return "";
    const scale = yScale();
    return pts
      .map(
        (p, i) =>
          `${i === 0 ? "M" : "L"}${p.x.toFixed(1)},${scale(p.bestSoFar).toFixed(1)}`,
      )
      .join(" ");
  });

  const confirmedPathD = createMemo(() => {
    const pts = chartPoints();
    if (pts.length === 0) return "";
    const scale = yScale();
    return pts
      .map(
        (p, i) =>
          `${i === 0 ? "M" : "L"}${p.x.toFixed(1)},${scale(p.confirmedFloor).toFixed(1)}`,
      )
      .join(" ");
  });

  // ── Y ticks ───────────────────────────────────────────────────

  const yTicks = createMemo(() => {
    const range = yMax() - yMin();
    const step = range / 4;
    return Array.from({ length: 5 }, (_, i) => yMin() + step * i);
  });

  // ── Rung boundaries ───────────────────────────────────────────

  const rungBoundaries = createMemo(() => {
    const pts = chartPoints();
    const boundaries: { x: number; rung: ChainRung }[] = [];
    for (let k = 1; k < props.chain.length; k++) {
      const rung = props.chain[k];
      if (!rung) continue;
      let prevLastX: number | null = null;
      let curFirstX: number | null = null;
      for (const p of pts) {
        if (p.rungIndex === k - 1) prevLastX = p.x;
        if (p.rungIndex === k && curFirstX === null) curFirstX = p.x;
      }
      if (curFirstX === null && prevLastX === null) continue;
      const x =
        curFirstX === null
          ? prevLastX!
          : prevLastX === null
            ? curFirstX
            : (prevLastX + curFirstX) / 2;
      boundaries.push({ x, rung });
    }
    return boundaries;
  });

  // ── Best-point check ──────────────────────────────────────────

  const isBestTrial = (t: TrialRow): boolean =>
    props.bestTrial !== null && t.trial_id === props.bestTrial.trial_id;

  // ── Help popover ──────────────────────────────────────────────

  const [helpOpen, setHelpOpen] = createSignal(false);

  return (
    <div id="tuner-chart-wrapper">
      <div id="tuner-chart-header">
        <span id="tuner-chart-title">Score history</span>
        <div id="tuner-chart-help">
          <button
            id="tuner-chart-help-btn"
            type="button"
            aria-expanded={helpOpen()}
            aria-label="How to read this chart"
            onClick={() => setHelpOpen((v) => !v)}
          >
            i
          </button>
          <Show when={helpOpen()}>
            <div id="tuner-chart-help-popover" role="tooltip">
              <button
                id="tuner-chart-help-close"
                type="button"
                aria-label="Close"
                onClick={() => setHelpOpen(false)}
              >
                ×
              </button>
              <dl>
                <dt><i class="legend-swatch legend-swatch-trial" /> Trial score</dt>
                <dd>
                  One dot per evaluated config: its OpenSkill estimate
                  <code> mu − 3σ</code> after matchmaking against the dynamic
                  opponent pool. Higher is better. A single trial's score is
                  still noisy but reflects play against sensible opponents
                  (not just one fixed baseline).
                </dd>
                <dt><i class="legend-swatch legend-swatch-floor" /> Confirmed lower bound</dt>
                <dd>
                  The bold line, and the one to trust. The running maximum of
                  the 95% confidence band's <em>lower</em> end
                  (<code> mu − 2σ</code>) — we're confident the true skill is
                  at least this high. Only rises once repeat evaluations (or a
                  clearly stronger config) actually confirm it.
                </dd>
                <dt><i class="legend-swatch legend-swatch-best" /> Best so far</dt>
                <dd>
                  The faint dashed line. The highest raw score observed, with
                  no confidence adjustment — de-emphasized because a single
                  lucky matchmaking sequence can inflate it.
                </dd>
                <dt><i class="legend-swatch legend-swatch-ci" /> 95% CI whisker</dt>
                <dd>
                  <code> mu ± 2σ</code>, the OpenSkill confidence band for
                  each trial. Wider means less certainty in the rating.
                </dd>
                <Show when={props.chain.length > 1}>
                  <dt><i class="legend-swatch legend-swatch-boundary" /> New baseline</dt>
                  <dd>
                    A vertical line marking where a run's incumbent was
                    promoted to a new baseline and tuning continued against it.
                  </dd>
                </Show>
                <dt>Incumbent</dt>
                <dd>
                  tuner's own tracked best config, aggregated across every
                  baseline instance and re-challenged as the run progresses.
                </dd>
              </dl>
            </div>
          </Show>
        </div>
      </div>

      <svg
        width={CHART_W}
        height={CHART_H}
        viewBox={`0 0 ${CHART_W} ${CHART_H}`}
        id="tuner-cost-chart"
      >
        {/* Y grid lines and labels */}
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
                  y={y + 3}
                  text-anchor="end"
                  fill="#8a8b96"
                  font-size="9"
                >
                  {tick.toFixed(2)}
                </text>
              </>
            );
          }}
        </For>

        {/* Best-so-far line — faint dashed (de-emphasized: no confidence
            behind it, just the raw maximum observed so far). */}
        <path
          d={bestPathD()}
          fill="none"
          stroke="#4caf7a"
          stroke-width="1.25"
          stroke-dasharray="3,3"
          stroke-opacity="0.55"
          stroke-linejoin="round"
        />

        {/* Confirmed lower bound — bold solid (this is the one to trust). */}
        <path
          d={confirmedPathD()}
          fill="none"
          stroke="#e0904a"
          stroke-width="2.5"
          stroke-linejoin="round"
        />

        {/* Per-point CI whiskers (mu ± 2σ) */}
        <For each={chartPoints()}>
          {(p) => {
            const g = groupFor(p.trial);
            return (
              <line
                x1={p.x}
                x2={p.x}
                y1={yScale()(g.ci.lower)}
                y2={yScale()(g.ci.upper)}
                stroke="rgba(91,127,214,0.35)"
                stroke-width="2"
                class="tuner-ci-whisker"
              />
            );
          }}
        </For>

        {/* Per-trial score dots */}
        <For each={chartPoints()}>
          {(p) => {
            const score = trialScore(p.trial) ?? 0;
            return (
              <circle
                cx={p.x}
                cy={yScale()(score)}
                r={isBestTrial(p.trial) ? 4 : 2.5}
                fill={isBestTrial(p.trial) ? "#4caf7a" : "#5b7fd6"}
              >
                <title>
                  {props.chain.length > 1
                    ? `${chainRungLabel(props.chain[p.rungIndex]!, p.rungIndex)} `
                    : ""}
                  Trial #{p.trial.trial_id}: score {fmtScore(trialScore(p.trial))}
                  {" (group of "}
                  {groupFor(p.trial).trials.length}
                  {")"}
                </title>
              </circle>
            );
          }}
        </For>

        {/* Rung-boundary markers */}
        <For each={rungBoundaries()}>
          {(b) => (
            <>
              <line
                x1={b.x}
                y1={PAD_TOP}
                x2={b.x}
                y2={PAD_TOP + PLOT_H}
                stroke="#c9a227"
                stroke-width="1"
                stroke-dasharray="3,3"
                class="tuner-rung-boundary"
              >
                <title>
                  New baseline from {b.rung.run_id}
                  {b.rung.incumbent
                    ? ` (promoted at score ${fmtScore(-b.rung.incumbent.cost!)})`
                    : ""}
                </title>
              </line>
              <circle cx={b.x} cy={PAD_TOP} r={2.5} fill="#c9a227" />
            </>
          )}
        </For>
      </svg>

      <div class="tuner-chart-legend">
        <span><i class="legend-swatch legend-swatch-trial" /> trial score</span>
        <span class="tuner-legend-emphasized"><i class="legend-swatch legend-swatch-floor" /> confirmed lower bound</span>
        <span class="tuner-legend-muted"><i class="legend-swatch legend-swatch-best" /> best so far</span>
        <span><i class="legend-swatch legend-swatch-ci" /> 95% CI</span>
        <Show when={props.chain.length > 1}>
          <span><i class="legend-swatch legend-swatch-boundary" /> new baseline</span>
        </Show>
      </div>
    </div>
  );
};