// Smac3RunDetail.tsx — Trial history for an open `kind: "smac3"` run:
// stats, a cost-over-trials chart (per-trial cost + running best-so-far),
// and parameter tables comparing candidates with the baseline they faced.
//
// Pure presentational component: `trials`/`tuner`/`launchConfig` are read
// from BenchState by RunDetailPanel.tsx (which owns the tail loop that
// keeps `trials` current — see reducer.ts's `tailTick`) and passed down,
// same convention as WinRateChart/LeaderboardTable reading their slice of
// BenchState directly, except this one takes props since it's nested
// inside another store-reading component rather than mounted as its own
// tab.
//
// Confidence: `cost` is itself already an aggregate win-rate estimate
// (`losses / (2 * rounds)`) over one trial's `rounds` self-play games, not
// a single observation — the intensifier re-evaluating the *same* config
// on a later seed (visible as two trial rows with identical `config`) is
// SMAC building more confidence in that estimate, not noise to ignore. So
// trials are grouped by identical `config` *and* baseline instance (a run
// using SMAC3's multi-instance mechanism scores the same config separately
// against each baseline, and those costs shouldn't be pooled together —
// see `groupKey`), each group's mean cost is treated as a pooled proportion
// over `n = evaluations * 2 * rounds` Bernoulli trials, and a Wilson score
// interval on that is rendered as a whisker per chart point plus a
// headline stat for the best trial's group.

import { createMemo, createSignal, For, Show, type Component } from "solid-js";
import type { ChainedTrial, ChainRung, IncumbentInfo, TrialRow, TunerInfo } from "./index.js";

/** 95% Wilson score interval for a proportion `phat` observed over `n`
 * Bernoulli trials — better-behaved than a normal approximation near 0/1,
 * which is exactly where a saturating cost estimate tends to sit. */
function wilsonInterval(phat: number, n: number, z = 1.96): { lower: number; upper: number } {
  if (n <= 0) return { lower: phat, upper: phat };
  const z2 = z * z;
  const denom = 1 + z2 / n;
  const center = (phat + z2 / (2 * n)) / denom;
  const margin = (z * Math.sqrt((phat * (1 - phat)) / n + z2 / (4 * n * n))) / denom;
  return { lower: Math.max(0, center - margin), upper: Math.min(1, center + margin) };
}

/** The `rounds` actually used for this run's trials — an operator can
 * `--override target.rounds=N` away from the tuner's declared
 * `eval_rounds` default at launch time, and only the run's own launch
 * config (not the tuner metadata) reflects that. Falls back to the
 * tuner's default when the run didn't override it. */
function resolveRounds(launchConfig: unknown, tuner: TunerInfo | null): number {
  const overrides = (launchConfig as { overrides?: unknown } | null)?.overrides;
  if (Array.isArray(overrides)) {
    for (const o of overrides) {
      if (typeof o === "string" && o.startsWith("target.rounds=")) {
        const n = Number(o.slice("target.rounds=".length));
        if (Number.isFinite(n) && n > 0) return n;
      }
    }
  }
  return tuner?.eval_rounds ?? 20;
}

interface TrialGroup {
  trials: TrialRow[];
  meanCost: number;
  n: number;
  ci: { lower: number; upper: number };
}

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

/** Short human label for one rung of a ladder chain, e.g. "Root
 * (root-1)" or "Rung 2 (root-1-ladder2)" -- used in tooltips and the
 * trials table's "Run" column once a chain has more than one rung. */
function chainRungLabel(rung: ChainRung, index: number): string {
  return index === 0 ? `Root (${rung.run_id})` : `Rung ${index + 1} (${rung.run_id})`;
}

interface ParamDiffRow {
  name: string;
  value: unknown;
  baseline: unknown;
  changed: boolean;
}

/** Diff a config's active parameters against the actual baseline config
 * used for this run. */
function paramsVsBaseline(
  config: Record<string, unknown> | undefined,
  baseline: Record<string, unknown> | null,
): ParamDiffRow[] {
  if (!config || !baseline) return [];
  return Object.entries(config).map(([name, value]) => {
    const baselineValue = baseline[name];
    const changed = baselineValue !== undefined && JSON.stringify(baselineValue) !== JSON.stringify(value);
    return { name, value, baseline: baselineValue, changed };
  });
}

/** The launch record stores the resolved parameter settings for each
 * baseline instance. A promoted rung also keeps the raw setting under
 * `baseline_configs`, because that is what the CLI forwards to the game.
 * A comparison only makes sense with exactly one baseline instance. */
function baselineConfig(launchConfig: unknown): Record<string, unknown> | null {
  const config = launchConfig as {
    baseline_settings?: unknown;
    baseline_configs?: unknown;
    overrides?: unknown;
  } | null;
  const settings = config?.baseline_settings ?? config?.baseline_configs;
  if (settings && typeof settings === "object" && !Array.isArray(settings)) {
    const entries = Object.values(settings as Record<string, unknown>);
    if (entries.length !== 1) return null;
    const baseline = entries[0];
    return baseline && typeof baseline === "object" && !Array.isArray(baseline)
      ? baseline as Record<string, unknown>
      : null;
  }

  // Runs launched before `baseline_settings` was persisted still identify
  // their floor opponent in the launch override. Only the two deliberately
  // supported floor families are reconstructable here; named game presets
  // are intentionally neither inferred nor displayed as tuning baselines.
  if (!Array.isArray(config?.overrides)) return null;
  let baselineOverride: string | undefined;
  for (let i = config.overrides.length - 1; i >= 0; i--) {
    const value = config.overrides[i];
    if (typeof value === "string" && value.startsWith("target.baselines=")) {
      baselineOverride = value;
      break;
    }
  }
  const match = baselineOverride?.match(/^target\.baselines=\[(['"])(flat_mc|random)\1\]$/);
  if (!match) return null;
  return { family: match[2], q_init: "Infinity" };
}

/** The baseline instance a trial's cost was measured against, when the run
 * used SMAC3's multi-instance mechanism (`Scenario(instances=...)`) --
 * `TrialTracker` (Python) stuffs it into the existing `extra` JSON column
 * as `{"instance": "master"}` rather than a new top-level DB column. `null`
 * for a single-instance run (nothing to disambiguate) or an older trial
 * recorded before this existed. */
function instanceOf(t: TrialRow): string | null {
  const extra = t.extra as { instance?: unknown } | null;
  return typeof extra?.instance === "string" ? extra.instance : null;
}

/** Grouping key for the CI-band pooling below: same `config` *and* same
 * baseline instance. Pooling across different instances would hide exactly
 * the signal multi-instance evaluation exists to expose -- a config that's
 * saturated against one baseline can still have a very different win rate
 * against another. */
function groupKey(t: TrialRow): string {
  return JSON.stringify(t.config) + "::" + (instanceOf(t) ?? "");
}

interface ChartPoint {
  x: number;
  trial: TrialRow;
  rungIndex: number;
  bestSoFar: number;
  /** Running minimum of each point's group's Wilson CI *upper* bound -- the
   * cost this config is confirmed, at 95% confidence, to be no worse than.
   * Unlike `bestSoFar` (the raw observed minimum, which a single lucky
   * evaluation can drag down), this only improves once repeat evaluations
   * have actually shrunk a group's interval -- see the chart's help
   * popover for the operator-facing explanation of why both lines exist. */
  confirmedFloor: number;
}

/** A vertical marker at the trial-index boundary between two rungs of a
 * ladder chain -- the moment an operator's "use best as new baseline"
 * (or the automated ladder driver) promoted the prior rung's incumbent
 * and relaunched against it. */
interface RungBoundary {
  x: number;
  rung: ChainRung;
}

export const Smac3RunDetail: Component<{
  /** This run's own trials -- used as a fallback source before `chain`/
   * `chainedTrials` resolve on the first tick (see `effectiveEntries`), and
   * always the value shown by the "Trials" stat when the chain is a single
   * rung (the common, non-laddered case). */
  trials: TrialRow[];
  /** This run's ladder chain, oldest rung first (one element for a plain
   * run). Empty before the first tick resolves. */
  chain: ChainRung[];
  /** Every rung's trials concatenated in chain order -- the data source for
   * the chart and trials table once non-empty, so pausing and resuming a
   * run (or manually advancing its baseline) renders as one continuous
   * timeline instead of a graph that resets per launch. */
  chainedTrials: ChainedTrial[];
  tuner: TunerInfo | null;
  /** `RunDetail.config` (the launch request body) — only consulted for a
   * `target.rounds=N` override; see `resolveRounds`. */
  launchConfig?: unknown;
  /** `RunDetail.incumbent` -- SMAC3's own tracked best config, distinct
   * from the "Best trial" stat below (which is just the lowest raw `cost`
   * among this run's trials, not aggregated across baseline instances).
   * `null`/absent before the run reports its first incumbent. */
  incumbent?: IncumbentInfo | null;
}> = (props) => {
  // `chainedTrials` covers the *entire* chain (this run plus every earlier
  // rung its baseline was advanced from), so it's preferred whenever
  // available; `trials` (this run alone) is only a fallback for the first
  // tick or two before the chain fetch resolves, to avoid a flash of "No
  // scored trials yet." Trial ids repeat across rungs (each is its own
  // SMAC3 run starting from trial 1), so entries are ordered by rung first,
  // trial_id within a rung second -- never by trial_id alone.
  const effectiveEntries = createMemo((): ChainedTrial[] => {
    if (props.chainedTrials.length > 0) {
      return [...props.chainedTrials].sort(
        (a, b) => a.rungIndex - b.rungIndex || a.trial.trial_id - b.trial.trial_id,
      );
    }
    return [...props.trials]
      .sort((a, b) => a.trial_id - b.trial_id)
      .map((trial) => ({ rungIndex: 0, trial }));
  });
  const sorted = createMemo(() => effectiveEntries().map((e) => e.trial));
  const scoredEntries = createMemo(() => effectiveEntries().filter((e) => e.trial.cost !== null));
  const scored = createMemo(() => scoredEntries().map((e) => e.trial));
  const rounds = createMemo(() => resolveRounds(props.launchConfig, props.tuner));

  // Trial ids restart at 1 in every rung, so "the best trial" has to be
  // tracked as a (rungIndex, trial_id) pair, not trial_id alone -- otherwise
  // two different rungs' trial #1 would both render as "the" best point.
  const bestEntry = createMemo((): ChainedTrial | null => {
    let best: ChainedTrial | null = null;
    for (const e of scoredEntries()) {
      if (best === null || (e.trial.cost as number) < (best.trial.cost as number)) best = e;
    }
    return best;
  });
  const bestTrial = createMemo((): TrialRow | null => bestEntry()?.trial ?? null);
  const isBest = (p: { rungIndex: number; trial: TrialRow }): boolean => {
    const best = bestEntry();
    return best !== null && best.rungIndex === p.rungIndex && best.trial.trial_id === p.trial.trial_id;
  };

  // A repeated `config` across trials (the intensifier re-evaluating the
  // same candidate on a later seed) is pooled into one group so its cost
  // estimate gets a tighter interval than any single evaluation -- but only
  // within the same baseline instance (see `groupKey`'s doc comment).
  const groups = createMemo((): Map<string, TrialGroup> => {
    const byKey = new Map<string, TrialRow[]>();
    for (const t of scored()) {
      const key = groupKey(t);
      const list = byKey.get(key);
      if (list) list.push(t);
      else byKey.set(key, [t]);
    }
    const r = rounds();
    const result = new Map<string, TrialGroup>();
    for (const [key, trials] of byKey) {
      const meanCost = trials.reduce((sum, t) => sum + (t.cost as number), 0) / trials.length;
      const n = trials.length * 2 * r;
      result.set(key, { trials, meanCost, n, ci: wilsonInterval(meanCost, n) });
    }
    return result;
  });

  const groupFor = (t: TrialRow): TrialGroup => groups().get(groupKey(t))!;

  const bestGroup = createMemo((): TrialGroup | null => {
    const best = bestTrial();
    return best ? (groups().get(groupKey(best)) ?? null) : null;
  });

  const chartPoints = createMemo((): ChartPoint[] => {
    const ts = scoredEntries();
    if (ts.length === 0) return [];
    let best = Infinity;
    let confirmedFloor = Infinity;
    return ts.map((entry, i) => {
      best = Math.min(best, entry.trial.cost as number);
      confirmedFloor = Math.min(confirmedFloor, groupFor(entry.trial).ci.upper);
      const x = PAD_LEFT + (ts.length > 1 ? (i / (ts.length - 1)) * PLOT_W : PLOT_W / 2);
      return { x, trial: entry.trial, rungIndex: entry.rungIndex, bestSoFar: best, confirmedFloor };
    });
  });

  // One marker per rung boundary (chain.length - 1 of them), positioned
  // midway between the last point of the earlier rung and the first point
  // of the later one -- `chain[k].incumbent` is exactly the cost that rung
  // was promoted to a baseline at (see `ChainRung`'s doc comment), which is
  // what makes it the right label for "why did the curve jump here."
  const rungBoundaries = createMemo((): RungBoundary[] => {
    const pts = chartPoints();
    const boundaries: RungBoundary[] = [];
    for (let k = 1; k < props.chain.length; k++) {
      const rung = props.chain[k];
      if (!rung) continue;
      let prevLastX: number | null = null;
      let curFirstX: number | null = null;
      for (const p of pts) {
        if (p.rungIndex === k - 1) prevLastX = p.x;
        if (p.rungIndex === k && curFirstX === null) curFirstX = p.x;
      }
      if (curFirstX === null) continue;
      boundaries.push({ x: prevLastX === null ? curFirstX : (prevLastX + curFirstX) / 2, rung });
    }
    return boundaries;
  });

  const yMax = createMemo(() => {
    const pts = chartPoints();
    if (pts.length === 0) return 1;
    // Include each point's CI upper bound -- a low-n group's interval can
    // reach above every observed cost in the run, and clipping it would
    // hide exactly the uncertainty this chart exists to show.
    const max = Math.max(...pts.map((p) => Math.max(p.trial.cost as number, groupFor(p.trial).ci.upper)));
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

  const confirmedFloorPathD = createMemo(() => {
    const pts = chartPoints();
    if (pts.length === 0) return "";
    const scale = yScale();
    return pts
      .map((p, i) => `${i === 0 ? "M" : "L"}${p.x.toFixed(1)},${scale(p.confirmedFloor).toFixed(1)}`)
      .join(" ");
  });

  const yTicks = createMemo(() => {
    const max = yMax();
    const step = max / 4;
    return [0, step, step * 2, step * 3, max];
  });

  // Two distinct "best" configs, deliberately not conflated (see the chart
  // help popover's confirmed-floor/best-so-far split for the same
  // distinction applied to the cost chart): the incumbent is SMAC3's own
  // tracked champion -- re-challenged as the run progresses, and exactly
  // what "Use best as new baseline" actually promotes -- while the lowest
  // trial is just this chart's single cheapest observed dot, with no
  // confirmation behind it.
  // This is recorded when the run is launched, before SMAC has an
  // incumbent. It therefore remains the authoritative opponent for the
  // root rung as well as for later promoted rungs.
  const currentBaselineConfig = createMemo(() => baselineConfig(props.launchConfig));

  const incumbentVsBaseline = createMemo(() => paramsVsBaseline(props.incumbent?.config, currentBaselineConfig()));
  const lowestTrialVsBaseline = createMemo(() => paramsVsBaseline(bestTrial()?.config, currentBaselineConfig()));

  const [helpOpen, setHelpOpen] = createSignal(false);

  const [copied, setCopied] = createSignal(false);
  async function copyIncumbentConfig(): Promise<void> {
    const incumbent = props.incumbent;
    if (!incumbent) return;
    await navigator.clipboard.writeText(JSON.stringify(incumbent.config));
    setCopied(true);
    setTimeout(() => setCopied(false), 1500);
  }

  return (
    <div id="smac3-run-detail">
      <Show when={props.incumbent}>
        {(incumbent) => (
          <div id="smac3-incumbent-row">
            <span class="smac3-stat-label">Incumbent</span>
            <span class="smac3-stat-value">{fmtCost(incumbent().cost)}</span>
            <button
              id="smac3-copy-incumbent-btn"
              onClick={copyIncumbentConfig}
              title="Copy this config for a later run's --baseline-config"
            >
              {copied() ? "Copied!" : "Copy as baseline config"}
            </button>
          </div>
        )}
      </Show>

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
          <Show when={bestGroup()}>
            {(group) => (
              <>
                <div class="smac3-stat">
                  <span class="smac3-stat-value">{group().trials.length}</span>
                  <span class="smac3-stat-label">Evaluations</span>
                </div>
                <div class="smac3-stat">
                  <span class="smac3-stat-value">
                    {fmtCost(group().ci.lower)} – {fmtCost(group().ci.upper)}
                  </span>
                  <span class="smac3-stat-label">95% CI</span>
                </div>
              </>
            )}
          </Show>
        </div>

        <div id="smac3-chart-wrapper">
          <div id="smac3-chart-header">
            <span id="smac3-chart-title">Cost history</span>
            <div id="smac3-chart-help">
              <button
                id="smac3-chart-help-btn"
                type="button"
                aria-expanded={helpOpen()}
                aria-label="How to read this chart"
                onClick={() => setHelpOpen((v) => !v)}
              >
                i
              </button>
              <Show when={helpOpen()}>
                <div id="smac3-chart-help-popover" role="tooltip">
                  <button
                    id="smac3-chart-help-close"
                    type="button"
                    aria-label="Close"
                    onClick={() => setHelpOpen(false)}
                  >
                    ×
                  </button>
                  <dl>
                    <dt><i class="legend-swatch legend-swatch-trial" /> Trial cost</dt>
                    <dd>One dot per evaluated config: its raw win/loss rate over that trial's games. Noisy on its own -- a single trial is a small sample.</dd>
                    <dt><i class="legend-swatch legend-swatch-floor" /> Confirmed floor</dt>
                    <dd>The bold line, and the one to trust. The most conservative this run can currently claim: the best 95%-confidence "no worse than this" bound, across every repeat evaluation of a config. Only drops once a result is actually confirmed, not just observed once -- use this to judge whether a config is safe to promote.</dd>
                    <dt><i class="legend-swatch legend-swatch-best" /> Best so far</dt>
                    <dd>The faint dashed line. The lowest cost observed at any single point, with no confirmation behind it -- deliberately de-emphasized, since one lucky trial can pull it down without the config actually being better.</dd>
                    <dt><i class="legend-swatch legend-swatch-ci" /> 95% CI whisker</dt>
                    <dd>Per trial, the plausible range for that config's true cost given how many games (and repeat evaluations) back it up -- wider means less confidence.</dd>
                    <Show when={props.chain.length > 1}>
                      <dt><i class="legend-swatch legend-swatch-boundary" /> New baseline</dt>
                      <dd>A vertical line marking where a run's incumbent was promoted to a new baseline and tuning continued against it.</dd>
                    </Show>
                    <dt>Incumbent</dt>
                    <dd>SMAC3's own tracked best config, aggregated across every baseline instance and re-challenged as the run progresses. This is what "Use best as new baseline" actually promotes -- not the same as "Lowest single trial" below, which is just this chart's single cheapest dot, unconfirmed.</dd>
                  </dl>
                </div>
              </Show>
            </div>
          </div>
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

            {/* Running best-so-far step line -- deliberately the fainter,
                thinner, dashed line: it's the raw observed minimum, which a
                single lucky trial can drag down with no confirmation behind
                it. Drawn first (underneath) so the bold confirmed-floor
                line on top doesn't get visually buried by it. */}
            <path
              d={bestPathD()}
              fill="none"
              stroke="#4caf7a"
              stroke-width="1.25"
              stroke-dasharray="3,3"
              stroke-opacity="0.55"
              stroke-linejoin="round"
            />

            {/* Confirmed floor: running-min of each point's CI *upper*
                bound -- the bold, solid, prominent line, since this is the
                one that's actually safe to act on (see the help popover).
                It sits at or above best-so-far by construction; the gap
                between the two is itself the "how much do I trust this"
                signal. */}
            <path
              d={confirmedFloorPathD()}
              fill="none"
              stroke="#e0904a"
              stroke-width="2.5"
              stroke-linejoin="round"
            />

            {/* Per-point confidence whisker -- the Wilson interval of the
                point's config group, shared across every point in that
                group (they're the same pooled estimate). Drawn under the
                dots so a tight interval doesn't get visually lost. */}
            <For each={chartPoints()}>
              {(p) => {
                const ci = () => groupFor(p.trial).ci;
                return (
                  <line
                    x1={p.x}
                    x2={p.x}
                    y1={yScale()(ci().lower)}
                    y2={yScale()(ci().upper)}
                    stroke="rgba(91,127,214,0.35)"
                    stroke-width="2"
                    class="smac3-ci-whisker"
                  />
                );
              }}
            </For>

            {/* Per-trial cost dots */}
            <For each={chartPoints()}>
              {(p) => (
                <circle
                  cx={p.x}
                  cy={yScale()(p.trial.cost as number)}
                  r={isBest(p) ? 4 : 2.5}
                  fill={isBest(p) ? "#4caf7a" : "#5b7fd6"}
                >
                  <title>
                    {props.chain.length > 1 ? `${chainRungLabel(props.chain[p.rungIndex]!, p.rungIndex)} ` : ""}
                    Trial #{p.trial.trial_id}: cost {fmtCost(p.trial.cost)} (group of{" "}
                    {groupFor(p.trial).trials.length}, 95% CI {fmtCost(groupFor(p.trial).ci.lower)}
                    {" – "}
                    {fmtCost(groupFor(p.trial).ci.upper)})
                  </title>
                </circle>
              )}
            </For>

            {/* Baseline-cutover markers -- one per rung boundary, labeled
                with the cost the prior rung's incumbent was promoted at. */}
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
                    class="smac3-rung-boundary"
                  >
                    <title>
                      New baseline from {b.rung.run_id}
                      {b.rung.incumbent ? ` (promoted at ${fmtCost(b.rung.incumbent.cost)} loss)` : ""}
                    </title>
                  </line>
                  <circle cx={b.x} cy={PAD_TOP} r={2.5} fill="#c9a227" />
                </>
              )}
            </For>
          </svg>
          <div class="smac3-chart-legend">
            <span><i class="legend-swatch legend-swatch-trial" /> trial cost</span>
            <span class="smac3-legend-emphasized"><i class="legend-swatch legend-swatch-floor" /> confirmed floor</span>
            <span class="smac3-legend-muted"><i class="legend-swatch legend-swatch-best" /> best so far</span>
            <span><i class="legend-swatch legend-swatch-ci" /> 95% CI</span>
            <Show when={props.chain.length > 1}>
              <span><i class="legend-swatch legend-swatch-boundary" /> new baseline</span>
            </Show>
          </div>
        </div>

        <Show when={incumbentVsBaseline().length > 0}>
          <table id="smac3-incumbent-diff-table" class="smac3-diff-table smac3-diff-table-emphasized">
            <caption>Incumbent vs. baseline</caption>
            <thead>
              <tr>
                <th>Parameter</th>
                <th>Incumbent</th>
                <th>Baseline</th>
              </tr>
            </thead>
            <tbody>
              <For each={incumbentVsBaseline()}>
                {(row) => (
                  <tr classList={{ "smac3-diff-changed": row.changed }}>
                    <td class="smac3-param-name">{row.name}</td>
                    <td>{String(row.value)}</td>
                    <td>{row.baseline === undefined ? "—" : String(row.baseline)}</td>
                  </tr>
                )}
              </For>
            </tbody>
          </table>
        </Show>

        <Show when={lowestTrialVsBaseline().length > 0}>
          <table id="smac3-lowest-trial-diff-table" class="smac3-diff-table">
            <caption>Lowest single trial (#{bestTrial()!.trial_id}) vs. baseline</caption>
            <thead>
              <tr>
                <th>Parameter</th>
                <th>Lowest</th>
                <th>Baseline</th>
              </tr>
            </thead>
            <tbody>
              <For each={lowestTrialVsBaseline()}>
                {(row) => (
                  <tr classList={{ "smac3-diff-changed": row.changed }}>
                    <td class="smac3-param-name">{row.name}</td>
                    <td>{String(row.value)}</td>
                    <td>{row.baseline === undefined ? "—" : String(row.baseline)}</td>
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
              <Show when={props.chain.length > 1}>
                <th>Run</th>
              </Show>
              <th>Family</th>
              <th>Baseline</th>
              <th>Cost</th>
              <th>Seed</th>
              <th>Time</th>
            </tr>
          </thead>
          <tbody>
            <For each={effectiveEntries().slice().reverse()}>
              {(e) => (
                <tr classList={{ "smac3-trial-best": isBest(e) }} title={JSON.stringify(e.trial.config)}>
                  <td>{e.trial.trial_id}</td>
                  <Show when={props.chain.length > 1}>
                    <td class="smac3-trial-rung">{chainRungLabel(props.chain[e.rungIndex]!, e.rungIndex)}</td>
                  </Show>
                  <td class="smac3-trial-family">{typeof e.trial.config.family === "string" ? e.trial.config.family : "—"}</td>
                  <td class="smac3-trial-baseline">{instanceOf(e.trial) ?? "—"}</td>
                  <td>{fmtCost(e.trial.cost)}</td>
                  <td>{e.trial.seed ?? "—"}</td>
                  <td>{e.trial.ts}</td>
                </tr>
              )}
            </For>
          </tbody>
        </table>
      </div>
    </div>
  );
};
