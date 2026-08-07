// AnalysisPanel.tsx — Analysis panel (PLAN-UI.md session 6): dispatches
// `analyze` (Session 3's job-poll wiring), renders a scrollable table of
// candidate moves by visit share / mean value, the principal variation, and
// an "Analyze" button -- analysis is real compute, so it never fires
// automatically on navigation (see GameShell.tsx, which only ever dispatches
// it in response to this panel's `onAnalyze`).
//
// Per PLAN-UI.md's hard rule, this component never touches the network --
// its only outputs are `onSelectPreset`/`onAnalyze`/`onHoverMove`, which
// GameShell wires to dispatch/local state. It also never builds the board
// heatmap itself: `analysisOverlay` (threaded into the renderer) is derived
// by GameShell from the same `analysis` state this panel reads, so both stay
// in sync off one source of truth.

import { type Component, createMemo, For, Show } from "solid-js";
import type { AiPresetInfo, Analysis } from "@mcts/game";
import { moveEquals } from "@mcts/game";
import type { JobPollState } from "@mcts/core";

type S = unknown;
type M = unknown;

interface CandidateRow {
  move: M;
  label: string;
  visitShare: number;
  meanValue: number;
  isProven: boolean;
  isSuggested: boolean;
}

export const AnalysisPanel: Component<{
  analysis: JobPollState<Analysis<M>>;
  presets: AiPresetInfo[];
  selectedPreset: string;
  before: S;
  formatMove?: (move: M, before: S) => string;
  busy: boolean;
  hoveredMove: M | null;
  onSelectPreset: (preset: string) => void;
  onAnalyze: () => void;
  onHoverMove: (move: M | null) => void;
}> = (props) => {
  const result = createMemo(() => (props.analysis.status === "done" ? props.analysis.result : null));

  const rows = createMemo((): CandidateRow[] => {
    const r = result();
    if (!r) return [];
    const total = r.total_visits || 1;
    return [...r.actions]
      .sort((a, b) => b.visits - a.visits)
      .map((c) => ({
        move: c.action,
        label: props.formatMove?.(c.action, props.before) ?? JSON.stringify(c.action),
        visitShare: c.visits / total,
        meanValue: c.mean_value,
        isProven: c.is_proven,
        isSuggested: r.suggested_move !== null && moveEquals(c.action, r.suggested_move),
      }));
  });

  // Only the PV's first move (the one played from `before`, the current
  // position) has a state this panel can format against -- later PV entries
  // are hypothetical future positions nothing here has a `before` for, so
  // they fall back to the same `JSON.stringify` this panel uses for a game
  // module that omits `formatMove` entirely.
  const pvLabel = createMemo(() => {
    const r = result();
    if (!r || r.principal_variation.length === 0) return null;
    return r.principal_variation
      .map((m, i) => (i === 0 ? (props.formatMove?.(m, props.before) ?? JSON.stringify(m)) : JSON.stringify(m)))
      .join(" → ");
  });

  return (
    <div id="analysis-panel">
      <div class="analysis-header">
        <select
          disabled={props.busy || props.presets.length === 0}
          value={props.selectedPreset}
          onChange={(e) => props.onSelectPreset(e.currentTarget.value)}
        >
          <For each={props.presets}>{(p) => <option value={p.id}>{p.label}</option>}</For>
        </select>
        <button disabled={props.busy || props.presets.length === 0} onClick={() => props.onAnalyze()}>
          {props.analysis.status === "pending" ? "Analyzing…" : "Analyze"}
        </button>
      </div>
      <Show when={props.analysis.status === "error"}>
        <div class="analysis-error">{props.analysis.error}</div>
      </Show>
      <Show when={result()}>
        {(r) => (
          <>
            <div class="analysis-summary">{r().total_visits} visits</div>
            <Show when={pvLabel()}>{(pv) => <div class="analysis-pv">PV: {pv()}</div>}</Show>
            <ul class="analysis-candidates">
              <For each={rows()}>
                {(row) => (
                  <li
                    classList={{ suggested: row.isSuggested, hovered: props.hoveredMove !== null && moveEquals(props.hoveredMove, row.move) }}
                    onMouseEnter={() => props.onHoverMove(row.move)}
                    onMouseLeave={() => props.onHoverMove(null)}
                  >
                    <span class="candidate-bar" style={{ width: `${Math.round(row.visitShare * 100)}%` }} />
                    <span class="candidate-label">{row.label}</span>
                    <Show when={row.isProven}>
                      <span class="candidate-proven" title="Proven win">
                        ✓
                      </span>
                    </Show>
                    <span class="candidate-share">{Math.round(row.visitShare * 100)}%</span>
                    <span class="candidate-value">{row.meanValue.toFixed(2)}</span>
                  </li>
                )}
              </For>
            </ul>
          </>
        )}
      </Show>
    </div>
  );
};
