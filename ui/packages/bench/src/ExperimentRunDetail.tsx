import { For, Show, type Component } from "solid-js";
import type { Store } from "@mcts/core";
import type { BenchAction, BenchState } from "./index.js";
import type { ExperimentCell } from "./types.js";

const prettyJson = (value: unknown): string => JSON.stringify(value ?? null, null, 2);
const statusLabel = (status: string): string => status.replaceAll("_", " ");
const formatTime = (value: string | null | undefined): string => value ? new Date(value).toLocaleString() : "Not available";

const StatusBadge: Component<{ status: string }> = (props) => <span class={`status-badge badge-${props.status}`}>{statusLabel(props.status)}</span>;

const CellMetrics: Component<{ cell: ExperimentCell }> = (props) => <div class="projects-result-metrics">
  <span><strong>{props.cell.completed_games} / {props.cell.planned_games}</strong><small>completed games</small></span>
  <span><strong>{props.cell.wins} / {props.cell.losses} / {props.cell.draws}</strong><small>W / L / D</small></span>
  <span><strong>{(props.cell.win_rate * 100).toFixed(1)}%</strong><small>win rate</small></span>
  <span><strong>{(props.cell.ci_lower * 100).toFixed(1)}–{(props.cell.ci_upper * 100).toFixed(1)}%</strong><small>95% confidence interval</small></span>
</div>;

export const ExperimentRunDetail: Component<{ store: Store<BenchState, BenchAction> }> = (props) => {
  const state = props.store.getState();
  const dispatch = props.store.dispatch;
  const open = () => state().openRun;
  const selected = () => open()?.cells.find((cell) => cell.cell_id === state().selectedCellId) ?? open()?.cells[0];
  const selectedGames = () => open()?.games.filter((game) => game.cell_id === selected()?.cell_id) ?? [];

  return <Show when={open()?.detail} fallback={<section class="projects-state"><span class="projects-state-title">No experiment run selected</span><span>Open a run from history to inspect its progress and source games.</span></section>}>
    <section class="projects-run-view">
      <header class="projects-page-header projects-run-header">
        <div><p class="projects-eyebrow">Experiment run</p><h1>{open()?.detail?.label ?? "Experiment run"}</h1><code class="projects-run-id">{open()?.detail?.run_id}</code></div>
        <StatusBadge status={open()?.detail?.status ?? "pending"} />
      </header>

      <Show when={selected()} fallback={<section class="projects-panel projects-state"><span class="projects-state-title">Waiting for cell results</span><span>The run has not reported its one-cell result yet.</span></section>}>
        <Show when={(open()?.cells.length ?? 0) > 1}>
          <section class="projects-panel projects-cell-selector" aria-label="Run cells"><For each={open()?.cells ?? []}>{(cell) => <button type="button" classList={{ "projects-cell-selected": cell.cell_id === selected()?.cell_id }} onClick={() => dispatch({ tag: "openCell", cellId: cell.cell_id })}>{cell.variant_label} · {cell.completed_games}/{cell.planned_games}</button>}</For></section>
        </Show>
        <section class="projects-panel" aria-labelledby="result-heading">
          <div class="projects-panel-heading"><div><h2 id="result-heading">{selected()?.variant_label ?? "Variant"} versus {selected()?.baseline_label ?? "Baseline"}</h2><p>{selected()?.game} · started {formatTime(selected()?.started_at)}</p></div><StatusBadge status={selected()?.status ?? "pending"} /></div>
          <button type="button" class="projects-cell-row" onClick={() => dispatch({ tag: "openCell", cellId: selected()!.cell_id })} aria-label={`Open cell ${selected()?.cell_id}: Variant: ${selected()?.completed_games}/${selected()?.planned_games} · ${selected()?.wins}/${selected()?.losses}/${selected()?.draws} · ${(selected()?.win_rate ?? 0) * 100}%`}><span>Variant: {selected()?.completed_games}/{selected()?.planned_games} · {selected()?.wins}/{selected()?.losses}/{selected()?.draws} · {(selected()?.win_rate ?? 0) * 100}%</span><span>Open cell <span aria-hidden="true">→</span></span></button>
          <progress class="projects-result-progress" max={selected()?.planned_games ?? 0} value={selected()?.completed_games ?? 0} aria-label={`${selected()?.completed_games ?? 0} of ${selected()?.planned_games ?? 0} games completed`} />
          <p class="projects-result-summary">Variant: {selected()?.completed_games}/{selected()?.planned_games} · {selected()?.wins}/{selected()?.losses}/{selected()?.draws} · {(selected()?.win_rate ?? 0) * 100}%</p>
          <CellMetrics cell={selected()!} />
        </section>

        <section class="projects-panel" aria-labelledby="cell-inspector-heading">
          <div class="projects-panel-heading"><div><h2 id="cell-inspector-heading">Cell inspector</h2><p>Inspect the exact saved inputs and the result recorded for this cell.</p></div></div>
          <dl class="projects-detail-grid"><div><dt>Game</dt><dd>{selected()?.game}</dd></div><div><dt>Budget</dt><dd>{selected()?.budget.kind === "iterations" ? `${selected()?.budget.value} iterations` : `${selected()?.budget.value} ms per move`}</dd></div><div><dt>Status</dt><dd><StatusBadge status={selected()?.status ?? "pending"} /></dd></div><div><dt>Started</dt><dd>{formatTime(selected()?.started_at)}</dd></div></dl>
          <div class="projects-json-grid"><div><h3>Candidate:</h3><pre><code>{prettyJson(selected()?.candidate_config)}</code></pre></div><div><h3>Baseline:</h3><pre><code>{prettyJson(selected()?.baseline_config)}</code></pre></div></div>
          <div class="projects-metrics-line"><strong>Result metrics</strong><span>{selected()?.wins} wins · {selected()?.losses} losses · {selected()?.draws} draws · {(selected()?.win_rate ?? 0) * 100}% win rate · 95% CI {(selected()?.ci_lower ?? 0) * 100}–{(selected()?.ci_upper ?? 0) * 100}%</span></div>
          <Show when={selected()?.error}><p class="projects-form-error" role="alert">{selected()?.error}</p></Show>
        </section>

        <section class="projects-panel" aria-labelledby="source-games-heading">
          <div class="projects-panel-heading"><div><h2 id="source-games-heading">Source games</h2><p>Each paired game contributing to the selected cell.</p></div></div>
          <Show when={selectedGames().length > 0} fallback={<div class="projects-state"><span class="projects-state-title">No source games reported</span><span>Games will appear here as the run records them.</span></div>}>
            <div class="projects-source-games"><For each={selectedGames()}>{(game) => <div class="projects-source-game"><span class="projects-source-sequence">Game {game.match_seq ?? game.game_seq}</span><span>{game.strategy_a ?? "Unknown"} vs {game.strategy_b ?? "Unknown"}</span><span class="projects-source-outcome">{game.outcome ?? "Pending"}</span><span>Seed {game.seed ?? "—"}</span><code>trace {game.game_seq}</code></div>}</For></div>
          </Show>
        </section>
      </Show>
    </section>
  </Show>;
};
