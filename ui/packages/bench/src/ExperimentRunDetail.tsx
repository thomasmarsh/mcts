import { createMemo, createSignal, For, Show, type Component, type JSX } from "solid-js";
import { Dynamic } from "solid-js/web";
import type { Store } from "@mcts/core";
import type { BenchAction, BenchState } from "./index.js";
import type { BenchSpectatorProps, ExperimentCell, GameTraceSummary } from "./types.js";
import { buildExperimentMatrix, budgetLabel } from "./experiment-matrix.js";
import { formatInterval, formatObservedResult, formatProgress, formatRate, formatTime, formatWld, statusLabel } from "./result-format.js";

const MAX_VISIBLE_LOG_LINES = 500;

function prettyJson(value: unknown): string {
  return JSON.stringify(value ?? null, null, 2);
}

const StatusBadge: Component<{ status: string }> = (props) => (
  <span class={`status-badge badge-${props.status}`}>
    {statusLabel(props.status)}
  </span>
);

function cellLabel(cell: ExperimentCell, budgetText: string): string {
  const observed = cell.completed_games === 0
    ? "no games yet"
    : `${formatRate(cell.win_rate)}, 95% interval ${formatInterval(cell.ci_lower, cell.ci_upper)}`;
  return `${cell.game}, ${budgetText}, ${cell.variant_label}, ${statusLabel(cell.status)}, ${formatProgress(cell.completed_games, cell.planned_games)} games, W/L/D ${formatWld(cell)}, ${observed}${cell.error ? `, error ${cell.error}` : ""}`;
}

const CellResult: Component<{ cell: ExperimentCell | null; budgetText: string; selected: boolean; onOpen: (cellId: string) => void }> = (props) => (
  <Show when={props.cell} fallback={<span class="projects-matrix-missing">Unavailable</span>}>
    {(cell) => <button
      type="button"
      class="projects-matrix-cell"
      classList={{ "projects-cell-selected": props.selected }}
      aria-pressed={props.selected}
      aria-label={cellLabel(cell(), props.budgetText)}
      onClick={() => props.onOpen(cell().cell_id)}
    >
      <span class="projects-matrix-cell-status"><StatusBadge status={cell().status} /></span>
      <span>{formatProgress(cell().completed_games, cell().planned_games)} games</span>
      <span>W/L/D {formatWld(cell())}</span>
      <span>{formatObservedResult(cell())}</span>
      <Show when={cell().error}><span class="projects-matrix-cell-error">{cell().error}</span></Show>
    </button>}
  </Show>
);

const DetailValue: Component<{ label: string; children: JSX.Element }> = (props) => (
  <div><dt>{props.label}</dt><dd>{props.children}</dd></div>
);

export const ExperimentRunDetail: Component<{
  store: Store<BenchState, BenchAction>;
  Spectator?: Component<BenchSpectatorProps>;
}> = (props) => {
  const state = props.store.getState();
  const dispatch = props.store.dispatch;
  const open = createMemo(() => state().openRun);
  const detail = createMemo(() => open()?.detail ?? null);
  const selected = createMemo(() => {
    const run = open();
    return run?.cells.find((cell) => cell.cell_id === state().selectedCellId) ?? run?.cells[0] ?? null;
  });
  const selectedGames = createMemo(() => {
    const cellId = selected()?.cell_id;
    return cellId ? (open()?.games.filter((game) => game.cell_id === cellId) ?? []) : [];
  });
  const matrix = createMemo(() => {
    const spec = detail()?.experiment_spec;
    return spec ? buildExperimentMatrix(spec, open()?.cells ?? []) : null;
  });
  const [replayGameSeq, setReplayGameSeq] = createSignal<number | null>(null);
  const totalCompleted = createMemo(() => (open()?.cells ?? []).reduce((sum, cell) => sum + cell.completed_games, 0));
  const totalPlanned = createMemo(() => (open()?.cells ?? []).reduce((sum, cell) => sum + cell.planned_games, 0));
  const statusCount = (status: string) => (open()?.cells ?? []).filter((cell) => cell.status === status).length;
  const exportPending = createMemo(() => state().experimentExportStatus === "pending");
  const exportAvailable = createMemo(() => Boolean(detail()?.experiment_spec && (open()?.cells.length ?? 0) > 0));
  const replayGame = createMemo(() => selectedGames().find((game) => game.game_seq === replayGameSeq()) ?? null);

  function openCell(cellId: string): void {
    setReplayGameSeq(null);
    dispatch({ tag: "openCell", cellId });
  }

  function replay(game: GameTraceSummary): void {
    setReplayGameSeq(game.game_seq);
  }

  return <Show when={detail()} fallback={<section class="projects-state"><span class="projects-state-title">No experiment run selected</span><span>Open a run from history to inspect its progress and source games.</span></section>}>
    {(runDetail) => <section class="projects-run-view">
      <header class="projects-page-header projects-run-header">
        <div><p class="projects-eyebrow">Experiment run</p><h1>{runDetail().label ?? "Experiment run"}</h1><code class="projects-run-id">{runDetail().run_id}</code></div>
        <div class="projects-run-actions"><StatusBadge status={runDetail().status} /><Show when={runDetail().status === "running"}><button id="stop-experiment-run-btn" type="button" onClick={() => dispatch({ tag: "stopRun", runId: runDetail().run_id })}>Stop</button></Show></div>
      </header>
      <Show when={state().stopError}><p class="projects-form-error" role="alert">{state().stopError}</p></Show>
      <Show when={state().experimentExportError}><p class="projects-form-error" role="alert">{state().experimentExportError}</p></Show>
      <section class="projects-panel projects-result-metrics" aria-label="Run summary">
        <span><strong>{totalCompleted()} / {totalPlanned()}</strong><small>completed games</small></span>
        <span><strong>{statusCount("pending")}</strong><small>pending cells</small></span>
        <span><strong>{statusCount("running")}</strong><small>running cells</small></span>
        <span><strong>{statusCount("completed")}</strong><small>completed cells</small></span>
        <span><strong>{statusCount("failed")}</strong><small>failed cells</small></span>
        <span><strong>{statusCount("cancelled")}</strong><small>cancelled cells</small></span>
      </section>

      <div class="projects-action-row projects-run-export-actions">
        <span>Export current run snapshot:</span>
        <button type="button" onClick={() => dispatch({ tag: "exportExperimentRun", format: "json" })} disabled={exportPending() || !exportAvailable()}>JSON</button>
        <button type="button" onClick={() => dispatch({ tag: "exportExperimentRun", format: "csv" })} disabled={exportPending() || !exportAvailable()}>CSV</button>
        <Show when={exportPending()}><span role="status">Preparing download…</span></Show>
      </div>

      <Show when={matrix()} fallback={<section class="projects-panel projects-state"><span class="projects-state-title">Waiting for experiment snapshot</span><span>The immutable experiment definition has not arrived yet.</span></section>}>
        {(model) => <>
          <p class="projects-matrix-context">Candidates versus {model().sections[0]?.columns.length ? runDetail().experiment_spec!.baseline.label : "the saved baseline"}. Select a result to inspect its exact inputs and source games.</p>
          <Show when={model().warnings.length > 0}>
            <div class="projects-matrix-warning" role="status"><strong>Data warning:</strong> {model().warnings.map((warning) => `${warning.kind} cell ${warning.cellId} at ${warning.coordinate.game} / ${budgetLabel(warning.coordinate.budget)} / ${warning.coordinate.variantId}`).join("; ")}</div>
          </Show>
          <For each={model().sections}>{(section, sectionIndex) => <section class="projects-panel projects-matrix-section" aria-labelledby={`matrix-heading-${sectionIndex()}`}>
            <div class="projects-panel-heading"><div><h2 id={`matrix-heading-${sectionIndex()}`}>{budgetLabel(section.budget)}</h2><p>Games are rows; variants are columns. Baseline: {runDetail().experiment_spec!.baseline.label}</p></div></div>
            <div class="projects-matrix-scroll">
              <table class="projects-matrix-table">
                <caption>{budgetLabel(section.budget)} results by game and variant</caption>
                <thead><tr><th scope="col">Game</th><For each={section.columns}>{(variant) => <th scope="col">{variant.label}</th>}</For></tr></thead>
                <tbody><For each={section.rows}>{(row) => <tr><th scope="row">{row.game.game}</th><For each={row.cells}>{(entry) => <td><CellResult cell={entry.cell} budgetText={budgetLabel(entry.coordinate.budget)} selected={entry.cell?.cell_id === state().selectedCellId} onOpen={openCell} /></td>}</For></tr>}</For></tbody>
              </table>
            </div>
          </section>}</For>
        </>}
      </Show>

      <Show when={selected()} fallback={<section class="projects-panel projects-state"><span class="projects-state-title">Waiting for cell results</span><span>The run has not reported a cell result yet.</span></section>}>
        {(cell) => <>
          <section class="projects-panel" aria-labelledby="cell-inspector-heading">
            <div class="projects-panel-heading"><div><h2 id="cell-inspector-heading">Cell inspector</h2><p>Every value below comes from the saved cell snapshot or its recorded result.</p></div><StatusBadge status={cell().status} /></div>
            <dl class="projects-detail-grid">
              <DetailValue label="Cell ID"><code>{cell().cell_id}</code></DetailValue>
              <DetailValue label="Cell seed">{cell().cell_seed ?? "Not recorded"}</DetailValue>
              <DetailValue label="Game">{cell().game}</DetailValue>
              <DetailValue label="Budget kind">{cell().budget.kind}</DetailValue>
              <DetailValue label="Budget value">{cell().budget.value}</DetailValue>
              <DetailValue label="Paired rounds">{cell().rounds}</DetailValue>
              <DetailValue label="Planned games">{cell().planned_games}</DetailValue>
              <DetailValue label="Completed games">{cell().completed_games}</DetailValue>
              <DetailValue label="Status"><StatusBadge status={cell().status} /></DetailValue>
              <DetailValue label="Started">{formatTime(cell().started_at)}</DetailValue>
              <DetailValue label="Ended">{formatTime(cell().ended_at)}</DetailValue>
              <DetailValue label="Error">{cell().error ?? "Not recorded"}</DetailValue>
            </dl>
            <div class="projects-json-grid">
              <div><h3>Game configuration</h3><pre><code>{prettyJson(cell().game_config)}</code></pre></div>
              <div><h3>Candidate configuration</h3><pre><code>{prettyJson(cell().candidate_config)}</code></pre></div>
              <div><h3>Baseline configuration</h3><pre><code>{prettyJson(cell().baseline_config)}</code></pre></div>
            </div>
            <dl class="projects-detail-grid projects-detail-grid-results">
              <DetailValue label="Candidate ID">{cell().variant_id}</DetailValue>
              <DetailValue label="Candidate label">{cell().variant_label}</DetailValue>
              <DetailValue label="Baseline ID">{cell().baseline_id}</DetailValue>
              <DetailValue label="Baseline label">{cell().baseline_label}</DetailValue>
              <DetailValue label="W / L / D">{formatWld(cell())}</DetailValue>
              <DetailValue label="Draw-as-half win rate">{formatObservedResult(cell())}</DetailValue>
              <DetailValue label="95% interval">{cell().completed_games === 0 ? "No games yet" : formatInterval(cell().ci_lower, cell().ci_upper)}</DetailValue>
              <DetailValue label="Progress">{formatProgress(cell().completed_games, cell().planned_games)}</DetailValue>
            </dl>
          </section>

          <details class="projects-panel projects-log-tail">
            <summary>Raw log tail ({Math.min(open()?.tail.lines.length ?? 0, MAX_VISIBLE_LOG_LINES)} lines)</summary>
            <p>Polling: {open()?.tail.active ? "live" : "complete"}</p>
            <Show when={open()?.tail.error}><p class="projects-form-error" role="alert">{open()?.tail.error}</p></Show>
            <pre><code>{(open()?.tail.lines ?? []).slice(-MAX_VISIBLE_LOG_LINES).join("\n") || "No log lines recorded."}</code></pre>
          </details>

          <section class="projects-panel" aria-labelledby="source-games-heading">
            <div class="projects-panel-heading"><div><h2 id="source-games-heading">Source games</h2><p>Traced games for {cell().cell_id}; metrics are recorded per game when available.</p></div></div>
            <Show when={selectedGames().length > 0} fallback={<div class="projects-state"><span class="projects-state-title">No source games reported</span><span>Games will appear here as the run records them.</span></div>}>
              <div class="projects-source-games"><For each={selectedGames()}>{(game) => <Show when={props.Spectator} fallback={<div class="projects-source-game"><SourceGame game={game} /></div>}>
                <button type="button" class="projects-source-game" aria-label={`Replay game ${game.match_seq ?? game.game_seq} (trace ${game.game_seq})`} onClick={() => replay(game)}><SourceGame game={game} /></button>
              </Show>}</For></div>
            </Show>
          </section>
          <Show when={props.Spectator && replayGame()}>
            <section class="projects-panel" aria-label="Game replay"><Dynamic component={props.Spectator} runId={runDetail().run_id} game={cell().game} kind={runDetail().kind} live={runDetail().status === "running"} cellId={cell().cell_id} initialGameSeq={replayGame()!.game_seq} /></section>
          </Show>
        </>}
      </Show>
    </section>}
  </Show>;
};

const SourceGame: Component<{ game: GameTraceSummary }> = (props) => <>
  <span class="projects-source-sequence">Match {props.game.match_seq ?? "—"} · Trace {props.game.game_seq}</span>
  <span>{props.game.strategy_a ?? "Unknown"} vs {props.game.strategy_b ?? "Unknown"}</span>
  <span class="projects-source-outcome">{props.game.outcome ?? "Pending"}{props.game.winner ? ` · winner ${props.game.winner}` : ""}</span>
  <span>Seed {props.game.seed ?? "Not recorded"}</span>
  <span>{props.game.ply_count} plies · {formatTime(props.game.started_at)} – {formatTime(props.game.ended_at)}</span>
  <Show when={props.game.metrics !== null && props.game.metrics !== undefined}><span class="projects-source-metrics"><code>{prettyJson(props.game.metrics)}</code></span></Show>
</>;
