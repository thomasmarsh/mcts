import { For, Show, type Component } from "solid-js";
import type { Store } from "@mcts/core";
import type { BenchAction, BenchState } from "./index.js";

export const ExperimentRunDetail: Component<{ store: Store<BenchState, BenchAction> }> = (props) => {
  const state = props.store.getState(); const dispatch = props.store.dispatch;
  const open = () => state().openRun;
  const selected = () => open()?.cells.find((cell) => cell.cell_id === state().selectedCellId) ?? open()?.cells[0];
  return <Show when={open()?.detail}>
    <section class="bench-experiment-run">
      <h2>Experiment run</h2><p>{open()?.detail?.status} · {open()?.detail?.run_id}</p>
      <For each={open()?.cells ?? []}>{(cell) => <button class="cell-row" onClick={() => dispatch({ tag: "openCell", cellId: cell.cell_id })}>{cell.variant_label}: {cell.completed_games}/{cell.planned_games} · {cell.wins}/{cell.losses}/{cell.draws} · {(cell.win_rate * 100).toFixed(1)}%</button>}</For>
      <Show when={selected()}>
        <div class="cell-inspector"><h3>Cell {selected()?.cell_id}</h3><p>Status: {selected()?.status}</p><p>Budget: {JSON.stringify(selected()?.budget)}</p><p>Candidate: <code>{JSON.stringify(selected()?.candidate_config)}</code></p><p>Baseline: <code>{JSON.stringify(selected()?.baseline_config)}</code></p><p>Metrics: {selected()?.wins} W / {selected()?.losses} L / {selected()?.draws} D, CI {(selected()?.ci_lower ?? 0) * 100}–{(selected()?.ci_upper ?? 1) * 100}%</p><Show when={selected()?.error}><p class="error">{selected()?.error}</p></Show></div>
        <div class="cell-games"><h4>Source games</h4><For each={open()?.games.filter((game) => game.cell_id === selected()?.cell_id)}>{(game) => <p>Game {game.match_seq ?? game.game_seq}: {game.strategy_a} vs {game.strategy_b} · {game.outcome} · trace {game.game_seq}</p>}</For></div>
      </Show>
    </section>
  </Show>;
};
