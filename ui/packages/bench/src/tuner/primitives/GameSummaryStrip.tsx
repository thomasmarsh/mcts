// GameSummaryStrip — the two seat-swapped game summaries of one tuner pair.
// Summary-only (outcome, plies, elapsed, iteration totals, seat held); the
// v4 tuner records no per-ply traces, so there is no board here. Pure
// layout — the caller derives the rows with `derivePairInspector`.

import { For, Show, type Component } from "solid-js";
import type { PairGameView } from "../models/pair-model.js";

const ms = (n: number): string => (n >= 1000 ? `${(n / 1000).toFixed(1)} s` : `${n} ms`);

export const GameSummaryStrip: Component<{ games: PairGameView[]; testid?: string }> = (props) => (
  <div class="tuner-game-strip" data-testid={props.testid ?? "game-summary-strip"}>
    <Show
      when={props.games.length > 0}
      fallback={<p class="tuner-fleet-empty">No game summaries recorded for this pair.</p>}
    >
      <For each={props.games}>
        {(g) => (
          <div class="tuner-game-card" classList={{ [`tuner-game-${g.result}`]: true }}>
            <div class="tuner-game-card-head">
              <span class="tuner-game-result">{g.resultLabel}</span>
              <span class="tuner-game-side">candidate {g.side}</span>
            </div>
            <dl class="tuner-game-card-meta">
              <dt>Plies</dt>
              <dd>{g.plies}</dd>
              <dt>Elapsed</dt>
              <dd>{ms(g.elapsedMs)}</dd>
              <dt>Cand. iters</dt>
              <dd>{g.candidateIterations.toLocaleString()}</dd>
              <dt>Opp. iters</dt>
              <dd>{g.opponentIterations.toLocaleString()}</dd>
            </dl>
          </div>
        )}
      </For>
    </Show>
  </div>
);
