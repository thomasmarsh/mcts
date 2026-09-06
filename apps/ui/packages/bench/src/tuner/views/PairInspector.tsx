// PairInspector — one tuner pair's evidence: the seat-swapped game
// summaries plus a headline W/D/L / ply / compute total. The tuner emits no
// per-ply move traces, so there is no board playback here; if traces are
// added, `PairGameView.hasTrace` flips and a `<BoardViewport>` drops in
// beside the strip (the same branch `SpectatorPanel` already makes).

import { createMemo, Show, type Component } from "solid-js";
import type { Store } from "@mcts/core";
import { peek, isLoading } from "../remote-data.js";
import type { TunerAction, TunerState } from "../tuner-reducer.js";
import { derivePairInspector } from "../models/pair-model.js";
import { GameSummaryStrip } from "../primitives/GameSummaryStrip.js";
import { KpiRow } from "../primitives/KpiRow.js";
import { CandidateChip } from "../primitives/CandidateChip.js";

export const PairInspector: Component<{
  store: Store<TunerState, TunerAction>;
  pairId: string;
  onClose: () => void;
  onOpenCandidate: (candidateId: string) => void;
}> = (props) => {
  const state = props.store.getState();

  const pairRow = createMemo(() =>
    (peek(state().pairs) ?? []).find((p) => p.pair_id === props.pairId),
  );
  const games = createMemo(() => peek(state().pairGames) ?? []);
  const loading = createMemo(() => isLoading(state().pairGames) && games().length === 0);
  const view = createMemo(() => {
    const row = pairRow();
    return row ? derivePairInspector(row, games()) : null;
  });

  return (
    <aside class="tuner-pair-inspector" data-testid="pair-inspector">
      <div class="tuner-candidate-drawer-head">
        <h3>Pair {props.pairId.replace(/^pair-/, "").slice(0, 12)}</h3>
        <button class="tuner-back" onClick={() => props.onClose()}>
          Close
        </button>
      </div>

      <Show
        when={view()}
        fallback={
          <p class="tuner-fleet-empty">
            {pairRow() ? "This pair is not in the loaded page." : "Pair not found."}
          </p>
        }
      >
        {(v) => (
          <>
            <dl class="tuner-candidate-meta">
              <dt>Phase</dt>
              <dd>{v().phase}</dd>
              <dt>Candidate</dt>
              <dd>
                <CandidateChip
                  candidateId={v().candidateId}
                  onClick={() => props.onOpenCandidate(v().candidateId)}
                />
              </dd>
              <dt>Opponent</dt>
              <dd class="tuner-mono">{v().opponentId}</dd>
              <dt>Task</dt>
              <dd class="tuner-mono">{v().taskId.replace(/^task-/, "").slice(0, 12)}</dd>
            </dl>

            <KpiRow
              testid="pair-kpis"
              items={[
                { label: "pair utility", value: v().pairUtility.toFixed(3) },
                { label: "W / D / L", value: `${v().wins} / ${v().draws} / ${v().losses}` },
                { label: "total plies", value: String(v().totalPlies) },
                {
                  label: "cand. iters",
                  value: v().candidateIterations.toLocaleString(),
                  hint: "summed across both seat-swapped games",
                },
              ]}
            />

            <Show when={loading()}>
              <p class="tuner-fleet-empty">Loading game summaries…</p>
            </Show>
            <GameSummaryStrip games={v().games} />
          </>
        )}
      </Show>
    </aside>
  );
};
