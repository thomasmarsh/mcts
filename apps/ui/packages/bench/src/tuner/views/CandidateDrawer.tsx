// CandidateDrawer — a right-hand drawer opened by `?candidate=<cid>` over a
// run view. Shows the candidate's config against the schema default, its
// validation interval when it reached validation, its lineage, and a
// copy-preset button. Pairs / per-prefix observation forests arrive in
// later evidence slices.

import { createMemo, For, Show, type Component } from "solid-js";
import type { Store } from "@mcts/core";
import { peek } from "../remote-data.js";
import type { TunerAction, TunerState } from "../tuner-reducer.js";
import { schemaDefaults } from "../models/config-diff-model.js";
import { deriveVerdict } from "../models/verdict-model.js";
import { deriveContenders, findContender, formatWDL } from "../models/contender-model.js";
import { ConfigDiff } from "../primitives/ConfigDiff.js";
import { CopyPresetButton } from "../primitives/CopyPresetButton.js";
import { IntervalBar } from "../primitives/IntervalBar.js";

export const CandidateDrawer: Component<{
  store: Store<TunerState, TunerAction>;
  candidateId: string;
  onClose: () => void;
}> = (props) => {
  const state = props.store.getState();

  const gameKind = createMemo(() => peek(state().projectionDetail)?.manifest?.game_kind ?? null);
  const baseConfig = createMemo(() => {
    const kind = gameKind();
    const info = (peek(state().tunableGames) ?? []).find((k) => k.game === kind);
    return info ? schemaDefaults(info.tuner.parameters) : {};
  });
  const candidate = createMemo(() =>
    (peek(state().candidates) ?? []).find((c) => c.candidate_id === props.candidateId),
  );
  const verdict = createMemo(() =>
    deriveVerdict({
      validation: peek(state().validation),
      candidates: peek(state().candidates),
      report: peek(state().report),
    }),
  );
  const validationRow = createMemo(() =>
    verdict().ranked.find((r) => r.candidateId === props.candidateId),
  );
  const contender = createMemo(() =>
    findContender(
      deriveContenders(peek(state().contenders) ?? [], peek(state().candidates)),
      props.candidateId,
    ),
  );

  return (
    <aside class="tuner-candidate-drawer" data-testid="candidate-drawer">
      <div class="tuner-candidate-drawer-head">
        <h3>{props.candidateId.replace(/^candidate-/, "").slice(0, 16)}</h3>
        <button class="tuner-back" onClick={() => props.onClose()}>
          Close
        </button>
      </div>

      <Show
        when={candidate()}
        fallback={<p class="tuner-fleet-empty">This candidate is not in the projection.</p>}
      >
        {(c) => (
          <>
            <dl class="tuner-candidate-meta">
              <dt>Source</dt>
              <dd>{c().source}</dd>
              <dt>Cohort</dt>
              <dd>
                {c().cohort_index} / slot {c().cohort_slot}
              </dd>
              <Show when={c().parent_candidate_id}>
                <dt>Parent</dt>
                <dd>
                  {c()
                    .parent_candidate_id!.replace(/^candidate-/, "")
                    .slice(0, 12)}
                </dd>
              </Show>
              <dt>Fingerprint</dt>
              <dd class="tuner-mono">{c().fingerprint}</dd>
            </dl>

            <Show when={validationRow()}>
              {(row) => (
                <div class="tuner-candidate-validation">
                  <h4>Validation (rank #{row().rank})</h4>
                  <IntervalBar
                    mean={row().estimate}
                    lower={row().lower}
                    upper={row().upper}
                    domain={verdict().domain}
                    reference={0}
                  />
                  <p>
                    {row().wins}W / {row().draws}D / {row().losses}L
                  </p>
                </div>
              )}
            </Show>

            <Show when={contender() && contender()!.games > 0}>
              <div class="tuner-candidate-record" data-testid="candidate-record">
                <h4>Record · {formatWDL(contender()!.overall)} (W / D / L)</h4>
                <table class="tuner-table">
                  <thead>
                    <tr>
                      <th>Opponent</th>
                      <th class="tuner-td-right">W / D / L</th>
                    </tr>
                  </thead>
                  <tbody>
                    <For each={contender()!.byOpponent}>
                      {(o) => (
                        <tr>
                          <td>{o.opponentId}</td>
                          <td class="tuner-td-right">{formatWDL(o)}</td>
                        </tr>
                      )}
                    </For>
                  </tbody>
                </table>
                <Show when={contender()!.byPhase.length > 1}>
                  <p class="tuner-candidate-record-phases">
                    <For each={contender()!.byPhase}>
                      {(p) => (
                        <span>
                          {p.phase}: {formatWDL(p)}
                          {"  "}
                        </span>
                      )}
                    </For>
                  </p>
                </Show>
              </div>
            </Show>

            <h4>Config vs default</h4>
            <ConfigDiff base={baseConfig()} candidate={c().canonical_config} />
            <CopyPresetButton
              candidateId={c().candidate_id}
              gameKind={gameKind()}
              config={c().canonical_config}
            />
          </>
        )}
      </Show>
    </aside>
  );
};
