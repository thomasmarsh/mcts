// RunEvidence — the drill-down view: every candidate and every pair of one
// run, plus the raw manifest / report escape hatch. Clicking a candidate
// opens the shared candidate drawer (`?candidate=`); clicking a pair opens
// the pair inspector (seat-swapped game summaries). No science here — this
// is the "find me this specific pair" screen.

import { createMemo, createSignal, For, Show, type Component } from "solid-js";
import type { Store } from "@mcts/core";
import { peek, isLoading } from "../remote-data.js";
import type { TunerAction, TunerState } from "../tuner-reducer.js";
import type { TunerRoute } from "../tuner-routes.js";
import type { ProjectionCandidate, ProjectionPairRow } from "../tuner-types.js";
import { shortCandidateId } from "../models/verdict-model.js";
import { DataTable } from "../primitives/DataTable.js";
import { JsonDrawer } from "../primitives/JsonDrawer.js";
import { PairInspector } from "./PairInspector.js";

export const RunEvidence: Component<{
  store: Store<TunerState, TunerAction>;
  runId: string;
  navigate: (route: TunerRoute) => void;
}> = (props) => {
  const state = props.store.getState();
  const dispatch = props.store.dispatch;

  const candidates = createMemo(() => peek(state().candidates) ?? []);
  const pairs = createMemo(() => peek(state().pairs) ?? []);
  const detail = createMemo(() => peek(state().projectionDetail));
  const pairsPending = createMemo(() => isLoading(state().pairs) && pairs().length === 0);
  const pairsPage = createMemo(() => state().pairsPage);
  // The `pairs` fetch is a bounded page (`pairsPage`), not the whole run's
  // table, so the true total comes from the run's compute-phase rollup
  // (already fetched for the overview) rather than the fetched page's
  // length -- otherwise the header would silently report a truncated count
  // once a run passes one page.
  const pairsTotal = createMemo(() => {
    const compute = detail()?.compute;
    if (!compute) return pairs().length;
    return compute.reduce((sum, phase) => sum + phase.completed_pairs, 0);
  });
  const pairsRangeEnd = createMemo(() =>
    Math.min(pairsPage().offset + pairsPage().limit, pairsTotal()),
  );

  const cohortOf = createMemo(() => {
    const map = new Map<string, number>();
    for (const c of candidates()) map.set(c.candidate_id, c.cohort_index);
    return map;
  });
  const phases = createMemo(() =>
    Array.from(new Set(pairs().map((p) => p.phase))).sort(),
  );
  const [phase, setPhase] = createSignal<string>("all");
  const filteredPairs = createMemo(() =>
    phase() === "all" ? pairs() : pairs().filter((p) => p.phase === phase()),
  );

  const openCandidate = (candidateId: string): void =>
    props.navigate({ view: "run", runId: props.runId, tab: "evidence", candidate: candidateId });
  const openPair = (pairId: string): void => dispatch({ tag: "selectPair", pairId });

  return (
    <div class="tuner-run-evidence" data-testid="tuner-run-evidence">
      <div class="tuner-run-overview-header">
        <button
          class="tuner-back"
          onClick={() => props.navigate({ view: "run", runId: props.runId, tab: "overview" })}
        >
          ← Overview
        </button>
        <h2>{props.runId} · evidence</h2>
        <button
          onClick={() => dispatch({ tag: "refreshProjection" })}
          disabled={state().refreshing}
        >
          {state().refreshing ? "Refreshing…" : "Refresh"}
        </button>
      </div>

      <section class="tuner-evidence-section">
        <h3>Candidates ({candidates().length})</h3>
        <DataTable<ProjectionCandidate>
          testid="evidence-candidates"
          rows={candidates()}
          rowKey={(c) => c.candidate_id}
          onRowClick={(c) => openCandidate(c.candidate_id)}
          empty="No candidates projected."
          pageSize={50}
          columns={[
            { key: "id", header: "Candidate", render: (c) => shortCandidateId(c.candidate_id) },
            { key: "src", header: "Source", render: (c) => c.source },
            {
              key: "slot",
              header: "Cohort / slot",
              align: "right",
              render: (c) => `${c.cohort_index} / ${c.cohort_slot}`,
            },
            {
              key: "parent",
              header: "Parent",
              render: (c) =>
                c.parent_candidate_id ? shortCandidateId(c.parent_candidate_id) : "—",
            },
            { key: "fp", header: "Fingerprint", render: (c) => c.fingerprint.slice(0, 12) },
          ]}
        />
      </section>

      <section class="tuner-evidence-section">
        <h3>Pairs ({pairsTotal()})</h3>
        <label class="tuner-evidence-filter">
          Phase{" "}
          <select value={phase()} onChange={(e) => setPhase(e.currentTarget.value)}>
            <option value="all">all</option>
            <For each={phases()}>{(p) => <option value={p}>{p}</option>}</For>
          </select>
        </label>
        <Show
          when={!pairsPending()}
          fallback={<p class="tuner-fleet-empty">Loading pairs…</p>}
        >
          <DataTable<ProjectionPairRow>
            testid="evidence-pairs"
            rows={filteredPairs()}
            rowKey={(p) => p.pair_id}
            onRowClick={(p) => openPair(p.pair_id)}
            empty="No pairs projected yet."
            columns={[
              { key: "pair", header: "Pair", render: (p) => p.pair_id.replace(/^pair-/, "").slice(0, 10) },
              { key: "phase", header: "Phase", render: (p) => p.phase },
              { key: "cand", header: "Candidate", render: (p) => shortCandidateId(p.candidate_id) },
              {
                key: "cohort",
                header: "Cohort",
                align: "right",
                render: (p) => cohortOf().get(p.candidate_id) ?? "—",
              },
              { key: "opp", header: "Opponent", render: (p) => p.opponent_id },
              {
                key: "util",
                header: "Utility",
                align: "right",
                render: (p) => p.pair_utility.toFixed(3),
              },
            ]}
          />
          <div class="tuner-table-pager" data-testid="evidence-pairs-pager">
            <button
              data-testid="evidence-pairs-prev"
              disabled={pairsPage().offset === 0}
              onClick={() =>
                dispatch({
                  tag: "pairsPageChanged",
                  offset: Math.max(0, pairsPage().offset - pairsPage().limit),
                })
              }
            >
              ← Prev
            </button>
            <span>
              {pairsTotal() === 0 ? "0–0" : `${pairsPage().offset + 1}–${pairsRangeEnd()}`} of{" "}
              {pairsTotal()}
            </span>
            <button
              data-testid="evidence-pairs-next"
              disabled={pairsRangeEnd() >= pairsTotal()}
              onClick={() =>
                dispatch({
                  tag: "pairsPageChanged",
                  offset: pairsPage().offset + pairsPage().limit,
                })
              }
            >
              Next →
            </button>
          </div>
        </Show>
        <Show when={state().openPairId}>
          {(pairId) => (
            <PairInspector
              store={props.store}
              pairId={pairId()}
              onClose={() => dispatch({ tag: "selectPair", pairId: null })}
              onOpenCandidate={openCandidate}
            />
          )}
        </Show>
      </section>

      <section class="tuner-evidence-section">
        <h3>Raw artifacts</h3>
        <JsonDrawer title="manifest (summary)" value={detail()?.manifest ?? null} testid="raw-manifest" />
        <JsonDrawer title="report.json" value={peek(state().report) ?? null} testid="raw-report" />
      </section>
    </div>
  );
};
