import { afterEach, describe, expect, it, vi } from "vitest";
import { cleanup, fireEvent, render, screen } from "@solidjs/testing-library";
import { createStore, Effect } from "@mcts/core";
import { benchReducer, initialBenchState, type BenchEnv, type TuningTrialDetail, type TuningTrialPage, type TuningTrialPageQuery, type TuningTrialSummary } from "@mcts/bench";
import { TuningSessionWorkbench } from "../packages/bench/src/tuning/TuningSessionWorkbench.js";

const counts = { total: 6, queued: 1, running: 1, terminal: 4, completed: 1, failed: 1, pruned: 1, cancelled: 1 };
const capabilities = { has_lifecycle: true, has_pairs: true, has_renderer_trace: true, has_search_reports: true, has_trial_reports: true };
const session = { session_id: "session-trials", game: "nim", label: "Trials", status: "completed", target_trial_count: 6, counts, created_at: "2026-08-23T00:00:00Z", last_activity_at: "2026-08-23T00:01:00Z", attempts: [], capabilities };
const overview = { schema_version: 1 as const, policy: null, objective: { metric: "score", direction: "max", complete_trials_only: true }, cursor: { session_sequence: 1 }, coverage: { trials: counts, reports: 0, pairs: { total: 0, running: 0, complete: 0, failed: 0, unmatched_pool_revisions: 0 }, points: { total: 0, returned: 0, sampled: false } }, bracket_resources: [], decision_groups: [], points: [], best: null, pool_revisions: [] };

function row(number: number, state = "complete", has_detail = true): TuningTrialSummary {
  return { trial_id: `trial-${number}`, trial_number: number, attempt_id: "attempt-1", state, reason: state === "complete" ? "max_pairs" : state, rating: state === "complete" ? { mu: 25, sigma: 1 } : null, score: state === "complete" ? 22 : null, family: number === 6 ? null : "ucb1", config_summary: null, bracket_id: number === 6 ? null : "b1", resource: number === 6 ? null : 4, pair_count: 2, wins: 2, losses: 1, draws: 1, elapsed_ms: 30, search_iterations_total: 40, search_move_time_ms: 5, has_detail };
}

const candidate = { trial_id: "trial-1", trial_number: 1, attempt_id: "attempt-1", state: "complete", config: { family: "ucb1", c: 1.4 }, score: 22, rating: { mu: 25, sigma: 1 }, reason: "max_pairs", failure: null, reports: [{ completed_pairs: 2, rating: { mu: 25, sigma: 1 }, score: 22, score_formula_version: 1, conservative_k: 3, decision: { outcome: "complete", reason: "max_pairs", pruning_exempt: false, bracket_id: "b1", rung_resource: 4 }, reported_at: "2026-08-23T00:01:00Z" }], pairs: [{ pair_id: "pair-1", pair_index: 0, state: "complete", seed: 7, round: 1, opponent: { anchor_id: "anchor-1", label: "Strong", config: { family: "rave", c: 2 }, mu: 24, sigma: 1, provenance: "pool" }, pool_snapshot_fingerprint: "pool-fp", pool_revision: { pool_snapshot_fingerprint: "pool-fp", display_ordinal: 2, observed_at: "2026-08-23T00:00:00Z", pair_count: 1, anchors: [{ anchor_ordinal: 0, anchor_id: "anchor-1", config: { family: "rave", c: 2 }, rating: { mu: 24, sigma: 1 }, provenance: "pool", insertion_reason: "seed", source_trial_id: null }] }, rating_before: { mu: 23, sigma: 2 }, rating_after: { mu: 25, sigma: 1 }, score: 22, failure: null, games: [
  { game_id: "game-1", candidate_side: "first", outcome: "candidate_win", seed: 7, round: 1, plies: 20, elapsed_ms: 11, candidate: { iterations_total: 20, iterations_first_half: 10, move_time_ms: 3 }, baseline: { iterations_total: 20, iterations_first_half: 10, move_time_ms: 3 }, replay: { run_id: "run-1", game_seq: 11, has_renderer_trace: true, has_search_reports: true } },
  { game_id: "game-2", candidate_side: "second", outcome: "candidate_loss", seed: 8, round: 1, plies: 21, elapsed_ms: 12, candidate: { iterations_total: 20, iterations_first_half: 10, move_time_ms: 3 }, baseline: { iterations_total: 20, iterations_first_half: 10, move_time_ms: 3 }, replay: null },
] }] } as const;

function page(rows: TuningTrialSummary[], next_cursor: string | null = null, total_count = rows.length, limit = 50): TuningTrialPage {
  return { schema_version: 1, trials: rows, total_count, limit, next_cursor, cursor: { session_sequence: 1 } };
}

function setup(getPage: (query: TuningTrialPageQuery | undefined) => TuningTrialPage, getDetail: () => TuningTrialDetail = () => ({ schema_version: 1, trial: candidate, cursor: { session_sequence: 1 } })): { queries: (TuningTrialPageQuery | undefined)[]; detailCalls: string[] } {
  const queries: (TuningTrialPageQuery | undefined)[] = [];
  const detailCalls: string[] = [];
  const env = {
    listTuningSessions: () => Effect.send({ schema_version: 1 as const, sessions: [session] }),
    getTuningSession: () => Effect.send({ schema_version: 1 as const, policy: null, summary: { session_id: session.session_id, status: session.status, target_trial_count: session.target_trial_count, counts }, attempts: [], trials: [], manifest: {}, fingerprint: null, capabilities, cursor: { session_sequence: 1 } }),
    getTuningAnalysisOverview: () => Effect.send(overview),
    getTuningTrialPage: (_id: string, query: TuningTrialPageQuery | undefined) => { queries.push(query); return Effect.send(getPage(query)); },
    getTuningTrialDetail: (_id: string, trialId: string) => { detailCalls.push(trialId); return Effect.send(getDetail()); },
  } as unknown as BenchEnv;
  const store = createStore(initialBenchState(), benchReducer, env);
  render(() => <TuningSessionWorkbench store={store} />);
  store.dispatch({ tag: "tuningNavigation", action: { tag: "listRequest" } });
  store.dispatch({ tag: "tuningNavigation", action: { tag: "selectSession", sessionId: session.session_id } });
  store.dispatch({ tag: "tuningNavigation", action: { tag: "setAnalysisTab", tab: "trials" } });
  return { queries, detailCalls };
}

afterEach(cleanup);

describe("paged tuning Trials view", () => {
  it("queries 50-row pages, sends filters and sorting to the server, and preserves no initial row selection", async () => {
    const { queries } = setup((query) => page(query?.cursor ? [row(2)] : [row(1), row(2)], query?.cursor ? null : "next", 83));
    await screen.findByText("83 results · page 1 · 2 rendered");
    expect(queries[0]).toMatchObject({ limit: 50, sort: "trial", direction: "desc", cursor: null });
    expect(screen.queryByRole("row", { selected: true })).not.toBeInTheDocument();
    fireEvent.input(screen.getByLabelText("Search trials"), { target: { value: "ucb" } });
    await vi.waitFor(() => expect(queries.at(-1)).toMatchObject({ q: "ucb", sort: "trial", direction: "desc" }));
    fireEvent.change(screen.getByLabelText("Sort trials"), { target: { value: "score" } });
    await vi.waitFor(() => expect(queries.at(-1)).toMatchObject({ q: "ucb", sort: "score", direction: "desc" }));
    fireEvent.change(screen.getByLabelText("Sort direction"), { target: { value: "asc" } });
    await vi.waitFor(() => expect(queries.at(-1)).toMatchObject({ q: "ucb", sort: "score", direction: "asc", limit: 50 }));
    fireEvent.click(screen.getByRole("button", { name: "Next page" }));
    await vi.waitFor(() => expect(queries.at(-1)).toMatchObject({ cursor: "next" }));
    expect(screen.getByText("83 results · showing 1 on page 2 (limit 50)")).toBeInTheDocument();
  });

  it("loads exactly one detail when expanded, retains selection across refreshes, and renders immutable pair evidence", async () => {
    const { detailCalls } = setup(() => page([row(1), row(6, "cancelled", false)], null, 2));
    await screen.findByRole("button", { name: "Expand trial 1" });
    fireEvent.click(screen.getByRole("button", { name: "Expand trial 1" }));
    await screen.findByText("Candidate configuration");
    expect(detailCalls).toEqual(["trial-1"]);
    expect(screen.getByText(/Pool revision/)).toBeInTheDocument();
    expect(screen.getAllByText("Replay game")).toHaveLength(1);
    expect(screen.getByText(/Not recorded — this game has no replay reference/)).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "Collapse trial 1" }));
    await screen.findByRole("button", { name: "Expand trial 1" });
    fireEvent.click(screen.getByRole("button", { name: "Expand trial 1" }));
    expect(detailCalls).toEqual(["trial-1"]);
    expect(screen.getByRole("button", { name: "Expand trial 6" })).toBeDisabled();
    expect(screen.getByText("Not recorded", { selector: "td" })).toBeInTheDocument();
  });

  it("copies the exact preset text and bounds a requested page to 200 rows", async () => {
    const { queries } = setup((query) => page([row(1)], null, 1, query?.limit ?? 50));
    await screen.findByRole("button", { name: "Expand trial 1" });
    fireEvent.change(screen.getByLabelText("Rows per page"), { target: { value: "200" } });
    await vi.waitFor(() => expect(queries.at(-1)?.limit).toBe(200));
    fireEvent.click(screen.getByRole("button", { name: "Expand trial 1" }));
    await screen.findByRole("button", { name: "Copy candidate preset" });
    const write = vi.spyOn(navigator.clipboard, "writeText").mockResolvedValue(undefined);
    fireEvent.click(screen.getByRole("button", { name: "Copy candidate preset" }));
    await vi.waitFor(() => expect(write).toHaveBeenCalledWith(`{
    "id": "candidate-trial_x2d_1",
    "label": "Tuned candidate",
    "description": "Candidate snapshot from trial 1 (trial-1).",
    "params": {
        "c": 1.4,
        "family": "ucb1"
    },
    "max_iterations": 10000,
    "threads": 1,
    "use_transpositions": false
}`));
    expect(await screen.findByText("Preset copied to clipboard.")).toBeInTheDocument();
  });
});
