import { afterEach, describe, expect, it, vi } from "vitest";
import { cleanup, fireEvent, render, screen } from "@solidjs/testing-library";
import { onMount } from "solid-js";
import { createStore, Effect, type Store } from "@mcts/core";
import { benchReducer, initialBenchState, type BenchAction, type BenchEnv, type BenchSpectatorProps, type BenchState, type RunSummary, type TuningSessionDetail, type TuningSessionListItem, type TuningSessionsResponse } from "@mcts/bench";
import { RunList } from "../packages/bench/src/RunList.js";
import { TuningSessionWorkbench } from "../packages/bench/src/tuning/TuningSessionWorkbench.js";
import { createMockBenchEnv } from "./fixtures/fake-bench.js";

const attemptA = { attempt_id: "attempt-a", bench_run_id: "physical-a", status: "completed", started_at: "2026-08-23T12:00:00Z", ended_at: "2026-08-23T12:10:00Z", failure: null };
const attemptB = { attempt_id: "attempt-b", bench_run_id: "physical-b", status: "failed", started_at: "2026-08-23T12:11:00Z", ended_at: "2026-08-23T12:12:00Z", failure: "worker stopped" };
const counts = { total: 3, queued: 0, running: 0, terminal: 3, completed: 2, failed: 1, pruned: 0, cancelled: 0 };
const session: TuningSessionListItem = {
  session_id: "session-a", game: "nim", label: "Observable tuning", status: "idle", target_trial_count: 3,
  counts, created_at: "2026-08-23T12:00:00Z", last_activity_at: "2026-08-23T12:12:00Z", attempts: [attemptA, attemptB],
  capabilities: { has_lifecycle: true, has_pairs: true, has_renderer_trace: true, has_search_reports: true, has_trial_reports: true },
};
const sessions: TuningSessionsResponse = { schema_version: 1, sessions: [session] };
const run = (run_id: string): RunSummary => ({ run_id, kind: "tuner", game: "nim", project_id: null, experiment_id: null, label: null, git_sha: "abc", git_dirty: false, host: "test", pid: null, started_at: "2026-08-23T12:00:00Z", ended_at: "2026-08-23T12:10:00Z", status: "completed", match_count: 0, trial_count: 3 });
const runs = [run("physical-a"), run("physical-b"), { ...run("legacy-tuner"), label: "old tuner" }];
const metrics = { iterations_total: 4, iterations_first_half: 2, move_time_ms: 3 };
const game = (id: string, trace_game_seq: number | null, candidate_side: "first" | "second" = "first") => ({ game_id: id, candidate_side, outcome: "candidate_win", seed: 7, round: 1, trace_game_seq, plies: 12, elapsed_ms: 30, candidate: metrics, baseline: metrics });
const pair = (id: string, status: string, games: ReturnType<typeof game>[]) => ({ pair_id: id, pair_index: 0, status, seed: 7, round: 1, opponent: { anchor_id: "anchor", config: {}, mu: 25, sigma: 1, label: "Anchor", provenance: null }, pool_snapshot_fingerprint: "pool", rating_before: { mu: 24, sigma: 2 }, rating_after: status === "complete" ? { mu: 25, sigma: 1 } : null, score: status === "complete" ? 1 : null, failure: status === "failed" ? "interrupted" : null, games });
function detail(extraPair = false): TuningSessionDetail {
  const trials = [
    { trial_id: "trial-a", trial_number: 1, attempt_id: "attempt-a", status: "complete", config: { c: 1 }, score: 1, mu: 25, sigma: 1, stop_reason: "max_pairs", failure: null, pairs: [pair("pair-two", "complete", [game("game-1", 41), game("game-2", 42, "second")]) ], reports: [
      { completed_pairs: 2, rating: { mu: 24, sigma: 2 }, score: 18, score_formula_version: 1, conservative_k: 3, decision: { outcome: "continue", reason: "startup_exempt", pruning_exempt: true, bracket_id: null, rung_resource: null }, reported_at: "2026-08-23T12:02:00Z" },
      { completed_pairs: 4, rating: { mu: 25, sigma: 1 }, score: 22, score_formula_version: 1, conservative_k: 3, decision: { outcome: "complete", reason: "max_pairs", pruning_exempt: false, bracket_id: "bracket-a", rung_resource: 4 }, reported_at: "2026-08-23T12:04:00Z" },
    ] },
    { trial_id: "trial-b", trial_number: 2, attempt_id: "attempt-b", status: "failed", config: { c: 2 }, score: null, mu: null, sigma: null, stop_reason: "hyperband_prune", failure: "worker stopped", pairs: [pair("pair-one", "failed", [game("game-3", null)])], reports: [
      { completed_pairs: 2, rating: { mu: 24, sigma: 2 }, score: 18, score_formula_version: 1, conservative_k: 3, decision: { outcome: "prune", reason: "hyperband_prune", pruning_exempt: false, bracket_id: null, rung_resource: null }, reported_at: "2026-08-23T12:03:00Z" },
    ] },
    { trial_id: "trial-c", trial_number: 3, attempt_id: "attempt-b", status: "failed", config: { c: 3 }, score: null, mu: null, sigma: null, stop_reason: null, failure: "cancelled", pairs: [pair("pair-zero", "failed", [])], reports: [] },
  ];
  if (extraPair) trials[0] = { ...trials[0]!, pairs: [...trials[0]!.pairs, pair("pair-new", "failed", [])] };
  return { schema_version: 1, policy: { resource: { min_pairs: 2, max_pairs: 6 }, rating: { model: "ThurstoneMostellerPart", score: "mu_minus_k_sigma", sigma_stop: 2, conservative_k: 3 }, sampler: { kind: "tpe", seed: 4, deterministic: true, startup_trials: 3 }, pruning: { enabled: true, kind: "hyperband", reduction_factor: 3, startup_terminal_trials: 5 } }, summary: { session_id: "session-a", status: "idle", target_trial_count: 3, counts }, attempts: [attemptA, attemptB], trials, manifest: { game: "nim" }, fingerprint: "fp", capabilities: session.capabilities, cursor: { session_sequence: extraPair ? 2 : 1 } } as TuningSessionDetail;
}

function setup(detailResponse: TuningSessionDetail | (() => TuningSessionDetail) = detail()): { store: Store<BenchState, BenchAction>; env: BenchEnv } {
  const env = createMockBenchEnv({ listRuns: () => Effect.send(runs), listTuningSessions: () => Effect.send(sessions), getTuningSession: () => Effect.send(typeof detailResponse === "function" ? detailResponse() : detailResponse) });
  const store = createStore<BenchState, BenchAction>(initialBenchState(), benchReducer, env);
  store.dispatch({ tag: "runs", action: { tag: "request" } });
  store.dispatch({ tag: "tuningNavigation", action: { tag: "listRequest" } });
  return { store, env };
}

afterEach(cleanup);

describe("observable tuning session workbench", () => {
  it("deduplicates associated runs, retains legacy runs, and exposes exact evidence counts", async () => {
    const { store } = setup();
    render(() => <><RunList store={store} onNewRun={() => undefined} /><TuningSessionWorkbench store={store} /></>);
    expect(await screen.findByText("Observable tuning")).toBeInTheDocument();
    expect(screen.getByText("Legacy tuner run")).toBeInTheDocument();
    expect(screen.queryByText("physical-a")).not.toBeInTheDocument();
    expect(screen.queryByText("physical-b")).not.toBeInTheDocument();
    expect(screen.queryByText("0 matches")).not.toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: /Observable tuning 3 \/ 3 terminal trials/ }));
    expect(await screen.findByText("queued 0 · running 0 · complete 2 · failed 1 · pruned 0 · cancelled 0")).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: /Expand attempts/ }));
    expect(screen.getByRole("button", { name: /Attempt attempt-a/ })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /Attempt attempt-b/ })).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "Expand attempt attempt-a" }));
    await vi.waitFor(() => expect(screen.getByRole("button", { name: "Expand trial 1" })).toBeInTheDocument());
    fireEvent.click(screen.getByRole("button", { name: "Expand trial 1" }));
    await vi.waitFor(() => expect(screen.getByRole("button", { name: "Expand pair 1" })).toBeInTheDocument());
    fireEvent.click(screen.getByRole("button", { name: "Expand pair 1" }));
    expect(screen.getByText(/2 of 2 games/)).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "Expand attempt attempt-b" }));
    await vi.waitFor(() => expect(screen.getByRole("button", { name: "Expand trial 2" })).toBeInTheDocument());
    fireEvent.click(screen.getByRole("button", { name: "Expand trial 2" }));
    await vi.waitFor(() => expect(screen.getByRole("button", { name: "Expand pair 1" })).toBeInTheDocument());
    fireEvent.click(screen.getAllByRole("button", { name: "Expand pair 1" }).at(-1)!);
    expect(screen.getByText(/failed after 1 of 2 games/)).toBeInTheDocument();
    await vi.waitFor(() => expect(screen.getByRole("button", { name: "Expand trial 3" })).toBeInTheDocument());
    fireEvent.click(screen.getByRole("button", { name: "Expand trial 3" }));
    await vi.waitFor(() => expect(screen.getByRole("button", { name: "Expand pair 1" })).toBeInTheDocument());
    fireEvent.click(screen.getAllByRole("button", { name: "Expand pair 1" }).at(-1)!);
    expect(screen.getByText(/failed after 0 of 2 games/)).toBeInTheDocument();
    expect(screen.queryByText(/\d+%/)).not.toBeInTheDocument();
    expect(screen.getAllByRole("treeitem").some((node) => node.getAttribute("aria-expanded") === "true")).toBe(true);
    expect(screen.getByRole("tree")).toHaveAttribute("aria-label", "Tuning evidence hierarchy");
    expect(screen.getAllByRole("treeitem").some((node) => node.getAttribute("aria-selected") === "false")).toBe(true);
  });

  it("hands replay the attempt run and trace sequence, and explains missing links", async () => {
    const seen: BenchSpectatorProps[] = [];
    const Spectator = (props: BenchSpectatorProps) => { seen.push(props); return <div data-testid="spectator" />; };
    const { store } = setup();
    render(() => <TuningSessionWorkbench store={store} Spectator={Spectator} />);
    store.dispatch({ tag: "tuningNavigation", action: { tag: "selectSession", sessionId: "session-a" } });
    await screen.findByText("Attempts and evidence");
    for (const action of [{ tag: "selectAttempt", attemptId: "attempt-a" }, { tag: "selectTrial", trialId: "trial-a" }, { tag: "selectPair", pairId: "pair-two" }, { tag: "selectGame", gameId: "game-1" }] as const) store.dispatch({ tag: "tuningNavigation", action });
    expect(await screen.findByTestId("spectator")).toBeInTheDocument();
    expect(seen.at(-1)).toMatchObject({ runId: "physical-a", game: "nim", initialGameSeq: 41 });
    for (const action of [{ tag: "selectTrial", trialId: "trial-b" }, { tag: "selectPair", pairId: "pair-one" }, { tag: "selectGame", gameId: "game-3" }] as const) store.dispatch({ tag: "tuningNavigation", action });
    await vi.waitFor(() => expect(screen.getByText("Replay unavailable: this game has no trace sequence.")).toBeInTheDocument());
    for (const action of [{ tag: "selectTrial", trialId: "trial-c" }, { tag: "selectPair", pairId: "pair-zero" }] as const) store.dispatch({ tag: "tuningNavigation", action });
    await vi.waitFor(() => expect(screen.getByText("Select a recorded game to inspect its replay.")).toBeInTheDocument());
  });

  it("explains when a replayable game has no associated attempt run", async () => {
    const missingRun = detail();
    missingRun.attempts = [{ ...attemptA, bench_run_id: null }, attemptB];
    const { store } = setup(missingRun);
    const Spectator = () => <div data-testid="spectator" />;
    render(() => <TuningSessionWorkbench store={store} Spectator={Spectator} />);
    store.dispatch({ tag: "tuningNavigation", action: { tag: "selectSession", sessionId: "session-a" } });
    await screen.findByText("Attempts and evidence");
    for (const action of [{ tag: "selectTrial", trialId: "trial-a" }, { tag: "selectPair", pairId: "pair-two" }, { tag: "selectGame", gameId: "game-1" }] as const) store.dispatch({ tag: "tuningNavigation", action });
    expect(await screen.findByText("Replay unavailable: the attempt has no associated physical run.")).toBeInTheDocument();
    expect(screen.queryByTestId("spectator")).not.toBeInTheDocument();
  });

  it("keeps replay identity and expanded hierarchy stable when detail gains a pair", async () => {
    let current = detail();
    const { store } = setup(() => current);
    let mounts = 0;
    const Spectator = (props: BenchSpectatorProps) => { onMount(() => { mounts += 1; }); return <div data-testid="spectator">{props.runId}:{props.initialGameSeq}</div>; };
    render(() => <TuningSessionWorkbench store={store} Spectator={Spectator} />);
    store.dispatch({ tag: "tuningNavigation", action: { tag: "selectSession", sessionId: "session-a" } });
    await screen.findByText("Attempts and evidence");
    expect(screen.getByRole("heading", { name: "Resolved policy" })).toBeInTheDocument();
    expect(screen.getByText("2–6 (4–12 physical games)")).toBeInTheDocument();
    expect(screen.getByText("ThurstoneMostellerPart")).toBeInTheDocument();
    store.dispatch({ tag: "tuningNavigation", action: { tag: "selectTrial", trialId: "trial-a" } });
    expect(await screen.findByText("After 2 completed pairs")).toBeInTheDocument();
    expect(screen.getByText("startup_exempt")).toBeInTheDocument();
    expect(screen.getAllByText("max_pairs")).toHaveLength(2);
    expect(screen.getAllByText("unknown")).toHaveLength(2);
    store.dispatch({ tag: "tuningNavigation", action: { tag: "selectTrial", trialId: "trial-b" } });
    expect(await screen.findAllByText("hyperband_prune")).toHaveLength(2);
    expect(screen.getByText("prune")).toBeInTheDocument();
    for (const action of [{ tag: "toggleExpanded", id: "attempt:attempt-a" }, { tag: "toggleExpanded", id: "trial:trial-a" }, { tag: "toggleExpanded", id: "pair:pair-two" }, { tag: "selectAttempt", attemptId: "attempt-a" }, { tag: "selectTrial", trialId: "trial-a" }, { tag: "selectPair", pairId: "pair-two" }] as const) store.dispatch({ tag: "tuningNavigation", action });
    const gameButton = await screen.findByRole("button", { name: /Game · candidate first/ });
    fireEvent.click(gameButton);
    expect(await screen.findByTestId("spectator")).toHaveTextContent("physical-a:41");
    expect(mounts).toBe(1);
    gameButton.focus();
    expect(document.activeElement).toBe(gameButton);
    current = detail(true);
    store.dispatch({ tag: "tuningNavigation", action: { tag: "detailRequest", sessionId: "session-a" } });
    await vi.waitFor(() => expect(store.getState()().tuningNavigation.detail.snapshot?.cursor.session_sequence).toBe(2));
    const navigation = store.getState()().tuningNavigation;
    expect(navigation.selection.gameId).toBe("game-1");
    expect(navigation.expandedIds).toEqual(expect.arrayContaining(["attempt:attempt-a", "trial:trial-a"]));
    expect(navigation.selection).toMatchObject({ sessionId: "session-a", attemptId: "attempt-a", trialId: "trial-a", pairId: "pair-two", gameId: "game-1" });
    expect(document.activeElement).toBe(gameButton);
    expect(screen.getByTestId("spectator")).toHaveTextContent("physical-a:41");
    expect(mounts).toBe(1);
  });
});
