import { afterEach, describe, expect, it, vi } from "vitest";
import { cleanup, fireEvent, render, screen } from "@solidjs/testing-library";
import { createStore, Effect } from "@mcts/core";
import { benchReducer, initialBenchState, type BenchAction, type BenchEnv, type BenchState, type TuningAnalysisOverview, type TuningSessionCommandResponse, type TuningSessionListItem } from "@mcts/bench";
import { TuningSessionWorkbench } from "../packages/bench/src/tuning/TuningSessionWorkbench.js";

const counts = { total: 2, queued: 0, running: 1, terminal: 1, completed: 1, failed: 0, pruned: 0, cancelled: 0 };
const capabilities = { has_lifecycle: true, has_pairs: true, has_renderer_trace: true, has_search_reports: true, has_trial_reports: true };
const control = (overrides: Partial<TuningSessionListItem["control"]> = {}) => ({
  version: 4,
  continuation: { target_trial_count: 5, consumed_trial_count: 2, remaining_trial_count: 3, active_attempt_id: "attempt-old", launch_reservation: null, stop_attempt_id: null, recovery_required: false },
  allowed_commands: [
    { command: "stop" as const, allowed: true, denial_reason: null },
    { command: "resume" as const, allowed: false, denial_reason: "active_attempt" },
    { command: "add_budget" as const, allowed: true, denial_reason: null },
  ],
  ...overrides,
});
const session = (id = "session-a", sessionControl = control()): TuningSessionListItem => ({
  session_id: id, game: "nim", label: "Controls", status: "active", target_trial_count: 5, counts,
  created_at: "2026-08-23T00:00:00Z", last_activity_at: "2026-08-23T00:01:00Z",
  attempts: [{ attempt_id: "attempt-old", bench_run_id: "run-old", status: "running", started_at: "2026-08-23T00:00:00Z", ended_at: null, failure: null }],
  capabilities, control: sessionControl,
});
const overview = (sessionControl = control()): TuningAnalysisOverview => ({
  schema_version: 1, policy: null, objective: { metric: "score", direction: "maximize", complete_trials_only: true }, cursor: { session_sequence: 1 },
  coverage: { trials: counts, reports: 0, pairs: { total: 0, running: 0, complete: 0, failed: 0, unmatched_pool_revisions: 0 }, points: { total: 0, returned: 0, sampled: false } },
  bracket_resources: [], decision_groups: [], points: [], best: null, pool_revisions: [], control: sessionControl,
});
const response = (sessionControl = control(), overrides: Partial<TuningSessionCommandResponse> = {}): TuningSessionCommandResponse => ({
  schema_version: 1, command_id: "server-command", replay: false, status: "extended", attempt_id: null, bench_run_id: null, signal: null,
  budget: { previous_target_trial_count: 5, delta: 2, target_trial_count: 7 }, launch_error: undefined, control: sessionControl, ...overrides,
});

function setup(options: {
  sessions?: TuningSessionListItem[];
  stop?: BenchEnv["stopTuningSession"];
  resume?: BenchEnv["resumeTuningSession"];
  budget?: BenchEnv["addTuningSessionBudget"];
} = {}) {
  const sessions = options.sessions ?? [session()];
  let lists = 0;
  let analyses = 0;
  const env = {
    listTuningSessions: () => { lists += 1; return Effect.send({ schema_version: 1 as const, sessions }); },
    getTuningAnalysisOverview: () => { analyses += 1; return Effect.send(overview(sessions[0]!.control)); },
    getTuningSession: () => Effect.none(), getTuningTrialPage: () => Effect.none(), getTuningTrialDetail: () => Effect.none(),
    stopTuningSession: options.stop ?? (() => Effect.send(response())),
    resumeTuningSession: options.resume ?? (() => Effect.send(response(
      control({ continuation: { ...control().continuation, active_attempt_id: null } }),
      { status: "resuming", attempt_id: "attempt-new", bench_run_id: "run-new" },
    ))),
    addTuningSessionBudget: options.budget ?? (() => Effect.send(response(
      control({ continuation: { ...control().continuation, target_trial_count: 7, remaining_trial_count: 5 } }),
    ))),
  } as unknown as BenchEnv;
  const store = createStore<BenchState, BenchAction>(initialBenchState(), benchReducer, env);
  store.dispatch({ tag: "tuningNavigation", action: { tag: "listRequest" } });
  store.dispatch({ tag: "tuningNavigation", action: { tag: "selectSession", sessionId: sessions[0]!.session_id } });
  render(() => <TuningSessionWorkbench store={store} />);
  return { store, lists: () => lists, analyses: () => analyses };
}

afterEach(cleanup);

describe("logical tuning session controls", () => {
  it("renders only projected commands and announces a pending Stop", async () => {
    let resolve!: (value: TuningSessionCommandResponse) => void;
    const { store } = setup({ stop: () => Effect.fromPromise(() => new Promise<TuningSessionCommandResponse>((done) => { resolve = done; })) });
    await screen.findByRole("button", { name: "Stop" });
    expect(screen.getByText("Resume unavailable: active attempt")).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "Stop" }));
    await vi.waitFor(() => expect(screen.getByText("Stop pending.")).toBeInTheDocument());
    expect(screen.getByRole("button", { name: "Stop" })).toBeDisabled();
    resolve(response(control({ version: 5 })));
    await vi.waitFor(() => expect(screen.getByText("Stop succeeded.")).toBeInTheDocument());
    expect(store.getState()().tuningNavigation.selection.attemptId).toBeNull();
  });

  it("validates and previews an additive target, then submits active extension without workers", async () => {
    const requests: unknown[] = [];
    const { store, lists, analyses } = setup({ budget: (_id, body) => { requests.push(body); return Effect.send(response(control({ version: 5, continuation: { ...control().continuation, target_trial_count: 8, remaining_trial_count: 6 } }))); } });
    await screen.findByText("5 + 1 = 6");
    const delta = screen.getByRole("textbox", { name: "Trials to add" });
    fireEvent.input(delta, { target: { value: "0" } });
    expect(screen.getByRole("alert")).toHaveTextContent("positive whole number");
    fireEvent.input(delta, { target: { value: "3" } });
    expect(screen.getByText("5 + 3 = 8")).toBeInTheDocument();
    expect(screen.queryByLabelText("Workers (optional)")).not.toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "Add N trials" }));
    await vi.waitFor(() => expect(requests).toHaveLength(1));
    expect(requests[0]).toMatchObject({ expected_version: 4, delta: 3, start: false });
    expect((requests[0] as { n_workers?: unknown }).n_workers).toBeUndefined();
    expect(store.getState()().tuningNavigation.list.snapshot?.sessions[0]?.target_trial_count).toBe(8);
    expect({ lists: lists(), analyses: analyses() }).toEqual({ lists: 2, analyses: 2 });
  });

  it("starts an idle extension only on explicit request and leaves the current selection unchanged until Open", async () => {
    const idleControl = control({ continuation: { ...control().continuation, active_attempt_id: null }, allowed_commands: [{ command: "resume", allowed: true, denial_reason: null }, { command: "add_budget", allowed: true, denial_reason: null }] });
    const requests: unknown[] = [];
    const { store } = setup({ sessions: [session("session-a", idleControl)], budget: (_id, body) => { requests.push(body); return Effect.send(response(control({ version: 5, continuation: { ...idleControl.continuation, target_trial_count: 7, remaining_trial_count: 5 } }), { status: "starting", attempt_id: "attempt-new", bench_run_id: "run-new" })); } });
    await screen.findByLabelText("Start a new attempt");
    fireEvent.click(screen.getByLabelText("Start a new attempt"));
    fireEvent.input(screen.getByLabelText("Workers (optional)"), { target: { value: "2" } });
    fireEvent.click(screen.getByRole("button", { name: "Add N trials and start" }));
    await vi.waitFor(() => expect(requests).toHaveLength(1));
    expect(requests[0]).toMatchObject({ delta: 1, start: true, n_workers: 2 });
    expect(store.getState()().tuningNavigation.selection.attemptId).toBeNull();
    expect(screen.getByText(/New attempt attempt-new is available/)).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "Open attempt" }));
    expect(store.getState()().tuningNavigation.selection.attemptId).toBe("attempt-new");
  });

  it("resumes only when the server projects Resume and announces the new attempt", async () => {
    const idleControl = control({ continuation: { ...control().continuation, active_attempt_id: null }, allowed_commands: [{ command: "resume", allowed: true, denial_reason: null }] });
    const requests: unknown[] = [];
    const { store } = setup({ sessions: [session("session-a", idleControl)], resume: (_id, body) => {
      requests.push(body);
      return Effect.send(response(control({ version: 5, continuation: { ...idleControl.continuation, active_attempt_id: null } }), { status: "resuming", attempt_id: "attempt-resumed", bench_run_id: "run-resumed" }));
    } });
    await screen.findByRole("button", { name: "Resume" });
    fireEvent.click(screen.getByRole("button", { name: "Resume" }));
    await vi.waitFor(() => expect(screen.getByText("Resume succeeded.")).toBeInTheDocument());
    expect(requests[0]).toMatchObject({ expected_version: 4 });
    expect(screen.getByText(/New attempt attempt-resumed is available/)).toBeInTheDocument();
    expect(store.getState()().tuningNavigation.selection.attemptId).toBeNull();
  });

  it("keeps a transport command id for retry and reports a replayed success", async () => {
    const ids: string[] = [];
    let calls = 0;
    const { store } = setup({ stop: (_id, body) => {
      ids.push(body.command_id);
      calls += 1;
      return calls === 1 ? Effect.fromPromise(() => Promise.reject(new Error("offline"))) : Effect.send(response(control({ version: 5 }), { command_id: body.command_id, replay: true }));
    } });
    await screen.findByRole("button", { name: "Stop" });
    fireEvent.click(screen.getByRole("button", { name: "Stop" }));
    await screen.findByRole("alert");
    expect(screen.getByRole("alert")).toHaveTextContent("Stop failed: Error: offline");
    fireEvent.click(screen.getByRole("button", { name: "Retry Stop" }));
    await vi.waitFor(() => expect(screen.getByText("Stop succeeded (replayed request).")).toBeInTheDocument());
    expect(ids).toHaveLength(2);
    expect(ids[0]).toBe(ids[1]);
    expect(store.getState()().tuningNavigation.selection.sessionId).toBe("session-a");
  });

  it.each([409, 422])("does not retry a definitive %i rejection", async (status) => {
    const { store } = setup({ stop: () => Effect.fromPromise(() => Promise.reject(Object.assign(new Error(`HTTP ${status}`), { status }))) });
    await screen.findByRole("button", { name: "Stop" });
    fireEvent.click(screen.getByRole("button", { name: "Stop" }));
    await screen.findByRole("alert");
    expect(screen.getByRole("alert")).toHaveTextContent(`Stop failed: Error: HTTP ${status}`);
    expect(screen.queryByRole("button", { name: "Retry Stop" })).not.toBeInTheDocument();
    expect(store.getState()().tuningNavigation.selection.sessionId).toBe("session-a");
  });

  it("reports a launch error while retaining the returned target and does not move selection", async () => {
    const launchedControl = control({ version: 5, continuation: { ...control().continuation, target_trial_count: 7, remaining_trial_count: 5 } });
    const { store } = setup({ budget: () => Effect.send(response(launchedControl, { status: "launch_failed", launch_error: "spawn failed", attempt_id: "attempt-new", bench_run_id: "run-new" })) });
    await screen.findByRole("button", { name: "Add N trials" });
    fireEvent.click(screen.getByRole("button", { name: "Add N trials" }));
    await screen.findByRole("alert");
    expect(screen.getByRole("alert")).toHaveTextContent("Add N trials failed: spawn failed");
    expect(screen.queryByRole("button", { name: "Open attempt" })).not.toBeInTheDocument();
    expect(store.getState()().tuningNavigation.list.snapshot?.sessions[0]?.target_trial_count).toBe(7);
    expect(store.getState()().tuningNavigation.selection).toMatchObject({ sessionId: "session-a", attemptId: null, trialId: null, pairId: null, gameId: null });
  });

  it("drops an old session command result after a rapid session change without changing the new selection", async () => {
    let resolve!: (value: TuningSessionCommandResponse) => void;
    const secondControl = control({ version: 9, continuation: { ...control().continuation, active_attempt_id: null } });
    const { store } = setup({ sessions: [session("session-a"), session("session-b", secondControl)], stop: () => Effect.fromPromise(() => new Promise<TuningSessionCommandResponse>((done) => { resolve = done; })) });
    await screen.findByRole("button", { name: "Stop" });
    fireEvent.click(screen.getByRole("button", { name: "Stop" }));
    await vi.waitFor(() => expect(screen.getByText("Stop pending.")).toBeInTheDocument());
    store.dispatch({ tag: "tuningNavigation", action: { tag: "selectSession", sessionId: "session-b" } });
    resolve(response(control({ version: 5 })));
    await vi.waitFor(() => expect(store.getState()().tuningNavigation.commands["session-a"]?.status).toBe("succeeded"));
    expect(store.getState()().tuningNavigation.selection).toEqual({ sessionId: "session-b", attemptId: null, trialId: null, pairId: null, gameId: null });
  });
});
