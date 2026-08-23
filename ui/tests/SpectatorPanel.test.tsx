import { createSignal } from "solid-js";
import { cleanup, fireEvent, render, screen } from "@solidjs/testing-library";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { SpectatorPanel, type TraceEnvironment, type TraceEventSource } from "../app/src/SpectatorPanel.js";
import type { GameMove, GameTraceSummary } from "../packages/bench/src/types.js";
import type { SearchReport } from "@mcts/game";

vi.mock("../app/src/games.js", () => import("./fixtures/fake-games-registry.js"));

function trace(gameSeq: number): GameTraceSummary {
  return {
    game_seq: gameSeq, match_seq: gameSeq, cell_id: null, seed: gameSeq, metrics: null, ply_count: 2,
    started_at: "2026-01-01T00:00:00Z", ended_at: "2026-01-01T00:00:01Z",
    strategy_a: "Candidate", strategy_b: "Base", outcome: "win_a", winner: "Candidate",
  };
}

function report(iterations: number, status: SearchReport<string>["status"] = "available"): SearchReport<string> {
  return {
    status, schema_version: 1, reason: status === "unavailable" ? "strategy_unsupported" : null,
    elapsed_seconds: 0.01, iteration_limit: iterations, time_limit_seconds: null, completed_iterations: iterations,
    termination: "iterations", selected_action: "inc", actions: [{ action: "inc", visits: iterations, share: 1, mean_value: 0.5, is_proven: false }],
    principal_variation: ["inc"], root_visits: iterations, tree_nodes: iterations, mean_depth: 1, max_depth: 1,
    graph_mode: "tree", tt_reads: 0, tt_writes: 0, tt_hits: 0, tt_hit_ratio: 0,
    iterations_per_second: iterations * 100, warnings: status === "partial" ? ["actions_truncated"] : [],
  };
}

function moves(...rows: Array<Partial<GameMove> & Pick<GameMove, "ply" | "state">>): GameMove[] {
  return rows.map((row) => ({ ts: "2026-01-01T00:00:00Z", mv: null, player: null, ...row }));
}

interface FakeSource extends TraceEventSource {
  url: string;
  emit(data: unknown): void;
  fail(): void;
  closed: boolean;
}

function environment(): { env: TraceEnvironment; api: { getRunGames: ReturnType<typeof vi.fn>; getRunGameMoves: ReturnType<typeof vi.fn> }; sources: FakeSource[] } {
  const sources: FakeSource[] = [];
  const api = { getRunGames: vi.fn(), getRunGameMoves: vi.fn() };
  return {
    api,
    sources,
    env: {
      api,
      eventSource: (url) => {
        const source: FakeSource = {
          url, closed: false, onmessage: null, onerror: null,
          emit(data) { this.onmessage?.({ data: JSON.stringify(data) } as MessageEvent<string>); },
          fail() { this.onerror?.(new Event("error")); },
          close() { this.closed = true; },
        };
        sources.push(source);
        return source;
      },
    },
  };
}

function deferred<T>(): { promise: Promise<T>; resolve(value: T): void; reject(error: unknown): void } {
  let resolve!: (value: T) => void;
  let reject!: (error: unknown) => void;
  return { promise: new Promise<T>((ok, fail) => { resolve = ok; reject = fail; }), resolve, reject };
}

afterEach(cleanup);

describe("SpectatorPanel trace boundary", () => {
  beforeEach(() => vi.restoreAllMocks());

  it("replays renderer state and every selected-ply search report with explicit ply controls", async () => {
    const { api, env } = environment();
    api.getRunGames.mockResolvedValue([trace(1)]);
    api.getRunGameMoves.mockResolvedValue(moves(
      { ply: 0, state: { board: 0 } },
      { ply: 1, state: { board: 1 }, mv: "inc", player: "A", search: report(12, "unavailable") },
      { ply: 2, state: { board: 2 }, mv: "inc", player: "B", search: report(18, "partial") },
    ));

    render(() => <SpectatorPanel runId="run-a" game="fake" kind="experiment" live={false} initialGameSeq={1} traceEnv={env} />);
    expect(await screen.findByTestId("fake-board")).toHaveTextContent('state:{"board":0}');
    expect(screen.getByRole("status")).toHaveTextContent("No final-search report");
    expect(screen.getByRole("button", { name: "First" })).toBeDisabled();
    expect(screen.getByRole("button", { name: "Previous" })).toBeDisabled();

    fireEvent.click(screen.getByRole("button", { name: "Next" }));
    expect(screen.getByRole("status")).toHaveTextContent("Final-search evidence unavailable");
    expect(screen.getAllByText("12").length).toBeGreaterThan(0);
    expect(screen.getByText("Per-ply search trend")).toBeInTheDocument();
    expect(screen.getByText(/1 newer ply/)).toBeInTheDocument();

    fireEvent.click(screen.getByRole("button", { name: "Last" }));
    expect(screen.getByRole("status")).toHaveTextContent("Final-search evidence is partial");
    expect(screen.getByRole("button", { name: "Next" })).toBeDisabled();
    fireEvent.click(screen.getByRole("button", { name: "First" }));
    expect(screen.getByText(/Ply 0 \/ 2/)).toBeInTheDocument();
  });

  it("uses the reduced-capability text fallback for old traces", async () => {
    const { api, env } = environment();
    api.getRunGames.mockResolvedValue([trace(2)]);
    api.getRunGameMoves.mockResolvedValue(moves({ ply: 0, state: "legacy trace text" }));

    render(() => <SpectatorPanel runId="run-a" game="fake" kind="tuner" live={false} initialGameSeq={2} traceEnv={env} />);
    expect(await screen.findByText("tuner trace", { exact: false })).toBeInTheDocument();
    expect(screen.getByText("legacy trace text")).toBeInTheDocument();
    expect(screen.queryByTestId("fake-board")).toBeNull();
  });

  it("keeps the chosen game and ply stable while live evidence adds games and plies", async () => {
    const { api, env, sources } = environment();
    api.getRunGames.mockResolvedValueOnce([trace(1)]).mockResolvedValueOnce([trace(3), trace(2), trace(1)]);
    api.getRunGameMoves.mockResolvedValue(moves(
      { ply: 0, state: { board: 0 } },
      { ply: 1, state: { board: 1 }, mv: "inc", player: "A", search: report(3) },
    ));

    render(() => <SpectatorPanel runId="run-a" game="fake" kind="experiment" live initialGameSeq={1} traceEnv={env} />);
    await screen.findByTestId("fake-board");
    fireEvent.click(screen.getByRole("button", { name: "Next" }));
    const selected = await vi.waitFor(() => {
      const source = sources.find((entry) => entry.url.includes("game_seq=1"));
      expect(source).toBeDefined();
      return source!;
    });
    selected.emit({ game_seq: 1, ply: 2, ts: "now", state: { board: 2 }, mv: "inc", player: "B", search: report(9) });

    await vi.waitFor(() => expect(screen.getByText("2 newer games")).toBeInTheDocument());
    expect(screen.getByText(/Ply 1 \/ 2 · 1 newer ply/)).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /#1/ })).toHaveClass("active");
    expect(screen.queryByRole("button", { name: /#3/ })).not.toBeNull();
  });

  it("ignores stale list and move completions, and closes stale selected-game streams", async () => {
    const { api, env, sources } = environment();
    const oldList = deferred<GameTraceSummary[]>();
    const newList = deferred<GameTraceSummary[]>();
    const oldMoves = deferred<GameMove[]>();
    const newMoves = deferred<GameMove[]>();
    api.getRunGames.mockReturnValueOnce(oldList.promise).mockReturnValueOnce(newList.promise).mockResolvedValue([trace(2), trace(1)]);
    api.getRunGameMoves.mockImplementation((_runId: string, gameSeq: number) => gameSeq === 1 ? oldMoves.promise : newMoves.promise);
    const [props, setProps] = createSignal({ runId: "run-a", game: "fake", kind: "experiment", live: true, initialGameSeq: undefined as number | undefined });
    render(() => <SpectatorPanel {...props()} traceEnv={env} />);

    setProps({ ...props(), runId: "run-b" });
    oldList.reject(new Error("old list failure"));
    newList.resolve([trace(1), trace(2)]);
    await vi.waitFor(() => expect(screen.queryByText(/old list failure/)).toBeNull());
    fireEvent.click(screen.getByRole("button", { name: /#1/ }));
    await vi.waitFor(() => expect(sources.some((source) => source.url.includes("game_seq=1"))).toBe(true));
    const oldSource = sources.find((source) => source.url.includes("game_seq=1"))!;
    fireEvent.click(screen.getByRole("button", { name: /#2/ }));
    expect(oldSource.closed).toBe(true);
    oldMoves.reject(new Error("old move failure"));
    newMoves.resolve(moves({ ply: 0, state: { board: 2 } }));
    expect(await screen.findByTestId("fake-board")).toHaveTextContent('state:{"board":2}');
    expect(screen.queryByText(/old move failure/)).toBeNull();

    oldSource.emit({ game_seq: 1, ply: 9, ts: "now", state: "stale", mv: null, player: null });
    expect(screen.queryByText("stale")).toBeNull();
  });
});
