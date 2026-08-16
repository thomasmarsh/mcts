import { createSignal } from "solid-js";
import { cleanup, render, screen } from "@solidjs/testing-library";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { SpectatorPanel } from "../app/src/SpectatorPanel.js";
import type { GameMove, GameTraceSummary } from "../packages/bench/src/types.js";

vi.mock("../app/src/games.js", () => import("./fixtures/fake-games-registry.js"));

const api = vi.hoisted(() => ({
  getRunGames: vi.fn(),
  getRunGameMoves: vi.fn(),
}));

vi.mock("../packages/bench/src/api-client.js", () => ({
  createBenchApiClient: () => api,
}));

afterEach(() => cleanup());

function trace(gameSeq: number, cellId: string): GameTraceSummary {
  return {
    game_seq: gameSeq,
    match_seq: gameSeq,
    cell_id: cellId,
    seed: gameSeq,
    metrics: null,
    ply_count: 1,
    started_at: "2026-01-01T00:00:00Z",
    ended_at: "2026-01-01T00:00:01Z",
    strategy_a: "Candidate",
    strategy_b: "Base",
    outcome: "win_a",
    winner: "Candidate",
  };
}

function move(text: string): GameMove {
  return { ply: 0, ts: "2026-01-01T00:00:00Z", state: text, mv: null, player: null };
}

describe("SpectatorPanel trace boundary", () => {
  beforeEach(() => {
    api.getRunGames.mockReset();
    api.getRunGameMoves.mockReset();
  });

  it("clears old moves across run/cell changes and reapplies the exact requested sequence", async () => {
    const requests: Array<{ runId: string; limit: number; cellId: string | undefined; resolve: (games: GameTraceSummary[]) => void }> = [];
    api.getRunGames.mockImplementation((runId: string, limit: number, cellId?: string) => new Promise<GameTraceSummary[]>((resolve) => {
      requests.push({ runId, limit, cellId, resolve });
    }));
    api.getRunGameMoves.mockImplementation((runId: string, gameSeq: number) => Promise.resolve([move(`${runId}:${gameSeq}`)]));

    const [props, setProps] = createSignal({ runId: "run-a", game: "fake", kind: "experiment", live: false, cellId: "cell-a", initialGameSeq: 7 });
    render(() => <SpectatorPanel {...props()} />);

    await vi.waitFor(() => expect(requests).toHaveLength(1));
    expect(requests[0]).toMatchObject({ runId: "run-a", limit: 100, cellId: "cell-a" });
    requests[0]!.resolve([trace(7, "cell-a")]);
    await vi.waitFor(() => expect(api.getRunGameMoves).toHaveBeenCalledWith("run-a", 7));
    await vi.waitFor(() => expect(screen.getByText("run-a:7")).toBeInTheDocument());

    setProps({ runId: "run-b", game: "fake", kind: "experiment", live: false, cellId: "cell-b", initialGameSeq: 7 });
    await vi.waitFor(() => expect(screen.queryByText("run-a:7")).not.toBeInTheDocument());
    await vi.waitFor(() => expect(requests).toHaveLength(2));
    expect(requests[1]).toMatchObject({ runId: "run-b", limit: 100, cellId: "cell-b" });

    requests[1]!.resolve([trace(7, "cell-b")]);
    await vi.waitFor(() => expect(api.getRunGameMoves).toHaveBeenLastCalledWith("run-b", 7));
    await vi.waitFor(() => expect(screen.getByText("run-b:7")).toBeInTheDocument());
  });
});
