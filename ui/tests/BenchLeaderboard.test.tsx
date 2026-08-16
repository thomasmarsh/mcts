import { cleanup, render, screen } from "@solidjs/testing-library";
import { afterEach, describe, expect, it } from "vitest";
import { createStore } from "@mcts/core";
import { LeaderboardTable, benchReducer, initialBenchState, type BenchAction, type BenchEnv, type BenchState } from "../packages/bench/src/index.js";
import { createMockBenchEnv } from "./fixtures/fake-bench.js";

afterEach(() => cleanup());

describe("LeaderboardTable result presentation", () => {
  it("keeps observed W/L/D and returned interval values while distinguishing no games", () => {
    const state = initialBenchState();
    state.leaderboard = {
      status: "done",
      jobId: null,
      result: [
        { strategy: "observed", total: 4, wins: 3, losses: 1, draws: 0, win_rate: 0.75, ci_lower: 0.4, ci_upper: 0.9 },
        { strategy: "empty", total: 0, wins: 0, losses: 0, draws: 0, win_rate: 0.5, ci_lower: 0, ci_upper: 1 },
      ],
      error: null,
      attempt: 0,
    };
    const env = createMockBenchEnv() as BenchEnv;
    const store = createStore<BenchState, BenchAction, BenchEnv>(state, benchReducer, env);
    render(() => <LeaderboardTable store={store} />);

    expect(screen.getByText("3/1/0")).toBeInTheDocument();
    expect(screen.getByText("75.0%")).toBeInTheDocument();
    expect(screen.getByText("40.0% – 90.0%")).toBeInTheDocument();
    expect(screen.getAllByText("No games yet")).toHaveLength(2);
    expect(document.querySelectorAll(".lb-bar-fill")[1]).toHaveStyle({ width: "0%" });
  });
});
