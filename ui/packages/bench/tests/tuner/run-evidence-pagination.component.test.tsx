// run-evidence-pagination.component.test.tsx — component test for
// RunEvidence's bounded pairs fetch (Task 14c). A real
// `createStore(tunerReducer, env)` backs the rendered view; the env is
// mocked (AGENTS.md "mock the environment"), no live server.
//
// Guards against the truncation bug this session fixed: the mocked env
// returns a 100-row page while `getProjectionRun`'s compute rollup reports
// 1690 completed pairs total, and the header must show the latter.

import { afterEach, describe, expect, it, vi } from "vitest";
import { cleanup, fireEvent, render, screen } from "@solidjs/testing-library";
import { Effect, createStore } from "@mcts/core";
import {
  initialTunerState,
  tunerReducer,
  type TunerAction,
  type TunerState,
} from "../../src/tuner/tuner-reducer.js";
import { RunEvidence } from "../../src/tuner/views/RunEvidence.js";
import { mockTunerEnv } from "./mock-tuner-env.js";
import type { TunerEnv } from "../../src/tuner/tuner-env.js";
import type {
  ProjectionPairQuery,
  ProjectionPairRow,
  ProjectionRunDetail,
} from "../../src/tuner/tuner-types.js";

afterEach(cleanup);

function pairRow(i: number): ProjectionPairRow {
  return {
    pair_id: `pair-${i}`,
    phase: "tuning",
    candidate_id: "cand-0",
    task_id: `task-${i}`,
    opponent_id: "baseline",
    pair_utility: 0.5,
  };
}

const TOTAL_PAIRS = 1690;

const detail: ProjectionRunDetail = {
  run_id: "r1",
  terminal_status: "complete",
  report_available: true,
  ingest_error: null,
  manifest: null,
  report: null,
  compute: [
    {
      phase: "tuning",
      pair_attempts: TOTAL_PAIRS,
      completed_pairs: TOTAL_PAIRS,
      failed_attempts: 0,
      censored_attempts: 0,
      physical_games: TOTAL_PAIRS * 2,
      search_iterations: 0,
      wall_time_ms: 0,
    },
  ],
};

function renderEvidence(calls: ProjectionPairQuery[]) {
  const getProjectionPairs: TunerEnv["getProjectionPairs"] = (_runId, query = {}) => {
    calls.push(query);
    const offset = query.offset ?? 0;
    const limit = query.limit ?? 100;
    const remaining = Math.max(0, Math.min(limit, TOTAL_PAIRS - offset));
    return Effect.send(Array.from({ length: remaining }, (_, i) => pairRow(offset + i)));
  };
  const env = mockTunerEnv({
    getProjectionRun: () => Effect.send(detail),
    getProjectionPairs,
  });
  const store = createStore<TunerState, TunerAction, TunerEnv>(
    initialTunerState(),
    tunerReducer,
    env,
  );
  store.dispatch({ tag: "openRun", runId: "r1" });
  render(() => <RunEvidence store={store} runId="r1" navigate={() => {}} />);
  return store;
}

describe("RunEvidence — pairs pagination", () => {
  it("shows the server-reported total, not the fetched page length", async () => {
    const calls: ProjectionPairQuery[] = [];
    renderEvidence(calls);

    await vi.waitFor(() =>
      expect(screen.getByText(`Pairs (${TOTAL_PAIRS})`)).toBeInTheDocument(),
    );
    expect(calls[0]).toMatchObject({ limit: 100, offset: 0 });
    expect(screen.getByText(`1–100 of ${TOTAL_PAIRS}`)).toBeInTheDocument();
  });

  it("Next fetches the next page and Prev returns to the first", async () => {
    const calls: ProjectionPairQuery[] = [];
    renderEvidence(calls);
    await vi.waitFor(() =>
      expect(screen.getByText(`Pairs (${TOTAL_PAIRS})`)).toBeInTheDocument(),
    );

    const prev = screen.getByTestId("evidence-pairs-prev") as HTMLButtonElement;
    const next = screen.getByTestId("evidence-pairs-next") as HTMLButtonElement;
    expect(prev).toBeDisabled();
    expect(next).not.toBeDisabled();

    fireEvent.click(next);
    await vi.waitFor(() =>
      expect(screen.getByText(`101–200 of ${TOTAL_PAIRS}`)).toBeInTheDocument(),
    );
    expect(calls.at(-1)).toMatchObject({ limit: 100, offset: 100 });
    expect(prev).not.toBeDisabled();

    fireEvent.click(prev);
    await vi.waitFor(() =>
      expect(screen.getByText(`1–100 of ${TOTAL_PAIRS}`)).toBeInTheDocument(),
    );
    expect(calls.at(-1)).toMatchObject({ limit: 100, offset: 0 });
  });
});
