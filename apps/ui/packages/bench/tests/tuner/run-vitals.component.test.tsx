// run-vitals.component.test.tsx — RunOverview renders the wall-clock
// Vitals panel (15f) once the telemetry rollup and projection detail have
// landed: a budget-anchored bar, a live ETA, and the performance table,
// all without opening Science or Evidence.

import { afterEach, describe, expect, it, vi } from "vitest";
import { cleanup, render, screen } from "@solidjs/testing-library";
import { Effect, createStore } from "@mcts/core";
import {
  initialTunerState,
  tunerReducer,
  type TunerAction,
  type TunerState,
} from "../../src/tuner/tuner-reducer.js";
import { RunOverview } from "../../src/tuner/views/RunOverview.js";
import { mockTunerEnv, runView } from "./mock-tuner-env.js";
import type { TunerEnv } from "../../src/tuner/tuner-env.js";
import type { ProjectionRunDetail } from "../../src/tuner/tuner-types.js";

afterEach(cleanup);

const detail: ProjectionRunDetail = {
  run_id: "r1",
  terminal_status: null,
  report_available: false,
  ingest_error: null,
  manifest: {
    manifest_run_id: "r1",
    manifest_fingerprint: "f",
    game_kind: "druid",
    objective_id: "o",
    cohort_size: 4,
    finalists: 2,
    seed: 1,
    task_seed: 2,
    shadow_policy_kind: "none",
    active_elimination: false,
    tuning_pair_budget: 84,
    validation_pair_budget: 4,
    diagnostic_pair_budget: 0,
  },
  report: null,
  compute: [
    {
      phase: "tuning",
      pair_attempts: 44,
      completed_pairs: 44,
      failed_attempts: 0,
      censored_attempts: 0,
      physical_games: 88,
      search_iterations: 0,
      wall_time_ms: 1_600_000,
    },
  ],
};

describe("RunOverview — Vitals panel", () => {
  it("shows the ETA, budget bar and performance KPIs from the telemetry rollup", async () => {
    const env: TunerEnv = mockTunerEnv({
      getProjectionRun: () => Effect.send(detail),
      getProjectionTelemetry: (runId) =>
        Effect.send({
          run_id: runId,
          sessions: 1,
          first_start_us: 1_000_000_000,
          last_end_us: 3_000_000_000,
          wall_span_us: 2_000_000_000,
          loop_active_us: 200_000_000,
          wait_us: 800_000_000,
          lanes: [],
        }),
    });
    const store = createStore<TunerState, TunerAction, TunerEnv>(
      initialTunerState(),
      tunerReducer,
      env,
    );
    store.dispatch({
      tag: "runsLoaded",
      runs: [runView({ run_id: "r1", status: "exited" })],
    });
    store.dispatch({ tag: "openRun", runId: "r1" });
    render(() => <RunOverview store={store} runId="r1" navigate={() => {}} />);

    await vi.waitFor(() => expect(screen.getByTestId("run-vitals")).toBeInTheDocument());
    // 44 completed / 88 budget.
    const bar = screen.getByTestId("run-vitals-bar");
    expect(bar.getAttribute("aria-valuenow")).toBe("50");
    // 44 pairs left * (2_000_000 ms / 44) ≈ 33m 20s.
    expect(screen.getByTestId("run-vitals-eta").textContent).toContain("ETA");
    const kpis = screen.getByTestId("run-vitals-kpis").textContent ?? "";
    expect(kpis).toContain("2.00×");
    expect(kpis).toContain("44 / 88");
  });
});
