// run-overview-skeleton.component.test.tsx — RunOverview should show a
// loading skeleton (not a misleading "0 / 0 pairs" / "waiting for first
// event" empty state) until the run's own detail has loaded for the first
// time, and it should stay off the DOM once real data lands so a background
// refresh never flashes it back in.

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

describe("RunOverview — initial-load skeleton", () => {
  it("shows a skeleton instead of the progress rail while detail is still loading", () => {
    let resolveDetail!: (value: ProjectionRunDetail) => void;
    const env: TunerEnv = mockTunerEnv({
      getProjectionRun: () =>
        Effect.fromPromise(
          () => new Promise<ProjectionRunDetail>((r) => (resolveDetail = r)),
        ),
    });
    const store = createStore<TunerState, TunerAction, TunerEnv>(
      initialTunerState(),
      tunerReducer,
      env,
    );
    store.dispatch({ tag: "runsLoaded", runs: [runView({ run_id: "r1", status: "live" })] });
    store.dispatch({ tag: "openRun", runId: "r1" });
    render(() => <RunOverview store={store} runId="r1" navigate={() => {}} />);

    expect(screen.getByTestId("run-overview-skeleton")).toBeInTheDocument();
    expect(screen.queryByTestId("progress-rail")).not.toBeInTheDocument();
    void resolveDetail;
  });

  it("swaps in the progress rail once detail has loaded, and never shows the skeleton again on refresh", async () => {
    const env: TunerEnv = mockTunerEnv({
      getProjectionRun: () =>
        Effect.send({
          run_id: "r1",
          terminal_status: null,
          report_available: false,
          ingest_error: null,
          manifest: null,
          report: null,
          compute: [],
        }),
    });
    const store = createStore<TunerState, TunerAction, TunerEnv>(
      initialTunerState(),
      tunerReducer,
      env,
    );
    store.dispatch({ tag: "runsLoaded", runs: [runView({ run_id: "r1", status: "live" })] });
    store.dispatch({ tag: "openRun", runId: "r1" });
    render(() => <RunOverview store={store} runId="r1" navigate={() => {}} />);

    await vi.waitFor(() => expect(screen.getByTestId("progress-rail")).toBeInTheDocument());
    expect(screen.queryByTestId("run-overview-skeleton")).not.toBeInTheDocument();

    // A background refresh (e.g. `refreshProjection`) re-fetches detail
    // without going through `loading` from the user's point of view — the
    // skeleton must not reappear.
    store.dispatch({ tag: "refreshProjection" });
    await vi.waitFor(() => expect(screen.getByTestId("progress-rail")).toBeInTheDocument());
    expect(screen.queryByTestId("run-overview-skeleton")).not.toBeInTheDocument();
  });
});
