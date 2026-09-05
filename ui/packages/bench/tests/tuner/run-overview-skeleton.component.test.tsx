// run-overview-skeleton.component.test.tsx — RunOverview should show a
// loading skeleton (not a misleading "0 / 0 pairs" / "waiting for first
// event" empty state) only while it has *no* signal at all — neither the
// run's own projection detail nor a live evidence event. As soon as either
// arrives it should show the real progress rail, which already knows how to
// render partial (ledger-not-ready) progress off the live evidence tally —
// including while detail is still loading, so a run's first evidence line
// proves the page is working well before its projection resolves. The
// skeleton should also stay off the DOM once real data lands, so a
// background refresh never flashes it back in.

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
import type { EvidenceEnvelope, ProjectionRunDetail } from "../../src/tuner/tuner-types.js";

afterEach(cleanup);

describe("RunOverview — initial-load skeleton", () => {
  it("shows a skeleton while neither detail nor any live evidence has arrived", () => {
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

  it("shows the live progress rail as soon as an evidence event arrives, even while detail is still loading", async () => {
    let sendEvents!: (events: EvidenceEnvelope[]) => void;
    const env: TunerEnv = mockTunerEnv({
      // Detail never resolves in this test — the point is that the live
      // evidence tally alone is enough to replace the skeleton.
      getProjectionRun: () => Effect.fromPromise(() => new Promise<ProjectionRunDetail>(() => {})),
      openEvidenceStream: () =>
        Effect.stream((send) => {
          sendEvents = (events) => send({ kind: "events", events });
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

    expect(screen.getByTestId("run-overview-skeleton")).toBeInTheDocument();

    sendEvents([
      {
        sequence: 1,
        type: "pair_started",
        payload: { pair_id: "p1", candidate_id: "c1", opponent_id: "c2" },
      },
    ]);

    await vi.waitFor(() =>
      expect(screen.getByTestId("progress-live-counts")).toBeInTheDocument(),
    );
    expect(screen.getByTestId("progress-live-counts").textContent).toContain(
      "1 evidence lines ingested so far",
    );
    expect(screen.queryByTestId("run-overview-skeleton")).not.toBeInTheDocument();
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

    // A background refresh (`autoRefreshProjection`) re-fetches detail
    // without going through `loading` from the user's point of view — the
    // skeleton must not reappear.
    store.dispatch({ tag: "autoRefreshProjection" });
    await vi.waitFor(() => expect(screen.getByTestId("progress-rail")).toBeInTheDocument());
    expect(screen.queryByTestId("run-overview-skeleton")).not.toBeInTheDocument();
  });
});
