// delete-run.test.ts — pure-reducer tests for the run-delete control
// (Task 13d). A `TestStore` drives `tunerReducer` against a mocked
// `TunerEnv`; no live server (AGENTS.md "mock the environment").

import { describe, expect, it, vi } from "vitest";
import { Effect } from "@mcts/core";
import { createTestStore } from "../../../../tests/test-store.js";
import {
  initialTunerState,
  tunerReducer,
  type TunerAction,
  type TunerState,
} from "../../src/tuner/tuner-reducer.js";
import { mockTunerEnv, runView } from "./mock-tuner-env.js";
import type { TunerEnv } from "../../src/tuner/tuner-env.js";

const store = (env: TunerEnv) =>
  createTestStore<TunerState, TunerAction, TunerEnv>(tunerReducer, env, initialTunerState());

describe("tunerReducer — delete run", () => {
  it("dispatches deleteRun, drops the run from the fleet, and reconciles from the server", async () => {
    const deleteRun = vi.fn((_runId: string) => Effect.send(undefined));
    const ts = store(
      mockTunerEnv({
        deleteRun,
        listRuns: () => Effect.send([]),
        listProjectionRuns: () => Effect.send([]),
      }),
    );

    // Seed a terminal run in the fleet, then delete it.
    ts.send({ tag: "runsLoaded", runs: [runView({ run_id: "old", status: "exited" })] }, (s) => {
      s.runs = {
        status: "ok",
        value: [runView({ run_id: "old", status: "exited" })],
        fetchedAt: expect.any(Number) as unknown as number,
      };
    });

    ts.send({ tag: "deleteRun", runId: "old" }, (s) => {
      s.deletingRunId = "old";
    });
    await ts.drain();

    expect(deleteRun).toHaveBeenCalledTimes(1);
    expect(deleteRun.mock.calls[0]![0]).toBe("old");

    ts.receive({ tag: "deleteRunOk", runId: "old" }, (s) => {
      s.deletingRunId = null;
      s.runs = { status: "ok", value: [], fetchedAt: expect.any(Number) as unknown as number };
    });
    ts.receive({ tag: "runsLoaded", runs: [] });
    ts.receive({ tag: "projectionLoaded", runs: [] }, (s) => {
      s.projectionRuns = {
        status: "ok",
        value: [],
        fetchedAt: expect.any(Number) as unknown as number,
      };
    });
    ts.receive({ tag: "projectionMetaLoaded", meta: { last_pass_at: null } });
  });

  it("surfaces a server rejection (a live run) inline and leaves the fleet untouched", async () => {
    const ts = store(
      mockTunerEnv({
        deleteRun: () =>
          Effect.fromPromise(() => Promise.reject(new Error("run is still live"))),
      }),
    );

    ts.send({ tag: "deleteRun", runId: "busy" }, (s) => {
      s.deletingRunId = "busy";
    });
    await ts.drain();

    ts.receive({ tag: "deleteRunFailed", error: "Error: run is still live" }, (s) => {
      s.deletingRunId = null;
      s.deleteError = "Error: run is still live";
    });
  });

  it("ignores a second deleteRun while one is in flight", async () => {
    const deleteRun = vi.fn(() => Effect.send(undefined));
    const ts = store(
      mockTunerEnv({
        deleteRun,
        listRuns: () => Effect.send([]),
        listProjectionRuns: () => Effect.send([]),
      }),
    );

    ts.send({ tag: "deleteRun", runId: "a" }, (s) => {
      s.deletingRunId = "a";
    });
    ts.send({ tag: "deleteRun", runId: "b" });
    expect(deleteRun).toHaveBeenCalledTimes(1);

    await ts.drain();
    ts.receive({ tag: "deleteRunOk", runId: "a" }, (s) => {
      s.deletingRunId = null;
      s.runs = { status: "ok", value: [], fetchedAt: expect.any(Number) as unknown as number };
    });
    ts.receive({ tag: "runsLoaded", runs: [] });
    ts.receive({ tag: "projectionLoaded", runs: [] }, (s) => {
      s.projectionRuns = {
        status: "ok",
        value: [],
        fetchedAt: expect.any(Number) as unknown as number,
      };
    });
    ts.receive({ tag: "projectionMetaLoaded", meta: { last_pass_at: null } });
  });
});
