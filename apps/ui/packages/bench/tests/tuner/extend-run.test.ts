// extend-run.test.ts — pure-reducer tests for the budget-extension control
// (Task 13b). A `TestStore` drives `tunerReducer` against a mocked
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
import type { TunerBudgetExtension } from "../../src/tuner/tuner-types.js";

const store = (env: TunerEnv) =>
  createTestStore<TunerState, TunerAction, TunerEnv>(tunerReducer, env, initialTunerState());

const extension: TunerBudgetExtension = {
  tuning_pair_attempts_delta: 40,
  validation_pair_attempts_delta: 0,
  diagnostic_pair_attempts_delta: 0,
  reason: "cohort race still open at the freeze",
};

describe("tunerReducer — extend budget", () => {
  it("dispatches extendRun with the exact body and re-arms the journal on success", async () => {
    const extendRun = vi.fn((_runId: string, _body: TunerBudgetExtension) =>
      Effect.send(runView({ run_id: "r1", status: "live" })),
    );
    const ts = store(
      mockTunerEnv({
        extendRun,
        // The relaunched run is back in the journal as live; keep it a single
        // poll (a follow-up exited poll would wind the loop straight down).
        listRuns: () => Effect.send([runView({ run_id: "r1", status: "exited" })]),
      }),
    );

    ts.send({ tag: "extendRun", runId: "r1", extension }, (s) => {
      s.extendBusy = true;
    });
    await ts.drain();

    expect(extendRun).toHaveBeenCalledTimes(1);
    expect(extendRun.mock.calls[0]![0]).toBe("r1");
    expect(extendRun.mock.calls[0]![1]).toEqual(extension);

    ts.receive({ tag: "extendOk" }, (s) => {
      s.extendBusy = false;
      s.extendSeq = 1;
      s.journalGeneration = 1;
    });
    ts.receive({ tag: "journalTick", generation: 1 });
    ts.receive({ tag: "runsLoaded", runs: [runView({ run_id: "r1", status: "exited" })] }, (s) => {
      s.runs = { status: "ok", value: [runView({ run_id: "r1", status: "exited" })], fetchedAt: expect.any(Number) as unknown as number };
    });
  });

  it("surfaces a server rejection inline and leaves the run untouched", async () => {
    const ts = store(
      mockTunerEnv({
        extendRun: () =>
          Effect.fromPromise(() =>
            Promise.reject(new Error("validation delta must be divisible by finalists (2)")),
          ),
      }),
    );

    ts.send({ tag: "extendRun", runId: "r1", extension }, (s) => {
      s.extendBusy = true;
    });
    await ts.drain();

    ts.receive(
      {
        tag: "extendFailed",
        error: "Error: validation delta must be divisible by finalists (2)",
      },
      (s) => {
        s.extendBusy = false;
        s.extendError = "Error: validation delta must be divisible by finalists (2)";
      },
    );
  });

  it("ignores a second extendRun while one is in flight", async () => {
    const extendRun = vi.fn(() => Effect.send(runView({ run_id: "r1", status: "exited" })));
    const ts = store(
      mockTunerEnv({
        extendRun,
        listRuns: () => Effect.send([runView({ run_id: "r1", status: "exited" })]),
      }),
    );

    ts.send({ tag: "extendRun", runId: "r1", extension }, (s) => {
      s.extendBusy = true;
    });
    // Second dispatch while the first is still busy is a no-op.
    ts.send({ tag: "extendRun", runId: "r1", extension });
    expect(extendRun).toHaveBeenCalledTimes(1);

    await ts.drain();
    ts.receive({ tag: "extendOk" }, (s) => {
      s.extendBusy = false;
      s.extendSeq = 1;
      s.journalGeneration = 1;
    });
    ts.receive({ tag: "journalTick", generation: 1 });
    ts.receive({ tag: "runsLoaded", runs: [runView({ run_id: "r1", status: "exited" })] }, (s) => {
      s.runs = { status: "ok", value: [runView({ run_id: "r1", status: "exited" })], fetchedAt: expect.any(Number) as unknown as number };
    });
  });
});
