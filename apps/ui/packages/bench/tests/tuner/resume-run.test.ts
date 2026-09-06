// resume-run.test.ts — pure-reducer tests for the plain-resume control (no
// budget change). A `TestStore` drives `tunerReducer` against a mocked
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

describe("tunerReducer — resume", () => {
  it("dispatches resumeRun and re-arms the journal on success", async () => {
    const resumeRun = vi.fn((_runId: string) =>
      Effect.send(runView({ run_id: "r1", status: "live" })),
    );
    const ts = store(
      mockTunerEnv({
        resumeRun,
        listRuns: () => Effect.send([runView({ run_id: "r1", status: "exited" })]),
      }),
    );

    ts.send({ tag: "resumeRun", runId: "r1" }, (s) => {
      s.resumeBusy = true;
    });
    await ts.drain();

    expect(resumeRun).toHaveBeenCalledTimes(1);
    expect(resumeRun.mock.calls[0]![0]).toBe("r1");

    ts.receive({ tag: "resumeOk" }, (s) => {
      s.resumeBusy = false;
      s.journalGeneration = 1;
    });
    ts.receive({ tag: "journalTick", generation: 1 });
    ts.receive({ tag: "runsLoaded", runs: [runView({ run_id: "r1", status: "exited" })] }, (s) => {
      s.runs = {
        status: "ok",
        value: [runView({ run_id: "r1", status: "exited" })],
        fetchedAt: expect.any(Number) as unknown as number,
      };
    });
  });

  it("surfaces a server rejection inline and leaves the run untouched", async () => {
    const ts = store(
      mockTunerEnv({
        resumeRun: () =>
          Effect.fromPromise(() =>
            Promise.reject(new Error("tuner run directory is missing")),
          ),
      }),
    );

    ts.send({ tag: "resumeRun", runId: "r1" }, (s) => {
      s.resumeBusy = true;
    });
    await ts.drain();

    ts.receive(
      { tag: "resumeFailed", error: "Error: tuner run directory is missing" },
      (s) => {
        s.resumeBusy = false;
        s.resumeError = "Error: tuner run directory is missing";
      },
    );
  });

  it("ignores a second resumeRun while one is in flight", async () => {
    const resumeRun = vi.fn(() => Effect.send(runView({ run_id: "r1", status: "exited" })));
    const ts = store(
      mockTunerEnv({
        resumeRun,
        listRuns: () => Effect.send([runView({ run_id: "r1", status: "exited" })]),
      }),
    );

    ts.send({ tag: "resumeRun", runId: "r1" }, (s) => {
      s.resumeBusy = true;
    });
    ts.send({ tag: "resumeRun", runId: "r1" });
    expect(resumeRun).toHaveBeenCalledTimes(1);

    await ts.drain();
    ts.receive({ tag: "resumeOk" }, (s) => {
      s.resumeBusy = false;
      s.journalGeneration = 1;
    });
    ts.receive({ tag: "journalTick", generation: 1 });
    ts.receive({ tag: "runsLoaded", runs: [runView({ run_id: "r1", status: "exited" })] }, (s) => {
      s.runs = {
        status: "ok",
        value: [runView({ run_id: "r1", status: "exited" })],
        fetchedAt: expect.any(Number) as unknown as number,
      };
    });
  });
});
