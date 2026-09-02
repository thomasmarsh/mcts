// tuner-reducer.test.ts — pure-reducer tests via the shared TestStore
// harness against a mocked `TunerEnv` (no live server, per AGENTS.md). The
// journal and log-tail poll loops schedule with `Effect.delay` and run on
// TestStore's manual scheduler; every test winds them down so no sleep is
// left pending.

import { describe, expect, it } from "vitest";
import { Effect } from "@mcts/core";
import { createTestStore } from "../../../../tests/test-store.js";
import {
  initialTunerState,
  tunerReducer,
  LOG_TAIL_MS,
  type TunerAction,
  type TunerState,
} from "../../src/tuner/tuner-reducer.js";
import { JOURNAL_POLL_MS } from "../../src/tuner/tuner-poll.js";
import type { RemoteData } from "../../src/tuner/remote-data.js";
import { mockTunerEnv, runView } from "./mock-tuner-env.js";
import type { TunerEnv } from "../../src/tuner/tuner-env.js";

const store = (env: TunerEnv) =>
  createTestStore<TunerState, TunerAction, TunerEnv>(tunerReducer, env, initialTunerState());

/** `{status:"ok", value, fetchedAt: <any number>}` — the reducer stamps
 * `Date.now()`, which `toEqual` matches with an asymmetric matcher. */
function ok<T>(value: T): RemoteData<T> {
  return { status: "ok", value, fetchedAt: expect.any(Number) as unknown as number };
}

const loadingAll = (s: TunerState): void => {
  s.kinds = { status: "loading" };
  s.objectives = { status: "loading" };
  s.runs = { status: "loading" };
  s.projectionRuns = { status: "loading" };
  s.journalGeneration = 1;
};

const drainInit = (ts: ReturnType<typeof store>): void => {
  ts.receive({ tag: "kindsLoaded", kinds: [] }, (s) => {
    s.kinds = ok([]);
  });
  ts.receive({ tag: "objectivesLoaded", objectives: [] }, (s) => {
    s.objectives = ok([]);
  });
};

describe("tunerReducer", () => {
  it("init loads every resource and stops polling when nothing is live", () => {
    const ts = store(
      mockTunerEnv({ listRuns: () => Effect.send([runView({ status: "exited" })]) }),
    );

    ts.send({ tag: "init" }, loadingAll);
    drainInit(ts);
    ts.receive({ tag: "runsLoaded", runs: [runView({ status: "exited" })] }, (s) => {
      s.runs = ok([runView({ status: "exited" })]);
    });
    ts.receive({ tag: "projectionLoaded", runs: [] }, (s) => {
      s.projectionRuns = ok([]);
    });
  });

  it("keeps polling the journal while a run is live, then stops and refreshes", () => {
    let call = 0;
    const ts = store(
      mockTunerEnv({
        listRuns: () => {
          call += 1;
          return Effect.send(
            call < 2 ? [runView({ status: "live" })] : [runView({ status: "exited" })],
          );
        },
      }),
    );

    ts.send({ tag: "init" }, loadingAll);
    drainInit(ts);
    ts.receive({ tag: "runsLoaded", runs: [runView({ status: "live" })] }, (s) => {
      s.runs = ok([runView({ status: "live" })]);
    });
    ts.receive({ tag: "projectionLoaded", runs: [] }, (s) => {
      s.projectionRuns = ok([]);
    });

    ts.advance(JOURNAL_POLL_MS);
    ts.receive({ tag: "journalTick", generation: 1 });
    ts.receive({ tag: "runsLoaded", runs: [runView({ status: "exited" })] }, (s) => {
      s.runs = ok([runView({ status: "exited" })]);
    });
    ts.receive({ tag: "refreshProjection" }, (s) => {
      s.refreshing = true;
    });
    ts.receive({ tag: "refreshDone" }, (s) => {
      s.refreshing = false;
      s.lastProjectionRefreshAt = expect.any(Number) as unknown as number;
    });
    ts.receive({ tag: "projectionLoaded", runs: [] });
  });

  it("optimistically inserts and opens a launched run, then tails its log", () => {
    const launched = runView({ run_id: "fresh", status: "live" });
    const detail = {
      run_id: "fresh",
      terminal_status: null,
      report_available: false,
      ingest_error: null,
      manifest: null,
      report: null,
      compute: [],
    };
    const ts = store(
      mockTunerEnv({
        launchRun: () => Effect.send(launched),
        getRunLog: () =>
          Effect.send({ lines: ["cohort 0 starting"], next_offset: 17, err_lines: [] }),
        listRuns: () => Effect.send([runView({ run_id: "fresh", status: "exited" })]),
        getProjectionRun: () => Effect.send(detail),
      }),
    );

    ts.send({ tag: "launch", request: launchRequest() }, (s) => {
      s.launch = { status: "pending", error: null, lastRunId: null };
    });
    ts.receive({ tag: "launchOk", run: launched }, (s) => {
      s.launch = { status: "done", error: null, lastRunId: "fresh" };
      s.runs = ok([launched]);
      s.openRunId = "fresh";
      s.logGeneration = 1;
      s.journalGeneration = 1;
      s.resourceGeneration = 1;
      s.projectionDetail = { status: "loading" };
      s.validation = { status: "loading" };
      s.candidates = { status: "loading" };
      s.report = { status: "loading" };
      s.log = { lines: [], errLines: [], offset: 0, error: null, active: true };
    });

    ts.receive({ tag: "logTick", generation: 1 });
    ts.receive({ tag: "journalTick", generation: 1 });
    ts.receive({ tag: "detailLoaded", generation: 1, detail }, (s) => {
      s.projectionDetail = ok(detail);
    });
    ts.receive(
      { tag: "validationLoaded", generation: 1, validation: { rows: [], unresolved_ties: null } },
      (s) => {
        s.validation = ok({ rows: [], unresolved_ties: null });
      },
    );
    ts.receive({ tag: "candidatesLoaded", generation: 1, candidates: [] }, (s) => {
      s.candidates = ok([]);
    });
    ts.receive({ tag: "reportLoaded", generation: 1, report: {} }, (s) => {
      s.report = ok({});
    });

    ts.receive(
      {
        tag: "logLoaded",
        generation: 1,
        lines: ["cohort 0 starting"],
        errLines: [],
        nextOffset: 17,
      },
      (s) => {
        s.log = {
          lines: ["cohort 0 starting"],
          errLines: [],
          offset: 17,
          error: null,
          active: true,
        };
      },
    );
    ts.receive(
      { tag: "runsLoaded", runs: [runView({ run_id: "fresh", status: "exited" })] },
      (s) => {
        s.runs = ok([runView({ run_id: "fresh", status: "exited" })]);
      },
    );
    ts.receive({ tag: "refreshProjection" }, (s) => {
      s.refreshing = true;
    });
    ts.receive({ tag: "refreshDone" }, (s) => {
      s.refreshing = false;
      s.lastProjectionRefreshAt = expect.any(Number) as unknown as number;
      s.resourceGeneration = 2;
      s.projectionDetail = { status: "loading", previous: detail };
      s.validation = { status: "loading", previous: { rows: [], unresolved_ties: null } };
      s.candidates = { status: "loading", previous: [] };
      s.report = { status: "loading", previous: {} };
    });
    ts.receive({ tag: "projectionLoaded", runs: [] });
    ts.receive({ tag: "detailLoaded", generation: 2, detail }, (s) => {
      s.projectionDetail = ok(detail);
    });
    ts.receive(
      { tag: "validationLoaded", generation: 2, validation: { rows: [], unresolved_ties: null } },
      (s) => {
        s.validation = ok({ rows: [], unresolved_ties: null });
      },
    );
    ts.receive({ tag: "candidatesLoaded", generation: 2, candidates: [] }, (s) => {
      s.candidates = ok([]);
    });
    ts.receive({ tag: "reportLoaded", generation: 2, report: {} }, (s) => {
      s.report = ok({});
    });
    expect(LOG_TAIL_MS).toBe(3000);
  });
});

function launchRequest() {
  return {
    game_kind: "nim",
    objective_key: "nim-v1",
    run_id: "fresh",
    task_seed: 1,
    tuning_pair_budget: 24,
    validation_pair_budget: 24,
    production_validation_pairs: 6,
  };
}
