// Shared mocked `TunerEnv` for the tuner reducer/component tests — every
// method returns a synchronous `Effect.send`, so no test touches the
// network (AGENTS.md "mock the environment").

import { Effect } from "@mcts/core";
import type { TunerEnv } from "../../src/tuner/tuner-env.js";
import type { TunerRunView } from "../../src/tuner/tuner-types.js";

export function runView(over: Partial<TunerRunView> = {}): TunerRunView {
  return {
    run_id: "r1",
    argv: ["uv"],
    run_dir: "/runs/r1",
    pid: 4242,
    started_at: "2026-01-01T00:00:00Z",
    terminal_outcome: null,
    status: "live",
    ...over,
  };
}

export function mockTunerEnv(over: Partial<TunerEnv> = {}): TunerEnv {
  const base: TunerEnv = {
    listKinds: () => Effect.send([]),
    listObjectives: () => Effect.send([]),
    getObjective: () =>
      Effect.send({ key: "o1", content: {}, updated_at: null, is_seed: false }),
    putObjective: (key, content) =>
      Effect.send({ key, content, updated_at: null, is_seed: false }),
    deleteObjective: () => Effect.send(undefined),
    validateObjective: () => Effect.send({ ok: true, errors: [] }),
    listRuns: () => Effect.send([]),
    getRun: () => Effect.send(runView()),
    launchRun: () => Effect.send(runView()),
    preflightRun: () => Effect.send({ ok: true, errors: [] }),
    stopRun: () => Effect.send(runView({ status: "exited" })),
    extendRun: () => Effect.send(runView()),
    getRunLog: () => Effect.send({ lines: [], next_offset: 0, err_lines: [] }),
    getRunEvidence: () => Effect.send({ events: [], next_seq: 0, run_status: "live" }),
    // Default: an evidence stream that opens and immediately closes (no
    // events, no terminal action). Tests that exercise the ticker override
    // this with a scripted `Effect.stream`.
    openEvidenceStream: () => Effect.stream((_send, done) => done()),
    refreshProjection: () =>
      Effect.send({ projected: 0, skipped: 0, ingest_errors: 0, pruned: 0 }),
    listProjectionRuns: () => Effect.send([]),
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
    getProjectionCohorts: () => Effect.send([]),
    getProjectionCandidates: () => Effect.send([]),
    getProjectionCandidate: () =>
      Effect.send({
        candidate_id: "c1",
        fingerprint: "f",
        canonical_config: {},
        cohort_index: 0,
        cohort_slot: 0,
        source: "schema_default",
        parent_candidate_id: null,
      }),
    getProjectionPairs: () => Effect.send([]),
    getProjectionPairGames: () => Effect.send([]),
    getProjectionValidation: () => Effect.send({ rows: [], unresolved_ties: null }),
    getProjectionReport: () => Effect.send({}),
  };
  return { ...base, ...over };
}
