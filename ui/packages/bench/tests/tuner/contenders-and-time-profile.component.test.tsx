// contenders-and-time-profile.component.test.tsx — 15g UI slices against a
// real `createStore(tunerReducer, env)` with a mocked env (AGENTS.md): the
// "All contenders" table on RunOverview, the CandidateDrawer record block,
// and the RunScience "Time profile" section (present + empty state).

import { afterEach, describe, expect, it, vi } from "vitest";
import { cleanup, fireEvent, render, screen } from "@solidjs/testing-library";
import { Effect, createStore } from "@mcts/core";
import {
  initialTunerState,
  tunerReducer,
  type TunerAction,
  type TunerState,
} from "../../src/tuner/tuner-reducer.js";
import { RunOverview } from "../../src/tuner/views/RunOverview.js";
import { RunScience } from "../../src/tuner/views/RunScience.js";
import { CandidateDrawer } from "../../src/tuner/views/CandidateDrawer.js";
import { mockTunerEnv, runView } from "./mock-tuner-env.js";
import type { TunerEnv } from "../../src/tuner/tuner-env.js";
import type {
  ProjectionCandidate,
  ProjectionContenderRow,
  ProjectionTelemetry,
} from "../../src/tuner/tuner-types.js";

afterEach(cleanup);

const candidates: ProjectionCandidate[] = [
  {
    candidate_id: "candidate-winner00000000",
    fingerprint: "f1",
    canonical_config: {},
    cohort_index: 0,
    cohort_slot: 0,
    source: "schema_default",
    parent_candidate_id: null,
  },
  {
    candidate_id: "candidate-loser000000000",
    fingerprint: "f2",
    canonical_config: {},
    cohort_index: 0,
    cohort_slot: 1,
    source: "smac",
    parent_candidate_id: null,
  },
];

const contenders: ProjectionContenderRow[] = [
  {
    candidate_id: "candidate-winner00000000",
    opponent_id: "op-a",
    phase: "tuning",
    wins: 4,
    losses: 1,
    draws: 0,
    sum_utility: 3,
  },
  {
    candidate_id: "candidate-loser000000000",
    opponent_id: "op-a",
    phase: "tuning",
    wins: 1,
    losses: 4,
    draws: 0,
    sum_utility: -3,
  },
];

const telemetry = (over: Partial<ProjectionTelemetry> = {}): ProjectionTelemetry => ({
  run_id: "r1",
  sessions: 1,
  first_start_us: 1_000_000,
  last_end_us: 11_000_000,
  wall_span_us: 10_000_000,
  loop_active_us: 10_000_000,
  wait_us: 4_000_000,
  lanes: [
    { name: "session", span_count: 1, total_us: 0, max_us: 0, first_start_us: 1_000_000, last_end_us: 1_000_000 },
    { name: "fold", span_count: 5, total_us: 3_000_000, max_us: 900_000, first_start_us: 1_000_000, last_end_us: 6_000_000 },
    { name: "wait", span_count: 5, total_us: 4_000_000, max_us: 1_500_000, first_start_us: 3_000_000, last_end_us: 11_000_000 },
  ],
  ...over,
});

function mount(view: "overview" | "science", env: TunerEnv) {
  const store = createStore<TunerState, TunerAction, TunerEnv>(
    initialTunerState(),
    tunerReducer,
    env,
  );
  store.dispatch({ tag: "runsLoaded", runs: [runView({ run_id: "r1", status: "exited" })] });
  store.dispatch({ tag: "openRun", runId: "r1" });
  const navigate = vi.fn();
  if (view === "overview") {
    render(() => <RunOverview store={store} runId="r1" navigate={navigate} />);
  } else {
    render(() => <RunScience store={store} runId="r1" navigate={navigate} />);
  }
  return { store, navigate };
}

describe("RunOverview — All contenders", () => {
  it("lists every evaluated candidate and row-click opens the drawer", async () => {
    const { navigate } = mount(
      "overview",
      mockTunerEnv({
        getProjectionContenders: () => Effect.send(contenders),
        getProjectionCandidates: () => Effect.send(candidates),
      }),
    );
    await vi.waitFor(() => expect(screen.getByTestId("run-contenders")).toBeInTheDocument());
    const table = screen.getByTestId("contenders-table");
    expect(table.textContent).toContain("winner000000");
    expect(table.textContent).toContain("loser0000000");
    // Winner (mean utility +0.6) sorts above loser (-0.6).
    const firstRow = table.querySelectorAll("tbody tr")[0]!;
    expect(firstRow.textContent).toContain("winner000000");
    expect(firstRow.textContent).toContain("4 / 0 / 1");

    fireEvent.click(firstRow);
    expect(navigate).toHaveBeenCalledWith(
      expect.objectContaining({ candidate: "candidate-winner00000000" }),
    );
  });
});

describe("CandidateDrawer — record block", () => {
  it("shows the per-opponent W/D/L for the selected candidate", async () => {
    const store = createStore<TunerState, TunerAction, TunerEnv>(
      initialTunerState(),
      tunerReducer,
      mockTunerEnv({
        getProjectionContenders: () => Effect.send(contenders),
        getProjectionCandidates: () => Effect.send(candidates),
      }),
    );
    store.dispatch({ tag: "runsLoaded", runs: [runView({ run_id: "r1", status: "exited" })] });
    store.dispatch({ tag: "openRun", runId: "r1" });
    await vi.waitFor(() =>
      expect(store.getState()().contenders.status).toBe("ok"),
    );
    render(() => (
      <CandidateDrawer
        store={store}
        candidateId="candidate-winner00000000"
        onClose={() => {}}
      />
    ));
    const block = await screen.findByTestId("candidate-record");
    expect(block.textContent).toContain("4 / 0 / 1");
    expect(block.textContent).toContain("op-a");
  });
});

describe("RunScience — Time profile", () => {
  it("renders the lanes, duty-cycle headline and stage treemap", async () => {
    mount(
      "science",
      mockTunerEnv({ getProjectionTelemetry: () => Effect.send(telemetry()) }),
    );
    await vi.waitFor(() =>
      expect(screen.getByTestId("science-time-profile")).toBeInTheDocument(),
    );
    expect(screen.getByTestId("time-profile-headline").textContent).toContain(
      "run loop 60% · waiting on games 40%",
    );
    expect(screen.getByTestId("timeline-lanes")).toBeInTheDocument();
    expect(screen.getByTestId("time-profile-treemap")).toBeInTheDocument();
  });

  it("shows the empty state when the run has no telemetry", async () => {
    mount(
      "science",
      mockTunerEnv({
        getProjectionTelemetry: () => Effect.send(telemetry({ lanes: [], wall_span_us: 0, loop_active_us: 0, wait_us: 0, first_start_us: null, last_end_us: null })),
      }),
    );
    await vi.waitFor(() =>
      expect(screen.getByTestId("science-time-profile")).toBeInTheDocument(),
    );
    expect(screen.getByTestId("science-time-profile").textContent).toContain(
      "No timing recorded for this run",
    );
  });
});
