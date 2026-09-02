// tests/TunerApp.test.tsx — component test for the version-4 tuner UI:
// a real `createStore(tunerReducer, env)` inside `<TunerApp>` driven by a
// mocked `TunerEnv` (no live server, no real timers relied on). Mirrors
// `BenchApp.test.tsx` / `GameShell.test.tsx`.

import { afterEach, describe, expect, it, vi } from "vitest";
import { cleanup, fireEvent, render, screen } from "@solidjs/testing-library";
import { Effect } from "@mcts/core";
import { TunerApp } from "@mcts/bench";
import { mockTunerEnv, runView } from "../packages/bench/tests/tuner/mock-tuner-env.js";
import type { TunerLaunchRequest } from "@mcts/bench";

afterEach(() => {
  cleanup();
  window.location.hash = "";
});

const kinds = [
  {
    game: "nim",
    tuner: {
      id: "strategy",
      baselines: ["strong"],
      eval_rounds: 20,
      parameters: [],
      conditions: [],
      game_config: {},
    },
  },
];
const objectives = [{ key: "nim-v1", objective_id: "nim-v1", game_kind: "nim" }];

describe("TunerApp fleet", () => {
  it("renders the KPI row and a completed run from the projection", async () => {
    render(() => (
      <TunerApp
        env={mockTunerEnv({
          listKinds: () => Effect.send(kinds),
          listObjectives: () => Effect.send(objectives),
          listRuns: () => Effect.send([]),
          listProjectionRuns: () =>
            Effect.send([
              {
                run_id: "done-1",
                terminal_status: "exited",
                report_available: true,
                ingest_error: null,
                game_kind: "nim",
                objective_id: "nim-v1",
                shadow_policy_kind: "paired_bootstrap",
                active_elimination: false,
                report_status: "complete",
                validation_claim: "production validation",
                total_pair_attempts: 240,
                total_completed_pairs: 236,
              },
            ]),
        })}
      />
    ));

    await vi.waitFor(() =>
      expect(screen.getByTestId("kpi-complete")).toHaveTextContent("1 complete"),
    );
    expect(screen.getByText("done-1")).toBeInTheDocument();
    expect(screen.getByText("production validation")).toBeInTheDocument();
  });

  it("launches a run and navigates to its overview", async () => {
    const launchRun = vi.fn((_req: TunerLaunchRequest) =>
      Effect.send(runView({ run_id: "launched-1", status: "live" })),
    );
    render(() => (
      <TunerApp
        env={mockTunerEnv({
          listKinds: () => Effect.send(kinds),
          listObjectives: () => Effect.send(objectives),
          listRuns: () => Effect.send([]),
          launchRun,
          getRunLog: () => Effect.send({ lines: ["booting"], next_offset: 7, err_lines: [] }),
        })}
      />
    ));

    await vi.waitFor(() => expect(screen.getByTestId("tuner-fleet")).toBeInTheDocument());
    fireEvent.click(screen.getByRole("button", { name: "New run" }));

    await vi.waitFor(() => expect(screen.getByText("Launch a tuner run")).toBeInTheDocument());
    fireEvent.click(screen.getByRole("button", { name: "Launch" }));

    await vi.waitFor(() => expect(launchRun).toHaveBeenCalledTimes(1));
    expect(launchRun.mock.calls[0]![0]).toMatchObject({
      game_kind: "nim",
      objective_key: "nim-v1",
    });

    await vi.waitFor(() =>
      expect(screen.getByTestId("tuner-run-overview")).toHaveTextContent("launched-1"),
    );
    await vi.waitFor(() =>
      expect(screen.getByTestId("tuner-log-lines")).toHaveTextContent("booting"),
    );
  });
});

const projectionRun = {
  run_id: "done-1",
  terminal_status: "exited",
  report_available: true,
  ingest_error: null,
  game_kind: "nim",
  objective_id: "nim-v1",
  shadow_policy_kind: "paired_bootstrap",
  active_elimination: false,
  report_status: "complete",
  validation_claim: "mechanics_smoke",
  total_pair_attempts: 24,
  total_completed_pairs: 24,
};

function completedRunEnv() {
  return mockTunerEnv({
    listKinds: () => Effect.send(kinds),
    listObjectives: () => Effect.send(objectives),
    listRuns: () => Effect.send([]),
    listProjectionRuns: () => Effect.send([projectionRun]),
    getProjectionRun: () =>
      Effect.send({
        run_id: "done-1",
        terminal_status: "exited",
        report_available: true,
        ingest_error: null,
        manifest: {
          manifest_run_id: "done-1",
          manifest_fingerprint: "mf",
          game_kind: "nim",
          objective_id: "nim-v1",
          cohort_size: 4,
          finalists: 2,
          seed: 1,
          task_seed: 1,
          shadow_policy_kind: "paired_bootstrap",
          active_elimination: false,
        },
        report: { schema_version: 4, status: "complete", validation_claim: "mechanics_smoke" },
        compute: [],
      }),
    getProjectionValidation: () =>
      Effect.send({
        rows: [
          {
            candidate_id: "candidate-aaaa1111",
            rank: 1,
            estimate: 0.5,
            lower: 0.1,
            upper: 0.9,
            wins: 3,
            draws: 1,
            losses: 0,
          },
          {
            candidate_id: "candidate-bbbb2222",
            rank: 2,
            estimate: 0.2,
            lower: -0.1,
            upper: 0.5,
            wins: 1,
            draws: 2,
            losses: 1,
          },
        ],
        unresolved_ties: [
          { left_candidate_id: "candidate-aaaa1111", right_candidate_id: "candidate-bbbb2222" },
        ],
      }),
    getProjectionCandidates: () =>
      Effect.send([
        {
          candidate_id: "candidate-aaaa1111",
          fingerprint: "aaaa1111",
          canonical_config: { family: "b" },
          cohort_index: 0,
          cohort_slot: 0,
          source: "smac_model",
          parent_candidate_id: null,
        },
      ]),
    getProjectionReport: () =>
      Effect.send({
        validation_claim: { claim: "mechanics_smoke", missing_production_axes: ["search_effort"] },
        limitations: ["default-only starting state"],
        unresolved_ties: [],
      }),
  });
}

describe("TunerApp run overview", () => {
  it("shows the ship verdict, validation ranking, and caveats for a completed run", async () => {
    render(() => <TunerApp env={completedRunEnv()} />);

    await vi.waitFor(() => expect(screen.getByText("done-1")).toBeInTheDocument());
    fireEvent.click(screen.getByText("done-1"));

    await vi.waitFor(() => expect(screen.getByTestId("ship-verdict")).toBeInTheDocument());
    expect(screen.getByTestId("ship-caveats")).toHaveTextContent("Mechanics smoke");
    expect(screen.getByTestId("ship-caveats")).toHaveTextContent("search effort");
    expect(screen.getByTestId("ship-ties")).toHaveTextContent(
      "Cannot distinguish aaaa1111 from bbbb2222",
    );
    expect(screen.getByTestId("validation-wdl")).toHaveTextContent("3 / 1 / 0");
  });

  it("opens the candidate drawer from a chip and closes it", async () => {
    render(() => <TunerApp env={completedRunEnv()} />);

    await vi.waitFor(() => expect(screen.getByText("done-1")).toBeInTheDocument());
    fireEvent.click(screen.getByText("done-1"));

    await vi.waitFor(() =>
      expect(screen.getAllByTestId("candidate-chip").length).toBeGreaterThan(0),
    );
    fireEvent.click(screen.getAllByTestId("candidate-chip")[0]!);

    await vi.waitFor(() => expect(screen.getByTestId("candidate-drawer")).toBeInTheDocument());
    expect(screen.getByTestId("candidate-drawer")).toHaveTextContent("smac_model");
    expect(window.location.hash).toContain("candidate=candidate-aaaa1111");

    fireEvent.click(screen.getByRole("button", { name: "Close" }));
    await vi.waitFor(() =>
      expect(screen.queryByTestId("candidate-drawer")).not.toBeInTheDocument(),
    );
  });
});
