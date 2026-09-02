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

const scienceReport = {
  validation_claim: { claim: "mechanics_smoke", missing_production_axes: [] },
  limitations: [],
  unresolved_ties: [],
  proposal_search: {
    configured: { bootstrap: 2, model: 2, random_reserve: 2, cohorts: 2, retained_elites: 2 },
    actual_source_attempts: { schema_default: 1, smac_model: 3 },
    rejections_by_source: { smac_model: 1 },
    accepted: [{ source: "schema_default" }, { source: "smac_model" }, { source: "smac_model" }],
    model_version: "smac-2.4-public-ask-v1",
    final_observation_count: 4,
    final_frontier_id: "frontier-deadbeef01234567",
  },
  shadow_elimination: {
    policy: { enforced: false, kind: "paired_bootstrap" },
    cohorts: [
      {
        cohort_index: 0,
        candidate_paths: [
          {
            candidate_id: "candidate-aaaa1111",
            final_top_set: true,
            looks: [{ prefix_id: "prefix-p6", disposition: "continue", maximum_mean_difference: 0.5 }],
          },
        ],
      },
    ],
  },
};

describe("TunerApp run science", () => {
  it("renders the convergence, proposal funnel, and cohort race from the report", async () => {
    render(() => (
      <TunerApp
        env={mockTunerEnv({
          ...completedRunEnv(),
          getProjectionReport: () => Effect.send(scienceReport),
        })}
      />
    ));

    await vi.waitFor(() => expect(screen.getByText("done-1")).toBeInTheDocument());
    fireEvent.click(screen.getByText("done-1"));

    await vi.waitFor(() => expect(screen.getByTestId("tuner-run-overview")).toBeInTheDocument());
    fireEvent.click(screen.getByRole("button", { name: "Full science →" }));

    await vi.waitFor(() => expect(screen.getByTestId("tuner-run-science")).toBeInTheDocument());
    expect(screen.getByTestId("science-convergence")).toBeInTheDocument();
    expect(screen.getByTestId("funnel-bars")).toBeInTheDocument();
    expect(screen.getByTestId("kpi-row")).toHaveTextContent("smac-2.4-public-ask-v1");
    expect(screen.getByTestId("race-strip")).toBeInTheDocument();

    fireEvent.click(screen.getAllByRole("button", { name: "Show numbers" })[0]!);
    await vi.waitFor(() =>
      expect(screen.getByTestId("convergence-numbers")).toBeInTheDocument(),
    );
  });
});

const sciencePart2Report = {
  validation_claim: { claim: "mechanics_smoke", missing_production_axes: [] },
  limitations: [],
  unresolved_ties: [],
  proposal_search: { actual_source_attempts: { schema_default: 1 }, accepted: [{ source: "schema_default" }] },
  shadow_elimination: {
    policy: { policy_kind: "paired_bootstrap", policy_version: "pb-v1" },
    summary: {
      counterfactual_eliminations: 1,
      top_set_false_eliminations: 0,
      top_set_false_elimination_rate: 0.0,
      trash_precision: null,
      true_trash_eliminations: 0,
      brier_score: 0.05,
    },
    scope: { completed_cohorts: 2, recorded_looks: 4, active_path_looks: 4, held_out_validation_used: false },
    strata: [{ reversals: 0, elimination_reversals: 0 }],
    calibration_bins: [
      { lower: 0.8, upper: 1.0, mean_prediction: 0.95, observed_success_rate: 0.6, count: 5 },
    ],
    cohorts: [],
  },
  opponent_response_analysis: {
    scope: { opponent_ids: ["schema-default", "historical"], interval_method: "hoeffding_pair_bound_v1" },
    candidates: [
      {
        candidate_id: "candidate-aaaa1111",
        opponent_responses: [
          { opponent_id: "schema-default", mean: 0.7, interval: { lower: 0.4, upper: 0.9 } },
          { opponent_id: "historical", mean: 0.4, interval: { lower: 0.1, upper: 0.7 } },
        ],
      },
    ],
    pairwise_interactions: [],
  },
  diagnostic_matchup_graph: {
    scope: { pair_attempt_budget: 12, search_effort: { kind: "iterations", value: 3 } },
    allocations: { count: 4 },
    nodes: [
      { candidate_id: "candidate-aaaa1111", objective_rank: 0 },
      { candidate_id: "candidate-bbbb2222", objective_rank: 1 },
    ],
    edges: [
      {
        left_candidate_id: "candidate-aaaa1111",
        right_candidate_id: "candidate-bbbb2222",
        material_direction: "left",
        estimate: 0.3,
        interval: { lower: 0.1, upper: 0.5 },
        pair_count: 4,
      },
    ],
    material_cycle_components: [],
    shortlist_effect: {},
  },
  compute: {
    policy_version: "safe-boundary-pair-attempts-v1",
    budget: { tuning_pair_attempts: 84, validation_pair_attempts: 4, diagnostic_pair_attempts: 12 },
    tuning: { pair_attempts: 84, completed_pairs: 82, failed_attempts: 1, censored_attempts: 1, overrun_pair_attempts: 0, unspent_pair_attempts: 0, physical_games: 168, search_iterations: 336, wall_time_ms: 168 },
    validation: { pair_attempts: 4, completed_pairs: 4, failed_attempts: 0, censored_attempts: 0, overrun_pair_attempts: 0, unspent_pair_attempts: 0, physical_games: 8, search_iterations: 16, wall_time_ms: 8 },
    diagnostic: { pair_attempts: 12, completed_pairs: 12, failed_attempts: 0, censored_attempts: 0, overrun_pair_attempts: 0, unspent_pair_attempts: 0, physical_games: 24, search_iterations: 36, wall_time_ms: 24 },
  },
};

describe("TunerApp run evidence", () => {
  it("lists candidates and pairs and opens the pair inspector with game summaries", async () => {
    const getProjectionPairGames = vi.fn(() =>
      Effect.send([
        {
          game_id: "game-a",
          pair_id: "pair-777aaa",
          candidate_side: "first",
          outcome: "candidate_win",
          plies: 9,
          elapsed_ms: 1200,
          candidate_iterations_total: 3000,
          opponent_iterations_total: 2800,
        },
        {
          game_id: "game-b",
          pair_id: "pair-777aaa",
          candidate_side: "second",
          outcome: "draw",
          plies: 11,
          elapsed_ms: 1400,
          candidate_iterations_total: 3100,
          opponent_iterations_total: 2900,
        },
      ]),
    );
    render(() => (
      <TunerApp
        env={mockTunerEnv({
          ...completedRunEnv(),
          getProjectionPairs: () =>
            Effect.send([
              {
                pair_id: "pair-777aaa",
                phase: "tuning",
                candidate_id: "candidate-aaaa1111",
                task_id: "task-1",
                opponent_id: "schema-default",
                pair_utility: 0.4,
              },
            ]),
          getProjectionPairGames,
        })}
      />
    ));

    await vi.waitFor(() => expect(screen.getByText("done-1")).toBeInTheDocument());
    fireEvent.click(screen.getByText("done-1"));
    await vi.waitFor(() => expect(screen.getByTestId("tuner-run-overview")).toBeInTheDocument());
    fireEvent.click(screen.getByRole("button", { name: "Raw evidence →" }));

    await vi.waitFor(() => expect(screen.getByTestId("tuner-run-evidence")).toBeInTheDocument());
    expect(screen.getByTestId("evidence-candidates")).toHaveTextContent("smac_model");
    expect(screen.getByTestId("evidence-pairs")).toHaveTextContent("schema-default");

    fireEvent.click(screen.getByText("777aaa"));
    await vi.waitFor(() => expect(getProjectionPairGames).toHaveBeenCalledWith("done-1", "pair-777aaa"));
    await vi.waitFor(() =>
      expect(screen.getByTestId("game-summary-strip")).toHaveTextContent("Candidate win"),
    );
    expect(screen.getByTestId("pair-kpis")).toHaveTextContent("1 / 1 / 0");
  });
});

describe("TunerApp run science part 2", () => {
  it("renders elimination calibration, opponent response, diagnostic graph, and compute", async () => {
    render(() => (
      <TunerApp
        env={mockTunerEnv({
          ...completedRunEnv(),
          getProjectionReport: () => Effect.send(sciencePart2Report),
        })}
      />
    ));

    await vi.waitFor(() => expect(screen.getByText("done-1")).toBeInTheDocument());
    fireEvent.click(screen.getByText("done-1"));
    await vi.waitFor(() => expect(screen.getByTestId("tuner-run-overview")).toBeInTheDocument());
    fireEvent.click(screen.getByRole("button", { name: "Full science →" }));
    await vi.waitFor(() => expect(screen.getByTestId("tuner-run-science")).toBeInTheDocument());

    expect(screen.getByTestId("calibration-heatmap")).toBeInTheDocument();
    expect(screen.getByTestId("elimination-kpis")).toHaveTextContent("Brier score");
    expect(screen.getByTestId("opponent-heatmap")).toBeInTheDocument();
    expect(screen.getByTestId("cycle-graph")).toBeInTheDocument();
    expect(screen.getByTestId("treemap")).toBeInTheDocument();
    expect(screen.getByTestId("compute-kpis")).toHaveTextContent("200");
  });
});
