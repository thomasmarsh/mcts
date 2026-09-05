// tests/TunerApp.test.tsx — component test for the version-4 tuner UI:
// a real `createStore(tunerReducer, env)` inside `<TunerApp>` driven by a
// mocked `TunerEnv` (no live server, no real timers relied on). Mirrors
// `GameShell.test.tsx`.

import { afterEach, describe, expect, it, vi } from "vitest";
import { cleanup, fireEvent, render, screen, within } from "@solidjs/testing-library";
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
const objectives = [
  {
    key: "nim-v1",
    objective_id: "nim-v1",
    game_kind: "nim",
    opponent_count: 2,
    updated_at: null,
    is_seed: false,
  },
];

describe("TunerApp fleet", () => {
  it("renders the KPI row and a completed run from the projection", async () => {
    render(() => (
      <TunerApp
        env={mockTunerEnv({
          listTunableGames: () => Effect.send(kinds),
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

  it("shows a run that died on startup as failed, with its launch.err", async () => {
    render(() => (
      <TunerApp
        env={mockTunerEnv({
          listTunableGames: () => Effect.send(kinds),
          listObjectives: () => Effect.send(objectives),
          listRuns: () =>
            Effect.send([
              runView({
                run_id: "doomed-1",
                status: "failed",
                terminal_outcome: "exited",
                error_detail: "tuner failed: run directory already exists",
              }),
            ]),
          listProjectionRuns: () => Effect.send([]),
        })}
      />
    ));

    await vi.waitFor(() => expect(screen.getByText("Failed to start")).toBeInTheDocument());
    expect(screen.getByText("doomed-1")).toBeInTheDocument();
    expect(screen.getByText("failed to start")).toBeInTheDocument();
    expect(
      screen.getByText(/tuner failed: run directory already exists/),
    ).toBeInTheDocument();
    expect(screen.getByTestId("kpi-failed")).toHaveTextContent("1 failed");
  });

  it("pre-checks the launch and blocks it when the config is invalid", async () => {
    const launchRun = vi.fn(() => Effect.send(runView({ run_id: "x", status: "live" })));
    render(() => (
      <TunerApp
        env={mockTunerEnv({
          listTunableGames: () => Effect.send(kinds),
          listObjectives: () => Effect.send(objectives),
          listRuns: () => Effect.send([]),
          launchRun,
          preflightRun: () =>
            Effect.send({
              ok: false,
              errors: ["validation pairs cannot exceed production validation pairs"],
            }),
        })}
      />
    ));

    await vi.waitFor(() => expect(screen.getByTestId("tuner-fleet")).toBeInTheDocument());
    fireEvent.click(screen.getByRole("button", { name: "New run" }));
    await vi.waitFor(() => expect(screen.getByText("Launch a tuner run")).toBeInTheDocument());

    await vi.waitFor(() =>
      expect(screen.getByTestId("preflight-errors")).toHaveTextContent(
        "validation pairs cannot exceed production validation pairs",
      ),
    );
    expect(screen.getByRole("button", { name: "Launch" })).toBeDisabled();
    fireEvent.click(screen.getByRole("button", { name: "Launch" }));
    expect(launchRun).not.toHaveBeenCalled();
  });

  it("launches a run and navigates to its overview", async () => {
    const launchRun = vi.fn((_req: TunerLaunchRequest) =>
      Effect.send(runView({ run_id: "launched-1", status: "live" })),
    );
    render(() => (
      <TunerApp
        env={mockTunerEnv({
          listTunableGames: () => Effect.send(kinds),
          listObjectives: () => Effect.send(objectives),
          listRuns: () => Effect.send([]),
          launchRun,
          getRunLog: () => Effect.send({ lines: ["booting"], next_offset: 7, err_lines: [], err_next_offset: 0 }),
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
    listTunableGames: () => Effect.send(kinds),
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
          canonical_config: { select: "b" },
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

describe("TunerApp objective manager", () => {
  const objectiveRow = (key: string) => ({
    key,
    objective_id: `${key}-id`,
    game_kind: "nim",
    opponent_count: 2,
    updated_at: "2026-09-02T00:00:00Z",
    is_seed: false,
  });

  it("lists objectives and removes a row after a confirmed delete", async () => {
    let corpus = [objectiveRow("nim-v1"), objectiveRow("nim-v2")];
    const deleteObjective = vi.fn((_key: string) => Effect.send(undefined));
    render(() => (
      <TunerApp
        env={mockTunerEnv({
          listTunableGames: () => Effect.send(kinds),
          listObjectives: () => Effect.send(corpus),
          listRuns: () => Effect.send([]),
          deleteObjective: (key) => {
            corpus = corpus.filter((o) => o.key !== key);
            return deleteObjective(key);
          },
        })}
      />
    ));

    await vi.waitFor(() => expect(screen.getByTestId("tuner-fleet")).toBeInTheDocument());
    fireEvent.click(screen.getByRole("button", { name: "Manage objectives" }));

    await vi.waitFor(() =>
      expect(screen.getByTestId("tuner-objective-manager")).toBeInTheDocument(),
    );
    expect(screen.getByText("nim-v1")).toBeInTheDocument();
    expect(screen.getByText("nim-v2")).toBeInTheDocument();

    const firstRow = screen.getByText("nim-v1").closest("tr")!;
    fireEvent.click(within(firstRow).getByRole("button", { name: "Delete" }));
    fireEvent.click(within(firstRow).getByRole("button", { name: "Confirm delete" }));

    await vi.waitFor(() => expect(deleteObjective).toHaveBeenCalledWith("nim-v1"));
    await vi.waitFor(() => expect(screen.queryByText("nim-v1")).not.toBeInTheDocument());
    expect(screen.getByText("nim-v2")).toBeInTheDocument();
  });

  it("opens the editor for an existing objective", async () => {
    render(() => (
      <TunerApp
        env={mockTunerEnv({
          listTunableGames: () => Effect.send(kinds),
          listObjectives: () => Effect.send([objectiveRow("nim-v1")]),
          listRuns: () => Effect.send([]),
          getObjective: () =>
            Effect.send({
              key: "nim-v1",
              content: { schema_version: 1, objective_id: "nim-v1" },
              updated_at: null,
              is_seed: false,
            }),
        })}
      />
    ));

    await vi.waitFor(() => expect(screen.getByTestId("tuner-fleet")).toBeInTheDocument());
    fireEvent.click(screen.getByRole("button", { name: "Manage objectives" }));
    await vi.waitFor(() =>
      expect(screen.getByTestId("tuner-objective-manager")).toBeInTheDocument(),
    );
    fireEvent.click(screen.getAllByRole("button", { name: "Edit" })[0]!);

    await vi.waitFor(() =>
      expect(screen.getByTestId("tuner-objective-editor")).toBeInTheDocument(),
    );
    await vi.waitFor(() =>
      expect(screen.getByTestId("objective-id-input")).toHaveValue("nim-v1"),
    );
  });
});

describe("TunerApp objective editor", () => {
  const existing = {
    schema_version: 1,
    objective_id: "nim-v1",
    game_kind: "nim",
    opponents: [
      {
        id: "schema-default",
        label: "Schema default",
        role: "default",
        weight: 2,
        config: { source: "schema_default" },
      },
      {
        id: "hist",
        label: "Historical",
        role: "historical_reference",
        weight: 4,
        config: { source: "inline", value: { c: 1.4 } },
      },
    ],
    start_distribution: { kind: "default_only" },
  };
  const objectiveRow = {
    key: "nim-v1",
    objective_id: "nim-v1",
    game_kind: "nim",
    opponent_count: 2,
    updated_at: null,
    is_seed: false,
  };

  async function openEditor(over: Parameters<typeof mockTunerEnv>[0]) {
    render(() => (
      <TunerApp
        env={mockTunerEnv({
          listTunableGames: () => Effect.send(kinds),
          listObjectives: () => Effect.send([objectiveRow]),
          listRuns: () => Effect.send([]),
          getObjective: () =>
            Effect.send({ key: "nim-v1", content: existing, updated_at: null, is_seed: false }),
          ...over,
        })}
      />
    ));
    await vi.waitFor(() => expect(screen.getByTestId("tuner-fleet")).toBeInTheDocument());
    fireEvent.click(screen.getByRole("button", { name: "Manage objectives" }));
    await vi.waitFor(() =>
      expect(screen.getByTestId("tuner-objective-manager")).toBeInTheDocument(),
    );
    fireEvent.click(screen.getAllByRole("button", { name: "Edit" })[0]!);
    await vi.waitFor(() =>
      expect(screen.getByTestId("tuner-objective-editor")).toBeInTheDocument(),
    );
  }

  it("saves the reduced-weight canonical body", async () => {
    const putObjective = vi.fn((key: string, content: unknown) =>
      Effect.send({ key, content, updated_at: null, is_seed: false }),
    );
    await openEditor({ putObjective });

    await vi.waitFor(() => expect(screen.getAllByTestId("objective-opponent").length).toBe(2));
    fireEvent.click(screen.getByRole("button", { name: "Save" }));

    await vi.waitFor(() => expect(putObjective).toHaveBeenCalledTimes(1));
    const [key, content] = putObjective.mock.calls[0]!;
    expect(key).toBe("nim-v1");
    expect((content as { opponents: { weight: number }[] }).opponents.map((o) => o.weight)).toEqual([
      1, 2,
    ]);
    // Save navigates back to the manager.
    await vi.waitFor(() =>
      expect(screen.getByTestId("tuner-objective-manager")).toBeInTheDocument(),
    );
  });

  it("disables Save and shows the reason for a bad draft", async () => {
    await openEditor({});
    const idInput = screen.getByTestId("objective-id-input") as HTMLInputElement;
    fireEvent.input(idInput, { target: { value: "" } });

    await vi.waitFor(() =>
      expect(screen.getByTestId("objective-validation")).toHaveTextContent(/objective id/i),
    );
    expect(screen.getByRole("button", { name: "Save" })).toBeDisabled();
  });

  it("renders a rejected server validation", async () => {
    await openEditor({
      validateObjective: () =>
        Effect.send({ ok: false, errors: ["objective needs at least two opponents"] }),
    });
    fireEvent.click(screen.getByRole("button", { name: "Validate on server" }));
    await vi.waitFor(() =>
      expect(screen.getByTestId("objective-server-validation")).toHaveTextContent(
        "at least two opponents",
      ),
    );
  });
});

describe("TunerApp objective editor — game setup", () => {
  const gameConfigKinds = [
    ...kinds,
    {
      game: "atarigo",
      tuner: {
        id: "strategy",
        baselines: ["strong"],
        eval_rounds: 20,
        parameters: [],
        conditions: [],
        game_config: { size: 13 },
        game_config_schema: {
          parameters: [{ name: "size", type: "int", bounds: [3, 19], default: 13 }],
          conditions: [],
        },
      },
    },
  ];

  async function openNewEditor(over: Parameters<typeof mockTunerEnv>[0] = {}) {
    render(() => (
      <TunerApp
        env={mockTunerEnv({
          listTunableGames: () => Effect.send(gameConfigKinds),
          listObjectives: () => Effect.send([]),
          listRuns: () => Effect.send([]),
          ...over,
        })}
      />
    ));
    await vi.waitFor(() => expect(screen.getByTestId("tuner-fleet")).toBeInTheDocument());
    fireEvent.click(screen.getByRole("button", { name: "Manage objectives" }));
    await vi.waitFor(() =>
      expect(screen.getByTestId("tuner-objective-manager")).toBeInTheDocument(),
    );
    fireEvent.click(screen.getByRole("button", { name: "New objective" }));
    await vi.waitFor(() =>
      expect(screen.getByTestId("tuner-objective-editor")).toBeInTheDocument(),
    );
  }

  it("shows the fixed-board caption until a configurable game is picked", async () => {
    await openNewEditor();
    expect(screen.getByTestId("objective-game-config")).toHaveTextContent(
      "This game's board is fixed",
    );

    fireEvent.input(screen.getByRole("combobox"), { target: { value: "nim" } });
    expect(screen.getByTestId("objective-game-config")).toHaveTextContent(
      "This game's board is fixed",
    );
  });

  it("renders a bounded size field for AtariGo and flags an out-of-bounds value", async () => {
    await openNewEditor();
    fireEvent.input(screen.getByRole("combobox"), { target: { value: "atarigo" } });

    const section = screen.getByTestId("objective-game-config");
    const sizeField = within(section).getByRole("spinbutton") as HTMLInputElement;
    expect(sizeField).toHaveAttribute("min", "3");
    expect(sizeField).toHaveAttribute("max", "19");

    fireEvent.input(sizeField, { target: { value: "25" } });
    await vi.waitFor(() =>
      expect(screen.getByTestId("objective-validation")).toHaveTextContent(/within \[3, 19\]/),
    );
    expect(screen.getByRole("button", { name: "Save" })).toBeDisabled();
  });

  it("keeps the size field mounted across keystrokes", async () => {
    await openNewEditor();
    fireEvent.input(screen.getByRole("combobox"), { target: { value: "atarigo" } });

    const section = screen.getByTestId("objective-game-config");
    const before = within(section).getByRole("spinbutton");
    fireEvent.input(before, { target: { value: "9" } });
    const after = within(screen.getByTestId("objective-game-config")).getByRole("spinbutton");
    expect(after).toBe(before);
  });

  it("keeps the objective-id field mounted across keystrokes", async () => {
    await openNewEditor();
    fireEvent.input(screen.getByRole("combobox"), { target: { value: "atarigo" } });
    const before = screen.getByTestId("objective-id-input");
    fireEvent.input(before, { target: { value: "foo" } });
    expect(screen.getByTestId("objective-id-input")).toBe(before);
  });

  it("threads game_config into the validated body", async () => {
    const validateObjective = vi.fn((_key: string, _content: unknown) =>
      Effect.send({ ok: true, errors: [] }),
    );
    await openNewEditor({ validateObjective });
    fireEvent.input(screen.getByRole("combobox"), { target: { value: "atarigo" } });

    const section = screen.getByTestId("objective-game-config");
    fireEvent.input(within(section).getByRole("spinbutton"), { target: { value: "9" } });

    fireEvent.click(screen.getByRole("button", { name: "Validate on server" }));
    await vi.waitFor(() => expect(validateObjective).toHaveBeenCalledTimes(1));
    expect(validateObjective.mock.calls[0]![1]).toMatchObject({ game_config: { size: 9 } });
  });
});

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

describe("TunerApp live run", () => {
  it("streams evidence into the ticker and advances the progress rail without a projection refresh", async () => {
    const events = [
      { sequence: 1, type: "proposal_created", payload: { source: "smac_model", candidate_id: "candidate-aaa111", cohort_index: 0 } },
      { sequence: 2, type: "pair_started", payload: { phase: "tuning", pair_id: "pair-abc123", candidate_id: "candidate-aaa111", opponent_id: "baseline" } },
      { sequence: 3, type: "pair_completed", payload: { phase: "tuning", pair_id: "pair-abc123", candidate_id: "candidate-aaa111", opponent_id: "baseline", pair_utility: 0.125 } },
      { sequence: 4, type: "cohort_completed", payload: { cohort_index: 0, candidate_ids: ["candidate-aaa111", "candidate-bbb222"], retained_candidate_ids: ["candidate-aaa111"] } },
    ];
    render(() => (
      <TunerApp
        env={mockTunerEnv({
          listTunableGames: () => Effect.send(kinds),
          listObjectives: () => Effect.send(objectives),
          listRuns: () => Effect.send([runView({ run_id: "live-1", status: "live" })]),
          listProjectionRuns: () => Effect.send([]),
          openEvidenceStream: () =>
            Effect.stream((send, done) => {
              send({ kind: "events", events });
              done();
            }),
        })}
      />
    ));

    await vi.waitFor(() => expect(screen.getByText("live-1")).toBeInTheDocument());
    fireEvent.click(screen.getByText("live-1"));

    await vi.waitFor(() => expect(screen.getByTestId("tuner-run-overview")).toBeInTheDocument());

    // The ticker shows the formatted lines off the raw evidence payloads.
    await vi.waitFor(() =>
      expect(screen.getByTestId("event-ticker")).toHaveTextContent(
        "pair abc123 done · aaa111 vs baseline · +0.125",
      ),
    );
    expect(screen.getByTestId("event-ticker")).toHaveTextContent(
      "cohort 0 complete — 1 promoted, 1 eliminated",
    );

    // The progress rail advances from the fold alone (the projection has no
    // compute ledger for this run yet).
    expect(screen.getByTestId("progress-cohort")).toHaveTextContent("cohort 0");
    expect(screen.getByTestId("progress-live-counts")).toHaveTextContent("1 pairs done");
  });
});

describe("TunerApp live science from projection rows", () => {
  it("renders convergence / funnel / observations from rows while the run is live, with 'in progress' badges", async () => {
    const proposals = [
      { proposal_index: 0, cohort_index: 0, cohort_slot: 0, candidate_id: "candidate-aaaa1111", source: "schema_default", source_attempt: 0, disposition: "accepted", frontier_id: "f", origin: null, acquisition: null, prediction: null, uncertainty: null, parent_candidate_id: null, refill_of_candidate_id: null },
      { proposal_index: 1, cohort_index: 0, cohort_slot: 1, candidate_id: "candidate-bbbb2222", source: "smac_model", source_attempt: 0, disposition: "accepted", frontier_id: "f", origin: null, acquisition: null, prediction: null, uncertainty: null, parent_candidate_id: null, refill_of_candidate_id: null },
    ];
    const observations = [
      { observation_id: "o1", candidate_id: "candidate-aaaa1111", phase: "tuning", prefix_id: "p1", mean: 0.4, lower: 0.2, upper: 0.6 },
      { observation_id: "o2", candidate_id: "candidate-bbbb2222", phase: "tuning", prefix_id: "p1", mean: 0.62, lower: 0.4, upper: 0.8 },
    ];
    const candidates = [
      { candidate_id: "candidate-aaaa1111", fingerprint: "f", canonical_config: {}, cohort_index: 0, cohort_slot: 0, source: "schema_default", parent_candidate_id: null },
      { candidate_id: "candidate-bbbb2222", fingerprint: "f", canonical_config: {}, cohort_index: 0, cohort_slot: 1, source: "smac_model", parent_candidate_id: null },
    ];

    render(() => (
      <TunerApp
        env={mockTunerEnv({
          listTunableGames: () => Effect.send(kinds),
          listObjectives: () => Effect.send(objectives),
          listRuns: () => Effect.send([runView({ run_id: "live-1", status: "live" })]),
          listProjectionRuns: () => Effect.send([]),
          getProjectionRun: () =>
            Effect.send({
              run_id: "live-1",
              terminal_status: "open",
              report_available: false,
              ingest_error: null,
              manifest: null,
              report: null,
              compute: [],
            }),
          // The report 404s until the run completes.
          getProjectionReport: () =>
            Effect.fromPromise(() => Promise.reject(new Error("no report"))),
          getProjectionProposals: () => Effect.send(proposals),
          getProjectionObservations: () => Effect.send(observations),
          getProjectionCandidates: () => Effect.send(candidates),
          openEvidenceStream: () => Effect.stream((_s, done) => done()),
        })}
      />
    ));

    await vi.waitFor(() => expect(screen.getByText("live-1")).toBeInTheDocument());
    fireEvent.click(screen.getByText("live-1"));
    await vi.waitFor(() => expect(screen.getByTestId("tuner-run-overview")).toBeInTheDocument());

    fireEvent.click(screen.getByRole("button", { name: "Full science →" }));
    await vi.waitFor(() => expect(screen.getByTestId("tuner-run-science")).toBeInTheDocument());

    // The view renders even though there is no report.
    await vi.waitFor(() =>
      expect(screen.getByTestId("science-convergence-liveness")).toHaveTextContent("in progress"),
    );
    expect(screen.getByTestId("science-proposal-search-liveness")).toHaveTextContent("in progress");
    // A 12e-8 section still waits for the report.
    expect(screen.getByTestId("science-opponent-response-liveness")).toHaveTextContent(
      "available when the run completes",
    );

    // The observation forest carries the row means.
    const obsSection = within(screen.getByTestId("science-observations"));
    fireEvent.click(obsSection.getByRole("button", { name: "Show numbers" }));
    await vi.waitFor(() =>
      expect(screen.getByTestId("observation-numbers")).toHaveTextContent("0.620"),
    );
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
