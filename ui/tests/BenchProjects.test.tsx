import { cleanup, fireEvent, render, screen } from "@solidjs/testing-library";
import { afterEach, describe, expect, it, vi } from "vitest";
import { Effect, createStore } from "@mcts/core";
import type { BenchEnv, BenchAction, BenchState } from "../packages/bench/src/reducer.js";
import { benchReducer } from "../packages/bench/src/reducer.js";
import { initialBenchState } from "../packages/bench/src/state.js";
import type { Experiment, ExperimentCell, Project, RunDetail } from "../packages/bench/src/types.js";
import { ProjectsApp } from "../packages/bench/src/ProjectsApp.js";
import { ExperimentEditor } from "../packages/bench/src/ExperimentEditor.js";
import { ExperimentRunDetail } from "../packages/bench/src/ExperimentRunDetail.js";

afterEach(() => {
  cleanup();
});

const noOpEnv: BenchEnv = {
  listProjects: () => Effect.none(), createProject: () => Effect.none(), getProject: () => Effect.none(), updateProject: () => Effect.none(),
  listExperiments: () => Effect.none(), createExperiment: () => Effect.none(), getExperiment: () => Effect.none(), updateExperiment: () => Effect.none(), launchExperiment: () => Effect.none(),
  getRunCells: () => Effect.send([]), listRuns: () => Effect.none(), getRun: () => Effect.none(), getRunLog: () => Effect.none(), getRunStdout: () => Effect.none(),
  getLeaderboard: () => Effect.none(), fetchCommitTrends: () => Effect.none(), launchRun: () => Effect.none(), stopRun: () => Effect.none(), resumeRun: () => Effect.none(), advanceBaseline: () => Effect.none(),
  getBenchKinds: () => Effect.none(), getSmac3Kinds: () => Effect.none(), getRunTrials: () => Effect.send([]), getRunChain: () => Effect.send([]), getRunGames: () => Effect.send([]), getRunGameMoves: () => Effect.none(), deleteRun: () => Effect.none(),
};

describe("persisted experiment components", () => {
  it("keeps all repeated removal controls visible and synchronizes indexed JSON after removal", async () => {
    const state = initialBenchState();
    state.selectedProjectId = "project-1";
    state.experimentDraft = { name: "Array editing", description: "", spec: {
      version: 1,
      games: [{ game: "nim", game_config: { marker: "first-game" } }],
      baseline: { id: "baseline", label: "Baseline", config: {} },
      variants: [{ id: "variant", label: "Variant", config: { marker: "first-variant" } }],
      budgets: [{ kind: "iterations", value: 5 }],
      rounds_per_cell: 1, base_seed: 42, max_parallel_cells: 1,
    } };
    state.smac3Kinds = { ...state.smac3Kinds, status: "done", result: [
      { game: "nim", tuner: { id: "nim", baselines: [], eval_rounds: 1, parameters: [], conditions: [], game_config: null } },
      { game: "druid", tuner: { id: "druid", baselines: [], eval_rounds: 1, parameters: [], conditions: [], game_config: { size: 7 } } },
    ] };
    const store = createStore<BenchState, BenchAction, BenchEnv>(state, benchReducer, noOpEnv);
    render(() => <ExperimentEditor store={store} />);

    expect(screen.getByRole("button", { name: "Remove game 1" })).toBeDisabled();
    expect(screen.getByRole("button", { name: "Remove variant 1" })).toBeDisabled();
    expect(screen.getByRole("button", { name: "Remove budget 1" })).toBeDisabled();

    fireEvent.click(screen.getByRole("button", { name: "Add game" }));
    fireEvent.click(screen.getByRole("button", { name: "Add variant" }));
    fireEvent.click(screen.getByRole("button", { name: "Add budget" }));
    await vi.waitFor(() => {
      expect(screen.getByRole("button", { name: "Remove game 2" })).toBeInTheDocument();
      expect(document.getElementById("game-config-1")).toBeInTheDocument();
      expect(document.getElementById("variant-config-1")).toBeInTheDocument();
    });

    const edited = JSON.parse(JSON.stringify(store.state.experimentDraft!)) as NonNullable<BenchState["experimentDraft"]>;
    edited.spec.games[1]!.game_config = { marker: "second-game" };
    edited.spec.variants[1]!.config = { marker: "second-variant" };
    store.dispatch({ tag: "experimentDraft", draft: edited });
    await vi.waitFor(() => {
      expect(store.state.experimentDraft?.spec.games[1]?.game_config).toEqual({ marker: "second-game" });
      expect(store.state.experimentDraft?.spec.variants[1]?.config).toEqual({ marker: "second-variant" });
    });

    fireEvent.click(screen.getByRole("button", { name: "Remove game 1" }));
    fireEvent.click(screen.getByRole("button", { name: "Remove variant 1" }));
    fireEvent.click(screen.getByRole("button", { name: "Remove budget 1" }));
    await vi.waitFor(() => {
      expect(store.state.experimentDraft?.spec.games).toHaveLength(1);
      expect(store.state.experimentDraft?.spec.games[0]?.game_config).toEqual({ marker: "second-game" });
      expect(store.state.experimentDraft?.spec.variants[0]?.config).toEqual({ marker: "second-variant" });
      expect((document.getElementById("game-config-0") as HTMLTextAreaElement).value).toContain("second-game");
      expect((document.getElementById("variant-config-0") as HTMLTextAreaElement).value).toContain("second-variant");
      expect(screen.getByRole("button", { name: "Remove game 1" })).toBeDisabled();
      expect(screen.getByRole("button", { name: "Remove variant 1" })).toBeDisabled();
      expect(screen.getByRole("button", { name: "Remove budget 1" })).toBeDisabled();
    });
  });

  it("renders a project shell with labelled controls and keeps an empty name unsubmitable", async () => {
    const store = createStore<BenchState, BenchAction, BenchEnv>(initialBenchState(), benchReducer, noOpEnv);
    render(() => <ProjectsApp store={store} />);

    expect(screen.getByRole("main")).toHaveClass("projects-page");
    expect(screen.getByLabelText("Project name")).toBeInTheDocument();
    const create = screen.getByRole("button", { name: "Create project" });
    expect(create).toBeDisabled();
    fireEvent.input(screen.getByLabelText("Project name"), { target: { value: "   " } });
    await vi.waitFor(() => expect(create).toBeDisabled());
  });

  it("keeps malformed strategy JSON visible, localizes the error, and re-enables saving after correction", async () => {
    const state = initialBenchState();
    state.selectedProjectId = "project-1";
    state.experimentDraft = { name: "JSON check", description: "", spec: {
      version: 1, games: [{ game: "nim", game_config: null }], baseline: { id: "baseline", label: "Baseline", config: {} }, variants: [{ id: "variant", label: "Variant", config: {} }], budgets: [{ kind: "iterations", value: 5 }], rounds_per_cell: 1, base_seed: 42, max_parallel_cells: 1,
    } };
    const store = createStore<BenchState, BenchAction, BenchEnv>(state, benchReducer, noOpEnv);
    render(() => <ProjectsApp store={store} />);

    const baseline = document.getElementById("baseline-config") as HTMLTextAreaElement;
    fireEvent.input(baseline, { target: { value: "{ broken" } });
    await vi.waitFor(() => expect(screen.getByText("Enter valid JSON.")).toBeInTheDocument());
    expect(baseline.value).toBe("{ broken");
    expect(screen.getByRole("button", { name: "Save" })).toBeDisabled();
    expect(screen.getByRole("button", { name: "Launch" })).toBeDisabled();

    fireEvent.input(baseline, { target: { value: '{"depth": 2}' } });
    await vi.waitFor(() => expect(store.state.experimentDraft?.spec.baseline.config).toEqual({ depth: 2 }));
    await vi.waitFor(() => expect(screen.getByRole("button", { name: "Save" })).not.toBeDisabled());
  });

  it("associates indexed server validation errors with repeated editor controls", () => {
    const state = initialBenchState();
    state.selectedProjectId = "project-1";
    state.experimentDraft = { name: "Field errors", description: "", spec: {
      version: 1,
      games: [{ game: "nim", game_config: null }, { game: "druid", game_config: { size: 7 } }],
      baseline: { id: "baseline", label: "Baseline", config: {} },
      variants: [{ id: "variant-1", label: "Variant 1", config: {} }, { id: "variant-2", label: "Variant 2", config: {} }],
      budgets: [{ kind: "iterations", value: 5 }, { kind: "time_per_move_ms", value: 10 }],
      rounds_per_cell: 1, base_seed: 42, max_parallel_cells: 1,
    } };
    state.experimentFieldErrors = {
      "spec.games[1].game": "games[1] is unavailable",
      "spec.variants[1].config": "candidate configuration is invalid",
      "spec.budgets[1].value": "budget must be positive",
    };
    const store = createStore<BenchState, BenchAction, BenchEnv>(state, benchReducer, noOpEnv);
    render(() => <ExperimentEditor store={store} />);

    for (const [controlId, errorId] of [
      ["game-1", "experiment-field-error-game-1"],
      ["variant-config-1", "experiment-field-error-variant-config-1"],
      ["budget-value-1", "experiment-field-error-budget-value-1"],
    ]) {
      const control = document.getElementById(controlId)!;
      expect(control.getAttribute("aria-invalid")).toBe("true");
      expect(control.getAttribute("aria-describedby")).toBe(errorId);
      expect(document.getElementById(errorId)).toBeInTheDocument();
    }
    expect(screen.getByText("candidate configuration is invalid")).toBeInTheDocument();
  });

  it("uses SMAC3 metadata for game defaults and preserves a positive budget when changing budget kinds", async () => {
    const state = initialBenchState();
    state.selectedProjectId = "project-1";
    state.experimentDraft = { name: "Metadata check", description: "", spec: {
      version: 1, games: [{ game: "nim", game_config: { stones: 7 } }], baseline: { id: "baseline", label: "Baseline", config: {} }, variants: [{ id: "variant", label: "Variant", config: {} }], budgets: [{ kind: "iterations", value: 9 }], rounds_per_cell: 1, base_seed: 42, max_parallel_cells: 1,
    } };
    state.smac3Kinds = { ...state.smac3Kinds, status: "done", result: [
      { game: "nim", tuner: { id: "nim", baselines: ["default"], eval_rounds: 1, parameters: [], conditions: [], game_config: { stones: 7 } } },
      { game: "druid", tuner: { id: "druid", baselines: ["default"], eval_rounds: 1, parameters: [], conditions: [], game_config: { size: 5 } } },
    ] };
    const store = createStore<BenchState, BenchAction, BenchEnv>(state, benchReducer, noOpEnv);
    render(() => <ProjectsApp store={store} />);

    fireEvent.change(screen.getByRole("combobox", { name: "Game" }), { target: { value: "druid" } });
    await vi.waitFor(() => expect(store.state.experimentDraft?.spec.games[0]).toEqual({ game: "druid", game_config: { size: 5 } }));
    fireEvent.change(screen.getByRole("combobox", { name: "Budget kind" }), { target: { value: "time_per_move_ms" } });
    await vi.waitFor(() => expect(store.state.experimentDraft?.spec.budgets[0]).toEqual({ kind: "time_per_move_ms", value: 9 }));
    expect((screen.getByLabelText("Budget value") as HTMLInputElement).value).toBe("9");
  });

  it("traverses project, experiment, launch, progress, and cell inspection with mocked effects", async () => {
    const project: Project = {
      project_id: "project-1", name: "Nim study", description: "small", archived: false,
      created_at: "2026-01-01T00:00:00Z", updated_at: "2026-01-01T00:00:00Z",
    };
    const experiment: Experiment = {
      experiment_id: "experiment-1", project_id: project.project_id, name: "Nim baseline", description: "one cell",
      spec: {
        version: 1, games: [{ game: "nim", game_config: null }],
        baseline: { id: "baseline", label: "Baseline", config: {} },
        variants: [{ id: "variant", label: "Variant", config: {} }],
        budgets: [{ kind: "iterations", value: 1 }], rounds_per_cell: 1, base_seed: 42, max_parallel_cells: 1,
      },
      created_at: project.created_at, updated_at: project.updated_at,
    };
    const cell: ExperimentCell = {
      cell_id: "cell-1", cell_seed: 7294331206661666, game: "nim", game_config: null, variant_id: "variant", variant_label: "Variant",
      candidate_config: {}, baseline_id: "baseline", baseline_label: "Baseline", baseline_config: {},
      budget: { kind: "iterations", value: 1 }, rounds: 1, planned_games: 2, completed_games: 1,
      status: "running", started_at: project.created_at, ended_at: null, error: null,
      wins: 1, losses: 0, draws: 0, win_rate: 1, ci_lower: 0.2, ci_upper: 1,
    };
    const detail: RunDetail = {
      run_id: "run-1", kind: "experiment", game: "nim", project_id: project.project_id,
      experiment_id: experiment.experiment_id, experiment_spec: experiment.spec, label: experiment.name,
      config: null, git_sha: "test", git_dirty: false, host: "test", pid: 1,
      started_at: project.created_at, ended_at: null, status: "running", log_path: "/tmp/log.jsonl",
      exit_code: null, match_count: 1, trial_count: 0, incumbent: null,
    };
    const env: BenchEnv = {
      listProjects: () => Effect.send([project]), createProject: () => Effect.send(project), getProject: () => Effect.send(project),
      updateProject: () => Effect.send(project), listExperiments: () => Effect.send([experiment]),
      createExperiment: () => Effect.send(experiment), getExperiment: () => Effect.send(experiment), updateExperiment: () => Effect.send(experiment),
      launchExperiment: () => Effect.send({ run_id: "run-1", pid: 1, log_path: "/tmp/log.jsonl" }),
      getRunCells: () => Effect.send([cell]), listRuns: () => Effect.send([]), getRun: () => Effect.send({ ...detail, status: "completed" }),
      getRunLog: () => Effect.send({ lines: [], next_offset: 0 }), getRunStdout: () => Effect.send(""), getLeaderboard: () => Effect.send([]),
      fetchCommitTrends: () => Effect.send({}), launchRun: () => Effect.none(), stopRun: () => Effect.none(), resumeRun: () => Effect.none(),
      advanceBaseline: () => Effect.none(), getBenchKinds: () => Effect.none(), getSmac3Kinds: () => Effect.none(),
      getRunTrials: () => Effect.send([]), getRunChain: () => Effect.send([]), getRunGames: () => Effect.send([]), getRunGameMoves: () => Effect.none(), deleteRun: () => Effect.none(),
    };
    const dispatched: BenchAction[] = [];
    const store = createStore<BenchState, BenchAction, BenchEnv>(initialBenchState(), benchReducer, env, (action) => dispatched.push(action));
    render(() => <><ProjectsApp store={store} /><ExperimentRunDetail store={store} /></>);

    fireEvent.input(screen.getByLabelText("Project name"), { target: { value: project.name } });
    await vi.waitFor(() => expect(screen.getByRole("button", { name: "Create project" })).not.toBeDisabled());
    fireEvent.click(screen.getByRole("button", { name: "Create project" }));
    await vi.waitFor(() => expect(store.state.selectedProjectId).toBe(project.project_id));
    await vi.waitFor(() => expect(screen.getByRole("button", { name: "New experiment" })).toBeInTheDocument());
    fireEvent.click(screen.getByRole("button", { name: "New experiment" }));
    await vi.waitFor(() => expect(screen.getByLabelText("Experiment name")).toBeInTheDocument());
    fireEvent.input(screen.getByLabelText("Experiment name"), { target: { value: experiment.name } });
    fireEvent.click(screen.getByRole("button", { name: "Save" }));
    await vi.waitFor(() => expect(screen.getByRole("button", { name: "Launch" })).not.toBeDisabled());
    fireEvent.click(screen.getByRole("button", { name: "Launch" }));
    await vi.waitFor(() => expect(dispatched).toContainEqual({ tag: "experimentLaunched", response: { run_id: "run-1", pid: 1, log_path: "/tmp/log.jsonl" } }));
    await vi.waitFor(() => expect(store.state.activeTab).toBe("runs"));
    await vi.waitFor(() => expect(screen.getByRole("button", { name: /Variant: 1\/2/ })).toBeInTheDocument());
    fireEvent.click(screen.getByRole("button", { name: /Variant: 1\/2/ }));
    expect(screen.getByText(/Candidate:/)).toBeInTheDocument();
  });

  it("renders a mocked 2x2x3 run, filters source games by cell, rejects stale polls, and stops", async () => {
    const project: Project = {
      project_id: "project-grid", name: "Grid", description: "", archived: false,
      created_at: "2026-01-01T00:00:00Z", updated_at: "2026-01-01T00:00:00Z",
    };
    const spec = {
      version: 1 as const,
      games: [{ game: "game-a", game_config: { board: "a" } }, { game: "game-b", game_config: { board: "b" } }],
      baseline: { id: "base", label: "Base", config: { family: "ucb1" } },
      variants: [
        { id: "v1", label: "V1", config: { family: "rave" } },
        { id: "v2", label: "V2", config: { family: "flat_mc" } },
        { id: "v3", label: "V3", config: { family: "random" } },
      ],
      budgets: [{ kind: "iterations" as const, value: 11 }, { kind: "time_per_move_ms" as const, value: 23 }],
      rounds_per_cell: 1, base_seed: 99, max_parallel_cells: 2,
    };
    const statuses = ["completed", "pending", "running", "failed", "cancelled", "completed", "completed", "completed", "completed", "completed", "completed", "completed"];
    const cells: ExperimentCell[] = Array.from({ length: 12 }, (_, ordinal) => ({
      cell_id: `cell-${String(ordinal + 1).padStart(6, "0")}`,
      cell_seed: 1000 + ordinal, game: ordinal < 6 ? "game-a" : "game-b",
      game_config: { board: ordinal < 6 ? "a" : "b" }, variant_id: `v${(ordinal % 3) + 1}`, variant_label: `V${(ordinal % 3) + 1}`,
      candidate_config: { variant: ordinal }, baseline_id: "base", baseline_label: "Base", baseline_config: { family: "ucb1" },
      budget: ordinal % 2 === 0 ? { kind: "iterations", value: 11 } : { kind: "time_per_move_ms", value: 23 },
      rounds: 1, planned_games: 2, completed_games: statuses[ordinal] === "completed" ? 2 : statuses[ordinal] === "running" || statuses[ordinal] === "failed" ? 1 : 0,
      status: statuses[ordinal]!, started_at: project.created_at, ended_at: null,
      error: statuses[ordinal] === "failed" ? "variant rejected" : statuses[ordinal] === "cancelled" ? "run stopped" : null,
      wins: 1, losses: 0, draws: 0, win_rate: 0.5, ci_lower: 0.1, ci_upper: 0.9,
    }));
    const detail: RunDetail = {
      run_id: "grid-run", kind: "experiment", game: null, project_id: project.project_id,
      experiment_id: "grid-experiment", experiment_spec: spec, label: "Grid run", config: null,
      git_sha: "test", git_dirty: false, host: "test", pid: 7, started_at: project.created_at,
      ended_at: null, status: "completed_with_errors", log_path: "/tmp/grid.log", exit_code: 0,
      match_count: 18, trial_count: 0, incumbent: null,
    };
    const calls: Array<[string, number | undefined, string | null | undefined]> = [];
    const stopCalls: string[] = [];
    const env: BenchEnv = {
      ...noOpEnv,
      getRunGames: (runId, limit, cellId) => { calls.push([runId, limit, cellId]); return Effect.send(cellId ? [{ game_seq: 2, match_seq: 2, cell_id: cellId, seed: 8, metrics: null, ply_count: 1, started_at: project.created_at, ended_at: project.created_at, strategy_a: "V2", strategy_b: "Base", outcome: "win_a", winner: "V2" }] : []); },
      stopRun: (runId) => { stopCalls.push(runId); return Effect.send({ run_id: runId, status: "stopped" }); },
    };
    const state = initialBenchState();
    state.openGeneration = 4;
    state.selectedCellId = cells[0]!.cell_id;
    state.openRun = {
      runId: detail.run_id, detail, trials: [], chain: [], chainedTrials: [], games: [{ game_seq: 1, match_seq: 1, cell_id: cells[0]!.cell_id, seed: 7, metrics: null, ply_count: 1, started_at: project.created_at, ended_at: project.created_at, strategy_a: "V1", strategy_b: "Base", outcome: "win_a", winner: "V1" }],
      cells,
      tail: { lines: [], offset: 0, active: true, error: null, idleAttempts: 0, failures: 0 },
    };
    const store = createStore<BenchState, BenchAction, BenchEnv>(state, benchReducer, env);
    const { container } = render(() => <ExperimentRunDetail store={store} />);

    expect(container.querySelectorAll(".projects-cell-selector button")).toHaveLength(12);
    expect(screen.getByText("18 / 24")).toBeInTheDocument();
    expect(screen.getByText("completed with errors")).toBeInTheDocument();
    expect(screen.getByText("variant rejected")).toBeInTheDocument();
    expect(screen.getByText("run stopped")).toBeInTheDocument();
    expect(screen.getByText("failed cells")).toBeInTheDocument();
    expect(screen.getByText("cancelled cells")).toBeInTheDocument();

    const secondCell = container.querySelector(".projects-cell-selector button:nth-child(2)") as HTMLButtonElement;
    fireEvent.click(secondCell);
    await vi.waitFor(() => expect(calls).toContainEqual(["grid-run", 5000, "cell-000002"]));
    await vi.waitFor(() => expect(screen.getByText("Game 2")).toBeInTheDocument());

    store.dispatch({ tag: "openRun", runId: "new-run" });
    store.dispatch({ tag: "tailed", generation: 4, lines: ["stale"], nextOffset: 9, detail, trials: [], chain: [], chainedTrials: [], cells });
    expect(store.state.openRun?.runId).toBe("new-run");
    expect(store.state.openRun?.detail).toBeNull();

    const stoppedDetail = { ...detail, status: "stopped" as const, ended_at: project.updated_at };
    store.dispatch({ tag: "openRun", runId: "grid-run" });
    const generation = store.state.openGeneration;
    store.dispatch({ tag: "tailed", generation, lines: [], nextOffset: 0, detail: { ...stoppedDetail, run_id: "grid-run" }, trials: [], chain: [], chainedTrials: [], cells: cells.map((cell) => ({ ...cell, status: cell.status === "pending" || cell.status === "running" ? "cancelled" : cell.status, error: cell.status === "pending" || cell.status === "running" ? "run stopped" : cell.error })) });
    await vi.waitFor(() => expect(screen.queryByRole("button", { name: "Stop" })).not.toBeInTheDocument());

    store.dispatch({ tag: "openRun", runId: "grid-run" });
    const runningGeneration = store.state.openGeneration;
    store.dispatch({ tag: "tailed", generation: runningGeneration, lines: [], nextOffset: 0, detail: { ...detail, status: "running" }, trials: [], chain: [], chainedTrials: [], cells });
    await vi.waitFor(() => expect(store.state.openRun?.detail?.status).toBe("running"));
    fireEvent.click(screen.getByRole("button", { name: "Stop" }));
    expect(stopCalls).toEqual(["grid-run"]);
  });
});
