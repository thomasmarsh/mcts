import { cleanup, fireEvent, render, screen } from "@solidjs/testing-library";
import { afterEach, describe, expect, it, vi } from "vitest";
import { Effect, createStore } from "@mcts/core";
import type { BenchEnv, BenchAction, BenchState } from "../packages/bench/src/reducer.js";
import { benchReducer } from "../packages/bench/src/reducer.js";
import { initialBenchState } from "../packages/bench/src/state.js";
import type { Experiment, ExperimentCell, Project, RunDetail } from "../packages/bench/src/types.js";
import { ProjectsApp } from "../packages/bench/src/ProjectsApp.js";
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
});
