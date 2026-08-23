import { cleanup, fireEvent, render, screen } from "@solidjs/testing-library";
import { afterEach, describe, expect, it, vi } from "vitest";
import type { Component } from "solid-js";
import { Effect, createStore } from "@mcts/core";
import type { BenchEnv, BenchAction, BenchState } from "../packages/bench/src/reducer.js";
import { benchReducer } from "../packages/bench/src/reducer.js";
import { initialBenchState } from "../packages/bench/src/state.js";
import type { BenchSpectatorProps, Experiment, ExperimentCell, ExperimentSpecV1, GameTraceSummary, Project, RunDetail } from "../packages/bench/src/types.js";
import { ProjectsApp } from "../packages/bench/src/ProjectsApp.js";
import { ExperimentEditor } from "../packages/bench/src/ExperimentEditor.js";
import { ExperimentRunDetail } from "../packages/bench/src/ExperimentRunDetail.js";
import { serializeExperimentRunCsv, serializeExperimentRunJson } from "../packages/bench/src/experiment-export.js";

afterEach(() => {
  cleanup();
});

const noOpEnv: BenchEnv = {
  listProjects: () => Effect.none(), createProject: () => Effect.none(), getProject: () => Effect.none(), updateProject: () => Effect.none(),
  listExperiments: () => Effect.none(), createExperiment: () => Effect.none(), getExperiment: () => Effect.none(), updateExperiment: () => Effect.none(), launchExperiment: () => Effect.none(),
  getRunCells: () => Effect.send([]), listRuns: () => Effect.none(), getRun: () => Effect.none(), getRunLog: () => Effect.none(), getRunStdout: () => Effect.none(), downloadFile: () => Effect.none(),
  getLeaderboard: () => Effect.none(), fetchCommitTrends: () => Effect.none(), launchRun: () => Effect.none(), stopRun: () => Effect.none(), resumeRun: () => Effect.none(), advanceBaseline: () => Effect.none(),
  getBenchKinds: () => Effect.none(), getTunerKinds: () => Effect.none(), listTuningSessions: () => Effect.none(), getTuningSession: () => Effect.none(), getRunTrials: () => Effect.send([]), getRunChain: () => Effect.send([]), getRunGames: () => Effect.send([]), getRunGameMoves: () => Effect.none(), deleteRun: () => Effect.none(),
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
    state.tunerKinds = { ...state.tunerKinds, status: "done", result: [
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

  it("uses tuner metadata for game defaults and preserves a positive budget when changing budget kinds", async () => {
    const state = initialBenchState();
    state.selectedProjectId = "project-1";
    state.experimentDraft = { name: "Metadata check", description: "", spec: {
      version: 1, games: [{ game: "nim", game_config: { stones: 7 } }], baseline: { id: "baseline", label: "Baseline", config: {} }, variants: [{ id: "variant", label: "Variant", config: {} }], budgets: [{ kind: "iterations", value: 9 }], rounds_per_cell: 1, base_seed: 42, max_parallel_cells: 1,
    } };
    state.tunerKinds = { ...state.tunerKinds, status: "done", result: [
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
      getRunLog: () => Effect.send({ lines: Array.from({ length: 502 }, (_, index) => `log-${index}`), next_offset: 14 }), getRunStdout: () => Effect.send(""), getLeaderboard: () => Effect.send([]),
      fetchCommitTrends: () => Effect.send({}), launchRun: () => Effect.none(), stopRun: () => Effect.none(), resumeRun: () => Effect.none(),
      advanceBaseline: () => Effect.none(), getBenchKinds: () => Effect.none(), getTunerKinds: () => Effect.none(), listTuningSessions: () => Effect.none(), getTuningSession: () => Effect.none(),
      getRunTrials: () => Effect.send([]), getRunChain: () => Effect.send([]), getRunGames: () => Effect.send([{ game_seq: 7, match_seq: 3, cell_id: cell.cell_id, seed: 101, metrics: { elapsed_ms: 22 }, ply_count: 9, started_at: project.created_at, ended_at: project.updated_at, strategy_a: "Variant", strategy_b: "Baseline", outcome: "win_a", winner: "Variant" }]), getRunGameMoves: () => Effect.none(), deleteRun: () => Effect.none(),
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
    await vi.waitFor(() => expect(screen.getByRole("button", { name: /nim, 1 iterations, Variant/ })).toBeInTheDocument());
    fireEvent.click(screen.getByRole("button", { name: /nim, 1 iterations, Variant/ }));
    const inspectorGeneration = store.state.openGeneration;
    store.dispatch({ tag: "tailed", generation: inspectorGeneration, lines: Array.from({ length: 502 }, (_, index) => `log-${index}`), nextOffset: 14, detail: { ...detail, status: "completed" }, trials: [], chain: [], chainedTrials: [], cells: [cell] });
    await vi.waitFor(() => expect(store.state.openRun?.tail.lines).toHaveLength(1004));
    expect(screen.getByText("Candidate configuration")).toBeInTheDocument();
    expect(screen.getByText("cell-1")).toBeInTheDocument();
    expect(screen.getByText("7294331206661666")).toBeInTheDocument();
    expect(screen.getByText("nim", { selector: "dd" })).toBeInTheDocument();
    expect(screen.getByText("Budget kind", { selector: "dt" }).parentElement).toHaveTextContent("iterations");
    expect(screen.getByText("Budget value", { selector: "dt" }).parentElement).toHaveTextContent("1");
    expect(screen.getByText("Paired rounds", { selector: "dt" }).parentElement).toHaveTextContent("1");
    expect(screen.getByText("Planned games", { selector: "dt" }).parentElement).toHaveTextContent("2");
    expect(screen.getByText("Completed games", { selector: "dt" }).parentElement).toHaveTextContent("1");
    expect(screen.getByText("completed")).toBeInTheDocument();
    expect(screen.getAllByText("Not recorded")).toHaveLength(2);
    expect(screen.getByText("W / L / D").parentElement).toHaveTextContent("1/0/0");
    expect(screen.getByText("Draw-as-half win rate").parentElement).toHaveTextContent("100.0%");
    expect(screen.getByText("95% interval").parentElement).toHaveTextContent("20.0% – 100.0%");
    expect(screen.getByText("Game configuration", { selector: "h3" }).parentElement).toHaveTextContent("null");
    expect(screen.getByText("Candidate configuration", { selector: "h3" }).parentElement).toHaveTextContent("{}");
    expect(screen.getByText("Baseline configuration", { selector: "h3" }).parentElement).toHaveTextContent("{}");
    expect(screen.getByText("Candidate ID").parentElement).toHaveTextContent("variant");
    expect(screen.getByText("Candidate label").parentElement).toHaveTextContent("Variant");
    expect(screen.getByText("Baseline ID").parentElement).toHaveTextContent("baseline");
    expect(screen.getByText("Baseline label").parentElement).toHaveTextContent("Baseline");
    const renderedLog = document.querySelector(".projects-log-tail code")?.textContent ?? "";
    const renderedLogLines = renderedLog.split("\n");
    expect(renderedLogLines).not.toContain("log-1");
    expect(renderedLogLines).toContain("log-2");
    expect(renderedLogLines).toContain("log-501");
    expect(screen.getByText(/Match 3 · Trace 7/)).toBeInTheDocument();
    expect(screen.getByText(/Variant vs Baseline/)).toBeInTheDocument();
    expect(screen.getByText(/win_a · winner Variant/)).toBeInTheDocument();
    expect(screen.getByText(/Seed 101/)).toBeInTheDocument();
    expect(screen.getByText(/9 plies/)).toBeInTheDocument();
    expect(screen.getByText(/elapsed_ms/)).toBeInTheDocument();
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
      cells: [...cells].reverse(),
      tail: { lines: [], offset: 0, active: true, error: null, idleAttempts: 0, failures: 0 },
    };
    const store = createStore<BenchState, BenchAction, BenchEnv>(state, benchReducer, env);
    const { container } = render(() => <ExperimentRunDetail store={store} />);

    expect(container.querySelectorAll(".projects-matrix-cell")).toHaveLength(12);
    expect(container.querySelectorAll(".projects-matrix-table")).toHaveLength(2);
    expect(screen.getByText("11 iterations results by game and variant")).toBeInTheDocument();
    expect(screen.getByText("23 ms per move results by game and variant")).toBeInTheDocument();
    expect(screen.getAllByRole("columnheader", { name: "Game" })).toHaveLength(2);
    expect(screen.getAllByRole("rowheader", { name: "game-a" })).toHaveLength(2);
    expect(screen.getAllByRole("columnheader", { name: "V1" })).toHaveLength(2);
    expect(screen.getByText(/Candidates versus/)).toHaveTextContent("Candidates versus Base");
    expect(screen.getByText("18 / 24")).toBeInTheDocument();
    expect(screen.getByText("completed with errors")).toBeInTheDocument();
    expect(screen.getByText("variant rejected")).toBeInTheDocument();
    expect(screen.getByText("run stopped")).toBeInTheDocument();
    expect(screen.getByText("failed cells")).toBeInTheDocument();
    expect(screen.getByText("cancelled cells")).toBeInTheDocument();
    expect(screen.getAllByText("No games yet").length).toBeGreaterThan(0);
    const firstCellButton = container.querySelector(".projects-matrix-cell") as HTMLButtonElement;
    expect(firstCellButton.getAttribute("aria-label")).toContain("game-a");
    expect(firstCellButton.getAttribute("aria-label")).toContain("11 iterations");
    expect(firstCellButton.getAttribute("aria-label")).toContain("V1");
    expect(firstCellButton.getAttribute("aria-label")).toContain("completed");
    expect(firstCellButton.getAttribute("aria-label")).toContain("2/2 games");
    expect(firstCellButton.getAttribute("aria-label")).toContain("W/L/D 1/0/0");
    expect(firstCellButton.getAttribute("aria-label")).toContain("50.0%");
    expect(firstCellButton.getAttribute("aria-label")).toContain("10.0% – 90.0%");

    const secondCell = container.querySelector(".projects-matrix-table tbody tr:first-child td:nth-child(3) button") as HTMLButtonElement;
    fireEvent.click(secondCell);
    await vi.waitFor(() => expect(calls).toContainEqual(["grid-run", 5000, "cell-000005"]));
    await vi.waitFor(() => expect(screen.getByText(/Match 2 · Trace 2/)).toBeInTheDocument());
    await vi.waitFor(() => expect(container.querySelector(".projects-matrix-table tbody tr:first-child td:nth-child(3) button")).toHaveAttribute("aria-pressed", "true"));
    expect(container.querySelector(".projects-matrix-cell")).toHaveAttribute("aria-pressed", "false");
    expect(screen.getByText("cell-000005")).toBeInTheDocument();

    const tailedCell = { ...cells[4]!, completed_games: 2, wins: 2, losses: 0, draws: 0, win_rate: 1, ci_lower: 0.3, ci_upper: 1 };
    store.dispatch({ tag: "tailed", generation: 4, lines: ["new-tail"], nextOffset: 12, detail, trials: [], chain: [], chainedTrials: [], cells: cells.map((cell) => cell.cell_id === tailedCell.cell_id ? tailedCell : cell) });
    await vi.waitFor(() => expect(screen.getByRole("button", { name: /game-a, 11 iterations, V2/ })).toHaveAccessibleName(expect.stringContaining("2/2 games")));
    expect(store.state.selectedCellId).toBe("cell-000005");

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

  it("derives the first valid matrix cell as selected without fetching it", () => {
    const spec: ExperimentSpecV1 = {
      version: 1,
      games: [{ game: "game-a", game_config: { board: "a" } }],
      baseline: { id: "base", label: "Base", config: {} },
      variants: [{ id: "v1", label: "V1", config: {} }, { id: "v2", label: "V2", config: {} }],
      budgets: [{ kind: "iterations", value: 10 }],
      rounds_per_cell: 1,
      base_seed: 44,
      max_parallel_cells: 1,
    };
    const makeCell = (cellId: string, game: string, variantId: string): ExperimentCell => ({
      cell_id: cellId, cell_seed: 1, game, game_config: { board: game }, variant_id: variantId, variant_label: variantId.toUpperCase(),
      candidate_config: { variantId }, baseline_id: "base", baseline_label: "Base", baseline_config: {}, budget: spec.budgets[0]!, rounds: 1,
      planned_games: 2, completed_games: 1, status: "running", started_at: "2026-01-01T00:00:00Z", ended_at: null, error: null,
      wins: 1, losses: 0, draws: 0, win_rate: 1, ci_lower: 0.2, ci_upper: 1,
    });
    const valid = makeCell("valid-v2", "game-a", "v2");
    const unexpected = makeCell("aaa-unexpected", "not-in-snapshot", "v1");
    const detail: RunDetail = {
      run_id: "fallback-run", kind: "experiment", game: null, project_id: null, experiment_id: "exp", experiment_spec: spec,
      label: "Fallback", config: null, git_sha: "sha", git_dirty: false, host: "test", pid: null,
      started_at: "2026-01-01T00:00:00Z", ended_at: null, status: "running", log_path: "", exit_code: null,
      match_count: 0, trial_count: 0, incumbent: null,
    };
    const getRunGames = vi.fn(() => Effect.send<GameTraceSummary[]>([]));
    const state = initialBenchState();
    state.openRun = {
      runId: detail.run_id, detail, cells: [unexpected, valid], games: [], trials: [], chain: [], chainedTrials: [],
      tail: { lines: [], offset: 0, active: true, error: null, idleAttempts: 0, failures: 0 },
    };
    const store = createStore<BenchState, BenchAction, BenchEnv>(state, benchReducer, { ...noOpEnv, getRunGames });
    render(() => <ExperimentRunDetail store={store} />);

    const cellButtons = screen.getAllByRole("button", { name: /game-a/ });
    expect(cellButtons).toHaveLength(1);
    expect(cellButtons[0]).toHaveAttribute("aria-pressed", "true");
    expect(screen.getByText("valid-v2")).toBeInTheDocument();
    expect(screen.getByText("Data warning:")).toBeInTheDocument();
    expect(store.state.selectedCellId).toBeNull();
    expect(getRunGames).not.toHaveBeenCalled();
  });

  it("exports every loaded cell through the component effect, including pending and failure recovery", async () => {
    const spec: ExperimentSpecV1 = {
      version: 1,
      games: [{ game: "game-a", game_config: null }],
      baseline: { id: "base", label: "Base", config: { depth: 1 } },
      variants: [{ id: "v1", label: "V1", config: { depth: 2 } }],
      budgets: [{ kind: "iterations", value: 10 }],
      rounds_per_cell: 1, base_seed: 44, max_parallel_cells: 1,
    };
    const makeCell = (cellId: string): ExperimentCell => ({
      cell_id: cellId, cell_seed: 1, game: "game-a", game_config: null, variant_id: "v1", variant_label: "V1",
      candidate_config: { depth: 2 }, baseline_id: "base", baseline_label: "Base", baseline_config: { depth: 1 }, budget: spec.budgets[0]!, rounds: 1,
      planned_games: 2, completed_games: 1, status: "running", started_at: "2026-01-01T00:00:00Z", ended_at: null, error: null,
      wins: 1, losses: 0, draws: 0, win_rate: 1, ci_lower: 0.2, ci_upper: 1,
    });
    const cells = [makeCell("cell-z"), makeCell("cell-a")];
    const detail: RunDetail = {
      run_id: "run/export", kind: "experiment", game: null, project_id: "project", experiment_id: "experiment", experiment_spec: spec,
      label: "Current snapshot", config: null, git_sha: "sha", git_dirty: true, host: "test", pid: 4,
      started_at: "2026-01-01T00:00:00Z", ended_at: null, status: "running", log_path: "", exit_code: null,
      match_count: 1, trial_count: 0, incumbent: null,
    };
    const requests: Array<[string, string, string]> = [];
    const pending: Array<{ resolve: () => void; reject: (error: Error) => void }> = [];
    let failNext = false;
    const downloadFile: BenchEnv["downloadFile"] = (filename, mimeType, contents) => {
      requests.push([filename, mimeType, contents]);
      if (failNext) {
        failNext = false;
        return Effect.fromPromise(() => Promise.reject(new Error("download failed")));
      }
      if (requests.length === 1) {
        return Effect.fromPromise(() => new Promise<void>((resolve, reject) => pending.push({ resolve, reject })));
      }
      return Effect.send(undefined);
    };
    const state = initialBenchState();
    state.selectedCellId = "cell-a";
    state.openRun = {
      runId: detail.run_id, detail, cells, games: [], trials: [], chain: [], chainedTrials: [],
      tail: { lines: [], offset: 0, active: true, error: null, idleAttempts: 0, failures: 0 },
    };
    const store = createStore<BenchState, BenchAction, BenchEnv>(state, benchReducer, { ...noOpEnv, downloadFile });
    render(() => <ExperimentRunDetail store={store} />);

    expect(screen.getByText("Export current run snapshot:")).toBeInTheDocument();
    expect(screen.getByText("Current snapshot")).toBeInTheDocument();
    const json = screen.getByRole("button", { name: "JSON" });
    const csv = screen.getByRole("button", { name: "CSV" });
    fireEvent.click(json);
    fireEvent.click(json);
    expect(requests).toHaveLength(1);
    await vi.waitFor(() => {
      expect(json).toBeDisabled();
      expect(csv).toBeDisabled();
      expect(document.querySelector(".projects-run-export-actions [role=status]")).toHaveTextContent("Preparing download");
    });
    expect(requests[0]).toEqual(["experiment-run-export.json", "application/json", serializeExperimentRunJson(detail, cells)]);
    pending[0]!.resolve();
    await vi.waitFor(() => expect(json).not.toBeDisabled());

    fireEvent.click(csv);
    await vi.waitFor(() => expect(requests).toHaveLength(2));
    expect(requests[1]).toEqual(["experiment-run-export.csv", "text/csv;charset=utf-8", serializeExperimentRunCsv(detail, cells)]);

    failNext = true;
    fireEvent.click(json);
    await vi.waitFor(() => expect(screen.getByRole("alert")).toHaveTextContent("download failed"));
    expect(json).not.toBeDisabled();
    expect(csv).not.toBeDisabled();
    expect(store.state.selectedCellId).toBe("cell-a");
    expect(store.state.openRun?.runId).toBe("run/export");
  });

  it("passes the selected trace across the replay boundary and clears it when the run changes", async () => {
    const spec: ExperimentSpecV1 = {
      version: 1,
      games: [{ game: "game-a", game_config: { board: "a" } }],
      baseline: { id: "base", label: "Base", config: {} },
      variants: [{ id: "v1", label: "V1", config: {} }],
      budgets: [{ kind: "iterations", value: 10 }],
      rounds_per_cell: 1, base_seed: 44, max_parallel_cells: 1,
    };
    const cell = (id: string): ExperimentCell => ({
      cell_id: id, cell_seed: 1, game: "game-a", game_config: { board: "a" }, variant_id: "v1", variant_label: "V1",
      candidate_config: {}, baseline_id: "base", baseline_label: "Base", baseline_config: {}, budget: spec.budgets[0]!, rounds: 1,
      planned_games: 1, completed_games: 1, status: "completed", started_at: "2026-01-01T00:00:00Z", ended_at: "2026-01-01T00:00:01Z", error: null,
      wins: 1, losses: 0, draws: 0, win_rate: 1, ci_lower: 0.2, ci_upper: 1,
    });
    const game = (gameSeq: number, cellId: string): GameTraceSummary => ({
      game_seq: gameSeq, match_seq: gameSeq, cell_id: cellId, seed: 9, metrics: { elapsed_ms: 12 }, ply_count: 3,
      started_at: "2026-01-01T00:00:00Z", ended_at: "2026-01-01T00:00:01Z", strategy_a: "V1", strategy_b: "Base", outcome: "win_a", winner: "V1",
    });
    const detail = (runId: string, cellId: string): RunDetail => ({
      run_id: runId, kind: "experiment", game: null, project_id: null, experiment_id: "exp", experiment_spec: spec,
      label: runId, config: null, git_sha: "sha", git_dirty: false, host: "test", pid: null,
      started_at: "2026-01-01T00:00:00Z", ended_at: "2026-01-01T00:00:01Z", status: "completed", log_path: "", exit_code: 0,
      match_count: 1, trial_count: 0, incumbent: null,
    });
    const spectatorProps: BenchSpectatorProps[] = [];
    const Spectator: Component<BenchSpectatorProps> = (props) => {
      spectatorProps.push({ ...props });
      return <div data-testid="fake-replay">{props.runId}:{props.cellId}:{props.initialGameSeq}</div>;
    };
    const state = initialBenchState();
    state.openRun = {
      runId: "run-a", detail: detail("run-a", "cell-a"), cells: [cell("cell-a")], games: [game(7, "cell-a")], trials: [], chain: [], chainedTrials: [],
      tail: { lines: [], offset: 0, active: false, error: null, idleAttempts: 0, failures: 0 },
    };
    const store = createStore<BenchState, BenchAction, BenchEnv>(state, benchReducer, noOpEnv);
    render(() => <ExperimentRunDetail store={store} Spectator={Spectator} />);

    fireEvent.click(screen.getByRole("button", { name: /Replay game 7/ }));
    await vi.waitFor(() => expect(screen.getByTestId("fake-replay")).toHaveTextContent("run-a:cell-a:7"));
    expect(spectatorProps.at(-1)).toEqual({ runId: "run-a", game: "game-a", kind: "experiment", live: false, cellId: "cell-a", initialGameSeq: 7 });

    store.dispatch({ tag: "openRun", runId: "run-b" });
    await vi.waitFor(() => expect(screen.queryByTestId("fake-replay")).not.toBeInTheDocument());
    const generation = store.state.openGeneration;
    store.dispatch({ tag: "tailed", generation, lines: [], nextOffset: 0, detail: detail("run-b", "cell-b"), trials: [], chain: [], chainedTrials: [], cells: [cell("cell-b")], games: [game(8, "cell-b")] });
    await vi.waitFor(() => expect(screen.queryByTestId("fake-replay")).not.toBeInTheDocument());
    fireEvent.click(screen.getByRole("button", { name: /Replay game 8/ }));
    await vi.waitFor(() => expect(screen.getByTestId("fake-replay")).toHaveTextContent("run-b:cell-b:8"));

    cleanup();
    const noSpectatorStore = createStore<BenchState, BenchAction, BenchEnv>(state, benchReducer, noOpEnv);
    render(() => <ExperimentRunDetail store={noSpectatorStore} />);
    expect(screen.queryByRole("button", { name: /Replay game/ })).not.toBeInTheDocument();
  });
});
