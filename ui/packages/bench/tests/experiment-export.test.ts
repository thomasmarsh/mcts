import { describe, expect, it } from "vitest";
import { serializeExperimentRunCsv, serializeExperimentRunJson } from "../src/experiment-export.js";
import type { ExperimentCell, RunDetail } from "../src/types.js";

const detail: RunDetail = {
  run_id: "run/export", kind: "experiment", project_id: null, experiment_id: "exp-1", experiment_spec: {
    version: 1, games: [{ game: "nim", game_config: { label: "a,b" } }], baseline: { id: "base", label: "Base", config: { quote: "yes" } },
    variants: [{ id: "v1", label: "Variant", config: { lines: "one\ntwo" } }], budgets: [{ kind: "iterations", value: 5 }], rounds_per_cell: 1, base_seed: 42, max_parallel_cells: 1,
  }, label: null, game: "nim", config: null, git_sha: "sha", git_dirty: true, host: "host", pid: null,
  started_at: "2026-01-01T00:00:00Z", ended_at: null, status: "completed", log_path: "", exit_code: 0, match_count: 0, trial_count: 0, incumbent: null,
};

function cell(cellId: string): ExperimentCell {
  return {
    cell_id: cellId, cell_seed: null, game: "nim", game_config: { text: "a,b" }, variant_id: "v1", variant_label: "Variant, \"quoted\"",
    candidate_config: { lines: "one\ntwo" }, baseline_id: "base", baseline_label: "Base", baseline_config: { quote: "yes" }, budget: { kind: "iterations", value: 5 }, rounds: 1,
    planned_games: 2, completed_games: 0, status: "pending", started_at: null, ended_at: null, error: "bad,\"line\"\nnext",
    wins: 0, losses: 0, draws: 0, win_rate: 0.5, ci_lower: 0, ci_upper: 1,
  };
}

describe("experiment exports", () => {
  it("sorts JSON cells, retains native configs and raw rates, and emits stable bytes", () => {
    const first = serializeExperimentRunJson(detail, [cell("z"), cell("a")]);
    const second = serializeExperimentRunJson(detail, [cell("a"), cell("z")]);
    const expected = {
      version: 1,
      run: {
        run_id: "run/export",
        project_id: null,
        experiment_id: "exp-1",
        label: null,
        status: "completed",
        git_sha: "sha",
        git_dirty: true,
        host: "host",
        started_at: "2026-01-01T00:00:00Z",
        ended_at: null,
        experiment_spec: detail.experiment_spec,
      },
      cells: [cell("a"), cell("z")],
    };
    expect(first).toBe(`${JSON.stringify(expected, null, 2)}\n`);
    expect(first).toBe(second);
    const parsed = JSON.parse(first) as { version: number; run: { experiment_spec: unknown }; cells: ExperimentCell[] };
    expect(parsed.version).toBe(1);
    expect(parsed.run.experiment_spec).toEqual(detail.experiment_spec);
    expect(parsed.cells.map((value) => value.cell_id)).toEqual(["a", "z"]);
    expect(parsed.cells[0]!.win_rate).toBe(0.5);
    expect(parsed.cells[0]!.game_config).toEqual({ text: "a,b" });
    expect(parsed.cells[0]!.cell_seed).toBeNull();
    expect(JSON.stringify(parsed)).toBe(JSON.stringify(expected));
  });

  it("uses the frozen CSV header, CRLF rows, null blanks, and RFC 4180 escaping", () => {
    const csv = serializeExperimentRunCsv(detail, [cell("cell-1")]);
    const expected = "run_id,run_status,cell_id,cell_status,game,game_config_json,budget_kind,budget_value,rounds,cell_seed,planned_games,completed_games,variant_id,variant_label,candidate_config_json,baseline_id,baseline_label,baseline_config_json,wins,losses,draws,win_rate,ci_lower,ci_upper,started_at,ended_at,error\r\n"
      + "run/export,completed,cell-1,pending,nim,\"{\"\"text\"\":\"\"a,b\"\"}\",iterations,5,1,,2,0,v1,\"Variant, \"\"quoted\"\"\",\"{\"\"lines\"\":\"\"one\\ntwo\"\"}\",base,Base,\"{\"\"quote\"\":\"\"yes\"\"}\",0,0,0,0.5,0,1,,,\"bad,\"\"line\"\"\nnext\"\r\n";
    expect(csv).toBe(expected);
    expect(csv.split("\r\n")).toHaveLength(3);
  });
});
