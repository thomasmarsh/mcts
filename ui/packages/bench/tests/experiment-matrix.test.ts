import { describe, expect, it } from "vitest";
import { buildExperimentMatrix } from "../src/experiment-matrix.js";
import type { ExperimentCell, ExperimentSpecV1 } from "../src/types.js";

const spec: ExperimentSpecV1 = {
  version: 1,
  games: [
    { game: "game-a", game_config: { board: "a" } },
    { game: "game-b", game_config: { board: "b" } },
  ],
  baseline: { id: "base", label: "Base", config: {} },
  variants: [
    { id: "v1", label: "V1", config: {} },
    { id: "v2", label: "V2", config: {} },
    { id: "v3", label: "V3", config: {} },
  ],
  budgets: [
    { kind: "iterations", value: 10 },
    { kind: "time_per_move_ms", value: 10 },
  ],
  rounds_per_cell: 1,
  base_seed: 42,
  max_parallel_cells: 1,
};

function cell(
  cellId: string,
  game: string,
  budget: ExperimentCell["budget"],
  variantId: string,
): ExperimentCell {
  return {
    cell_id: cellId,
    cell_seed: 1,
    game,
    game_config: {},
    variant_id: variantId,
    variant_label: variantId,
    candidate_config: {},
    baseline_id: "base",
    baseline_label: "Base",
    baseline_config: {},
    budget,
    rounds: 1,
    planned_games: 2,
    completed_games: 1,
    status: "running",
    started_at: null,
    ended_at: null,
    error: null,
    wins: 1,
    losses: 0,
    draws: 0,
    win_rate: 1,
    ci_lower: 0,
    ci_upper: 1,
  };
}

describe("buildExperimentMatrix", () => {
  it("builds budget sections with game rows and variant columns in spec order", () => {
    const cells = spec.budgets.flatMap((budget, budgetIndex) =>
      spec.games.flatMap((game, gameIndex) =>
        spec.variants.map((variant, variantIndex) =>
          cell(`cell-${budgetIndex}-${gameIndex}-${variantIndex}`, game.game, budget, variant.id),
        ),
      ),
    );
    const model = buildExperimentMatrix(spec, [...cells].reverse());
    expect(model.sections).toHaveLength(2);
    expect(model.sections.map((section) => [section.budget.kind, section.budget.value])).toEqual([
      ["iterations", 10],
      ["time_per_move_ms", 10],
    ]);
    expect(
      model.sections.every(
        (section) =>
          section.rows.length === 2 && section.rows.every((row) => row.cells.length === 3),
      ),
    ).toBe(true);
    expect(model.sections[0]!.rows.map((row) => row.game.game)).toEqual(["game-a", "game-b"]);
    expect(model.sections[0]!.columns.map((variant) => variant.id)).toEqual(["v1", "v2", "v3"]);
    expect(model.warnings).toEqual([]);
  });

  it("does not collide iteration and time budgets with the same value", () => {
    const model = buildExperimentMatrix(spec, [
      cell("time", "game-a", { kind: "time_per_move_ms", value: 10 }, "v1"),
      cell("iterations", "game-a", { kind: "iterations", value: 10 }, "v1"),
    ]);
    expect(model.sections[0]!.rows[0]!.cells[0]!.cell?.cell_id).toBe("iterations");
    expect(model.sections[1]!.rows[0]!.cells[0]!.cell?.cell_id).toBe("time");
  });

  it("leaves missing coordinates visible and reports unexpected and duplicate cells deterministically", () => {
    const model = buildExperimentMatrix(spec, [
      cell("z-duplicate", "game-a", spec.budgets[0]!, "v1"),
      cell("a-first", "game-a", spec.budgets[0]!, "v1"),
      cell("unexpected", "other", spec.budgets[0]!, "v1"),
    ]);
    expect(model.sections[0]!.rows[0]!.cells[0]!.cell?.cell_id).toBe("a-first");
    expect(model.sections[0]!.rows[0]!.cells[1]!.cell).toBeNull();
    expect(model.warnings.map((warning) => [warning.kind, warning.cellId])).toEqual([
      ["unexpected", "unexpected"],
      ["duplicate", "z-duplicate"],
    ]);
  });

  it("keeps a one-cell snapshot as a one-by-one matrix", () => {
    const oneSpec = {
      ...spec,
      games: [spec.games[0]!],
      budgets: [spec.budgets[0]!],
      variants: [spec.variants[0]!],
    };
    const model = buildExperimentMatrix(oneSpec, [
      cell("only", "game-a", oneSpec.budgets[0]!, "v1"),
    ]);
    expect(model.sections).toHaveLength(1);
    expect(model.sections[0]!.rows).toHaveLength(1);
    expect(model.sections[0]!.rows[0]!.cells).toHaveLength(1);
    expect(model.sections[0]!.rows[0]!.cells[0]!.cell?.cell_id).toBe("only");
  });
});
