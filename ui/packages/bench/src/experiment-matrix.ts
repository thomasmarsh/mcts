import type { Budget, ExperimentCell, ExperimentGame, ExperimentSpecV1, NamedStrategyConfig } from "./types.js";

export interface MatrixCoordinate {
  game: string;
  budget: Budget;
  variantId: string;
}

export interface MatrixCell {
  coordinate: MatrixCoordinate;
  cell: ExperimentCell | null;
}

export interface MatrixRow {
  game: ExperimentGame;
  cells: MatrixCell[];
}

export interface ExperimentMatrixSection {
  budget: Budget;
  columns: NamedStrategyConfig[];
  rows: MatrixRow[];
}

export interface MatrixWarning {
  kind: "unexpected" | "duplicate";
  cellId: string;
  coordinate: MatrixCoordinate;
}

export interface ExperimentMatrix {
  sections: ExperimentMatrixSection[];
  warnings: MatrixWarning[];
}

function coordinateKey(game: string, budget: Budget, variantId: string): string {
  return JSON.stringify([game, budget.kind, budget.value, variantId]);
}

function coordinateForCell(cell: ExperimentCell): MatrixCoordinate {
  return { game: cell.game, budget: cell.budget, variantId: cell.variant_id };
}

export function buildExperimentMatrix(spec: ExperimentSpecV1, responseCells: ExperimentCell[]): ExperimentMatrix {
  const expected = new Set<string>();
  const sections: ExperimentMatrixSection[] = spec.budgets.map((budget) => ({
    budget,
    columns: spec.variants,
    rows: spec.games.map((game) => ({
      game,
      cells: spec.variants.map((variant) => {
        expected.add(coordinateKey(game.game, budget, variant.id));
        return { coordinate: { game: game.game, budget, variantId: variant.id }, cell: null };
      }),
    })),
  }));

  const cellsByCoordinate = new Map<string, ExperimentCell[]>();
  const warnings: MatrixWarning[] = [];
  for (const cell of [...responseCells].sort((left, right) => left.cell_id.localeCompare(right.cell_id))) {
    const key = coordinateKey(cell.game, cell.budget, cell.variant_id);
    if (!expected.has(key)) {
      warnings.push({ kind: "unexpected", cellId: cell.cell_id, coordinate: coordinateForCell(cell) });
      continue;
    }
    const existing = cellsByCoordinate.get(key) ?? [];
    existing.push(cell);
    cellsByCoordinate.set(key, existing);
  }

  for (const [key, cells] of cellsByCoordinate) {
    if (cells.length > 1) {
      for (const cell of cells.slice(1)) warnings.push({ kind: "duplicate", cellId: cell.cell_id, coordinate: coordinateForCell(cell) });
    }
    cellsByCoordinate.set(key, cells);
  }

  for (const section of sections) {
    for (const row of section.rows) {
      for (const entry of row.cells) {
        entry.cell = cellsByCoordinate.get(coordinateKey(entry.coordinate.game, entry.coordinate.budget, entry.coordinate.variantId))?.[0] ?? null;
      }
    }
  }

  warnings.sort((left, right) => left.cellId.localeCompare(right.cellId) || left.kind.localeCompare(right.kind));
  return { sections, warnings };
}

export function budgetLabel(budget: Budget): string {
  return budget.kind === "iterations" ? `${budget.value} iterations` : `${budget.value} ms per move`;
}
