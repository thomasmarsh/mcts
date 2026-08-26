import type { Budget, ExperimentCell, ExperimentSpecV1 } from "./types.js";

export const JS_MAX_SAFE_INTEGER = Number.MAX_SAFE_INTEGER;
const MASK_53 = (1n << 53n) - 1n;
const MASK_64 = (1n << 64n) - 1n;

/** Mirrors game-host's SplitMix64 finalizer and returns a JSON-safe integer. */
export function deriveSeed(seed: number, ordinal: number): number {
  let value = (BigInt(seed) + BigInt(ordinal) * 0x9e3779b97f4a7c15n) & MASK_64;
  value = ((value ^ (value >> 30n)) * 0xbf58476d1ce4e5b9n) & MASK_64;
  value = ((value ^ (value >> 27n)) * 0x94d049bb133111ebn) & MASK_64;
  return Number((value ^ (value >> 31n)) & MASK_53);
}

export interface ExpandedCellPreview {
  ordinal: number;
  cell_id: string;
  game: string;
  variant_id: string;
  variant_label: string;
  budget: Budget;
  rounds: number;
  planned_games: number;
  cell_seed: number;
  round_seeds: number[];
}

export interface ExpandedExperimentPreview {
  cells: ExpandedCellPreview[];
  total_planned_games: number;
}

export function expandExperimentSpec(spec: ExperimentSpecV1): ExpandedExperimentPreview {
  const count = spec.games.length * spec.budgets.length * spec.variants.length;
  const width = Math.max(6, String(Math.max(1, count)).length);
  const plannedGames = spec.rounds_per_cell * 2;
  const cells: ExpandedCellPreview[] = [];
  let ordinal = 0;
  for (const game of spec.games) {
    for (const budget of spec.budgets) {
      for (const variant of spec.variants) {
        const cellSeed = deriveSeed(spec.base_seed, ordinal);
        cells.push({
          ordinal,
          cell_id: `cell-${String(ordinal + 1).padStart(width, "0")}`,
          game: game.game,
          variant_id: variant.id,
          variant_label: variant.label,
          budget,
          rounds: spec.rounds_per_cell,
          planned_games: plannedGames,
          cell_seed: cellSeed,
          round_seeds: Array.from({ length: spec.rounds_per_cell }, (_, round) =>
            deriveSeed(cellSeed, round),
          ),
        });
        ordinal += 1;
      }
    }
  }
  return { cells, total_planned_games: count * plannedGames };
}

export function cellFromResponse(cell: ExperimentCell): ExpandedCellPreview {
  return {
    ordinal: Number.parseInt(cell.cell_id.replace("cell-", ""), 10) - 1,
    cell_id: cell.cell_id,
    game: cell.game,
    variant_id: cell.variant_id,
    variant_label: cell.variant_label,
    budget: cell.budget,
    rounds: cell.rounds,
    planned_games: cell.planned_games,
    cell_seed: cell.cell_seed ?? 0,
    round_seeds: [],
  };
}
