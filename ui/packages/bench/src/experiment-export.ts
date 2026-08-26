import type { ExperimentCell, ExperimentSpecV1, RunDetail, RunStatus } from "./types.js";

export interface ExperimentRunExportV1 {
  version: 1;
  run: {
    run_id: string;
    project_id: string | null;
    experiment_id: string | null;
    label: string | null;
    status: RunStatus;
    git_sha: string;
    git_dirty: boolean;
    host: string;
    started_at: string;
    ended_at: string | null;
    experiment_spec: ExperimentSpecV1;
  };
  cells: ExperimentCell[];
}

function sortedCells(cells: ExperimentCell[]): ExperimentCell[] {
  return [...cells].sort((left, right) => left.cell_id.localeCompare(right.cell_id));
}

function exportEnvelope(detail: RunDetail, cells: ExperimentCell[]): ExperimentRunExportV1 {
  if (!detail.experiment_spec) throw new Error("The run snapshot is not available yet.");
  return {
    version: 1,
    run: {
      run_id: detail.run_id,
      project_id: detail.project_id,
      experiment_id: detail.experiment_id,
      label: detail.label,
      status: detail.status,
      git_sha: detail.git_sha,
      git_dirty: detail.git_dirty,
      host: detail.host,
      started_at: detail.started_at,
      ended_at: detail.ended_at,
      experiment_spec: detail.experiment_spec,
    },
    cells: sortedCells(cells),
  };
}

export function serializeExperimentRunJson(detail: RunDetail, cells: ExperimentCell[]): string {
  return `${JSON.stringify(exportEnvelope(detail, cells), null, 2)}\n`;
}

const CSV_HEADER = [
  "run_id",
  "run_status",
  "cell_id",
  "cell_status",
  "game",
  "game_config_json",
  "budget_kind",
  "budget_value",
  "rounds",
  "cell_seed",
  "planned_games",
  "completed_games",
  "variant_id",
  "variant_label",
  "candidate_config_json",
  "baseline_id",
  "baseline_label",
  "baseline_config_json",
  "wins",
  "losses",
  "draws",
  "win_rate",
  "ci_lower",
  "ci_upper",
  "started_at",
  "ended_at",
  "error",
];

function compactJson(value: unknown): string {
  return JSON.stringify(value === undefined ? null : value);
}

function csvField(value: unknown): string {
  const text = value === null || value === undefined ? "" : String(value);
  return /[",\r\n]/.test(text) ? `"${text.replaceAll('"', '""')}"` : text;
}

export function serializeExperimentRunCsv(detail: RunDetail, cells: ExperimentCell[]): string {
  if (!detail.experiment_spec) throw new Error("The run snapshot is not available yet.");
  const rows = sortedCells(cells).map((cell) =>
    [
      detail.run_id,
      detail.status,
      cell.cell_id,
      cell.status,
      cell.game,
      compactJson(cell.game_config),
      cell.budget.kind,
      cell.budget.value,
      cell.rounds,
      cell.cell_seed,
      cell.planned_games,
      cell.completed_games,
      cell.variant_id,
      cell.variant_label,
      compactJson(cell.candidate_config),
      cell.baseline_id,
      cell.baseline_label,
      compactJson(cell.baseline_config),
      cell.wins,
      cell.losses,
      cell.draws,
      cell.win_rate,
      cell.ci_lower,
      cell.ci_upper,
      cell.started_at,
      cell.ended_at,
      cell.error,
    ]
      .map(csvField)
      .join(","),
  );
  return [CSV_HEADER.join(","), ...rows].join("\r\n") + "\r\n";
}

export function sanitizeExportRunId(runId: string): string {
  const safe = runId.replace(/[^A-Za-z0-9._-]+/g, "-").replace(/^-+|-+$/g, "");
  return safe || "run";
}
