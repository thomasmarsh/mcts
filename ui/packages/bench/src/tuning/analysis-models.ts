import type {
  TuningAnalysisOverview,
  TuningAnalysisPoint,
  TuningBracketResourceAggregate,
  TuningPoolRevision,
  TuningTrialDetailPair,
  TuningTrialDetailView,
  TuningTrialPage,
  TuningTrialSummary,
} from "../types.js";

/** The wire value used by the trial-page route for reports without a bracket. */
export const UNASSIGNED_BRACKET = "unassigned";

export interface AnalysisSampleMetadata {
  total: number;
  returned: number;
  sampled: boolean;
}

export interface BracketFacet {
  id: string | null;
  key: string;
  label: string;
  resources: number[];
  reports: number;
  trials: number;
  points: number;
  empty: boolean;
}

export interface ResourceDomains {
  shared: number[];
  local: BracketFacet[];
}

export interface AnalysisPlotRow extends TuningAnalysisPoint {
  best: boolean;
  selected: boolean;
}

export interface DecisionSymbol {
  key: string;
  symbol: string;
  label: string;
}

export interface DecisionGroupRow {
  outcome: string;
  reason: string;
  pruning_exempt: boolean;
  reports: number;
  outcomeSymbol: DecisionSymbol;
  reasonSymbol: DecisionSymbol;
}

export interface RungFunnelRow extends TuningBracketResourceAggregate {
  bracket: string;
  selected: boolean;
}

export interface TrialTrajectory {
  trialId: string;
  trialNumber: number;
  trialStatus: string;
  selected: boolean;
  points: Array<AnalysisPlotRow & { terminal: boolean }>;
  reportCount: number;
}

export type PruningDecisionKey =
  | "below_minimum"
  | "startup_exempt"
  | "pruning_disabled"
  | "continued"
  | "pruned"
  | "confidence_completed"
  | "max_completed";

export interface PruningFunnelRow {
  key: PruningDecisionKey;
  label: string;
  description: string;
  reason: string;
  reports: number;
}

export interface WldSummary {
  wins: number;
  losses: number;
  draws: number;
  total: number;
}

export interface ComputeSummary {
  elapsedMs: number;
  searchIterationsTotal: number;
  searchMoveTimeMs: number;
}

export interface TrialSummaryRow extends TuningTrialSummary {
  wld: WldSummary;
  compute: ComputeSummary;
  selected: boolean;
}

export interface TrialPageSummary {
  rows: TrialSummaryRow[];
  wld: WldSummary;
  compute: ComputeSummary;
  totalCount: number;
  returnedCount: number;
  sampled: boolean;
}

export interface PoolRevisionCoverage {
  revisions: Array<TuningPoolRevision & { gapBefore: number[]; anchorCount: number }>;
  revisionCount: number;
  pairCount: number;
  anchorCount: number;
  unmatchedPoolRevisions: number;
}

export interface OpponentDistance {
  pairId: string;
  pairIndex: number;
  opponentId: string;
  candidateMu: number;
  opponentMu: number;
  deltaMu: number;
  absoluteMuDistance: number;
  candidateSigma: number;
  opponentSigma: number;
  deltaSigma: number;
}

function compareText(a: string, b: string): number {
  return a.localeCompare(b, undefined, { numeric: true, sensitivity: "base" });
}

function bracketKey(bracketId: string | null): string {
  return bracketId ?? UNASSIGNED_BRACKET;
}

function bracketLabel(bracketId: string | null): string {
  return bracketId === null ? "Unassigned" : bracketId;
}

function numericDomain(values: Iterable<number>): number[] {
  return [...new Set([...values].filter(Number.isFinite))].sort((a, b) => a - b);
}

function sortBracketIds(ids: Iterable<string | null>): Array<string | null> {
  return [...new Set(ids)].sort((a, b) => {
    if (a === null) return b === null ? 0 : -1;
    if (b === null) return 1;
    return compareText(a, b);
  });
}

/** Coverage attached to every chart that consumes the compact point sample. */
export function analysisSampleMetadata(overview: TuningAnalysisOverview): AnalysisSampleMetadata {
  return { ...overview.coverage.points };
}

/**
 * Bracket filter facets. Explicit `facetIds` make a selected but currently
 * empty bracket visible instead of silently dropping its filter control.
 */
export function bracketFacets(
  overview: TuningAnalysisOverview,
  facetIds: readonly (string | null)[] = [],
): BracketFacet[] {
  const ids = sortBracketIds([
    ...facetIds,
    ...overview.bracket_resources.map((row) => row.bracket_id),
    ...overview.points.map((point) => point.bracket_id),
  ]);
  return ids.map((id) => {
    const resources = overview.bracket_resources.filter((row) => row.bracket_id === id);
    const points = overview.points.filter((point) => point.bracket_id === id);
    return {
      id,
      key: bracketKey(id),
      label: bracketLabel(id),
      resources: numericDomain([...resources.map((row) => row.resource), ...points.map((point) => point.resource)]),
      reports: resources.reduce((total, row) => total + row.reports, 0),
      trials: resources.reduce((total, row) => total + row.trials, 0),
      points: points.length,
      empty: resources.length === 0 && points.length === 0,
    };
  });
}

/** Shared resource scale plus every bracket's local scale, both numerically sorted. */
export function resourceDomains(
  overview: TuningAnalysisOverview,
  facetIds: readonly (string | null)[] = [],
): ResourceDomains {
  const local = bracketFacets(overview, facetIds);
  return {
    shared: numericDomain(local.flatMap((facet) => facet.resources)),
    local,
  };
}

/** Exact, unrounded rows for score plots. */
export function exactPlotRows(overview: TuningAnalysisOverview, selectedTrialId: string | null = null): AnalysisPlotRow[] {
  const bestIds = new Set(overview.best?.trial_ids ?? []);
  return overview.points
    .map((point) => ({ ...point, best: bestIds.has(point.trial_id), selected: point.trial_id === selectedTrialId }))
    .sort((a, b) => a.trial_number - b.trial_number || a.resource - b.resource || compareText(a.trial_id, b.trial_id));
}

/** Small semantic symbols keep labels available to non-visual consumers. */
export function stateSymbol(state: string): DecisionSymbol {
  const normalized = state.toLowerCase();
  const known: Record<string, Omit<DecisionSymbol, "key">> = {
    queued: { symbol: "○", label: "Queued" },
    running: { symbol: "◌", label: "Running" },
    complete: { symbol: "✓", label: "Complete" },
    completed: { symbol: "✓", label: "Completed" },
    prune: { symbol: "↯", label: "Pruned" },
    pruned: { symbol: "↯", label: "Pruned" },
    failed: { symbol: "×", label: "Failed" },
    cancelled: { symbol: "−", label: "Cancelled" },
  };
  return { key: state, ...(known[normalized] ?? { symbol: "?", label: state || "Unknown" }) };
}

export function reasonSymbol(reason: string | null): DecisionSymbol {
  if (reason === null || reason === "") return { key: "unreported", symbol: "−", label: "Not reported" };
  const normalized = reason.toLowerCase();
  if (normalized === "max_pairs") return { key: reason, symbol: "✓", label: "Maximum pairs" };
  if (normalized.includes("prune")) return { key: reason, symbol: "↯", label: "Pruned" };
  if (normalized.includes("cancel")) return { key: reason, symbol: "−", label: "Cancelled" };
  if (normalized.includes("fail")) return { key: reason, symbol: "×", label: "Failed" };
  return { key: reason, symbol: "•", label: reason };
}

/** Plain-language meanings for the typed, persisted lifecycle reasons. */
export function decisionReasonDescription(reason: string): string {
  const descriptions: Record<string, string> = {
    below_min_pairs: "The report was recorded before the minimum pair count, so pruning did not apply.",
    pruning_disabled: "Pruning was disabled for this report, so the candidate continued.",
    startup_exempt: "The startup allowance exempted this candidate from pruning at this report.",
    hyperband_keep: "The candidate survived the observed Hyperband rung and continued.",
    confidence: "The candidate completed because the recorded confidence criterion was met.",
    max_pairs: "The candidate completed after reaching the configured maximum pair count.",
    hyperband_prune: "The candidate was pruned at the observed Hyperband rung.",
  };
  return descriptions[reason] ?? "The server recorded this decision reason without additional explanatory evidence.";
}

/**
 * Disjoint report-decision buckets. They deliberately key off the stored
 * reason rather than guessed thresholds, so their counts can be reconciled
 * exactly with the full-population report total.
 */
export function pruningFunnelRows(overview: TuningAnalysisOverview): PruningFunnelRow[] {
  const counts = new Map<string, number>();
  for (const group of overview.decision_groups) counts.set(group.reason, (counts.get(group.reason) ?? 0) + group.reports);
  const definitions: Array<Omit<PruningFunnelRow, "reports">> = [
    { key: "below_minimum", label: "Below minimum", reason: "below_min_pairs", description: decisionReasonDescription("below_min_pairs") },
    { key: "startup_exempt", label: "Startup exempt", reason: "startup_exempt", description: decisionReasonDescription("startup_exempt") },
    { key: "pruning_disabled", label: "Pruning disabled", reason: "pruning_disabled", description: decisionReasonDescription("pruning_disabled") },
    { key: "continued", label: "Continued", reason: "hyperband_keep", description: decisionReasonDescription("hyperband_keep") },
    { key: "pruned", label: "Pruned", reason: "hyperband_prune", description: decisionReasonDescription("hyperband_prune") },
    { key: "confidence_completed", label: "Confidence-completed", reason: "confidence", description: decisionReasonDescription("confidence") },
    { key: "max_completed", label: "Max-completed", reason: "max_pairs", description: decisionReasonDescription("max_pairs") },
  ];
  return definitions.map((definition) => ({ ...definition, reports: counts.get(definition.reason) ?? 0 }));
}

/** Every server-provided decision group, including zero-count groups. */
export function decisionGroupRows(overview: TuningAnalysisOverview): DecisionGroupRow[] {
  return overview.decision_groups
    .map((group) => ({ ...group, outcomeSymbol: stateSymbol(group.outcome), reasonSymbol: reasonSymbol(group.reason) }))
    .sort((a, b) => compareText(a.outcome, b.outcome) || compareText(a.reason, b.reason) || Number(a.pruning_exempt) - Number(b.pruning_exempt));
}

/** Funnel rows retain the API's exact aggregate values and only add display identity. */
export function rungFunnelRows(overview: TuningAnalysisOverview, selectedBracketId: string | null = null): RungFunnelRow[] {
  return overview.bracket_resources
    .map((row) => ({ ...row, bracket: bracketLabel(row.bracket_id), selected: row.bracket_id === selectedBracketId }))
    .sort((a, b) => {
      const bracket = a.bracket_id === null
        ? (b.bracket_id === null ? 0 : -1)
        : (b.bracket_id === null ? 1 : compareText(a.bracket_id, b.bracket_id));
      return bracket || a.resource - b.resource || (a.rung_resource ?? -1) - (b.rung_resource ?? -1);
    });
}

/**
 * One point per recorded report. No extrapolated terminal point is added, so
 * each path ends at its final recorded report even while a session is live.
 */
export function trialTrajectories(overview: TuningAnalysisOverview, selectedTrialId: string | null = null): TrialTrajectory[] {
  return trialTrajectoriesFromRows(exactPlotRows(overview, selectedTrialId));
}

/** Groups already-selected exact rows without changing their source snapshot. */
export function trialTrajectoriesFromRows(rows: readonly AnalysisPlotRow[]): TrialTrajectory[] {
  const grouped = new Map<string, AnalysisPlotRow[]>();
  for (const row of rows) {
    const rows = grouped.get(row.trial_id);
    if (rows) rows.push(row);
    else grouped.set(row.trial_id, [row]);
  }
  return [...grouped.entries()]
    .map(([trialId, rows]) => {
      const first = rows[0]!;
      const points = rows.map((row, index) => ({ ...row, terminal: index === rows.length - 1 }));
      return {
        trialId,
        trialNumber: first.trial_number,
        trialStatus: first.trial_status,
        selected: first.selected,
        points,
        reportCount: points.length,
      };
    })
    .sort((a, b) => a.trialNumber - b.trialNumber || compareText(a.trialId, b.trialId));
}

function addWld(left: WldSummary, right: WldSummary): WldSummary {
  const wins = left.wins + right.wins;
  const losses = left.losses + right.losses;
  const draws = left.draws + right.draws;
  return { wins, losses, draws, total: wins + losses + draws };
}

function trialWld(trial: TuningTrialSummary): WldSummary {
  return { wins: trial.wins, losses: trial.losses, draws: trial.draws, total: trial.wins + trial.losses + trial.draws };
}

function trialCompute(trial: TuningTrialSummary): ComputeSummary {
  return {
    elapsedMs: trial.elapsed_ms,
    searchIterationsTotal: trial.search_iterations_total,
    searchMoveTimeMs: trial.search_move_time_ms,
  };
}

function addCompute(left: ComputeSummary, right: ComputeSummary): ComputeSummary {
  return {
    elapsedMs: left.elapsedMs + right.elapsedMs,
    searchIterationsTotal: left.searchIterationsTotal + right.searchIterationsTotal,
    searchMoveTimeMs: left.searchMoveTimeMs + right.searchMoveTimeMs,
  };
}

/** W/L/D and compute totals for the currently returned page, never unseen rows. */
export function trialPageSummary(page: TuningTrialPage, selectedTrialId: string | null = null): TrialPageSummary {
  const rows = page.trials.map((trial) => ({ ...trial, wld: trialWld(trial), compute: trialCompute(trial), selected: trial.trial_id === selectedTrialId }));
  return {
    rows,
    wld: rows.reduce((total, row) => addWld(total, row.wld), { wins: 0, losses: 0, draws: 0, total: 0 }),
    compute: rows.reduce((total, row) => addCompute(total, row.compute), { elapsedMs: 0, searchIterationsTotal: 0, searchMoveTimeMs: 0 }),
    totalCount: page.total_count,
    returnedCount: rows.length,
    sampled: page.total_count > rows.length,
  };
}

/** Revision order and gaps are derived from persisted ordinals, not wall-clock time. */
export function poolRevisionCoverage(overview: TuningAnalysisOverview): PoolRevisionCoverage {
  const revisions = [...overview.pool_revisions]
    .sort((a, b) => a.display_ordinal - b.display_ordinal || compareText(a.pool_snapshot_fingerprint, b.pool_snapshot_fingerprint));
  let previous: number | null = null;
  const rows = revisions.map((revision) => {
    const gapBefore = previous === null ? [] : Array.from({ length: Math.max(0, revision.display_ordinal - previous - 1) }, (_, i) => previous! + i + 1);
    previous = revision.display_ordinal;
    return { ...revision, gapBefore, anchorCount: revision.anchors.length };
  });
  return {
    revisions: rows,
    revisionCount: rows.length,
    pairCount: rows.reduce((total, row) => total + row.pair_count, 0),
    anchorCount: rows.reduce((total, row) => total + row.anchorCount, 0),
    unmatchedPoolRevisions: overview.coverage.pairs.unmatched_pool_revisions,
  };
}

/** Distance is the recorded candidate rating before the pair versus that pair's recorded opponent. */
export function opponentDistances(trial: Pick<TuningTrialDetailView, "pairs">): OpponentDistance[] {
  return [...trial.pairs]
    .sort((a, b) => a.pair_index - b.pair_index || compareText(a.pair_id, b.pair_id))
    .map((pair: TuningTrialDetailPair) => ({
      pairId: pair.pair_id,
      pairIndex: pair.pair_index,
      opponentId: pair.opponent.anchor_id,
      candidateMu: pair.rating_before.mu,
      opponentMu: pair.opponent.mu,
      deltaMu: pair.rating_before.mu - pair.opponent.mu,
      absoluteMuDistance: Math.abs(pair.rating_before.mu - pair.opponent.mu),
      candidateSigma: pair.rating_before.sigma,
      opponentSigma: pair.opponent.sigma,
      deltaSigma: pair.rating_before.sigma - pair.opponent.sigma,
    }));
}

/** Adds selection identity without changing the source row objects. */
export function highlightSelectedTrial<T extends { trial_id: string }>(rows: readonly T[], selectedTrialId: string | null): Array<T & { selected: boolean }> {
  return rows.map((row) => ({ ...row, selected: row.trial_id === selectedTrialId }));
}
