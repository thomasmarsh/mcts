// tuner/index.ts — public surface of the version-4 tuner UI.

export { TunerApp } from "./TunerApp.js";
export { FleetDashboard } from "./views/FleetDashboard.js";
export { LaunchForm as TunerLaunchForm } from "./views/LaunchForm.js";
export { ObjectiveManager } from "./views/ObjectiveManager.js";
export { ObjectiveEditor } from "./views/ObjectiveEditor.js";
export { RunOverview } from "./views/RunOverview.js";
export { RunScience } from "./views/RunScience.js";
export { RunEvidence } from "./views/RunEvidence.js";
export { CandidateDrawer } from "./views/CandidateDrawer.js";
export { PairInspector } from "./views/PairInspector.js";

export { RunStatusBadge } from "./primitives/RunStatusBadge.js";
export { ProgressRail } from "./primitives/ProgressRail.js";
export { RunCard } from "./primitives/RunCard.js";
export { IntervalBar } from "./primitives/IntervalBar.js";
export { Forest, type ForestRow } from "./primitives/Forest.js";
export { DataTable, type DataColumn } from "./primitives/DataTable.js";
export { CandidateChip } from "./primitives/CandidateChip.js";
export { ConfigDiff } from "./primitives/ConfigDiff.js";
export { ShipVerdict } from "./primitives/ShipVerdict.js";
export { CopyPresetButton } from "./primitives/CopyPresetButton.js";
export { StepLine, type StepPoint } from "./primitives/StepLine.js";
export { FunnelBars, type FunnelRow } from "./primitives/FunnelBars.js";
export { KpiRow, type KpiItem } from "./primitives/KpiRow.js";
export { RaceStrip, dispositionClass, type RaceStripRow } from "./primitives/RaceStrip.js";
export { Heatmap, type HeatmapCell, type HeatmapRow } from "./primitives/Heatmap.js";
export { CycleGraph, type CycleGraphNode, type CycleGraphEdge } from "./primitives/CycleGraph.js";
export { Treemap, type TreemapGroup, type TreemapChild } from "./primitives/Treemap.js";
export { GameSummaryStrip } from "./primitives/GameSummaryStrip.js";
export { JsonDrawer } from "./primitives/JsonDrawer.js";

export {
  deriveVerdict,
  shortCandidateId,
  type ShipVerdict as ShipVerdictModel,
  type VerdictCandidate,
  type VerdictInput,
} from "./models/verdict-model.js";
export {
  flattenConfig,
  schemaDefaults,
  configDiffRows,
  type ConfigDiffRow,
} from "./models/config-diff-model.js";
export { buildPreset, type PresetSpec, type PresetCopyResult } from "./models/preset-copy.js";
export {
  derivePairInspector,
  type PairInspectorView,
  type PairGameView,
  type PairResult,
} from "./models/pair-model.js";
export { deriveProposalFunnel, type ProposalFunnel, type ProposalStage } from "./models/funnel-model.js";
export {
  deriveCohortRace,
  type RaceGraph,
  type CohortRace,
  type RaceRow,
} from "./models/race-model.js";
export {
  deriveConvergence,
  deriveObservations,
  type Convergence,
  type ConvergenceStep,
  type Observations,
  type ObservationRow,
} from "./models/science-models.js";
export { deriveElimination, type EliminationView, type CalibrationBin } from "./models/elimination-model.js";
export {
  deriveOpponentResponse,
  type OpponentResponseView,
  type OpponentRow,
  type OpponentCell,
} from "./models/opponent-model.js";
export {
  deriveDiagnosticGraph,
  type DiagnosticGraphView,
  type DiagnosticNode,
  type DiagnosticEdge,
} from "./models/diagnostic-model.js";
export {
  deriveComputeLedger,
  type ComputeLedgerView,
  type PhaseLedger,
} from "./models/compute-model.js";

export {
  tunerReducer,
  initialTunerState,
  LOG_TAIL_MS,
  type TunerState,
  type TunerAction,
  type TunerLaunchState,
  type TunerLogTailState,
} from "./tuner-reducer.js";
export { createTunerApiClient, type TunerApiClient } from "./tuner-api-client.js";
export { createTunerEnv, type TunerEnv } from "./tuner-env.js";
export { parseTunerHash, tunerHash, type TunerRoute, type RunTab } from "./tuner-routes.js";
export {
  JOURNAL_POLL_MS,
  PROJECTION_REFRESH_MS,
  journalPollDelayMs,
  projectionRefreshDelayMs,
  type OpenRunLiveness,
} from "./tuner-poll.js";
export { idle, toLoading, toOk, toErr, peek, isLoading, type RemoteData } from "./remote-data.js";
export {
  deriveProgress,
  formatWall,
  type ProgressSummary,
  type ProgressInput,
} from "./models/progress-model.js";
export type * from "./tuner-types.js";
