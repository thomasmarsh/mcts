// progress-model.ts — pure derivation behind `<ProgressRail>`. The rail
// summarises "how far along is this run" from two inputs the UI already
// has: the journal (liveness + started-at) and, once the projection has
// caught up, the per-phase compute ledger. No arithmetic lives in the
// component -- the presentational primitives stay layout-only.

import type { ProjectionComputePhase, TunerRunLiveness } from "../tuner-types.js";

export interface ProgressSummary {
  /** Coarse phase label for the rail's headline. */
  phase: string;
  live: boolean;
  wallMs: number;
  pairs: {
    completed: number;
    attempted: number;
    failed: number;
    censored: number;
  };
  /** completed / attempted, 0 when nothing has been attempted yet. */
  fraction: number;
}

export interface ProgressInput {
  status: TunerRunLiveness | null;
  /** ISO timestamp from the journal row. */
  startedAt: string | null;
  nowMs: number;
  /** Per-phase compute ledger from the projection, when available. */
  compute?: ProjectionComputePhase[];
}

const sum = (xs: number[]): number => xs.reduce((a, b) => a + b, 0);

export function deriveProgress(input: ProgressInput): ProgressSummary {
  const live = input.status === "live";
  const compute = input.compute ?? [];

  const completed = sum(compute.map((c) => c.completed_pairs));
  const attempted = sum(compute.map((c) => c.pair_attempts));
  const failed = sum(compute.map((c) => c.failed_attempts));
  const censored = sum(compute.map((c) => c.censored_attempts));

  const startedMs = input.startedAt ? Date.parse(input.startedAt) : NaN;
  const wallMs = live
    ? Number.isNaN(startedMs)
      ? 0
      : Math.max(0, input.nowMs - startedMs)
    : sum(compute.map((c) => c.wall_time_ms));

  const phase =
    compute.length > 0
      ? compute[compute.length - 1]!.phase
      : live
        ? "starting"
        : "—";

  return {
    phase,
    live,
    wallMs,
    pairs: { completed, attempted, failed, censored },
    fraction: attempted > 0 ? completed / attempted : 0,
  };
}

export function formatWall(ms: number): string {
  const total = Math.floor(ms / 1000);
  const h = Math.floor(total / 3600);
  const m = Math.floor((total % 3600) / 60);
  const s = total % 60;
  if (h > 0) return `${h}h ${m}m`;
  if (m > 0) return `${m}m ${s}s`;
  return `${s}s`;
}
