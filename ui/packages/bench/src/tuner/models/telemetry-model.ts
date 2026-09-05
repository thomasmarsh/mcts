// telemetry-model.ts — pure derivations over the wall-clock telemetry
// rollup (`GET .../projection/runs/{id}/telemetry`) and the per-phase
// compute ledger. Feeds the run header's vitals table and ETA (15f) and
// the per-contender W/L/D records (15g). No rendering, no `fetch` — the
// arithmetic the run loop deliberately keeps out of `evidence.jsonl`
// lives here instead.

import type {
  ProjectionComputePhase,
  ProjectionPairRow,
  ProjectionTelemetry,
} from "../tuner-types.js";

const sum = (xs: number[]): number => xs.reduce((a, b) => a + b, 0);

export interface VitalsInput {
  telemetry: ProjectionTelemetry;
  /** Per-phase compute ledger from `GET .../runs/{id}` — cumulative game
   * compute and the completed-pair count come from here. */
  compute: ProjectionComputePhase[];
  /** Resolved total pair budget (tuning + validation + diagnostic) when it
   * is known — from the run plan or the finished report. Drives the ETA;
   * `null` leaves `etaMs` null rather than guessing. */
  budgetPairs?: number | null;
  /** `"live"` uses `nowMs - startedAt` for the elapsed wall; otherwise the
   * telemetry sidecar's own extent is used. */
  live: boolean;
  /** ISO timestamp from the run's journal row. */
  startedAt?: string | null;
  nowMs: number;
}

export interface RunVitals {
  /** Wall time the run has been going, spanning any resume sleep gap. */
  elapsedWallMs: number;
  /** Cumulative game-subprocess compute (`compute_phases.wall_time_ms`) --
   * exceeds elapsed wall once parallelism > 1. */
  gameComputeMs: number;
  /** Time the loop blocked on game subprocesses (the `wait` lane). */
  waitMs: number;
  /** Run-loop thread time excluding the games it waits on. */
  loopActiveMs: number;
  /** `gameComputeMs / waitMs` -- games in flight per wall-second of waiting.
   * `null` before any wait span is recorded. */
  effectiveParallelism: number | null;
  /** `loopActiveMs / (loopActiveMs + waitMs)` -- how much of the critical
   * path is the loop's own single-threaded work. `null` before any span. */
  loopOverheadFraction: number | null;
  completedPairs: number;
  /** Lifetime average, `null` until a pair completes. */
  secPerPair: number | null;
  /** `(budgetPairs - completedPairs) * secPerPair * 1000`, clamped at 0.
   * `null` when the budget or the rate is unknown. */
  etaMs: number | null;
  /** Run-loop processes that have touched the run; > 1 means it resumed. */
  sessions: number;
}

export function deriveVitals(input: VitalsInput): RunVitals {
  const { telemetry, compute } = input;
  const waitMs = telemetry.wait_us / 1000;
  const loopActiveMs = telemetry.loop_active_us / 1000;
  const gameComputeMs = sum(compute.map((c) => c.wall_time_ms));
  const completedPairs = sum(compute.map((c) => c.completed_pairs));

  const startedMs = input.startedAt ? Date.parse(input.startedAt) : NaN;
  const elapsedWallMs =
    input.live && !Number.isNaN(startedMs)
      ? Math.max(0, input.nowMs - startedMs)
      : telemetry.wall_span_us / 1000;

  const effectiveParallelism = waitMs > 0 ? gameComputeMs / waitMs : null;
  const critical = loopActiveMs + waitMs;
  const loopOverheadFraction = critical > 0 ? loopActiveMs / critical : null;

  const secPerPair = completedPairs > 0 ? elapsedWallMs / 1000 / completedPairs : null;
  const remaining =
    input.budgetPairs != null ? Math.max(0, input.budgetPairs - completedPairs) : null;
  const etaMs =
    remaining != null && secPerPair != null ? remaining * secPerPair * 1000 : null;

  return {
    elapsedWallMs,
    gameComputeMs,
    waitMs,
    loopActiveMs,
    effectiveParallelism,
    loopOverheadFraction,
    completedPairs,
    secPerPair,
    etaMs,
    sessions: telemetry.sessions,
  };
}

export interface ContenderRecord {
  opponentId: string;
  wins: number;
  losses: number;
  draws: number;
}

export interface ContenderRecords {
  candidateId: string;
  /** One entry per distinct opponent this contender was paired against,
   * ordered by opponent id. */
  byOpponent: ContenderRecord[];
  total: ContenderRecord;
}

/** Roll a candidate's pair rows up into a W/L/D record per panel opponent.
 * Each pair is one paired (seat-swapped) match; its `pair_utility` sign is
 * the paired outcome from the candidate's side (`> 0` win, `< 0` loss, `0`
 * draw). Pass the rows for a single candidate. */
export function deriveContenderRecords(
  candidateId: string,
  pairs: ProjectionPairRow[],
): ContenderRecords {
  const mine = pairs.filter((p) => p.candidate_id === candidateId);
  const byId = new Map<string, ContenderRecord>();
  const total: ContenderRecord = { opponentId: "", wins: 0, losses: 0, draws: 0 };

  for (const pair of mine) {
    let record = byId.get(pair.opponent_id);
    if (!record) {
      record = { opponentId: pair.opponent_id, wins: 0, losses: 0, draws: 0 };
      byId.set(pair.opponent_id, record);
    }
    const bucket = pair.pair_utility > 0 ? "wins" : pair.pair_utility < 0 ? "losses" : "draws";
    record[bucket] += 1;
    total[bucket] += 1;
  }

  const byOpponent = [...byId.values()].sort((a, b) =>
    a.opponentId < b.opponentId ? -1 : a.opponentId > b.opponentId ? 1 : 0,
  );
  return { candidateId, byOpponent, total };
}

export function formatRecord(record: ContenderRecord): string {
  return `${record.wins}–${record.losses}–${record.draws}`;
}
