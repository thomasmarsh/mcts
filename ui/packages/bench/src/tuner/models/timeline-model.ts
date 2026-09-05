// timeline-model.ts — pure derivation over the wall-clock telemetry rollup
// (`GET .../projection/runs/{id}/telemetry`). Answers "where does the run's
// wall time go": the run loop's own work versus waiting on game
// subprocesses, and the split across loop stages. Non-scientific — the
// sidecar carries the timestamps `evidence.jsonl` omits and no replay reads
// it. A real flame graph is one Perfetto download away; this is the
// glanceable in-app summary.

import type { ProjectionTelemetry } from "../tuner-types.js";

export interface TimelineLane {
  name: string;
  /** ms from the run's first recorded span to this lane's first span. */
  offsetMs: number;
  /** ms from this lane's first span to its last span's end. */
  extentMs: number;
  /** Total recorded span time in this lane (≤ `extentMs`). */
  activeMs: number;
  spanCount: number;
  maxMs: number;
}

export interface TimelineStage {
  stage: string;
  ms: number;
  /** `ms / Σ activeMs`; the fractions sum to 1 when any stage has time. */
  fraction: number;
}

export interface Timeline {
  /** At least one non-`session` lane has spans. */
  present: boolean;
  /** `wall_span_us` in ms — spans a resumed run's sleep gap. */
  wallMs: number;
  sessions: number;
  lanes: TimelineLane[];
  /** Descending by time. */
  byStage: TimelineStage[];
  duty: {
    /** Run-loop thread time excluding the `wait` lane. */
    loopMs: number;
    /** `wait` lane total — blocked on game subprocesses. */
    waitMs: number;
    /** `loopMs / (loopMs + waitMs)`; `null` before any span. */
    loopFraction: number | null;
    /** "run loop N% · waiting on games M%", or the empty-state line. */
    headline: string;
  };
}

const NON_STAGE = new Set(["session"]);

export function deriveTimeline(
  telemetry: ProjectionTelemetry | null | undefined,
): Timeline {
  const origin = telemetry?.first_start_us ?? 0;
  const wallMs = (telemetry?.wall_span_us ?? 0) / 1000;
  const stageLanes = (telemetry?.lanes ?? []).filter(
    (l) => !NON_STAGE.has(l.name) && l.span_count > 0,
  );

  const lanes: TimelineLane[] = stageLanes
    .map((l) => ({
      name: l.name,
      offsetMs: Math.max(0, (l.first_start_us - origin) / 1000),
      extentMs: Math.max(0, (l.last_end_us - l.first_start_us) / 1000),
      activeMs: l.total_us / 1000,
      spanCount: l.span_count,
      maxMs: l.max_us / 1000,
    }))
    .sort((a, b) => a.offsetMs - b.offsetMs || a.name.localeCompare(b.name));

  const totalActive = lanes.reduce((s, l) => s + l.activeMs, 0);
  const byStage: TimelineStage[] = lanes
    .map((l) => ({
      stage: l.name,
      ms: l.activeMs,
      fraction: totalActive > 0 ? l.activeMs / totalActive : 0,
    }))
    .sort((a, b) => b.ms - a.ms);

  const waitMs = (telemetry?.wait_us ?? 0) / 1000;
  const loopMs = Math.max(0, (telemetry?.loop_active_us ?? 0) / 1000 - waitMs);
  const critical = loopMs + waitMs;
  const loopFraction = critical > 0 ? loopMs / critical : null;
  const pct = (x: number): string => `${Math.round((x / critical) * 100)}%`;
  const headline =
    critical > 0
      ? `run loop ${pct(loopMs)} · waiting on games ${pct(waitMs)}`
      : "no timing recorded for this run";

  return {
    present: lanes.length > 0,
    wallMs,
    sessions: telemetry?.sessions ?? 0,
    lanes,
    byStage,
    duty: { loopMs, waitMs, loopFraction, headline },
  };
}
