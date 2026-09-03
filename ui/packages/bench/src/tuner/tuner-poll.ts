// tuner-poll.ts — pure "how long until the next poll" decisions for the
// tuner fleet. No timers, no fetch: the reducer calls these to size an
// `Effect.delay`, exactly as the round-robin reducer uses `tailDelayMs`.
//
// Two independent cadences:
//   - the fleet journal (`GET /runs`): cheap, liveness only. Poll while any
//     run is `live`; stop once every run has exited.
//   - the projection refresh for the *open* run: expensive (re-runs the
//     projector). Auto-refresh only while that run is `live`; once it has
//     exited the run-dir is authority and won't change, so switch to a
//     manual "Refresh science" button.

export const JOURNAL_POLL_MS = 3_000;
/** Auto-refresh cadence for the open run's projection while it is live. Fast
 * enough that convergence / funnel / race / observations move on-screen every
 * few pair completions without a manual "Refresh science". */
export const PROJECTION_REFRESH_MS = 6_000;
/** One accelerated cycle immediately after the evidence stream reports a new
 * *scientific* event, so a burst of pair completions pulls the science
 * forward within ~3 s instead of waiting the full cadence. */
export const SCIENCE_STALE_REFRESH_MS = 3_000;
/** Fallback cadence for the open run's evidence tail when the SSE stream is
 * degraded (push failed). While the stream is healthy the UI does not poll. */
export const EVIDENCE_POLL_MS = 1_500;

/** Milliseconds until the next fleet-journal poll, or `null` to stop
 * polling. `liveRunCount` is how many journal rows report `status: "live"`. */
export function journalPollDelayMs(liveRunCount: number): number | null {
  return liveRunCount > 0 ? JOURNAL_POLL_MS : null;
}

export type OpenRunLiveness = "live" | "exited" | "unknown" | "none";

/** Milliseconds until the next automatic projection refresh for the open
 * run, or `null` when the UI should fall back to a manual refresh button
 * (no run open, or the open run has finished). `scienceStale` shortens the
 * next cycle to `SCIENCE_STALE_REFRESH_MS` after a new scientific evidence
 * event has landed since the last refresh. */
export function projectionRefreshDelayMs(
  open: OpenRunLiveness,
  scienceStale = false,
): number | null {
  if (open !== "live") return null;
  return scienceStale ? SCIENCE_STALE_REFRESH_MS : PROJECTION_REFRESH_MS;
}

/** Milliseconds until the next evidence-tail poll, or `null` to not poll.
 * `null` while the SSE stream is healthy (push, no poll) or the run is not
 * `live`; `EVIDENCE_POLL_MS` only when the stream is degraded and the run is
 * still live. */
export function evidencePollDelayMs(
  streamOk: boolean,
  status: OpenRunLiveness,
): number | null {
  if (streamOk || status !== "live") return null;
  return EVIDENCE_POLL_MS;
}
