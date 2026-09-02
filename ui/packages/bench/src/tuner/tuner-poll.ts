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
export const PROJECTION_REFRESH_MS = 20_000;

/** Milliseconds until the next fleet-journal poll, or `null` to stop
 * polling. `liveRunCount` is how many journal rows report `status: "live"`. */
export function journalPollDelayMs(liveRunCount: number): number | null {
  return liveRunCount > 0 ? JOURNAL_POLL_MS : null;
}

export type OpenRunLiveness = "live" | "exited" | "unknown" | "none";

/** Milliseconds until the next automatic projection refresh for the open
 * run, or `null` when the UI should fall back to a manual refresh button
 * (no run open, or the open run has finished). */
export function projectionRefreshDelayMs(open: OpenRunLiveness): number | null {
  return open === "live" ? PROJECTION_REFRESH_MS : null;
}
