import { describe, expect, it } from "vitest";
import {
  JOURNAL_POLL_MS,
  journalPollDelayMs,
  PROJECTION_REFRESH_MS,
  SCIENCE_STALE_REFRESH_MS,
  projectionRefreshDelayMs,
  EVIDENCE_POLL_MS,
  evidencePollDelayMs,
} from "../../src/tuner/tuner-poll.js";

describe("tuner-poll", () => {
  it("polls the journal on a fixed cadence while any run is live", () => {
    expect(journalPollDelayMs(1)).toBe(JOURNAL_POLL_MS);
    expect(journalPollDelayMs(4)).toBe(JOURNAL_POLL_MS);
  });

  it("stops journal polling once every run has exited", () => {
    expect(journalPollDelayMs(0)).toBeNull();
  });

  it("auto-refreshes the projection every few seconds while the open run is live", () => {
    expect(projectionRefreshDelayMs("live")).toBe(PROJECTION_REFRESH_MS);
    expect(PROJECTION_REFRESH_MS).toBe(6_000);
  });

  it("shortens the next refresh cycle when a new scientific event has landed", () => {
    expect(projectionRefreshDelayMs("live", true)).toBe(SCIENCE_STALE_REFRESH_MS);
    expect(SCIENCE_STALE_REFRESH_MS).toBeLessThan(PROJECTION_REFRESH_MS);
    // Stale only matters while live.
    expect(projectionRefreshDelayMs("exited", true)).toBeNull();
  });

  it("falls back to manual refresh when the open run is finished or absent", () => {
    expect(projectionRefreshDelayMs("exited")).toBeNull();
    expect(projectionRefreshDelayMs("unknown")).toBeNull();
    expect(projectionRefreshDelayMs("none")).toBeNull();
  });

  it("does not run the client loop while the evidence stream is healthy", () => {
    // The headless follower + `projection-updated` frame own freshness then.
    expect(projectionRefreshDelayMs("live", false, true)).toBeNull();
    expect(projectionRefreshDelayMs("live", true, true)).toBeNull();
    // Stream down while live -> the fallback loop runs.
    expect(projectionRefreshDelayMs("live", false, false)).toBe(PROJECTION_REFRESH_MS);
  });

  it("does not poll evidence while the SSE stream is healthy", () => {
    expect(evidencePollDelayMs(true, "live")).toBeNull();
  });

  it("polls the evidence tail only when the stream is degraded and the run is live", () => {
    expect(evidencePollDelayMs(false, "live")).toBe(EVIDENCE_POLL_MS);
    expect(evidencePollDelayMs(false, "exited")).toBeNull();
    expect(evidencePollDelayMs(false, "none")).toBeNull();
  });
});
