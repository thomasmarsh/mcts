import { describe, expect, it } from "vitest";

import { deriveTimeline } from "../../src/tuner/models/timeline-model.js";
import type {
  ProjectionTelemetry,
  ProjectionTelemetryLane,
} from "../../src/tuner/tuner-types.js";

const lane = (over: Partial<ProjectionTelemetryLane>): ProjectionTelemetryLane => ({
  name: "fold",
  span_count: 1,
  total_us: 0,
  max_us: 0,
  first_start_us: 1_000_000,
  last_end_us: 1_000_000,
  ...over,
});

const telemetry = (over: Partial<ProjectionTelemetry>): ProjectionTelemetry => ({
  run_id: "r",
  sessions: 1,
  first_start_us: 1_000_000,
  last_end_us: 11_000_000,
  wall_span_us: 10_000_000,
  loop_active_us: 0,
  wait_us: 0,
  lanes: [],
  ...over,
});

describe("deriveTimeline", () => {
  it("normalises lanes to [0, wall], computes byStage fractions, and the loop/wait split", () => {
    const t = telemetry({
      loop_active_us: 10_000_000, // 4s wait + 6s other loop work
      wait_us: 4_000_000,
      lanes: [
        lane({ name: "session", span_count: 2, total_us: 0 }),
        lane({
          name: "fold",
          span_count: 5,
          total_us: 2_000_000,
          max_us: 800_000,
          first_start_us: 1_000_000,
          last_end_us: 6_000_000,
        }),
        lane({
          name: "wait",
          span_count: 5,
          total_us: 4_000_000,
          first_start_us: 3_000_000,
          last_end_us: 11_000_000,
        }),
      ],
    });
    const out = deriveTimeline(t);
    expect(out.present).toBe(true);
    expect(out.wallMs).toBe(10_000);
    // `session` is dropped; lanes ordered by offset.
    expect(out.lanes.map((l) => l.name)).toEqual(["fold", "wait"]);
    const wait = out.lanes[1]!;
    expect(wait.offsetMs).toBe(2_000);
    expect(wait.extentMs).toBe(8_000);
    expect(wait.activeMs).toBe(4_000);
    const totalFraction = out.byStage.reduce((s, x) => s + x.fraction, 0);
    expect(totalFraction).toBeCloseTo(1);
    expect(out.duty.waitMs).toBe(4_000);
    expect(out.duty.loopMs).toBe(6_000);
    expect(out.duty.loopFraction).toBeCloseTo(0.6);
    expect(out.duty.headline).toBe("run loop 60% · waiting on games 40%");
  });

  it("reports the empty state with no telemetry", () => {
    const out = deriveTimeline(null);
    expect(out.present).toBe(false);
    expect(out.lanes).toEqual([]);
    expect(out.byStage).toEqual([]);
    expect(out.duty.headline).toBe("no timing recorded for this run");
  });

  it("treats a sidecar with only a session marker as not present", () => {
    const out = deriveTimeline(
      telemetry({ lanes: [lane({ name: "session", span_count: 1, total_us: 0 })] }),
    );
    expect(out.present).toBe(false);
  });
});
