import { describe, expect, it } from "vitest";
import { parseTunerHash, tunerHash } from "../../src/tuner/tuner-routes.js";

describe("tuner-routes", () => {
  it("defaults anything non-tuner to the fleet", () => {
    expect(parseTunerHash("")).toEqual({ view: "fleet" });
    expect(parseTunerHash("#/games")).toEqual({ view: "fleet" });
    expect(parseTunerHash("#/tuner")).toEqual({ view: "fleet" });
  });

  it("parses the launch route", () => {
    expect(parseTunerHash("#/tuner/launch")).toEqual({ view: "launch" });
  });

  it("parses a run route with its tab, decoding the id", () => {
    expect(parseTunerHash("#/tuner/run/nim%2Fabc")).toEqual({
      view: "run",
      runId: "nim/abc",
      tab: "overview",
    });
    expect(parseTunerHash("#/tuner/run/r1/science")).toEqual({
      view: "run",
      runId: "r1",
      tab: "science",
    });
    expect(parseTunerHash("#/tuner/run/r1/bogus")).toEqual({
      view: "run",
      runId: "r1",
      tab: "overview",
    });
  });

  it("carries the candidate drawer query param", () => {
    expect(parseTunerHash("#/tuner/run/r1/science?candidate=candidate-abc")).toEqual({
      view: "run",
      runId: "r1",
      tab: "science",
      candidate: "candidate-abc",
    });
    expect(parseTunerHash("#/tuner/run/r1?candidate=c1")).toEqual({
      view: "run",
      runId: "r1",
      tab: "overview",
      candidate: "c1",
    });
  });

  it("round-trips through tunerHash", () => {
    for (const route of [
      { view: "fleet" as const },
      { view: "launch" as const },
      { view: "run" as const, runId: "r/1", tab: "overview" as const },
      { view: "run" as const, runId: "r1", tab: "evidence" as const },
      { view: "run" as const, runId: "r1", tab: "overview" as const, candidate: "candidate-x/y" },
      { view: "run" as const, runId: "r1", tab: "science" as const, candidate: "c2" },
    ]) {
      expect(parseTunerHash(tunerHash(route))).toEqual(route);
    }
  });
});
