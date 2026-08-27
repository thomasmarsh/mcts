import { describe, expect, it } from "vitest";
import { formatObservedResult, formatProgress, statusLabel } from "../src/result-format.js";

describe("result formatting", () => {
  it("presents returned rates and intervals without recalculating them", () => {
    expect(formatProgress(4, 10)).toBe("4/10");
    expect(
      formatObservedResult({ completed_games: 4, win_rate: 0.75, ci_lower: 0.4, ci_upper: 0.9 }),
    ).toBe("75.0% (95% CI 40.0% – 90.0%)");
  });

  it("does not present the neutral default as an observation", () => {
    const cell = { completed_games: 0, win_rate: 0.5, ci_lower: 0, ci_upper: 1 };
    expect(formatObservedResult(cell)).toBe("No games yet");
  });

  it("keeps status labels readable", () => {
    expect(statusLabel("completed_with_errors")).toBe("completed with errors");
    expect(statusLabel("cancelled")).toBe("cancelled");
  });
});
