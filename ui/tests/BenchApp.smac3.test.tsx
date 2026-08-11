// tests/BenchApp.smac3.test.tsx — Component-level tests for the SMAC3
// launch fields and run-detail panel, following the same pattern as
// BenchApp.test.tsx: `@solidjs/testing-library` + a real `createStore`
// against a mocked `BenchEnv` from `fixtures/fake-bench.js`, no live
// server or browser.

import { afterEach, describe, expect, it } from "vitest";
import { render, screen, fireEvent, cleanup } from "@solidjs/testing-library";
import { createStore, Effect } from "@mcts/core";
import {
  benchReducer,
  initialBenchState,
  type BenchState,
  type BenchAction,
  type BenchEnv,
  LaunchForm,
  RunDetailPanel,
} from "@mcts/bench";
import {
  createMockBenchEnv,
  FAKE_SMAC3_RUN_ID,
  fakeKinds,
  fakeTrialRowsWithRepeats,
} from "./fixtures/fake-bench.js";

function createTestStore(envOverrides?: Partial<BenchEnv>) {
  const env = createMockBenchEnv(envOverrides);
  const store = createStore<BenchState, BenchAction>(initialBenchState(), benchReducer, env);
  store.dispatch({ tag: "kinds", action: { tag: "request" } });
  store.dispatch({ tag: "smac3Kinds", action: { tag: "request" } });
  store.dispatch({ tag: "runs", action: { tag: "request" } });
  return { store, env };
}

afterEach(() => {
  cleanup();
});

describe("LaunchForm / smac3", () => {
  it("selecting the smac3 kind auto-selects the tunable game and renders its parameter space", () => {
    const { store } = createTestStore();
    render(() => <LaunchForm store={store} />);

    fireEvent.change(screen.getByLabelText("Run Kind"), { target: { value: "smac3" } });

    // The regular round_robin strategy picker never appears for smac3.
    expect(screen.queryByText(/select at least 2/i)).not.toBeInTheDocument();

    // Game auto-selects the (only) tunable game, and its search space
    // renders read-only, driven purely by the /smac3/kinds metadata.
    const gameSelect = screen.getByLabelText("Game") as HTMLSelectElement;
    expect(gameSelect.value).toBe("traffic-lights");
    expect(screen.getByText("strong")).toBeInTheDocument(); // baseline
    expect(screen.getByText("schedule")).toBeInTheDocument(); // a parameter name
    expect(screen.getByText("epsilon")).toBeInTheDocument(); // another parameter name
    expect(screen.getByText(/schedule = threshold/)).toBeInTheDocument(); // a condition

    // Budget fields are present with their documented defaults.
    expect(screen.getByLabelText("Trials")).toBeInTheDocument();
    expect(screen.getByLabelText("Seed")).toBeInTheDocument();
  });

  it("submitting builds --override argv from the budget fields, not a strategies list", () => {
    const seen: { kind: string; game: string; config?: unknown }[] = [];
    const { store } = createTestStore({
      launchRun: (kind, game, config) => {
        seen.push({ kind, game, config });
        return createMockBenchEnv().launchRun(kind, game, config);
      },
    });
    render(() => <LaunchForm store={store} />);

    fireEvent.change(screen.getByLabelText("Run Kind"), { target: { value: "smac3" } });
    fireEvent.input(screen.getByLabelText("Trials"), { target: { value: "25" } });
    fireEvent.click(screen.getByLabelText(/Deterministic/));

    const launchBtn = screen.getByText("Launch") as HTMLButtonElement;
    expect(launchBtn.disabled).toBe(false);
    fireEvent.click(launchBtn);

    expect(seen).toEqual([
      {
        kind: "smac3",
        game: "traffic-lights",
        config: { overrides: ["optimizer.n_trials=25", "optimizer.deterministic=True", "optimizer.seed=42"] },
      },
    ]);
  });

  it("omits optimizer.n_workers entirely when the Workers field is left blank", () => {
    const seen: unknown[] = [];
    const { store } = createTestStore({
      launchRun: (_kind, _game, config) => {
        seen.push(config);
        return createMockBenchEnv().launchRun(_kind, _game, config);
      },
    });
    render(() => <LaunchForm store={store} />);

    fireEvent.change(screen.getByLabelText("Run Kind"), { target: { value: "smac3" } });
    fireEvent.click(screen.getByText("Launch"));

    const overrides = (seen[0] as { overrides: string[] }).overrides;
    expect(overrides.some((o) => o.startsWith("optimizer.n_workers"))).toBe(false);

    // ... and includes it when the field has a value.
    seen.length = 0;
    fireEvent.input(screen.getByLabelText("Workers"), { target: { value: "4" } });
    fireEvent.click(screen.getByText("Launch"));
    const overridesWithWorkers = (seen[0] as { overrides: string[] }).overrides;
    expect(overridesWithWorkers).toContain("optimizer.n_workers=4");
  });

  it("falls back to the round_robin games list for other kinds, unaffected by smac3Kinds", () => {
    const { store } = createTestStore();
    render(() => <LaunchForm store={store} />);

    fireEvent.change(screen.getByLabelText("Run Kind"), { target: { value: "round_robin" } });
    expect((screen.getByLabelText("Game") as HTMLSelectElement).value).toBe(fakeKinds[0]!.games[0]!.game);
    expect(screen.getByText("Strong")).toBeInTheDocument();
  });
});

describe("RunDetailPanel / smac3", () => {
  it("renders trial stats, the best-vs-default table, and the trial history once trials land", async () => {
    const { store } = createTestStore();
    render(() => <RunDetailPanel store={store} />);

    store.dispatch({ tag: "openRun", runId: FAKE_SMAC3_RUN_ID });

    // fakeSmac3RunDetail is already terminal, so the single tail tick fetches
    // trials in the same round-trip -- no manual tick-forcing needed here.
    await screen.findByText("Best cost (loss rate)");

    // Best trial is #2 (cost 0.3 -- the lowest of the three fixture rows).
    // Scoped by selector since e.g. "3" (trial count) and "30.0%" (best
    // cost) also appear elsewhere (a trial_id cell, a per-trial cost cell).
    expect(screen.getByText("#2", { selector: ".smac3-stat-value" })).toBeInTheDocument();
    expect(screen.getByText("30.0%", { selector: ".smac3-stat-value" })).toBeInTheDocument();
    expect(screen.getByText("3", { selector: ".smac3-stat-value" })).toBeInTheDocument(); // trial count

    // Trial history lists all three rows' costs.
    for (const pct of ["55.0%", "30.0%", "40.0%"]) {
      expect(screen.getAllByText(pct).length).toBeGreaterThanOrEqual(1);
    }

    // The trial table's Family column shows each trial's family, not just
    // RAVE's -- fixture spans rave/ucb1_tuned/ucb1.
    const familyCells = document.querySelectorAll(".smac3-trial-family");
    expect(Array.from(familyCells).map((c) => c.textContent)).toEqual(["ucb1", "ucb1_tuned", "rave"]);

    // Best trial (#2) is `family: "ucb1_tuned"`, not the search space's
    // default `family: "rave"` -- the best-vs-default diff table must
    // compare across two different families' configs, not two RAVE
    // configs, and flag every param that differs from its own default.
    const diffTable = document.querySelector("#smac3-diff-table")!;
    expect(diffTable.textContent).toContain("ucb1_tuned");
    const familyRow = Array.from(diffTable.querySelectorAll("tbody tr")).find(
      (row) => row.querySelector(".smac3-param-name")?.textContent === "family",
    )!;
    expect(familyRow.classList.contains("smac3-diff-changed")).toBe(true);
    expect(familyRow.textContent).toContain("rave"); // the default, shown in the Default column
  });

  it("pools trials with an identical config into one confidence-band group", async () => {
    const { store } = createTestStore({
      getRunTrials: () => Effect.send(fakeTrialRowsWithRepeats),
    });
    render(() => <RunDetailPanel store={store} />);

    store.dispatch({ tag: "openRun", runId: FAKE_SMAC3_RUN_ID });

    await screen.findByText("Best cost (loss rate)");

    // fakeTrialRowsWithRepeats adds two more evaluations (#4, #5) of trial
    // #2's exact config (cost 0.25/0.35 vs #2's 0.3) -- #4 is now the
    // lowest single cost, but all three share one group.
    expect(screen.getByText("#4", { selector: ".smac3-stat-value" })).toBeInTheDocument();
    expect(screen.getByText("3", { selector: ".smac3-stat-value" })).toBeInTheDocument(); // Evaluations

    expect(screen.getByText("Evaluations", { selector: ".smac3-stat-label" })).toBeInTheDocument();
    expect(screen.getByText("95% CI", { selector: ".smac3-stat-label" })).toBeInTheDocument();

    // The pooled interval must actually straddle the group's mean cost
    // (30.0%, i.e. (0.25 + 0.3 + 0.35) / 3) -- not be a degenerate
    // single-point estimate.
    const ciStat = Array.from(document.querySelectorAll(".smac3-stat")).find(
      (el) => el.querySelector(".smac3-stat-label")?.textContent === "95% CI",
    )!;
    const ciText = ciStat.querySelector(".smac3-stat-value")!.textContent!;
    const [lo, hi] = ciText.split("–").map((s) => parseFloat(s));
    expect(lo).toBeLessThan(30.0);
    expect(hi).toBeGreaterThan(30.0);

    // Every point sharing that config renders a (non-degenerate) whisker.
    const whiskers = document.querySelectorAll(".smac3-ci-whisker");
    expect(whiskers.length).toBe(5); // one per scored trial (#1, #2, #3, #4, #5)
  });
});
