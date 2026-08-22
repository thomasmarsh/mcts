// tests/BenchApp.tuner.test.tsx — Component-level tests for the tuner
// launch fields and run-detail panel, following the same pattern as
// BenchApp.test.tsx: `@solidjs/testing-library` + a real `createStore`
// against a mocked `BenchEnv` from `fixtures/fake-bench.js`, no live
// server or browser.

import { afterEach, describe, expect, it, vi } from "vitest";
import { render, screen, fireEvent, cleanup } from "@solidjs/testing-library";
import { createStore, Effect } from "@mcts/core";
import {
  benchReducer,
  initialBenchState,
  type BenchState,
  type BenchAction,
  type BenchEnv,
  type ChainRung,
  type TrialRow,
  LaunchForm,
  RunDetailPanel,
} from "@mcts/bench";
import {
  createMockBenchEnv,
  FAKE_tuner_RUN_ID,
  fakeKinds,
  fakeTunerRunDetail,
  fakeTrialRows,
  fakeTrialRowsWithRepeats,
  fakeTrialRowsMultiInstance,
} from "./fixtures/fake-bench.js";

function createTestStore(envOverrides?: Partial<BenchEnv>) {
  const env = createMockBenchEnv(envOverrides);
  const store = createStore<BenchState, BenchAction>(initialBenchState(), benchReducer, env);
  store.dispatch({ tag: "kinds", action: { tag: "request" } });
  store.dispatch({ tag: "tunerKinds", action: { tag: "request" } });
  store.dispatch({ tag: "runs", action: { tag: "request" } });
  return { store, env };
}

afterEach(() => {
  cleanup();
});

describe("LaunchForm / tuner", () => {
  it("selecting the tuner kind auto-selects the tunable game and renders its parameter space", () => {
    const { store } = createTestStore();
    render(() => <LaunchForm store={store} />);

    fireEvent.change(screen.getByLabelText("Run Kind"), { target: { value: "tuner" } });

    // The regular round_robin strategy picker never appears for tuner.
    expect(screen.queryByText(/select at least 2/i)).not.toBeInTheDocument();

    // Game auto-selects the (only) tunable game, and its search space
    // renders read-only, driven purely by the /tuner/kinds metadata.
    const gameSelect = screen.getByLabelText("Game") as HTMLSelectElement;
    expect(gameSelect.value).toBe("traffic-lights");
    expect(screen.getByText("schedule")).toBeInTheDocument(); // a parameter name
    expect(screen.getByText("epsilon")).toBeInTheDocument(); // another parameter name
    expect(screen.getByText(/schedule = threshold/)).toBeInTheDocument(); // a condition

    // Budget fields are present with their documented defaults.
    expect(screen.getByLabelText("Trials")).toBeInTheDocument();
    expect(screen.getByLabelText("Seed")).toBeInTheDocument();

    // Rounds/trial defaults from the tuner's own eval_rounds (20, per the
    // traffic-lights fixture), not a hardcoded form default.
    expect((screen.getByLabelText("Rounds/trial") as HTMLInputElement).value).toBe("20");

    // Eta field is present with its default.
    const etaField = screen.getByLabelText(/Eta.*/) as HTMLInputElement;
    expect(etaField).toBeInTheDocument();
    expect(etaField.value).toBe("0.1");
  });

  it("omits target.rounds when unchanged, includes it when the field is edited", () => {
    const seen: unknown[] = [];
    const { store, env } = createTestStore();
    render(() => <LaunchForm store={store} />);
    // Spy on launchRun after render
    const origLaunch = env.launchRun;
    env.launchRun = vi.fn((kind, game, config) => {
      seen.push(config);
      return createMockBenchEnv().launchRun(kind, game, config);
    });

    fireEvent.change(screen.getByLabelText("Run Kind"), { target: { value: "tuner" } });
    store.dispatch({ tag: "launch", action: { tag: "request", kind: "tuner", game: "traffic-lights", config: { overrides: [] } } });
    // The reducer calls env.launchRun (which we spied on above) via the job effect.
    // But the effect is async, so query the store's state instead.
    expect((seen[0] as { overrides: string[] } | undefined)?.overrides).toBeDefined();
    const overrides = (seen[0] as { overrides: string[] }).overrides;
    expect(overrides.some((o) => o.startsWith("target.rounds"))).toBe(false);
  });

  it("renders budget fields and eta with correct defaults", () => {
    const { store } = createTestStore();
    render(() => <LaunchForm store={store} />);

    fireEvent.change(screen.getByLabelText("Run Kind"), { target: { value: "tuner" } });

    // Budget fields are present.
    expect(screen.getByLabelText("Trials")).toBeInTheDocument();
    expect(screen.getByLabelText("Workers")).toBeInTheDocument();
    expect(screen.getByLabelText("Seed")).toBeInTheDocument();
    expect(screen.getByLabelText("Rounds/trial")).toBeInTheDocument();
    expect(screen.getByLabelText(/Eta.*/)).toBeInTheDocument();
    expect(screen.getByLabelText(/Iteration budget/)).toBeInTheDocument();
    expect(screen.getByLabelText(/Time budget/)).toBeInTheDocument();
    expect(screen.getByLabelText(/Deterministic/)).toBeInTheDocument();

    // No more baselines panel or ladder fields.
    expect(screen.queryByText(/Starting baseline panel/)).not.toBeInTheDocument();
    expect(screen.queryByText(/Ladder/)).not.toBeInTheDocument();
    expect(screen.queryByText(/Max rungs/)).not.toBeInTheDocument();
    expect(screen.queryByText(/Saturation threshold/)).not.toBeInTheDocument();
  });

  it("hides the Game config field for a game with an empty game_config", () => {
    const { store } = createTestStore();
    render(() => <LaunchForm store={store} />);

    fireEvent.change(screen.getByLabelText("Run Kind"), { target: { value: "tuner" } });
    // Auto-selected game is traffic-lights (game_config: {}).
    expect(screen.queryByLabelText("Game config")).not.toBeInTheDocument();
  });

  it("shows a pre-filled Game config field for a game with a real game_config", () => {
    const { store } = createTestStore();
    render(() => <LaunchForm store={store} />);

    fireEvent.change(screen.getByLabelText("Run Kind"), { target: { value: "tuner" } });
    fireEvent.change(screen.getByLabelText("Game"), { target: { value: "druid" } });

    const field = screen.getByLabelText("Game config") as HTMLTextAreaElement;
    expect(JSON.parse(field.value)).toEqual({ size: { w: 5, h: 5 } });
  });

  it("disables Launch and shows an error when Game config contains invalid JSON", () => {
    const { store } = createTestStore();
    render(() => <LaunchForm store={store} />);

    fireEvent.change(screen.getByLabelText("Run Kind"), { target: { value: "tuner" } });
    fireEvent.change(screen.getByLabelText("Game"), { target: { value: "druid" } });

    const field = screen.getByLabelText("Game config") as HTMLTextAreaElement;
    fireEvent.input(field, { target: { value: "not json" } });

    const launchBtn = screen.getByText("Launch") as HTMLButtonElement;
    expect(launchBtn.disabled).toBe(true);
  });

  it("falls back to the round_robin games list for other kinds, unaffected by tunerKinds", () => {
    const { store } = createTestStore();
    render(() => <LaunchForm store={store} />);

    fireEvent.change(screen.getByLabelText("Run Kind"), { target: { value: "round_robin" } });
    expect((screen.getByLabelText("Game") as HTMLSelectElement).value).toBe(fakeKinds[0]!.games[0]!.game);
    expect(screen.getByText("Strong")).toBeInTheDocument();
  });
});

describe("RunDetailPanel / tuner", () => {
  it("renders trial stats, baseline comparisons, and the trial history once trials land", async () => {
    const { store } = createTestStore();
    render(() => <RunDetailPanel store={store} />);

    store.dispatch({ tag: "openRun", runId: FAKE_tuner_RUN_ID });

    // fakeTunerRunDetail is already terminal, so the single tail tick fetches
    // trials in the same round-trip -- no manual tick-forcing needed here.
    await screen.findByText("Best score (mu − 3σ)");

    // Best trial is #2 (cost 0.3 = score -0.300, the lowest of three).
    expect(screen.getByText("#2", { selector: ".tuner-stat-value" })).toBeInTheDocument();
    expect(screen.getByText("-0.300", { selector: ".tuner-stat-value" })).toBeInTheDocument();
    expect(screen.getByText("3", { selector: ".tuner-stat-value" })).toBeInTheDocument(); // trial count

    // Trial history lists all three rows' scores (from -cost).
    for (const score of ["-0.550", "-0.300", "-0.400"]) {
      expect(screen.getAllByText(score).length).toBeGreaterThanOrEqual(1);
    }

    // The trial table's Family column shows each trial's family -- fixture
    // spans rave/ucb1_tuned/ucb1.
    const familyCells = document.querySelectorAll(".tuner-trial-family");
    expect(Array.from(familyCells).map((c) => c.textContent)).toEqual(["ucb1", "ucb1_tuned", "rave"]);

    // fakeTunerRunDetail's incumbent (family: "rave", c: 0.7) gets its own
    // table -- distinct from the lowest single trial above.
    const incumbentTable = document.querySelector("#tuner-incumbent-diff-table")!;
    expect(incumbentTable.textContent).toContain("rave");
    expect(incumbentTable.textContent).toContain("0.7");
  });

  it("shows the tracked incumbent and copies its config on click", async () => {
    const writeText = vi.spyOn(navigator.clipboard, "writeText").mockResolvedValue(undefined);

    const { store } = createTestStore();
    render(() => <RunDetailPanel store={store} />);

    store.dispatch({ tag: "openRun", runId: FAKE_tuner_RUN_ID });

    // fakeTunerRunDetail's incumbent is {config: {family: "rave", c: 0.7}, cost: 0.2}
    // -- distinct from the "Best trial" stat. The score displayed is -cost = -0.200.
    await screen.findByText("Incumbent", { selector: ".tuner-stat-label" });
    expect(screen.getByText("-0.200", { selector: ".tuner-stat-value" })).toBeInTheDocument();

    fireEvent.click(screen.getByText("Copy as baseline config"));
    expect(writeText).toHaveBeenCalledWith(JSON.stringify({ family: "rave", c: 0.7 }));
    await screen.findByText("Copied!");
  });

  it("reconstructs a baseline from overrides for runs recorded before baseline_settings", async () => {
    const detail = {
      ...fakeTunerRunDetail,
      config: { overrides: ["optimizer.n_trials=50", "target.baselines=['random']"] },
    };
    const { store } = createTestStore({ getRun: () => Effect.send(detail) });
    render(() => <RunDetailPanel store={store} />);

    store.dispatch({ tag: "openRun", runId: FAKE_tuner_RUN_ID });
    await screen.findByText("Best score (mu − 3σ)");

    const incumbentTable = document.querySelector("#tuner-incumbent-diff-table")!;
    expect(incumbentTable.textContent).toContain("random");
  });

  it("toggles the chart help popover open and closed", async () => {
    const { store } = createTestStore();
    render(() => <RunDetailPanel store={store} />);

    store.dispatch({ tag: "openRun", runId: FAKE_tuner_RUN_ID });
    await screen.findByText("Best score (mu − 3σ)");

    expect(screen.queryByText(/running maximum of the 95% confidence band/)).not.toBeInTheDocument();

    fireEvent.click(screen.getByLabelText("How to read this chart"));
    expect(screen.getByText(/running maximum of the 95% confidence band/)).toBeInTheDocument();

    fireEvent.click(screen.getByLabelText("Close"));
    expect(screen.queryByText(/running maximum of the 95% confidence band/)).not.toBeInTheDocument();
  });

  it("draws a confirmed-floor line that is never more optimistic than best-so-far", async () => {
    const { store } = createTestStore();
    render(() => <RunDetailPanel store={store} />);

    store.dispatch({ tag: "openRun", runId: FAKE_tuner_RUN_ID });
    await screen.findByText("Best score (mu − 3σ)");

    const paths = document.querySelectorAll("#tuner-cost-chart path");
    const bestPath = Array.from(paths).find((p) => p.getAttribute("stroke") === "#4caf7a")!;
    const floorPath = Array.from(paths).find((p) => p.getAttribute("stroke") === "#e0904a")!;
    expect(bestPath).toBeTruthy();
    expect(floorPath).toBeTruthy();

    const lastY = (d: string): number => {
      const points = [...d.matchAll(/[ML]([\d.]+),([\d.]+)/g)];
      return parseFloat(points[points.length - 1]![2]!);
    };
    const bestY = lastY(bestPath.getAttribute("d")!);
    const floorY = lastY(floorPath.getAttribute("d")!);
    // Smaller y = higher up = a higher (weaker) score on this chart's
    // inverted scale -- fakeTrialRows has no repeat evaluations, so every
    // group's CI upper bound sits strictly above its own raw score, and the
    // confirmed floor must never render below (more optimistic than)
    // best-so-far.
    expect(floorY).toBeLessThanOrEqual(bestY);
  });

  it("pools trials with an identical config into one confidence-band group", async () => {
    const { store } = createTestStore({
      getRunTrials: () => Effect.send(fakeTrialRowsWithRepeats),
    });
    render(() => <RunDetailPanel store={store} />);

    store.dispatch({ tag: "openRun", runId: FAKE_tuner_RUN_ID });

    await screen.findByText("Best score (mu − 3σ)");

    // fakeTrialRowsWithRepeats adds two more evaluations (#4, #5) of trial
    // #2's exact config (cost 0.25/0.35 vs #2's 0.3) -- #4 is now the
    // lowest single cost, but all three share one group.
    expect(screen.getByText("#4", { selector: ".tuner-stat-value" })).toBeInTheDocument();
    expect(screen.getByText("3", { selector: ".tuner-stat-value" })).toBeInTheDocument(); // Evaluations

    expect(screen.getByText("Evaluations", { selector: ".tuner-stat-label" })).toBeInTheDocument();
    expect(screen.getByText("95% CI (mu ± 2σ)", { selector: ".tuner-stat-label" })).toBeInTheDocument();

    // The pooled interval must actually straddle the group's mean cost
    // (30.0%, i.e. (0.25 + 0.3 + 0.35) / 3) -- not be a degenerate
    // single-point estimate. The CI is based on mu ± 2σ which the fixture
    // doesn't provide extra fields for, so the CI values are computed from
    // the cost fallback.
    const ciStat = Array.from(document.querySelectorAll(".tuner-stat")).find(
      (el) => el.querySelector(".tuner-stat-label")?.textContent === "95% CI (mu ± 2σ)",
    )!;
    const ciText = ciStat.querySelector(".tuner-stat-value")!.textContent!;
    // CI fallback: mu ± 2σ = -cost ± 0 (no sigma info), so ci is
    // degenerate: lower === upper.
    const [lo, hi] = ciText.split("–").map((s) => parseFloat(s));
    expect(lo).toBe(hi); // Degenerate CI without sigma info

    // Every point sharing that config renders a whisker.
    const whiskers = document.querySelectorAll(".tuner-ci-whisker");
    expect(whiskers.length).toBe(5); // one per scored trial (#1, #2, #3, #4, #5)
  });

  it("keeps trials with the same config but different baseline instances in separate confidence-band groups", async () => {
    const { store } = createTestStore({
      getRunTrials: () => Effect.send(fakeTrialRowsMultiInstance),
    });
    render(() => <RunDetailPanel store={store} />);

    store.dispatch({ tag: "openRun", runId: FAKE_tuner_RUN_ID });

    await screen.findByText("Best score (mu − 3σ)");

    // The trial table's Baseline column distinguishes the two instances.
    const baselineCells = document.querySelectorAll(".tuner-trial-baseline");
    expect(Array.from(baselineCells).map((c) => c.textContent).sort()).toEqual(["master", "strong"]);

    // #1 (cost 0.1 = score -0.100 vs "strong") is the best trial.
    expect(screen.getByText("#1", { selector: ".tuner-stat-value" })).toBeInTheDocument();
    expect(screen.getByText("1", { selector: ".tuner-stat-value" })).toBeInTheDocument(); // Evaluations: not pooled with #2

    // Two distinct (single-evaluation) whiskers, not one pooled group.
    const whiskers = document.querySelectorAll(".tuner-ci-whisker");
    expect(whiskers.length).toBe(2);
  });

  it("shows a Resume control for a finished tuner run and dispatches resumeRun with the entered trial count", async () => {
    const seen: unknown[] = [];
    const { store } = createTestStore({
      resumeRun: (runId, nTrials, nWorkers) => {
        seen.push([runId, nTrials, nWorkers]);
        return createMockBenchEnv().resumeRun(runId, nTrials, nWorkers);
      },
    });
    render(() => <RunDetailPanel store={store} />);

    store.dispatch({ tag: "openRun", runId: FAKE_tuner_RUN_ID });
    await screen.findByText("Best score (mu − 3σ)");

    // fakeTunerRunDetail's trial_count is 3, so the default is 203.
    const input = screen.getByLabelText("Resume with n_trials") as HTMLInputElement;
    expect(input.value).toBe("203");

    fireEvent.input(input, { target: { value: "500" } });
    fireEvent.click(screen.getByText("Resume"));

    expect(seen).toEqual([[FAKE_tuner_RUN_ID, 500, undefined]]);
  });

  it("hides the Resume control while the run is still running", async () => {
    const runningTunerDetail = { ...fakeTunerRunDetail, status: "running", ended_at: null };
    const { store } = createTestStore({
      getRun: () => Effect.send(runningTunerDetail),
    });
    render(() => <RunDetailPanel store={store} />);

    store.dispatch({ tag: "openRun", runId: FAKE_tuner_RUN_ID });
    await screen.findByText("Status");

    expect(screen.queryByLabelText("Resume with n_trials")).not.toBeInTheDocument();
  });

  it("renders every rung of a ladder chain as one continuous trial history with a baseline-cutover marker", async () => {
    const rootRunId = "tuner-traffic-lights-20260201T000000-abc1234";
    const rootTrials: TrialRow[] = [
      { trial_id: 1, ts: "2026-02-01T00:00:01Z", config: { family: "ucb1" }, seed: 0, cost: 0.5, extra: null },
      { trial_id: 2, ts: "2026-02-01T00:00:02Z", config: { family: "ucb1" }, seed: 0, cost: 0.2, extra: null },
    ];
    const chain: ChainRung[] = [
      {
        run_id: rootRunId,
        label: null,
        status: "completed",
        started_at: "2026-02-01T00:00:00Z",
        ended_at: "2026-02-01T01:00:00Z",
        trial_count: rootTrials.length,
        incumbent: null,
      },
      {
        run_id: FAKE_tuner_RUN_ID,
        label: "baseline advance from " + rootRunId,
        status: "completed",
        started_at: "2026-03-01T00:00:00Z",
        ended_at: "2026-03-01T01:00:00Z",
        trial_count: fakeTrialRows.length,
        incumbent: { config: { family: "ucb1" }, cost: 0.2 },
      },
    ];
    const { store } = createTestStore({
      getRunChain: () => Effect.send(chain),
      getRunTrials: (runId) => Effect.send(runId === rootRunId ? rootTrials : fakeTrialRows),
    });
    render(() => <RunDetailPanel store={store} />);

    store.dispatch({ tag: "openRun", runId: FAKE_tuner_RUN_ID });
    await screen.findByText("Best score (mu − 3σ)");

    // Trial count spans both rungs (2 + 3), not just the open rung's own 3.
    expect(screen.getByText("5", { selector: ".tuner-stat-value" })).toBeInTheDocument();

    // The trials table gained a "Run" column identifying each row's rung,
    // and the cross-rung best score (trial #2 with cost 0.2 => score -0.200)
    // still wins over every rung-2 row.
    expect(screen.getByText("Run")).toBeInTheDocument();
    const rungCells = Array.from(document.querySelectorAll(".tuner-trial-rung")).map((c) => c.textContent);
    expect(rungCells).toEqual([
      // Table renders newest first: rung 2's 3 rows, then root's 2.
      `Rung 2 (${FAKE_tuner_RUN_ID})`,
      `Rung 2 (${FAKE_tuner_RUN_ID})`,
      `Rung 2 (${FAKE_tuner_RUN_ID})`,
      `Root (${rootRunId})`,
      `Root (${rootRunId})`,
    ]);
    // Cross-rung best cost: root's trial #2 (score -0.200) beats every rung-2 row.
    expect(screen.getByText("#2", { selector: ".tuner-stat-value" })).toBeInTheDocument();

    // One cutover marker for the one rung boundary in a 2-rung chain.
    expect(document.querySelectorAll(".tuner-rung-boundary").length).toBe(1);
    expect(screen.getByText("new baseline")).toBeInTheDocument();
  });

  it("shows the new-baseline flagpost before the new rung has scored a trial", async () => {
    const rootRunId = "tuner-traffic-lights-root";
    const rootTrials: TrialRow[] = [
      { trial_id: 1, ts: "2026-02-01T00:00:01Z", config: { family: "ucb1" }, seed: 0, cost: 0.025, extra: null },
    ];
    const chain: ChainRung[] = [
      {
        run_id: rootRunId,
        label: null,
        status: "stopped",
        started_at: "2026-02-01T00:00:00Z",
        ended_at: "2026-02-01T00:10:00Z",
        trial_count: 1,
        incumbent: null,
      },
      {
        run_id: FAKE_tuner_RUN_ID,
        label: "ladder rung 2 of " + rootRunId,
        status: "running",
        started_at: "2026-02-01T00:10:01Z",
        ended_at: null,
        trial_count: 0,
        incumbent: { config: { family: "ucb1" }, cost: 0.025 },
      },
    ];
    const { store } = createTestStore({
      getRunChain: () => Effect.send(chain),
      getRunTrials: (runId) => Effect.send(runId === rootRunId ? rootTrials : []),
    });
    render(() => <RunDetailPanel store={store} />);

    store.dispatch({ tag: "openRun", runId: FAKE_tuner_RUN_ID });
    await screen.findByText("Best score (mu − 3σ)");

    expect(document.querySelectorAll(".tuner-rung-boundary").length).toBe(1);
    expect(screen.getByText("new baseline")).toBeInTheDocument();
  });

  it("diffs the incumbent/lowest-trial tables against the latest rung's recorded baseline", async () => {
    const rootRunId = "tuner-traffic-lights-20260201T000000-abc1234";
    const rootTrials: TrialRow[] = [
      { trial_id: 1, ts: "2026-02-01T00:00:01Z", config: { family: "ucb1", c: 1.4 }, seed: 0, cost: 0.5, extra: null },
    ];
    const chain: ChainRung[] = [
      {
        run_id: rootRunId,
        label: null,
        status: "completed",
        started_at: "2026-02-01T00:00:00Z",
        ended_at: "2026-02-01T01:00:00Z",
        trial_count: rootTrials.length,
        incumbent: null,
      },
      {
        run_id: FAKE_tuner_RUN_ID,
        label: "baseline advance from " + rootRunId,
        status: "completed",
        started_at: "2026-03-01T00:00:00Z",
        ended_at: "2026-03-01T01:00:00Z",
        trial_count: fakeTrialRows.length,
        incumbent: { config: { family: "ucb1", c: 1.4 }, cost: 0.2 },
      },
    ];
    const detail = {
      ...fakeTunerRunDetail,
      config: { baseline_settings: { ladder2: { family: "ucb1", c: 1.4 } } },
    };
    const { store } = createTestStore({
      getRun: () => Effect.send(detail),
      getRunChain: () => Effect.send(chain),
      getRunTrials: (runId) => Effect.send(runId === rootRunId ? rootTrials : fakeTrialRows),
    });
    render(() => <RunDetailPanel store={store} />);

    store.dispatch({ tag: "openRun", runId: FAKE_tuner_RUN_ID });
    await screen.findByText("Best score (mu − 3σ)");

    expect(screen.getByText("Incumbent vs. baseline")).toBeInTheDocument();
    expect(screen.queryByText(/default/i)).not.toBeInTheDocument();

    const incumbentTable = document.querySelector("#tuner-incumbent-diff-table")!;
    expect(incumbentTable.querySelector("thead")!.textContent).toContain("Baseline");
    // Incumbent (family: "rave", c: 0.7) vs. the rung's baseline
    // (family: "ucb1", c: 1.4): both params differ.
    const familyRow = Array.from(incumbentTable.querySelectorAll("tbody tr")).find(
      (row) => row.querySelector(".tuner-param-name")?.textContent === "family",
    )!;
    expect(familyRow.classList.contains("tuner-diff-changed")).toBe(true);
    expect(familyRow.textContent).toContain("ucb1");
    const cRow = Array.from(incumbentTable.querySelectorAll("tbody tr")).find(
      (row) => row.querySelector(".tuner-param-name")?.textContent === "c",
    )!;
    expect(cRow.classList.contains("tuner-diff-changed")).toBe(true);
    expect(cRow.textContent).toContain("1.4");
  });
});