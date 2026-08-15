// tests/BenchApp.smac3.test.tsx — Component-level tests for the SMAC3
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
  FAKE_SMAC3_RUN_ID,
  fakeKinds,
  fakeSmac3RunDetail,
  fakeTrialRows,
  fakeTrialRowsWithRepeats,
  fakeTrialRowsMultiInstance,
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
    expect(screen.getByText("strong", { selector: ".meta-value" })).toBeInTheDocument(); // named preset
    expect(screen.getByText("schedule")).toBeInTheDocument(); // a parameter name
    expect(screen.getByText("epsilon")).toBeInTheDocument(); // another parameter name
    expect(screen.getByText(/schedule = threshold/)).toBeInTheDocument(); // a condition

    // Budget fields are present with their documented defaults.
    expect(screen.getByLabelText("Trials")).toBeInTheDocument();
    expect(screen.getByLabelText("Seed")).toBeInTheDocument();

    // Rounds/trial defaults from the tuner's own eval_rounds (20, per the
    // traffic-lights fixture), not a hardcoded form default.
    expect((screen.getByLabelText("Rounds/trial") as HTMLInputElement).value).toBe("20");
  });

  it("omits target.rounds when unchanged, includes it when the field is edited", () => {
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
    expect(overrides.some((o) => o.startsWith("target.rounds"))).toBe(false);

    seen.length = 0;
    fireEvent.input(screen.getByLabelText("Rounds/trial"), { target: { value: "5" } });
    fireEvent.click(screen.getByText("Launch"));
    const overridesWithRounds = (seen[0] as { overrides: string[] }).overrides;
    expect(overridesWithRounds).toContain("target.rounds=5");
  });

  it("omits target.max_iterations when blank, includes it when the field is set", () => {
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
    expect(overrides.some((o) => o.startsWith("target.max_iterations"))).toBe(false);

    seen.length = 0;
    fireEvent.input(screen.getByLabelText(/Iteration budget/), { target: { value: "1000" } });
    fireEvent.click(screen.getByText("Launch"));
    const overridesWithBudget = (seen[0] as { overrides: string[] }).overrides;
    expect(overridesWithBudget).toContain("target.max_iterations=1000");
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
        config: {
          // Includes `target.baselines=['flat_mc']` -- the default
          // starting baseline (the floor rung) differs from traffic-lights'
          // own default named preset ("strong"), so the override is needed.
          overrides: [
            "optimizer.n_trials=25",
            "optimizer.deterministic=True",
            "optimizer.seed=42",
            "target.baselines=['flat_mc']",
          ],
        },
      },
    ]);
  });

  it("sends target.baselines when the starting baseline is a named preset", () => {
    const seen: unknown[] = [];
    const { store } = createTestStore({
      launchRun: (_kind, _game, config) => {
        seen.push(config);
        return createMockBenchEnv().launchRun(_kind, _game, config);
      },
    });
    render(() => <LaunchForm store={store} />);

    fireEvent.change(screen.getByLabelText("Run Kind"), { target: { value: "smac3" } });
    fireEvent.change(screen.getByLabelText("Starting baseline"), { target: { value: "strong" } });
    fireEvent.click(screen.getByText("Launch"));

    const overrides = (seen[0] as { overrides: string[] }).overrides;
    expect(overrides).toContain("target.baselines=['strong']");
  });

  it("sets config.ladder only when the ladder checkbox is enabled", () => {
    const seen: { config?: { ladder?: unknown } }[] = [];
    const { store } = createTestStore({
      launchRun: (kind, game, config) => {
        seen.push({ config: config as { ladder?: unknown } });
        return createMockBenchEnv().launchRun(kind, game, config);
      },
    });
    render(() => <LaunchForm store={store} />);

    fireEvent.change(screen.getByLabelText("Run Kind"), { target: { value: "smac3" } });
    fireEvent.click(screen.getByText("Launch"));
    expect(seen[0]!.config!.ladder).toBeUndefined();

    seen.length = 0;
    fireEvent.click(screen.getByLabelText(/Ladder/));
    fireEvent.input(screen.getByLabelText("Max rungs"), { target: { value: "8" } });
    fireEvent.input(screen.getByLabelText(/Saturation threshold/), { target: { value: "0.1" } });
    fireEvent.click(screen.getByText("Launch"));

    expect(seen[0]!.config!.ladder).toEqual({ max_rungs: 8, saturation_threshold: 0.1 });
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

  it("hides the Game config field for a game with an empty game_config", () => {
    const { store } = createTestStore();
    render(() => <LaunchForm store={store} />);

    fireEvent.change(screen.getByLabelText("Run Kind"), { target: { value: "smac3" } });
    // Auto-selected game is traffic-lights (game_config: {}).
    expect(screen.queryByLabelText("Game config")).not.toBeInTheDocument();
  });

  it("shows a pre-filled Game config field for a game with a real game_config, and includes it at launch", () => {
    const seen: { kind: string; game: string; config?: unknown }[] = [];
    const { store } = createTestStore({
      launchRun: (kind, game, config) => {
        seen.push({ kind, game, config });
        return createMockBenchEnv().launchRun(kind, game, config);
      },
    });
    render(() => <LaunchForm store={store} />);

    fireEvent.change(screen.getByLabelText("Run Kind"), { target: { value: "smac3" } });
    fireEvent.change(screen.getByLabelText("Game"), { target: { value: "druid" } });

    const field = screen.getByLabelText("Game config") as HTMLTextAreaElement;
    expect(JSON.parse(field.value)).toEqual({ size: { w: 5, h: 5 } });

    fireEvent.input(field, { target: { value: '{"size":{"w":9,"h":9}}' } });
    fireEvent.click(screen.getByText("Launch"));

    expect(seen).toHaveLength(1);
    expect((seen[0]!.config as { game_config: unknown }).game_config).toEqual({
      size: { w: 9, h: 9 },
    });
  });

  it("disables Launch and shows an error when Game config contains invalid JSON", () => {
    const { store } = createTestStore();
    render(() => <LaunchForm store={store} />);

    fireEvent.change(screen.getByLabelText("Run Kind"), { target: { value: "smac3" } });
    fireEvent.change(screen.getByLabelText("Game"), { target: { value: "druid" } });

    const field = screen.getByLabelText("Game config") as HTMLTextAreaElement;
    fireEvent.input(field, { target: { value: "not json" } });

    const launchBtn = screen.getByText("Launch") as HTMLButtonElement;
    expect(launchBtn.disabled).toBe(true);
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
  it("renders trial stats, baseline comparisons, and the trial history once trials land", async () => {
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

    // Best trial (#2) is `family: "ucb1_tuned"`; its table compares it
    // against the actual floor baseline selected when the run started.
    const diffTable = document.querySelector("#smac3-lowest-trial-diff-table")!;
    expect(diffTable.textContent).toContain("ucb1_tuned");
    const familyRow = Array.from(diffTable.querySelectorAll("tbody tr")).find(
      (row) => row.querySelector(".smac3-param-name")?.textContent === "family",
    )!;
    expect(familyRow.classList.contains("smac3-diff-changed")).toBe(true);
    expect(familyRow.textContent).toContain("flat_mc");

    // fakeSmac3RunDetail's incumbent (family: "rave", c: 0.7) gets its own
    // table -- the config "Use best as new baseline" would actually
    // promote, distinct from (and here, different family than) the lowest
    // single trial above.
    const incumbentTable = document.querySelector("#smac3-incumbent-diff-table")!;
    expect(incumbentTable.textContent).toContain("rave");
    expect(incumbentTable.textContent).toContain("0.7");
    expect(incumbentTable.textContent).toContain("flat_mc");
    expect(incumbentTable.querySelector("thead")!.textContent).toContain("Baseline");
    expect(incumbentTable.textContent).not.toMatch(/default/i);
  });

  it("shows the SMAC-tracked incumbent and copies its config on click", async () => {
    const writeText = vi.spyOn(navigator.clipboard, "writeText").mockResolvedValue(undefined);

    const { store } = createTestStore();
    render(() => <RunDetailPanel store={store} />);

    store.dispatch({ tag: "openRun", runId: FAKE_SMAC3_RUN_ID });

    // fakeSmac3RunDetail's incumbent is {config: {family: "rave", c: 0.7}, cost: 0.2}
    // -- distinct from the "Best trial" stat, which is derived from the
    // trial fixture rows, not this field. Scoped by selector since the
    // incumbent diff table's caption also says "Incumbent".
    await screen.findByText("Incumbent", { selector: ".smac3-stat-label" });
    expect(screen.getByText("20.0%", { selector: ".smac3-stat-value" })).toBeInTheDocument();

    fireEvent.click(screen.getByText("Copy as baseline config"));
    expect(writeText).toHaveBeenCalledWith(JSON.stringify({ family: "rave", c: 0.7 }));
    await screen.findByText("Copied!");
  });

  it("reconstructs a floor baseline for runs recorded before baseline settings", async () => {
    const detail = {
      ...fakeSmac3RunDetail,
      config: { overrides: ["optimizer.n_trials=50", "target.baselines=['random']"] },
    };
    const { store } = createTestStore({ getRun: () => Effect.send(detail) });
    render(() => <RunDetailPanel store={store} />);

    store.dispatch({ tag: "openRun", runId: FAKE_SMAC3_RUN_ID });
    await screen.findByText("Incumbent vs. baseline");

    const incumbentTable = document.querySelector("#smac3-incumbent-diff-table")!;
    expect(incumbentTable.textContent).toContain("random");
    expect(incumbentTable.textContent).not.toMatch(/default|strong|master|easy/i);
  });

  it("toggles the chart help popover open and closed", async () => {
    const { store } = createTestStore();
    render(() => <RunDetailPanel store={store} />);

    store.dispatch({ tag: "openRun", runId: FAKE_SMAC3_RUN_ID });
    await screen.findByText("Best cost (loss rate)");

    expect(screen.queryByText(/95%-confidence "no worse than this" bound/)).not.toBeInTheDocument();

    fireEvent.click(screen.getByLabelText("How to read this chart"));
    expect(screen.getByText(/95%-confidence "no worse than this" bound/)).toBeInTheDocument();

    fireEvent.click(screen.getByLabelText("Close"));
    expect(screen.queryByText(/95%-confidence "no worse than this" bound/)).not.toBeInTheDocument();
  });

  it("draws a confirmed-floor line that is never more optimistic than best-so-far", async () => {
    const { store } = createTestStore();
    render(() => <RunDetailPanel store={store} />);

    store.dispatch({ tag: "openRun", runId: FAKE_SMAC3_RUN_ID });
    await screen.findByText("Best cost (loss rate)");

    const paths = document.querySelectorAll("#smac3-cost-chart path");
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
    // Smaller y = higher up = a higher (worse) cost on this chart's
    // inverted scale -- fakeTrialRows has no repeat evaluations, so every
    // group's CI upper bound sits strictly above its own raw cost, and the
    // confirmed floor must never render below (more optimistic than)
    // best-so-far.
    expect(floorY).toBeLessThanOrEqual(bestY);
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

  it("keeps trials with the same config but different baseline instances in separate confidence-band groups", async () => {
    const { store } = createTestStore({
      getRunTrials: () => Effect.send(fakeTrialRowsMultiInstance),
    });
    render(() => <RunDetailPanel store={store} />);

    store.dispatch({ tag: "openRun", runId: FAKE_SMAC3_RUN_ID });

    await screen.findByText("Best cost (loss rate)");

    // The trial table's Baseline column distinguishes the two instances.
    const baselineCells = document.querySelectorAll(".smac3-trial-baseline");
    expect(Array.from(baselineCells).map((c) => c.textContent).sort()).toEqual(["master", "strong"]);

    // #1 (cost 0.1 vs "strong") is the best trial -- if the two same-config
    // trials had been pooled across instances, the group's mean/CI would be
    // (0.1 + 0.6) / 2 instead of a single-evaluation estimate per instance.
    expect(screen.getByText("#1", { selector: ".smac3-stat-value" })).toBeInTheDocument();
    expect(screen.getByText("1", { selector: ".smac3-stat-value" })).toBeInTheDocument(); // Evaluations: not pooled with #2

    // Two distinct (single-evaluation) whiskers, not one pooled group.
    const whiskers = document.querySelectorAll(".smac3-ci-whisker");
    expect(whiskers.length).toBe(2);
  });

  it("shows a Resume control for a finished smac3 run and dispatches resumeRun with the entered trial count", async () => {
    const seen: unknown[] = [];
    const { store } = createTestStore({
      resumeRun: (runId, nTrials, nWorkers) => {
        seen.push([runId, nTrials, nWorkers]);
        return createMockBenchEnv().resumeRun(runId, nTrials, nWorkers);
      },
    });
    render(() => <RunDetailPanel store={store} />);

    store.dispatch({ tag: "openRun", runId: FAKE_SMAC3_RUN_ID });
    await screen.findByText("Best cost (loss rate)");

    // fakeSmac3RunDetail's trial_count is 3, so the default is 203.
    const input = screen.getByLabelText("Resume with n_trials") as HTMLInputElement;
    expect(input.value).toBe("203");

    fireEvent.input(input, { target: { value: "500" } });
    fireEvent.click(screen.getByText("Resume"));

    expect(seen).toEqual([[FAKE_SMAC3_RUN_ID, 500, undefined]]);
  });

  it("hides the Resume control while the run is still running", async () => {
    const runningSmac3Detail = { ...fakeSmac3RunDetail, status: "running", ended_at: null };
    const { store } = createTestStore({
      getRun: () => Effect.send(runningSmac3Detail),
    });
    render(() => <RunDetailPanel store={store} />);

    store.dispatch({ tag: "openRun", runId: FAKE_SMAC3_RUN_ID });
    await screen.findByText("Status");

    expect(screen.queryByLabelText("Resume with n_trials")).not.toBeInTheDocument();
  });

  it("shows a Use best as new baseline control once an incumbent exists, even while the run is still running", async () => {
    const runningSmac3Detail = { ...fakeSmac3RunDetail, status: "running", ended_at: null };
    const seen: unknown[] = [];
    const { store } = createTestStore({
      getRun: () => Effect.send(runningSmac3Detail),
      advanceBaseline: (runId, nTrials, nWorkers) => {
        seen.push([runId, nTrials, nWorkers]);
        return createMockBenchEnv().advanceBaseline(runId, nTrials, nWorkers);
      },
    });
    render(() => <RunDetailPanel store={store} />);

    store.dispatch({ tag: "openRun", runId: FAKE_SMAC3_RUN_ID });
    await screen.findByText("Status");

    // Unlike Resume (hidden above for the same running detail), this
    // button doesn't require the run to have stopped first -- the route
    // handles that itself.
    const btn = screen.getByText("Use best as new baseline");
    fireEvent.click(btn);
    expect(seen).toEqual([[FAKE_SMAC3_RUN_ID, undefined, undefined]]);
  });

  it("hides the Use best as new baseline control before any incumbent has been reported", async () => {
    const { store } = createTestStore({
      getRun: () => Effect.send({ ...fakeSmac3RunDetail, incumbent: null }),
    });
    render(() => <RunDetailPanel store={store} />);

    store.dispatch({ tag: "openRun", runId: FAKE_SMAC3_RUN_ID });
    await screen.findByText("Status");

    expect(screen.queryByText("Use best as new baseline")).not.toBeInTheDocument();
  });

  it("renders every rung of a ladder chain as one continuous trial history with a baseline-cutover marker", async () => {
    const rootRunId = "smac3-traffic-lights-20260201T000000-abc1234";
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
        run_id: FAKE_SMAC3_RUN_ID,
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

    store.dispatch({ tag: "openRun", runId: FAKE_SMAC3_RUN_ID });
    await screen.findByText("Best cost (loss rate)");

    // Trial count spans both rungs (2 + 3), not just the open rung's own 3.
    expect(screen.getByText("5", { selector: ".smac3-stat-value" })).toBeInTheDocument();

    // The trials table gained a "Run" column identifying each row's rung,
    // and the cross-rung best cost (20.0%, root-1's trial #2) still wins
    // over every rung-2 row.
    expect(screen.getByText("Run")).toBeInTheDocument();
    const rungCells = Array.from(document.querySelectorAll(".smac3-trial-rung")).map((c) => c.textContent);
    expect(rungCells).toEqual([
      // Table renders newest first: rung 2's 3 rows, then root's 2.
      `Rung 2 (${FAKE_SMAC3_RUN_ID})`,
      `Rung 2 (${FAKE_SMAC3_RUN_ID})`,
      `Rung 2 (${FAKE_SMAC3_RUN_ID})`,
      `Root (${rootRunId})`,
      `Root (${rootRunId})`,
    ]);
    // Cross-rung best cost: root's trial #2 (20.0%) beats every rung-2 row.
    expect(screen.getByText("#2", { selector: ".smac3-stat-value" })).toBeInTheDocument();

    // One cutover marker for the one rung boundary in a 2-rung chain.
    expect(document.querySelectorAll(".smac3-rung-boundary").length).toBe(1);
    expect(screen.getByText("new baseline")).toBeInTheDocument();
  });

  it("diffs the incumbent/lowest-trial tables against the latest rung's recorded baseline", async () => {
    const rootRunId = "smac3-traffic-lights-20260201T000000-abc1234";
    const rootTrials: TrialRow[] = [
      { trial_id: 1, ts: "2026-02-01T00:00:01Z", config: { family: "ucb1", c: 1.4 }, seed: 0, cost: 0.5, extra: null },
    ];
    // Rung 2's baseline is rung 1's own incumbent, stored in the current
    // run's launch config as the resolved baseline settings.
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
        run_id: FAKE_SMAC3_RUN_ID,
        label: "baseline advance from " + rootRunId,
        status: "completed",
        started_at: "2026-03-01T00:00:00Z",
        ended_at: "2026-03-01T01:00:00Z",
        trial_count: fakeTrialRows.length,
        incumbent: { config: { family: "ucb1", c: 1.4 }, cost: 0.2 },
      },
    ];
    const detail = {
      ...fakeSmac3RunDetail,
      config: { baseline_settings: { ladder2: { family: "ucb1", c: 1.4 } } },
    };
    const { store } = createTestStore({
      getRun: () => Effect.send(detail),
      getRunChain: () => Effect.send(chain),
      getRunTrials: (runId) => Effect.send(runId === rootRunId ? rootTrials : fakeTrialRows),
    });
    render(() => <RunDetailPanel store={store} />);

    store.dispatch({ tag: "openRun", runId: FAKE_SMAC3_RUN_ID });
    await screen.findByText("Best cost (loss rate)");

    expect(screen.getByText("Incumbent vs. baseline")).toBeInTheDocument();
    expect(screen.queryByText(/default/i)).not.toBeInTheDocument();

    const incumbentTable = document.querySelector("#smac3-incumbent-diff-table")!;
    expect(incumbentTable.querySelector("thead")!.textContent).toContain("Baseline");
    // Incumbent (family: "rave", c: 0.7) vs. the rung's baseline
    // (family: "ucb1", c: 1.4): both params differ.
    const familyRow = Array.from(incumbentTable.querySelectorAll("tbody tr")).find(
      (row) => row.querySelector(".smac3-param-name")?.textContent === "family",
    )!;
    expect(familyRow.classList.contains("smac3-diff-changed")).toBe(true);
    expect(familyRow.textContent).toContain("ucb1");
    const cRow = Array.from(incumbentTable.querySelectorAll("tbody tr")).find(
      (row) => row.querySelector(".smac3-param-name")?.textContent === "c",
    )!;
    expect(cRow.classList.contains("smac3-diff-changed")).toBe(true);
    expect(cRow.textContent).toContain("1.4");
  });
});
