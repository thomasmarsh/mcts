// launch-budget-fields.component.test.tsx — the three launch-budget number
// inputs (tuning/validation/production pair budgets) carry `min`/`step`
// attributes derived from the resolved run plan's opponent panel weight and
// the finalists/cohort-size fields, and snap a typed value to the nearest
// valid one on blur — so the operator's arrow keys land on values the
// server's launch validation (`tuner_cli.run.validate_objective_options`)
// actually accepts instead of "Wrong!" round trips. Real
// `createStore(tunerReducer, env)`, mocked `TunerEnv` (AGENTS.md).

import { afterEach, describe, expect, it, vi } from "vitest";
import { cleanup, fireEvent, render, screen } from "@solidjs/testing-library";
import { Effect, createStore } from "@mcts/core";
import {
  initialTunerState,
  tunerReducer,
  type TunerAction,
  type TunerState,
} from "../../src/tuner/tuner-reducer.js";
import { LaunchForm } from "../../src/tuner/views/LaunchForm.js";
import { mockTunerEnv } from "./mock-tuner-env.js";
import type { TunerEnv } from "../../src/tuner/tuner-env.js";
import type { TunableGame } from "../../src/types.js";
import type { RunPlan, TunerLaunchRequest } from "../../src/tuner/tuner-types.js";

afterEach(cleanup);

const KIND: TunableGame = {
  game: "atari-go",
  tuner: {
    id: "atari-go",
    baselines: ["strong"],
    eval_rounds: 1,
    parameters: [{ name: "algorithm", type: "categorical", choices: ["mcts", "rave"] }],
    conditions: [],
    game_config: {},
  },
};

// An opponent panel with total weight 4 (2 + 2), so validation/production
// steps land somewhere other than the weight-1 fallback.
const PLAN: RunPlan = {
  ok: true,
  errors: [],
  game_kind: "atari-go",
  objective_id: "atari-go-reference-v1",
  opponents: [
    { id: "a", role: "default", weight: 2, source: "schema_default", config: "{}" },
    { id: "b", role: "historical_reference", weight: 2, source: "inline", config: "{}" },
  ],
  budgets: {
    cohort_size: 8,
    finalists: 3,
    bootstrap_candidates: 3,
    random_reserve_candidates: 2,
    tuning_pairs: 4,
    tuning_pair_budget: 32,
    validation_pair_budget: 24,
    diagnostic_pair_budget: 0,
    production_validation_pairs: 8,
    proposer_policy: "smac_mixed",
    derived: { initial_cohort_pairs: 32, validation_pairs_per_finalist: 8, production_pairs: 8 },
  },
};

function setup(over: Partial<TunerEnv> = {}) {
  const store = createStore<TunerState, TunerAction, TunerEnv>(
    initialTunerState(),
    tunerReducer,
    mockTunerEnv(over),
  );
  store.dispatch({ tag: "tunableGamesLoaded", tunableGames: [KIND] });
  store.dispatch({
    tag: "objectivesLoaded",
    objectives: [
      {
        key: "atari-go-obj",
        objective_id: "atari-go-reference-v1",
        game_kind: "atari-go",
        opponent_count: 2,
        updated_at: null,
        is_seed: false,
      },
    ],
  });
  render(() => <LaunchForm store={store} />);
  return store;
}

describe("LaunchForm — budget field steppers", () => {
  it("steps validation/production budgets by the resolved panel weight, and floors the tuning budget", async () => {
    const planRun = vi.fn((_b: TunerLaunchRequest) => Effect.send(PLAN));
    setup({ planRun });
    await vi.waitFor(() => expect(planRun).toHaveBeenCalled());

    const validation = (await screen.findByLabelText(
      "Validation pair budget",
    )) as HTMLInputElement;
    const production = screen.getByLabelText("Production validation pairs") as HTMLInputElement;
    const tuning = screen.getByLabelText("Tuning pair budget") as HTMLInputElement;

    // finalists default 3 x panel weight 4 = 12; production step = panel weight 4.
    await vi.waitFor(() => expect(validation.step).toBe("12"));
    expect(validation.min).toBe("12");
    expect(production.step).toBe("4");
    expect(production.min).toBe("4");
    // cohort default 8 x tuning_pairs 4 = 32.
    expect(tuning.min).toBe("32");
  });

  it("snaps a typed validation budget up to the nearest multiple of finalists × panel weight on blur", async () => {
    const planRun = vi.fn((_b: TunerLaunchRequest) => Effect.send(PLAN));
    setup({ planRun });
    await vi.waitFor(() => expect(planRun).toHaveBeenCalled());
    const validation = (await screen.findByLabelText(
      "Validation pair budget",
    )) as HTMLInputElement;
    await vi.waitFor(() => expect(validation.step).toBe("12"));

    fireEvent.input(validation, { target: { value: "50" } });
    fireEvent.blur(validation);
    // 50 / 12 = 4.17 -> rounds to 4 -> 48.
    expect(validation.value).toBe("48");
  });

  it("snaps a too-low tuning budget up to the cohort floor on blur", async () => {
    setup();
    const tuning = screen.getByLabelText("Tuning pair budget") as HTMLInputElement;

    fireEvent.input(tuning, { target: { value: "5" } });
    fireEvent.blur(tuning);
    expect(tuning.value).toBe("32");
  });
});
