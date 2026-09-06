// run-plan.component.test.tsx — the LaunchForm "Run plan" panel (Task 13h).
// A real `createStore(tunerReducer, env)` backs the render; `planRun` is
// mocked (AGENTS.md "mock the environment"), no live server.

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

const PLAN: RunPlan = {
  ok: true,
  errors: [],
  game_kind: "atari-go",
  objective_id: "atari-go-reference-v1",
  game_config: "{}",
  game_config_is_override: false,
  opponents: [
    {
      id: "schema-default",
      label: "Default",
      role: "default",
      weight: 1,
      source: "schema_default",
      config: '{"select":"rave"}',
    },
    {
      id: "historical",
      role: "historical_reference",
      weight: 1,
      source: "inline",
      config: '{"algorithm":"mcts"}',
    },
  ],
  space: {
    schema_id: "strategy",
    algorithms: ["mcts", "bandit"],
    residual_categoricals: { algorithm: ["mcts", "bandit"], select: ["ucb1", "rave"] },
    constraints: [{ set: { select: { choices: ["ucb1", "rave"] } } }],
    parameters: [
      {
        name: "select",
        kind: "categorical",
        bounds: null,
        choices: ["ucb1", "rave"],
        default: "ucb1",
        constant_value: null,
        active_when: "algorithm in ['mcts']",
      },
    ],
  },
  efforts: {
    tuning: { kind: "iterations", value: 1000 },
    validation: { kind: "iterations", value: 10000 },
    production: { kind: "iterations", value: 10000 },
  },
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
  epoch: { epoch_id: "epoch-x", fingerprint: "deadbeefcafef00d0000" },
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

describe("LaunchForm — Run plan panel", () => {
  it("resolves the plan on form settle and shows the expanded opponent config", async () => {
    const planRun = vi.fn((_b: TunerLaunchRequest) => Effect.send(PLAN));
    setup({ planRun });

    await vi.waitFor(() => expect(planRun).toHaveBeenCalled());
    const table = await screen.findByTestId("run-plan-opponents");
    expect(table).toHaveTextContent('{"select":"rave"}');
    const space = screen.getByTestId("run-plan-tuning-space");
    expect(space).toHaveTextContent("algorithm: mcts, bandit");
    expect(space).toHaveTextContent("variants: select ∈ [ucb1, rave]");
    expect(screen.getByTestId("run-plan-budgets")).toHaveTextContent("32 pairs");
  });

  it("stays a hint (no table) while the request is incomplete", () => {
    const planRun = vi.fn((_b: TunerLaunchRequest) => Effect.send(PLAN));
    setup({ planRun });
    // Clear the required run id so `buildRequest` returns null.
    fireEvent.input(screen.getByLabelText("Run id"), { target: { value: "" } });
    expect(screen.queryByTestId("run-plan-opponents")).not.toBeInTheDocument();
  });
});
