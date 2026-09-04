// profile-editor.component.test.tsx — component test for the launch-profile
// editor. Real `createStore(tunerReducer, env)`, mocked `TunerEnv`, no live
// server (AGENTS.md "mock the environment").

import { afterEach, describe, expect, it, vi } from "vitest";
import { cleanup, fireEvent, render, screen } from "@solidjs/testing-library";
import { Effect, createStore } from "@mcts/core";
import {
  initialTunerState,
  tunerReducer,
  type TunerAction,
  type TunerState,
} from "../../src/tuner/tuner-reducer.js";
import { ProfileEditor } from "../../src/tuner/views/ProfileEditor.js";
import { mockTunerEnv } from "./mock-tuner-env.js";
import type { TunerEnv } from "../../src/tuner/tuner-env.js";
import type { TunableGame } from "../../src/types.js";
import type { JsonValue } from "../../src/types.js";

afterEach(cleanup);

const KIND: TunableGame = {
  game: "nim",
  tuner: {
    id: "nim",
    baselines: ["strong"],
    eval_rounds: 1,
    parameters: [
      { name: "algorithm", type: "categorical", choices: ["mcts", "random", "negamax"] },
      { name: "select", type: "categorical", choices: ["ucb1", "rave"] },
      { name: "c", type: "float", bounds: [0.5, 3.0], default: 1.4 },
    ],
    conditions: [
      { if: { algorithm: "mcts" }, then: ["select"] },
      { if: { select: ["ucb1"] }, then: ["c"] },
    ],
    game_config: {},
  },
};

function setup(over: Partial<TunerEnv>, profileKey: string | null) {
  const store = createStore<TunerState, TunerAction, TunerEnv>(
    initialTunerState(),
    tunerReducer,
    mockTunerEnv(over),
  );
  store.dispatch({ tag: "tunableGamesLoaded", tunableGames: [KIND] });
  store.dispatch({
    tag: "objectivesLoaded",
    objectives: [
      { key: "nim-obj", objective_id: "nim-obj", game_kind: null, opponent_count: 1, updated_at: null, is_seed: false },
    ],
  });
  if (profileKey) store.dispatch({ tag: "openProfile", key: profileKey });
  const navigate = vi.fn();
  render(() => <ProfileEditor store={store} profileKey={profileKey} navigate={navigate} />);
  return { store, navigate };
}

describe("ProfileEditor", () => {
  it("saves a new profile with the form's constraints, efforts, and budgets", async () => {
    const putProfile = vi.fn((key: string, content: JsonValue) =>
      Effect.send({ key, content, updated_at: null, is_seed: false }),
    );
    const { navigate } = setup({ putProfile }, null);

    fireEvent.input(screen.getByTestId("profile-id-input"), { target: { value: "nim sweep" } });
    fireEvent.input(screen.getByTestId("profile-budget-tuningPairBudget"), {
      target: { value: "40" },
    });
    fireEvent.input(screen.getByTestId("profile-effort-tuning-value"), { target: { value: "500" } });
    fireEvent.input(screen.getByTestId("constraint-mode-algorithm"), {
      target: { value: "choices" },
    });
    fireEvent.click(screen.getByTestId("constraint-choice-algorithm-negamax"));

    fireEvent.click(screen.getByText("Save"));

    await vi.waitFor(() => expect(putProfile).toHaveBeenCalled());
    const [key, content] = putProfile.mock.calls[0]! as [string, Record<string, unknown>];
    expect(key).toBe("nim-sweep");
    expect(content.game_kind).toBe("nim");
    expect(content.objective_key).toBe("nim-obj");
    expect(content.constraints).toEqual([{ set: { algorithm: { choices: ["mcts", "random"] } } }]);
    expect(content.efforts).toEqual({ tuning: { kind: "iterations", value: 500 } });
    expect((content.budgets as Record<string, number>).tuning_pair_budget).toBe(40);

    await vi.waitFor(() => expect(navigate).toHaveBeenCalledWith({ view: "profiles" }));
  });

  it("seeds the constraint editor from a loaded profile", async () => {
    const getProfile = vi.fn(() =>
      Effect.send({
        key: "nim-sweep",
        content: {
          profile_id: "nim-sweep",
          game_kind: "nim",
          objective_key: "nim-obj",
          constraints: [{ set: { c: { range: [1.2, 1.8] } } }],
          budgets: { tuning_pair_budget: 40, validation_pair_budget: 24, production_validation_pairs: 8 },
        } as JsonValue,
        updated_at: null,
        is_seed: false,
      }),
    );
    setup({ getProfile }, "nim-sweep");

    await vi.waitFor(() =>
      expect((screen.getByTestId("constraint-mode-c") as HTMLSelectElement).value).toBe("range"),
    );
    expect((screen.getByTestId("constraint-low-c") as HTMLInputElement).value).toBe("1.2");
    expect((screen.getByTestId("profile-budget-tuningPairBudget") as HTMLInputElement).value).toBe(
      "40",
    );
  });

  it("blocks Save on a local constraint error", () => {
    setup({}, null);
    fireEvent.input(screen.getByTestId("constraint-mode-algorithm"), {
      target: { value: "choices" },
    });
    for (const c of ["mcts", "random", "negamax"]) {
      fireEvent.click(screen.getByTestId(`constraint-choice-algorithm-${c}`));
    }
    expect(screen.getByText("Save")).toBeDisabled();
  });
});
