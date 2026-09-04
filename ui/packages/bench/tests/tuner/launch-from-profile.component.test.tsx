// launch-from-profile.component.test.tsx — "Start from profile" seeds every
// LaunchForm field from a saved profile, and "Save these settings as a
// profile…" writes the current form state back. Real `createStore`, mocked
// `TunerEnv`, no live server (AGENTS.md).

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
import { mockTunerEnv, runView } from "./mock-tuner-env.js";
import type { TunerEnv } from "../../src/tuner/tuner-env.js";
import type { TunableGame, JsonValue } from "../../src/types.js";
import type { TunerLaunchRequest, TunerProfileFile } from "../../src/tuner/tuner-types.js";

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

const PROFILE: TunerProfileFile = {
  key: "nim-sweep",
  profile_id: "nim-sweep",
  game_kind: "nim",
  objective_key: "nim-obj",
  constraint_count: 1,
  updated_at: null,
  is_seed: false,
};

const PROFILE_CONTENT: JsonValue = {
  profile_id: "nim-sweep",
  game_kind: "nim",
  objective_key: "nim-obj",
  constraints: [{ set: { c: { range: [1.2, 1.8] } } }],
  efforts: { tuning: { kind: "iterations", value: 500 } },
  budgets: { tuning_pair_budget: 40, validation_pair_budget: 24, production_validation_pairs: 8 },
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
      { key: "nim-obj", objective_id: "nim-obj", game_kind: null, opponent_count: 1, updated_at: null, is_seed: false },
    ],
  });
  store.dispatch({ tag: "profilesLoaded", profiles: [PROFILE] });
  const navigate = vi.fn();
  render(() => <LaunchForm store={store} navigate={navigate} />);
  return { store, navigate };
}

describe("LaunchForm — launch profiles", () => {
  it("seeds fields and the launch payload from a picked profile", async () => {
    const getProfile = vi.fn(() =>
      Effect.send({ key: "nim-sweep", content: PROFILE_CONTENT, updated_at: null, is_seed: false }),
    );
    const launchRun = vi.fn((_b: TunerLaunchRequest) => Effect.send(runView()));
    setup({ getProfile, launchRun });

    fireEvent.input(screen.getByTestId("from-profile"), { target: { value: "nim-sweep" } });

    await vi.waitFor(() =>
      expect((screen.getByTestId("effort-tuning-value") as HTMLInputElement).value).toBe("500"),
    );
    expect((screen.getByLabelText("Tuning pair budget") as HTMLInputElement).value).toBe("40");
    expect((screen.getByTestId("constraint-mode-c") as HTMLSelectElement).value).toBe("range");

    await vi.waitFor(() =>
      expect(screen.getByRole("button", { name: /^Launch$/ })).toBeEnabled(),
    );
    fireEvent.submit(document.getElementById("tuner-launch-form")!);
    await vi.waitFor(() => expect(launchRun).toHaveBeenCalled());
    const body = launchRun.mock.calls[0]![0] as TunerLaunchRequest;
    expect(body.constraints).toEqual([{ set: { c: { range: [1.2, 1.8] } } }]);
    expect(body.tuning_pair_budget).toBe(40);
    expect(body.tuning_max_iterations).toBe(500);
  });

  it("writes the current form state back as a profile", async () => {
    const putProfile = vi.fn((key: string, content: JsonValue) =>
      Effect.send({ key, content, updated_at: null, is_seed: false }),
    );
    setup({ putProfile });

    fireEvent.click(screen.getByTestId("save-as-profile-toggle"));
    fireEvent.input(screen.getByTestId("save-profile-key"), { target: { value: "my-profile" } });
    fireEvent.click(screen.getByTestId("save-profile-submit"));

    await vi.waitFor(() => expect(putProfile).toHaveBeenCalled());
    const [key, content] = putProfile.mock.calls[0]! as [string, Record<string, unknown>];
    expect(key).toBe("my-profile");
    expect(content.game_kind).toBe("nim");
    expect(content.objective_key).toBe("nim-obj");
    await vi.waitFor(() =>
      expect(screen.getByTestId("save-profile-done")).toHaveTextContent("my-profile"),
    );
  });
});
