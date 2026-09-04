// launch-space-overrides.test.tsx — component test for the "Constrain
// parameters" panel added to LaunchForm's advanced section (Task 13g). Real
// `createStore`, mocked `TunerEnv`, no live server (AGENTS.md).

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
import type { TunableGame } from "../../src/types.js";
import type { TunerLaunchRequest } from "../../src/tuner/tuner-types.js";

afterEach(cleanup);

const KIND: TunableGame = {
  game: "nim",
  tuner: {
    id: "nim",
    baselines: ["strong"],
    eval_rounds: 1,
    parameters: [
      { name: "family", type: "categorical", choices: ["ucb1", "rave"] },
      { name: "c", type: "float", bounds: [0.5, 3.0], default: 1.4 },
    ],
    conditions: [],
    game_config: {},
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
        key: "nim-obj",
        objective_id: "nim-obj",
        game_kind: null,
        opponent_count: 1,
        updated_at: null,
        is_seed: false,
      },
    ],
  });
  render(() => <LaunchForm store={store} />);
  fireEvent.click(screen.getByText("Show advanced options"));
  return store;
}

describe("LaunchForm — constrain parameters", () => {
  it("assembles a valid space_overrides map from the JSON textarea", async () => {
    const launchRun = vi.fn((_body: TunerLaunchRequest) => Effect.send(runView()));
    setup({ launchRun });

    fireEvent.input(screen.getByTestId("space-overrides-input"), {
      target: { value: '{ "c": { "range": [1.2, 1.8] } }' },
    });
    fireEvent.click(screen.getByRole("button", { name: /^Launch$/ }));
    await vi.waitFor(() => expect(launchRun).toHaveBeenCalled());
    expect((launchRun.mock.calls[0]![0] as TunerLaunchRequest).space_overrides).toEqual({
      c: { range: [1.2, 1.8] },
    });
  });

  it("omits space_overrides when the textarea is blank", async () => {
    const launchRun = vi.fn((_body: TunerLaunchRequest) => Effect.send(runView()));
    setup({ launchRun });
    fireEvent.click(screen.getByRole("button", { name: /^Launch$/ }));
    await vi.waitFor(() => expect(launchRun).toHaveBeenCalled());
    expect(
      (launchRun.mock.calls[0]![0] as TunerLaunchRequest).space_overrides,
    ).toBeUndefined();
  });

  it("blocks launch on an obvious local error", () => {
    setup();
    fireEvent.input(screen.getByTestId("space-overrides-input"), {
      target: { value: '{ "c": { "range": [2, 1] } }' },
    });
    expect(screen.getByTestId("space-overrides-error")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /Launch/ })).toBeDisabled();
  });
});
