// launch-constraints.test.tsx — component test for the schema-driven
// constraint editor wired into LaunchForm (replacing the old categorical
// exclusion checklist and the free-text "Constrain parameters" textarea). Real
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

describe("LaunchForm — constraint editor", () => {
  it("emits a `constraints` array from an unticked categorical (exclude an algorithm)", async () => {
    const launchRun = vi.fn((_b: TunerLaunchRequest) => Effect.send(runView()));
    setup({ launchRun });

    fireEvent.input(screen.getByTestId("constraint-mode-algorithm"), {
      target: { value: "choices" },
    });
    fireEvent.click(screen.getByTestId("constraint-choice-algorithm-negamax"));

    fireEvent.click(screen.getByRole("button", { name: /^Launch$/ }));
    await vi.waitFor(() => expect(launchRun).toHaveBeenCalled());
    const body = launchRun.mock.calls[0]![0] as TunerLaunchRequest;
    expect(body.constraints).toEqual([{ set: { algorithm: { choices: ["mcts", "random"] } } }]);
  });

  it("omits `constraints` when every row is free", async () => {
    const launchRun = vi.fn((_b: TunerLaunchRequest) => Effect.send(runView()));
    setup({ launchRun });
    fireEvent.click(screen.getByRole("button", { name: /^Launch$/ }));
    await vi.waitFor(() => expect(launchRun).toHaveBeenCalled());
    expect((launchRun.mock.calls[0]![0] as TunerLaunchRequest).constraints).toBeUndefined();
  });

  it("feeds a narrowed range into the debounced plan/preflight round-trip", async () => {
    const planRun = vi.fn((_b: TunerLaunchRequest) => Effect.send({ ok: true, errors: [] }));
    const preflightRun = vi.fn((_b: TunerLaunchRequest) => Effect.send({ ok: true, errors: [] }));
    setup({ planRun, preflightRun });

    fireEvent.input(screen.getByTestId("constraint-mode-c"), { target: { value: "range" } });
    fireEvent.input(screen.getByTestId("constraint-low-c"), { target: { value: "1.2" } });
    fireEvent.input(screen.getByTestId("constraint-high-c"), { target: { value: "1.8" } });

    await vi.waitFor(() => {
      const last = planRun.mock.calls.at(-1)?.[0] as TunerLaunchRequest | undefined;
      expect(last?.constraints).toEqual([{ set: { c: { range: [1.2, 1.8] } } }]);
    });
    expect(preflightRun).toHaveBeenCalled();
  });

  it("blocks launch on a local constraint error (every box unticked)", () => {
    setup();
    fireEvent.input(screen.getByTestId("constraint-mode-algorithm"), {
      target: { value: "choices" },
    });
    for (const c of ["mcts", "random", "negamax"]) {
      fireEvent.click(screen.getByTestId(`constraint-choice-algorithm-${c}`));
    }
    expect(screen.getByTestId("constraint-error-algorithm")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /Launch/ })).toBeDisabled();
  });

  it("surfaces the server preflight error inline", async () => {
    const preflightRun = vi.fn(() =>
      Effect.send({ ok: false, errors: ["constraint on 'select' leaves no residual choice"] }),
    );
    setup({ preflightRun });
    fireEvent.input(screen.getByTestId("constraint-mode-select"), { target: { value: "choices" } });
    fireEvent.click(screen.getByTestId("constraint-choice-select-rave"));

    await vi.waitFor(() =>
      expect(screen.getByTestId("preflight-errors")).toHaveTextContent(
        "constraint on 'select' leaves no residual choice",
      ),
    );
  });

  it("hides the fieldset for a game with no tunable parameters", () => {
    const store = createStore<TunerState, TunerAction, TunerEnv>(
      initialTunerState(),
      tunerReducer,
      mockTunerEnv(),
    );
    store.dispatch({
      tag: "tunableGamesLoaded",
      tunableGames: [{ game: "ttt", tuner: { ...KIND.tuner, id: "ttt", parameters: [], conditions: [] } }],
    });
    store.dispatch({
      tag: "objectivesLoaded",
      objectives: [
        {
          key: "o",
          objective_id: "o",
          game_kind: null,
          opponent_count: 1,
          updated_at: null,
          is_seed: false,
        },
      ],
    });
    render(() => <LaunchForm store={store} />);
    fireEvent.click(screen.getByText("Show advanced options"));

    expect(screen.queryByTestId("constraint-editor-fieldset")).not.toBeInTheDocument();
    expect(screen.getByTestId("effort-rows")).toBeInTheDocument();
  });
});
