// launch-exclusions.test.tsx — component test for the excluded-families
// checklist added to LaunchForm's advanced section (Task 13c). Real
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
import type { TunerGameInfo } from "../../src/types.js";
import type { TunerLaunchRequest } from "../../src/tuner/tuner-types.js";

afterEach(cleanup);

const KIND: TunerGameInfo = {
  game: "nim",
  tuner: {
    id: "nim",
    baselines: ["strong"],
    eval_rounds: 1,
    parameters: [
      { name: "family", type: "categorical", choices: ["ucb1", "rave", "negamax"] },
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
  store.dispatch({ tag: "kindsLoaded", kinds: [KIND] });
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

describe("LaunchForm — excluded families", () => {
  it("puts checked families into exclude_family in selection order", async () => {
    const launchRun = vi.fn((_body: TunerLaunchRequest) => Effect.send(runView()));
    setup({ launchRun });

    fireEvent.click(screen.getByTestId("exclude-family-negamax"));
    fireEvent.click(screen.getByTestId("exclude-family-ucb1"));

    fireEvent.click(screen.getByRole("button", { name: /^Launch$/ }));
    await vi.waitFor(() => expect(launchRun).toHaveBeenCalled());
    const body = launchRun.mock.calls[0]![0] as TunerLaunchRequest;
    expect(body.exclude_family).toEqual(["negamax", "ucb1"]);
  });

  it("omits exclude_family when nothing is checked", async () => {
    const launchRun = vi.fn((_body: TunerLaunchRequest) => Effect.send(runView()));
    setup({ launchRun });
    fireEvent.click(screen.getByRole("button", { name: /^Launch$/ }));
    await vi.waitFor(() => expect(launchRun).toHaveBeenCalled());
    expect((launchRun.mock.calls[0]![0] as TunerLaunchRequest).exclude_family).toBeUndefined();
  });

  it("disables launch when every family is excluded", () => {
    setup();
    for (const f of ["ucb1", "rave", "negamax"]) {
      fireEvent.click(screen.getByTestId(`exclude-family-${f}`));
    }
    expect(screen.getByTestId("exclude-all-error")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /Launch/ })).toBeDisabled();
  });

  it("shows the server preflight error for an illegal exclusion inline", async () => {
    const preflightRun = vi.fn(() =>
      Effect.send({ ok: false, errors: ["family 'ucb1' is required by parameter 'c'"] }),
    );
    setup({ preflightRun });
    fireEvent.click(screen.getByTestId("exclude-family-rave"));

    await vi.waitFor(() =>
      expect(screen.getByTestId("preflight-errors")).toHaveTextContent(
        "family 'ucb1' is required by parameter 'c'",
      ),
    );
  });

  it("hides the checklist entirely for a game with no family axis", () => {
    const store = createStore<TunerState, TunerAction, TunerEnv>(
      initialTunerState(),
      tunerReducer,
      mockTunerEnv(),
    );
    store.dispatch({
      tag: "kindsLoaded",
      kinds: [{ game: "ttt", tuner: { ...KIND.tuner, id: "ttt", parameters: [] } }],
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

    expect(screen.queryByTestId("family-checklist")).not.toBeInTheDocument();
    expect(screen.getByTestId("effort-rows")).toBeInTheDocument();
  });
});
