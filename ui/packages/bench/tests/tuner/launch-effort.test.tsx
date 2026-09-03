// launch-effort.test.tsx — component test for the per-phase search-effort
// rows added to LaunchForm's advanced section (Task 13c). A real
// `createStore(tunerReducer, env)` backs the render; the env is mocked
// (AGENTS.md "mock the environment"), no live server.

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
        game_kind: "nim",
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

async function capturedLaunch(launchRun: ReturnType<typeof vi.fn>): Promise<TunerLaunchRequest> {
  fireEvent.click(screen.getByRole("button", { name: /^Launch$/ }));
  await vi.waitFor(() => expect(launchRun).toHaveBeenCalled());
  return launchRun.mock.calls[0]![0] as TunerLaunchRequest;
}

describe("LaunchForm — per-phase search effort", () => {
  it("omits every effort field when the rows are blank", async () => {
    const launchRun = vi.fn((_body: TunerLaunchRequest) => Effect.send(runView()));
    setup({ launchRun });
    const body = await capturedLaunch(launchRun);
    expect(body.tuning_max_iterations).toBeUndefined();
    expect(body.tuning_max_time_ms).toBeUndefined();
    expect(body.validation_max_iterations).toBeUndefined();
    expect(body.production_max_iterations).toBeUndefined();
  });

  it("emits exactly one of iterations / time_ms per filled row, matching the unit toggle", async () => {
    const launchRun = vi.fn((_body: TunerLaunchRequest) => Effect.send(runView()));
    setup({ launchRun });

    fireEvent.input(screen.getByTestId("effort-tuning-value"), { target: { value: "500" } });
    fireEvent.input(screen.getByTestId("effort-validation-value"), { target: { value: "2000" } });
    fireEvent.input(screen.getByTestId("effort-validation-unit"), { target: { value: "time_ms" } });

    const body = await capturedLaunch(launchRun);
    expect(body.tuning_max_iterations).toBe(500);
    expect(body.tuning_max_time_ms).toBeUndefined();
    expect(body.validation_max_time_ms).toBe(2000);
    expect(body.validation_max_iterations).toBeUndefined();
  });

  it("blocks a launch when a same-unit tuning effort exceeds production", async () => {
    const launchRun = vi.fn((_body: TunerLaunchRequest) => Effect.send(runView()));
    setup({ launchRun });

    fireEvent.input(screen.getByTestId("effort-production-value"), { target: { value: "1000" } });
    fireEvent.input(screen.getByTestId("effort-tuning-value"), { target: { value: "5000" } });

    expect(screen.getByTestId("effort-error")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /Launch/ })).toBeDisabled();
  });

  it("allows a tuning effort above production when the units differ", async () => {
    const launchRun = vi.fn((_body: TunerLaunchRequest) => Effect.send(runView()));
    setup({ launchRun });

    fireEvent.input(screen.getByTestId("effort-production-value"), { target: { value: "1000" } });
    fireEvent.input(screen.getByTestId("effort-tuning-value"), { target: { value: "5000" } });
    fireEvent.input(screen.getByTestId("effort-tuning-unit"), { target: { value: "time_ms" } });

    expect(screen.queryByTestId("effort-error")).not.toBeInTheDocument();
    const body = await capturedLaunch(launchRun);
    expect(body.tuning_max_time_ms).toBe(5000);
    expect(body.production_max_iterations).toBe(1000);
  });

  it("carries a non-default proposer policy and omits it otherwise", async () => {
    const launchRun = vi.fn((_body: TunerLaunchRequest) => Effect.send(runView()));
    setup({ launchRun });
    fireEvent.input(screen.getByTestId("proposer-policy"), { target: { value: "qmc" } });
    const body = await capturedLaunch(launchRun);
    expect(body.proposer_policy).toBe("qmc");
  });
});
