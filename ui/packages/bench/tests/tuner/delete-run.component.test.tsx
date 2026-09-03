// delete-run.component.test.tsx — component test for the delete control on
// terminal RunCards in the FleetDashboard (Task 13d). A real
// `createStore(tunerReducer, env)` backs the rendered view; the env is
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
import { FleetDashboard } from "../../src/tuner/views/FleetDashboard.js";
import { mockTunerEnv } from "./mock-tuner-env.js";
import type { TunerEnv } from "../../src/tuner/tuner-env.js";
import type { ProjectionRunListItem } from "../../src/tuner/tuner-types.js";

afterEach(() => {
  cleanup();
  vi.unstubAllGlobals();
});

/** happy-dom has no `window.confirm`; install a mock and return it. */
function stubConfirm(fn: (message?: string) => boolean) {
  const mock = vi.fn(fn);
  vi.stubGlobal("confirm", mock);
  return mock;
}

const finished: ProjectionRunListItem = {
  run_id: "done-1",
  terminal_status: "complete",
  report_available: true,
  ingest_error: null,
  game_kind: "druid",
  objective_id: "druid-default",
  shadow_policy_kind: null,
  active_elimination: null,
  report_status: "complete",
  validation_claim: "ship",
  total_pair_attempts: 120,
  total_completed_pairs: 120,
};

function renderFleet(env: TunerEnv = mockTunerEnv()) {
  const store = createStore<TunerState, TunerAction, TunerEnv>(
    initialTunerState(),
    tunerReducer,
    env,
  );
  store.dispatch({ tag: "projectionLoaded", runs: [finished] });
  render(() => <FleetDashboard store={store} navigate={() => {}} />);
  return store;
}

describe("FleetDashboard — delete run", () => {
  it("deletes a terminal run only after the confirm is accepted, then refetches", async () => {
    const deleteRun = vi.fn((_runId: string) => Effect.send(undefined));
    const listProjectionRuns = vi.fn(() => Effect.send<ProjectionRunListItem[]>([]));
    const env = mockTunerEnv({ deleteRun, listProjectionRuns, listRuns: () => Effect.send([]) });

    // Reject the first confirm, accept the second.
    let calls = 0;
    const confirm = stubConfirm(() => ++calls > 1);

    renderFleet(env);
    expect(listProjectionRuns).not.toHaveBeenCalled();

    const button = () => screen.getByTestId("run-card-delete");
    fireEvent.click(button());
    expect(confirm).toHaveBeenCalledTimes(1);
    expect(deleteRun).not.toHaveBeenCalled();

    fireEvent.click(button());
    expect(confirm).toHaveBeenCalledTimes(2);
    await vi.waitFor(() => expect(deleteRun).toHaveBeenCalledTimes(1));
    expect(deleteRun.mock.calls[0]![0]).toBe("done-1");
    // The success path reconciles the fleet from the server.
    await vi.waitFor(() => expect(listProjectionRuns).toHaveBeenCalled());
    await vi.waitFor(() =>
      expect(screen.queryByTestId("run-card-delete")).not.toBeInTheDocument(),
    );
  });

  it("surfaces a 409 for a live run inline", async () => {
    const env = mockTunerEnv({
      deleteRun: () =>
        Effect.fromPromise(() => Promise.reject(new Error("run is still live -- stop it first"))),
    });
    stubConfirm(() => true);

    renderFleet(env);
    fireEvent.click(screen.getByTestId("run-card-delete"));

    await vi.waitFor(() =>
      expect(screen.getByRole("alert")).toHaveTextContent("run is still live"),
    );
    // The run card is still there.
    expect(screen.getByTestId("run-card-delete")).toBeInTheDocument();
  });
});
