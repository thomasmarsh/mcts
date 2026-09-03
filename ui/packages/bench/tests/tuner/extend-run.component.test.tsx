// extend-run.component.test.tsx — component test for the "Extend budget"
// control on RunOverview (Task 13b). A real `createStore(tunerReducer, env)`
// backs the rendered view; the env is mocked (AGENTS.md "mock the
// environment"), no live server.

import { afterEach, describe, expect, it, vi } from "vitest";
import { cleanup, fireEvent, render, screen } from "@solidjs/testing-library";
import { Effect, createStore } from "@mcts/core";
import {
  initialTunerState,
  tunerReducer,
  type TunerAction,
  type TunerState,
} from "../../src/tuner/tuner-reducer.js";
import { RunOverview } from "../../src/tuner/views/RunOverview.js";
import { mockTunerEnv, runView } from "./mock-tuner-env.js";
import type { TunerEnv } from "../../src/tuner/tuner-env.js";
import type { TunerBudgetExtension, TunerRunView } from "../../src/tuner/tuner-types.js";

afterEach(cleanup);

function renderOverview(run: TunerRunView, env: TunerEnv = mockTunerEnv()) {
  const store = createStore<TunerState, TunerAction, TunerEnv>(
    initialTunerState(),
    tunerReducer,
    env,
  );
  store.dispatch({ tag: "runsLoaded", runs: [run] });
  render(() => <RunOverview store={store} runId={run.run_id} navigate={() => {}} />);
  return store;
}

const frozen = runView({
  run_id: "frozen-1",
  status: "exited",
  terminal_outcome: "exited",
});

describe("RunOverview — extend budget", () => {
  it("shows the form for a frozen run and enables submit only with a positive delta and a reason", () => {
    renderOverview(frozen);

    expect(screen.getByTestId("extend-budget-form")).toBeInTheDocument();
    const submit = screen.getByTestId("extend-submit") as HTMLButtonElement;
    expect(submit).toBeDisabled();

    // Reason alone: still nothing to add.
    fireEvent.input(screen.getByTestId("extend-reason"), {
      target: { value: "cohort race unresolved" },
    });
    expect(submit).toBeDisabled();

    // A positive delta with the reason present: enabled.
    fireEvent.input(screen.getByTestId("extend-tuning-delta"), { target: { value: "40" } });
    expect(submit).not.toBeDisabled();

    // Clearing the reason disables it again.
    fireEvent.input(screen.getByTestId("extend-reason"), { target: { value: "  " } });
    expect(submit).toBeDisabled();
  });

  it("dispatches extendRun with the typed deltas and clears the form on success", async () => {
    const extendRun = vi.fn((_id: string, _body: TunerBudgetExtension) =>
      Effect.send(runView({ run_id: "frozen-1", status: "live" })),
    );
    renderOverview(frozen, mockTunerEnv({ extendRun, listRuns: () => Effect.send([frozen]) }));

    fireEvent.input(screen.getByTestId("extend-tuning-delta"), { target: { value: "40" } });
    fireEvent.input(screen.getByTestId("extend-diagnostic-delta"), { target: { value: "8" } });
    fireEvent.input(screen.getByTestId("extend-reason"), { target: { value: "more diagnostics" } });
    fireEvent.click(screen.getByTestId("extend-submit"));

    await vi.waitFor(() => expect(extendRun).toHaveBeenCalledTimes(1));
    expect(extendRun.mock.calls[0]![1]).toEqual({
      tuning_pair_attempts_delta: 40,
      validation_pair_attempts_delta: 0,
      diagnostic_pair_attempts_delta: 8,
      reason: "more diagnostics",
    });
    await vi.waitFor(() =>
      expect(screen.getByTestId("extend-tuning-delta")).toHaveValue(null),
    );
    expect(screen.getByTestId("extend-reason")).toHaveValue("");
  });

  it("surfaces a server rejection inline and keeps the form values", async () => {
    const extendRun = vi.fn(() =>
      Effect.fromPromise(() => Promise.reject(new Error("delta not divisible by finalists"))),
    );
    renderOverview(frozen, mockTunerEnv({ extendRun }));

    fireEvent.input(screen.getByTestId("extend-tuning-delta"), { target: { value: "41" } });
    fireEvent.input(screen.getByTestId("extend-reason"), { target: { value: "retry" } });
    fireEvent.click(screen.getByTestId("extend-submit"));

    await vi.waitFor(() =>
      expect(screen.getByTestId("extend-error")).toHaveTextContent(
        "delta not divisible by finalists",
      ),
    );
    expect(screen.getByTestId("extend-tuning-delta")).toHaveValue(41);
    expect(screen.getByTestId("extend-reason")).toHaveValue("retry");
  });

  it("hides the form for a live run", () => {
    renderOverview(runView({ run_id: "live-1", status: "live" }));
    expect(screen.queryByTestId("extend-budget-form")).not.toBeInTheDocument();
  });

  it("hides the form for a run that failed before writing a manifest", () => {
    renderOverview(
      runView({
        run_id: "doomed-1",
        status: "failed",
        terminal_outcome: "exited",
        error_detail: "run directory already exists",
      }),
    );
    expect(screen.queryByTestId("extend-budget-form")).not.toBeInTheDocument();
  });

  it("hides the form for a run that was killed", () => {
    renderOverview(
      runView({ run_id: "killed-1", status: "exited", terminal_outcome: "signalled" }),
    );
    expect(screen.queryByTestId("extend-budget-form")).not.toBeInTheDocument();
  });
});
