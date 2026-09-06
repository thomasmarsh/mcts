// resume-run.component.test.tsx — component test for the plain "Resume"
// button on RunOverview. A real `createStore(tunerReducer, env)` backs the
// rendered view; the env is mocked (AGENTS.md "mock the environment"), no
// live server.

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
import type { TunerRunView } from "../../src/tuner/tuner-types.js";

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

describe("RunOverview — resume", () => {
  it("shows Resume for a run its reaper marked lost", () => {
    renderOverview(
      runView({ run_id: "lost-1", status: "exited", terminal_outcome: "lost" }),
    );
    expect(screen.getByTestId("resume-run")).toBeInTheDocument();
  });

  it("shows Resume for a cleanly-frozen run", () => {
    renderOverview(
      runView({ run_id: "frozen-1", status: "exited", terminal_outcome: "exited" }),
    );
    expect(screen.getByTestId("resume-run")).toBeInTheDocument();
  });

  it("dispatches resumeRun on click", async () => {
    const resumeRun = vi.fn((_id: string) =>
      Effect.send(runView({ run_id: "lost-1", status: "live" })),
    );
    renderOverview(
      runView({ run_id: "lost-1", status: "exited", terminal_outcome: "lost" }),
      mockTunerEnv({ resumeRun, listRuns: () => Effect.send([]) }),
    );

    fireEvent.click(screen.getByTestId("resume-run"));

    await vi.waitFor(() => expect(resumeRun).toHaveBeenCalledTimes(1));
    expect(resumeRun.mock.calls[0]![0]).toBe("lost-1");
  });

  it("surfaces a server rejection inline", async () => {
    const resumeRun = vi.fn(() =>
      Effect.fromPromise(() => Promise.reject(new Error("tuner run directory is missing"))),
    );
    renderOverview(
      runView({ run_id: "lost-1", status: "exited", terminal_outcome: "lost" }),
      mockTunerEnv({ resumeRun }),
    );

    fireEvent.click(screen.getByTestId("resume-run"));

    await vi.waitFor(() =>
      expect(screen.getByTestId("resume-error")).toHaveTextContent(
        "tuner run directory is missing",
      ),
    );
  });

  it("hides Resume for a live run", () => {
    renderOverview(runView({ run_id: "live-1", status: "live" }));
    expect(screen.queryByTestId("resume-run")).not.toBeInTheDocument();
  });

  it("hides Resume for a run that failed before writing a manifest", () => {
    renderOverview(
      runView({
        run_id: "doomed-1",
        status: "failed",
        terminal_outcome: "exited",
        error_detail: "run directory already exists",
      }),
    );
    expect(screen.queryByTestId("resume-run")).not.toBeInTheDocument();
  });

  it("hides Resume for a run that was killed", () => {
    renderOverview(
      runView({ run_id: "killed-1", status: "exited", terminal_outcome: "signalled" }),
    );
    expect(screen.queryByTestId("resume-run")).not.toBeInTheDocument();
  });
});
