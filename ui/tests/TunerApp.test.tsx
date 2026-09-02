// tests/TunerApp.test.tsx — component test for the version-4 tuner UI:
// a real `createStore(tunerReducer, env)` inside `<TunerApp>` driven by a
// mocked `TunerEnv` (no live server, no real timers relied on). Mirrors
// `BenchApp.test.tsx` / `GameShell.test.tsx`.

import { afterEach, describe, expect, it, vi } from "vitest";
import { cleanup, fireEvent, render, screen } from "@solidjs/testing-library";
import { Effect } from "@mcts/core";
import { TunerApp } from "@mcts/bench";
import { mockTunerEnv, runView } from "../packages/bench/tests/tuner/mock-tuner-env.js";
import type { TunerLaunchRequest } from "@mcts/bench";

afterEach(() => {
  cleanup();
  window.location.hash = "";
});

const kinds = [
  {
    game: "nim",
    tuner: { id: "strategy", baselines: ["strong"], eval_rounds: 20, parameters: [], conditions: [], game_config: {} },
  },
];
const objectives = [{ key: "nim-v1", objective_id: "nim-v1", game_kind: "nim" }];

describe("TunerApp fleet", () => {
  it("renders the KPI row and a completed run from the projection", async () => {
    render(() => (
      <TunerApp
        env={mockTunerEnv({
          listKinds: () => Effect.send(kinds),
          listObjectives: () => Effect.send(objectives),
          listRuns: () => Effect.send([]),
          listProjectionRuns: () =>
            Effect.send([
              {
                run_id: "done-1",
                terminal_status: "exited",
                report_available: true,
                ingest_error: null,
                game_kind: "nim",
                objective_id: "nim-v1",
                shadow_policy_kind: "paired_bootstrap",
                active_elimination: false,
                report_status: "complete",
                validation_claim: "production validation",
                total_pair_attempts: 240,
                total_completed_pairs: 236,
              },
            ]),
        })}
      />
    ));

    await vi.waitFor(() => expect(screen.getByTestId("kpi-complete")).toHaveTextContent("1 complete"));
    expect(screen.getByText("done-1")).toBeInTheDocument();
    expect(screen.getByText("production validation")).toBeInTheDocument();
  });

  it("launches a run and navigates to its overview", async () => {
    const launchRun = vi.fn((_req: TunerLaunchRequest) =>
      Effect.send(runView({ run_id: "launched-1", status: "live" })),
    );
    render(() => (
      <TunerApp
        env={mockTunerEnv({
          listKinds: () => Effect.send(kinds),
          listObjectives: () => Effect.send(objectives),
          listRuns: () => Effect.send([]),
          launchRun,
          getRunLog: () => Effect.send({ lines: ["booting"], next_offset: 7, err_lines: [] }),
        })}
      />
    ));

    await vi.waitFor(() => expect(screen.getByTestId("tuner-fleet")).toBeInTheDocument());
    fireEvent.click(screen.getByRole("button", { name: "New run" }));

    await vi.waitFor(() => expect(screen.getByText("Launch a tuner run")).toBeInTheDocument());
    fireEvent.click(screen.getByRole("button", { name: "Launch" }));

    await vi.waitFor(() => expect(launchRun).toHaveBeenCalledTimes(1));
    expect(launchRun.mock.calls[0]![0]).toMatchObject({ game_kind: "nim", objective_key: "nim-v1" });

    await vi.waitFor(() =>
      expect(screen.getByTestId("tuner-run-overview")).toHaveTextContent("launched-1"),
    );
    await vi.waitFor(() => expect(screen.getByTestId("tuner-log-lines")).toHaveTextContent("booting"));
  });
});
