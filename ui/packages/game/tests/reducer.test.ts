// tests/reducer.test.ts — Tests for appReducer's aiMove/analysis job-poll
// wiring: PLAN-UI.md session 3 wires ai_move/analyze through
// `@mcts/core`'s `jobPollReduce` even though the transport is a single
// blocking request, not a real submit-then-poll pair -- `submitJob()`
// resolves directly to `{status: "done", result}`. These tests confirm that
// synchronous-resolve path produces the same status transitions a real poll
// loop would (mirrors pb/ui/tests/features/diagrams.test.ts's
// "populates ... via the job-poll cache-hit ('done') path" test).

import { describe, it, expect } from "vitest";
import { Effect } from "@mcts/core";
import { createTestStore } from "../../../tests/test-store.js";
import { appReducer, type Env } from "../src/reducer.js";
import { initialAppState } from "../src/state.js";
import type { AiMoveResult, Analysis } from "../src/types.js";

// Test-only state/move types -- appReducer never inspects their shape.
type S = number;
type M = string;

const mockEnv: Env = {
  getGames: () => Effect.none(),
  newGame: () => Effect.none(),
  legalMoves: () => Effect.none(),
  view: () => Effect.none(),
  apply: () => Effect.none(),
  aiPresets: () => Effect.none(),
  aiMove: () => Effect.none(),
  analyze: () => Effect.none(),
};

describe("appReducer / aiMove", () => {
  it("request -> submitted('done') resolves synchronously, same as a real poll loop's terminal state", () => {
    const result: AiMoveResult<S, M> = { move: "b", state: 1, view: {} };
    const seen: { kind: string; state: S; preset: string }[] = [];
    const env: Env = {
      ...mockEnv,
      // `Env.aiMove` is generic per call (it serves every game kind); this
      // mock only ever runs against this test's own S/M, so it captures the
      // call and returns this test's fixed `result` regardless of what the
      // caller's own S2/M2/V2 happen to be -- sound here because the caller
      // (appReducer, below) is itself instantiated at S/M for this test.
      aiMove: <S2, M2, V2 = unknown>(kind: string, state: S2, preset: string) => {
        seen.push({ kind, state: state as unknown as S, preset });
        return Effect.send(result) as unknown as Effect<AiMoveResult<S2, M2, V2>>;
      },
    };
    const init = initialAppState<S, M>("druid", 7);
    const ts = createTestStore(appReducer<S, M>, env, init);

    ts.send({ tag: "aiMove", action: { tag: "request", preset: "master" } }, (s) => {
      s.aiMove.status = "pending";
    });
    ts.receive(
      {
        tag: "aiMove",
        action: { tag: "job", action: { tag: "submitted", result: { status: "done", result } } },
      },
      (s) => {
        s.aiMove.status = "done";
        s.aiMove.result = result;
      },
    );

    // env.aiMove is called with the tree's current state and gameKind, not
    // just the preset -- confirms the dynamic jobEnv construction (reading
    // draft.gameKind / draft.tree.nodes[draft.tree.currentId]) actually wires
    // through, not just the job-poll status machinery.
    expect(seen).toEqual([{ kind: "druid", state: 7, preset: "master" }]);
  });
});

describe("appReducer / analysis", () => {
  it("request -> submitted('done') resolves synchronously, same as a real poll loop's terminal state", () => {
    const result: Analysis<M> = { actions: [], principal_variation: [], total_visits: 5, suggested_move: null };
    const env: Env = {
      ...mockEnv,
      analyze: <M2>() => Effect.send(result) as unknown as Effect<Analysis<M2>>,
    };
    const init = initialAppState<S, M>("test-kind", 0);
    const ts = createTestStore(appReducer<S, M>, env, init);

    ts.send({ tag: "analysis", action: { tag: "request", preset: "strong", budgetMs: 1000 } }, (s) => {
      s.analysis.status = "pending";
    });
    ts.receive(
      {
        tag: "analysis",
        action: { tag: "job", action: { tag: "submitted", result: { status: "done", result } } },
      },
      (s) => {
        s.analysis.status = "done";
        s.analysis.result = result;
      },
    );
  });
});
