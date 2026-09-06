// tests/reducer.test.ts — Tests for appReducer's aiMove/analysis job-poll
// wiring: ai_move/analyze are wired through
// `@mcts/core`'s `jobPollReduce` even though the transport is a single
// blocking request, not a real submit-then-poll pair -- `submitJob()`
// resolves directly to `{status: "done", result}`. These tests confirm that
// synchronous-resolve path produces the same status transitions a real poll
// loop would (mirrors pb/ui/tests/features/diagrams.test.ts's
// "populates ... via the job-poll cache-hit ('done') path" test).

import { describe, it, expect } from "vitest";
import { Effect } from "@mcts/core";
import { createTestStore } from "../../../tests/test-store.js";
import { appReducer, type AppAction, type Env } from "../src/reducer.js";
import { initialAppState, type AppState } from "../src/state.js";
import { gameTreeReducer, initialGameTree } from "../src/game-tree.js";
import type {
  AiMoveResult,
  AiStrategyRef,
  Analysis,
  SearchReport,
  StateAndView,
} from "../src/types.js";

// Test-only state/move types -- appReducer never inspects their shape.
type S = number;
type M = string;

function searchReport(selectedAction: M): SearchReport<M> {
  return {
    status: "available",
    schema_version: 1,
    reason: null,
    elapsed_seconds: 0.5,
    iteration_limit: 100,
    time_limit_seconds: null,
    completed_iterations: 100,
    termination: "iterations",
    selected_action: selectedAction,
    actions: [],
    principal_variation: [selectedAction],
    root_visits: 100,
    tree_nodes: 101,
    mean_depth: 2,
    max_depth: 4,
    graph_mode: "tree",
    tt_reads: 0,
    tt_writes: 0,
    tt_hits: 0,
    tt_hit_ratio: null,
    iterations_per_second: 200,
    warnings: [],
  };
}

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
    const report = searchReport("b");
    const result: AiMoveResult<S, M> = { move: "b", state: 1, view: {}, search: report };
    const seen: { kind: string; state: S; strategy: AiStrategyRef }[] = [];
    const env: Env = {
      ...mockEnv,
      // `Env.aiMove` is generic per call (it serves every game kind); this
      // mock only ever runs against this test's own S/M, so it captures the
      // call and returns this test's fixed `result` regardless of what the
      // caller's own S2/M2/V2 happen to be -- sound here because the caller
      // (appReducer, below) is itself instantiated at S/M for this test.
      aiMove: <S2, M2, V2 = unknown>(kind: string, state: S2, strategy: AiStrategyRef) => {
        seen.push({ kind, state: state as unknown as S, strategy });
        return Effect.send(result) as unknown as Effect<AiMoveResult<S2, M2, V2>>;
      },
    };
    const init = initialAppState<S, M>("druid", 7);
    const ts = createTestStore(appReducer<S, M>, env, init);

    ts.send(
      { tag: "aiMove", action: { tag: "request", strategy: { kind: "preset", id: "master" } } },
      (s) => {
        s.aiMove.status = "pending";
      },
    );
    ts.receive(
      {
        tag: "aiMove",
        action: { tag: "job", action: { tag: "submitted", result: { status: "done", result } } },
        epoch: 0,
      },
      (s) => {
        s.aiMove.status = "done";
        s.aiMove.result = result;
        // appReducer folds a completed aiMove straight into the tree in the
        // same reduction, same as a human "move" -- see reducer.ts.
        const rootId = s.tree.rootId;
        const nextId = `n${s.tree.nextId}`;
        s.tree.nodes[rootId]!.childIds.push(nextId);
        s.tree.nodes[nextId] = {
          id: nextId,
          state: result.state,
          move: result.move,
          search: report,
          parentId: rootId,
          childIds: [],
        };
        s.tree.currentId = nextId;
        s.tree.nextId += 1;
      },
    );

    // env.aiMove is called with the tree's current state and gameKind, not
    // just the strategy -- confirms the dynamic jobEnv construction (reading
    // draft.gameKind / draft.tree.nodes[draft.tree.currentId]) actually wires
    // through, not just the job-poll status machinery.
    expect(seen).toEqual([{ kind: "druid", state: 7, strategy: { kind: "preset", id: "master" } }]);
  });

  it("drops a stale AI completion without creating a move or retaining its report", () => {
    const init = initialAppState<S, M>("druid", 7);
    init.epoch = 1;
    const ts = createTestStore(appReducer<S, M>, mockEnv, init);
    const result: AiMoveResult<S, M> = { move: "b", state: 1, view: {}, search: searchReport("b") };

    ts.send({
      tag: "aiMove",
      action: { tag: "job", action: { tag: "submitted", result: { status: "done", result } } },
      epoch: 0,
    });

    expect(ts.getState().tree.nodes[ts.getState().tree.rootId]?.search).toBeNull();
    expect(Object.keys(ts.getState().tree.nodes)).toHaveLength(1);
  });

  // A `{kind: "custom", spec}` strategy must reach `env.aiMove` unchanged --
  // `appReducer` itself never inspects an `AiStrategyRef`'s shape, only
  // forwards it (the `preset`/`custom` split happens at the `api-client.ts`
  // HTTP boundary, one layer below this reducer test).
  it("forwards a custom AiStrategyRef to env.aiMove unchanged", () => {
    const result: AiMoveResult<S, M> = { move: "b", state: 1, view: {} };
    const customStrategy: AiStrategyRef = {
      kind: "custom",
      spec: {
        search: {
          select: { kind: "ucb1", c: 1.4 },
          simulate: { kind: "uniform" },
          backprop: { kind: "classic" },
          final_action: { kind: "robust_child" },
        },
        max_iterations: 500,
      },
    };
    const seen: AiStrategyRef[] = [];
    const env: Env = {
      ...mockEnv,
      aiMove: <S2, M2, V2 = unknown>(_kind: string, _state: S2, strategy: AiStrategyRef) => {
        seen.push(strategy);
        return Effect.send(result) as unknown as Effect<AiMoveResult<S2, M2, V2>>;
      },
    };
    const init = initialAppState<S, M>("druid", 7);
    const ts = createTestStore(appReducer<S, M>, env, init);

    ts.send({ tag: "aiMove", action: { tag: "request", strategy: customStrategy } }, (s) => {
      s.aiMove.status = "pending";
    });
    ts.receive(
      {
        tag: "aiMove",
        action: { tag: "job", action: { tag: "submitted", result: { status: "done", result } } },
        epoch: 0,
      },
      (s) => {
        s.aiMove.status = "done";
        s.aiMove.result = result;
        const rootId = s.tree.rootId;
        const nextId = `n${s.tree.nextId}`;
        s.tree.nodes[rootId]!.childIds.push(nextId);
        s.tree.nodes[nextId] = {
          id: nextId,
          state: result.state,
          move: result.move,
          search: null,
          parentId: rootId,
          childIds: [],
        };
        s.tree.currentId = nextId;
        s.tree.nextId += 1;
      },
    );

    expect(seen).toEqual([customStrategy]);
  });

  // Regression test: `GameShell`'s autoplay effect re-fires whenever the
  // store updates and it's still an AI-controlled seat's turn -- a failure
  // (bad custom config, a crashing subprocess, any transport error) flips
  // `aiMove.status` from "pending" to "error", which by itself looks
  // identical to "safe to try again" from that effect's point of view.
  // `aiMoveFailedNodeId` is the state this reducer owns so `GameShell` can
  // tell "already failed at this exact node" apart from "a fresh node,
  // worth trying" without re-deriving it locally -- see `AppState.
  // aiMoveFailedNodeId`'s doc comment.
  it("a failed aiMove records the node it failed at, in aiMoveFailedNodeId", async () => {
    const env: Env = {
      ...mockEnv,
      aiMove: () => Effect.fromPromise(() => Promise.reject(new Error("subprocess crashed"))),
    };
    const init = initialAppState<S, M>("druid", 7);
    const ts = createTestStore(appReducer<S, M>, env, init);
    const rootId = init.tree.rootId;

    ts.send(
      { tag: "aiMove", action: { tag: "request", strategy: { kind: "preset", id: "master" } } },
      (s) => {
        s.aiMove.status = "pending";
      },
    );
    await ts.drain();
    ts.receive(
      {
        tag: "aiMove",
        action: { tag: "job", action: { tag: "failed", error: "Error: subprocess crashed" } },
        epoch: 0,
      },
      (s) => {
        s.aiMove.status = "error";
        s.aiMove.error = "Error: subprocess crashed";
        s.aiMoveFailedNodeId = rootId;
      },
    );
  });
});

describe("appReducer / analysis", () => {
  it("request -> submitted('done') resolves synchronously, same as a real poll loop's terminal state", () => {
    const result: Analysis<M> = {
      actions: [],
      principal_variation: [],
      total_visits: 5,
      suggested_move: null,
      search: searchReport("analysis-only"),
    };
    const env: Env = {
      ...mockEnv,
      analyze: <M2>() => Effect.send(result) as unknown as Effect<Analysis<M2>>,
    };
    const init = initialAppState<S, M>("test-kind", 0);
    const ts = createTestStore(appReducer<S, M>, env, init);

    ts.send(
      {
        tag: "analysis",
        action: { tag: "request", strategy: { kind: "preset", id: "strong" }, budgetMs: 1000 },
      },
      (s) => {
        s.analysis.status = "pending";
      },
    );
    ts.receive(
      {
        tag: "analysis",
        action: { tag: "job", action: { tag: "submitted", result: { status: "done", result } } },
        epoch: 0,
      },
      (s) => {
        s.analysis.status = "done";
        s.analysis.result = result;
      },
    );
  });

  it("forwards a custom AiStrategyRef to env.analyze unchanged", () => {
    const result: Analysis<M> = {
      actions: [],
      principal_variation: [],
      total_visits: 5,
      suggested_move: null,
    };
    const customStrategy: AiStrategyRef = {
      kind: "custom",
      spec: {
        search: {
          select: { kind: "epsilon_greedy", epsilon: 0.1, inner: { kind: "ucb1", c: 1.4 } },
          simulate: { kind: "mast" },
          backprop: { kind: "classic" },
          final_action: { kind: "secure_child", a: 4.0 },
        },
        max_time_ms: 2000,
      },
    };
    const seen: { strategy: AiStrategyRef; budgetMs: number | undefined }[] = [];
    const env: Env = {
      ...mockEnv,
      analyze: <M2>(_kind: string, _state: unknown, strategy: AiStrategyRef, budgetMs?: number) => {
        seen.push({ strategy, budgetMs });
        return Effect.send(result) as unknown as Effect<Analysis<M2>>;
      },
    };
    const init = initialAppState<S, M>("test-kind", 0);
    const ts = createTestStore(appReducer<S, M>, env, init);

    ts.send(
      { tag: "analysis", action: { tag: "request", strategy: customStrategy, budgetMs: 1500 } },
      (s) => {
        s.analysis.status = "pending";
      },
    );
    ts.receive(
      {
        tag: "analysis",
        action: { tag: "job", action: { tag: "submitted", result: { status: "done", result } } },
        epoch: 0,
      },
      (s) => {
        s.analysis.status = "done";
        s.analysis.result = result;
      },
    );

    expect(seen).toEqual([{ strategy: customStrategy, budgetMs: 1500 }]);
  });

  // A stale `analysis` result would otherwise go on labeling a
  // position it was never computed for once the tree moves on -- the
  // heatmap overlay/suggested-move highlight derive straight from this
  // field, so a leftover result renders against the wrong board.
  it("resets once a tree navigation changes the current position", () => {
    const result: Analysis<M> = {
      actions: [],
      principal_variation: [],
      total_visits: 5,
      suggested_move: null,
    };
    const env: Env = {
      ...mockEnv,
      analyze: <M2>() => Effect.send(result) as unknown as Effect<Analysis<M2>>,
    };
    const init = initialAppState<S, M>("druid", 0);
    const ts = createTestStore(appReducer<S, M>, env, init);

    ts.send(
      { tag: "analysis", action: { tag: "request", strategy: { kind: "preset", id: "strong" } } },
      (s) => {
        s.analysis.status = "pending";
      },
    );
    ts.receive(
      {
        tag: "analysis",
        action: { tag: "job", action: { tag: "submitted", result: { status: "done", result } } },
        epoch: 0,
      },
      (s) => {
        s.analysis.status = "done";
        s.analysis.result = result;
      },
    );

    // `undo` at the root is itself a no-op for `currentId`, but the reset is
    // unconditional on any tree action (see reducer.ts) rather than trying
    // to detect whether `currentId` actually moved.
    ts.send({ tag: "tree", action: { tag: "undo" } }, (s) => {
      s.analysis.status = "idle";
      s.analysis.result = null;
    });
  });

  it("resets once a human move completes and advances the tree", () => {
    const analysisResult: Analysis<M> = {
      actions: [],
      principal_variation: [],
      total_visits: 5,
      suggested_move: null,
    };
    const moveResult: StateAndView<S> = { state: 1, view: {} };
    const env: Env = {
      ...mockEnv,
      analyze: <M2>() => Effect.send(analysisResult) as unknown as Effect<Analysis<M2>>,
      apply: <S2, V2 = unknown>() =>
        Effect.send(moveResult) as unknown as Effect<StateAndView<S2, V2>>,
    };
    const init = initialAppState<S, M>("druid", 0);
    const ts = createTestStore(appReducer<S, M>, env, init);

    ts.send(
      { tag: "analysis", action: { tag: "request", strategy: { kind: "preset", id: "strong" } } },
      (s) => {
        s.analysis.status = "pending";
      },
    );
    ts.receive(
      {
        tag: "analysis",
        action: {
          tag: "job",
          action: { tag: "submitted", result: { status: "done", result: analysisResult } },
        },
        epoch: 0,
      },
      (s) => {
        s.analysis.status = "done";
        s.analysis.result = analysisResult;
      },
    );

    ts.send({ tag: "move", action: { tag: "request", move: "a" } }, (s) => {
      s.move.status = "pending";
    });
    ts.receive(
      {
        tag: "move",
        action: {
          tag: "job",
          action: { tag: "submitted", result: { status: "done", result: moveResult } },
        },
        move: "a",
      },
      (s) => {
        s.move.status = "done";
        s.move.result = moveResult;
        const rootId = s.tree.rootId;
        const nextId = `n${s.tree.nextId}`;
        s.tree.nodes[rootId]!.childIds.push(nextId);
        s.tree.nodes[nextId] = {
          id: nextId,
          state: moveResult.state,
          move: "a",
          search: null,
          parentId: rootId,
          childIds: [],
        };
        s.tree.currentId = nextId;
        s.tree.nextId += 1;
        s.analysis.status = "idle";
        s.analysis.result = null;
      },
    );
  });
});

// A regression suite for the bug this invariant fixes: `GameShell`'s
// autoplay effect derives `currentPlayer` from `position`, and used to be
// able to read it one node behind `tree.currentId` right after an aiMove
// folded into the tree (the `position/request` re-fetch it triggers doesn't
// resolve synchronously) -- which, for a config where one seat is
// AI-controlled and the other human, could fire an *extra* AI move on the
// human seat's turn before the human ever saw a legal-moves list that
// matched the real board. Every branch below that can move `tree.currentId`
// must null `position` in the same reduction, so `position`/`summary` are
// never non-null for a stale node -- callers get this for free rather than
// each having to compare `position.nodeId` to `tree.currentId` themselves.
describe("appReducer / position", () => {
  // `position`'s "request" branch awaits real `view`/`legalMoves` effects
  // (via `Promise.all`, see reducer.ts) rather than resolving synchronously
  // like the job-poll-wrapped calls elsewhere in this file -- `mockEnv`'s
  // `Effect.none()` stubs for both never resolve, so every test below needs
  // an `env` that actually answers them.
  const positionEnv: Env = {
    ...mockEnv,
    view: <V2 = unknown>() => Effect.send({}) as unknown as Effect<V2>,
    legalMoves: <M2>() => Effect.send({ moves: [] }) as unknown as Effect<{ moves: M2[] }>,
  };

  // `position`'s "loaded" arrives via an `Effect.fromPromise` (unlike the
  // job-poll-wrapped calls elsewhere in this file, which resolve inline
  // through `Effect.send`), so it needs a real microtask flush -- `drain()`
  // -- before `receive()` can see it queued.
  async function loadPosition<S2, M2>(
    ts: ReturnType<typeof createTestStore<AppState<S2, M2>, AppAction<S2, M2>, Env>>,
    nodeId: string,
  ) {
    ts.send({ tag: "position", action: { tag: "request" } });
    await ts.drain();
    ts.receive(
      { tag: "position", action: { tag: "loaded", nodeId, epoch: 0, view: {}, moves: [] } },
      (s) => {
        s.position = { nodeId, view: {}, legalMoves: [] };
      },
    );
  }

  it("is nulled by a tree navigation that changes currentId", async () => {
    const init = initialAppState<S, M>("druid", 0);
    const ts = createTestStore(appReducer<S, M>, positionEnv, init);
    await loadPosition(ts, "n0");

    ts.send({ tag: "tree", action: { tag: "undo" } }, (s) => {
      s.position = null;
    });
  });

  it("is nulled in the same reduction a human move advances the tree", async () => {
    const moveResult: StateAndView<S> = { state: 1, view: {} };
    const env: Env = {
      ...positionEnv,
      apply: <S2, V2 = unknown>() =>
        Effect.send(moveResult) as unknown as Effect<StateAndView<S2, V2>>,
    };
    const init = initialAppState<S, M>("druid", 0);
    const ts = createTestStore(appReducer<S, M>, env, init);
    await loadPosition(ts, "n0");

    ts.send({ tag: "move", action: { tag: "request", move: "a" } }, (s) => {
      s.move.status = "pending";
    });
    ts.receive(
      {
        tag: "move",
        action: {
          tag: "job",
          action: { tag: "submitted", result: { status: "done", result: moveResult } },
        },
        move: "a",
      },
      (s) => {
        s.move.status = "done";
        s.move.result = moveResult;
        const nextId = `n${s.tree.nextId}`;
        s.tree.nodes[s.tree.rootId]!.childIds.push(nextId);
        s.tree.nodes[nextId] = {
          id: nextId,
          state: moveResult.state,
          move: "a",
          search: null,
          parentId: s.tree.rootId,
          childIds: [],
        };
        s.tree.currentId = nextId;
        s.tree.nextId += 1;
        s.position = null;
      },
    );
  });

  it("is nulled in the same reduction an aiMove advances the tree", async () => {
    const result: AiMoveResult<S, M> = { move: "b", state: 1, view: {} };
    const env: Env = {
      ...positionEnv,
      aiMove: <S2, M2, V2 = unknown>() =>
        Effect.send(result) as unknown as Effect<AiMoveResult<S2, M2, V2>>,
    };
    const init = initialAppState<S, M>("druid", 0);
    const ts = createTestStore(appReducer<S, M>, env, init);
    await loadPosition(ts, "n0");

    ts.send(
      { tag: "aiMove", action: { tag: "request", strategy: { kind: "preset", id: "strong" } } },
      (s) => {
        s.aiMove.status = "pending";
      },
    );
    ts.receive(
      {
        tag: "aiMove",
        action: { tag: "job", action: { tag: "submitted", result: { status: "done", result } } },
        epoch: 0,
      },
      (s) => {
        s.aiMove.status = "done";
        s.aiMove.result = result;
        const nextId = `n${s.tree.nextId}`;
        s.tree.nodes[s.tree.rootId]!.childIds.push(nextId);
        s.tree.nodes[nextId] = {
          id: nextId,
          state: result.state,
          move: result.move,
          search: null,
          parentId: s.tree.rootId,
          childIds: [],
        };
        s.tree.currentId = nextId;
        s.tree.nextId += 1;
        s.position = null;
      },
    );
  });
});

describe("appReducer / newGame", () => {
  it("stores the request's config alongside the fresh tree, for save/load", () => {
    const result: StateAndView<S> = { state: 9, view: {} };
    const env: Env = {
      ...mockEnv,
      newGame: <S2, V2 = unknown>() =>
        Effect.send(result) as unknown as Effect<StateAndView<S2, V2>>,
    };
    const init = initialAppState<S, M>("druid", 0);
    gameTreeReducer(
      init.tree,
      { tag: "applyMove", move: "old-ai", state: 1, search: searchReport("old-ai") },
      undefined,
    );
    const ts = createTestStore(appReducer<S, M>, env, init);

    ts.send(
      { tag: "newGame", action: { tag: "request", config: { size: { w: 7, h: 7 } } } },
      (s) => {
        s.newGame.status = "pending";
      },
    );
    ts.receive(
      {
        tag: "newGame",
        action: { tag: "job", action: { tag: "submitted", result: { status: "done", result } } },
        config: { size: { w: 7, h: 7 } },
      },
      (s) => {
        s.tree = initialAppState<S, M>("druid", 9).tree;
        s.config = { size: { w: 7, h: 7 } };
        s.epoch = 1;
        // Folded into tree/config/epoch in the same reduction that observed
        // "done" -- draft.newGame itself resets back to idle, unlike
        // aiMove/analysis which stay "done" (see reducer.ts).
        s.newGame = initialAppState<S, M>("druid", 0).newGame;
      },
    );
  });
});

describe("appReducer / load", () => {
  it("rehydrates gameKind/config/tree and resets in-flight job-poll state, bumping epoch", () => {
    const init = initialAppState<S, M>("druid", 0);
    const ts = createTestStore(appReducer<S, M>, mockEnv, init);

    const loadedTree = initialGameTree<S, M>(5);
    gameTreeReducer(loadedTree, { tag: "applyMove", move: "z", state: 6 }, undefined);

    ts.send({ tag: "load", gameKind: "ttt", config: { n: 3 }, tree: loadedTree }, (s) => {
      s.gameKind = "ttt";
      s.config = { n: 3 };
      s.tree = loadedTree;
      s.epoch = 1;
    });
  });
});

describe("appReducer / switchGame", () => {
  it("changes gameKind, resets per-kind slices, and drops epoch to 0 until newGame bumps it back up", () => {
    const init = initialAppState<S, M>("druid", 0);
    init.epoch = 2;
    init.seats = { Black: { kind: "preset", id: "strong" } };
    init.aiPresets.status = "done";
    init.aiPresets.result = [];
    init.analysis.status = "done";
    init.analysis.result = {
      actions: [],
      principal_variation: [],
      total_visits: 0,
      suggested_move: null,
    };
    init.position = { nodeId: init.tree.rootId, view: {}, legalMoves: [] };
    const ts = createTestStore(appReducer<S, M>, mockEnv, init);

    ts.send({ tag: "switchGame", gameKind: "ttt" }, (s) => {
      s.gameKind = "ttt";
      // position/aiPresets/analysis (and move/aiMove/newGame, already idle
      // here) all reset back to idle -- a different kind's dialog needs its
      // own aiPresets fetched fresh and shouldn't render the outgoing kind's
      // position/analysis, which the new kind's S/M/V can't interpret.
      s.position = null;
      s.aiPresets = initialAppState<S, M>("ttt", 0).aiPresets;
      s.analysis = initialAppState<S, M>("ttt", 0).analysis;
      // Seats are per-kind (player ids differ, e.g. Druid's Black/White vs.
      // ttt's X/O) -- carrying them over would misassign control of a seat
      // id the new kind doesn't have.
      s.seats = {};
      // epoch drops to 0 (not just left at its prior value): `tree` still
      // holds the outgoing kind's nodes here (only `newGame`'s own "done"
      // branch replaces it), so `GameShell`'s `position/request` effect --
      // which re-fires on every store update, not just when `currentId`
      // actually changes -- must stay suppressed (`epoch < 1`) until
      // `newGame` completes and bumps epoch back up in the same reduction
      // that finally makes `tree` match `gameKind` again.
      s.epoch = 0;
    });
  });
});

describe("appReducer / setPreset", () => {
  it("stores the selection in ui.selectedPreset", () => {
    const init = initialAppState<S, M>("druid", 0);
    const ts = createTestStore(appReducer<S, M>, mockEnv, init);
    ts.send({ tag: "setPreset", preset: "master" }, (s) => {
      s.ui.selectedPreset = "master";
    });
  });
});
