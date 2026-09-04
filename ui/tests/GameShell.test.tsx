// tests/GameShell.test.tsx — Component-level regression tests for two bugs
// only visible at the GameShell effect layer, not in appReducer's own
// contract (packages/game/tests/reducer.test.ts already covers that layer):
//
//   1. The board renderer (DruidRenderer in production) used to unmount and
//      remount on every single move, because `state().position` legitimately
//      goes `null` for one reduction after *every* move/nav while GameShell
//      re-fetches it (see reducer.ts), and the renderer was gated directly
//      on `<Show when={position()}>`. For DruidRenderer that meant a fresh
//      three.js scene/camera/OrbitControls each time -- a visible flash/tear
//      and the camera snapping back to its default framing after every AI
//      move. Fixed by `heldPosition`, a signal that only clears on a real
//      `epoch` change (see GameShell.tsx).
//
//   2 & 3. GameShell's autoplay effect (fire an aiMove whenever it's an
//      AI-controlled seat's turn) used to fire unconditionally, with no
//      check for whether `tree.currentId` was actually the live frontier of
//      play. Navigating back into history (undo/redo/jumpTo -- e.g. the
//      ArrowLeft hotkey, or clicking an earlier move in the history panel)
//      to a node whose mover happened to be AI-controlled immediately
//      re-triggered an aiMove *from that node*, which replayed the existing
//      child and snapped straight back to (or forked past) wherever the
//      user had just navigated to. Fixed by gating the effect on
//      `isFrontier(tree)` (game-tree.ts).
//
// Both are exercised here through the real `appReducer`/`createStore` (no
// TestStore stubbing) against a mocked `Env` -- a fake, rule-free game
// module stands in for Druid/tic-tac-toe (see tests/fixtures/fake-game.tsx)
// so this never needs a real renderer or a live server, only
// `@solidjs/testing-library` + happy-dom.

import { beforeEach, describe, expect, it, vi } from "vitest";
import { fireEvent, render, screen, within } from "@solidjs/testing-library";
import { Effect } from "@mcts/core";
import type {
  AiMoveResult,
  Analysis,
  Env,
  LegalMovesResult,
  SearchReport,
  StateAndView,
} from "@mcts/game";
import {
  createTestStore,
  fixtureAxisSchema,
  mockEnv,
  mockFetchStrategyAlgorithms,
  mockFetchStrategySchema,
} from "./helpers.js";
import {
  type FakeView,
  mountLog,
  resetMountLog,
  TERMINAL_AT,
  viewFor,
} from "./fixtures/fake-game.js";

vi.mock("../app/src/games.js", () => import("./fixtures/fake-games-registry.js"));

const { GameShell } = await import("../app/src/GameShell.js");

function makeFakeEnv(): Env {
  return {
    ...mockEnv,
    newGame: <S2, V2 = unknown>() =>
      Effect.send({ state: 0, view: viewFor(0) }) as unknown as Effect<StateAndView<S2, V2>>,
    legalMoves: <S2, M2>(_kind: string, state: S2) =>
      Effect.send({
        moves: (state as unknown as number) >= TERMINAL_AT ? [] : ["inc"],
      }) as unknown as Effect<LegalMovesResult<M2>>,
    view: <S2, V2 = unknown>(_kind: string, state: S2) =>
      Effect.send(viewFor(state as unknown as number)) as unknown as Effect<V2>,
    apply: <S2, M2, V2 = unknown>(_kind: string, state: S2) => {
      const next = (state as unknown as number) + 1;
      return Effect.send({ state: next, view: viewFor(next) }) as unknown as Effect<
        StateAndView<S2, V2>
      >;
    },
    aiPresets: () => Effect.send([]),
    aiMove: <S2, M2, V2 = unknown>(_kind: string, state: S2) => {
      const cur = state as unknown as number;
      // If the frontier guard regresses, autoplay could fire from a
      // terminal (or otherwise past-the-end) position -- fail loudly rather
      // than silently returning nonsense.
      if (cur >= TERMINAL_AT)
        throw new Error("aiMove called at/after TERMINAL_AT -- the frontier guard regressed");
      const next = cur + 1;
      return Effect.send({ move: "inc", state: next, view: viewFor(next) }) as unknown as Effect<
        AiMoveResult<S2, M2, V2>
      >;
    },
  };
}

function searchReport(
  iterations: number,
  status: SearchReport<string>["status"] = "available",
): SearchReport<string> {
  return {
    status,
    schema_version: 1,
    reason: status === "unavailable" ? "strategy_unsupported" : null,
    elapsed_seconds: 0.01,
    iteration_limit: iterations,
    time_limit_seconds: null,
    completed_iterations: iterations,
    termination: "iterations",
    selected_action: "inc",
    actions: [{ action: "inc", visits: iterations, share: 1, mean_value: 0.5, is_proven: false }],
    principal_variation: ["inc"],
    root_visits: iterations,
    tree_nodes: iterations,
    mean_depth: 1,
    max_depth: 1,
    graph_mode: "tree",
    tt_reads: 0,
    tt_writes: 0,
    tt_hits: 0,
    tt_hit_ratio: 0,
    iterations_per_second: iterations * 100,
    warnings: status === "partial" ? ["actions_truncated"] : [],
  };
}

function analysisResult(search?: SearchReport<string> | null): Analysis<string> {
  return {
    actions: [{ action: "inc", visits: 12, mean_value: 0.5, is_proven: false }],
    principal_variation: ["inc"],
    total_visits: 12,
    suggested_move: "inc",
    ...(search === undefined ? {} : { search }),
  };
}

function makeInspectorEnv(
  options: { analysis?: Analysis<string>; aiSearch?: SearchReport<string> | null } = {},
): Env {
  return {
    ...makeFakeEnv(),
    aiPresets: () => Effect.send([{ id: "strong", label: "Strong", description: "test" }]),
    analyze: <M2,>() =>
      Effect.send(options.analysis ?? analysisResult(searchReport(17))) as unknown as Effect<
        Analysis<M2>
      >,
    aiMove: <S2, M2, V2 = unknown>(_kind: string, state: S2) => {
      const current = state as unknown as number;
      const result: AiMoveResult<number, string, FakeView> = {
        move: "inc",
        state: current + 1,
        view: viewFor(current + 1),
        ...(options.aiSearch === undefined
          ? { search: searchReport(5) }
          : { search: options.aiSearch }),
      };
      return Effect.send(result) as unknown as Effect<AiMoveResult<S2, M2, V2>>;
    },
  };
}

/** Both seats AI-controlled -- lets autoplay alone drive the game to
 * TERMINAL_AT, deterministically, one linear branch (n0..n6, no forks). */
function makeBothSeatsAi(store: ReturnType<typeof createTestStore>["store"]): void {
  store.dispatch({ tag: "setSeat", player: "A", control: { kind: "preset", id: "ai" } });
  store.dispatch({ tag: "setSeat", player: "B", control: { kind: "preset", id: "ai" } });
}

/** Polls `getValue()` for `ms` of real wall-clock time, asserting it equals
 * `expected` on every poll -- fails at the first tick it drifts, rather than
 * just checking the final value (which the old autoplay bug could, after a
 * few ticks, settle back to something indistinguishable from "never
 * navigated at all" -- see GameShell.test.tsx's header comment). */
async function holdsSteadyAt(getValue: () => unknown, expected: unknown, ms = 300): Promise<void> {
  const start = Date.now();
  while (Date.now() - start < ms) {
    expect(getValue()).toBe(expected);
    await new Promise((resolve) => setTimeout(resolve, 20));
  }
}

beforeEach(() => {
  resetMountLog();
});

describe("GameShell autoplay/history bugs (fake game, no real server)", () => {
  it("never remounts the board renderer across autoplay moves or history navigation", async () => {
    const { store } = createTestStore("fake", makeFakeEnv());
    render(() => (
      <GameShell
        store={store}
        fetchStrategySchema={mockFetchStrategySchema}
        fetchStrategyAlgorithms={mockFetchStrategyAlgorithms}
      />
    ));

    await vi.waitFor(() => expect(screen.getByTestId("fake-board")).toBeInTheDocument());
    expect(mountLog).toEqual(["mount"]);

    makeBothSeatsAi(store);
    await vi.waitFor(() => expect(store.state.tree.nextId).toBe(TERMINAL_AT + 1), {
      timeout: 5000,
    });
    await vi.waitFor(() => expect(store.state.position?.nodeId).toBe(store.state.tree.currentId));
    expect(mountLog).toEqual(["mount"]);

    store.dispatch({ tag: "tree", action: { tag: "undo" } });
    await vi.waitFor(() => expect(store.state.tree.currentId).toBe("n5"));

    store.dispatch({ tag: "tree", action: { tag: "jumpTo", id: "n0" } });
    await vi.waitFor(() => expect(store.state.tree.currentId).toBe("n0"));

    // Every move above (6 of them) plus two history jumps each put
    // `position` through the transient-null gap the old code unmounted the
    // renderer for -- it should still be exactly the one mount from initial
    // load.
    expect(mountLog).toEqual(["mount"]);
  });

  it("undo (ArrowLeft) moves back one ply and holds -- it used to immediately snap forward again", async () => {
    const { store } = createTestStore("fake", makeFakeEnv());
    render(() => (
      <GameShell
        store={store}
        fetchStrategySchema={mockFetchStrategySchema}
        fetchStrategyAlgorithms={mockFetchStrategyAlgorithms}
      />
    ));
    await vi.waitFor(() => expect(screen.getByTestId("fake-board")).toBeInTheDocument());

    makeBothSeatsAi(store);
    await vi.waitFor(() => expect(store.state.tree.currentId).toBe("n6"), { timeout: 5000 });

    window.dispatchEvent(new KeyboardEvent("keydown", { key: "ArrowLeft", bubbles: true }));

    await vi.waitFor(() => expect(store.state.tree.currentId).toBe("n5"));
    // n5's mover ("B", AI-controlled) is a leaf-adjacent AI turn -- exactly
    // the case that used to re-fire an aiMove from n5 and snap straight
    // back to n6, one reactive tick later.
    await holdsSteadyAt(() => store.state.tree.currentId, "n5");
  });

  it("clicking a move in the history panel jumps there and holds -- it used to go nowhere the user could see", async () => {
    const { store } = createTestStore("fake", makeFakeEnv());
    render(() => (
      <GameShell
        store={store}
        fetchStrategySchema={mockFetchStrategySchema}
        fetchStrategyAlgorithms={mockFetchStrategyAlgorithms}
      />
    ));
    await vi.waitFor(() => expect(screen.getByTestId("fake-board")).toBeInTheDocument());

    makeBothSeatsAi(store);
    await vi.waitFor(() => expect(store.state.tree.currentId).toBe("n6"), { timeout: 5000 });

    const startRow = screen.getByText("Start").closest("button");
    expect(startRow).not.toBeNull();
    startRow!.click();

    await vi.waitFor(() => expect(store.state.tree.currentId).toBe("n0"));
    // Root's mover ("A", AI-controlled) used to cascade autoplay all the
    // way back to n6 within a couple of reactive ticks, making the click
    // look like it did nothing.
    await holdsSteadyAt(() => store.state.tree.currentId, "n0");
  });
});

describe("GameShell autoplay: a failing aiMove must not retry forever (fake game, no real server)", () => {
  it("attempts an AI-controlled seat's move exactly once and surfaces the error, instead of retrying in an unbounded loop", async () => {
    let aiMoveCalls = 0;
    const env: Env = {
      ...makeFakeEnv(),
      aiMove: () => {
        aiMoveCalls++;
        return Effect.fromPromise(() => Promise.reject(new Error("subprocess crashed")));
      },
    };
    const { store } = createTestStore("fake", env);
    render(() => (
      <GameShell
        store={store}
        fetchStrategySchema={mockFetchStrategySchema}
        fetchStrategyAlgorithms={mockFetchStrategyAlgorithms}
      />
    ));
    await vi.waitFor(() => expect(screen.getByTestId("fake-board")).toBeInTheDocument());

    makeBothSeatsAi(store);
    await vi.waitFor(() => expect(store.state.aiMove.status).toBe("error"));
    await vi.waitFor(() => expect(screen.getByText(/AI move failed/)).toBeInTheDocument());

    // The old autoplay effect re-fired on every store update with no
    // backoff -- an "error" status clears `busy()`, which let the same
    // doomed request fire again immediately, forever. Holding steady here
    // (not just checking the count once) is what actually catches a
    // regression back to that unbounded loop, the same reasoning
    // `holdsSteadyAt` documents above.
    await holdsSteadyAt(() => aiMoveCalls, 1);
  });
});

describe("GameShell New Game dialog: 'Custom…' seat option (fake game, no real server)", () => {
  it("builds an AiStrategyRef from the schema-driven editor and dispatches it as that seat's control", async () => {
    const { store, captured } = createTestStore("fake", makeFakeEnv());
    render(() => (
      <GameShell
        store={store}
        fetchStrategySchema={mockFetchStrategySchema}
        fetchStrategyAlgorithms={mockFetchStrategyAlgorithms}
      />
    ));
    await vi.waitFor(() => expect(screen.getByTestId("fake-board")).toBeInTheDocument());

    fireEvent.click(screen.getByText("New Game"));
    const seatASelect = await vi.waitFor(() => {
      const el = screen.getByLabelText("A") as HTMLSelectElement;
      expect(el.querySelector('option[value="custom"]')).not.toBeDisabled();
      return el;
    });
    fireEvent.change(seatASelect, { target: { value: "custom" } });

    // Selecting "custom" seeds the seat's editor from the schema's own
    // first-listed variant per axis (`defaultCustomStrategySpec`) -- "ucb1"
    // in this fixture (see helpers.ts's `fixtureAxisSchema`).
    const selectAxis = await screen.findByLabelText("Select");
    expect((selectAxis as HTMLSelectElement).value).toBe(
      fixtureAxisSchema.select.variants[0]!.kind,
    );

    // Switching to the wrapper variant reveals the nested `select_base`
    // picker -- the one real level of recursion `config_ir.rs` allows.
    fireEvent.change(selectAxis, { target: { value: "epsilon_greedy" } });
    const innerAxis = await screen.findByLabelText("wraps");
    expect((innerAxis as HTMLSelectElement).value).toBe(
      fixtureAxisSchema.select_base.variants[0]!.kind,
    );

    fireEvent.click(document.getElementById("new-game-start")!);

    const setSeatA = captured.findLast(
      (a) => a.tag === "setSeat" && "player" in a && a.player === "A",
    ) as { tag: "setSeat"; player: string; control: unknown } | undefined;
    expect(setSeatA?.control).toMatchObject({
      kind: "custom",
      spec: { search: { select: { kind: "epsilon_greedy", inner: { kind: "ucb1" } } } },
    });
  });
});

describe("GameShell live search inspection (fake game, no real server)", () => {
  it("keeps a completed explicit analysis distinct from the retained AI report", async () => {
    const { store } = createTestStore("fake", makeInspectorEnv());
    render(() => (
      <GameShell
        store={store}
        fetchStrategySchema={mockFetchStrategySchema}
        fetchStrategyAlgorithms={mockFetchStrategyAlgorithms}
      />
    ));
    await vi.waitFor(() => expect(screen.getByTestId("fake-board")).toBeInTheDocument());

    fireEvent.click(screen.getByRole("button", { name: "AI Move" }));
    await vi.waitFor(() => expect(store.state.tree.currentId).toBe("n1"));
    await vi.waitFor(() =>
      expect(
        within(document.getElementById("move-search-panel")!).getByText("Completed iterations"),
      ).toBeInTheDocument(),
    );
    expect(
      within(document.getElementById("move-search-panel")!).getAllByText("5").length,
    ).toBeGreaterThan(0);
    expect(
      within(document.getElementById("move-search-panel")!).getAllByText("inc#0").length,
    ).toBeGreaterThan(0);

    fireEvent.click(screen.getByRole("button", { name: "Analyze" }));
    await vi.waitFor(() =>
      expect(
        within(document.getElementById("analysis-panel")!).getByText("Completed iterations"),
      ).toBeInTheDocument(),
    );
    expect(
      within(document.getElementById("analysis-panel")!).getAllByText("17").length,
    ).toBeGreaterThan(0);
    expect(
      within(document.getElementById("move-search-panel")!).getAllByText("5").length,
    ).toBeGreaterThan(0);
    const summaryIds = Array.from(document.querySelectorAll("h3"))
      .filter((heading) => heading.textContent === "Search summary")
      .map((heading) => heading.id);
    expect(new Set(summaryIds).size).toBe(2);
  });

  it("keeps legacy analysis actions and PV under a reduced-capability label", async () => {
    const { store } = createTestStore("fake", makeInspectorEnv({ analysis: analysisResult() }));
    render(() => (
      <GameShell
        store={store}
        fetchStrategySchema={mockFetchStrategySchema}
        fetchStrategyAlgorithms={mockFetchStrategyAlgorithms}
      />
    ));
    await vi.waitFor(() => expect(screen.getByTestId("fake-board")).toBeInTheDocument());

    fireEvent.click(screen.getByRole("button", { name: "Analyze" }));
    await vi.waitFor(() =>
      expect(
        screen.getByRole("heading", { name: "Legacy analysis (reduced capability)" }),
      ).toBeInTheDocument(),
    );
    const panel = document.getElementById("analysis-panel")!;
    expect(within(panel).getByText("inc#0")).toBeInTheDocument();
    expect(within(panel).queryByText("Search summary")).toBeNull();
  });

  it("shows that a human move has no retained search report", async () => {
    const { store } = createTestStore("fake", makeInspectorEnv());
    render(() => (
      <GameShell
        store={store}
        fetchStrategySchema={mockFetchStrategySchema}
        fetchStrategyAlgorithms={mockFetchStrategyAlgorithms}
      />
    ));
    await vi.waitFor(() => expect(screen.getByTestId("fake-board")).toBeInTheDocument());

    store.dispatch({ tag: "move", action: { tag: "request", move: "inc" } });
    await vi.waitFor(() => expect(store.state.tree.currentId).toBe("n1"));
    const panel = document.getElementById("move-search-panel")!;
    expect(within(panel).getByText("Search that selected this move")).toBeInTheDocument();
    expect(within(panel).getByRole("status")).toHaveTextContent("played by a human");
  });

  it("changes the retained report with undo, redo, and history selection without analyzing", async () => {
    let analyzeCalls = 0;
    const env: Env = {
      ...makeInspectorEnv(),
      analyze: () => {
        analyzeCalls++;
        return Effect.none();
      },
    };
    const { store } = createTestStore("fake", env);
    render(() => (
      <GameShell
        store={store}
        fetchStrategySchema={mockFetchStrategySchema}
        fetchStrategyAlgorithms={mockFetchStrategyAlgorithms}
      />
    ));
    await vi.waitFor(() => expect(screen.getByTestId("fake-board")).toBeInTheDocument());

    store.dispatch({ tag: "move", action: { tag: "request", move: "inc" } });
    await vi.waitFor(() => expect(store.state.tree.currentId).toBe("n1"));
    fireEvent.click(screen.getByRole("button", { name: "AI Move" }));
    await vi.waitFor(() => expect(store.state.tree.currentId).toBe("n2"));
    expect(
      within(document.getElementById("move-search-panel")!).getAllByText("5").length,
    ).toBeGreaterThan(0);

    window.dispatchEvent(new KeyboardEvent("keydown", { key: "ArrowLeft", bubbles: true }));
    await vi.waitFor(() => expect(store.state.tree.currentId).toBe("n1"));
    expect(
      within(document.getElementById("move-search-panel")!).getByRole("status"),
    ).toHaveTextContent("played by a human");

    window.dispatchEvent(new KeyboardEvent("keydown", { key: "ArrowRight", bubbles: true }));
    await vi.waitFor(() => expect(store.state.tree.currentId).toBe("n2"));
    expect(
      within(document.getElementById("move-search-panel")!).getAllByText("5").length,
    ).toBeGreaterThan(0);

    fireEvent.click(screen.getByText("Start").closest("button")!);
    await vi.waitFor(() => expect(store.state.tree.currentId).toBe("n0"));
    expect(
      within(document.getElementById("move-search-panel")!).getByRole("status"),
    ).toHaveTextContent("starting position");
    expect(analyzeCalls).toBe(0);
  });

  it("renders unavailable and partial explicit reports without falling back to legacy output", async () => {
    const partial = createTestStore(
      "fake",
      makeInspectorEnv({ analysis: analysisResult(searchReport(3, "partial")) }),
    );
    const rendered = render(() => (
      <GameShell
        store={partial.store}
        fetchStrategySchema={mockFetchStrategySchema}
        fetchStrategyAlgorithms={mockFetchStrategyAlgorithms}
      />
    ));
    await vi.waitFor(() => expect(screen.getByTestId("fake-board")).toBeInTheDocument());
    fireEvent.click(screen.getByRole("button", { name: "Analyze" }));
    await vi.waitFor(() =>
      expect(
        within(document.getElementById("analysis-panel")!).getByRole("status"),
      ).toHaveTextContent("evidence is partial"),
    );
    expect(
      screen.getByText("The action list was truncated before every root action could be retained."),
    ).toBeInTheDocument();
    rendered.unmount();

    const unavailable = createTestStore(
      "fake",
      makeInspectorEnv({ analysis: analysisResult(searchReport(3, "unavailable")) }),
    );
    render(() => (
      <GameShell
        store={unavailable.store}
        fetchStrategySchema={mockFetchStrategySchema}
        fetchStrategyAlgorithms={mockFetchStrategyAlgorithms}
      />
    ));
    await vi.waitFor(() => expect(screen.getByTestId("fake-board")).toBeInTheDocument());
    fireEvent.click(screen.getByRole("button", { name: "Analyze" }));
    await vi.waitFor(() =>
      expect(
        within(document.getElementById("analysis-panel")!).getByRole("status"),
      ).toHaveTextContent("evidence unavailable"),
    );
    expect(
      screen.queryByRole("heading", { name: "Legacy analysis (reduced capability)" }),
    ).toBeNull();
  });

  it("drops an analysis response that completes after a new game", async () => {
    let resolveAnalysis: ((result: Analysis<string>) => void) | undefined;
    const env: Env = {
      ...makeInspectorEnv(),
      analyze: <M2,>() =>
        Effect.fromPromise(
          () =>
            new Promise<Analysis<string>>((resolve) => {
              resolveAnalysis = resolve;
            }),
        ) as unknown as Effect<Analysis<M2>>,
    };
    const { store } = createTestStore("fake", env);
    render(() => (
      <GameShell
        store={store}
        fetchStrategySchema={mockFetchStrategySchema}
        fetchStrategyAlgorithms={mockFetchStrategyAlgorithms}
      />
    ));
    await vi.waitFor(() => expect(screen.getByTestId("fake-board")).toBeInTheDocument());

    fireEvent.click(screen.getByRole("button", { name: "Analyze" }));
    await vi.waitFor(() => expect(store.state.analysis.status).toBe("pending"));
    fireEvent.click(screen.getByRole("button", { name: "New Game" }));
    fireEvent.click(document.getElementById("new-game-start")!);
    await vi.waitFor(() => expect(store.state.epoch).toBe(2));
    resolveAnalysis!(analysisResult(searchReport(99)));
    await vi.waitFor(() => expect(store.state.analysis.status).toBe("idle"));
    expect(document.getElementById("analysis-panel")?.textContent).not.toContain("99");
  });
});
