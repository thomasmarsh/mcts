// GameShell.tsx — Game-kind-agnostic chrome: HUD (turn indicator, hand
// counts via a per-game summary, mode buttons), New Game dialog, AI-move/
// autoplay controls, and the renderer registry lookup. Ported from
// server/static/app.js's DOM-manipulation HUD logic, now
// driven by `store.dispatch`/reactive effects instead of direct DOM writes
// and global mutable state — and generalized to work for any
// `GAME_MODULES` entry, not just Druid.
//
// Game modules are loaded lazily (see `games.ts`) — `createResource` drives
// the async fetch, and the existing `<Show when={mod()}>` fallback doubles as
// the loading indicator.
//
// Per the hard rule, this component never touches the network
// itself: every effect below only ever calls `props.store.dispatch(...)`.

import { type Component, createEffect, createMemo, createResource, createSignal, For, lazy, onCleanup, onMount, Show } from "solid-js";
import { Dynamic } from "solid-js/web";
import type { Store } from "@mcts/core";
import type { AiStrategyRef, AnalysisOverlayEntry, AppAction, AppState, AxisSchema, GameTreeNode, MoveStep } from "@mcts/game";
import { isFrontier, moveEquals } from "@mcts/game";
import { defaultCustomStrategySpec, StrategyConfigEditor } from "@mcts/strategy-config";
import { GAME_META, GAME_MODULES } from "./games.js";

// Panels are lazy-loaded so they only pull in their own dependencies when
// the user actually starts a game (`state().epoch >= 1`).
const MoveListPanel = lazy(() => import("./MoveListPanel.js").then((m) => ({ default: m.MoveListPanel })));
const AnalysisPanel = lazy(() => import("./AnalysisPanel.js").then((m) => ({ default: m.AnalysisPanel })));
const SaveLoadPanel = lazy(() => import("./SaveLoadPanel.js").then((m) => ({ default: m.SaveLoadPanel })));

type S = unknown;
type M = unknown;
type V = unknown;

/** Wraps a bare preset id as an `AiStrategyRef` -- the New Game dialog's
 * seat pickers only ever choose a named preset, never build an `AiStrategyRef`
 * directly, so this is the one place that boundary is crossed.
 * TODO: a "Custom…" option would need to build other `AiStrategyRef` kinds
 * here too. */
function presetStrategy(id: string): AiStrategyRef {
  return { kind: "preset", id };
}

/** Walks `tree`'s root-to-current path into the `MoveStep[]` shape
 * `GameRendererProps.history` expects — the root itself (whose `move` is
 * always `null`) never appears as a step, only as the first step's
 * `before`. */
function historyPath(tree: AppState<S, M, V>["tree"]): MoveStep<S, M>[] {
  const chain: GameTreeNode<S, M>[] = [];
  let node: GameTreeNode<S, M> | undefined = tree.nodes[tree.currentId];
  while (node) {
    chain.push(node);
    node = node.parentId ? tree.nodes[node.parentId] : undefined;
  }
  chain.reverse();
  const steps: MoveStep<S, M>[] = [];
  for (let i = 1; i < chain.length; i++) {
    const n = chain[i]!;
    const before = chain[i - 1]!;
    steps.push({ move: n.move as M, before: before.state });
  }
  return steps;
}

export const GameShell: Component<{
  store: Store<AppState<S, M, V>, AppAction<S, M, V>>;
  fetchStrategySchema: () => Promise<AxisSchema>;
}> = (props) => {
  const state = props.store.getState();
  const dispatch = props.store.dispatch;

  // Fetched once, game-independent (`config_ir`'s shape doesn't depend on
  // which game kind is selected) -- fed to every seat's "Custom…" editor
  // below, not re-fetched per seat or per game-kind switch. Injected as a
  // prop rather than read via `env`/dispatch because it's a one-shot static
  // read with no job-poll/error-retry semantics worth the `AppState` slice
  // `aiPresets` has; `App.tsx` fetches `GET /api/games` the same way.
  const [schema] = createResource(props.fetchStrategySchema);

  // Asynchronously load the game-kind module. The resource's source tracks
  // `state().gameKind`, so switching kinds (via the new-game dialog) triggers
  // a re-fetch. `modData()` returns `undefined` while loading, which the
  // `<Show when={mod()}>` wrapper displays as the "Unknown game." fallback.
  const [modData] = createResource(
    () => state().gameKind,
    async (kind: string) => {
      const load = GAME_MODULES[kind];
      if (!load) throw new Error(`Unknown game kind: ${kind}`);
      return load();
    },
  );
  const mod = () => modData();

  const position = createMemo(() => state().position);
  const summary = createMemo(() => {
    const p = position();
    const m = mod();
    return p && m ? m.summarize(p.view) : null;
  });

  const busy = createMemo(
    () =>
      state().move.status === "pending" ||
      state().aiMove.status === "pending" ||
      state().analysis.status === "pending" ||
      state().newGame.status === "pending",
  );

  const [activeMode, setActiveMode] = createSignal<string | null>(null);
  const [hoveredMove, setHoveredMove] = createSignal<M | null>(null);
  const [autoplayPaused, setAutoplayPaused] = createSignal(false);
  const [dialogOpen, setDialogOpen] = createSignal(false);
  const [pendingConfig, setPendingConfig] = createSignal<unknown>(undefined);
  const [pendingSeats, setPendingSeats] = createSignal<Record<string, "human" | AiStrategyRef>>({});

  // `state().position` goes `null` for one reduction after *every* move/nav
  // (reducer.ts nulls it to preserve the "position matches currentId"
  // invariant, then GameShell's own position/request effect below re-fetches
  // it) — not just when a new game starts. Gating the renderer directly on
  // `position()` therefore unmounted/remounted `Dynamic`'s `GameRenderer` on
  // *every* move: DruidRenderer's `onMount` rebuilds its three.js scene,
  // camera, and OrbitControls from scratch each time, which read as a
  // flash/tear and a snapped-back camera after every AI move. `heldPosition`
  // keeps the last-known position around across that brief gap so the
  // renderer stays mounted continuously; it's only cleared back to `null` on
  // a genuine new game (an `epoch` bump), which is when the fallback should
  // actually reappear.
  const [heldPosition, setHeldPosition] = createSignal<AppState<S, M, V>["position"]>(null);
  let lastEpoch = state().epoch;
  createEffect(() => {
    const epoch = state().epoch;
    if (epoch !== lastEpoch) {
      lastEpoch = epoch;
      setHeldPosition(null);
    }
    const p = position();
    if (p) setHeldPosition(p);
  });

  const legalMoves = createMemo(() => {
    const p = position();
    if (!p) return [];
    const modeDef = mod()?.modes?.find((md) => md.id === activeMode());
    return modeDef ? p.legalMoves.filter(modeDef.filter) : p.legalMoves;
  });

  // Default to the first mode whenever the current module's `modes` don't
  // include whatever `activeMode` is currently set to -- both the initial
  // `null` case and a kind switch where the outgoing kind's mode id (e.g.
  // Druid's "sarsen") doesn't exist in the incoming kind's own `modes` (e.g.
  // Tak's "flat"/"wall"/"cap"/"move"). Re-deriving on every `mod()` change
  // (not just once, guarded on `activeMode() === null`) is what makes this
  // self-healing across kind switches: with a once-only guard, switching
  // from Druid to Tak left `activeMode` stuck on Druid's mode id, which
  // matched none of Tak's -- `legalMoves` below then fell through to its
  // unfiltered branch, silently handing `TakRenderer` a mix of Place *and*
  // Spread moves that its own placement/spread-mode split can't represent
  // (see that component's `isSpreadMode` check), making placement moves
  // vanish entirely as soon as any spread became legal.
  createEffect(() => {
    const modes = mod()?.modes;
    if (!modes || modes.length === 0) return;
    if (!modes.some((md) => md.id === activeMode())) setActiveMode(modes[0]!.id);
  });

  // Bootstrap: fetch this kind's AI presets once, and start the very first
  // game with the server's own default config (an empty `newGame` request —
  // see server/main.rs's `post_new`, which fills in `adapter.default_config()`
  // when `config` is omitted).
  onMount(() => {
    dispatch({ tag: "aiPresets", action: { tag: "request" } });
    dispatch({ tag: "newGame", action: { tag: "request" } });
  });

  // Re-derive view/legalMoves for whatever node is current, on every
  // navigation. Gated on `epoch >= 1` so this never fires against the
  // pre-bootstrap placeholder root (see state.ts's `initialAppState` —
  // `epoch` only advances once a real `newGame` has completed).
  //
  // `state()` (the store's `useSnapshot` signal) is coarse: it changes on
  // *every* mutation anywhere in `AppState`, not just `tree.currentId`, so
  // this effect body reruns on every store update regardless of the `void
  // s.tree.currentId` read below — that read alone doesn't scope the
  // dependency (see `reducer.ts`'s `switchGame` comment, which already
  // documents this). Without the `lastPositionKey` guard, the `position/
  // request` dispatch below is itself a store update, which reruns this
  // same effect, which dispatches again — a self-sustaining loop that a
  // real network round trip happens to rate-limit, but which spins as fast
  // as the event loop allows against a synchronously-resolving `Env` (e.g.
  // a mocked one in a test), consuming memory until the process OOMs.
  let lastPositionKey = "";
  createEffect(() => {
    const s = state();
    const key = `${s.tree.currentId}:${s.epoch}`;
    if (key === lastPositionKey) return;
    lastPositionKey = key;
    setHoveredMove(null);
    if (s.epoch < 1) return;
    dispatch({ tag: "position", action: { tag: "request" } });
  });

  // Auto-play: if the position isn't terminal and the player to move is
  // AI-controlled, fire an aiMove request — mirrors app.js's
  // `maybeTriggerAiTurn`, re-checked after every position/seat/pause change.
  // Safe to trust `summary()`'s `currentPlayer` here without separately
  // checking it against `tree.currentId`: `appReducer` nulls `position`
  // (which `summary` reads through) in the same reduction as every
  // `currentId`-changing action, so a non-null `position`/`summary` is
  // always for the *current* node, by construction (see reducer.ts).
  //
  // Gated on `childIds.length === 0` (the current node being a leaf) so this
  // only ever drives the live *frontier* of the game forward — without this,
  // navigating back into history (undo/redo/jumpTo/arrow keys) to a node
  // that happens to be an AI seat's turn immediately re-triggered an aiMove
  // from there, which either replayed the same historical branch (undo
  // looking like it "snapped back" to the last move) or forked a new one
  // (a history click silently going nowhere the user could see).
  // `state().aiMoveFailedNodeId` (set by reducer.ts's `aiMove` handling)
  // is what keeps this effect from retrying a doomed request forever: a
  // failure (bad custom config, a crashing subprocess, any transport
  // error) flips `aiMove.status` to `"error"`, which clears `busy()` --
  // and since nothing about the tree changed, this effect reruns on that
  // same store update. Without the node check below, it fired the
  // identical request again immediately, with no backoff, forever -- see
  // `AppState.aiMoveFailedNodeId`'s doc comment. A fresh attempt at a
  // *different* node, or a deliberate manual retry (which dispatches a
  // fresh `request`, resetting `error` -- `jobPollReduce`'s `"start"`
  // case), both still go through normally.
  createEffect(() => {
    if (busy() || autoplayPaused()) return;
    if (!isFrontier(state().tree)) return;
    const sum = summary();
    if (!sum || sum.currentPlayer === null) return;
    const seat = state().seats[sum.currentPlayer] ?? "human";
    if (seat === "human") return;
    if (state().aiMove.status === "error" && state().aiMoveFailedNodeId === state().tree.currentId) return;
    dispatch({ tag: "aiMove", action: { tag: "request", strategy: seat } });
  });

  function onKeyDown(event: KeyboardEvent): void {
    if (busy()) return;
    const tag = (event.target as HTMLElement | null)?.tagName;
    if (tag === "SELECT" || tag === "INPUT" || tag === "TEXTAREA") return;
    if (event.key === "ArrowLeft") {
      dispatch({ tag: "tree", action: { tag: "undo" } });
      return;
    }
    if (event.key === "ArrowRight") {
      dispatch({ tag: "tree", action: { tag: "redo" } });
      return;
    }
    const hit = mod()?.modes?.find((md) => md.hotkey === event.key);
    if (hit) setActiveMode(hit.id);
  }
  onMount(() => window.addEventListener("keydown", onKeyDown));
  onCleanup(() => window.removeEventListener("keydown", onKeyDown));

  /** Selects a `<select>` option value for `pendingSeats()[player]`'s current
   * control -- "human", a preset id, or the "custom" sentinel (the actual
   * spec, if any, is rendered by the `StrategyConfigEditor` block below the
   * select, not carried in the option value itself). */
  function seatSelectValue(control: "human" | AiStrategyRef): string {
    if (control === "human") return "human";
    return control.kind === "preset" ? control.id : "custom";
  }

  function openDialog(): void {
    setPendingConfig(undefined);
    const seats: Record<string, "human" | AiStrategyRef> = {};
    for (const p of GAME_META[state().gameKind]?.players ?? []) seats[p] = state().seats[p] ?? "human";
    setPendingSeats(seats);
    setDialogOpen(true);
  }

  // Switches which kind the (still-open) New Game dialog is about to start
  // — the game-kind picker. Dispatches `switchGame` immediately
  // (rather than deferring to `startNewGame`) so the dialog's seat pickers
  // re-fetch the new kind's own `aiPresets` and its player list updates via
  // `GAME_META` while still open. `state().tree` still holds the outgoing
  // game's nodes until `newGame` (dispatched from `startNewGame` below)
  // completes and replaces it — `switchGame` drops `epoch` to 0 for that
  // whole window (see its own comment in reducer.ts) so nothing tries to
  // read that stale tree under the new `gameKind` in the meantime.
  function onGameKindChange(kind: string): void {
    if (kind === state().gameKind) return;
    dispatch({ tag: "switchGame", gameKind: kind });
    dispatch({ tag: "aiPresets", action: { tag: "request" } });
    setPendingConfig(undefined);
    const seats: Record<string, "human" | AiStrategyRef> = {};
    for (const p of GAME_META[kind]?.players ?? []) seats[p] = "human";
    setPendingSeats(seats);
  }

  function startNewGame(): void {
    for (const [player, control] of Object.entries(pendingSeats())) {
      dispatch({ tag: "setSeat", player, control });
    }
    setAutoplayPaused(false);
    dispatch({ tag: "newGame", action: { tag: "request", config: pendingConfig() } });
    setDialogOpen(false);
  }

  const manualMoveStrategy = (): AiStrategyRef => {
    const sum = summary();
    if (!sum || sum.currentPlayer === null) return presetStrategy("strong");
    const seat = state().seats[sum.currentPlayer] ?? "human";
    return seat === "human" ? presetStrategy("strong") : seat;
  };

  const presetOptions = () => (state().aiPresets.status === "done" ? (state().aiPresets.result ?? []) : []);

  // Falls back to "strong" the same way `manualMovePreset` above does, only
  // once the user hasn't picked one yet (`ui.selectedPreset`, a
  // slice of `AppState` — see state.ts).
  const analysisPreset = createMemo(() => {
    const chosen = state().ui.selectedPreset;
    if (chosen && presetOptions().some((p) => p.id === chosen)) return chosen;
    const options = presetOptions();
    return options.find((p) => p.id === "strong")?.id ?? options[0]?.id ?? "strong";
  });

  // Feeds `DruidRenderer`'s heatmap overlay — one source of truth (`analysis`
  // job-poll state) for both this and `AnalysisPanel`'s own candidate table.
  // `undefined` (not `[]`) when there's no completed analysis for the
  // *current* position — reducer.ts resets `analysis` on every tree
  // navigation/move, so a stale result never lingers past the position it
  // was computed for.
  const analysisOverlay = createMemo((): AnalysisOverlayEntry<M>[] | undefined => {
    const a = state().analysis;
    if (a.status !== "done" || !a.result) return undefined;
    const total = a.result.total_visits || 1;
    const suggested = a.result.suggested_move;
    return a.result.actions.map(
      (c): AnalysisOverlayEntry<M> => ({
        move: c.action,
        visitShare: c.visits / total,
        isProven: c.is_proven,
        isSuggested: suggested !== null && moveEquals(c.action, suggested),
      }),
    );
  });

  return (
    <>
      <Show when={mod()} fallback={<div class="loading">Loading game…</div>}>
        {(m) => (
          <>
            <Show when={heldPosition()} fallback={<div class="loading">Starting a new game…</div>}>
              {(p) => (
                <Dynamic
                  component={m().Renderer}
                  state={state().tree.nodes[state().tree.currentId]?.state}
                  view={p().view}
                  history={historyPath(state().tree)}
                  legalMoves={legalMoves()}
                  busy={busy()}
                  onMove={(move: M) => dispatch({ tag: "move", action: { tag: "request", move } })}
                  hoveredMove={hoveredMove()}
                  onHover={setHoveredMove}
                  analysisOverlay={analysisOverlay()}
                />
              )}
            </Show>

            <div id="hud">
              <div id="turn">{summary()?.turnText ?? ""}</div>
              <div id="hands">
                <For each={summary()?.lines ?? []}>
                  {(line) => (
                    <div class="hand" style={{ "--swatch": line.swatch ?? "transparent" }}>
                      {line.text}
                    </div>
                  )}
                </For>
              </div>
              <Show when={m().modes && m().modes!.length > 0}>
                <div id="modes">
                  <For each={m().modes}>
                    {(md) => (
                      <button
                        class="mode"
                        classList={{ active: activeMode() === md.id }}
                        disabled={busy()}
                        onClick={() => setActiveMode(md.id)}
                      >
                        {md.label} {md.hotkey && <span class="hotkey">{md.hotkey}</span>}
                      </button>
                    )}
                  </For>
                </div>
              </Show>
              <div id="actions">
                <button id="ai-move" disabled={busy()} onClick={() => dispatch({ tag: "aiMove", action: { tag: "request", strategy: manualMoveStrategy() } })}>
                  AI Move
                </button>
                <button id="autoplay-toggle" classList={{ paused: autoplayPaused() }} onClick={() => setAutoplayPaused((v) => !v)}>
                  {autoplayPaused() ? "Resume" : "Pause"}
                </button>
                <button id="new-game" onClick={openDialog}>
                  New Game
                </button>
              </div>
              <Show when={state().epoch >= 1}>
                <SaveLoadPanel
                  gameKind={state().gameKind}
                  config={state().config}
                  tree={state().tree}
                  onLoad={(gameKind, config, tree) => dispatch({ tag: "load", gameKind, config, tree })}
                />
              </Show>
              <div id="banner" style={{ color: summary()?.bannerColor }}>
                {summary()?.bannerText ?? ""}
              </div>
              <Show when={state().aiMove.status === "error" && state().aiMoveFailedNodeId === state().tree.currentId}>
                <div id="ai-move-error" style={{ color: "#c0392b" }}>
                  AI move failed: {state().aiMove.error}
                </div>
              </Show>
            </div>

            <Show when={state().epoch >= 1}>
              <MoveListPanel
                tree={state().tree}
                formatMove={m().formatMove}
                onJump={(id) => dispatch({ tag: "tree", action: { tag: "jumpTo", id } })}
              />
              <AnalysisPanel
                analysis={state().analysis}
                presets={presetOptions()}
                selectedPreset={analysisPreset()}
                before={state().tree.nodes[state().tree.currentId]?.state}
                formatMove={m().formatMove}
                busy={busy()}
                hoveredMove={hoveredMove()}
                onSelectPreset={(preset) => dispatch({ tag: "setPreset", preset })}
                onAnalyze={() => dispatch({ tag: "analysis", action: { tag: "request", strategy: presetStrategy(analysisPreset()) } })}
                onHoverMove={setHoveredMove}
              />
            </Show>
          </>
        )}
      </Show>

      {/* Dialog is outside the mod() Show wrapper so it stays open across
          game-kind switches even while the new module is still loading. */}
      <Show when={dialogOpen()}>
        <dialog
          id="new-game-dialog"
          ref={(el) => queueMicrotask(() => el.showModal())}
        >
          <form
            id="new-game-form"
            onSubmit={(e) => {
              e.preventDefault();
              startNewGame();
            }}
          >
            <h2>New Game</h2>
            <Show when={Object.keys(GAME_MODULES).length > 1}>
              <label>
                Game
                <select value={state().gameKind} onChange={(e) => onGameKindChange(e.currentTarget.value)}>
                  <For each={Object.keys(GAME_MODULES)}>
                    {(kind) => {
                      const label = state().gamesInfo.find((g) => g.kind === kind)?.label ?? kind;
                      return <option value={kind}>{label}</option>;
                    }}
                  </For>
                </select>
              </label>
            </Show>
            <Show when={mod()?.NewGameFields}>
              {(Fields) => <Dynamic component={Fields()} config={pendingConfig()} onChange={setPendingConfig} />}
            </Show>
            <For each={GAME_META[state().gameKind]?.players ?? []}>
              {(player) => (
                <>
                  <label>
                    {player}
                    <select
                      value={seatSelectValue(pendingSeats()[player] ?? "human")}
                      onChange={(e) => {
                        const value = e.currentTarget.value;
                        setPendingSeats((s) => {
                          if (value === "human") return { ...s, [player]: "human" };
                          if (value === "custom") {
                            const sch = schema();
                            if (!sch) return s;
                            return { ...s, [player]: { kind: "custom", spec: defaultCustomStrategySpec(sch) } };
                          }
                          return { ...s, [player]: { kind: "preset", id: value } };
                        });
                      }}
                    >
                      <option value="human">Human</option>
                      <For each={presetOptions()}>{(preset) => <option value={preset.id}>AI: {preset.label}</option>}</For>
                      <option value="custom" disabled={!schema()}>
                        Custom…
                      </option>
                    </select>
                  </label>
                  <Show when={(() => {
                    const control = pendingSeats()[player];
                    return control !== "human" && control?.kind === "custom" ? control : undefined;
                  })()}>
                    {(custom) => (
                      <Show when={schema()}>
                        {(sch) => (
                          <StrategyConfigEditor
                            schema={sch()}
                            config={custom().spec}
                            onChange={(spec) => setPendingSeats((s) => ({ ...s, [player]: { kind: "custom", spec } }))}
                          />
                        )}
                      </Show>
                    )}
                  </Show>
                </>
              )}
            </For>
            <div class="dialog-actions">
              <button type="button" onClick={() => setDialogOpen(false)}>
                Cancel
              </button>
              <button type="submit" id="new-game-start">
                Start
              </button>
            </div>
          </form>
        </dialog>
      </Show>
    </>
  );
};