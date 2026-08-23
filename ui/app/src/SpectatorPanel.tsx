// SpectatorPanel.tsx — Read-only playback for persisted game traces.

import { Dynamic } from "solid-js/web";
import { createEffect, createMemo, createResource, createSignal, For, onCleanup, Show, type Component } from "solid-js";
import { createBenchApiClient } from "../../packages/bench/src/api-client.js";
import type { BenchSpectatorProps, GameMove, GameTraceSummary, LiveGameMove } from "../../packages/bench/src/types.js";
import type { GameKindModule, MoveStep, SearchReport } from "@mcts/game";
import { SearchInspector, type SearchInspectorPoint } from "@mcts/search-inspector";
import { GAME_MODULES } from "./games.js";

export interface TraceApi {
  getRunGames(runId: string, limit?: number, cellId?: string | null): Promise<GameTraceSummary[]>;
  getRunGameMoves(runId: string, gameSeq: number): Promise<GameMove[]>;
}

export interface TraceEventSource {
  onmessage: ((event: MessageEvent<string>) => unknown) | null;
  onerror: ((event: Event) => unknown) | null;
  close(): void;
}

/** Browser-facing trace dependencies, kept injectable for component tests. */
export interface TraceEnvironment {
  api: TraceApi;
  eventSource(url: string): TraceEventSource;
}

export interface SpectatorPanelProps extends BenchSpectatorProps {
  traceEnv?: TraceEnvironment;
}

function defaultTraceEnvironment(): TraceEnvironment {
  return {
    api: createBenchApiClient(),
    eventSource: (url) => new EventSource(url),
  };
}

function readonlyView(state: unknown): unknown {
  return state && typeof state === "object"
    ? { ...(state as Record<string, unknown>), terminal: false, winner: null }
    : { terminal: false, winner: null };
}

function mergeMoves(rows: GameMove[], row: GameMove): GameMove[] {
  return [...rows.filter((existing) => existing.ply !== row.ply), row].sort((a, b) => a.ply - b.ply);
}

function countLabel(count: number, noun: string): string {
  return `${count} newer ${noun}${count === 1 ? "" : "s"}`;
}

export const SpectatorPanel: Component<SpectatorPanelProps> = (props) => {
  const traceEnv = props.traceEnv ?? defaultTraceEnvironment();
  const [games, setGames] = createSignal<GameTraceSummary[]>([]);
  const [gamesLoading, setGamesLoading] = createSignal(true);
  const [gamesError, setGamesError] = createSignal<string | null>(null);
  const [selectedSeq, setSelectedSeq] = createSignal<number | null>(null);
  const [moves, setMoves] = createSignal<GameMove[]>([]);
  const [moveError, setMoveError] = createSignal<string | null>(null);
  const [currentPly, setCurrentPly] = createSignal(0);
  const [liveError, setLiveError] = createSignal<string | null>(null);
  const [appliedInitialKey, setAppliedInitialKey] = createSignal<string | null>(null);
  let gameListGeneration = 0;
  let moveGeneration = 0;
  let selectedStreamGeneration = 0;
  const listKey = createMemo(() => JSON.stringify([props.runId, props.cellId ?? null]));
  const selectionKey = createMemo(() => JSON.stringify([props.runId, props.cellId ?? null, props.initialGameSeq ?? null]));
  const [module] = createResource(() => props.game, async (game): Promise<GameKindModule<unknown, unknown, unknown> | null> => {
    const load = GAME_MODULES[game];
    return load ? load() : null;
  });

  const currentIndex = createMemo(() => moves().findIndex((row) => row.ply === currentPly()));
  const currentMove = createMemo(() => moves()[currentIndex()] ?? null);
  const previousState = createMemo(() => {
    const index = currentIndex();
    return index > 0 ? moves()[index - 1]!.state : currentMove()?.state;
  });
  const history = createMemo((): MoveStep<unknown, unknown>[] => {
    const index = currentIndex();
    const trace = moves();
    const path: MoveStep<unknown, unknown>[] = [];
    for (let i = 1; i <= index; i++) {
      const row = trace[i]!;
      if (row.mv !== null) path.push({ move: row.mv, before: trace[i - 1]!.state });
    }
    return path;
  });
  const searchPoints = createMemo((): SearchInspectorPoint<unknown>[] => moves().flatMap((row) => row.mv === null
    ? []
    : [{ ply: row.ply, player: row.player ?? "Unknown", move: row.mv, report: row.search }]));
  const selectedGame = createMemo(() => games().find((game) => game.game_seq === selectedSeq()) ?? null);
  const isRendererTrace = createMemo(() => currentMove() !== null && typeof currentMove()!.state !== "string");
  const newerGames = createMemo(() => selectedSeq() === null ? 0 : games().filter((game) => game.game_seq > selectedSeq()!).length);
  const newerPlies = createMemo(() => currentMove() === null ? 0 : moves().filter((row) => row.ply > currentPly()).length);

  function requestGames(): void {
    const key = listKey();
    const [runId, cellId] = JSON.parse(key) as [string, string | null];
    const generation = ++gameListGeneration;
    setGamesLoading(true);
    setGamesError(null);
    void traceEnv.api.getRunGames(runId, 100, cellId).then(
      (rows) => {
        if (generation !== gameListGeneration || key !== listKey()) return;
        setGames(rows);
        setGamesLoading(false);
      },
      (error: unknown) => {
        if (generation !== gameListGeneration || key !== listKey()) return;
        setGames([]);
        setGamesError(String(error));
        setGamesLoading(false);
      },
    );
  }

  createEffect(() => {
    listKey();
    requestGames();
  });

  createEffect(() => {
    selectionKey();
    ++moveGeneration;
    ++selectedStreamGeneration;
    setAppliedInitialKey(null);
    setSelectedSeq(null);
    setMoves([]);
    setCurrentPly(0);
    setMoveError(null);
    setLiveError(null);
  });

  createEffect(() => {
    const key = selectionKey();
    const requested = props.initialGameSeq;
    if (gamesLoading() || appliedInitialKey() === key || requested === undefined) return;
    if (games().some((game) => game.game_seq === requested)) {
      setAppliedInitialKey(key);
      void selectGame(requested);
    }
  });

  async function selectGame(gameSeq: number): Promise<void> {
    const key = selectionKey();
    const generation = ++moveGeneration;
    setSelectedSeq(gameSeq);
    setMoves([]);
    setCurrentPly(0);
    setMoveError(null);
    try {
      const rows = await traceEnv.api.getRunGameMoves(props.runId, gameSeq);
      if (generation !== moveGeneration || key !== selectionKey() || selectedSeq() !== gameSeq) return;
      const sorted = [...rows].sort((a, b) => a.ply - b.ply);
      setMoves(sorted);
      setCurrentPly(sorted[0]?.ply ?? 0);
    } catch (error: unknown) {
      if (generation !== moveGeneration || key !== selectionKey() || selectedSeq() !== gameSeq) return;
      setMoves([]);
      setMoveError(String(error));
    }
  }

  createEffect(() => {
    const runId = props.runId;
    const gameSeq = selectedSeq();
    const key = selectionKey();
    if (gameSeq === null || !props.live) return;
    const generation = ++selectedStreamGeneration;
    setLiveError(null);
    const source = traceEnv.eventSource(`/api/bench/runs/${encodeURIComponent(runId)}/live?game_seq=${encodeURIComponent(gameSeq)}`);
    source.onmessage = (event) => {
      if (generation !== selectedStreamGeneration || key !== selectionKey() || selectedSeq() !== gameSeq) return;
      try {
        const row = JSON.parse(event.data) as LiveGameMove;
        if (row.game_seq !== gameSeq) return;
        setMoves((old) => mergeMoves(old, row));
        requestGames();
      } catch (error: unknown) {
        if (generation === selectedStreamGeneration) setLiveError(`Invalid live trace event: ${String(error)}`);
      }
    };
    source.onerror = () => {
      if (generation === selectedStreamGeneration && key === selectionKey()) setLiveError("Live trace connection lost.");
    };
    onCleanup(() => source.close());
  });

  createEffect(() => {
    const runId = props.runId;
    if (!props.live) return;
    const source = traceEnv.eventSource(`/api/bench/runs/${encodeURIComponent(runId)}/live`);
    source.onmessage = () => requestGames();
    onCleanup(() => source.close());
  });

  function setPly(index: number): void {
    const row = moves()[index];
    if (row) setCurrentPly(row.ply);
  }

  return <section id="spectator-panel">
    <div id="spectator-header"><strong>Game traces</strong><Show when={newerGames() > 0}><span>{countLabel(newerGames(), "game")}</span></Show></div>
    <Show when={liveError()}><div class="log-error">{liveError()}</div></Show>
    <div id="spectator-layout">
      <div id="spectator-games">
        <Show when={gamesError()}><div class="log-error">{gamesError()}</div></Show>
        <Show when={!gamesLoading()} fallback={<div class="log-empty">Loading games…</div>}><Show when={games().length > 0} fallback={<div class="log-empty">No traced games yet.</div>}><For each={games()}>{(game) => <button class="spectator-game" classList={{ active: selectedSeq() === game.game_seq }} onClick={() => void selectGame(game.game_seq)}>#{game.game_seq} · {game.strategy_a && game.strategy_b ? `${game.strategy_a} vs ${game.strategy_b}` : props.kind === "tuner" ? `${game.ply_count} plies · tuning worker` : `${game.ply_count} plies`}</button>}</For></Show></Show>
      </div>
      <div id="spectator-board">
        <Show when={moveError()}><div class="log-error">{moveError()}</div></Show>
        <Show when={currentMove()} fallback={<div class="log-empty">Choose a game to inspect. Selecting it never switches to another running game.</div>}>{(row) => <><Show when={isRendererTrace()} fallback={<div class="spectator-text-trace"><div><strong>tuner trace</strong>{selectedGame() ? ` · game #${selectedGame()!.game_seq}` : ""}</div><div>Player: {row().player ?? "initial state"}</div><pre>{String(row().state)}</pre><Show when={row().mv !== null}><pre>Move: {JSON.stringify(row().mv)}</pre></Show></div>}><Show when={module()} fallback={<div class="log-empty">Loading board…</div>}>{(mod) => <Dynamic component={mod().Renderer} state={row().state} view={readonlyView(row().state)} history={history()} legalMoves={[]} busy={true} onMove={() => undefined} hoveredMove={null} onHover={() => undefined} />}</Show></Show>
          <section class="spectator-search"><SearchInspector report={row().search as SearchReport<unknown> | null | undefined} points={searchPoints()} before={previousState()} formatMove={module()?.formatMove} /></section>
        </>}</Show>
        <Show when={moves().length > 0}><div id="spectator-controls"><button disabled={currentIndex() <= 0} onClick={() => setPly(0)}>First</button><button disabled={currentIndex() <= 0} onClick={() => setPly(currentIndex() - 1)}>Previous</button><span>Ply {currentMove()?.ply ?? 0} / {moves()[moves().length - 1]?.ply ?? 0}<Show when={newerPlies() > 0}> · {countLabel(newerPlies(), "ply")}</Show></span><button disabled={currentIndex() < 0 || currentIndex() >= moves().length - 1} onClick={() => setPly(currentIndex() + 1)}>Next</button><button disabled={currentIndex() < 0 || currentIndex() >= moves().length - 1} onClick={() => setPly(moves().length - 1)}>Last</button></div></Show>
      </div>
    </div>
  </section>;
};
