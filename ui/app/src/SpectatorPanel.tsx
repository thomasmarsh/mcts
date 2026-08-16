// SpectatorPanel.tsx — Read-only playback for persisted game traces.

import { Dynamic } from "solid-js/web";
import { createEffect, createMemo, createResource, createSignal, For, onCleanup, Show, type Component } from "solid-js";
import { createBenchApiClient } from "../../packages/bench/src/api-client.js";
import type { GameMove, GameTraceSummary, LiveGameMove } from "../../packages/bench/src/types.js";
import type { GameKindModule, MoveStep } from "@mcts/game";
import { GAME_MODULES } from "./games.js";

function readonlyView(state: unknown): unknown {
  return state && typeof state === "object"
    ? { ...(state as Record<string, unknown>), terminal: false, winner: null }
    : { terminal: false, winner: null };
}

export const SpectatorPanel: Component<{ runId: string; game: string; kind: string; live: boolean; cellId?: string; initialGameSeq?: number }> = (props) => {
  const api = createBenchApiClient();
  const [games, { refetch: refetchGames }] = createResource(() => JSON.stringify([props.runId, props.cellId ?? null]), (key) => {
    const [runId, cellId] = JSON.parse(key) as [string, string | null];
    return api.getRunGames(runId, 100, cellId);
  });
  const [selectedSeq, setSelectedSeq] = createSignal<number | null>(null);
  const [moves, setMoves] = createSignal<GameMove[]>([]);
  const [moveError, setMoveError] = createSignal<string | null>(null);
  const [currentPly, setCurrentPly] = createSignal(0);
  const [liveError, setLiveError] = createSignal<string | null>(null);
  const [appliedInitialKey, setAppliedInitialKey] = createSignal<string | null>(null);
  const traceRequestKey = createMemo(() => JSON.stringify([props.runId, props.cellId ?? null, props.initialGameSeq ?? null]));
  const [module] = createResource(() => props.game, async (game): Promise<GameKindModule<unknown, unknown, unknown> | null> => {
    const load = GAME_MODULES[game];
    return load ? load() : null;
  });

  const currentMove = createMemo(() => moves()[currentPly()] ?? null);
  const history = createMemo((): MoveStep<unknown, unknown>[] => {
    const trace = moves();
    const path: MoveStep<unknown, unknown>[] = [];
    for (let i = 1; i <= Math.min(currentPly(), trace.length - 1); i++) {
      const row = trace[i]!;
      if (row.mv !== null) path.push({ move: row.mv, before: trace[i - 1]!.state });
    }
    return path;
  });

  const selectedGame = createMemo(() => (games() ?? []).find((game) => game.game_seq === selectedSeq()) ?? null);
  const isRendererTrace = createMemo(() => currentMove() !== null && typeof currentMove()!.state !== "string");

  createEffect(() => {
    traceRequestKey();
    setAppliedInitialKey(null);
    setSelectedSeq(null);
    setMoves([]);
    setCurrentPly(0);
    setMoveError(null);
    setLiveError(null);
  });

  createEffect(() => {
    const key = traceRequestKey();
    const requested = props.initialGameSeq;
    const available = games() ?? [];
    if (games.loading || appliedInitialKey() === key) return;
    if (requested !== undefined && available.some((game) => game.game_seq === requested)) {
      setAppliedInitialKey(key);
      void selectGame(requested);
    }
  });

  async function selectGame(gameSeq: number): Promise<void> {
    const requestKey = traceRequestKey();
    setSelectedSeq(gameSeq);
    setMoveError(null);
    try {
      const rows = await api.getRunGameMoves(props.runId, gameSeq);
      if (requestKey !== traceRequestKey()) return;
      setMoves(rows);
      setCurrentPly(0);
    } catch (e: unknown) {
      if (requestKey !== traceRequestKey()) return;
      setMoves([]);
      setMoveError(String(e));
    }
  }

  createEffect(() => {
    const gameSeq = selectedSeq();
    if (gameSeq === null || !props.live) return;
    setLiveError(null);
    const source = new EventSource(`/api/bench/runs/${encodeURIComponent(props.runId)}/live?game_seq=${encodeURIComponent(gameSeq)}`);
    source.onmessage = (event) => {
      try {
        const row = JSON.parse(event.data) as LiveGameMove;
        if (row.game_seq !== gameSeq) return;
        setMoves((old) => [...old.filter((existing) => existing.ply !== row.ply), {
          ply: row.ply, ts: row.ts, state: row.state, mv: row.mv, player: row.player,
        }].sort((a, b) => a.ply - b.ply));
        void refetchGames();
      } catch (e: unknown) {
        setLiveError(`Invalid live trace event: ${String(e)}`);
      }
    };
    source.onerror = () => setLiveError("Live trace connection lost.");
    onCleanup(() => source.close());
  });

  // Keep the picker current for a running run without selecting, rendering,
  // or otherwise taking focus away from the game the operator chose.
  createEffect(() => {
    if (!props.live) return;
    const source = new EventSource(`/api/bench/runs/${encodeURIComponent(props.runId)}/live`);
    source.onmessage = () => void refetchGames();
    onCleanup(() => source.close());
  });

  function newestGame(): void {
    const game = games()?.[0];
    if (game) void selectGame(game.game_seq);
  }

  return <section id="spectator-panel">
    <div id="spectator-header"><strong>Game traces</strong><Show when={props.live}><button id="spectator-live-btn" onClick={newestGame}>Follow newest game</button></Show></div>
    <Show when={liveError()}><div class="log-error">{liveError()}</div></Show>
    <div id="spectator-layout">
      <div id="spectator-games">
        <Show when={!games.loading} fallback={<div class="log-empty">Loading games…</div>}><Show when={(games() ?? []).length > 0} fallback={<div class="log-empty">No traced games yet.</div>}><For each={games() ?? []}>{(game: GameTraceSummary) => <button class="spectator-game" classList={{ active: selectedSeq() === game.game_seq }} onClick={() => void selectGame(game.game_seq)}>#{game.game_seq} · {game.strategy_a && game.strategy_b ? `${game.strategy_a} vs ${game.strategy_b}` : props.kind === "smac3" ? `${game.ply_count} plies · tuning worker` : `${game.ply_count} plies`}</button>}</For></Show></Show>
      </div>
      <div id="spectator-board">
        <Show when={moveError()}><div class="log-error">{moveError()}</div></Show>
        <Show when={currentMove()} fallback={<div class="log-empty">Choose a game to inspect. Selecting it never switches to another running game.</div>}>{(row) => <Show when={isRendererTrace()} fallback={<div class="spectator-text-trace"><div><strong>SMAC3 trace</strong>{selectedGame() ? ` · game #${selectedGame()!.game_seq}` : ""}</div><div>Player: {row().player ?? "initial state"}</div><pre>{String(row().state)}</pre><Show when={row().mv !== null}><pre>Move: {JSON.stringify(row().mv)}</pre></Show></div>}><Show when={module()} fallback={<div class="log-empty">Loading board…</div>}>{(mod) => <Dynamic component={mod().Renderer} state={row().state} view={readonlyView(row().state)} history={history()} legalMoves={[]} busy={true} onMove={() => undefined} hoveredMove={null} onHover={() => undefined} />}</Show></Show>}</Show>
        <Show when={moves().length > 0}><div id="spectator-controls"><button disabled={currentPly() === 0} onClick={() => setCurrentPly((ply) => ply - 1)}>Previous</button><span>Ply {currentMove()?.ply ?? 0} / {moves()[moves().length - 1]?.ply ?? 0}</span><button disabled={currentPly() >= moves().length - 1} onClick={() => setCurrentPly((ply) => ply + 1)}>Next</button></div></Show>
      </div>
    </div>
  </section>;
};
