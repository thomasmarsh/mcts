// api-client.ts — Typed fetch wrapper for the mcts stateless game server
// (server/main.rs / server/adapters/mod.rs's `GameAdapter` contract).
// PLAN-UI.md's "Hard rule": this is the *only* file in this package allowed
// to reference `fetch` -- enforced by the fetch-ban eslint rule in
// ui/eslint.config.js. Three layers, mirroring pb/ui/app/src/api-client.ts:
//   1. `ApiClient` -- a plain interface of `Promise`-returning methods.
//   2. `createApiClient(): ApiClient` -- the one concrete implementation.
//   3. `createEnv(api): Env` -- lifts every method into an `Effect`. `Env`
//      (the type reducers actually receive) is defined in reducer.ts, not
//      here -- see that file's header comment for why.

import { Effect } from "@mcts/core";
import type { Env } from "./reducer.js";
import type {
  AiMoveResult,
  AiPresetInfo,
  Analysis,
  GameInfo,
  LegalMovesResult,
  StateAndView,
} from "./types.js";

export interface ApiClient {
  getGames(): Promise<GameInfo[]>;
  newGame<S, V = unknown>(kind: string, config?: unknown): Promise<StateAndView<S, V>>;
  legalMoves<S, M>(kind: string, state: S): Promise<LegalMovesResult<M>>;
  view<S, V = unknown>(kind: string, state: S): Promise<V>;
  apply<S, M, V = unknown>(kind: string, state: S, move: M): Promise<StateAndView<S, V>>;
  aiPresets(kind: string): Promise<AiPresetInfo[]>;
  aiMove<S, M, V = unknown>(kind: string, state: S, preset: string): Promise<AiMoveResult<S, M, V>>;
  analyze<S, M>(kind: string, state: S, preset: string, budgetMs?: number): Promise<Analysis<M>>;
}

/** The server (`AdapterError`'s `IntoResponse` impl, `server/adapters/mod.rs`)
 * returns a bare-string error body today, not a JSON envelope -- structured
 * `{error, code}` bodies are PLAN-UI.md session 9's job. Read as text rather
 * than attempting to parse JSON. */
async function errorMessage(r: Response): Promise<string> {
  const text = await r.text().catch(() => "");
  return text || `API ${r.status}`;
}

async function fetchJson<T>(url: string): Promise<T> {
  const r = await fetch(url);
  if (!r.ok) throw new Error(await errorMessage(r));
  return r.json() as Promise<T>;
}

async function postJson<T>(url: string, body: unknown): Promise<T> {
  const r = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!r.ok) throw new Error(await errorMessage(r));
  return r.json() as Promise<T>;
}

/** `baseUrl` defaults to `""` (relative URLs, resolved against whatever
 * origin the page loads from -- the Vite dev proxy or the server's own
 * `ServeDir` in production). A non-empty override exists only so
 * `tests/integration.test.ts` can point this at a `cargo run` server
 * directly from vitest's node/happy-dom environment, which has no page
 * origin for a relative URL to resolve against. */
export function createApiClient(baseUrl = ""): ApiClient {
  const url = (path: string): string => baseUrl + path;
  return {
    async getGames(): Promise<GameInfo[]> {
      return fetchJson(url("/api/games"));
    },
    async newGame<S, V = unknown>(kind: string, config?: unknown): Promise<StateAndView<S, V>> {
      return postJson(url(`/api/games/${encodeURIComponent(kind)}/new`), { config });
    },
    async legalMoves<S, M>(kind: string, state: S): Promise<LegalMovesResult<M>> {
      return postJson(url(`/api/games/${encodeURIComponent(kind)}/legal_moves`), { state });
    },
    async view<S, V = unknown>(kind: string, state: S): Promise<V> {
      return postJson(url(`/api/games/${encodeURIComponent(kind)}/view`), { state });
    },
    async apply<S, M, V = unknown>(kind: string, state: S, move: M): Promise<StateAndView<S, V>> {
      return postJson(url(`/api/games/${encodeURIComponent(kind)}/apply`), { state, move });
    },
    async aiPresets(kind: string): Promise<AiPresetInfo[]> {
      return fetchJson(url(`/api/games/${encodeURIComponent(kind)}/ai_presets`));
    },
    async aiMove<S, M, V = unknown>(kind: string, state: S, preset: string): Promise<AiMoveResult<S, M, V>> {
      return postJson(url(`/api/games/${encodeURIComponent(kind)}/ai_move`), { state, preset });
    },
    async analyze<S, M>(kind: string, state: S, preset: string, budgetMs?: number): Promise<Analysis<M>> {
      return postJson(url(`/api/games/${encodeURIComponent(kind)}/analyze`), {
        state,
        preset,
        budget_ms: budgetMs,
      });
    },
  };
}

export function createEnv(api: ApiClient): Env {
  const lift = <T>(thunk: () => Promise<T>): Effect<T> => Effect.fromPromise(thunk);
  return {
    getGames: () => lift(() => api.getGames()),
    newGame: <S, V = unknown>(kind: string, config?: unknown) => lift(() => api.newGame<S, V>(kind, config)),
    legalMoves: <S, M>(kind: string, state: S) => lift(() => api.legalMoves<S, M>(kind, state)),
    view: <S, V = unknown>(kind: string, state: S) => lift(() => api.view<S, V>(kind, state)),
    apply: <S, M, V = unknown>(kind: string, state: S, move: M) => lift(() => api.apply<S, M, V>(kind, state, move)),
    aiPresets: (kind: string) => lift(() => api.aiPresets(kind)),
    aiMove: <S, M, V = unknown>(kind: string, state: S, preset: string) =>
      lift(() => api.aiMove<S, M, V>(kind, state, preset)),
    analyze: <S, M>(kind: string, state: S, preset: string, budgetMs?: number) =>
      lift(() => api.analyze<S, M>(kind, state, preset, budgetMs)),
  };
}
