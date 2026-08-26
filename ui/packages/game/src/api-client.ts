// api-client.ts — Typed fetch wrapper for the mcts stateless game server
// (server/main.rs / server/adapters/mod.rs's `GameAdapter` contract).
// Hard rule: this is the *only* file in this package allowed
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
  AiStrategyRef,
  Analysis,
  AxisSchema,
  GameInfo,
  LegalMovesResult,
  StateAndView,
  TunerInfo,
} from "./types.js";

/** Flattens an `AiStrategyRef` into `server::main::AiMoveRequest`/
 * `AnalyzeRequest`'s actual `{preset, custom?}` wire shape -- `preset` stays
 * a required, non-empty string even for the custom path (a literal
 * `"custom"` sentinel, kept for server-side logging -- see that struct's doc
 * comment), with `custom` carrying the real spec alongside it. */
function strategyBody(strategy: AiStrategyRef): { preset: string; custom?: unknown } {
  return strategy.kind === "preset" ? { preset: strategy.id } : { preset: "custom", custom: strategy.spec };
}

export interface ApiClient {
  getGames(): Promise<GameInfo[]>;
  newGame<S, V = unknown>(kind: string, config?: unknown): Promise<StateAndView<S, V>>;
  legalMoves<S, M>(kind: string, state: S): Promise<LegalMovesResult<M>>;
  view<S, V = unknown>(kind: string, state: S): Promise<V>;
  apply<S, M, V = unknown>(kind: string, state: S, move: M): Promise<StateAndView<S, V>>;
  aiPresets(kind: string): Promise<AiPresetInfo[]>;
  aiMove<S, M, V = unknown>(kind: string, state: S, strategy: AiStrategyRef): Promise<AiMoveResult<S, M, V>>;
  analyze<S, M>(kind: string, state: S, strategy: AiStrategyRef, budgetMs?: number): Promise<Analysis<M>>;
  fetchStrategySchema(): Promise<AxisSchema>;
  fetchStrategyFamilies(kind: string): Promise<TunerInfo | null>;
}

/** The server (`AdapterError`'s `IntoResponse` impl, `server/adapters/mod.rs`)
 * returns a structured `{error, code}` JSON body.
 * Read as text first (a body-limit/timeout rejection, or anything below the
 * `AdapterError` layer, may not be JSON at all) and only then try to parse
 * it as `{error}`, falling back to the raw text. */
async function errorMessage(r: Response): Promise<string> {
  const text = await r.text().catch(() => "");
  if (text) {
    try {
      const body: unknown = JSON.parse(text);
      if (body && typeof body === "object" && typeof (body as { error?: unknown }).error === "string") {
        return (body as { error: string }).error;
      }
    } catch {
      // Not JSON (e.g. a plain-text timeout/body-limit rejection) -- fall
      // through to the raw text below.
    }
  }
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
    async aiMove<S, M, V = unknown>(kind: string, state: S, strategy: AiStrategyRef): Promise<AiMoveResult<S, M, V>> {
      return postJson(url(`/api/games/${encodeURIComponent(kind)}/ai_move`), { state, ...strategyBody(strategy) });
    },
    async analyze<S, M>(kind: string, state: S, strategy: AiStrategyRef, budgetMs?: number): Promise<Analysis<M>> {
      return postJson(url(`/api/games/${encodeURIComponent(kind)}/analyze`), {
        state,
        ...strategyBody(strategy),
        budget_ms: budgetMs,
      });
    },
    async fetchStrategySchema(): Promise<AxisSchema> {
      return fetchJson(url("/api/strategy-schema"));
    },
    async fetchStrategyFamilies(kind: string): Promise<TunerInfo | null> {
      return fetchJson(url(`/api/games/${encodeURIComponent(kind)}/strategy-families`));
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
    aiMove: <S, M, V = unknown>(kind: string, state: S, strategy: AiStrategyRef) =>
      lift(() => api.aiMove<S, M, V>(kind, state, strategy)),
    analyze: <S, M>(kind: string, state: S, strategy: AiStrategyRef, budgetMs?: number) =>
      lift(() => api.analyze<S, M>(kind, state, strategy, budgetMs)),
  };
}
