// types.ts — Wire types mirroring server/adapters/mod.rs's `GameAdapter`
// contract and server/main.rs's route response shapes -- typed against the
// real contract, not a sketch. Field names
// match the JSON wire format exactly (snake_case): none of the Rust structs
// this mirrors set `#[serde(rename_all = ...)]`, so there's no
// camelCase-translation layer here either. Kept separate from both
// api-client.ts and reducer.ts (rather than folded into either) so neither
// needs to import from the other -- see reducer.ts's `Env` / api-client.ts's
// `createEnv` for why that avoided circular import matters here.

/** `GET /api/games` element. */
export interface GameInfo {
  kind: string;
  label: string;
  description: string;
  config_schema: unknown;
}

/** `GET /api/games/{kind}/ai_presets` element. */
export interface AiPresetInfo {
  id: string;
  label: string;
  description: string;
}

/** One candidate root action from `analyze`, mirroring
 * `server::adapters::AnalysisAction` (action encoded as the game's own move
 * type `M`, not `Value` -- that erasure only exists on the Rust side of the
 * wire). */
export interface AnalysisAction<M> {
  action: M;
  visits: number;
  mean_value: number;
  is_proven: boolean;
}

/** `POST /api/games/{kind}/analyze` response, mirroring `server::adapters::Analysis`. */
export interface Analysis<M> {
  actions: AnalysisAction<M>[];
  principal_variation: M[];
  total_visits: number;
  suggested_move: M | null;
}

/** `POST /api/games/{kind}/new` and `POST /api/games/{kind}/apply` share this
 * response shape -- see server/main.rs's `post_new`/`post_apply`. */
export interface StateAndView<S, V = unknown> {
  state: S;
  view: V;
}

/** `POST /api/games/{kind}/ai_move` response -- `StateAndView` plus the move
 * the AI chose. */
export interface AiMoveResult<S, M, V = unknown> {
  move: M;
  state: S;
  view: V;
}

/** `POST /api/games/{kind}/legal_moves` response. */
export interface LegalMovesResult<M> {
  moves: M[];
}
