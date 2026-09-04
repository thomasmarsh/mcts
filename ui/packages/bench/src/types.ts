// types.ts — Wire types mirroring server/bench/mod.rs's response shapes and
// query parameters. Field names match the JSON wire format exactly
// (snake_case), same convention as @mcts/game's types.ts: the Rust structs
// don't set `#[serde(rename_all)]`, so there's no camelCase translation
// layer anywhere. Kept separate from api-client.ts and reducer.ts so
// neither needs to import from the other.
//
// Everything here is shared by the tuner UI (`src/tuner/`), which has its
// own wire types for run/trial/evidence data (`tuner-types.ts`) built on a
// separate `tuner-api-client.ts`.

/** JSON carried by persisted tuning manifests and configuration snapshots.
 * Keeping this boundary explicit prevents presentation code from treating
 * lifecycle payloads as arbitrary records. */
export type JsonValue =
  null | boolean | number | string | JsonValue[] | { [key: string]: JsonValue };

/** One entry in a tuner's parameter space (`game_host::TunerParameter`).
 * `spec` is `#[serde(flatten)]`ed on the Rust side, so `type`/`bounds`/
 * `choices`/`value`/`default` land as top-level sibling fields of `name`
 * rather than nested — this type mirrors the wire shape directly. Only
 * `type` is guaranteed; which other fields are present depends on it
 * (`float`/`int` -> `bounds`+`default`, `categorical` -> `choices`+
 * `default`, `constant` -> `value`). */
export interface TunerParameter {
  name: string;
  type: "float" | "int" | "categorical" | "constant" | (string & {});
  bounds?: [number, number];
  choices?: string[];
  default?: unknown;
  value?: unknown;
}

/** A conditional activation rule (`game_host::TunerCondition`): when the
 * trial's config matches every `if` entry, every parameter named in `then`
 * is also active (and therefore present in a trial's `config`). `if`'s
 * values are either a single value or a list of values (any-of). */
export interface TunerCondition {
  if: Record<string, unknown>;
  then: string[];
}

/** Bounds and types for a game's `game_config` setup axis
 * (`game_host::GameConfigSchema`), reusing the `TunerParameter` /
 * `TunerCondition` shapes verbatim. An empty `parameters` list means the
 * board is fixed at compile time — nothing to configure. Druid's `{w, h}`
 * size is two dotted `size.w` / `size.h` parameters. */
export interface GameConfigSchema {
  parameters: TunerParameter[];
  conditions: TunerCondition[];
}

/** A game's tunable search space, as reported by `tune describe`
 * (`game_host::TunerInfo`) and surfaced through `GET
 * /api/bench/tuner/kinds`. `baselines` is a list rather than a single id so
 * tuner can evaluate each trial against multiple opponent strengths
 * (`Scenario(instances=...)`) -- most games report one entry; a game with a
 * genuine second, harder preset (e.g. druid's "master") can list it as a
 * second instance. */
export interface TunerInfo {
  id: string;
  baselines: string[];
  eval_rounds: number;
  parameters: TunerParameter[];
  conditions: TunerCondition[];
  /** The game's own `default_config()` -- a game-setup axis (e.g. Druid's
   * board size) tuner never searches over, unlike `parameters`. `{}` means
   * the game's board is fixed at compile time and there's nothing to
   * configure here. */
  game_config: unknown;
  /** Bounds and types for the `game_config` axis, so a launch form or the
   * objective editor can accept and validate a non-default value (e.g.
   * AtariGo on 9×9). Absent / empty `parameters` for a fixed-board game. */
  game_config_schema?: GameConfigSchema;
}

/** `GET /api/bench/tuner/kinds` element — one tunable game. */
export interface TunableGame {
  game: string;
  tuner: TunerInfo;
}
