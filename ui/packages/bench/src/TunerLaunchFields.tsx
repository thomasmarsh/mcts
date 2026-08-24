// TunerLaunchFields.tsx — The `tuner`-kind portion of the launch form.
//
// Unlike round_robin (pick strategies to pit against each other), a tuner
// run's "config" is an optimizer budget over a search space the *game*
// defines, not the launcher. So this renders:
//   - a game picker restricted to games that report tuner metadata
//     (`GET /api/bench/tuner/kinds` — only games with a `tuner()` impl
//     appear at all, so there's nothing to disable/grey out here);
//   - a read-only summary of that game's search space (parameters,
//     conditions, the tuner's *default* eval rounds/trial) so the
//     operator can see what a trial actually varies before committing a
//     budget to it;
//   - the actual editable fields: target trials, workers, resolved resource
//     limits, sampler/pruning inputs, and the per-run compute budget
//     (`optimizer.*`/`target.*` overrides). The parameter *values*
//     themselves aren't editable here — `tuner`'s CLI `--override` only
//     reaches dotted dataclass attributes, not the list-shaped
//     `parameters:` search space, so there is nothing for a form field to
//     write to;
//   - a raw-JSON "Game config" textarea, only for a game whose `tuner().
//     game_config` isn't `{}` (today, only Druid) — a game-setup axis (e.g.
//     board size) separate from the strategy search space above: tuner
//     never searches over it, it just pins every trial in the run to it.
//     Deliberately a generic JSON field rather than a typed per-game picker
//     (e.g. a board-size dropdown) so a future game with its own config
//     needs no new UI code here.
//
// Former SMAC3-era fields (starting baselines panel, ladder/rung/saturation)
// have been removed — the pool auto-seeds with "default" and "random"
// anchors, and matchmaking is a flat iterative loop against the closest
// anchor, not a multi-rung automated ladder.
//
// LaunchForm owns all of this component's state (same lifted-state
// convention as the strategy picker) and builds the `--override` argv from
// it at submit time — see `buildTunerOverrides` there.

import { createMemo, For, Show, type Component } from "solid-js";
import type { TunerGameInfo, TunerParameter } from "./index.js";

/** Whether a `tuner().game_config` value means "nothing to configure" —
 * every game but Druid reports `{}` here. */
export function isEmptyGameConfig(gameConfig: unknown): boolean {
  return (
    !!gameConfig &&
    typeof gameConfig === "object" &&
    !Array.isArray(gameConfig) &&
    Object.keys(gameConfig).length === 0
  );
}

/** Render one parameter's range/choices/value as a compact string. Full
 * text (not truncated) — callers that display this in a narrow column
 * truncate it themselves and rely on this being the `title` tooltip. */
function paramRange(p: TunerParameter): string {
  switch (p.type) {
    case "float":
    case "int":
      return p.bounds ? `${p.bounds[0]} – ${p.bounds[1]}` : "";
    case "categorical":
      return p.choices ? p.choices.join(" / ") : "";
    case "constant":
      return "fixed";
    default:
      return "";
  }
}

function paramDefault(p: TunerParameter): string {
  const v = p.type === "constant" ? p.value : p.default;
  return v === undefined || v === null ? "—" : String(v);
}

/** Human-readable summary of one condition, e.g. "schedule = threshold -> rave". */
function conditionLabel(cond: TunerGameInfo["tuner"]["conditions"][number]): string {
  const [parent, value] = Object.entries(cond.if)[0] ?? ["?", "?"];
  const valueLabel = Array.isArray(value) ? value.join(" / ") : String(value);
  return `${parent} = ${valueLabel} → ${cond.then.join(", ")}`;
}

export const TunerLaunchFields: Component<{
  games: TunerGameInfo[];
  gamesLoading: boolean;
  game: string;
  onGameChange: (game: string) => void;
  nTrials: number;
  onNTrialsChange: (n: number) => void;
  /** Empty string means "auto" (server picks cpu_count // 2). */
  nWorkers: string;
  onNWorkersChange: (v: string) => void;
  deterministic: boolean;
  onDeterministicChange: (v: boolean) => void;
  seed: number;
  onSeedChange: (n: number) => void;
  /** Per-run MCTS iteration ceiling (`mcts_tune::SearchBudget::max_iterations`
   * on the Rust side) — how much compute *every* trial's candidate (and,
   * for a `baseline_config`-backed opponent, that opponent too) gets, not a
   * hyperparameter tuner searches over. Empty string means "unset" (use the
   * game binary's own historical default, `mcts-tune`'s `MAX_ITER`
   * constant) — forwarded as `target.max_iterations=N` only when set, same
   * convention as `nWorkers`'s "auto". */
  maxIterations: string;
  onMaxIterationsChange: (v: string) => void;
  /** Per-run wall-clock search budget in milliseconds, per move
   * (`mcts_tune::SearchBudget::max_time` on the Rust side), forwarded as
   * `target.max_time_ms=N` — mutually exclusive with `maxIterations`
   * (`game-host::run_tune_eval` rejects a `tune eval` invocation that sets
   * both `--max-iterations` and `--max-time-ms`). Empty string means
   * "unset", same convention as `maxIterations`. */
  maxTimeMs: string;
  onMaxTimeMsChange: (v: string) => void;
  /** Raw JSON text for the "Game config" field — only rendered when the
   * selected game's `tuner().game_config` isn't `{}`. */
  gameConfig: string;
  onGameConfigChange: (v: string) => void;
  /** Parse error for `gameConfig`, or `null` when it's valid JSON. */
  gameConfigError: string | null;
  minPairs: number;
  onMinPairsChange: (n: number) => void;
  maxPairs: number;
  onMaxPairsChange: (n: number) => void;
  pruningEnabled: boolean;
  onPruningEnabledChange: (v: boolean) => void;
  reductionFactor: number;
  onReductionFactorChange: (n: number) => void;
  pruningStartupTrials: number;
  onPruningStartupTrialsChange: (n: number) => void;
  /** Empty means that the resolved rating policy does not stop on sigma. */
  sigmaStop: string;
  onSigmaStopChange: (v: string) => void;
  tpeStartupTrials: number;
  onTpeStartupTrialsChange: (n: number) => void;
  validationError: string | null;
  disabled: boolean;
}> = (props) => {
  const currentTuner = createMemo(() => props.games.find((g) => g.game === props.game)?.tuner ?? null);

  return (
    <div id="tuner-launch-fields">
      <Show when={props.gamesLoading}>
        <div class="loading-bench">Loading tunable games…</div>
      </Show>

      <Show when={!props.gamesLoading && props.games.length === 0}>
        <div class="launch-empty">
          No game implements a tuner tuner yet — see <code>tuner()</code> on <code>GameAdapter</code>.
        </div>
      </Show>

      <Show when={props.games.length > 0}>
        <label>
          Game
          <select
            value={props.game}
            onChange={(e) => props.onGameChange(e.currentTarget.value)}
            disabled={props.disabled}
          >
            <option value="">— Select —</option>
            <For each={props.games}>{(g) => <option value={g.game}>{g.game}</option>}</For>
          </select>
        </label>

        <Show when={currentTuner()}>
          {(tuner) => (
            <div id="tuner-tuner-summary">
              <div class="tuner-tuner-meta">
                <span class="meta-label">Tuner</span>
                <span class="meta-value"><code>{tuner().id}</code></span>
              </div>

              <table id="tuner-param-table">
                <thead>
                  <tr>
                    <th>Parameter</th>
                    <th>Type</th>
                    <th>Range</th>
                    <th>Default</th>
                  </tr>
                </thead>
                <tbody>
                  <For each={tuner().parameters}>
                    {(p) => (
                      <tr>
                        <td class="tuner-param-name" title={p.name}>{p.name}</td>
                        <td class="tuner-param-type">{p.type}</td>
                        <td class="tuner-param-range" title={paramRange(p)}>{paramRange(p)}</td>
                        <td class="tuner-param-default" title={paramDefault(p)}>{paramDefault(p)}</td>
                      </tr>
                    )}
                  </For>
                </tbody>
              </table>

              <Show when={tuner().conditions.length > 0}>
                <details id="tuner-conditions">
                  <summary>Parameter conditions ({tuner().conditions.length})</summary>
                  <ul>
                    <For each={tuner().conditions}>{(c) => <li>{conditionLabel(c)}</li>}</For>
                  </ul>
                </details>
              </Show>

              <Show when={!isEmptyGameConfig(tuner().game_config)}>
                <label>
                  Game config
                  <textarea
                    id="tuner-game-config"
                    rows={4}
                    value={props.gameConfig}
                    onInput={(e) => props.onGameConfigChange(e.currentTarget.value)}
                    disabled={props.disabled}
                  />
                </label>
                <Show when={props.gameConfigError}>
                  <div class="launch-error">{props.gameConfigError}</div>
                </Show>
              </Show>
            </div>
          )}
        </Show>

        <Show when={props.game}>
          <div id="tuner-budget-fields">
            <label>
              Target trials
              <input
                type="number"
                min={1}
                value={props.nTrials}
                onInput={(e) => props.onNTrialsChange(Math.max(1, parseInt(e.currentTarget.value) || 1))}
                disabled={props.disabled}
              />
            </label>

            <label>
              Evaluation slots
              <input
                type="number"
                min={1}
                placeholder="auto"
                value={props.nWorkers}
                onInput={(e) => props.onNWorkersChange(e.currentTarget.value)}
                disabled={props.disabled}
              />
              <span class="tuner-field-hint">Concurrent pair evaluations, not MCTS search threads.</span>
            </label>

            <label>
              Seed
              <input
                type="number"
                value={props.seed}
                onInput={(e) => props.onSeedChange(parseInt(e.currentTarget.value) || 0)}
                disabled={props.disabled}
              />
            </label>

            <label>
              Minimum pairs
              <input
                type="number"
                min={1}
                value={props.minPairs}
                onInput={(e) => props.onMinPairsChange(Math.max(1, parseInt(e.currentTarget.value) || 1))}
                disabled={props.disabled}
              />
              <span class="tuner-field-hint">{props.minPairs * 2} physical games</span>
            </label>

            <label>
              Maximum pairs
              <input
                type="number"
                min={1}
                value={props.maxPairs}
                onInput={(e) => props.onMaxPairsChange(Math.max(1, parseInt(e.currentTarget.value) || 1))}
                disabled={props.disabled}
              />
              <span class="tuner-field-hint">{props.maxPairs * 2} physical games</span>
            </label>

            <label>
              TPE startup trials
              <input
                type="number"
                min={0}
                value={props.tpeStartupTrials}
                onInput={(e) => props.onTpeStartupTrialsChange(Math.max(0, parseInt(e.currentTarget.value) || 0))}
                disabled={props.disabled}
              />
            </label>

            <label>
              Sigma stop
              <input
                type="number"
                min={0}
                step="any"
                placeholder="disabled"
                value={props.sigmaStop}
                onInput={(e) => props.onSigmaStopChange(e.currentTarget.value)}
                disabled={props.disabled}
              />
              <span class="tuner-field-hint">Leave blank to use no uncertainty stop.</span>
            </label>

            <label class="tuner-checkbox-field">
              <input
                type="checkbox"
                checked={props.pruningEnabled}
                onChange={(e) => props.onPruningEnabledChange(e.currentTarget.checked)}
                disabled={props.disabled}
              />
              Enable Hyperband pruning
            </label>

            <Show when={props.pruningEnabled}>
              <label>
                Reduction factor
                <input
                  type="number"
                  min={2}
                  step="any"
                  value={props.reductionFactor}
                  onInput={(e) => props.onReductionFactorChange(parseFloat(e.currentTarget.value) || 2)}
                  disabled={props.disabled}
                />
              </label>

              <label>
                Pruning startup trials
                <input
                  type="number"
                  min={0}
                  value={props.pruningStartupTrials}
                  onInput={(e) => props.onPruningStartupTrialsChange(Math.max(0, parseInt(e.currentTarget.value) || 0))}
                  disabled={props.disabled}
                />
              </label>
            </Show>

            <Show when={props.validationError}>
              {(error) => <div class="launch-error">{error()}</div>}
            </Show>

            <label>
              Iteration budget
              <input
                type="number"
                min={1}
                placeholder="auto"
                value={props.maxIterations}
                onInput={(e) => props.onMaxIterationsChange(e.currentTarget.value)}
                disabled={props.disabled || props.maxTimeMs.trim() !== ""}
              />
              <span class="tuner-field-hint">
                MCTS iterations per move, applied to every trial's candidate (and its opponent, when
                self-play). Blank uses the game binary's own default -- this is a compute budget, not
                something tuner tunes for you.
              </span>
            </label>

            <label>
              Time budget (ms)
              <input
                type="number"
                min={1}
                placeholder="auto"
                value={props.maxTimeMs}
                onInput={(e) => props.onMaxTimeMsChange(e.currentTarget.value)}
                disabled={props.disabled || props.maxIterations.trim() !== ""}
              />
              <span class="tuner-field-hint">
                Wall-clock milliseconds per move instead of a fixed iteration count -- mutually
                exclusive with the iteration-budget field above (set one, leave the other blank).
              </span>
            </label>

            <label class="tuner-checkbox-field">
              <input
                type="checkbox"
                checked={props.deterministic}
                onChange={(e) => props.onDeterministicChange(e.currentTarget.checked)}
                disabled={props.disabled}
              />
              Deterministic (single seed per config)
            </label>
          </div>
        </Show>
      </Show>
    </div>
  );
};
