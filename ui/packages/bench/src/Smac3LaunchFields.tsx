// Smac3LaunchFields.tsx — The `smac3`-kind portion of the launch form.
//
// Unlike round_robin (pick strategies to pit against each other), a SMAC3
// run's "config" is an optimizer budget over a search space the *game*
// defines, not the launcher. So this renders:
//   - a game picker restricted to games that report tuner metadata
//     (`GET /api/bench/smac3/kinds` — only games with a `tuner()` impl
//     appear at all, so there's nothing to disable/grey out here);
//   - a read-only summary of that game's search space (parameters,
//     conditions, baselines, the tuner's *default* eval rounds/trial) so the
//     operator can see what a trial actually varies before committing a
//     budget to it;
//   - the actual editable fields: n_trials/n_workers/deterministic/seed and
//     rounds/trial (`optimizer.*`/`target.rounds` overrides). The parameter
//     *values* themselves aren't editable here — `smac3`'s CLI `--override`
//     only reaches dotted dataclass attributes, not the list-shaped
//     `parameters:` search space, so there is nothing for a form field to
//     write to;
//   - a raw-JSON "Game config" textarea, only for a game whose `tuner().
//     game_config` isn't `{}` (today, only Druid) — a game-setup axis (e.g.
//     board size) separate from the strategy search space above: SMAC3
//     never searches over it, it just pins every trial in the run to it.
//     Deliberately a generic JSON field rather than a typed per-game picker
//     (e.g. a board-size dropdown) so a future game with its own config
//     needs no new UI code here.
//
// LaunchForm owns all of this component's state (same lifted-state
// convention as the strategy picker) and builds the `--override` argv from
// it at submit time — see `buildSmac3Overrides` there.

import { createMemo, For, Show, type Component } from "solid-js";
import type { Smac3GameInfo, TunerParameter } from "./index.js";

/** Baseline-only families (`mcts-tune`'s `make_candidate`) that exist purely
 * as ladder floor rungs -- never in any game's `tuner().baselines` (those
 * are named presets), never SMAC3-searchable, but always launchable as the
 * starting opponent via `target.baselines=[...]`. See `mcts-tune/src/
 * lib.rs`'s `"random"`/`"flat_mc"` match arms. */
export const FLOOR_BASELINES = ["flat_mc", "random"] as const;

/** Whether a `tuner().game_config` value means "nothing to configure" --
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
 * text (not truncated) -- callers that display this in a narrow column
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
function conditionLabel(cond: Smac3GameInfo["tuner"]["conditions"][number]): string {
  const [parent, value] = Object.entries(cond.if)[0] ?? ["?", "?"];
  const valueLabel = Array.isArray(value) ? value.join(" / ") : String(value);
  return `${parent} = ${valueLabel} → ${cond.then.join(", ")}`;
}

export const Smac3LaunchFields: Component<{
  games: Smac3GameInfo[];
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
  rounds: number;
  onRoundsChange: (n: number) => void;
  /** Raw JSON text for the "Game config" field -- only rendered when the
   * selected game's `tuner().game_config` isn't `{}`. */
  gameConfig: string;
  onGameConfigChange: (v: string) => void;
  /** Parse error for `gameConfig`, or `null` when it's valid JSON. */
  gameConfigError: string | null;
  /** Which opponent a fresh run's root rung starts against -- one of the
   * selected game's own named presets (`tuner().baselines`) or a floor
   * family (`FLOOR_BASELINES`). Forwarded as a `target.baselines=[...]`
   * override at submit time (see `LaunchForm.tsx`'s `buildSmac3Overrides`)
   * rather than defaulting to the tuner's full baseline list. */
  startingBaseline: string;
  onStartingBaselineChange: (v: string) => void;
  /** Whether this launch opts into the automated ladder driver
   * (`server/src/bench/mod.rs`'s `plan_ladder_advances`) -- when on, the
   * run's `config.ladder = {max_rungs, saturation_threshold}` is set, and
   * once this rung saturates, the driver stops it and relaunches facing
   * only its own incumbent (see `replace_baseline_with_incumbent`). */
  ladderEnabled: boolean;
  onLadderEnabledChange: (v: boolean) => void;
  maxRungs: number;
  onMaxRungsChange: (n: number) => void;
  saturationThreshold: number;
  onSaturationThresholdChange: (n: number) => void;
  disabled: boolean;
}> = (props) => {
  const currentTuner = createMemo(() => props.games.find((g) => g.game === props.game)?.tuner ?? null);

  return (
    <div id="smac3-launch-fields">
      <Show when={props.gamesLoading}>
        <div class="loading-bench">Loading tunable games…</div>
      </Show>

      <Show when={!props.gamesLoading && props.games.length === 0}>
        <div class="launch-empty">
          No game implements a SMAC3 tuner yet — see <code>tuner()</code> on <code>GameAdapter</code>.
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
            <div id="smac3-tuner-summary">
              <div class="smac3-tuner-meta">
                <span class="meta-label">Tuner</span>
                <span class="meta-value"><code>{tuner().id}</code></span>
                <span class="meta-label">
                  {tuner().baselines.length > 1 ? "Named presets" : "Named preset"}
                </span>
                <span class="meta-value">{tuner().baselines.join(", ")}</span>
              </div>
              {/* This is metadata about the game's tuner, not what the run
                  actually starts against -- that's the "Starting baseline"
                  selector below. Listing every named preset here just
                  documents what's available to pick from. */}
              <label>
                Starting baseline
                <select
                  value={props.startingBaseline}
                  onChange={(e) => props.onStartingBaselineChange(e.currentTarget.value)}
                  disabled={props.disabled}
                >
                  <For each={tuner().baselines}>{(b) => <option value={b}>{b}</option>}</For>
                  <For each={FLOOR_BASELINES}>{(b) => <option value={b}>{b} (floor)</option>}</For>
                </select>
              </label>

              <table id="smac3-param-table">
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
                        <td class="smac3-param-name" title={p.name}>{p.name}</td>
                        <td class="smac3-param-type">{p.type}</td>
                        <td class="smac3-param-range" title={paramRange(p)}>{paramRange(p)}</td>
                        <td class="smac3-param-default" title={paramDefault(p)}>{paramDefault(p)}</td>
                      </tr>
                    )}
                  </For>
                </tbody>
              </table>

              {/* Collapsible (open by default) rather than an always-expanded
                  list -- a family parameter's condition can read like
                  "family = ucb1 / ucb1_dm / ucb1_mast / ... → final_action"
                  and a handful of those eat a lot of vertical space in a
                  ~380px-wide sidebar for something most launches don't need
                  to re-check every time. */}
              <Show when={tuner().conditions.length > 0}>
                <details id="smac3-conditions">
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
                    id="smac3-game-config"
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
          <div id="smac3-budget-fields">
            <label>
              Trials
              <input
                type="number"
                min={1}
                value={props.nTrials}
                onInput={(e) => props.onNTrialsChange(Math.max(1, parseInt(e.currentTarget.value) || 1))}
                disabled={props.disabled}
              />
            </label>

            <label>
              Workers
              <input
                type="number"
                min={1}
                placeholder="auto"
                value={props.nWorkers}
                onInput={(e) => props.onNWorkersChange(e.currentTarget.value)}
                disabled={props.disabled}
              />
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
              Rounds/trial
              <input
                type="number"
                min={1}
                value={props.rounds}
                onInput={(e) => props.onRoundsChange(Math.max(1, parseInt(e.currentTarget.value) || 1))}
                disabled={props.disabled}
              />
            </label>

            <label class="smac3-checkbox-field">
              <input
                type="checkbox"
                checked={props.deterministic}
                onChange={(e) => props.onDeterministicChange(e.currentTarget.checked)}
                disabled={props.disabled}
              />
              Deterministic (single seed per config)
            </label>
          </div>

          <div id="smac3-ladder-fields">
            <label class="smac3-checkbox-field">
              <input
                type="checkbox"
                checked={props.ladderEnabled}
                onChange={(e) => props.onLadderEnabledChange(e.currentTarget.checked)}
                disabled={props.disabled}
              />
              Ladder (auto-widen to face own incumbent once saturated)
            </label>

            <Show when={props.ladderEnabled}>
              <label>
                Max rungs
                <input
                  type="number"
                  min={1}
                  value={props.maxRungs}
                  onInput={(e) => props.onMaxRungsChange(Math.max(1, parseInt(e.currentTarget.value) || 1))}
                  disabled={props.disabled}
                />
              </label>

              <label>
                Saturation threshold (max incumbent loss rate, 0–1)
                <input
                  type="number"
                  min={0}
                  max={1}
                  step={0.05}
                  value={props.saturationThreshold}
                  onInput={(e) => props.onSaturationThresholdChange(parseFloat(e.currentTarget.value) || 0)}
                  disabled={props.disabled}
                />
                <span class="smac3-field-hint">
                  Widens once the incumbent's loss rate against the current baseline is at or below
                  this (0 = must go undefeated; 0.1 = widen once losing at most 10% of games) -- a
                  fraction of games lost, not a percent-complete or 0–100 scale.
                </span>
              </label>
            </Show>
          </div>
        </Show>
      </Show>
    </div>
  );
};
