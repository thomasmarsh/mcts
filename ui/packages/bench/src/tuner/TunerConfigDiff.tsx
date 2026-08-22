/** TunerConfigDiff — parameter comparison tables showing how the
 * incumbent's config (and/or the lowest single trial's config) differs
 * from the baseline the run started against.
 *
 * Two tables are rendered when both are available: an emphasized one for
 * the incumbent (the config "Use best as new baseline" actually promotes)
 * and a plain one for the lowest single trial (this chart's cheapest dot,
 * for reference).
 *
 * Only rendered when the run launched with exactly one baseline instance
 * (comparison is meaningless against a multi-instance panel). */

import { For, Show, type Component } from "solid-js";
import type { IncumbentInfo, TrialRow } from "../index.js";

// ── Types ─────────────────────────────────────────────────────────

interface ParamDiffRow {
  name: string;
  value: unknown;
  baseline: unknown;
  changed: boolean;
}

// ── Helpers ───────────────────────────────────────────────────────

/** Diff a config's active parameters against the actual baseline config
 * used for this run. */
function paramsVsBaseline(
  config: Record<string, unknown> | undefined,
  baseline: Record<string, unknown> | null,
): ParamDiffRow[] {
  if (!config || !baseline) return [];
  return Object.entries(config).map(([name, value]) => {
    const baselineValue = baseline[name];
    const changed =
      baselineValue !== undefined &&
      JSON.stringify(baselineValue) !== JSON.stringify(value);
    return { name, value, baseline: baselineValue, changed };
  });
}

/** Extract the baseline config from the launch record.
 *
 * The launch record stores the resolved parameter settings for each
 * baseline instance. A promoted rung also keeps the raw setting under
 * `baseline_configs`. A comparison only makes sense with exactly one
 * baseline instance. */
function baselineConfig(
  launchConfig: unknown,
): Record<string, unknown> | null {
  const config = launchConfig as {
    baseline_settings?: unknown;
    baseline_configs?: unknown;
    overrides?: unknown;
  } | null;
  const settings = config?.baseline_settings ?? config?.baseline_configs;
  if (settings && typeof settings === "object" && !Array.isArray(settings)) {
    const entries = Object.values(settings as Record<string, unknown>);
    if (entries.length !== 1) return null;
    const baseline = entries[0];
    return baseline && typeof baseline === "object" && !Array.isArray(baseline)
      ? (baseline as Record<string, unknown>)
      : null;
  }

  // Runs launched before `baseline_settings` was persisted still identify
  // their floor opponent in the launch override.
  if (!Array.isArray(config?.overrides)) return null;
  let baselineOverride: string | undefined;
  for (let i = config.overrides.length - 1; i >= 0; i--) {
    const value = config.overrides[i];
    if (typeof value === "string" && value.startsWith("target.baselines=")) {
      baselineOverride = value;
      break;
    }
  }
  const match = baselineOverride?.match(
    /^target\.baselines=\[(['"])(flat_mc|random)\1\]$/,
  );
  if (!match) return null;
  return { family: match[2], q_init: "Infinity" };
}

// ── Diff table sub-component ──────────────────────────────────────

const DiffTable: Component<{
  id: string;
  caption: string;
  emphasized: boolean;
  rows: ParamDiffRow[];
}> = (props) => {
  if (props.rows.length === 0) return null;
  return (
    <table
      id={props.id}
      classList={{
        "tuner-diff-table": true,
        "tuner-diff-table-emphasized": props.emphasized,
      }}
    >
      <caption>{props.caption}</caption>
      <thead>
        <tr>
          <th>Parameter</th>
          <th>Value</th>
          <th>Baseline</th>
        </tr>
      </thead>
      <tbody>
        <For each={props.rows}>
          {(row) => (
            <tr classList={{ "tuner-diff-changed": row.changed }}>
              <td class="tuner-param-name">{row.name}</td>
              <td>{String(row.value)}</td>
              <td>
                {row.baseline === undefined ? "—" : String(row.baseline)}
              </td>
            </tr>
          )}
        </For>
      </tbody>
    </table>
  );
};

// ── Main component ────────────────────────────────────────────────

export const TunerConfigDiff: Component<{
  incumbent: IncumbentInfo | null;
  bestTrial: TrialRow | null;
  launchConfig?: unknown;
}> = (props) => {
  const currentBaseline = baselineConfig(props.launchConfig);

  const incumbentRows = () =>
    paramsVsBaseline(props.incumbent?.config, currentBaseline);

  const lowestTrialRows = () =>
    paramsVsBaseline(props.bestTrial?.config, currentBaseline);

  return (
    <>
      <Show when={incumbentRows().length > 0}>
        <DiffTable
          id="tuner-incumbent-diff-table"
          caption="Incumbent vs. baseline"
          emphasized={true}
          rows={incumbentRows()}
        />
      </Show>

      <Show when={lowestTrialRows().length > 0}>
        <DiffTable
          id="tuner-lowest-trial-diff-table"
          caption={`Lowest single trial (#${props.bestTrial!.trial_id}) vs. baseline`}
          emphasized={false}
          rows={lowestTrialRows()}
        />
      </Show>
    </>
  );
};