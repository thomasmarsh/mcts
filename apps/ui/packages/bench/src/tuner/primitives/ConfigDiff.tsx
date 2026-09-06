// ConfigDiff — a parameter table comparing the schema default against a
// candidate's canonical config, changed rows highlighted, with an
// only-changed toggle. All flattening/pairing lives in
// `config-diff-model.ts`; this is layout only.

import { createSignal, For, Show, type Component } from "solid-js";
import type { JsonValue } from "../../types.js";
import { configDiffRows } from "../models/config-diff-model.js";

export interface ConfigDiffProps {
  /** Schema-default config as a flat path→string map (see `schemaDefaults`). */
  base: Record<string, string>;
  candidate: JsonValue | null;
  /** Start with only the changed rows shown (default true). */
  onlyChangedDefault?: boolean;
}

export const ConfigDiff: Component<ConfigDiffProps> = (props) => {
  const [onlyChanged, setOnlyChanged] = createSignal(props.onlyChangedDefault ?? true);
  const rows = () => configDiffRows(props.base, props.candidate);
  const shown = () => (onlyChanged() ? rows().filter((r) => r.changed) : rows());
  return (
    <div class="tuner-config-diff" data-testid="config-diff">
      <label class="tuner-config-diff-toggle">
        <input
          type="checkbox"
          checked={onlyChanged()}
          onChange={(e) => setOnlyChanged(e.currentTarget.checked)}
        />
        Only changed
      </label>
      <Show
        when={shown().length > 0}
        fallback={<p class="tuner-fleet-empty">No parameters differ from the default.</p>}
      >
        <table class="tuner-table">
          <thead>
            <tr>
              <th>Parameter</th>
              <th>Default</th>
              <th>Candidate</th>
            </tr>
          </thead>
          <tbody>
            <For each={shown()}>
              {(row) => (
                <tr classList={{ "tuner-config-diff-changed": row.changed }}>
                  <td>{row.path}</td>
                  <td>{row.base ?? "—"}</td>
                  <td>{row.candidate ?? "—"}</td>
                </tr>
              )}
            </For>
          </tbody>
        </table>
      </Show>
    </div>
  );
};
