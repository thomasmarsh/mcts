// ConstraintEditor — the schema-driven replacement for the launch form's
// free-text "Constrain parameters" textarea. One row per tunable parameter,
// grouped by the axis that gates it. Each row picks a narrowing mode
// (`free` / `fix` / `range` / `choices`) whose inputs are bounded by the
// schema, so an out-of-domain constraint can't be expressed. An optional
// `when` predicate scopes a row to specific values of an upstream categorical.
//
// Controlled: the parent owns the `rows` state and runs `deriveConstraints`
// to get the wire form. This component only renders and edits rows, and shows
// the same validation errors inline. Not yet wired into `LaunchForm`.

import { For, Show, type Component } from "solid-js";
import type { TunerParameter } from "../../types.js";
import {
  axisGroups,
  deriveConstraints,
  emptyRow,
  modesFor,
  predicateParents,
  type ConstraintMode,
  type ConstraintRow,
  type ConstraintRows,
  type ParamSchema,
} from "../models/constraint-editor-model.js";

function schemaHint(p: TunerParameter): string {
  if (p.type === "constant") return `constant ${String(p.value)}`;
  if (Array.isArray(p.bounds)) {
    const dflt = p.default === undefined ? "" : `, default ${String(p.default)}`;
    return `${p.type} ${p.bounds[0]}–${p.bounds[1]}${dflt}`;
  }
  if (Array.isArray(p.choices)) return `one of ${p.choices.join(", ")}`;
  return p.type;
}

export const ConstraintEditor: Component<{
  schema: ParamSchema;
  rows: ConstraintRows;
  onChange: (rows: ConstraintRows) => void;
}> = (props) => {
  const groups = () => axisGroups(props.schema);
  const rowErrors = () => {
    const map = new Map<string, string>();
    for (const err of deriveConstraints(props.schema, props.rows).errors) {
      const [name, ...rest] = err.split(": ");
      map.set(name!, rest.join(": "));
    }
    return map;
  };

  const rowFor = (name: string): ConstraintRow => {
    const existing = props.rows[name];
    if (existing) return existing;
    const spec = props.schema.parameters.find((p) => p.name === name)!;
    return emptyRow(spec);
  };

  function patch(name: string, next: Partial<ConstraintRow>): void {
    props.onChange({ ...props.rows, [name]: { ...rowFor(name), ...next } });
  }

  function toggleChoice(name: string, choice: string, keep: boolean): void {
    const row = rowFor(name);
    patch(name, {
      retained: keep
        ? [...new Set([...row.retained, choice])]
        : row.retained.filter((c) => c !== choice),
    });
  }

  function toggleWhenValue(name: string, parent: string, value: string, on: boolean): void {
    const row = rowFor(name);
    const current = row.when[parent] ?? [];
    const nextValues = on ? [...new Set([...current, value])] : current.filter((v) => v !== value);
    const when = { ...row.when };
    if (nextValues.length > 0) when[parent] = nextValues;
    else delete when[parent];
    patch(name, { when });
  }

  return (
    <div class="constraint-editor" data-testid="constraint-editor">
      <For each={groups()}>
        {(group) => (
          <fieldset class="constraint-editor-group">
            <legend>{group.axis ?? "algorithm & axes"}</legend>
            <For each={group.parameters}>
              {(param) => {
                const row = () => rowFor(param.name);
                const modes = modesFor(param);
                const parents = () => predicateParents(props.schema, param.name);
                return (
                  <div class="constraint-editor-row" data-testid={`constraint-row-${param.name}`}>
                    <div class="constraint-editor-head">
                      <code>{param.name}</code>
                      <Show when={modes.length > 1} fallback={<span class="tuner-launch-hint">not tunable</span>}>
                        <select
                          data-testid={`constraint-mode-${param.name}`}
                          value={row().mode}
                          onInput={(e) =>
                            patch(param.name, { mode: e.currentTarget.value as ConstraintMode })
                          }
                        >
                          <For each={modes}>{(m) => <option value={m}>{m}</option>}</For>
                        </select>
                      </Show>
                      <span class="tuner-launch-hint">{schemaHint(param)}</span>
                    </div>

                    <Show when={row().mode === "fix"}>
                      <Show
                        when={Array.isArray(param.choices)}
                        fallback={
                          <input
                            type="number"
                            data-testid={`constraint-fix-${param.name}`}
                            value={row().fix}
                            onInput={(e) => patch(param.name, { fix: e.currentTarget.value })}
                          />
                        }
                      >
                        <select
                          data-testid={`constraint-fix-${param.name}`}
                          value={row().fix}
                          onInput={(e) => patch(param.name, { fix: e.currentTarget.value })}
                        >
                          <option value="">choose…</option>
                          <For each={param.choices}>{(c) => <option value={c}>{c}</option>}</For>
                        </select>
                      </Show>
                    </Show>

                    <Show when={row().mode === "range"}>
                      <div class="constraint-editor-range">
                        <input
                          type="number"
                          data-testid={`constraint-low-${param.name}`}
                          value={row().low}
                          onInput={(e) => patch(param.name, { low: e.currentTarget.value })}
                        />
                        <span>–</span>
                        <input
                          type="number"
                          data-testid={`constraint-high-${param.name}`}
                          value={row().high}
                          onInput={(e) => patch(param.name, { high: e.currentTarget.value })}
                        />
                      </div>
                    </Show>

                    <Show when={row().mode === "choices" && Array.isArray(param.choices)}>
                      <div class="constraint-editor-choices">
                        <For each={param.choices}>
                          {(choice) => (
                            <label>
                              <input
                                type="checkbox"
                                data-testid={`constraint-choice-${param.name}-${choice}`}
                                checked={row().retained.includes(choice)}
                                onChange={(e) =>
                                  toggleChoice(param.name, choice, e.currentTarget.checked)
                                }
                              />
                              {choice}
                            </label>
                          )}
                        </For>
                      </div>
                    </Show>

                    <Show when={row().mode !== "free" && parents().length > 0}>
                      <details class="constraint-editor-when">
                        <summary>only when…</summary>
                        <For each={parents()}>
                          {(parent) => (
                            <div class="constraint-editor-when-parent">
                              <code>{parent.name}</code>
                              <For each={parent.choices}>
                                {(value) => (
                                  <label>
                                    <input
                                      type="checkbox"
                                      data-testid={`constraint-when-${param.name}-${parent.name}-${value}`}
                                      checked={(row().when[parent.name] ?? []).includes(value)}
                                      onChange={(e) =>
                                        toggleWhenValue(
                                          param.name,
                                          parent.name,
                                          value,
                                          e.currentTarget.checked,
                                        )
                                      }
                                    />
                                    {value}
                                  </label>
                                )}
                              </For>
                            </div>
                          )}
                        </For>
                      </details>
                    </Show>

                    <Show when={rowErrors().get(param.name)}>
                      <p
                        class="launch-error"
                        role="alert"
                        data-testid={`constraint-error-${param.name}`}
                      >
                        {rowErrors().get(param.name)}
                      </p>
                    </Show>
                  </div>
                );
              }}
            </For>
          </fieldset>
        )}
      </For>
    </div>
  );
};
