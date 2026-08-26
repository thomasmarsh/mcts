import { createEffect, createMemo, createSignal, For, Show, type Component } from "solid-js";
import type { Store } from "@mcts/core";
import type { BenchAction, BenchState } from "./index.js";
import { expandExperimentSpec } from "./experiment-grid.js";
import type { ExperimentSpecV1 } from "./types.js";

type JsonKey = string;
const prettyJson = (value: unknown): string => JSON.stringify(value ?? null, null, 2);
const friendlyStatus = (status: string): string => status.replaceAll("_", " ");

export const ExperimentEditor: Component<{ store: Store<BenchState, BenchAction> }> = (props) => {
  const state = props.store.getState();
  const dispatch = props.store.dispatch;
  const draft = createMemo(() => state().experimentDraft);
  const [jsonText, setJsonText] = createSignal<Record<JsonKey, string>>({});
  const [jsonErrors, setJsonErrors] = createSignal<Record<JsonKey, string>>({});
  let loadedKey = "";

  createEffect(() => {
    const current = draft();
    const key = current
      ? `${state().selectedExperimentId ?? "new"}:${JSON.stringify(current.spec)}`
      : "empty";
    if (key === loadedKey) return;
    loadedKey = key;
    const next: Record<string, string> = {
      baseline: current ? prettyJson(current.spec.baseline.config) : "",
    };
    current?.spec.games.forEach((game, index) => {
      next[`game-${index}`] = prettyJson(game.game_config);
    });
    current?.spec.variants.forEach((variant, index) => {
      next[`variant-${index}`] = prettyJson(variant.config);
    });
    setJsonText(next);
    setJsonErrors({});
  });

  const update = (change: (spec: ExperimentSpecV1) => void) => {
    const current = draft();
    if (!current) return;
    const spec = structuredClone(current.spec);
    change(spec);
    dispatch({ tag: "experimentDraft", draft: { ...current, spec } });
  };
  const editJson = (
    key: JsonKey,
    text: string,
    apply: (spec: ExperimentSpecV1, value: unknown) => void,
    objectOnly: boolean,
  ) => {
    setJsonText((old) => ({ ...old, [key]: text }));
    try {
      const value: unknown = JSON.parse(text);
      if (objectOnly && (!value || typeof value !== "object" || Array.isArray(value)))
        throw new Error("Strategy configuration must be a JSON object.");
      setJsonErrors((old) => {
        const next = { ...old };
        delete next[key];
        return next;
      });
      update((spec) => apply(spec, value));
    } catch (error) {
      setJsonErrors((old) => ({
        ...old,
        [key]:
          error instanceof SyntaxError
            ? "Enter valid JSON."
            : error instanceof Error
              ? error.message
              : "Enter valid JSON.",
      }));
    }
  };
  const gameOptions = createMemo(() => state().tunerKinds.result ?? []);
  const dirty = () => {
    const current = draft();
    return (
      current !== null && JSON.stringify(current) !== JSON.stringify(state().experimentSavedDraft)
    );
  };
  const localInvalid = () => Object.keys(jsonErrors()).length > 0;
  const fieldError = (path: string) => state().experimentFieldErrors[path];
  const errorFor = (path: string, key?: string) =>
    fieldError(path) ?? (key ? jsonErrors()[key] : undefined);
  const errorId = (key: string) => `experiment-field-error-${key.replace(/[^a-zA-Z0-9_-]/g, "-")}`;
  const fieldAttrs = (error: string | undefined, key: string) => ({
    "aria-invalid": (error ? "true" : undefined) as "true" | undefined,
    "aria-describedby": error ? errorId(key) : undefined,
  });
  const FieldError: Component<{ error: string | undefined; id: string }> = (props) => (
    <Show when={props.error}>
      <span id={props.id} class="projects-field-error">
        {props.error}
      </span>
    </Show>
  );
  const saveDisabled = () => state().experimentSaveStatus === "saving" || localInvalid();
  const launchDisabled = () =>
    !state().selectedExperimentId ||
    state().experimentSaveStatus === "saving" ||
    state().experimentLaunchStatus === "launching" ||
    dirty() ||
    localInvalid();
  const preview = createMemo(() => {
    const current = draft();
    return current ? expandExperimentSpec(current.spec) : null;
  });
  const budgetSummary = (kind: string, value: number) =>
    kind === "iterations" ? `${value} iterations` : `${value} ms per move`;
  const nextGame = () =>
    gameOptions().find(
      (item) => !(draft()?.spec.games ?? []).some((game) => game.game === item.game),
    );

  return (
    <Show
      when={draft()}
      fallback={
        <div
          class="projects-state projects-state-error"
          role={state().experimentError ? "alert" : undefined}
        >
          <span class="projects-state-title">
            {state().experimentError
              ? "Experiment could not be loaded"
              : "Select or create an experiment"}
          </span>
          <span>
            {state().experimentError ?? "Return to the project to choose a saved definition."}
          </span>
        </div>
      }
    >
      <form class="projects-editor" onSubmit={(event) => event.preventDefault()}>
        <header class="projects-page-header projects-editor-header">
          <div>
            <button
              class="projects-back-link"
              type="button"
              onClick={() =>
                dispatch({ tag: "openProject", projectId: state().selectedProjectId ?? "" })
              }
            >
              ← Project
            </button>
            <p class="projects-eyebrow">
              {state().selectedExperimentId ? "Saved experiment" : "New experiment"}
            </p>
            <h1>{draft()?.name || "Untitled experiment"}</h1>
            <p class="projects-lede">
              Define a repeatable grid of candidate-versus-baseline cells and launch an exact saved
              snapshot.
            </p>
          </div>
        </header>
        <section class="projects-panel" aria-labelledby="identity-heading">
          <div class="projects-panel-heading">
            <div>
              <h2 id="identity-heading">Identity</h2>
              <p>Name the experiment and record the question it is meant to answer.</p>
            </div>
          </div>
          <div class="projects-form-grid projects-form-grid-two">
            <div class="projects-field">
              <label for="experiment-name">Experiment name</label>
              <input
                id="experiment-name"
                value={draft()?.name}
                onInput={(e) =>
                  dispatch({
                    tag: "experimentDraft",
                    draft: { ...draft()!, name: e.currentTarget.value },
                  })
                }
              />{" "}
              <Show when={fieldError("name")}>
                <span class="projects-field-error">{fieldError("name")}</span>
              </Show>
            </div>
            <div class="projects-field">
              <label for="experiment-description">
                Description <span class="projects-optional">Optional</span>
              </label>
              <textarea
                id="experiment-description"
                rows="2"
                value={draft()?.description}
                onInput={(e) =>
                  dispatch({
                    tag: "experimentDraft",
                    draft: { ...draft()!, description: e.currentTarget.value },
                  })
                }
              />
            </div>
          </div>
        </section>

        <section class="projects-panel" aria-labelledby="game-heading">
          <div class="projects-panel-heading">
            <div>
              <h2 id="game-heading">Games</h2>
              <p>Every game is expanded against every budget and variant.</p>
            </div>
            <button
              class="projects-button projects-button-secondary"
              type="button"
              disabled={!nextGame()}
              onClick={() => {
                const item = nextGame();
                if (item)
                  dispatch({
                    tag: "experimentGameAdded",
                    game: item.game,
                    gameConfig: JSON.parse(JSON.stringify(item.tuner.game_config ?? null)),
                  });
              }}
            >
              Add game
            </button>
          </div>
          <For each={draft()?.spec.games ?? []}>
            {(game, index) => {
              const gameError = fieldError(`spec.games[${index()}].game`);
              const configError = errorFor(`spec.games[${index()}].game_config`, `game-${index()}`);
              const applyGameConfig = (text: string) =>
                editJson(
                  `game-${index()}`,
                  text,
                  (spec, value) => {
                    spec.games[index()]!.game_config = value;
                  },
                  false,
                );
              return (
                <fieldset class="projects-grid-item">
                  <legend>Game {index() + 1}</legend>
                  <div class="projects-form-grid projects-form-grid-two">
                    <div class="projects-field">
                      <label for={`game-${index()}`}>Game</label>
                      <select
                        id={`game-${index()}`}
                        value={game.game}
                        {...fieldAttrs(gameError, `game-${index()}`)}
                        onChange={(e) => {
                          const item = gameOptions().find(
                            (candidate) => candidate.game === e.currentTarget.value,
                          );
                          const config = item?.tuner.game_config ?? null;
                          dispatch({
                            tag: "experimentGameEdited",
                            index: index(),
                            game: e.currentTarget.value,
                            gameConfig: JSON.parse(JSON.stringify(config)),
                          });
                        }}
                      >
                        <option value={game.game}>{game.game}</option>
                        <For
                          each={gameOptions().filter(
                            (item) =>
                              item.game !== game.game &&
                              !(draft()?.spec.games ?? []).some(
                                (other, otherIndex) =>
                                  otherIndex !== index() && other.game === item.game,
                              ),
                          )}
                        >
                          {(item) => <option value={item.game}>{item.game}</option>}
                        </For>
                      </select>
                      <FieldError error={gameError} id={errorId(`game-${index()}`)} />
                    </div>
                    <div class="projects-field">
                      <label for={`game-config-${index()}`}>Game configuration</label>
                      <textarea
                        id={`game-config-${index()}`}
                        class="projects-json-editor"
                        rows="5"
                        {...fieldAttrs(configError, `game-config-${index()}`)}
                        value={jsonText()[`game-${index()}`] ?? prettyJson(game.game_config)}
                        onInput={(e) => applyGameConfig(e.currentTarget.value)}
                        onChange={(e) => applyGameConfig(e.currentTarget.value)}
                      />
                      <FieldError error={configError} id={errorId(`game-config-${index()}`)} />
                    </div>
                  </div>
                  <button
                    class="projects-button projects-button-secondary"
                    type="button"
                    aria-label={`Remove game ${index() + 1}`}
                    disabled={(draft()?.spec.games.length ?? 0) <= 1}
                    onClick={() => dispatch({ tag: "experimentGameRemoved", index: index() })}
                  >
                    Remove game
                  </button>
                </fieldset>
              );
            }}
          </For>
        </section>

        <section class="projects-panel" aria-labelledby="comparison-heading">
          <div class="projects-panel-heading">
            <div>
              <h2 id="comparison-heading">Comparison</h2>
              <p>
                The baseline is shared by every cell; each variant creates another candidate row.
              </p>
            </div>
          </div>
          <fieldset class="projects-strategy-panel">
            <legend>Baseline</legend>
            <div class="projects-form-grid projects-form-grid-two">
              <div class="projects-field">
                <label for="baseline-id">Strategy ID</label>
                <input
                  id="baseline-id"
                  {...fieldAttrs(fieldError("spec.baseline.id"), "baseline-id")}
                  value={draft()?.spec.baseline.id}
                  onInput={(e) =>
                    update((spec) => {
                      spec.baseline.id = e.currentTarget.value;
                    })
                  }
                />
                <FieldError error={fieldError("spec.baseline.id")} id={errorId("baseline-id")} />
              </div>
              <div class="projects-field">
                <label for="baseline-label">Label</label>
                <input
                  id="baseline-label"
                  {...fieldAttrs(fieldError("spec.baseline.label"), "baseline-label")}
                  value={draft()?.spec.baseline.label}
                  onInput={(e) =>
                    update((spec) => {
                      spec.baseline.label = e.currentTarget.value;
                    })
                  }
                />
                <FieldError
                  error={fieldError("spec.baseline.label")}
                  id={errorId("baseline-label")}
                />
              </div>
            </div>
            <div class="projects-field">
              <label for="baseline-config">Raw strategy JSON</label>
              <textarea
                id="baseline-config"
                class="projects-json-editor"
                rows="6"
                {...fieldAttrs(errorFor("spec.baseline.config", "baseline"), "baseline-config")}
                value={jsonText().baseline ?? prettyJson(draft()?.spec.baseline.config)}
                onInput={(e) =>
                  editJson(
                    "baseline",
                    e.currentTarget.value,
                    (spec, value) => {
                      spec.baseline.config = value as Record<string, unknown>;
                    },
                    true,
                  )
                }
              />
              <FieldError
                error={errorFor("spec.baseline.config", "baseline")}
                id={errorId("baseline-config")}
              />
            </div>
          </fieldset>
          <For each={draft()?.spec.variants ?? []}>
            {(variant, index) => {
              const idError = fieldError(`spec.variants[${index()}].id`);
              const labelError = fieldError(`spec.variants[${index()}].label`);
              const configError = errorFor(
                `spec.variants[${index()}].config`,
                `variant-${index()}`,
              );
              const applyVariantConfig = (text: string) =>
                editJson(
                  `variant-${index()}`,
                  text,
                  (spec, value) => {
                    spec.variants[index()]!.config = value as Record<string, unknown>;
                  },
                  true,
                );
              return (
                <fieldset class="projects-strategy-panel">
                  <legend>Variant {index() + 1}</legend>
                  <div class="projects-form-grid projects-form-grid-two">
                    <div class="projects-field">
                      <label for={`variant-id-${index()}`}>Strategy ID</label>
                      <input
                        id={`variant-id-${index()}`}
                        {...fieldAttrs(idError, `variant-id-${index()}`)}
                        value={variant.id}
                        onInput={(e) =>
                          dispatch({
                            tag: "experimentVariantEdited",
                            index: index(),
                            field: "id",
                            value: e.currentTarget.value,
                          })
                        }
                      />
                      <FieldError error={idError} id={errorId(`variant-id-${index()}`)} />
                    </div>
                    <div class="projects-field">
                      <label for={`variant-label-${index()}`}>Label</label>
                      <input
                        id={`variant-label-${index()}`}
                        {...fieldAttrs(labelError, `variant-label-${index()}`)}
                        value={variant.label}
                        onInput={(e) =>
                          dispatch({
                            tag: "experimentVariantEdited",
                            index: index(),
                            field: "label",
                            value: e.currentTarget.value,
                          })
                        }
                      />
                      <FieldError error={labelError} id={errorId(`variant-label-${index()}`)} />
                    </div>
                  </div>
                  <div class="projects-field">
                    <label for={`variant-config-${index()}`}>Raw strategy JSON</label>
                    <textarea
                      id={`variant-config-${index()}`}
                      class="projects-json-editor"
                      rows="6"
                      {...fieldAttrs(configError, `variant-config-${index()}`)}
                      value={jsonText()[`variant-${index()}`] ?? prettyJson(variant.config)}
                      onInput={(e) => applyVariantConfig(e.currentTarget.value)}
                      onChange={(e) => applyVariantConfig(e.currentTarget.value)}
                    />
                    <FieldError error={configError} id={errorId(`variant-config-${index()}`)} />
                  </div>
                  <div class="projects-action-row">
                    <button
                      class="projects-button projects-button-secondary"
                      type="button"
                      aria-label={`Remove variant ${index() + 1}`}
                      disabled={(draft()?.spec.variants.length ?? 0) <= 1}
                      onClick={() => dispatch({ tag: "experimentVariantRemoved", index: index() })}
                    >
                      Remove variant
                    </button>
                  </div>
                </fieldset>
              );
            }}
          </For>
          <button
            class="projects-button projects-button-secondary"
            type="button"
            onClick={() => dispatch({ tag: "experimentVariantAdded" })}
          >
            Add variant
          </button>
        </section>

        <section class="projects-panel" aria-labelledby="execution-heading">
          <div class="projects-panel-heading">
            <div>
              <h2 id="execution-heading">Execution grid</h2>
              <p>Each budget is paired with every game and variant.</p>
            </div>
            <button
              class="projects-button projects-button-secondary"
              type="button"
              onClick={() => dispatch({ tag: "experimentBudgetAdded" })}
            >
              Add budget
            </button>
          </div>
          <For each={draft()?.spec.budgets ?? []}>
            {(budget, index) => {
              const kindError =
                fieldError(`spec.budgets[${index()}].kind`) ??
                fieldError(`spec.budgets[${index()}]`);
              const valueError = fieldError(`spec.budgets[${index()}].value`);
              return (
                <div class="projects-grid-item projects-form-grid projects-execution-grid">
                  <div class="projects-field">
                    <label for={`budget-kind-${index()}`}>Budget kind</label>
                    <select
                      id={`budget-kind-${index()}`}
                      {...fieldAttrs(kindError, `budget-kind-${index()}`)}
                      value={budget.kind}
                      onChange={(e) =>
                        dispatch({
                          tag: "experimentBudgetEdited",
                          index: index(),
                          field: "kind",
                          value: e.currentTarget.value,
                        })
                      }
                    >
                      <option value="iterations">Iterations</option>
                      <option value="time_per_move_ms">Time per move</option>
                    </select>
                    <FieldError error={kindError} id={errorId(`budget-kind-${index()}`)} />
                  </div>
                  <div class="projects-field projects-field-number">
                    <label for={`budget-value-${index()}`}>Budget value</label>
                    <input
                      id={`budget-value-${index()}`}
                      {...fieldAttrs(valueError, `budget-value-${index()}`)}
                      type="number"
                      min="1"
                      step="1"
                      value={budget.value}
                      onInput={(e) =>
                        dispatch({
                          tag: "experimentBudgetEdited",
                          index: index(),
                          field: "value",
                          value: Number(e.currentTarget.value),
                        })
                      }
                    />
                    <FieldError error={valueError} id={errorId(`budget-value-${index()}`)} />
                  </div>
                  <button
                    class="projects-button projects-button-secondary"
                    type="button"
                    aria-label={`Remove budget ${index() + 1}`}
                    disabled={(draft()?.spec.budgets.length ?? 0) <= 1}
                    onClick={() => dispatch({ tag: "experimentBudgetRemoved", index: index() })}
                  >
                    Remove budget
                  </button>
                </div>
              );
            }}
          </For>
          <div class="projects-form-grid projects-execution-grid">
            <div class="projects-field projects-field-number">
              <label for="rounds-per-cell">Paired rounds</label>
              <input
                id="rounds-per-cell"
                {...fieldAttrs(fieldError("spec.rounds_per_cell"), "rounds-per-cell")}
                type="number"
                min="1"
                step="1"
                value={draft()?.spec.rounds_per_cell}
                onInput={(e) =>
                  update((spec) => {
                    spec.rounds_per_cell = Number(e.currentTarget.value);
                  })
                }
              />
              <FieldError
                error={fieldError("spec.rounds_per_cell")}
                id={errorId("rounds-per-cell")}
              />
            </div>
            <div class="projects-field projects-field-number">
              <label for="base-seed">Base seed</label>
              <input
                id="base-seed"
                {...fieldAttrs(fieldError("spec.base_seed"), "base-seed")}
                type="number"
                min="0"
                step="1"
                value={draft()?.spec.base_seed}
                onInput={(e) =>
                  update((spec) => {
                    spec.base_seed = Number(e.currentTarget.value);
                  })
                }
              />
              <FieldError error={fieldError("spec.base_seed")} id={errorId("base-seed")} />
            </div>
            <div class="projects-field projects-field-number">
              <label for="max-parallel-cells">Max parallel cells</label>
              <input
                id="max-parallel-cells"
                {...fieldAttrs(fieldError("spec.max_parallel_cells"), "max-parallel-cells")}
                type="number"
                min="1"
                step="1"
                value={draft()?.spec.max_parallel_cells}
                onInput={(e) =>
                  update((spec) => {
                    spec.max_parallel_cells = Number(e.currentTarget.value);
                  })
                }
              />
              <FieldError
                error={fieldError("spec.max_parallel_cells")}
                id={errorId("max-parallel-cells")}
              />
            </div>
          </div>
        </section>

        <section class="projects-panel projects-plan-panel" aria-labelledby="plan-heading">
          <div class="projects-panel-heading">
            <div>
              <h2 id="plan-heading">Exact grid preview</h2>
              <p>
                Cells expand in game, budget, variant order with deterministic identifiers and
                seeds.
              </p>
            </div>
          </div>
          <Show when={preview()}>
            {(plan) => (
              <>
                <div class="projects-plan-summary">
                  <span class="projects-plan-number">{plan().total_planned_games}</span>
                  <span>
                    <strong>planned games</strong>
                    <small>
                      {plan().cells.length} cells · {draft()?.spec.rounds_per_cell} paired rounds
                    </small>
                  </span>
                </div>
                <div class="projects-cell-preview" aria-label="Grid preview">
                  <For each={plan().cells}>
                    {(cell) => (
                      <div class="projects-cell-preview-row">
                        <code>{cell.cell_id}</code>
                        <span>{cell.game}</span>
                        <span>{cell.variant_label}</span>
                        <span>{budgetSummary(cell.budget.kind, cell.budget.value)}</span>
                        <span>{cell.planned_games} games</span>
                      </div>
                    )}
                  </For>
                </div>
              </>
            )}
          </Show>
          <Show when={state().experimentError}>
            <p class="projects-form-error" role="alert">
              {state().experimentError}
            </p>
          </Show>
          <Show when={state().experimentRunError}>
            <p class="projects-form-error" role="alert">
              {state().experimentRunError}
            </p>
          </Show>
          <div class="projects-action-row">
            <button
              class="projects-button projects-button-secondary"
              type="button"
              disabled={saveDisabled()}
              onClick={() => dispatch({ tag: "saveExperiment" })}
            >
              {state().experimentSaveStatus === "saving" ? "Saving…" : "Save"}
            </button>
            <button
              class="projects-button projects-button-primary"
              type="button"
              disabled={launchDisabled()}
              onClick={() => dispatch({ tag: "launchExperiment" })}
            >
              {state().experimentLaunchStatus === "launching" ? "Launching…" : "Launch"}
            </button>
            <Show when={dirty()}>
              <span class="projects-dirty-note">Unsaved changes</span>
            </Show>
            <Show when={!dirty() && state().selectedExperimentId}>
              <span class="projects-saved-note">Saved and ready to launch</span>
            </Show>
          </div>
        </section>
        <section class="projects-panel" aria-labelledby="run-history-heading">
          <div class="projects-panel-heading">
            <div>
              <h2 id="run-history-heading">Run history</h2>
              <p>Recent launches for this project and experiment.</p>
            </div>
          </div>
          <Show
            when={state().runs.status === "done"}
            fallback={<div class="projects-state">Loading run history…</div>}
          >
            <div class="projects-run-list">
              <For each={state().runs.result ?? []}>
                {(run) => (
                  <button
                    class="projects-run-row"
                    type="button"
                    onClick={() => dispatch({ tag: "openRun", runId: run.run_id })}
                  >
                    <span class="projects-run-main">
                      <strong>{run.label ?? "Experiment run"}</strong>
                      <code>{run.run_id}</code>
                    </span>
                    <span class={`status-badge badge-${run.status}`}>
                      {friendlyStatus(run.status)}
                    </span>
                    <span>{run.match_count} matches</span>
                    <time datetime={run.started_at}>
                      {new Date(run.started_at).toLocaleString()}
                    </time>
                  </button>
                )}
              </For>
            </div>
          </Show>
        </section>
      </form>
    </Show>
  );
};
