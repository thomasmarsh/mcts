import { createEffect, createMemo, createSignal, For, Show, type Component } from "solid-js";
import type { Store } from "@mcts/core";
import type { BenchAction, BenchState } from "./index.js";
import type { ExperimentSpecV1 } from "./types.js";

type JsonField = "gameConfig" | "baselineConfig" | "variantConfig";

const prettyJson = (value: unknown): string => JSON.stringify(value ?? null, null, 2);

const friendlyStatus = (status: string): string => status.replaceAll("_", " ");

export const ExperimentEditor: Component<{ store: Store<BenchState, BenchAction> }> = (props) => {
  const state = props.store.getState();
  const dispatch = props.store.dispatch;
  const draft = createMemo(() => state().experimentDraft);
  const [gameConfigText, setGameConfigText] = createSignal("");
  const [baselineConfigText, setBaselineConfigText] = createSignal("");
  const [variantConfigText, setVariantConfigText] = createSignal("");
  const [jsonErrors, setJsonErrors] = createSignal<Partial<Record<JsonField, string>>>({});
  let lastLoadedKey = "";

  createEffect(() => {
    const current = draft();
    const spec = current?.spec;
    const key = current ? `${state().selectedExperimentId ?? "new"}:draft` : "empty";
    if (key === lastLoadedKey) return;
    lastLoadedKey = key;
    setGameConfigText(spec ? prettyJson(spec.games[0]?.game_config) : "");
    setBaselineConfigText(spec ? prettyJson(spec.baseline.config) : "");
    setVariantConfigText(spec ? prettyJson(spec.variants[0]?.config) : "");
    setJsonErrors({});
  });

  const update = (change: (spec: ExperimentSpecV1) => void) => {
    const current = draft();
    if (!current) return;
    const spec = structuredClone(current.spec);
    change(spec);
    dispatch({ tag: "experimentDraft", draft: { ...current, spec } });
  };

  const updateText = (field: JsonField, text: string) => {
    if (field === "gameConfig") setGameConfigText(text);
    if (field === "baselineConfig") setBaselineConfigText(text);
    if (field === "variantConfig") setVariantConfigText(text);
    try {
      const value: unknown = JSON.parse(text);
      if ((field === "baselineConfig" || field === "variantConfig") && (!value || typeof value !== "object" || Array.isArray(value))) {
        setJsonErrors((errors) => ({ ...errors, [field]: "Strategy configuration must be a JSON object." }));
        return;
      }
      setJsonErrors((errors) => { const next = { ...errors }; delete next[field]; return next; });
      update((spec) => {
        if (field === "gameConfig") spec.games[0]!.game_config = value;
        if (field === "baselineConfig") spec.baseline.config = value as Record<string, unknown>;
        if (field === "variantConfig") spec.variants[0]!.config = value as Record<string, unknown>;
      });
    } catch {
      setJsonErrors((errors) => ({ ...errors, [field]: "Enter valid JSON." }));
    }
  };

  const gameOptions = createMemo(() => {
    const options = [...(state().smac3Kinds.result ?? [])];
    const current = draft()?.spec.games[0]?.game;
    return current && !options.some((item) => item.game === current) ? [{ game: current, tuner: { id: "", baselines: [], eval_rounds: 0, parameters: [], conditions: [], game_config: draft()?.spec.games[0]?.game_config } }, ...options] : options;
  });
  const currentBudget = () => draft()?.spec.budgets[0] ?? { kind: "iterations" as const, value: 25 };
  const dirty = () => {
    const current = draft();
    return current !== null && JSON.stringify(current) !== JSON.stringify(state().experimentSavedDraft);
  };
  const localJsonInvalid = () => Object.keys(jsonErrors()).length > 0;
  const saveDisabled = () => state().experimentSaveStatus === "saving" || localJsonInvalid();
  const launchDisabled = () => !state().selectedExperimentId || state().experimentSaveStatus === "saving" || state().experimentLaunchStatus === "launching" || dirty() || localJsonInvalid();
  const fieldError = (path: string) => state().experimentFieldErrors[path];
  const combinedError = (path: string, localField?: JsonField) => fieldError(path) ?? (localField ? jsonErrors()[localField] : undefined);

  return <Show when={draft()} fallback={<div class="projects-state projects-state-error" role={state().experimentError ? "alert" : undefined}><span class="projects-state-title">{state().experimentError ? "Experiment could not be loaded" : "Select or create an experiment"}</span><span>{state().experimentError ?? "Return to the project to choose a saved definition."}</span></div>}>
    <form class="projects-editor" onSubmit={(event) => event.preventDefault()}>
      <header class="projects-page-header projects-editor-header">
        <div>
          <button class="projects-back-link" type="button" onClick={() => dispatch({ tag: "openProject", projectId: state().selectedProjectId ?? "" })}><span aria-hidden="true">←</span> Project</button>
          <p class="projects-eyebrow">{state().selectedExperimentId ? "Saved experiment" : "New experiment"}</p>
          <h1>{draft()?.name || "Untitled experiment"}</h1>
          <p class="projects-lede">Define one candidate-versus-baseline game cell, save it, and then launch an exact saved snapshot.</p>
        </div>
      </header>

      <section class="projects-panel" aria-labelledby="identity-heading">
        <div class="projects-panel-heading"><div><h2 id="identity-heading">Identity</h2><p>Name the experiment and record the question it is meant to answer.</p></div></div>
        <div class="projects-form-grid projects-form-grid-two">
          <div class="projects-field"><label for="experiment-name">Experiment name</label><input id="experiment-name" aria-invalid={Boolean(fieldError("name"))} aria-describedby={fieldError("name") ? "experiment-name-error" : undefined} value={draft()?.name} onInput={(e) => dispatch({ tag: "experimentDraft", draft: { ...draft()!, name: e.currentTarget.value } })} /> <Show when={fieldError("name")}><span id="experiment-name-error" class="projects-field-error">{fieldError("name")}</span></Show></div>
          <div class="projects-field"><label for="experiment-description">Description <span class="projects-optional">Optional</span></label><textarea id="experiment-description" rows="2" value={draft()?.description} onInput={(e) => dispatch({ tag: "experimentDraft", draft: { ...draft()!, description: e.currentTarget.value } })} /></div>
        </div>
      </section>

      <section class="projects-panel" aria-labelledby="game-heading">
        <div class="projects-panel-heading"><div><h2 id="game-heading">Game</h2><p>Choose from the SMAC3-compatible game catalog. Changing games restores that game's default setup.</p></div></div>
        <div class="projects-form-grid projects-form-grid-two">
          <div class="projects-field"><label for="experiment-game">Game</label><select id="experiment-game" value={draft()?.spec.games[0]?.game} onChange={(e) => { const selected = gameOptions().find((item) => item.game === e.currentTarget.value); const gameConfig = selected?.tuner.game_config ?? null; dispatch({ tag: "experimentGameChanged", game: e.currentTarget.value, gameConfig }); setGameConfigText(prettyJson(gameConfig)); setJsonErrors((errors) => { const next = { ...errors }; delete next.gameConfig; return next; }); }}><For each={gameOptions()}>{(item) => <option value={item.game}>{item.game}</option>}</For></select><span class="projects-help">Available games come from the loaded Bench metadata.</span></div>
          <div class="projects-field"><label for="game-config">Game configuration</label><textarea id="game-config" class="projects-json-editor" rows="6" value={gameConfigText()} aria-invalid={Boolean(combinedError("spec.games[0].game_config", "gameConfig"))} aria-describedby="game-config-help game-config-error" onInput={(e) => updateText("gameConfig", e.currentTarget.value)} /><span id="game-config-help" class="projects-help">Any valid JSON value is accepted by the game contract.</span><Show when={combinedError("spec.games[0].game_config", "gameConfig")}><span id="game-config-error" class="projects-field-error" role="alert">{combinedError("spec.games[0].game_config", "gameConfig")}</span></Show></div>
        </div>
      </section>

      <section class="projects-panel" aria-labelledby="comparison-heading">
        <div class="projects-panel-heading"><div><h2 id="comparison-heading">Comparison</h2><p>Both strategies play the same paired rounds, switching sides to keep the comparison balanced.</p></div></div>
        <div class="projects-strategy-grid">
          <fieldset class="projects-strategy-panel"><legend>Baseline</legend><div class="projects-field"><label for="baseline-label">Human-readable label</label><input id="baseline-label" value={draft()?.spec.baseline.label} onInput={(e) => update((spec) => { spec.baseline.label = e.currentTarget.value; })} aria-invalid={Boolean(fieldError("spec.baseline.label"))} /><Show when={fieldError("spec.baseline.label")}><span class="projects-field-error">{fieldError("spec.baseline.label")}</span></Show></div><div class="projects-field"><label for="baseline-config">Opaque strategy configuration</label><textarea id="baseline-config" class="projects-json-editor" rows="7" value={baselineConfigText()} aria-invalid={Boolean(combinedError("spec.baseline.config", "baselineConfig"))} aria-describedby="baseline-config-help baseline-config-error" onInput={(e) => updateText("baselineConfig", e.currentTarget.value)} /><span id="baseline-config-help" class="projects-help">A JSON object supplied to the baseline strategy.</span><Show when={combinedError("spec.baseline.config", "baselineConfig")}><span id="baseline-config-error" class="projects-field-error" role="alert">{combinedError("spec.baseline.config", "baselineConfig")}</span></Show></div></fieldset>
          <fieldset class="projects-strategy-panel"><legend>Variant</legend><div class="projects-field"><label for="variant-label">Human-readable label</label><input id="variant-label" value={draft()?.spec.variants[0]?.label} onInput={(e) => update((spec) => { spec.variants[0]!.label = e.currentTarget.value; })} aria-invalid={Boolean(fieldError("spec.variants[0].label"))} /><Show when={fieldError("spec.variants[0].label")}><span class="projects-field-error">{fieldError("spec.variants[0].label")}</span></Show></div><div class="projects-field"><label for="variant-config">Opaque strategy configuration</label><textarea id="variant-config" class="projects-json-editor" rows="7" value={variantConfigText()} aria-invalid={Boolean(combinedError("spec.variants[0].config", "variantConfig"))} aria-describedby="variant-config-help variant-config-error" onInput={(e) => updateText("variantConfig", e.currentTarget.value)} /><span id="variant-config-help" class="projects-help">A JSON object supplied to the candidate strategy.</span><Show when={combinedError("spec.variants[0].config", "variantConfig")}><span id="variant-config-error" class="projects-field-error" role="alert">{combinedError("spec.variants[0].config", "variantConfig")}</span></Show></div></fieldset>
        </div>
      </section>

      <section class="projects-panel" aria-labelledby="execution-heading">
        <div class="projects-panel-heading"><div><h2 id="execution-heading">Execution</h2><p>Choose the per-move budget and how many paired rounds to run.</p></div></div>
        <div class="projects-form-grid projects-execution-grid">
          <div class="projects-field"><label for="budget-kind">Budget kind</label><select id="budget-kind" value={currentBudget().kind} onChange={(e) => update((spec) => { const value = Math.max(1, spec.budgets[0]?.value ?? 25); spec.budgets[0] = e.currentTarget.value === "iterations" ? { kind: "iterations", value } : { kind: "time_per_move_ms", value }; })}><option value="iterations">Iterations</option><option value="time_per_move_ms">Time per move</option></select><span class="projects-help">Iterations count search steps; time uses milliseconds per move.</span></div>
          <div class="projects-field projects-field-number"><label for="budget-value">Budget value</label><input id="budget-value" type="number" min="1" step="1" value={currentBudget().value} onInput={(e) => update((spec) => { spec.budgets[0]!.value = Number(e.currentTarget.value); })} aria-invalid={Boolean(fieldError("spec.budgets[0].value"))} /><Show when={fieldError("spec.budgets[0].value")}><span class="projects-field-error">{fieldError("spec.budgets[0].value")}</span></Show></div>
          <div class="projects-field projects-field-number"><label for="rounds-per-cell">Paired rounds</label><input id="rounds-per-cell" type="number" min="1" step="1" value={draft()?.spec.rounds_per_cell} onInput={(e) => update((spec) => { spec.rounds_per_cell = Number(e.currentTarget.value); })} aria-invalid={Boolean(fieldError("spec.rounds_per_cell"))} /><span class="projects-help">Each round produces two games.</span></div>
          <div class="projects-field projects-field-number"><label for="base-seed">Base seed</label><input id="base-seed" type="number" step="1" value={draft()?.spec.base_seed} onInput={(e) => update((spec) => { spec.base_seed = Number(e.currentTarget.value); })} /><span class="projects-help">Seeds are advanced for each paired game.</span></div>
        </div>
      </section>

      <section class="projects-panel projects-plan-panel" aria-labelledby="plan-heading">
        <div class="projects-panel-heading"><div><h2 id="plan-heading">Plan summary and actions</h2><p>One cell compares the variant against the baseline across the planned paired games.</p></div></div>
        <div class="projects-plan-summary"><span class="projects-plan-number">{2 * (draft()?.spec.rounds_per_cell ?? 0)}</span><span><strong>planned games</strong><small>1 cell · {draft()?.spec.rounds_per_cell} paired rounds</small></span></div>
        <Show when={state().experimentError}><p class="projects-form-error" role="alert">{state().experimentError}</p></Show>
        <Show when={state().experimentRunError}><p class="projects-form-error" role="alert">{state().experimentRunError}</p></Show>
        <div class="projects-action-row"><button class="projects-button projects-button-secondary" type="button" disabled={saveDisabled()} onClick={() => dispatch({ tag: "saveExperiment" })}>{state().experimentSaveStatus === "saving" ? "Saving…" : "Save"}</button><button class="projects-button projects-button-primary" type="button" disabled={launchDisabled()} onClick={() => dispatch({ tag: "launchExperiment" })}>{state().experimentLaunchStatus === "launching" ? "Launching…" : "Launch"}</button><Show when={dirty()}><span class="projects-dirty-note">Unsaved changes</span></Show><Show when={!dirty() && state().selectedExperimentId}><span class="projects-saved-note">Saved and ready to launch</span></Show></div>
      </section>

      <section class="projects-panel" aria-labelledby="run-history-heading">
        <div class="projects-panel-heading"><div><h2 id="run-history-heading">Run history</h2><p>Runs for this experiment are loaded from the filtered Bench history.</p></div></div>
        <Show when={state().runs.status === "done"} fallback={<Show when={state().runs.status === "error"} fallback={<div class="projects-state"><span class="projects-state-title">Loading run history</span><span>Checking for previous launches…</span></div>}><div class="projects-state projects-state-error" role="alert"><span class="projects-state-title">Run history unavailable</span><span>{state().runs.error ?? "Try opening the experiment again."}</span></div></Show>}>
          <Show when={(state().runs.result ?? []).length > 0} fallback={<div class="projects-state"><span class="projects-state-title">No runs yet</span><span>Save this definition, then Launch to create its first run.</span></div>}>
            <div class="projects-run-list"><For each={state().runs.result ?? []}>{(run) => <button class="projects-run-row" type="button" onClick={() => dispatch({ tag: "openRun", runId: run.run_id })} aria-label={`Open run ${run.label ?? run.run_id}`}><span class="projects-run-main"><strong>{run.label ?? "Experiment run"}</strong><code>{run.run_id}</code></span><span class={`status-badge badge-${run.status}`}>{friendlyStatus(run.status)}</span><span class="projects-run-matches">{run.match_count} matches</span><time datetime={run.started_at}>{new Date(run.started_at).toLocaleString()}</time></button>}</For></div>
          </Show>
        </Show>
      </section>
    </form>
  </Show>;
};
