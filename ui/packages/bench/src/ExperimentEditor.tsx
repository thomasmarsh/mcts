import { createMemo, For, Show, type Component } from "solid-js";
import type { Store } from "@mcts/core";
import type { BenchAction, BenchState } from "./index.js";
import type { ExperimentSpecV1 } from "./types.js";

export const ExperimentEditor: Component<{ store: Store<BenchState, BenchAction> }> = (props) => {
  const state = props.store.getState(); const dispatch = props.store.dispatch;
  const draft = createMemo(() => state().experimentDraft);
  const update = (change: (spec: ExperimentSpecV1) => void) => { const current = draft(); if (!current) return; const spec = structuredClone(current.spec); change(spec); dispatch({ tag: "experimentDraft", draft: { ...current, spec } }); };
  const jsonText = (value: unknown) => JSON.stringify(value, null, 2);
  const updateConfig = (side: "baseline" | "variant", text: string) => { try { const value = JSON.parse(text) as unknown; if (!value || typeof value !== "object" || Array.isArray(value)) return; update((spec) => { if (side === "baseline") spec.baseline.config = value as Record<string, unknown>; else spec.variants[0]!.config = value as Record<string, unknown>; }); } catch { /* Keep invalid text local to the input until it is corrected. */ } };
  return <Show when={draft()} fallback={<p>Select or create an experiment.</p>}>
    <section class="bench-experiment-editor">
      <button onClick={() => dispatch({ tag: "openProject", projectId: state().selectedProjectId ?? "" })}>← Project</button>
      <h2>Experiment</h2>
      <input aria-label="Experiment name" value={draft()?.name} onInput={(e) => dispatch({ tag: "experimentDraft", draft: { ...draft()!, name: e.currentTarget.value } })} placeholder="Experiment name" />
      <textarea aria-label="Experiment description" value={draft()?.description} onInput={(e) => dispatch({ tag: "experimentDraft", draft: { ...draft()!, description: e.currentTarget.value } })} placeholder="Description" />
      <label>Game <input value={draft()?.spec.games[0]?.game} onInput={(e) => update((spec) => { spec.games[0]!.game = e.currentTarget.value; })} /></label>
      <label>Game config <textarea value={jsonText(draft()?.spec.games[0]?.game_config)} onInput={(e) => { try { update((spec) => { spec.games[0]!.game_config = JSON.parse(e.currentTarget.value) as unknown; }); } catch { /* invalid JSON waits for correction */ } }} /></label>
      <label>Baseline label <input value={draft()?.spec.baseline.label} onInput={(e) => update((spec) => { spec.baseline.label = e.currentTarget.value; })} /></label>
      <label>Baseline config <textarea value={jsonText(draft()?.spec.baseline.config)} onInput={(e) => updateConfig("baseline", e.currentTarget.value)} /></label>
      <label>Variant label <input value={draft()?.spec.variants[0]?.label} onInput={(e) => update((spec) => { spec.variants[0]!.label = e.currentTarget.value; })} /></label>
      <label>Variant config <textarea value={jsonText(draft()?.spec.variants[0]?.config)} onInput={(e) => updateConfig("variant", e.currentTarget.value)} /></label>
      <label>Budget iterations <input type="number" value={draft()?.spec.budgets[0]?.kind === "iterations" ? draft()?.spec.budgets[0]?.value : 25} onInput={(e) => update((spec) => { spec.budgets[0] = { kind: "iterations", value: Number(e.currentTarget.value) }; })} /></label>
      <label>Rounds <input type="number" min="1" value={draft()?.spec.rounds_per_cell} onInput={(e) => update((spec) => { spec.rounds_per_cell = Number(e.currentTarget.value); })} /></label>
      <label>Seed <input type="number" value={draft()?.spec.base_seed} onInput={(e) => update((spec) => { spec.base_seed = Number(e.currentTarget.value); })} /></label>
      <p>One cell, {2 * (draft()?.spec.rounds_per_cell ?? 0)} planned games.</p>
      <Show when={state().experimentError}><p class="error">{state().experimentError}</p></Show>
      <button onClick={() => dispatch({ tag: "saveExperiment" })}>Save</button>
      <button disabled={!state().selectedExperimentId} onClick={() => dispatch({ tag: "launchExperiment" })}>Launch</button>
      <Show when={state().experimentRunError}><p class="error">{state().experimentRunError}</p></Show>
      <h3>Runs</h3>
      <Show when={state().runs.status === "done"} fallback={<p>Loading runs…</p>}>
        <For each={state().runs.result ?? []}>{(run) => <button onClick={() => dispatch({ tag: "openRun", runId: run.run_id })}>{run.run_id} · {run.status} · {run.match_count} matches</button>}</For>
        <Show when={(state().runs.result ?? []).length === 0}><p>No runs yet.</p></Show>
      </Show>
    </section>
  </Show>;
};
