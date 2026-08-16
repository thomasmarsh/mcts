import { For, Show, type Component } from "solid-js";
import type { Store } from "@mcts/core";
import type { BenchAction, BenchState } from "./index.js";

export const ProjectDetail: Component<{ store: Store<BenchState, BenchAction> }> = (props) => {
  const state = props.store.getState(); const dispatch = props.store.dispatch;
  const budgetSummary = (kind: string, value: number) => kind === "iterations" ? `${value} iterations` : `${value} ms per move`;
  return <>
    <header class="projects-page-header projects-detail-header">
      <div>
        <button class="projects-back-link" onClick={() => dispatch({ tag: "setTab", tab: "projects" })}><span aria-hidden="true">←</span> Projects</button>
        <p class="projects-eyebrow">Project</p>
        <Show when={state().selectedProject} fallback={<div class="projects-state"><span>Loading project…</span></div>}>
          <h1>{state().selectedProject?.name}</h1>
          <p class="projects-lede">{state().selectedProject?.description?.trim() || "No description for this project."}</p>
        </Show>
      </div>
      <button class="projects-button projects-button-primary" onClick={() => dispatch({ tag: "newExperiment" })}>New experiment</button>
    </header>

    <section class="projects-panel" aria-labelledby="saved-experiments-heading">
      <div class="projects-panel-heading"><div><h2 id="saved-experiments-heading">Saved experiments</h2><p>Definitions are loaded from the project and can be reopened without relaunching them.</p></div></div>
      <Show when={state().experiments.status === "done"} fallback={<Show when={state().experiments.status === "error"} fallback={<div class="projects-state"><span class="projects-state-title">Loading experiments</span><span>Reading saved definitions…</span></div>}><div class="projects-state projects-state-error" role="alert"><span class="projects-state-title">Experiments could not be loaded</span><span>{state().experiments.error ?? "Try opening the project again."}</span></div></Show>}>
        <Show when={(state().experiments.result ?? []).length > 0} fallback={<div class="projects-state"><span class="projects-state-title">No saved experiments</span><span>Choose New experiment to define the first one for this project.</span></div>}>
          <div class="projects-card-list"><For each={state().experiments.result ?? []}>{(experiment) => {
            const game = experiment.spec.games[0]?.game ?? "Unknown game";
            const budget = experiment.spec.budgets[0];
            return <button class="projects-card projects-card-button experiment-card" onClick={() => dispatch({ tag: "openExperiment", experimentId: experiment.experiment_id })} aria-label={`Open experiment ${experiment.name}`}>
              <span class="projects-card-main"><strong>{experiment.name}</strong><span>{experiment.description.trim() || "No description"}</span><span class="experiment-summary-meta"><span>{game}</span><span>{budget ? budgetSummary(budget.kind, budget.value) : "No budget"}</span><span>{experiment.spec.rounds_per_cell} paired rounds</span></span></span>
              <span class="projects-card-action">Open experiment <span aria-hidden="true">→</span></span>
            </button>;
          }}</For></div>
        </Show>
      </Show>
    </section>
    <section class="projects-panel" aria-labelledby="recent-runs-heading">
      <div class="projects-panel-heading"><div><h2 id="recent-runs-heading">Recent runs</h2><p>Project runs remain available while definitions evolve.</p></div></div>
      <Show when={state().runs.status === "done"} fallback={<div class="projects-state">Loading recent runs…</div>}>
        <Show when={(state().runs.result ?? []).length > 0} fallback={<div class="projects-state"><span class="projects-state-title">No runs yet</span><span>Launch a saved experiment to create the first project run.</span></div>}>
          <div class="projects-run-list"><For each={state().runs.result ?? []}>{(run) => <button class="projects-run-row" type="button" onClick={() => dispatch({ tag: "openRun", runId: run.run_id })}><span class="projects-run-main"><strong>{run.label ?? "Bench run"}</strong><code>{run.run_id}</code></span><span class={`status-badge badge-${run.status}`}>{run.status.replaceAll("_", " ")}</span><span>{run.match_count} matches</span><time datetime={run.started_at}>{new Date(run.started_at).toLocaleString()}</time></button>}</For></div>
        </Show>
      </Show>
    </section>
  </>;
};
