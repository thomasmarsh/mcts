import { For, Show, type Component } from "solid-js";
import type { Store } from "@mcts/core";
import type { BenchAction, BenchState } from "./index.js";

export const ProjectDetail: Component<{ store: Store<BenchState, BenchAction> }> = (props) => {
  const state = props.store.getState(); const dispatch = props.store.dispatch;
  return <section class="bench-project-detail">
    <button onClick={() => dispatch({ tag: "setTab", tab: "projects" })}>← Projects</button>
    <Show when={state().selectedProject}><h2>{state().selectedProject?.name}</h2><p>{state().selectedProject?.description}</p></Show>
    <button onClick={() => dispatch({ tag: "newExperiment" })}>New experiment</button>
    <Show when={state().experimentError}><p class="error">{state().experimentError}</p></Show>
    <h3>Experiments</h3>
    <For each={state().experiments.result ?? []}>{(experiment) => <button class="experiment-card" onClick={() => dispatch({ tag: "openExperiment", experimentId: experiment.experiment_id })}><strong>{experiment.name}</strong><span>{experiment.description}</span></button>}</For>
    <Show when={state().experiments.status === "done" && (state().experiments.result ?? []).length === 0}><p>No saved experiments.</p></Show>
  </section>;
};
