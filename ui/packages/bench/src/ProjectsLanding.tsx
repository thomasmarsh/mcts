import { For, Show, type Component } from "solid-js";
import type { Store } from "@mcts/core";
import type { BenchAction, BenchState } from "./index.js";

export const ProjectsLanding: Component<{ store: Store<BenchState, BenchAction> }> = (props) => {
  const state = props.store.getState();
  const dispatch = props.store.dispatch;
  return <section class="bench-projects">
    <h2>Projects</h2>
    <p>Durable research projects and their saved experiment definitions.</p>
    <div class="project-create">
      <input aria-label="Project name" placeholder="Project name" value={state().projectDraft.name} onInput={(e) => dispatch({ tag: "projectDraft", name: e.currentTarget.value, description: state().projectDraft.description })} />
      <textarea aria-label="Project description" placeholder="Description" value={state().projectDraft.description} onInput={(e) => dispatch({ tag: "projectDraft", name: state().projectDraft.name, description: e.currentTarget.value })} />
      <button onClick={() => dispatch({ tag: "createProject" })}>Create project</button>
    </div>
    <Show when={state().projectError}><p class="error">{state().projectError}</p></Show>
    <Show when={state().projects.status === "done"} fallback={<p>Loading projects…</p>}>
      <div class="project-list"><For each={state().projects.result ?? []}>{(project) => <button class="project-card" onClick={() => dispatch({ tag: "openProject", projectId: project.project_id })}><strong>{project.name}</strong><span>{project.description}</span></button>}</For></div>
      <Show when={(state().projects.result ?? []).length === 0}><p>No active projects yet.</p></Show>
    </Show>
  </section>;
};
