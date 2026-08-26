import { For, Show, type Component } from "solid-js";
import type { Store } from "@mcts/core";
import type { BenchAction, BenchState } from "./index.js";

export const ProjectsLanding: Component<{ store: Store<BenchState, BenchAction> }> = (props) => {
  const state = props.store.getState();
  const dispatch = props.store.dispatch;
  return (
    <>
      <header class="projects-page-header">
        <div>
          <p class="projects-eyebrow">Bench / Projects</p>
          <h1>Projects</h1>
          <p class="projects-lede">
            Organize durable experiment definitions and compare their saved runs over time.
          </p>
        </div>
      </header>

      <section class="projects-panel" aria-labelledby="create-project-heading">
        <div class="projects-panel-heading">
          <div>
            <h2 id="create-project-heading">Create project</h2>
            <p>Give a research thread a home before defining its first experiment.</p>
          </div>
        </div>
        <form
          class="project-create"
          onSubmit={(event) => {
            event.preventDefault();
            dispatch({ tag: "createProject" });
          }}
          onClick={(event) => {
            if ((event.target as HTMLElement).closest(".project-create-submit"))
              dispatch({ tag: "createProject" });
          }}
        >
          <div class="projects-field projects-field-grow">
            <label for="project-name">Project name</label>
            <input
              id="project-name"
              value={state().projectDraft.name}
              onInput={(e) =>
                dispatch({
                  tag: "projectDraft",
                  name: e.currentTarget.value,
                  description: state().projectDraft.description,
                })
              }
            />
          </div>
          <div class="projects-field projects-field-grow">
            <label for="project-description">
              Description <span class="projects-optional">Optional</span>
            </label>
            <textarea
              id="project-description"
              rows="2"
              value={state().projectDraft.description}
              onInput={(e) =>
                dispatch({
                  tag: "projectDraft",
                  name: state().projectDraft.name,
                  description: e.currentTarget.value,
                })
              }
            />
          </div>
          <button
            class="projects-button projects-button-primary project-create-submit"
            type="button"
            disabled={!state().projectDraft.name.trim()}
          >
            Create project
          </button>
        </form>
        <Show when={state().projectError}>
          <p class="projects-form-error" role="alert">
            {state().projectError}
          </p>
        </Show>
      </section>

      <section class="projects-panel" aria-labelledby="active-projects-heading">
        <div class="projects-panel-heading">
          <div>
            <h2 id="active-projects-heading">Active projects</h2>
            <p>Open a project to review its saved experiments.</p>
          </div>
        </div>
        <Show
          when={state().projects.status === "done"}
          fallback={
            <Show
              when={state().projects.status === "error"}
              fallback={
                <div class="projects-state">
                  <span class="projects-state-title">Loading projects</span>
                  <span>Fetching your active research projects…</span>
                </div>
              }
            >
              <div class="projects-state projects-state-error" role="alert">
                <span class="projects-state-title">Projects could not be loaded</span>
                <span>{state().projects.error ?? "Try refreshing the Bench view."}</span>
              </div>
            </Show>
          }
        >
          <Show
            when={(state().projects.result ?? []).length > 0}
            fallback={
              <div class="projects-state">
                <span class="projects-state-title">No active projects yet</span>
                <span>Create a project above to begin a saved experiment workflow.</span>
              </div>
            }
          >
            <div class="projects-card-list">
              <For each={state().projects.result ?? []}>
                {(project) => (
                  <button
                    class="projects-card projects-card-button"
                    onClick={() => dispatch({ tag: "openProject", projectId: project.project_id })}
                    aria-label={`Open project ${project.name}`}
                  >
                    <span class="projects-card-main">
                      <strong>{project.name}</strong>
                      <span>{project.description.trim() || "No description"}</span>
                    </span>
                    <span class="projects-card-action">
                      Open project <span aria-hidden="true">→</span>
                    </span>
                  </button>
                )}
              </For>
            </div>
          </Show>
        </Show>
      </section>
    </>
  );
};
