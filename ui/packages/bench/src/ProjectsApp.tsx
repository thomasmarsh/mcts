import { Show, type Component } from "solid-js";
import type { Store } from "@mcts/core";
import type { BenchAction, BenchState } from "./index.js";
import { ProjectsLanding } from "./ProjectsLanding.js";
import { ProjectDetail } from "./ProjectDetail.js";
import { ExperimentEditor } from "./ExperimentEditor.js";

export const ProjectsApp: Component<{ store: Store<BenchState, BenchAction> }> = (props) => {
  const state = props.store.getState();
  return (
    <main class="projects-page">
      <Show
        when={state().selectedExperimentId || state().experimentDraft}
        fallback={
          <Show when={state().selectedProjectId} fallback={<ProjectsLanding store={props.store} />}>
            <ProjectDetail store={props.store} />
          </Show>
        }
      >
        <ExperimentEditor store={props.store} />
      </Show>
    </main>
  );
};
