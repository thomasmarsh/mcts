// ObjectiveEditor — placeholder shell for the objective create/edit form.
// The data layer (routes, reducer slots, api client) it will build on lands
// with the manager; the opponent-panel form itself is a later slice.

import { Show, type Component } from "solid-js";
import type { Store } from "@mcts/core";
import { peek } from "../remote-data.js";
import type { TunerAction, TunerState } from "../tuner-reducer.js";
import type { TunerRoute } from "../tuner-routes.js";

export const ObjectiveEditor: Component<{
  store: Store<TunerState, TunerAction>;
  objectiveKey: string | null;
  game?: string;
  navigate: (route: TunerRoute) => void;
}> = (props) => {
  const state = props.store.getState();
  const detail = () => peek(state().objectiveDetail);

  return (
    <div class="tuner-objective-editor" data-testid="tuner-objective-editor">
      <button class="tuner-back" onClick={() => props.navigate({ view: "objectives" })}>
        ← Objectives
      </button>
      <h3>
        {props.objectiveKey === null
          ? `New objective${props.game ? ` for ${props.game}` : ""}`
          : `Edit ${props.objectiveKey}`}
      </h3>
      <p class="tuner-fleet-empty">The objective editor form is not built yet.</p>
      <Show when={detail()}>
        <pre class="tuner-objective-editor-raw" data-testid="objective-editor-raw">
          {JSON.stringify(detail()!.content, null, 2)}
        </pre>
      </Show>
    </div>
  );
};
