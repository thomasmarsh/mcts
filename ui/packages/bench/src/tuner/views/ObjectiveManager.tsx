// ObjectiveManager — the objective corpus screen (`#/tuner/objectives`). A
// `DataTable` over `GET /api/bench/tuner/objectives` with per-row Edit /
// Duplicate / Delete and a "New objective" entry point. Composition only:
// every mutation is a reducer action, every fetch an `Effect` on the env.

import { createMemo, createSignal, Show, type Component } from "solid-js";
import type { Store } from "@mcts/core";
import { peek } from "../remote-data.js";
import type { TunerAction, TunerState } from "../tuner-reducer.js";
import type { TunerRoute } from "../tuner-routes.js";
import type { TunerObjectiveFile } from "../tuner-types.js";
import { DataTable, type DataColumn } from "../primitives/DataTable.js";

export const ObjectiveManager: Component<{
  store: Store<TunerState, TunerAction>;
  navigate: (route: TunerRoute) => void;
}> = (props) => {
  const state = props.store.getState();
  const dispatch = props.store.dispatch;

  const objectives = createMemo(() => peek(state().objectives) ?? []);
  const [confirming, setConfirming] = createSignal<string | null>(null);

  const edit = (row: TunerObjectiveFile): void =>
    props.navigate({ view: "objective", key: row.key });
  const duplicate = (row: TunerObjectiveFile): void =>
    props.navigate(
      row.game_kind
        ? { view: "objective", key: null, game: row.game_kind }
        : { view: "objective", key: null },
    );

  const columns: DataColumn<TunerObjectiveFile>[] = [
    { key: "key", header: "Key", render: (r) => r.key },
    { key: "objective_id", header: "Objective id", render: (r) => r.objective_id ?? "—" },
    { key: "game_kind", header: "Game", render: (r) => r.game_kind ?? "—" },
    { key: "opponent_count", header: "Opponents", align: "right", render: (r) => r.opponent_count },
    { key: "updated_at", header: "Updated", render: (r) => r.updated_at ?? "—" },
    {
      key: "is_seed",
      header: "",
      render: (r) => (
        <Show when={r.is_seed}>
          <span class="tuner-badge">seed</span>
        </Show>
      ),
    },
    {
      key: "actions",
      header: "",
      render: (r) => (
        <span class="tuner-objective-actions">
          <button onClick={() => edit(r)}>Edit</button>
          <button onClick={() => duplicate(r)}>Duplicate</button>
          <Show
            when={confirming() === r.key}
            fallback={<button onClick={() => setConfirming(r.key)}>Delete</button>}
          >
            <button
              class="tuner-objective-delete-confirm"
              disabled={state().objectiveMutating === r.key}
              onClick={() => {
                setConfirming(null);
                dispatch({ tag: "deleteObjective", key: r.key });
              }}
            >
              Confirm delete
            </button>
            <button onClick={() => setConfirming(null)}>Cancel</button>
          </Show>
        </span>
      ),
    },
  ];

  return (
    <div class="tuner-objective-manager" data-testid="tuner-objective-manager">
      <div class="tuner-fleet-header">
        <h3>Objectives</h3>
        <div class="tuner-fleet-actions">
          <button class="tuner-back" onClick={() => props.navigate({ view: "fleet" })}>
            ← Fleet
          </button>
          <button
            class="tuner-fleet-new"
            onClick={() => props.navigate({ view: "objective", key: null })}
          >
            New objective
          </button>
        </div>
      </div>

      <Show when={state().objectiveMutateError}>
        <div class="launch-error" role="alert">
          {state().objectiveMutateError}
        </div>
      </Show>

      <DataTable
        testid="objective-table"
        columns={columns}
        rows={objectives()}
        rowKey={(r) => r.key}
        empty="No objectives yet — create one."
      />
    </div>
  );
};
