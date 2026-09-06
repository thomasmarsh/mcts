// ProfileManager — the launch-profile corpus screen (`#/tuner/profiles`). A
// `DataTable` over `GET /api/bench/tuner/profiles` with per-row Edit /
// Duplicate / Delete and a "New profile" entry point. The launch-form
// counterpart to `ObjectiveManager`: composition only, every mutation a
// reducer action, every fetch an `Effect` on the env.

import { createMemo, createSignal, Show, type Component } from "solid-js";
import type { Store } from "@mcts/core";
import { peek } from "../remote-data.js";
import type { TunerAction, TunerState } from "../tuner-reducer.js";
import type { TunerRoute } from "../tuner-routes.js";
import type { TunerProfileFile } from "../tuner-types.js";
import { DataTable, type DataColumn } from "../primitives/DataTable.js";

export const ProfileManager: Component<{
  store: Store<TunerState, TunerAction>;
  navigate: (route: TunerRoute) => void;
}> = (props) => {
  const state = props.store.getState();
  const dispatch = props.store.dispatch;

  const profiles = createMemo(() => peek(state().profiles) ?? []);
  const [confirming, setConfirming] = createSignal<string | null>(null);

  const edit = (row: TunerProfileFile): void =>
    props.navigate({ view: "profile", key: row.key });
  const duplicate = (): void => props.navigate({ view: "profile", key: null });

  const columns: DataColumn<TunerProfileFile>[] = [
    { key: "key", header: "Key", render: (r) => r.key },
    { key: "profile_id", header: "Profile id", render: (r) => r.profile_id ?? "—" },
    { key: "game_kind", header: "Game", render: (r) => r.game_kind ?? "—" },
    { key: "objective_key", header: "Objective", render: (r) => r.objective_key ?? "—" },
    {
      key: "constraint_count",
      header: "Constraints",
      align: "right",
      render: (r) => r.constraint_count,
    },
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
          <button onClick={() => duplicate()}>Duplicate</button>
          <Show
            when={confirming() === r.key}
            fallback={<button onClick={() => setConfirming(r.key)}>Delete</button>}
          >
            <button
              class="tuner-objective-delete-confirm"
              disabled={state().profileMutating === r.key}
              onClick={() => {
                setConfirming(null);
                dispatch({ tag: "deleteProfile", key: r.key });
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
    <div class="tuner-profile-manager" data-testid="tuner-profile-manager">
      <div class="tuner-fleet-header">
        <h3>Launch profiles</h3>
        <div class="tuner-fleet-actions">
          <button class="tuner-back" onClick={() => props.navigate({ view: "fleet" })}>
            ← Fleet
          </button>
          <button
            class="tuner-fleet-new"
            onClick={() => props.navigate({ view: "profile", key: null })}
          >
            New profile
          </button>
        </div>
      </div>

      <Show when={state().profileMutateError}>
        <div class="launch-error" role="alert">
          {state().profileMutateError}
        </div>
      </Show>

      <DataTable
        testid="profile-table"
        columns={columns}
        rows={profiles()}
        rowKey={(r) => r.key}
        empty="No launch profiles yet — create one."
      />
    </div>
  );
};
