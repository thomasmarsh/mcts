// profile-manager.component.test.tsx — component test for the launch-profile
// corpus screen. A real `createStore(tunerReducer, env)` backs the rendered
// view; the env is mocked (AGENTS.md "mock the environment"), no live server.

import { afterEach, describe, expect, it, vi } from "vitest";
import { cleanup, fireEvent, render, screen } from "@solidjs/testing-library";
import { Effect, createStore } from "@mcts/core";
import {
  initialTunerState,
  tunerReducer,
  type TunerAction,
  type TunerState,
} from "../../src/tuner/tuner-reducer.js";
import { ProfileManager } from "../../src/tuner/views/ProfileManager.js";
import { mockTunerEnv } from "./mock-tuner-env.js";
import type { TunerEnv } from "../../src/tuner/tuner-env.js";
import type { TunerProfileFile } from "../../src/tuner/tuner-types.js";

afterEach(() => cleanup());

const profile: TunerProfileFile = {
  key: "nim-sweep",
  profile_id: "nim-sweep",
  game_kind: "nim",
  objective_key: "nim-v1",
  constraint_count: 2,
  updated_at: null,
  is_seed: false,
};

function renderManager(env: TunerEnv = mockTunerEnv()) {
  const store = createStore<TunerState, TunerAction, TunerEnv>(
    initialTunerState(),
    tunerReducer,
    env,
  );
  store.dispatch({ tag: "profilesLoaded", profiles: [profile] });
  const navigate = vi.fn();
  render(() => <ProfileManager store={store} navigate={navigate} />);
  return { store, navigate };
}

describe("ProfileManager", () => {
  it("lists the profile corpus", () => {
    renderManager();
    expect(screen.getByTestId("profile-table")).toHaveTextContent("nim-sweep");
    expect(screen.getByTestId("profile-table")).toHaveTextContent("nim-v1");
  });

  it("navigates to the editor on Edit and New profile", () => {
    const { navigate } = renderManager();
    fireEvent.click(screen.getByText("Edit"));
    expect(navigate).toHaveBeenCalledWith({ view: "profile", key: "nim-sweep" });
    fireEvent.click(screen.getByText("New profile"));
    expect(navigate).toHaveBeenCalledWith({ view: "profile", key: null });
  });

  it("deletes a profile only after the inline confirm, then re-lists", async () => {
    const deleteProfile = vi.fn((_key: string) => Effect.send(undefined));
    const listProfiles = vi.fn(() => Effect.send<TunerProfileFile[]>([]));
    const env = mockTunerEnv({ deleteProfile, listProfiles });
    renderManager(env);

    fireEvent.click(screen.getByText("Delete"));
    expect(deleteProfile).not.toHaveBeenCalled();

    fireEvent.click(screen.getByText("Confirm delete"));
    await vi.waitFor(() => expect(deleteProfile).toHaveBeenCalledWith("nim-sweep"));
    await vi.waitFor(() => expect(listProfiles).toHaveBeenCalled());
    await vi.waitFor(() =>
      expect(screen.queryByText("nim-sweep")).not.toBeInTheDocument(),
    );
  });

  it("surfaces a delete failure inline", async () => {
    const env = mockTunerEnv({
      deleteProfile: () =>
        Effect.fromPromise(() => Promise.reject(new Error("profile in use"))),
    });
    renderManager(env);
    fireEvent.click(screen.getByText("Delete"));
    fireEvent.click(screen.getByText("Confirm delete"));
    await vi.waitFor(() =>
      expect(screen.getByRole("alert")).toHaveTextContent("profile in use"),
    );
  });
});
