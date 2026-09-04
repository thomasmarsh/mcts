// ProfileEditor — build or edit a launch profile from the UI
// (`#/tuner/profiles/new` or `.../profiles/<key>`). The form state is a pure
// `ProfileDraft` (see `models/profile-model.ts`), not the wire JSON; this
// view is the glue: game + objective pickers, the schema-driven constraint
// editor, the effort/budget grid, the client-side validator on every
// keystroke, and `saveProfile` / `validateProfile` dispatch.

import { createEffect, createMemo, createSignal, For, Show, type Component } from "solid-js";
import type { Store } from "@mcts/core";
import { peek } from "../remote-data.js";
import type { TunerAction, TunerState } from "../tuner-reducer.js";
import type { TunerRoute } from "../tuner-routes.js";
import { ConstraintEditor } from "./ConstraintEditor.js";
import type { ParamSchema } from "../models/constraint-editor-model.js";
import {
  draftFromProfileContent,
  draftToProfileContent,
  emptyProfileDraft,
  profileSlugKey,
  validateProfileDraft,
  PROFILE_PHASES,
  type EffortUnit,
  type ProfileDraft,
  type ProfilePhase,
} from "../models/profile-model.js";

const EMPTY_SCHEMA: ParamSchema = { parameters: [], conditions: [] };

const BUDGET_FIELDS: Array<[keyof ProfileDraft["budgets"], string]> = [
  ["tuningPairBudget", "Tuning pair budget"],
  ["validationPairBudget", "Validation pair budget"],
  ["productionValidationPairs", "Production validation pairs"],
  ["cohortSize", "Cohort size (optional)"],
  ["finalists", "Finalists (optional)"],
];

export const ProfileEditor: Component<{
  store: Store<TunerState, TunerAction>;
  profileKey: string | null;
  navigate: (route: TunerRoute) => void;
}> = (props) => {
  const state = props.store.getState();
  const dispatch = props.store.dispatch;
  const isCreate = (): boolean => props.profileKey === null;

  const [draft, setDraft] = createSignal<ProfileDraft>(emptyProfileDraft());
  const [keyText, setKeyText] = createSignal(props.profileKey ?? "");
  const [keyEdited, setKeyEdited] = createSignal(!isCreate());
  const [seededKey, setSeededKey] = createSignal<string | null>(isCreate() ? "" : null);
  const [warnings, setWarnings] = createSignal<string[]>([]);

  const tunableGames = createMemo(() => peek(state().tunableGames) ?? []);
  const objectives = createMemo(() => peek(state().objectives) ?? []);
  const schema = createMemo<ParamSchema>(
    () => tunableGames().find((k) => k.game === draft().gameKind)?.tuner ?? EMPTY_SCHEMA,
  );

  // Seed the draft from the loaded detail once, in edit mode.
  createEffect(() => {
    if (isCreate()) return;
    const detail = peek(state().profileDetail);
    if (!detail || detail.key !== props.profileKey) return;
    if (seededKey() === detail.key) return;
    const probe = draftFromProfileContent(detail.content, EMPTY_SCHEMA);
    const sch = tunableGames().find((k) => k.game === probe.draft.gameKind)?.tuner ?? EMPTY_SCHEMA;
    const parsed = draftFromProfileContent(detail.content, sch);
    setDraft(parsed.draft);
    setWarnings(parsed.warnings);
    setSeededKey(detail.key);
  });

  // In create mode, keep the key suggestion tracking the profile id.
  createEffect(() => {
    if (!keyEdited()) setKeyText(profileSlugKey(draft().profileId));
  });

  // Default the game / objective once the metadata loads (create mode).
  createEffect(() => {
    if (draft().gameKind === "" && tunableGames().length > 0) {
      setDraft((d) => ({ ...d, gameKind: tunableGames()[0]!.game }));
    }
  });
  const matchingObjectives = createMemo(() => {
    const k = draft().gameKind;
    return objectives().filter((o) => !k || o.game_kind == null || o.game_kind === k);
  });
  createEffect(() => {
    const opts = matchingObjectives();
    if (
      (draft().objectiveKey === "" || !opts.some((o) => o.key === draft().objectiveKey)) &&
      opts.length > 0
    ) {
      setDraft((d) => ({ ...d, objectiveKey: opts[0]!.key }));
    }
  });

  const clientErrors = createMemo(() => validateProfileDraft(draft(), schema()));
  const effectiveKey = (): string => (isCreate() ? keyText().trim() : props.profileKey!);
  const saveStatus = (): TunerState["profileSave"]["status"] => state().profileSave.status;
  const validation = () => peek(state().profileValidation);

  // Navigate back once a save lands.
  let navigatedSave = false;
  createEffect(() => {
    if (saveStatus() === "done" && !navigatedSave) {
      navigatedSave = true;
      props.navigate({ view: "profiles" });
    }
  });

  function setEffort(phase: ProfilePhase, patch: Partial<{ value: string; unit: EffortUnit }>): void {
    setDraft((d) => ({
      ...d,
      efforts: { ...d.efforts, [phase]: { ...d.efforts[phase], ...patch } },
    }));
  }
  function setBudget(key: keyof ProfileDraft["budgets"], value: string): void {
    setDraft((d) => ({ ...d, budgets: { ...d.budgets, [key]: value } }));
  }

  function save(): void {
    if (clientErrors().length > 0 || effectiveKey() === "") return;
    dispatch({
      tag: "saveProfile",
      key: effectiveKey(),
      content: draftToProfileContent(draft(), schema()),
    });
  }
  function validateOnServer(): void {
    dispatch({
      tag: "validateProfile",
      key: effectiveKey(),
      content: draftToProfileContent(draft(), schema()),
    });
  }

  return (
    <div class="tuner-profile-editor" data-testid="tuner-profile-editor">
      <button class="tuner-back" onClick={() => props.navigate({ view: "profiles" })}>
        ← Profiles
      </button>
      <h3>{isCreate() ? "New launch profile" : `Edit ${props.profileKey}`}</h3>

      <Show when={warnings().length > 0}>
        <ul class="tuner-launch-hint" data-testid="profile-editor-warnings">
          <For each={warnings()}>{(w) => <li>{w}</li>}</For>
        </ul>
      </Show>

      <div class="tuner-launch-grid">
        <label>
          Profile id
          <input
            type="text"
            data-testid="profile-id-input"
            value={draft().profileId}
            onInput={(e) => setDraft((d) => ({ ...d, profileId: e.currentTarget.value }))}
          />
        </label>
        <label>
          Game
          <select
            data-testid="profile-game"
            value={draft().gameKind}
            onInput={(e) => setDraft((d) => ({ ...d, gameKind: e.currentTarget.value }))}
          >
            <Show when={draft().gameKind === ""}>
              <option value="">(pick a game)</option>
            </Show>
            <For each={tunableGames()}>{(k) => <option value={k.game}>{k.game}</option>}</For>
          </select>
        </label>
        <label>
          Objective
          <select
            data-testid="profile-objective"
            value={draft().objectiveKey}
            onInput={(e) => setDraft((d) => ({ ...d, objectiveKey: e.currentTarget.value }))}
          >
            <Show when={matchingObjectives().length === 0}>
              <option value="">(no objective for this game)</option>
            </Show>
            <For each={matchingObjectives()}>
              {(o) => (
                <option value={o.key}>
                  {o.objective_id ?? o.key}
                  {o.game_kind ? ` (${o.game_kind})` : ""}
                </option>
              )}
            </For>
          </select>
        </label>
        <Show when={isCreate()}>
          <label>
            File key
            <input
              type="text"
              data-testid="profile-key-input"
              value={keyText()}
              onInput={(e) => {
                setKeyEdited(true);
                setKeyText(e.currentTarget.value);
              }}
            />
          </label>
        </Show>
      </div>

      <h4>Budgets</h4>
      <div class="tuner-launch-grid">
        <For each={BUDGET_FIELDS}>
          {([key, label]) => (
            <label>
              {label}
              <input
                type="number"
                data-testid={`profile-budget-${key}`}
                value={draft().budgets[key]}
                onInput={(e) => setBudget(key, e.currentTarget.value)}
              />
            </label>
          )}
        </For>
      </div>

      <h4>Per-phase search effort</h4>
      <fieldset class="tuner-launch-effort" data-testid="profile-effort-rows">
        <For each={PROFILE_PHASES}>
          {(phase) => (
            <div class="tuner-launch-effort-row">
              <span class="tuner-launch-effort-phase">{phase}</span>
              <input
                type="number"
                data-testid={`profile-effort-${phase}-value`}
                placeholder="CLI default"
                value={draft().efforts[phase].value}
                onInput={(e) => setEffort(phase, { value: e.currentTarget.value })}
              />
              <select
                data-testid={`profile-effort-${phase}-unit`}
                value={draft().efforts[phase].unit}
                onInput={(e) => setEffort(phase, { unit: e.currentTarget.value as EffortUnit })}
              >
                <option value="iterations">iterations</option>
                <option value="time_ms">time (ms)</option>
              </select>
            </div>
          )}
        </For>
      </fieldset>

      <Show when={schema().parameters.length > 0}>
        <fieldset class="tuner-launch-overrides" data-testid="profile-constraint-fieldset">
          <legend>Constrain parameters</legend>
          <p class="tuner-launch-hint">
            Narrow the tuning space this profile launches with — fix a value,
            restrict a range, or drop choices. Bounds come from the game's
            schema; the launch preflight has the final say.
          </p>
          <ConstraintEditor
            schema={schema()}
            rows={draft().constraintRows}
            onChange={(rows) => setDraft((d) => ({ ...d, constraintRows: rows }))}
          />
        </fieldset>
      </Show>

      <div class="tuner-objective-validation" data-testid="profile-validation">
        <Show
          when={clientErrors().length > 0}
          fallback={<p class="tuner-objective-caption">No client-side problems.</p>}
        >
          <ul class="launch-error" role="alert">
            <For each={clientErrors()}>{(e) => <li>{e}</li>}</For>
          </ul>
        </Show>
        <button
          type="button"
          onClick={validateOnServer}
          disabled={state().profileValidation.status === "loading" || clientErrors().length > 0}
        >
          Validate on server
        </button>
        <Show when={validation()}>
          {(v) => (
            <p data-testid="profile-server-validation">
              <Show
                when={v().ok}
                fallback={<span class="launch-error">{v().errors.join("; ")}</span>}
              >
                server accepted
              </Show>
            </p>
          )}
        </Show>
      </div>

      <Show when={state().profileSave.status === "error" && state().profileSave.error}>
        <div class="launch-error" role="alert">
          {state().profileSave.error}
        </div>
      </Show>

      <div class="tuner-objective-editor-footer">
        <button
          type="button"
          id="profile-save"
          disabled={
            clientErrors().length > 0 || effectiveKey() === "" || saveStatus() === "pending"
          }
          onClick={save}
        >
          {saveStatus() === "pending" ? "Saving…" : "Save"}
        </button>
        <button
          type="button"
          class="tuner-back"
          onClick={() => props.navigate({ view: "profiles" })}
        >
          Cancel
        </button>
      </div>
    </div>
  );
};
