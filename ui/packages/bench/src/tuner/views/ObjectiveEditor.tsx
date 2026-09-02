// ObjectiveEditor — build or edit a frozen objective from the UI
// (`#/tuner/objectives/new` or `.../objectives/<key>`). The form state is a
// pure `ObjectiveDraft` (see `models/objective-model.ts`), not the wire JSON;
// this view is the glue: it renders the opponent panel, drives the config
// form off the game's tuner schema, runs the client-side validator on every
// keystroke, and dispatches `saveObjective` / `validateObjective`.

import { createEffect, createMemo, createSignal, For, Show, type Component } from "solid-js";
import type { Store } from "@mcts/core";
import { peek } from "../remote-data.js";
import type { TunerAction, TunerState } from "../tuner-reducer.js";
import type { TunerRoute } from "../tuner-routes.js";
import type { JsonValue, TunerInfo, TunerParameter } from "../../types.js";
import {
  activeParamNames,
  blankInlineOpponent,
  draftFromContent,
  draftToContent,
  emptyDraft,
  slugKey,
  validateDraft,
  type ObjectiveDraft,
  type OpponentDraft,
} from "../models/objective-model.js";

function paramDefault(p: TunerParameter): JsonValue | undefined {
  return (p.default !== undefined ? p.default : p.value) as JsonValue | undefined;
}

export const ObjectiveEditor: Component<{
  store: Store<TunerState, TunerAction>;
  objectiveKey: string | null;
  game?: string;
  navigate: (route: TunerRoute) => void;
}> = (props) => {
  const state = props.store.getState();
  const dispatch = props.store.dispatch;
  const isCreate = (): boolean => props.objectiveKey === null;

  const [draft, setDraft] = createSignal<ObjectiveDraft>(emptyDraft(props.game ?? ""));
  const [keyText, setKeyText] = createSignal(props.objectiveKey ?? "");
  const [keyEdited, setKeyEdited] = createSignal(!isCreate());
  const [seededKey, setSeededKey] = createSignal<string | null>(isCreate() ? "" : null);
  const [warnings, setWarnings] = createSignal<string[]>([]);

  // Seed the draft from the loaded detail once, in edit mode.
  createEffect(() => {
    if (isCreate()) return;
    const detail = peek(state().objectiveDetail);
    if (!detail || detail.key !== props.objectiveKey) return;
    if (seededKey() === detail.key) return;
    const parsed = draftFromContent(detail.content, props.game ?? "");
    setDraft(parsed.draft);
    setWarnings(parsed.warnings);
    setSeededKey(detail.key);
  });

  // In create mode, keep the key suggestion tracking the objective id.
  createEffect(() => {
    if (!keyEdited()) setKeyText(slugKey(draft().objectiveId));
  });

  const kinds = createMemo(() => peek(state().kinds) ?? []);
  const schema = createMemo<TunerInfo | null>(
    () => kinds().find((k) => k.game === draft().gameKind)?.tuner ?? null,
  );
  const clientErrors = createMemo(() => validateDraft(draft(), schema()));
  const hasInlineConfig = createMemo(() =>
    draft().opponents.some((o) => o.kind === "inline" && Object.keys(o.config).length > 0),
  );

  const effectiveKey = (): string => (isCreate() ? keyText().trim() : props.objectiveKey!);
  const saveStatus = (): TunerState["objectiveSave"]["status"] => state().objectiveSave.status;
  const validation = () => peek(state().objectiveValidation);

  // Navigate back once a save lands.
  let navigatedSave = false;
  createEffect(() => {
    if (saveStatus() === "done" && !navigatedSave) {
      navigatedSave = true;
      props.navigate({ view: "objectives" });
    }
  });

  function patchOpponent(index: number, patch: Partial<OpponentDraft>): void {
    setDraft((d) => ({
      ...d,
      opponents: d.opponents.map((o, i) => (i === index ? { ...o, ...patch } : o)),
    }));
  }

  function setParam(index: number, name: string, value: JsonValue | undefined): void {
    setDraft((d) => ({
      ...d,
      opponents: d.opponents.map((o, i) => {
        if (i !== index) return o;
        const config = { ...o.config };
        if (value === undefined || value === "") delete config[name];
        else config[name] = value;
        return { ...o, config, configText: JSON.stringify(config, null, 2) };
      }),
    }));
  }

  function syncRaw(index: number): void {
    setDraft((d) => ({
      ...d,
      opponents: d.opponents.map((o, i) => {
        if (i !== index) return o;
        try {
          const parsed = JSON.parse(o.configText);
          if (parsed && typeof parsed === "object" && !Array.isArray(parsed)) {
            return { ...o, config: parsed as Record<string, JsonValue> };
          }
        } catch {
          /* leave the buffer; validateDraft surfaces the parse error */
        }
        return o;
      }),
    }));
  }

  function addOpponent(): void {
    setDraft((d) => ({
      ...d,
      opponents: [...d.opponents, blankInlineOpponent(d.opponents.length)],
    }));
  }

  function removeOpponent(index: number): void {
    setDraft((d) => ({ ...d, opponents: d.opponents.filter((_, i) => i !== index) }));
  }

  function save(): void {
    if (clientErrors().length > 0) return;
    dispatch({ tag: "saveObjective", key: effectiveKey(), content: draftToContent(draft()) });
  }

  function validateOnServer(): void {
    dispatch({ tag: "validateObjective", key: effectiveKey(), content: draftToContent(draft()) });
  }

  const renderField = (opponent: OpponentDraft, index: number, p: TunerParameter) => {
    const current = opponent.config[p.name];
    const def = paramDefault(p);
    if (p.type === "categorical") {
      return (
        <select
          value={current === undefined ? "" : String(current)}
          onInput={(e) =>
            setParam(index, p.name, e.currentTarget.value === "" ? undefined : e.currentTarget.value)
          }
        >
          <option value="">(default{def !== undefined ? `: ${String(def)}` : ""})</option>
          <For each={p.choices ?? []}>{(c) => <option value={c}>{c}</option>}</For>
        </select>
      );
    }
    if (p.type === "bool") {
      return (
        <input
          type="checkbox"
          checked={current === true || (current === undefined && def === true)}
          onInput={(e) => setParam(index, p.name, e.currentTarget.checked)}
        />
      );
    }
    if (p.type === "constant") {
      return <input type="text" readOnly value={def === undefined ? "" : String(def)} />;
    }
    // float / int
    return (
      <input
        type="number"
        step={p.type === "int" ? 1 : "any"}
        min={p.bounds?.[0]}
        max={p.bounds?.[1]}
        placeholder={def === undefined ? "" : String(def)}
        value={current === undefined ? "" : String(current)}
        onInput={(e) => {
          const raw = e.currentTarget.value;
          if (raw === "") return setParam(index, p.name, undefined);
          const n = Number(raw);
          if (Number.isFinite(n)) setParam(index, p.name, p.type === "int" ? Math.trunc(n) : n);
        }}
      />
    );
  };

  const configForm = (opponent: OpponentDraft, index: number) => {
    const sch = schema();
    if (!sch || sch.parameters.length === 0) {
      return (
        <p class="tuner-fleet-empty">
          No tuner schema for this game — use raw JSON.
        </p>
      );
    }
    const active = activeParamNames(sch, opponent.config);
    return (
      <div class="tuner-objective-param-grid">
        <For each={sch.parameters.filter((p) => active.has(p.name))}>
          {(p) => (
            <label class="tuner-objective-param">
              <span>{p.name}</span>
              {renderField(opponent, index, p)}
            </label>
          )}
        </For>
      </div>
    );
  };

  return (
    <div class="tuner-objective-editor" data-testid="tuner-objective-editor">
      <button class="tuner-back" onClick={() => props.navigate({ view: "objectives" })}>
        ← Objectives
      </button>
      <h3>{isCreate() ? "New objective" : `Edit ${props.objectiveKey}`}</h3>

      <Show when={warnings().length > 0}>
        <ul class="tuner-launch-hint" data-testid="objective-editor-warnings">
          <For each={warnings()}>{(w) => <li>{w}</li>}</For>
        </ul>
      </Show>

      <div class="tuner-launch-grid">
        <label>
          Objective id
          <input
            type="text"
            data-testid="objective-id-input"
            value={draft().objectiveId}
            onInput={(e) => setDraft((d) => ({ ...d, objectiveId: e.currentTarget.value }))}
          />
        </label>
        <label>
          Game
          <select
            value={draft().gameKind}
            disabled={hasInlineConfig()}
            onInput={(e) => setDraft((d) => ({ ...d, gameKind: e.currentTarget.value }))}
          >
            <Show when={draft().gameKind === ""}>
              <option value="">(pick a game)</option>
            </Show>
            <For each={kinds()}>{(k) => <option value={k.game}>{k.game}</option>}</For>
          </select>
        </label>
        <Show when={isCreate()}>
          <label>
            File key
            <input
              type="text"
              data-testid="objective-key-input"
              value={keyText()}
              onInput={(e) => {
                setKeyEdited(true);
                setKeyText(e.currentTarget.value);
              }}
            />
          </label>
        </Show>
      </div>
      <Show when={hasInlineConfig()}>
        <p class="tuner-launch-hint">
          Game is locked while opponents carry inline config (changing it would invalidate the
          schema). Clear the inline configs to switch games.
        </p>
      </Show>

      <h4>Opponent panel</h4>
      <For each={draft().opponents}>
        {(opponent, index) => (
          <div class="tuner-objective-opponent" data-testid="objective-opponent">
            <Show
              when={opponent.kind === "inline"}
              fallback={
                <div class="tuner-objective-opponent-head">
                  <strong>Schema default</strong>
                  <label>
                    Weight
                    <input
                      type="number"
                      min="1"
                      step="1"
                      value={String(opponent.weight)}
                      onInput={(e) =>
                        patchOpponent(index(), { weight: Math.trunc(Number(e.currentTarget.value)) })
                      }
                    />
                  </label>
                  <span class="tuner-objective-caption">
                    role: default — config is the game's schema default
                  </span>
                </div>
              }
            >
              <div class="tuner-objective-opponent-head">
                <label>
                  Id
                  <input
                    type="text"
                    value={opponent.id}
                    onInput={(e) => patchOpponent(index(), { id: e.currentTarget.value })}
                  />
                </label>
                <label>
                  Label
                  <input
                    type="text"
                    value={opponent.label}
                    onInput={(e) => patchOpponent(index(), { label: e.currentTarget.value })}
                  />
                </label>
                <label>
                  Weight
                  <input
                    type="number"
                    min="1"
                    step="1"
                    value={String(opponent.weight)}
                    onInput={(e) =>
                      patchOpponent(index(), { weight: Math.trunc(Number(e.currentTarget.value)) })
                    }
                  />
                </label>
                <button
                  type="button"
                  class="tuner-objective-delete-confirm"
                  onClick={() => removeOpponent(index())}
                >
                  Remove
                </button>
              </div>

              <div class="tuner-objective-config">
                <div class="tuner-config-diff-toggle">
                  <button
                    type="button"
                    classList={{ "tuner-toggle-active": opponent.configMode === "form" }}
                    onClick={() => patchOpponent(index(), { configMode: "form" })}
                  >
                    Form
                  </button>
                  <button
                    type="button"
                    classList={{ "tuner-toggle-active": opponent.configMode === "raw" }}
                    onClick={() =>
                      patchOpponent(index(), {
                        configMode: "raw",
                        configText: JSON.stringify(opponent.config, null, 2),
                      })
                    }
                  >
                    Raw JSON
                  </button>
                </div>
                <Show
                  when={opponent.configMode === "form"}
                  fallback={
                    <textarea
                      class="tuner-objective-editor-raw"
                      rows="6"
                      data-testid="objective-config-raw"
                      value={opponent.configText}
                      onInput={(e) => patchOpponent(index(), { configText: e.currentTarget.value })}
                      onBlur={() => syncRaw(index())}
                    />
                  }
                >
                  {configForm(opponent, index())}
                </Show>
              </div>
            </Show>
          </div>
        )}
      </For>
      <button type="button" class="tuner-fleet-new" onClick={addOpponent}>
        Add opponent
      </button>

      <p class="tuner-objective-caption">start distribution: default only (the only supported value)</p>

      <div class="tuner-objective-validation" data-testid="objective-validation">
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
          disabled={state().objectiveValidation.status === "loading"}
        >
          Validate on server
        </button>
        <Show when={validation()}>
          {(v) => (
            <p data-testid="objective-server-validation">
              <Show
                when={v().ok}
                fallback={<span class="launch-error">{v().errors.join("; ")}</span>}
              >
                server accepted
                {v().panel_fingerprint ? ` — panel ${v().panel_fingerprint}` : ""}
              </Show>
            </p>
          )}
        </Show>
      </div>

      <Show when={state().objectiveSave.status === "error" && state().objectiveSave.error}>
        <div class="launch-error" role="alert">
          {state().objectiveSave.error}
        </div>
      </Show>

      <div class="tuner-fleet-actions">
        <button
          type="button"
          id="objective-save"
          disabled={
            clientErrors().length > 0 ||
            effectiveKey() === "" ||
            saveStatus() === "pending"
          }
          onClick={save}
        >
          {saveStatus() === "pending" ? "Saving…" : "Save"}
        </button>
        <button type="button" class="tuner-back" onClick={() => props.navigate({ view: "objectives" })}>
          Cancel
        </button>
      </div>
    </div>
  );
};
