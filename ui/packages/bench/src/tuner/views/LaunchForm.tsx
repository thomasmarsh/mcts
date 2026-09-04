// LaunchForm — rebuilt for the version-4 tuner. Every option is driven by
// server configuration: the game picker from `GET /api/bench/tuner/kinds`
// (built-in `game-<kind>` binaries), the objective picker from `GET
// /api/bench/tuner/objectives` (the server's configured objectives dir). A
// launch request carries a `game_kind` + `objective_key`, never a
// filesystem path.

import {
  createEffect,
  createMemo,
  createSignal,
  For,
  onCleanup,
  Show,
  untrack,
  type Component,
} from "solid-js";
import type { Store } from "@mcts/core";
import { peek } from "../remote-data.js";
import type { TunerAction, TunerState } from "../tuner-reducer.js";
import type { TunerLaunchRequest } from "../tuner-types.js";
import type { TunerRoute } from "../tuner-routes.js";
import { summarizeRunPlan } from "../models/run-plan-model.js";
import { ConstraintEditor } from "./ConstraintEditor.js";
import {
  deriveConstraints,
  emptyRows,
  type ConstraintRows,
  type ParamSchema,
} from "../models/constraint-editor-model.js";
import {
  draftFromProfileContent,
  draftToProfileContent,
  type ProfileDraft,
} from "../models/profile-model.js";
import { OpponentPanelTable } from "../primitives/OpponentPanelTable.js";
import { KpiRow } from "../primitives/KpiRow.js";

// These mirror the tuner CLI's own defaults (cohort 8, finalists 3,
// tuning_pairs 4). They satisfy every launch constraint for a panel whose
// opponent weights are all 1 (total weight 1 or 2): tuning_pair_budget covers
// one initial cohort (8 x 4), validation_pair_budget is a multiple of
// finalists, and validation_pair_budget / finalists is both <= the production
// count and a multiple of the panel weight. A panel with larger opponent
// weights needs all three scaled up to stay divisible by that weight.
const DEFAULTS = {
  task_seed: 1,
  tuning_pair_budget: 32,
  validation_pair_budget: 24,
  production_validation_pairs: 8,
};

/** `1` → `1`, `""` → undefined, junk → undefined. */
function optInt(raw: string): number | undefined {
  const t = raw.trim();
  if (t === "") return undefined;
  const n = Number(t);
  return Number.isFinite(n) ? Math.trunc(n) : undefined;
}

/** The three search-effort phases, each an either/or iterations-or-time pair
 * on the Rust `TunerLaunchRequest`. */
const EFFORT_PHASES = ["tuning", "validation", "production"] as const;
type EffortPhase = (typeof EFFORT_PHASES)[number];
type EffortUnit = "iterations" | "time_ms";

const PROPOSER_POLICIES = ["smac_mixed", "random", "qmc", "irace_generational"] as const;

function suggestRunId(kind: string, objectiveKey: string): string {
  const stamp = new Date()
    .toISOString()
    .replace(/[-:]/g, "")
    .replace(/\.\d+Z$/, "Z");
  const base = objectiveKey || kind || "tuner";
  return `${base}-${stamp}`;
}

export const LaunchForm: Component<{
  store: Store<TunerState, TunerAction>;
  navigate?: (route: TunerRoute) => void;
}> = (props) => {
  const state = props.store.getState();
  const dispatch = props.store.dispatch;

  const tunableGames = createMemo(() => peek(state().tunableGames) ?? []);
  const objectives = createMemo(() => peek(state().objectives) ?? []);
  const profiles = createMemo(() => peek(state().profiles) ?? []);

  const [gameKind, setGameKind] = createSignal("");
  const [objectiveKey, setObjectiveKey] = createSignal("");
  const [runId, setRunId] = createSignal("");
  const [runIdEdited, setRunIdEdited] = createSignal(false);
  const [taskSeed, setTaskSeed] = createSignal(String(DEFAULTS.task_seed));
  const [tuningBudget, setTuningBudget] = createSignal(String(DEFAULTS.tuning_pair_budget));
  const [validationBudget, setValidationBudget] = createSignal(
    String(DEFAULTS.validation_pair_budget),
  );
  const [productionPairs, setProductionPairs] = createSignal(
    String(DEFAULTS.production_validation_pairs),
  );
  const [seed, setSeed] = createSignal("");
  const [cohortSize, setCohortSize] = createSignal("");
  const [finalists, setFinalists] = createSignal("");
  const [evaluatorWorkers, setEvaluatorWorkers] = createSignal("");
  const [proposerPolicy, setProposerPolicy] = createSignal("");
  // Per-phase effort: a value string + a unit toggle for each phase.
  const [effortValue, setEffortValue] = createSignal<Record<EffortPhase, string>>({
    tuning: "",
    validation: "",
    production: "",
  });
  const [effortUnit, setEffortUnit] = createSignal<Record<EffortPhase, EffortUnit>>({
    tuning: "iterations",
    validation: "iterations",
    production: "iterations",
  });
  const [showAdvanced, setShowAdvanced] = createSignal(false);

  // The picked game's tuning schema (`tuner.parameters` + `tuner.conditions`),
  // already shipped in `GET /api/bench/tuner/kinds`. Drives the constraint
  // editor entirely — no parameter or algorithm name is hardcoded here.
  const EMPTY_SCHEMA: ParamSchema = { parameters: [], conditions: [] };
  const schema = createMemo<ParamSchema>(
    () => tunableGames().find((k) => k.game === gameKind())?.tuner ?? EMPTY_SCHEMA,
  );

  // Run-scoped tuning-space constraints, authored row-by-row against the
  // schema. A different game has a different schema, so start fresh on a
  // game switch.
  const [constraintRows, setConstraintRows] = createSignal<ConstraintRows>({});
  // Set while a "start from profile" seed is switching the game, so the
  // game-switch reset below doesn't wipe the constraint rows the seed just
  // installed. Reset by that same effect on its next run.
  const [profileSeeding, setProfileSeeding] = createSignal(false);
  createEffect(() => {
    gameKind();
    if (untrack(profileSeeding)) {
      setProfileSeeding(false);
      return;
    }
    setConstraintRows(emptyRows(schema()));
  });
  const constraintResult = createMemo(() => deriveConstraints(schema(), constraintRows()));

  function setPhaseValue(phase: EffortPhase, raw: string): void {
    setEffortValue({ ...effortValue(), [phase]: raw });
  }
  function setPhaseUnit(phase: EffortPhase, unit: EffortUnit): void {
    setEffortUnit({ ...effortUnit(), [phase]: unit });
  }

  /** A filled, positive effort for a phase, as `{ unit, value }`, else null. */
  const phaseEffort = (phase: EffortPhase): { unit: EffortUnit; value: number } | null => {
    const n = optInt(effortValue()[phase]);
    if (n === undefined || n <= 0) return null;
    return { unit: effortUnit()[phase], value: n };
  };

  // The one effort rule an operator can check locally: a filled tuning or
  // validation effort of the *same unit* as a filled production effort must
  // not exceed it. Mixed units or blanks are the server's problem.
  const effortError = createMemo((): string | null => {
    const prod = phaseEffort("production");
    if (!prod) return null;
    for (const phase of ["tuning", "validation"] as const) {
      const e = phaseEffort(phase);
      if (e && e.unit === prod.unit && e.value > prod.value) {
        return `${phase} effort (${e.value}) cannot exceed production effort (${prod.value})`;
      }
    }
    return null;
  });

  // --- Start from / save as a launch profile ---------------------------
  const [fromProfileKey, setFromProfileKey] = createSignal("");
  const [savePanelOpen, setSavePanelOpen] = createSignal(false);
  const [saveKey, setSaveKey] = createSignal("");
  let seededProfileKey: string | null = null;

  function selectProfile(key: string): void {
    setFromProfileKey(key);
    seededProfileKey = null;
    if (key) dispatch({ tag: "openProfile", key });
  }

  /** The current form state as a `ProfileDraft`, for "save as profile". */
  const currentProfileDraft = (): ProfileDraft => ({
    profileId: saveKey().trim(),
    gameKind: gameKind(),
    objectiveKey: objectiveKey(),
    constraintRows: constraintRows(),
    efforts: {
      tuning: { value: effortValue().tuning, unit: effortUnit().tuning },
      validation: { value: effortValue().validation, unit: effortUnit().validation },
      production: { value: effortValue().production, unit: effortUnit().production },
    },
    budgets: {
      tuningPairBudget: tuningBudget(),
      validationPairBudget: validationBudget(),
      productionValidationPairs: productionPairs(),
      cohortSize: cohortSize(),
      finalists: finalists(),
    },
  });

  function saveAsProfile(): void {
    const key = saveKey().trim();
    if (key === "") return;
    dispatch({
      tag: "saveProfile",
      key,
      content: draftToProfileContent(currentProfileDraft(), schema()),
    });
  }

  // Seed every field from the selected profile once its detail loads. The
  // operator can still override anything before launching.
  createEffect(() => {
    const key = fromProfileKey();
    if (!key || seededProfileKey === key) return;
    const detail = peek(state().profileDetail);
    if (!detail || detail.key !== key) return;
    seededProfileKey = key;

    const probe = draftFromProfileContent(detail.content, EMPTY_SCHEMA);
    const sch = tunableGames().find((k) => k.game === probe.draft.gameKind)?.tuner ?? EMPTY_SCHEMA;
    const { draft } = draftFromProfileContent(detail.content, sch);

    if (draft.gameKind && draft.gameKind !== gameKind()) setProfileSeeding(true);
    if (draft.gameKind) setGameKind(draft.gameKind);
    if (draft.objectiveKey) setObjectiveKey(draft.objectiveKey);
    setConstraintRows(draft.constraintRows);
    setTuningBudget(draft.budgets.tuningPairBudget);
    setValidationBudget(draft.budgets.validationPairBudget);
    setProductionPairs(draft.budgets.productionValidationPairs);
    setCohortSize(draft.budgets.cohortSize);
    setFinalists(draft.budgets.finalists);
    setEffortValue({
      tuning: draft.efforts.tuning.value,
      validation: draft.efforts.validation.value,
      production: draft.efforts.production.value,
    });
    setEffortUnit({
      tuning: draft.efforts.tuning.unit,
      validation: draft.efforts.validation.unit,
      production: draft.efforts.production.unit,
    });
    if (draft.constraintRows && Object.keys(draft.constraintRows).length > 0) {
      setShowAdvanced(true);
    }
  });

  // Objectives whose `game_kind` matches the picked game come first; an
  // objective with no declared kind is always offered.
  const matchingObjectives = createMemo(() => {
    const k = gameKind();
    return objectives().filter((o) => !k || o.game_kind == null || o.game_kind === k);
  });

  // Default the game/objective once the metadata loads.
  createEffect(() => {
    if (gameKind() === "" && tunableGames().length > 0) setGameKind(tunableGames()[0]!.game);
  });
  createEffect(() => {
    const opts = matchingObjectives();
    if ((objectiveKey() === "" || !opts.some((o) => o.key === objectiveKey())) && opts.length > 0) {
      setObjectiveKey(opts[0]!.key);
    }
  });
  // Keep the suggested run id fresh until the operator types their own.
  createEffect(() => {
    if (!runIdEdited()) setRunId(suggestRunId(gameKind(), objectiveKey()));
  });

  const busy = () => state().launch.status === "pending";
  const launchError = () => (state().launch.status === "error" ? state().launch.error : null);
  const preflight = createMemo(() => state().preflight);
  const runPlan = createMemo(() => summarizeRunPlan(peek(state().runPlan)));
  const planPending = createMemo(() => state().runPlan.status === "loading");

  /** The launch request for the current form values, or `null` if a required
   * field is missing / not a positive integer. */
  const buildRequest = (): TunerLaunchRequest | null => {
    if (
      gameKind() === "" ||
      objectiveKey() === "" ||
      runId().trim() === "" ||
      optInt(taskSeed()) === undefined ||
      (optInt(tuningBudget()) ?? 0) <= 0 ||
      (optInt(validationBudget()) ?? 0) <= 0 ||
      (optInt(productionPairs()) ?? 0) <= 0 ||
      effortError() !== null ||
      constraintResult().errors.length > 0
    ) {
      return null;
    }
    const effortFields: Record<string, number> = {};
    for (const phase of EFFORT_PHASES) {
      const e = phaseEffort(phase);
      if (e) effortFields[`${phase}_max_${e.unit}`] = e.value;
    }
    return {
      game_kind: gameKind(),
      objective_key: objectiveKey(),
      run_id: runId().trim(),
      task_seed: optInt(taskSeed()) ?? DEFAULTS.task_seed,
      tuning_pair_budget: optInt(tuningBudget()) ?? DEFAULTS.tuning_pair_budget,
      validation_pair_budget: optInt(validationBudget()) ?? DEFAULTS.validation_pair_budget,
      production_validation_pairs:
        optInt(productionPairs()) ?? DEFAULTS.production_validation_pairs,
      ...(optInt(seed()) !== undefined ? { seed: optInt(seed()) } : {}),
      ...(optInt(cohortSize()) !== undefined ? { cohort_size: optInt(cohortSize()) } : {}),
      ...(optInt(finalists()) !== undefined ? { finalists: optInt(finalists()) } : {}),
      ...(optInt(evaluatorWorkers()) !== undefined
        ? { evaluator_workers: optInt(evaluatorWorkers()) }
        : {}),
      ...(proposerPolicy() !== "" ? { proposer_policy: proposerPolicy() } : {}),
      ...(constraintResult().constraints.length > 0
        ? { constraints: constraintResult().constraints }
        : {}),
      ...effortFields,
    };
  };

  const canLaunch = createMemo(
    () =>
      !busy() &&
      buildRequest() !== null &&
      preflight().status !== "checking" &&
      preflight().status !== "invalid",
  );

  // Dry-run the launch against the server whenever the form settles, so the
  // operator sees "validation pairs cannot exceed production validation
  // pairs" here instead of a dead run in the fleet.
  let debounce: ReturnType<typeof setTimeout> | undefined;
  onCleanup(() => clearTimeout(debounce));
  createEffect(() => {
    const request = buildRequest();
    clearTimeout(debounce);
    if (!request) {
      dispatch({ tag: "resetPreflight" });
      dispatch({ tag: "resetPlan" });
      return;
    }
    debounce = setTimeout(() => {
      dispatch({ tag: "preflight", request });
      dispatch({ tag: "planRun", request });
    }, 350);
  });

  function onSubmit(e: Event): void {
    e.preventDefault();
    const request = buildRequest();
    if (!request || !canLaunch()) return;
    dispatch({ tag: "launch", request });
  }

  return (
    <form id="tuner-launch-form" class="tuner-launch-form" onSubmit={onSubmit}>
      <h3>Launch a tuner run</h3>

      <Show when={profiles().length > 0}>
        <label class="tuner-launch-from-profile">
          Start from profile
          <select
            data-testid="from-profile"
            value={fromProfileKey()}
            onInput={(e) => selectProfile(e.currentTarget.value)}
            disabled={busy()}
          >
            <option value="">(none — blank form)</option>
            <For each={profiles()}>
              {(p) => <option value={p.key}>{p.profile_id ?? p.key}</option>}
            </For>
          </select>
        </label>
      </Show>

      <Show when={launchError()}>
        <div class="launch-error" role="alert">
          {launchError()}
        </div>
      </Show>

      <Show when={preflight().status === "invalid"}>
        <div class="launch-error" role="alert" data-testid="preflight-errors">
          <strong>This launch would fail:</strong>
          <ul>
            <For each={preflight().errors}>{(msg) => <li>{msg}</li>}</For>
          </ul>
        </div>
      </Show>
      <Show when={preflight().status === "error"}>
        <div class="tuner-launch-hint" role="status">
          Could not pre-check this launch ({preflight().error}). The server will still validate it.
        </div>
      </Show>

      <label>
        Game
        <select
          data-testid="game-kind"
          value={gameKind()}
          onInput={(e) => setGameKind(e.currentTarget.value)}
          disabled={busy()}
        >
          <For each={tunableGames()}>{(k) => <option value={k.game}>{k.game}</option>}</For>
        </select>
      </label>

      <label>
        Objective
        <select
          value={objectiveKey()}
          onInput={(e) => setObjectiveKey(e.currentTarget.value)}
          disabled={busy()}
        >
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
      <p class="tuner-launch-hint">
        <Show
          when={matchingObjectives().length === 0}
          fallback={<a href="#/tuner/objectives">Manage objectives</a>}
        >
          No objective for <strong>{gameKind() || "this game"}</strong> yet —{" "}
          <a
            href={`#/tuner/objectives/new${
              gameKind() ? `?game=${encodeURIComponent(gameKind())}` : ""
            }`}
          >
            create one
          </a>
          .
        </Show>
      </p>

      <label>
        Run id
        <input
          type="text"
          value={runId()}
          onInput={(e) => {
            setRunIdEdited(true);
            setRunId(e.currentTarget.value);
          }}
          disabled={busy()}
        />
      </label>

      <div class="tuner-launch-grid">
        <label>
          Task seed
          <input type="number" value={taskSeed()} onInput={(e) => setTaskSeed(e.currentTarget.value)} />
        </label>
        <label>
          Tuning pair budget
          <input
            type="number"
            value={tuningBudget()}
            onInput={(e) => setTuningBudget(e.currentTarget.value)}
          />
        </label>
        <label>
          Validation pair budget
          <input
            type="number"
            value={validationBudget()}
            onInput={(e) => setValidationBudget(e.currentTarget.value)}
          />
        </label>
        <label>
          Production validation pairs
          <input
            type="number"
            value={productionPairs()}
            onInput={(e) => setProductionPairs(e.currentTarget.value)}
          />
        </label>
      </div>

      <button
        type="button"
        class="tuner-launch-advanced-toggle"
        onClick={() => setShowAdvanced((v) => !v)}
      >
        {showAdvanced() ? "Hide" : "Show"} advanced options
      </button>
      <Show when={showAdvanced()}>
        <div class="tuner-launch-grid">
          <label>
            Proposer seed
            <input type="number" value={seed()} onInput={(e) => setSeed(e.currentTarget.value)} />
          </label>
          <label>
            Cohort size
            <input
              type="number"
              value={cohortSize()}
              onInput={(e) => setCohortSize(e.currentTarget.value)}
            />
          </label>
          <label>
            Finalists
            <input
              type="number"
              value={finalists()}
              onInput={(e) => setFinalists(e.currentTarget.value)}
            />
          </label>
          <label>
            Evaluator workers
            <input
              type="number"
              value={evaluatorWorkers()}
              onInput={(e) => setEvaluatorWorkers(e.currentTarget.value)}
            />
          </label>
          <label>
            Proposer policy
            <select
              data-testid="proposer-policy"
              value={proposerPolicy()}
              onInput={(e) => setProposerPolicy(e.currentTarget.value)}
            >
              <option value="">default (smac_mixed)</option>
              <For each={PROPOSER_POLICIES}>{(p) => <option value={p}>{p}</option>}</For>
            </select>
          </label>
        </div>

        <fieldset class="tuner-launch-effort" data-testid="effort-rows">
          <legend>Per-phase search effort</legend>
          <For each={EFFORT_PHASES}>
            {(phase) => (
              <div class="tuner-launch-effort-row">
                <span class="tuner-launch-effort-phase">{phase}</span>
                <input
                  type="number"
                  data-testid={`effort-${phase}-value`}
                  placeholder="CLI default"
                  value={effortValue()[phase]}
                  onInput={(e) => setPhaseValue(phase, e.currentTarget.value)}
                />
                <select
                  data-testid={`effort-${phase}-unit`}
                  value={effortUnit()[phase]}
                  onInput={(e) => setPhaseUnit(phase, e.currentTarget.value as EffortUnit)}
                >
                  <option value="iterations">iterations</option>
                  <option value="time_ms">time (ms)</option>
                </select>
              </div>
            )}
          </For>
          <Show when={effortError()}>
            <p class="launch-error" role="alert" data-testid="effort-error">
              {effortError()}
            </p>
          </Show>
        </fieldset>

        <Show when={schema().parameters.length > 0}>
          <fieldset class="tuner-launch-overrides" data-testid="constraint-editor-fieldset">
            <legend>Constrain parameters</legend>
            <p class="tuner-launch-hint">
              Narrow the tuning space for this run — fix a value, restrict a
              range, or drop choices (unticking every box off a categorical
              excludes an algorithm or variant). Bounds come from the game's
              schema; the launch preflight has the final say.
            </p>
            <ConstraintEditor
              schema={schema()}
              rows={constraintRows()}
              onChange={setConstraintRows}
            />
          </fieldset>
        </Show>
      </Show>

      <section class="tuner-run-plan" data-testid="run-plan">
        <h4>Run plan{planPending() ? " (resolving…)" : ""}</h4>
        <Show
          when={runPlan().resolved}
          fallback={
            <p class="tuner-launch-hint">
              {runPlan().errors.length > 0
                ? "Resolve the errors above to see the resolved run plan."
                : "Fill in the required fields to see what this run will do."}
            </p>
          }
        >
          <p class="tuner-launch-hint">
            {runPlan().gameKind} · objective <code>{runPlan().objectiveId}</code>
            <Show when={runPlan().epochFingerprint}>
              {" "}
              · epoch <code class="tuner-mono">{runPlan().epochFingerprint!.slice(0, 12)}</code>
            </Show>
          </p>

          <h5>Opponent panel</h5>
          <OpponentPanelTable opponents={runPlan().opponents} testid="run-plan-opponents" />

          <h5>Tuning space</h5>
          <p class="tuner-launch-hint" data-testid="run-plan-families">
            families: {runPlan().families.join(", ") || "—"}
            <Show when={runPlan().excludedFamilies.length > 0}>
              {" "}
              (excluded: {runPlan().excludedFamilies.join(", ")})
            </Show>
          </p>
          <ul class="tuner-run-plan-params">
            <For each={runPlan().parameters}>
              {(p) => (
                <li>
                  <code>{p.name}</code> — {p.kind}
                  <Show when={p.bounds}>
                    {" "}
                    [{p.bounds![0]}, {p.bounds![1]}]
                  </Show>
                  <Show when={p.choices}> {`{${p.choices!.join(", ")}}`}</Show>
                  <Show when={p.active_when}>
                    {" "}
                    <span class="tuner-launch-hint">when {p.active_when}</span>
                  </Show>
                </li>
              )}
            </For>
          </ul>

          <h5>Effort &amp; budget</h5>
          <KpiRow items={runPlan().effortKpis} testid="run-plan-efforts" />
          <KpiRow items={runPlan().budgetKpis} testid="run-plan-budgets" />

          <Show when={runPlan().gameConfigIsOverride}>
            <p class="tuner-launch-hint">
              game_config override: <code class="tuner-mono">{runPlan().gameConfig}</code>
            </p>
          </Show>
        </Show>
      </section>

      <section class="tuner-launch-save-profile">
        <Show
          when={savePanelOpen()}
          fallback={
            <button
              type="button"
              class="tuner-launch-advanced-toggle"
              data-testid="save-as-profile-toggle"
              onClick={() => {
                setSaveKey(fromProfileKey() || runId().trim());
                setSavePanelOpen(true);
              }}
            >
              Save these settings as a profile…
            </button>
          }
        >
          <div class="tuner-launch-grid">
            <label>
              Profile key
              <input
                type="text"
                data-testid="save-profile-key"
                value={saveKey()}
                onInput={(e) => setSaveKey(e.currentTarget.value)}
              />
            </label>
          </div>
          <Show when={state().profileSave.status === "error" && state().profileSave.error}>
            <p class="launch-error" role="alert" data-testid="save-profile-error">
              {state().profileSave.error}
            </p>
          </Show>
          <Show when={state().profileSave.status === "done"}>
            <p class="tuner-launch-hint" data-testid="save-profile-done">
              Saved profile <code>{saveKey().trim()}</code>.
              <Show when={props.navigate}>
                {" "}
                <a
                  href="#/tuner/profiles"
                  onClick={(e) => {
                    e.preventDefault();
                    props.navigate!({ view: "profiles" });
                  }}
                >
                  Manage profiles
                </a>
              </Show>
            </p>
          </Show>
          <button
            type="button"
            data-testid="save-profile-submit"
            disabled={
              saveKey().trim() === "" ||
              buildRequest() === null ||
              state().profileSave.status === "pending"
            }
            onClick={saveAsProfile}
          >
            {state().profileSave.status === "pending" ? "Saving…" : "Save profile"}
          </button>
          <button type="button" onClick={() => setSavePanelOpen(false)}>
            Cancel
          </button>
        </Show>
      </section>

      <button type="submit" id="tuner-launch-button" disabled={!canLaunch()}>
        {busy()
          ? "Launching…"
          : preflight().status === "checking"
            ? "Checking…"
            : "Launch"}
      </button>
    </form>
  );
};
