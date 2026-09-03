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
  type Component,
} from "solid-js";
import type { Store } from "@mcts/core";
import { peek } from "../remote-data.js";
import type { TunerAction, TunerState } from "../tuner-reducer.js";
import type { TunerLaunchRequest } from "../tuner-types.js";

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

export const LaunchForm: Component<{ store: Store<TunerState, TunerAction> }> = (props) => {
  const state = props.store.getState();
  const dispatch = props.store.dispatch;

  const kinds = createMemo(() => peek(state().kinds) ?? []);
  const objectives = createMemo(() => peek(state().objectives) ?? []);

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
  const [excludedFamilies, setExcludedFamilies] = createSignal<string[]>([]);
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

  // The tunable `family` categorical's choices for the picked game, sourced
  // from the schema already shipped in `GET /api/bench/tuner/kinds`. Empty
  // when the game has no `family` axis (nothing to exclude).
  const familyChoices = createMemo(() => {
    const info = kinds().find((k) => k.game === gameKind());
    const param = info?.tuner.parameters.find((p) => p.name === "family");
    return param?.choices ?? [];
  });

  // A different game has a different family list; drop stale exclusions.
  createEffect(() => {
    const choices = familyChoices();
    setExcludedFamilies((prev) => prev.filter((f) => choices.includes(f)));
  });

  function toggleFamily(family: string, checked: boolean): void {
    setExcludedFamilies((prev) =>
      checked ? [...prev, family] : prev.filter((f) => f !== family),
    );
  }

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

  const excludesEveryFamily = createMemo(
    () => familyChoices().length > 0 && excludedFamilies().length >= familyChoices().length,
  );

  // Objectives whose `game_kind` matches the picked game come first; an
  // objective with no declared kind is always offered.
  const matchingObjectives = createMemo(() => {
    const k = gameKind();
    return objectives().filter((o) => !k || o.game_kind == null || o.game_kind === k);
  });

  // Default the game/objective once the metadata loads.
  createEffect(() => {
    if (gameKind() === "" && kinds().length > 0) setGameKind(kinds()[0]!.game);
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
      excludesEveryFamily()
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
      ...(excludedFamilies().length > 0 ? { exclude_family: excludedFamilies() } : {}),
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
      return;
    }
    debounce = setTimeout(() => dispatch({ tag: "preflight", request }), 350);
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
          <For each={kinds()}>{(k) => <option value={k.game}>{k.game}</option>}</For>
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

        <Show when={familyChoices().length > 0}>
          <fieldset class="tuner-launch-families" data-testid="family-checklist">
            <legend>Excluded families</legend>
            <For each={familyChoices()}>
              {(family) => (
                <label class="tuner-launch-family">
                  <input
                    type="checkbox"
                    data-testid={`exclude-family-${family}`}
                    checked={excludedFamilies().includes(family)}
                    onChange={(e) => toggleFamily(family, e.currentTarget.checked)}
                  />
                  {family}
                </label>
              )}
            </For>
            <Show when={excludesEveryFamily()}>
              <p class="launch-error" role="alert" data-testid="exclude-all-error">
                A run must leave at least one family available.
              </p>
            </Show>
          </fieldset>
        </Show>
      </Show>

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
