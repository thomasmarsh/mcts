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
  const [showAdvanced, setShowAdvanced] = createSignal(false);

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
      (optInt(productionPairs()) ?? 0) <= 0
    ) {
      return null;
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
        </div>
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
