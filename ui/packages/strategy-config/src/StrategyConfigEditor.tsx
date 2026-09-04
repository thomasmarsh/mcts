// StrategyConfigEditor.tsx — an interactive, schema-driven editor for
// `config_ir::SearchSpec`'s free composition of the four MCTS axes (select/
// simulate/backprop/final_action), plus the surrounding `CustomStrategySpec`
// budget/thread/transposition-table fields. Game-agnostic: `config_ir`'s
// shape doesn't depend on which game is being configured, so unlike
// `NewGameFields` this takes no `GameKindModule` slot.
//
// Entirely schema-driven off `AxisSchema` (`GET /api/strategy-schema`,
// `mcts_tune::config_ir_schema::axis_schema()`) rather than hand-coding any
// per-algorithm knowledge -- the same variant-picker logic renders a top-level
// axis (`select`/`simulate`/`backprop`/`final_action`), a wrapped inner spec
// (`epsilon_greedy`'s `inner: BaseSelectSpec`), and a nested non-axis enum
// field (`RaveSchedule`/`RaveUcb`/`DecisiveMoveMode`) -- because all three
// are the same shape in `AxisSchema`: pick a `kind`, render that variant's
// fields. Recursion is bounded to exactly the one level `config_ir.rs`
// itself allows: only variants carrying a `wraps` key recurse into another
// axis's variant list, and neither `select_base` nor `simulate_base` (the
// only `wraps` targets) contains a further wrapping variant, so
// `VariantPicker` terminates without needing an explicit depth guard.

import "./strategy-config.css";

import { createMemo, createSignal, For, Show, type Component } from "solid-js";
import type {
  AxisFieldSchema,
  AxisSchema,
  AxisVariantSchema,
  BackpropSpec,
  CustomStrategySpec,
  FinalActionSpec,
  SearchSpec,
  SelectSpec,
  SimulateSpec,
  TunerInfo,
  TunerParameter,
} from "@mcts/game";
import { activeNames, withDefaultsFilled } from "./axis-activation.js";

/** A variant value as this editor manipulates it -- `kind` plus whatever
 * fields that variant's schema entry declares, generically. Each concrete
 * axis (`SelectSpec`, `SimulateSpec`, ...) is a narrower union of exactly
 * this shape; the schema-driven editor can't statically know which one it's
 * building; `StrategyConfigEditor`'s axis-level callers cast the result back
 * to the concrete field it's storing into `search`. */
export type VariantValue = { kind: string; [field: string]: unknown };

function findVariant(variants: AxisVariantSchema[], kind: string): AxisVariantSchema {
  return variants.find((v) => v.kind === kind) ?? variants[0]!;
}

/** Build a fresh, fully-defaulted value for `variant` -- used both to seed a
 * newly-selected variant's own scalar fields and, for a `wraps` variant, its
 * nested `inner` spec (defaulted to `schema[wraps].variants[0]`, since a
 * `wraps` target has no axis-level "default variant" of its own in the
 * schema, only per-field defaults). */
function buildDefaultVariantValue(variant: AxisVariantSchema, schema: AxisSchema): VariantValue {
  const value: VariantValue = { kind: variant.kind };
  for (const field of variant.fields) {
    value[field.name] = buildDefaultFieldValue(field, schema);
  }
  if (variant.wraps) {
    const innerVariants = schema[variant.wraps].variants;
    value.inner = buildDefaultVariantValue(innerVariants[0]!, schema);
  }
  return value;
}

function buildDefaultFieldValue(field: AxisFieldSchema, schema: AxisSchema): unknown {
  if (field.type === "enum") {
    // A `bare` enum's value *is* the plain variant-name string -- see
    // `AxisFieldSchema`'s doc comment. Only a real tagged union
    // (`RaveSchedule`/`RaveUcb`) needs the `{kind, ...fields}` shape
    // `buildDefaultVariantValue` builds.
    if (field.bare) return field.default;
    return buildDefaultVariantValue(findVariant(field.variants, field.default), schema);
  }
  return field.default;
}

/** A freshly-defaulted `SearchSpec` -- each axis seeded from its schema's
 * first listed variant. Shared by `defaultCustomStrategySpec` (initial
 * config) and by switching from by-algorithm mode back to free composition
 * (same reset-to-first-variant behavior either way). */
function buildDefaultSearchSpec(schema: AxisSchema): SearchSpec {
  return {
    select: buildDefaultVariantValue(schema.select.variants[0]!, schema) as unknown as SelectSpec,
    simulate: buildDefaultVariantValue(
      schema.simulate.variants[0]!,
      schema,
    ) as unknown as SimulateSpec,
    backprop: buildDefaultVariantValue(
      schema.backprop.variants[0]!,
      schema,
    ) as unknown as BackpropSpec,
    final_action: buildDefaultVariantValue(
      schema.final_action.variants[0]!,
      schema,
    ) as unknown as FinalActionSpec,
  };
}

/** A freshly-defaulted `CustomStrategySpec` -- each axis seeded from its
 * schema's first listed variant (`buildDefaultVariantValue`, the same
 * defaulting this editor uses when a variant is newly selected), budgeted by
 * *time* rather than iteration count, and running on all cores. This
 * mirrors every real preset in each game's `presets.json` (`easy`/`medium`/
 * `strong`/`master` -- all `max_time_ms`, never `max_iterations`, and
 * `threads: 0` for the NST-based `strong`/`master`), for the same reason:
 * a user-composed axis combination's per-iteration cost is unknowable ahead
 * of time (`decisive_move_nst`'s bigram lookups alone are ~15x a plain
 * `uniform` simulate's cost per iteration, measured against Druid), so an
 * iteration cap can turn into an unpredictable, effectively unbounded wall-
 * clock wait -- which is exactly what "Custom…" used to do at its own
 * defaults (10,000 iterations, single-threaded) before this default
 * existed. `1000`ms matches this editor's own fallback when a user
 * manually re-checks "Time limit" with no prior value (see `timeMs` below)
 * -- the same "reasonable first guess" used in both places. The one shared
 * place callers (e.g. `GameShell`'s New Game dialog) seed a new seat's
 * "Custom…" config, so that shape isn't hand-duplicated per caller. */
export function defaultCustomStrategySpec(schema: AxisSchema): CustomStrategySpec {
  return {
    search: buildDefaultSearchSpec(schema),
    max_time_ms: 1000,
    threads: 0,
  };
}

/** Recursive variant-kind picker: a `<select>` over `variants`, the chosen
 * variant's own scalar/enum fields, and (only for a `wraps` variant) a
 * nested `VariantPicker` for its `inner` spec. Used both at the top level
 * (an axis of `SearchSpec`) and for a nested non-axis enum field
 * (`RaveSchedule`/`RaveUcb`/`DecisiveMoveMode`) -- both are "pick a kind,
 * render that variant's fields" against an `AxisVariantSchema[]`. */
const VariantPicker: Component<{
  label: string;
  variants: AxisVariantSchema[];
  schema: AxisSchema;
  value: VariantValue;
  onChange: (value: VariantValue) => void;
}> = (props) => {
  const current = createMemo(() => findVariant(props.variants, props.value.kind));

  function setField(name: string, fieldValue: unknown) {
    props.onChange({ ...props.value, [name]: fieldValue });
  }

  return (
    <div class="strategy-variant-picker">
      <label>
        {props.label}
        <select
          value={current().kind}
          onChange={(e) => {
            const next = findVariant(props.variants, e.currentTarget.value);
            props.onChange(buildDefaultVariantValue(next, props.schema));
          }}
        >
          <For each={props.variants}>{(v) => <option value={v.kind}>{v.kind}</option>}</For>
        </select>
      </label>

      <div class="strategy-variant-fields">
        <For each={current().fields}>
          {(field) => (
            <Show
              when={
                field.type === "enum" && (field as Extract<AxisFieldSchema, { type: "enum" }>).bare
              }
              fallback={
                <Show
                  when={field.type === "enum"}
                  fallback={
                    <ScalarField
                      field={field as Extract<AxisFieldSchema, { type: "float" | "int" | "bool" }>}
                      value={props.value[field.name]}
                      onChange={(v) => setField(field.name, v)}
                    />
                  }
                >
                  <VariantPicker
                    label={field.name}
                    variants={(field as Extract<AxisFieldSchema, { type: "enum" }>).variants}
                    schema={props.schema}
                    value={
                      (props.value[field.name] as VariantValue | undefined) ??
                      (buildDefaultFieldValue(field, props.schema) as VariantValue)
                    }
                    onChange={(v) => setField(field.name, v)}
                  />
                </Show>
              }
            >
              <BareEnumField
                field={field as Extract<AxisFieldSchema, { type: "enum" }>}
                value={props.value[field.name]}
                onChange={(v) => setField(field.name, v)}
              />
            </Show>
          )}
        </For>
      </div>

      <Show when={current().wraps}>
        {(wraps) => (
          <VariantPicker
            label="wraps"
            variants={props.schema[wraps()].variants}
            schema={props.schema}
            value={
              (props.value.inner as VariantValue | undefined) ??
              buildDefaultVariantValue(props.schema[wraps()].variants[0]!, props.schema)
            }
            onChange={(v) => setField("inner", v)}
          />
        )}
      </Show>
    </div>
  );
};

/** A `bare` enum field (`DecisiveMoveMode` -- see `AxisFieldSchema`'s doc
 * comment): a plain `<select>` whose value *is* the chosen variant's `kind`
 * string, not a `{kind, ...fields}` object -- there are no fields to carry,
 * every `bare` variant's `fields` array is empty by construction. */
const BareEnumField: Component<{
  field: Extract<AxisFieldSchema, { type: "enum" }>;
  value: unknown;
  onChange: (value: unknown) => void;
}> = (props) => {
  return (
    <label>
      {props.field.name}
      <select
        value={(props.value as string | undefined) ?? props.field.default}
        onChange={(e) => props.onChange(e.currentTarget.value)}
      >
        <For each={props.field.variants}>{(v) => <option value={v.kind}>{v.kind}</option>}</For>
      </select>
    </label>
  );
};

const NumberField: Component<{
  field: Extract<AxisFieldSchema, { type: "float" | "int" }>;
  value: unknown;
  onChange: (value: unknown) => void;
}> = (props) => {
  return (
    <label>
      {props.field.name}
      <input
        type="number"
        min={props.field.bounds[0]}
        max={props.field.bounds[1]}
        step={props.field.type === "int" ? 1 : "any"}
        value={(props.value as number | undefined) ?? props.field.default}
        onInput={(e) => {
          const n =
            props.field.type === "int"
              ? parseInt(e.currentTarget.value)
              : parseFloat(e.currentTarget.value);
          props.onChange(Number.isFinite(n) ? n : props.field.default);
        }}
      />
    </label>
  );
};

const BoolField: Component<{
  field: Extract<AxisFieldSchema, { type: "bool" }>;
  value: unknown;
  onChange: (value: unknown) => void;
}> = (props) => {
  return (
    <label class="strategy-checkbox-field">
      <input
        type="checkbox"
        checked={(props.value as boolean | undefined) ?? props.field.default}
        onChange={(e) => props.onChange(e.currentTarget.checked)}
      />
      {props.field.name}
    </label>
  );
};

const ScalarField: Component<{
  field: Extract<AxisFieldSchema, { type: "float" | "int" | "bool" }>;
  value: unknown;
  onChange: (value: unknown) => void;
}> = (props) => {
  return (
    <Show
      when={
        props.field.type === "bool"
          ? (props.field as Extract<AxisFieldSchema, { type: "bool" }>)
          : undefined
      }
      fallback={
        <NumberField
          field={props.field as Extract<AxisFieldSchema, { type: "float" | "int" }>}
          value={props.value}
          onChange={props.onChange}
        />
      }
    >
      {(field) => <BoolField field={field()} value={props.value} onChange={props.onChange} />}
    </Show>
  );
};

/** One `TunerParameter` rendered by type, generically -- covers every axis variant
 * and every conditioned field (`q_init`, `mcgs`, `rave_ucb`, `c`, ...) with
 * one control per wire type rather than per field name, since `TunerInfo`
 * carries no more structure than name/type/bounds-or-choices/default. */
const TunerFieldControl: Component<{
  param: TunerParameter;
  value: unknown;
  onChange: (value: unknown) => void;
}> = (props) => {
  return (
    <Show
      when={props.param.type === "categorical" ? props.param : undefined}
      fallback={
        <Show
          when={props.param.type === "bool" ? props.param : undefined}
          fallback={
            <label>
              {props.param.name}
              <input
                type="number"
                min={(props.param as Extract<TunerParameter, { type: "int" | "float" }>).bounds[0]}
                max={(props.param as Extract<TunerParameter, { type: "int" | "float" }>).bounds[1]}
                step={props.param.type === "int" ? 1 : "any"}
                value={
                  (props.value as number | undefined) ??
                  (props.param as Extract<TunerParameter, { type: "int" | "float" }>).default
                }
                onInput={(e) => {
                  const n =
                    props.param.type === "int"
                      ? parseInt(e.currentTarget.value)
                      : parseFloat(e.currentTarget.value);
                  props.onChange(
                    Number.isFinite(n)
                      ? n
                      : (props.param as Extract<TunerParameter, { type: "int" | "float" }>).default,
                  );
                }}
              />
            </label>
          }
        >
          {(param) => (
            <label class="strategy-checkbox-field">
              <input
                type="checkbox"
                checked={(props.value as boolean | undefined) ?? param().default}
                onChange={(e) => props.onChange(e.currentTarget.checked)}
              />
              {param().name}
            </label>
          )}
        </Show>
      }
    >
      {(param) => (
        <label>
          {param().name}
          <select
            value={(props.value as string | undefined) ?? param().default}
            onChange={(e) => props.onChange(e.currentTarget.value)}
          >
            <For each={param().choices}>{(choice) => <option value={choice}>{choice}</option>}</For>
          </select>
        </label>
      )}
    </Show>
  );
};

/** By-algorithm mode's own fields: the `algorithm` picker (always first,
 * always the root categorical) plus every other currently-active parameter --
 * the four MCTS axis categoricals (`select`/`simulate`/`backprop`/
 * `final_action`) and their per-variant knobs -- in `info.parameters`
 * declaration order. Picking an `algorithm` fully replaces `values`
 * (discarding every other field, same "changing the kind rebuilds the value"
 * precedent `VariantPicker` sets for axis variants); editing any other field
 * merges the edit into `values` and fills in defaults for anything it newly
 * reveals. */
const AlgorithmFields: Component<{
  info: TunerInfo;
  values: Record<string, unknown>;
  onChange: (values: Record<string, unknown>) => void;
}> = (props) => {
  const algorithmParam = createMemo(
    () => props.info.parameters.find((p) => p.name === "algorithm")!,
  );
  const active = createMemo(() =>
    activeNames(props.info.parameters, props.info.conditions, props.values),
  );

  function setField(name: string, value: unknown) {
    props.onChange(
      withDefaultsFilled(props.info.parameters, props.info.conditions, {
        ...props.values,
        [name]: value,
      }),
    );
  }

  return (
    <div class="strategy-algorithm-fields">
      <TunerFieldControl
        param={algorithmParam()}
        value={props.values.algorithm}
        onChange={(value) =>
          props.onChange(
            withDefaultsFilled(props.info.parameters, props.info.conditions, { algorithm: value }),
          )
        }
      />
      <For
        each={props.info.parameters.filter((p) => p.name !== "algorithm" && active().has(p.name))}
      >
        {(param) => (
          <TunerFieldControl
            param={param}
            value={props.values[param.name]}
            onChange={(value) => setField(param.name, value)}
          />
        )}
      </For>
    </div>
  );
};

/** Which budget field(s) are enabled -- not itself part of `CustomStrategySpec`,
 * since "both boxes unchecked" is a transient invalid UI state this editor
 * must be able to represent (to show the validation error) without ever
 * committing it through `onChange`. */
interface BudgetEnabled {
  time: boolean;
  iterations: boolean;
}

export const StrategyConfigEditor: Component<{
  schema: AxisSchema;
  config: CustomStrategySpec;
  /** The game's tuner algorithm/axis catalog (`GET /api/games/{kind}/
   * strategy-algorithms`), `null` for a game with no tuning support. Drives
   * the mode toggle below `null` hides it entirely, leaving free composition
   * as the only mode, exactly today's behavior. */
  tunerInfo: TunerInfo | null;
  onChange: (config: CustomStrategySpec) => void;
}> = (props) => {
  const [budgetEnabled, setBudgetEnabled] = createSignal<BudgetEnabled>({
    time: props.config.max_time_ms !== undefined,
    iterations: props.config.max_iterations !== undefined,
  });

  const budgetError = createMemo(() => {
    const e = budgetEnabled();
    return e.time || e.iterations
      ? null
      : "At least one of time limit or iteration limit must be set.";
  });

  // Mirrors `build_custom`'s own mutual-exclusivity contract: `search` and
  // `params` are never both set, so which one is present *is* the mode --
  // there is no separate signal to fall out of sync with `config`.
  const mode = createMemo<"free" | "params">(() =>
    props.config.search !== undefined ? "free" : "params",
  );

  function switchToFree() {
    props.onChange({
      ...props.config,
      search: buildDefaultSearchSpec(props.schema),
      params: undefined,
    });
  }

  function switchToNamed(info: TunerInfo) {
    const algorithm = info.parameters.find((p) => p.name === "algorithm")!;
    props.onChange({
      ...props.config,
      search: undefined,
      params: withDefaultsFilled(info.parameters, info.conditions, {
        algorithm: algorithm.default,
      }),
    });
  }

  function setAxis<K extends keyof SearchSpec>(axis: K, value: VariantValue) {
    props.onChange({
      ...props.config,
      search: {
        ...props.config.search!,
        [axis]: value as unknown as SearchSpec[K],
      },
    });
  }

  function commitBudget(enabled: BudgetEnabled, timeMs: number, iterations: number) {
    setBudgetEnabled(enabled);
    if (!enabled.time && !enabled.iterations) return;
    props.onChange({
      ...props.config,
      max_time_ms: enabled.time ? timeMs : undefined,
      max_iterations: enabled.iterations ? iterations : undefined,
    });
  }

  const timeMs = () => props.config.max_time_ms ?? 1000;
  const iterations = () => props.config.max_iterations ?? 10_000;

  return (
    <div class="strategy-config-editor">
      <Show when={props.tunerInfo}>
        {(info) => (
          <div class="strategy-mode-toggle" role="tablist">
            <button
              type="button"
              role="tab"
              aria-selected={mode() === "free"}
              classList={{ active: mode() === "free" }}
              onClick={switchToFree}
            >
              Free composition
            </button>
            <button
              type="button"
              role="tab"
              aria-selected={mode() === "params"}
              classList={{ active: mode() === "params" }}
              onClick={() => switchToNamed(info())}
            >
              By algorithm
            </button>
          </div>
        )}
      </Show>

      <Show when={mode() === "free"}>
        <VariantPicker
          label="Select"
          variants={props.schema.select.variants}
          schema={props.schema}
          value={props.config.search!.select as VariantValue}
          onChange={(v) => setAxis("select", v)}
        />
        <VariantPicker
          label="Simulate"
          variants={props.schema.simulate.variants}
          schema={props.schema}
          value={props.config.search!.simulate as VariantValue}
          onChange={(v) => setAxis("simulate", v)}
        />
        <VariantPicker
          label="Backprop"
          variants={props.schema.backprop.variants}
          schema={props.schema}
          value={props.config.search!.backprop as VariantValue}
          onChange={(v) => setAxis("backprop", v)}
        />
        <VariantPicker
          label="Final action"
          variants={props.schema.final_action.variants}
          schema={props.schema}
          value={props.config.search!.final_action as VariantValue}
          onChange={(v) => setAxis("final_action", v)}
        />
      </Show>

      <Show when={mode() === "params" && props.tunerInfo}>
        {(info) => (
          <AlgorithmFields
            info={info()}
            values={props.config.params ?? {}}
            onChange={(params) => props.onChange({ ...props.config, params })}
          />
        )}
      </Show>

      <div class="strategy-budget-fields">
        <label class="strategy-checkbox-field">
          <input
            type="checkbox"
            checked={budgetEnabled().time}
            onChange={(e) =>
              commitBudget(
                { ...budgetEnabled(), time: e.currentTarget.checked },
                timeMs(),
                iterations(),
              )
            }
          />
          Time limit (ms)
        </label>
        <Show when={budgetEnabled().time}>
          <input
            type="number"
            min={1}
            value={timeMs()}
            onInput={(e) => {
              const n = parseInt(e.currentTarget.value);
              if (Number.isFinite(n)) commitBudget(budgetEnabled(), n, iterations());
            }}
          />
        </Show>

        <label class="strategy-checkbox-field">
          <input
            type="checkbox"
            checked={budgetEnabled().iterations}
            onChange={(e) =>
              commitBudget(
                { ...budgetEnabled(), iterations: e.currentTarget.checked },
                timeMs(),
                iterations(),
              )
            }
          />
          Iteration limit
        </label>
        <Show when={budgetEnabled().iterations}>
          <input
            type="number"
            min={1}
            value={iterations()}
            onInput={(e) => {
              const n = parseInt(e.currentTarget.value);
              if (Number.isFinite(n)) commitBudget(budgetEnabled(), timeMs(), n);
            }}
          />
        </Show>

        <Show when={budgetError()}>
          <div class="strategy-budget-error">{budgetError()}</div>
        </Show>

        <label>
          Threads (0 = auto)
          <input
            type="number"
            min={0}
            value={props.config.threads ?? 0}
            onInput={(e) => {
              const n = parseInt(e.currentTarget.value);
              props.onChange({ ...props.config, threads: Number.isFinite(n) ? n : 0 });
            }}
          />
        </label>

        <label class="strategy-checkbox-field">
          <input
            type="checkbox"
            checked={props.config.use_transpositions ?? false}
            onChange={(e) => {
              const use_transpositions = e.currentTarget.checked;
              // `mcgs` requires `use_transpositions` (see the field below
              // and `CustomStrategySpec::mcgs`'s doc comment) -- clearing
              // transpositions clears it too, so this editor can never
              // produce the rejected `mcgs && !use_transpositions`
              // combination in the first place.
              props.onChange({
                ...props.config,
                use_transpositions,
                mcgs: use_transpositions ? props.config.mcgs : false,
              });
            }}
          />
          Use transpositions
        </label>

        {/* `q_init`/`mcgs`/`state_only_keying` are ordinary conditioned
            `TunerParameter`s in by-algorithm mode, rendered generically by
            `AlgorithmFields` above -- these hand-written rows write to
            `CustomStrategySpec`'s top-level fields, which the `params`
            branch of `build_custom` never reads for these three names, so
            showing both would be redundant and silently ignored. */}
        <Show when={mode() === "free"}>
          <label class="strategy-checkbox-field">
            <input
              type="checkbox"
              disabled={!(props.config.use_transpositions ?? false)}
              checked={props.config.mcgs ?? false}
              onChange={(e) => props.onChange({ ...props.config, mcgs: e.currentTarget.checked })}
            />
            Graph search (MCGS)
          </label>

          <label>
            Q init
            <select
              value={props.config.q_init ?? "Parent"}
              onChange={(e) => props.onChange({ ...props.config, q_init: e.currentTarget.value })}
            >
              {/* `QInit`'s wire form -- see `mcts-tune/src/presets.rs`'s
                  `CustomStrategySpec::q_init` doc comment. */}
              <option value="Parent">Parent</option>
              <option value="Win">Win</option>
              <option value="Loss">Loss</option>
              <option value="Draw">Draw</option>
              <option value="Infinity">Infinity</option>
            </select>
          </label>
        </Show>
      </div>
    </div>
  );
};
