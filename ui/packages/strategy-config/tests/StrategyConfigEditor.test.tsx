// StrategyConfigEditor.test.tsx — component tests against the hand-built
// `fixtureSchema` (see schema-fixture.ts), per AGENTS.md's UI-testing rule:
// a real `@solidjs/testing-library` render, no live server, no mocked `Env`
// needed since this component takes its schema/config as plain props.

import { afterEach, describe, expect, it } from "vitest";
import { createSignal } from "solid-js";
import { cleanup, fireEvent, render, screen } from "@solidjs/testing-library";
import type { CustomStrategySpec } from "@mcts/game";
import { defaultCustomStrategySpec, StrategyConfigEditor } from "../src/index.js";
import { fixtureDefaultConfig, fixtureSchema } from "./schema-fixture.js";

afterEach(() => cleanup());

/** A minimal controlled harness: owns the signal, re-renders on `onChange`,
 * same convention a real caller (e.g. GameShell's New Game dialog) would
 * use. Also exposes the latest committed config for assertions. */
function renderEditor(initial: CustomStrategySpec) {
  const [config, setConfig] = createSignal<CustomStrategySpec>(initial);
  let latest = initial;
  const onChange = (next: CustomStrategySpec) => {
    latest = next;
    setConfig(next);
  };
  render(() => <StrategyConfigEditor schema={fixtureSchema} config={config()} onChange={onChange} />);
  return { getLatest: () => latest };
}

describe("defaultCustomStrategySpec", () => {
  it("seeds a time budget (never an unbounded iteration count) and all-core threads", () => {
    // Regression test: a user-composed axis combination's per-iteration
    // cost is unknowable ahead of time (e.g. `decisive_move_nst` is ~15x
    // a plain `uniform` simulate's cost per iteration, measured against
    // Druid) -- seeding an iteration cap here (the previous default,
    // 10,000 iterations, single-threaded) turned into an effectively
    // unbounded wall-clock wait for an expensive combination, which read
    // as "the custom game doesn't run." Every real preset in
    // `games/*/presets.json` budgets by `max_time_ms`, never
    // `max_iterations`, for exactly this reason -- this default now
    // matches that.
    const spec = defaultCustomStrategySpec(fixtureSchema);
    expect(spec.max_time_ms).toBeTypeOf("number");
    expect(spec.max_iterations).toBeUndefined();
    expect(spec.threads).toBe(0);
  });
});

describe("StrategyConfigEditor", () => {
  it("selecting epsilon_greedy reveals the inner select_base picker", async () => {
    renderEditor(fixtureDefaultConfig() as unknown as CustomStrategySpec);

    expect(screen.queryByText("wraps")).toBeNull();

    const selectSelect = screen.getAllByRole("combobox")[0]!;
    fireEvent.change(selectSelect, { target: { value: "epsilon_greedy" } });

    await screen.findByText("wraps");
  });

  it("deselecting a wrapper discards its now-orphaned inner config", async () => {
    const { getLatest } = renderEditor(fixtureDefaultConfig() as unknown as CustomStrategySpec);

    const selectSelect = screen.getAllByRole("combobox")[0]!;
    fireEvent.change(selectSelect, { target: { value: "epsilon_greedy" } });
    await screen.findByText("wraps");
    expect(getLatest().search.select).toMatchObject({ kind: "epsilon_greedy", inner: { kind: "ucb1" } });

    fireEvent.change(selectSelect, { target: { value: "ucb1" } });
    expect(getLatest().search.select).toEqual({ kind: "ucb1", c: 1.4142135623730951 });
    expect((getLatest().search.select as { inner?: unknown }).inner).toBeUndefined();
  });

  // Regression test for a live bug: a `bare` field (`DecisiveMoveMode`) used
  // to be rendered the same way as a real tagged union (`RaveSchedule`/
  // `RaveUcb`), emitting `{kind, ...fields}` -- which the server rejected
  // ("unknown variant `kind`, expected one of `win`, `win_loss`,
  // `win_loss_draw`", since `DecisiveMoveMode` has no serde tag). This
  // exercises the actual editor path a user hit: pick `decisive_move_nst`,
  // then check what `mode` actually serializes as.
  it("a bare enum field (DecisiveMoveMode) renders as a plain <select> and emits a plain string, not an object", async () => {
    const { getLatest } = renderEditor(fixtureDefaultConfig() as unknown as CustomStrategySpec);

    const simulateSelect = screen.getByLabelText("Simulate") as HTMLSelectElement;
    fireEvent.change(simulateSelect, { target: { value: "decisive_move_nst" } });

    const modeSelect = (await screen.findByLabelText("mode")) as HTMLSelectElement;
    expect(modeSelect.value).toBe("win");
    expect(getLatest().search.simulate).toMatchObject({ kind: "decisive_move_nst", mode: "win" });

    fireEvent.change(modeSelect, { target: { value: "win_loss" } });
    const mode = (getLatest().search.simulate as { mode: unknown }).mode;
    expect(mode).toBe("win_loss");
    expect(typeof mode).toBe("string");
  });

  it("unchecking both budget boxes shows a validation error and withholds a valid onChange", () => {
    const { getLatest } = renderEditor(fixtureDefaultConfig() as unknown as CustomStrategySpec);

    const iterationsCheckbox = screen.getByLabelText("Iteration limit") as HTMLInputElement;
    expect(iterationsCheckbox.checked).toBe(true);

    const before = getLatest();
    fireEvent.click(iterationsCheckbox);

    screen.getByText(/at least one/i);
    // No valid config was committed -- onChange was withheld, not fired with
    // both fields unset.
    expect(getLatest()).toBe(before);
  });

  it("MCGS checkbox is disabled until transpositions are enabled, and unchecking transpositions clears mcgs", () => {
    const { getLatest } = renderEditor(fixtureDefaultConfig() as unknown as CustomStrategySpec);

    const transpositionsCheckbox = screen.getByLabelText("Use transpositions") as HTMLInputElement;
    const mcgsCheckbox = screen.getByLabelText("Graph search (MCGS)") as HTMLInputElement;
    expect(transpositionsCheckbox.checked).toBe(false);
    expect(mcgsCheckbox.disabled).toBe(true);

    fireEvent.click(transpositionsCheckbox);
    expect(mcgsCheckbox.disabled).toBe(false);

    fireEvent.click(mcgsCheckbox);
    expect(getLatest().mcgs).toBe(true);

    // Turning transpositions back off must clear `mcgs` too -- the server
    // (`mcts_tune::presets::build_custom`) rejects `mcgs && !use_transpositions`,
    // so this editor must never be able to produce that combination.
    fireEvent.click(transpositionsCheckbox);
    expect(getLatest().mcgs).toBe(false);
    expect(getLatest().use_transpositions).toBe(false);
  });

  it("checking both budget boxes simultaneously is accepted", () => {
    const { getLatest } = renderEditor(fixtureDefaultConfig() as unknown as CustomStrategySpec);

    const timeCheckbox = screen.getByLabelText("Time limit (ms)") as HTMLInputElement;
    fireEvent.click(timeCheckbox);

    expect(screen.queryByText(/at least one/i)).toBeNull();
    const latest = getLatest();
    expect(latest.max_time_ms).toBeTypeOf("number");
    expect(latest.max_iterations).toBeTypeOf("number");
  });
});
