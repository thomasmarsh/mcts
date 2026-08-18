// StrategyConfigEditor.test.tsx — component tests against the hand-built
// `fixtureSchema` (see schema-fixture.ts), per AGENTS.md's UI-testing rule:
// a real `@solidjs/testing-library` render, no live server, no mocked `Env`
// needed since this component takes its schema/config as plain props.

import { afterEach, describe, expect, it } from "vitest";
import { createSignal } from "solid-js";
import { cleanup, fireEvent, render, screen } from "@solidjs/testing-library";
import type { CustomStrategySpec } from "@mcts/game";
import { StrategyConfigEditor } from "../src/index.js";
import { fixtureDefaultConfig, fixtureSchema } from "./schema-fixture.js";

afterEach(() => cleanup());

/** A minimal controlled harness: owns the signal, re-renders on `onChange`,
 * same convention a real caller (GameShell's New Game dialog, Phase 5) would
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
