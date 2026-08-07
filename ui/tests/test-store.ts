// tests/test-store.ts — TCA-style exhaustive TestStore for feature reducers.
// Ported from pb/ui/tests/test-store.ts -- mirror, don't redesign.
//
// Pattern:
//   ts.send(action, state => { state.foo = "bar"; })
//     — dispatch action; mutate the state clone to express the full expected
//       state after the action; TestStore internally validates actual == expected.
//   ts.receive(expectedAction, state => { state.foo = "bar"; })
//     — assert the next pending effect-dispatched action matches expectedAction,
//       then dispatch it with the same state-mutation validation.
//   ts.assertDrained()
//     — fail if any effect-dispatched actions were not consumed via .receive().

import { afterEach, expect } from "vitest";
import type { Reducer } from "@mcts/core";

export class TestStore<S, A, Env> {
  private state: S;
  private reducer: Reducer<S, A, Env>;
  private environment: Env;
  private pending: A[] = [];
  private effectPromises: Promise<void>[] = [];

  constructor(reducer: Reducer<S, A, Env>, environment: Env, initialState: S) {
    this.state = structuredClone(initialState);
    this.reducer = reducer;
    this.environment = environment;
  }

  /** Dispatch action. If assert is provided, mutate the expected-state clone to
   *  describe all state changes; TestStore compares actual vs expected. */
  send(action: A, assert?: (state: S) => void): this {
    const expected = structuredClone(this.state);
    if (assert) assert(expected);

    const draft = structuredClone(this.state) as S;
    const effect = this.reducer(draft, action, this.environment);

    if (assert) expect(draft).toEqual(expected);

    this.state = draft;
    if (effect) {
      const p = effect.execute((a) => this.pending.push(a)).catch(() => {});
      this.effectPromises.push(p);
    }

    return this;
  }

  /** Settle all pending async effects (Promise-based). Call before receive()
   *  when the effect under test is async (e.g. a rejection path). */
  async drain(): Promise<void> {
    while (this.effectPromises.length > 0) {
      const batch = this.effectPromises.splice(0);
      await Promise.all(batch);
    }
  }

  /** Assert the next pending action equals expected, then dispatch it. */
  receive(expected: A, assert?: (state: S) => void): this {
    if (this.pending.length === 0) {
      throw new Error(
        `Expected to receive ${JSON.stringify(expected)}, but no effects have been dispatched`,
      );
    }
    const actual = this.pending.shift() as A;
    expect(actual).toEqual(expected);
    return this.send(actual, assert);
  }

  assertDrained(): void {
    if (this.pending.length > 0) {
      throw new Error(
        `TestStore has ${this.pending.length} unhandled action(s): ${JSON.stringify(this.pending)}`,
      );
    }
  }

  /** Read current state after send() without the assert callback. */
  getState(): S {
    return this.state;
  }
}

export function createTestStore<S, A, Env>(
  reducer: Reducer<S, A, Env>,
  environment: Env,
  initialState: S,
): TestStore<S, A, Env> {
  const store = new TestStore(reducer, environment, initialState);
  // async afterEach: settle any stray async effects before asserting empty queue
  afterEach(async () => {
    await store.drain();
    store.assertDrained();
  });
  return store;
}
