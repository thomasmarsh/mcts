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
//
// Virtual time: every effect runs against the store's `TestScheduler`, a
// manual `Scheduler` that `Effect.delay` sleeps on instead of a real timer.
// A delayed action only becomes receivable when the test advances virtual
// time past its due time:
//   ts.advance(1000);  // fires any Effect.delay(<=1000ms, ...) sends
// Delivery is synchronous — advance() returns with the action already in the
// pending queue, so no awaiting or timer mocking is ever involved. Tests
// must end with no sleep still pending (drive the poll loop to a terminal
// state, or advance past the last delay and consume what fires); the
// afterEach hook fails the test otherwise rather than hanging on a sleep
// that can never resolve.

import { afterEach, expect } from "vitest";
import type { Reducer, Scheduler } from "@mcts/core";

/** A manual `Scheduler`: `schedule`d callbacks queue up and only fire when
 * `advance` moves virtual time past their due time, in (due-time,
 * insertion) order. Firing is synchronous — a callback that sends an action
 * delivers it before `advance` returns. */
export class TestScheduler implements Scheduler {
  private currentMs = 0;
  private seq = 0;
  private sleeps: { at: number; order: number; cb: () => void }[] = [];

  schedule(ms: number, cb: () => void): void {
    this.sleeps.push({ at: this.currentMs + Math.max(0, ms), order: this.seq++, cb });
  }

  /** Callbacks scheduled but not yet fired. */
  get pendingCount(): number {
    return this.sleeps.length;
  }

  /** Advance virtual time by `ms`, firing every callback due within that
   * window (including anything a fired callback itself schedules deeper
   * into the same window). */
  advance(ms: number): void {
    const target = this.currentMs + ms;
    for (;;) {
      let next = -1;
      for (let i = 0; i < this.sleeps.length; i++) {
        const s = this.sleeps[i]!;
        if (s.at > target) continue;
        const best = this.sleeps[next];
        if (next === -1 || s.at < best!.at || (s.at === best!.at && s.order < best!.order))
          next = i;
      }
      if (next === -1) break;
      const [s] = this.sleeps.splice(next, 1);
      this.currentMs = Math.max(this.currentMs, s!.at);
      s!.cb();
    }
    this.currentMs = target;
  }
}

export class TestStore<S, A, Env> {
  private state: S;
  private reducer: Reducer<S, A, Env>;
  private environment: Env;
  private pending: A[] = [];
  private effectPromises: Promise<void>[] = [];
  readonly scheduler = new TestScheduler();

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
      const p = effect
        .execute((a) => this.pending.push(a), { scheduler: this.scheduler })
        .catch(() => {});
      this.effectPromises.push(p);
    }

    return this;
  }

  /** Settle all pending async effects (Promise-based). Call before receive()
   *  when the effect under test is async (e.g. a rejection path). Not needed
   *  for actions delivered by advance(), which land synchronously. */
  async drain(): Promise<void> {
    while (this.effectPromises.length > 0) {
      const batch = this.effectPromises.splice(0);
      await Promise.all(batch);
    }
  }

  /** Advance the test scheduler by `ms`, synchronously delivering any
   *  `Effect.delay` actions that come due. */
  advance(ms: number): this {
    this.scheduler.advance(ms);
    return this;
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
    const stranded = store.scheduler.pendingCount;
    if (stranded > 0) {
      throw new Error(
        `TestStore finished with ${stranded} delayed effect(s) still sleeping on the test scheduler; ` +
          `advance() past them or drive the poll loop to completion`,
      );
    }
    await store.drain();
    store.assertDrained();
  });
  return store;
}
