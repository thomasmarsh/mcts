// core/effect.ts — Effect class for async side effects.
// Supports .map(), .catch(), .merge() for composition.

/** What `Effect.delay` sleeps on. The real store uses `realScheduler`
 * (setTimeout); tests substitute a manual scheduler (see
 * ui/tests/test-store.ts's `TestScheduler`) so delayed effects fire on
 * virtual time the test advances explicitly — no real waiting, no timer
 * mocking. */
export interface Scheduler {
  /** Invoke `cb` after `ms` milliseconds (real or virtual). */
  schedule(ms: number, cb: () => void): void;
}

export const realScheduler: Scheduler = {
  schedule: (ms, cb) => {
    setTimeout(cb, ms);
  },
};

/** Everything an effect needs from whoever executes it. Threaded through
 * the runner/combinators rather than read from a module global so the
 * choice is made once, at the store boundary. */
export interface EffectContext {
  scheduler: Scheduler;
}

const defaultContext: EffectContext = { scheduler: realScheduler };

type Runner<A> = (send: (a: A) => void, ctx: EffectContext) => Promise<void>;

export class Effect<A> {
  private constructor(private readonly runner: Runner<A>) {}

  /** An effect that does nothing. */
  static none<A>(): Effect<A> {
    return new Effect(() => Promise.resolve());
  }

  /** An effect that immediately sends a single value. Useful in tests. */
  static send<A>(a: A): Effect<A> {
    return new Effect((send) => {
      send(a);
      return Promise.resolve();
    });
  }

  /** Lift a promise thunk; errors propagate (store catches unhandled rejections). */
  static fromPromise<A>(thunk: () => Promise<A>): Effect<A> {
    return new Effect((send) =>
      thunk().then((a) => {
        send(a);
      }),
    );
  }

  /** Lift a push subscription that emits many values then finishes -- e.g. an
   * SSE stream. `start` is handed a `send` and a `done`; call `send` for each
   * value and `done` once when the source closes (the effect's promise
   * resolves then). Values sent after `done` are dropped. There is no
   * cancellation channel: the source itself must terminate (server closes the
   * stream, or it errors), so use this only for sources that do. */
  static stream<A>(start: (send: (a: A) => void, done: () => void) => void): Effect<A> {
    return new Effect(
      (send) =>
        new Promise((resolve) => {
          let finished = false;
          start(
            (a) => {
              if (!finished) send(a);
            },
            () => {
              if (!finished) {
                finished = true;
                resolve();
              }
            },
          );
        }),
    );
  }

  /** An effect that sends a single value after `ms` milliseconds. For poll-loop backoff. */
  static delay<A>(ms: number, a: A): Effect<A> {
    return new Effect(
      (send, ctx) =>
        new Promise((resolve) => {
          ctx.scheduler.schedule(ms, () => {
            send(a);
            resolve();
          });
        }),
    );
  }

  /** Run all effects concurrently; each sends into the same channel. */
  static merge<A>(...effects: Effect<A>[]): Effect<A> {
    return new Effect((send, ctx) =>
      Promise.all(effects.map((e) => e.runner(send, ctx))).then(() => {}),
    );
  }

  /** Transform the output value of this effect. */
  map<B>(f: (a: A) => B): Effect<B> {
    return new Effect((send, ctx) => this.runner((a) => send(f(a)), ctx));
  }

  /** Convert a rejected promise into a sent value rather than a thrown error. */
  catch(onReject: (e: unknown) => A): Effect<A> {
    return new Effect((send, ctx) =>
      this.runner(send, ctx).catch((e) => {
        send(onReject(e));
      }),
    );
  }

  /** @internal — called by the store (or a test harness, with its own
   * scheduler in `ctx`). */
  execute(send: (a: A) => void, ctx: EffectContext = defaultContext): Promise<void> {
    return this.runner(send, ctx);
  }
}
