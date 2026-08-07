// core/effect.ts — Effect class for async side effects.
// Supports .map(), .catch(), .merge() for composition.

type Runner<A> = (send: (a: A) => void) => Promise<void>;

export class Effect<A> {
  private constructor(private readonly runner: Runner<A>) {}

  /** An effect that does nothing. */
  static none<A>(): Effect<A> {
    return new Effect(() => Promise.resolve());
  }

  /** An effect that immediately sends a single value. Useful in tests. */
  static send<A>(a: A): Effect<A> {
    return new Effect(send => { send(a); return Promise.resolve(); });
  }

  /** Lift a promise thunk; errors propagate (store catches unhandled rejections). */
  static fromPromise<A>(thunk: () => Promise<A>): Effect<A> {
    return new Effect(send => thunk().then(a => { send(a); }));
  }

  /** An effect that sends a single value after `ms` milliseconds. For poll-loop backoff. */
  static delay<A>(ms: number, a: A): Effect<A> {
    return new Effect(send => new Promise(resolve => {
      setTimeout(() => { send(a); resolve(); }, ms);
    }));
  }

  /** Run all effects concurrently; each sends into the same channel. */
  static merge<A>(...effects: Effect<A>[]): Effect<A> {
    return new Effect(send =>
      Promise.all(effects.map(e => e.runner(send))).then(() => {})
    );
  }

  /** Transform the output value of this effect. */
  map<B>(f: (a: A) => B): Effect<B> {
    return new Effect(send => this.runner(a => send(f(a))));
  }

  /** Convert a rejected promise into a sent value rather than a thrown error. */
  catch(onReject: (e: unknown) => A): Effect<A> {
    return new Effect(send =>
      this.runner(send).catch(e => { send(onReject(e)); })
    );
  }

  /** @internal — called by the store. */
  execute(send: (a: A) => void): Promise<void> {
    return this.runner(send);
  }
}
