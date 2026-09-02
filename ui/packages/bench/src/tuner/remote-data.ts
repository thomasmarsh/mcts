// remote-data.ts — a four-state wrapper for one asynchronously loaded
// resource, used throughout `tuner-reducer.ts`. The reducer holds one
// `RemoteData<T>` per endpoint; components read `.status` and dispatch, they
// never fetch. Mirrors the `idle | loading | ok | err` shape the bench
// reducer's job-poll sub-reducers already use, extracted here so the tuner
// views can share it without pulling in the round-robin reducer.

export type RemoteData<T> =
  | { status: "idle" }
  | { status: "loading"; previous?: T }
  | { status: "ok"; value: T; fetchedAt: number }
  | { status: "err"; message: string; previous?: T };

export const idle = <T>(): RemoteData<T> => ({ status: "idle" });

/** Move to `loading`, preserving any previously loaded value so a view can
 * keep showing stale data (dimmed) while a refresh is in flight. */
export function toLoading<T>(current: RemoteData<T>): RemoteData<T> {
  return { status: "loading", previous: peek(current) };
}

export function toOk<T>(value: T, fetchedAt: number): RemoteData<T> {
  return { status: "ok", value, fetchedAt };
}

export function toErr<T>(message: string, current: RemoteData<T>): RemoteData<T> {
  return { status: "err", message, previous: peek(current) };
}

/** The last successfully loaded value, if any — from an `ok` state directly,
 * or carried on a subsequent `loading` / `err`. */
export function peek<T>(data: RemoteData<T>): T | undefined {
  switch (data.status) {
    case "ok":
      return data.value;
    case "loading":
    case "err":
      return data.previous;
    case "idle":
      return undefined;
  }
}

export const isLoading = (data: RemoteData<unknown>): boolean => data.status === "loading";
