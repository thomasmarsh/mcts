// core/job-poll.ts — Generic background-job submit/poll state machine.
//
// Domain-agnostic: no game knowledge. Any consumer with a job that a server
// executes asynchronously and reports back via a job id embeds this reducer
// via its own keyed slice of app state (e.g. `aiMove`/`analysis` in the
// game UI's `AppState`).

import { Effect } from "./effect.js";

export interface JobPollState<T> {
  status: "idle" | "pending" | "done" | "error";
  jobId: string | null;
  result: T | null;
  error: string | null;
  attempt: number;
}

export function initialJobPollState<T>(): JobPollState<T> {
  return { status: "idle", jobId: null, result: null, error: null, attempt: 0 };
}

export type JobSubmitResult<T> =
  { status: "done"; result: T } | { status: "pending"; jobId: string };

export type JobPollResult<T> =
  { status: "pending" } | { status: "done"; result: T } | { status: "error"; error: string };

export type JobPollAction<T> =
  | { tag: "start" }
  | { tag: "submitted"; result: JobSubmitResult<T> }
  | { tag: "tick"; jobId: string }
  | { tag: "polled"; jobId: string; result: JobPollResult<T> }
  | { tag: "failed"; error: string }; // submitJob/pollJob itself rejected (network/transport failure)

export interface JobPollEnv<T> {
  submitJob(): Effect<JobSubmitResult<T>>;
  pollJob(jobId: string): Effect<JobPollResult<T>>;
}

export const JOB_POLL_BACKOFF_START_MS = 300;
export const JOB_POLL_BACKOFF_MAX_MS = 2000;
export const JOB_POLL_MAX_ATTEMPTS = 30;

function nextDelayMs(attempt: number): number {
  return Math.min(JOB_POLL_BACKOFF_START_MS * 2 ** (attempt - 1), JOB_POLL_BACKOFF_MAX_MS);
}

export function jobPollReduce<T>(
  draft: JobPollState<T>,
  action: JobPollAction<T>,
  env: JobPollEnv<T>,
): Effect<JobPollAction<T>> | null {
  switch (action.tag) {
    case "start": {
      if (draft.status === "pending") return null;
      draft.status = "pending";
      draft.jobId = null;
      draft.result = null;
      draft.error = null;
      draft.attempt = 0;
      return env
        .submitJob()
        .map((result): JobPollAction<T> => ({ tag: "submitted", result }))
        .catch((e): JobPollAction<T> => ({ tag: "failed", error: String(e) }));
    }
    case "submitted": {
      if (action.result.status === "done") {
        draft.status = "done";
        draft.result = action.result.result;
        return null;
      }
      draft.jobId = action.result.jobId;
      return Effect.send<JobPollAction<T>>({ tag: "tick", jobId: action.result.jobId });
    }
    case "tick": {
      if (draft.jobId !== action.jobId) return null; // superseded by a newer job
      return env
        .pollJob(action.jobId)
        .map((result): JobPollAction<T> => ({ tag: "polled", jobId: action.jobId, result }))
        .catch((e): JobPollAction<T> => ({ tag: "failed", error: String(e) }));
    }
    case "polled": {
      if (draft.jobId !== action.jobId) return null; // stale poll from a superseded job
      if (action.result.status === "done") {
        draft.status = "done";
        draft.result = action.result.result;
        return null;
      }
      if (action.result.status === "error") {
        draft.status = "error";
        draft.error = action.result.error;
        return null;
      }
      draft.attempt += 1;
      if (draft.attempt >= JOB_POLL_MAX_ATTEMPTS) {
        draft.status = "error";
        draft.error = "Timed out waiting for job to complete";
        return null;
      }
      return Effect.delay(nextDelayMs(draft.attempt), {
        tag: "tick",
        jobId: action.jobId,
      } as JobPollAction<T>);
    }
    case "failed": {
      draft.status = "error";
      draft.error = action.error;
      return null;
    }
  }
}
