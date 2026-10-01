//! In-memory async job store backing `ai_move`/`analyze`'s submit+poll API
//! (see `main.rs`'s route handlers). A long-running MCTS search (tens of
//! seconds to minutes, for a big preset or a large custom iteration count)
//! must never hold an HTTP request open -- the old `TimeoutLayer` just
//! traded "the UI hangs" for "the UI silently breaks" once a legitimate
//! search ran past 30s. `submit` instead starts the work on a blocking
//! thread and returns right away: either the finished result, if it
//! happened to land within a short grace period, or a job id the client
//! polls later via `poll`. `poll` only ever does a map lookup -- it never
//! blocks on the search itself.
//!
//! Abandoned jobs (the tab closed, or the client simply never polls again)
//! are never cancelled: the search has no cooperative-cancellation hook
//! today (threading one through every algorithm's hot loop would be a much
//! larger change than this job queue), and letting it run to completion is
//! harmless -- it occupies one blocking-pool thread until it finishes,
//! exactly as it did before this change, and its result is then swept away
//! unread. `sweep` removes any job older than a TTL, bounding the store for
//! a client that never comes back; a hard cap on live entries is a backstop
//! in case the sweep ever falls behind a burst of abandoned jobs.

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use axum::http::StatusCode;
use serde::Serialize;
use serde_json::Value;
use tokio::sync::Notify;

use crate::adapter::AdapterError;

/// How long `submit` waits to see whether the work finishes inline before
/// falling back to a `pending` response. Comfortably above typical
/// "fast preset" latency and far below anything a client should ever
/// perceive as a hang.
pub const SUBMIT_GRACE: Duration = Duration::from_millis(250);

/// How long a job is kept around after it's done (for a client that never
/// comes back to poll it) or while it's still pending (as a backstop
/// against a search that somehow never finishes).
pub const JOB_TTL: Duration = Duration::from_secs(10 * 60);

/// Hard cap on live entries -- a backstop against `sweep` ever lagging
/// behind a burst of abandoned jobs, not a limit normal use should approach
/// (one browser tab submits one `ai_move`/`analyze` at a time).
pub const MAX_JOBS: usize = 1000;

#[derive(Clone)]
enum JobState {
    Pending,
    Done(Value),
    Error { status: StatusCode, message: String },
}

struct Job {
    state: Mutex<JobState>,
    /// Fires once, when `state` leaves `Pending`. `Notify::notify_one`
    /// stores a permit when there's no waiter yet, so a waiter that checks
    /// `state` (finds it still `Pending`) and only then awaits `notified()`
    /// can never miss the completion that raced it -- see tokio's `Notify`
    /// docs for why this check-then-await order is safe without a lock
    /// held across both steps.
    notify: Notify,
    created_at: Instant,
    completed_at: Mutex<Option<Instant>>,
}

impl Job {
    fn new() -> Arc<Self> {
        Arc::new(Self {
            state: Mutex::new(JobState::Pending),
            notify: Notify::new(),
            created_at: Instant::now(),
            completed_at: Mutex::new(None),
        })
    }

    fn complete(&self, state: JobState) {
        *self.state.lock().unwrap() = state;
        *self.completed_at.lock().unwrap() = Some(Instant::now());
        self.notify.notify_one();
    }

    fn snapshot(&self) -> JobState {
        self.state.lock().unwrap().clone()
    }

    /// Waits until `state` leaves `Pending`, then returns it. Only ever used
    /// by `submit`'s grace-period race -- `poll` must never block.
    async fn wait_done(&self) -> JobState {
        loop {
            match self.snapshot() {
                JobState::Pending => self.notify.notified().await,
                done => return done,
            }
        }
    }

    /// Age since completion, or since creation if still pending.
    fn age(&self, now: Instant) -> Duration {
        let completed = *self.completed_at.lock().unwrap();
        now.duration_since(completed.unwrap_or(self.created_at))
    }
}

/// `submit`'s client-visible outcome -- the wire shape `job-poll.ts` calls
/// `JobSubmitResult`. No `error` variant: an error discovered within the
/// grace period surfaces as an ordinary HTTP error response instead (see
/// `submit`), the same as every other adapter error in this API; one
/// discovered later is only ever observed through `poll`.
#[derive(Debug, Serialize)]
#[serde(tag = "status", rename_all = "lowercase")]
pub enum JobSubmitResponse {
    Done {
        result: Value,
    },
    Pending {
        #[serde(rename = "jobId")]
        job_id: String,
    },
}

/// `poll`'s client-visible outcome -- the wire shape `job-poll.ts` calls
/// `JobPollResult`. An unknown job id (never issued, already consumed by an
/// earlier poll, or swept for age) isn't a variant here: `poll` returns
/// `None` for that, and the route handler turns it into a 404, the same way
/// every other "no such X" in this API is reported.
#[derive(Debug, Serialize)]
#[serde(tag = "status", rename_all = "lowercase")]
pub enum JobPollResponse {
    Pending,
    Done { result: Value },
    Error { error: String },
}

fn next_job_id() -> String {
    static NEXT: AtomicU64 = AtomicU64::new(1);
    format!("job-{}", NEXT.fetch_add(1, Ordering::Relaxed))
}

#[derive(Clone)]
pub struct JobStore {
    jobs: Arc<Mutex<HashMap<String, Arc<Job>>>>,
}

impl Default for JobStore {
    fn default() -> Self {
        Self::new()
    }
}

impl JobStore {
    pub fn new() -> Self {
        Self {
            jobs: Arc::new(Mutex::new(HashMap::new())),
        }
    }

    /// Runs `work` on a blocking thread and reports the outcome. `work`
    /// builds the exact JSON `Value` a `done` response (inline here, or
    /// later via `poll`) hands the client -- this store never knows the
    /// shape of an `ai_move`/`analyze` result, only that it's some `Value`.
    ///
    /// Races the work against `grace`: finishing inline lets the client skip
    /// a poll round trip entirely (true for most presets most of the time),
    /// but the work has already been started on its own task either way, so
    /// a slow search is never held up waiting for this call to return.
    pub async fn submit<F>(
        &self,
        grace: Duration,
        work: F,
    ) -> Result<JobSubmitResponse, AdapterError>
    where
        F: FnOnce() -> Result<Value, AdapterError> + Send + 'static,
    {
        self.sweep(JOB_TTL, MAX_JOBS);

        let job_id = next_job_id();
        let job = Job::new();
        self.jobs.lock().unwrap().insert(job_id.clone(), job.clone());

        let job_for_task = job.clone();
        tokio::spawn(async move {
            let outcome = tokio::task::spawn_blocking(work).await;
            let state = match outcome {
                Ok(Ok(value)) => JobState::Done(value),
                Ok(Err(e)) => JobState::Error {
                    status: e.status,
                    message: e.message,
                },
                Err(join_err) => JobState::Error {
                    status: StatusCode::INTERNAL_SERVER_ERROR,
                    message: format!("job panicked: {join_err}"),
                },
            };
            job_for_task.complete(state);
        });

        match tokio::time::timeout(grace, job.wait_done()).await {
            Ok(JobState::Done(value)) => {
                self.remove(&job_id);
                Ok(JobSubmitResponse::Done { result: value })
            }
            Ok(JobState::Error { status, message }) => {
                self.remove(&job_id);
                Err(AdapterError { status, message })
            }
            Ok(JobState::Pending) => unreachable!("wait_done never returns Pending"),
            Err(_elapsed) => Ok(JobSubmitResponse::Pending { job_id }),
        }
    }

    /// Never blocks: a map lookup plus a mutex-guarded state read. Removes
    /// the job once it reports `done`/`error` -- a terminal result only
    /// ever needs to be delivered once, and dropping it here keeps the
    /// store from growing with results nobody will read again.
    pub fn poll(&self, job_id: &str) -> Option<JobPollResponse> {
        let job = self.jobs.lock().unwrap().get(job_id).cloned()?;
        match job.snapshot() {
            JobState::Pending => Some(JobPollResponse::Pending),
            JobState::Done(value) => {
                self.remove(job_id);
                Some(JobPollResponse::Done { result: value })
            }
            JobState::Error { message, .. } => {
                self.remove(job_id);
                Some(JobPollResponse::Error { error: message })
            }
        }
    }

    fn remove(&self, job_id: &str) {
        self.jobs.lock().unwrap().remove(job_id);
    }

    /// Drops any job older than `ttl` (since completion if it has finished,
    /// otherwise since creation), then -- only if that still leaves more
    /// than `max_jobs` -- evicts the oldest survivors down to that cap.
    /// Called on every `submit` and from a periodic background task in
    /// `main.rs`, so an abandoned job's result can't accumulate forever
    /// even if nothing ever polls it.
    pub fn sweep(&self, ttl: Duration, max_jobs: usize) {
        let now = Instant::now();
        let mut jobs = self.jobs.lock().unwrap();
        jobs.retain(|_, job| job.age(now) < ttl);

        if jobs.len() > max_jobs {
            let mut by_age: Vec<(String, Instant)> = jobs
                .iter()
                .map(|(id, job)| (id.clone(), job.created_at))
                .collect();
            by_age.sort_by_key(|(_, created_at)| *created_at);
            for (id, _) in by_age.into_iter().take(jobs.len() - max_jobs) {
                jobs.remove(&id);
            }
        }
    }

    #[cfg(test)]
    fn len(&self) -> usize {
        self.jobs.lock().unwrap().len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ok_work(value: Value) -> impl FnOnce() -> Result<Value, AdapterError> + Send + 'static {
        move || Ok(value)
    }

    #[tokio::test]
    async fn submit_resolves_inline_when_work_finishes_within_the_grace_period() {
        let store = JobStore::new();
        let result = store
            .submit(Duration::from_millis(200), ok_work(serde_json::json!({"a": 1})))
            .await
            .unwrap();
        match result {
            JobSubmitResponse::Done { result } => assert_eq!(result, serde_json::json!({"a": 1})),
            JobSubmitResponse::Pending { .. } => panic!("expected an inline done response"),
        }
        // Resolved inline -- nothing left to poll for.
        assert_eq!(store.len(), 0);
    }

    #[tokio::test]
    async fn submit_surfaces_an_inline_error_as_err_not_a_done_envelope() {
        let store = JobStore::new();
        let err = store
            .submit(Duration::from_millis(200), || {
                Err(AdapterError::bad_request("bad state"))
            })
            .await
            .unwrap_err();
        assert_eq!(err.status, StatusCode::BAD_REQUEST);
        assert_eq!(err.message, "bad state");
        assert_eq!(store.len(), 0);
    }

    #[tokio::test]
    async fn a_job_stays_pending_across_many_polls_then_completes_and_is_consumed_once() {
        let store = JobStore::new();
        let job_id = match store
            .submit(Duration::ZERO, move || {
                std::thread::sleep(Duration::from_millis(80));
                Ok(serde_json::json!({"done": true}))
            })
            .await
            .unwrap()
        {
            JobSubmitResponse::Pending { job_id } => job_id,
            JobSubmitResponse::Done { .. } => panic!("expected pending with a zero grace period"),
        };

        // Poll several times while the work is still running -- every poll
        // must report pending without blocking or consuming the job.
        for _ in 0..5 {
            match store.poll(&job_id) {
                Some(JobPollResponse::Pending) => {}
                other => panic!("expected pending, got a terminal response too early: {other:?}"),
            }
        }

        // Wait past the work's own sleep, then poll until done (bounded,
        // so a regression that never completes fails instead of hanging).
        let deadline = Instant::now() + Duration::from_secs(2);
        loop {
            match store.poll(&job_id) {
                Some(JobPollResponse::Done { result }) => {
                    assert_eq!(result, serde_json::json!({"done": true}));
                    break;
                }
                Some(JobPollResponse::Pending) => {
                    assert!(Instant::now() < deadline, "job never completed");
                    tokio::time::sleep(Duration::from_millis(10)).await;
                }
                other => panic!("unexpected poll result: {other:?}"),
            }
        }

        // A done job is consumed by the poll that first observes it.
        assert!(store.poll(&job_id).is_none());
        assert_eq!(store.len(), 0);
    }

    #[tokio::test]
    async fn an_error_discovered_after_the_grace_period_is_reported_once_then_consumed() {
        let store = JobStore::new();
        let job_id = match store
            .submit(Duration::ZERO, || {
                std::thread::sleep(Duration::from_millis(50));
                Err(AdapterError::internal("search crashed"))
            })
            .await
            .unwrap()
        {
            JobSubmitResponse::Pending { job_id } => job_id,
            JobSubmitResponse::Done { .. } => panic!("expected pending with a zero grace period"),
        };

        let deadline = Instant::now() + Duration::from_secs(2);
        loop {
            match store.poll(&job_id) {
                Some(JobPollResponse::Error { error }) => {
                    assert_eq!(error, "search crashed");
                    break;
                }
                Some(JobPollResponse::Pending) => {
                    assert!(Instant::now() < deadline, "job never completed");
                    tokio::time::sleep(Duration::from_millis(10)).await;
                }
                other => panic!("unexpected poll result: {other:?}"),
            }
        }

        assert!(store.poll(&job_id).is_none());
    }

    #[tokio::test]
    async fn poll_of_an_unknown_job_id_is_none() {
        let store = JobStore::new();
        assert!(store.poll("no-such-job").is_none());
    }

    #[tokio::test]
    async fn sweep_evicts_jobs_older_than_ttl_but_leaves_fresh_ones() {
        let store = JobStore::new();
        let stale_id = match store
            .submit(Duration::ZERO, || {
                std::thread::sleep(Duration::from_millis(10));
                Ok(serde_json::json!(1))
            })
            .await
            .unwrap()
        {
            JobSubmitResponse::Pending { job_id } => job_id,
            JobSubmitResponse::Done { .. } => panic!("expected pending"),
        };
        tokio::time::sleep(Duration::from_millis(30)).await;

        let fresh_id = match store
            .submit(Duration::ZERO, || {
                std::thread::sleep(Duration::from_millis(500));
                Ok(serde_json::json!(2))
            })
            .await
            .unwrap()
        {
            JobSubmitResponse::Pending { job_id } => job_id,
            JobSubmitResponse::Done { .. } => panic!("expected pending"),
        };

        // A TTL shorter than the stale job's age (it finished its 10ms
        // sleep well before the 30ms wait above) but longer than the fresh
        // job's.
        store.sweep(Duration::from_millis(20), MAX_JOBS);

        assert!(store.poll(&stale_id).is_none(), "stale job should be swept");
        assert!(
            matches!(store.poll(&fresh_id), Some(JobPollResponse::Pending)),
            "fresh job should survive the sweep"
        );
    }

    #[tokio::test]
    async fn sweep_backstop_evicts_the_oldest_entries_once_over_the_cap() {
        let store = JobStore::new();
        let mut ids = Vec::new();
        for n in 0..3 {
            let id = match store
                .submit(Duration::ZERO, move || {
                    std::thread::sleep(Duration::from_millis(500));
                    Ok(serde_json::json!(n))
                })
                .await
                .unwrap()
            {
                JobSubmitResponse::Pending { job_id } => job_id,
                JobSubmitResponse::Done { .. } => panic!("expected pending"),
            };
            ids.push(id);
            // Keep insertion order distinguishable by creation time.
            tokio::time::sleep(Duration::from_millis(5)).await;
        }

        // A generous TTL (nothing is actually stale) but a cap of 1 -- only
        // the backstop, not age, should be doing the evicting here.
        store.sweep(Duration::from_secs(60), 1);

        assert!(store.poll(&ids[0]).is_none(), "oldest job should be evicted");
        assert!(store.poll(&ids[1]).is_none(), "middle job should be evicted");
        assert!(
            matches!(store.poll(&ids[2]), Some(JobPollResponse::Pending)),
            "newest job should survive the cap"
        );
    }
}
