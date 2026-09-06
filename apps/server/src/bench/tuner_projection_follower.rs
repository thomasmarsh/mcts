//! Server-owned headless projection follower.
//!
//! `report.json` lands once, at run end, and 12e-6..8 kept a live run's
//! science moving only while a browser tab was open on it -- the tab's own
//! refresh loop drove `POST /projection/refresh`. That leaves two gaps:
//! opening the Fleet dashboard cold after an unattended run is stale until the
//! loop catches up, and every viewer pays a fresh `uv run` per tick.
//!
//! This module closes both. While any tuner launch journal shows `live`,
//! exactly one long-lived `tuner-project --watch` child keeps the SQLite
//! projection fresh for every viewer at once. The child is reaped once no run
//! is live and respawned on the next launch; a crash is restarted with a
//! bounded budget that resets once the child has run healthily for a while.
//!
//! The spawner and the "is any run live" probe are injected so the supervisor
//! state machine is unit-tested with no real `uv` and no real journal.

use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use mcts_bench::tuner_launch;

use super::tuner_runs;

/// A running `tuner-project --watch` child. The production implementation
/// wraps [`std::process::Child`]; a test one wraps a flag it flips to
/// simulate a crash.
pub trait FollowerChild: Send {
    /// `true` while the child is still running.
    fn is_running(&mut self) -> bool;
    /// Ask the child to stop and reap it.
    fn terminate(&mut self);
}

/// Spawns one detached watch child. Injected so tests drive the supervisor
/// without a real `uv`.
pub type WatchSpawner =
    Arc<dyn Fn() -> std::io::Result<Box<dyn FollowerChild>> + Send + Sync>;

/// Probe for "does any launch journal row report `live`". Injected for the
/// same reason.
pub type LivenessProbe = Arc<dyn Fn() -> bool + Send + Sync>;

/// How long a child must have been running before a crash resets its restart
/// budget, and the window `reprojected_recently` treats as "the follower has
/// had time to commit at least one pass".
const HEALTHY_WINDOW: Duration = Duration::from_secs(30);
/// Restarts allowed inside one unhealthy streak before the supervisor gives
/// up until the next launch (or the next healthy period).
const MAX_RESTARTS: u32 = 5;
/// Supervisor tick cadence.
pub const TICK: Duration = Duration::from_secs(2);

struct Inner {
    child: Option<Box<dyn FollowerChild>>,
    spawned_at: Option<Instant>,
    restarts: u32,
}

pub struct ProjectionFollower {
    spawn: WatchSpawner,
    any_live: LivenessProbe,
    inner: Mutex<Inner>,
}

impl ProjectionFollower {
    pub fn new(spawn: WatchSpawner, any_live: LivenessProbe) -> Arc<Self> {
        Arc::new(Self {
            spawn,
            any_live,
            inner: Mutex::new(Inner {
                child: None,
                spawned_at: None,
                restarts: 0,
            }),
        })
    }

    /// The production follower: shells `tuner-project --watch` from the repo
    /// root and reads liveness straight from the launch journal.
    pub fn production(runs_root: PathBuf, db: PathBuf, interval_secs: f64) -> Arc<Self> {
        let probe_root = runs_root.clone();
        Self::new(
            Arc::new(move || spawn_watch_process(&runs_root, &db, interval_secs)),
            Arc::new(move || any_run_live(&probe_root)),
        )
    }

    /// One supervisor step: reconcile the child against journal liveness.
    /// Spawns when a run is live and nothing is running, reaps when no run is
    /// live, and restarts a crashed child within its budget. Called on a
    /// timer and directly after a launch returns.
    pub fn tick(&self) {
        let mut inner = self.inner.lock().expect("follower mutex");
        let any_live = (self.any_live)();

        let running = match inner.child.as_mut() {
            Some(child) => child.is_running(),
            None => false,
        };

        if running {
            let healthy = inner
                .spawned_at
                .is_some_and(|at| at.elapsed() >= HEALTHY_WINDOW);
            if healthy {
                inner.restarts = 0;
            }
            if !any_live {
                if let Some(mut child) = inner.child.take() {
                    child.terminate();
                }
                inner.spawned_at = None;
                inner.restarts = 0;
            }
            return;
        }

        // Nothing running. Drop a dead handle and decide whether to (re)spawn.
        let crashed = inner.child.is_some();
        inner.child = None;
        inner.spawned_at = None;
        if !any_live {
            inner.restarts = 0;
            return;
        }
        if crashed {
            inner.restarts += 1;
        }
        if inner.restarts > MAX_RESTARTS {
            return;
        }
        match (self.spawn)() {
            Ok(child) => {
                inner.child = Some(child);
                inner.spawned_at = Some(Instant::now());
            }
            Err(error) => {
                eprintln!("projection follower: spawn failed: {error}");
                inner.restarts += 1;
            }
        }
    }

    /// True when a watch child has been running long enough to have committed
    /// at least one projection pass -- the signal `POST /projection/refresh`
    /// uses to skip a redundant out-of-band shell-out.
    pub fn reprojected_recently(&self) -> bool {
        let inner = self.inner.lock().expect("follower mutex");
        inner.child.is_some()
            && inner
                .spawned_at
                .is_some_and(|at| at.elapsed() >= HEALTHY_WINDOW.min(Duration::from_secs(6)))
    }

    #[cfg(test)]
    fn child_present(&self) -> bool {
        self.inner.lock().expect("follower mutex").child.is_some()
    }

    #[cfg(test)]
    fn restarts(&self) -> u32 {
        self.inner.lock().expect("follower mutex").restarts
    }
}

/// Drive [`ProjectionFollower::tick`] on a timer for the life of the server.
pub fn spawn_supervisor(follower: Arc<ProjectionFollower>) {
    tokio::spawn(async move {
        let mut interval = tokio::time::interval(TICK);
        loop {
            interval.tick().await;
            let follower = follower.clone();
            // `tick` does blocking file / process work; keep it off the
            // async worker.
            let _ = tokio::task::spawn_blocking(move || follower.tick()).await;
        }
    });
}

fn any_run_live(runs_root: &Path) -> bool {
    tuner_launch::records(runs_root)
        .map(|records| {
            records
                .iter()
                .any(|record| tuner_runs::liveness(record) == "live")
        })
        .unwrap_or(false)
}

fn spawn_watch_process(
    runs_root: &Path,
    db: &Path,
    interval_secs: f64,
) -> std::io::Result<Box<dyn FollowerChild>> {
    let child = std::process::Command::new("uv")
        .args([
            "run",
            "--project",
            "tuner",
            "tuner-project",
            "--watch",
            "--interval",
        ])
        .arg(format!("{interval_secs}"))
        .arg("--runs-root")
        .arg(runs_root)
        .arg("--db")
        .arg(db)
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .spawn()?;
    Ok(Box::new(ProcessChild(child)))
}

struct ProcessChild(std::process::Child);

impl FollowerChild for ProcessChild {
    fn is_running(&mut self) -> bool {
        matches!(self.0.try_wait(), Ok(None))
    }

    fn terminate(&mut self) {
        let _ = self.0.kill();
        let _ = self.0.wait();
    }
}

impl Drop for ProcessChild {
    /// Don't outlive the server: a dropped follower (shutdown, or a reap that
    /// takes the `Box`) takes its `tuner-project --watch` child with it.
    fn drop(&mut self) {
        let _ = self.0.kill();
        let _ = self.0.wait();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

    /// A fake child whose `alive` flag a test flips to simulate a crash.
    struct FakeChild {
        alive: Arc<AtomicBool>,
    }

    impl FollowerChild for FakeChild {
        fn is_running(&mut self) -> bool {
            self.alive.load(Ordering::SeqCst)
        }
        fn terminate(&mut self) {
            self.alive.store(false, Ordering::SeqCst);
        }
    }

    struct Harness {
        live: Arc<AtomicBool>,
        spawns: Arc<AtomicUsize>,
        last_child: Arc<Mutex<Option<Arc<AtomicBool>>>>,
        fail_next: Arc<AtomicBool>,
    }

    fn harness() -> (Arc<ProjectionFollower>, Harness) {
        let live = Arc::new(AtomicBool::new(false));
        let spawns = Arc::new(AtomicUsize::new(0));
        let last_child: Arc<Mutex<Option<Arc<AtomicBool>>>> = Arc::new(Mutex::new(None));
        let fail_next = Arc::new(AtomicBool::new(false));

        let spawn_spawns = spawns.clone();
        let spawn_last = last_child.clone();
        let spawn_fail = fail_next.clone();
        let spawn: WatchSpawner = Arc::new(move || {
            if spawn_fail.swap(false, Ordering::SeqCst) {
                return Err(std::io::Error::other("injected spawn failure"));
            }
            spawn_spawns.fetch_add(1, Ordering::SeqCst);
            let alive = Arc::new(AtomicBool::new(true));
            *spawn_last.lock().unwrap() = Some(alive.clone());
            Ok(Box::new(FakeChild { alive }) as Box<dyn FollowerChild>)
        });

        let probe_live = live.clone();
        let any_live: LivenessProbe = Arc::new(move || probe_live.load(Ordering::SeqCst));

        (
            ProjectionFollower::new(spawn, any_live),
            Harness {
                live,
                spawns,
                last_child,
                fail_next,
            },
        )
    }

    #[test]
    fn spawns_only_while_a_run_is_live_and_reaps_when_none_is() {
        let (follower, h) = harness();

        follower.tick();
        assert!(!follower.child_present(), "no child while nothing is live");
        assert_eq!(h.spawns.load(Ordering::SeqCst), 0);

        h.live.store(true, Ordering::SeqCst);
        follower.tick();
        assert!(follower.child_present());
        assert_eq!(h.spawns.load(Ordering::SeqCst), 1);

        // Idempotent while still live -- no second child.
        follower.tick();
        assert_eq!(h.spawns.load(Ordering::SeqCst), 1);

        h.live.store(false, Ordering::SeqCst);
        follower.tick();
        assert!(!follower.child_present(), "reaped once no run is live");
        let last = h.last_child.lock().unwrap().clone().unwrap();
        assert!(!last.load(Ordering::SeqCst), "the child was terminated");
    }

    #[test]
    fn restarts_a_crashed_child_within_budget_then_gives_up() {
        let (follower, h) = harness();
        h.live.store(true, Ordering::SeqCst);
        follower.tick();
        assert_eq!(h.spawns.load(Ordering::SeqCst), 1);

        for expected in 2..=(MAX_RESTARTS + 1) {
            h.last_child
                .lock()
                .unwrap()
                .clone()
                .unwrap()
                .store(false, Ordering::SeqCst);
            follower.tick();
            assert_eq!(h.spawns.load(Ordering::SeqCst), expected as usize);
        }

        // Budget exhausted: the next crash is not restarted.
        h.last_child
            .lock()
            .unwrap()
            .clone()
            .unwrap()
            .store(false, Ordering::SeqCst);
        follower.tick();
        assert_eq!(h.spawns.load(Ordering::SeqCst), (MAX_RESTARTS + 1) as usize);
        assert!(!follower.child_present());

        // A fresh launch (still live, but the operator relaunched) does not by
        // itself reset the budget; going quiet does.
        h.live.store(false, Ordering::SeqCst);
        follower.tick();
        assert_eq!(follower.restarts(), 0);
        h.live.store(true, Ordering::SeqCst);
        follower.tick();
        assert_eq!(h.spawns.load(Ordering::SeqCst), (MAX_RESTARTS + 2) as usize);
    }

    #[test]
    fn a_spawn_error_counts_against_the_budget_but_recovers() {
        let (follower, h) = harness();
        h.live.store(true, Ordering::SeqCst);
        h.fail_next.store(true, Ordering::SeqCst);
        follower.tick();
        assert!(!follower.child_present());
        assert_eq!(h.spawns.load(Ordering::SeqCst), 0);

        follower.tick();
        assert!(follower.child_present(), "next tick spawns cleanly");
        assert_eq!(h.spawns.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn reprojected_recently_is_false_until_a_child_has_run_a_while() {
        let (follower, h) = harness();
        h.live.store(true, Ordering::SeqCst);
        follower.tick();
        // Just spawned -- not yet long enough to have committed a pass.
        assert!(!follower.reprojected_recently());
    }
}
