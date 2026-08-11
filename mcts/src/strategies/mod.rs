pub mod flat_mc;
pub mod human;
pub mod mcts;
pub mod random;

use crate::game::Game;

/// One root action's statistics after a search has run, for reporting
/// candidate moves (e.g. a UI's analysis panel) rather than just the single
/// action `choose_action` picked.
#[derive(Debug, Clone)]
pub struct ActionReport<A> {
    pub action: A,
    /// Number of times this action was selected from the root.
    pub visits: u32,
    /// Expected value from the root's mover's perspective, in [-1, 1].
    pub mean_value: f64,
    /// Whether this action's outcome is proven (MCTS-Solver), i.e. its true
    /// value is known rather than an empirical estimate. Doesn't say
    /// win/loss/draw on its own -- `mean_value` collapses to (approximately)
    /// +1./-1./0. once proven, since backprop keeps biasing search toward a
    /// proven-win child, driving its average toward the true outcome.
    pub is_proven: bool,
}

/// A search's full root report: every explored action's stats, the
/// principal variation, and how much total search went into producing them.
#[derive(Debug, Clone)]
pub struct RootReport<A> {
    pub actions: Vec<ActionReport<A>>,
    pub principal_variation: Vec<A>,
    pub total_visits: u32,
}

pub trait Search: Sync + Send {
    type G: Game;

    fn friendly_name(&self) -> String;

    fn choose_action(&mut self, state: &<Self::G as Game>::S) -> <Self::G as Game>::A;

    fn principle_variation(&self) -> Vec<<Self::G as Game>::A> {
        vec![]
    }

    /// Structured per-root-action statistics from the most recent
    /// `choose_action` call, for callers that want every candidate (e.g. an
    /// analysis panel) rather than just the one action that was picked.
    /// `state` must be the same state `choose_action` was last called with --
    /// this reads existing search state rather than searching again.
    /// Default empty, matching `principle_variation`'s default: strategies
    /// that don't keep a persistent tree (`flat_mc`, `random`, `human`) have
    /// nothing structured to report.
    #[allow(unused_variables)]
    fn root_report(&self, state: &<Self::G as Game>::S) -> RootReport<<Self::G as Game>::A> {
        RootReport {
            actions: vec![],
            principal_variation: vec![],
            total_visits: 0,
        }
    }

    fn estimated_depth(&self) -> usize {
        0
    }

    /// Number of nodes held in this search's arena, for callers that only
    /// have a type-erased `Box<dyn Search>` and want to observe tree reuse
    /// (`mcts::SearchConfig::reuse_tree`) without downcasting. `0` for every
    /// strategy that doesn't keep a persistent arena.
    fn arena_len(&self) -> usize {
        0
    }

    fn set_friendly_name(&mut self, name: &str);

    #[allow(unused_variables)]
    fn make_book_entry(
        &mut self,
        state: &<Self::G as Game>::S,
    ) -> (Vec<<Self::G as Game>::A>, Vec<f64>) {
        unimplemented!();
    }
}

/// See `parallel_test_guard`.
static PARALLEL_TEST_MUTEX: std::sync::OnceLock<std::sync::Mutex<()>> = std::sync::OnceLock::new();

/// Serialises tests that spawn their own thread pools, so cargo's
/// default per-binary test concurrency never overlaps two
/// thread-spawning tests' worker bursts and exhausts RAM.
///
/// Intentionally not gated behind `#[cfg(test)]` so that
/// `mcts-tests` (a separate crate) can use it too.
pub fn parallel_test_guard() -> std::sync::MutexGuard<'static, ()> {
    PARALLEL_TEST_MUTEX
        .get_or_init(|| std::sync::Mutex::new(()))
        .lock()
        .unwrap()
}

#[cfg(test)]
mod tests;
