pub mod flat_mc;
pub mod human;
pub mod mcts;
pub mod negamax;
pub mod random;

use crate::game::Game;
use serde::Serialize;

/// Versioned final evidence from a strategy's most recent action choice.
///
/// This is deliberately engine-facing rather than a wire-format type: callers
/// can serialize a game's action type however their own boundary requires.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum SearchReportStatus {
    Available,
    Partial,
    Unavailable,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum SearchReportReason {
    StrategyUnsupported,
    SearchNotRun,
    RootParallelPvSingleTree,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum SearchTermination {
    Iterations,
    Time,
    Solved,
    Unknown,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum SearchGraphMode {
    Tree,
    Transpositions,
    DagEdges,
    DagNodes,
    DagBoth,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum SearchWarning {
    ActionsTruncated,
    PrincipalVariationTruncated,
    StructuralDiagnosticsOmitted,
}

/// One root action's final-search evidence. `share` is based on the visits
/// across all explored actions, including rows omitted by the report cap.
#[derive(Debug, Clone, Serialize)]
pub struct SearchActionReport<A> {
    pub action: A,
    pub visits: u32,
    pub share: f64,
    /// Root mover-relative expected value in [-1, 1].
    pub mean_value: f64,
    pub is_proven: bool,
}

/// Schema version 1 of the final report for one `choose_action` call.
#[derive(Debug, Clone, Serialize)]
pub struct SearchReport<A> {
    pub schema_version: u8,
    pub status: SearchReportStatus,
    pub reason: Option<SearchReportReason>,
    pub elapsed_seconds: Option<f64>,
    pub iteration_limit: Option<usize>,
    pub time_limit_seconds: Option<f64>,
    pub completed_iterations: usize,
    pub termination: Option<SearchTermination>,
    pub selected_action: Option<A>,
    pub actions: Vec<SearchActionReport<A>>,
    pub principal_variation: Vec<A>,
    /// Visits held at the root after the search, which can include work
    /// retained from earlier calls when tree reuse is enabled.
    pub root_visits: u32,
    /// Nodes retained after the search, rather than nodes newly allocated by
    /// this particular call (so tree reuse remains observable).
    pub tree_nodes: usize,
    pub mean_depth: Option<f64>,
    pub max_depth: Option<usize>,
    pub graph_mode: Option<SearchGraphMode>,
    pub tt_reads: usize,
    pub tt_writes: usize,
    pub tt_hits: usize,
    pub tt_hit_ratio: Option<f64>,
    pub iterations_per_second: Option<f64>,
    pub warnings: Vec<SearchWarning>,
}

/// Kept as a descriptive alias for consumers that call this a final report.
pub type FinalSearchReport<A> = SearchReport<A>;

impl<A> SearchReport<A> {
    pub fn unavailable(reason: SearchReportReason) -> Self {
        Self {
            schema_version: 1,
            status: SearchReportStatus::Unavailable,
            reason: Some(reason),
            elapsed_seconds: None,
            iteration_limit: None,
            time_limit_seconds: None,
            completed_iterations: 0,
            termination: None,
            selected_action: None,
            actions: vec![],
            principal_variation: vec![],
            root_visits: 0,
            tree_nodes: 0,
            mean_depth: None,
            max_depth: None,
            graph_mode: None,
            tt_reads: 0,
            tt_writes: 0,
            tt_hits: 0,
            tt_hit_ratio: None,
            iterations_per_second: None,
            warnings: vec![],
        }
    }
}

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

    /// Final evidence from the most recent `choose_action`. `state` and
    /// `selected_action` are the pre-move state and action returned by that
    /// call, respectively. Strategies without persistent search evidence
    /// return an explicit unsupported report.
    #[allow(unused_variables)]
    fn search_report(
        &self,
        state: &<Self::G as Game>::S,
        selected_action: &<Self::G as Game>::A,
    ) -> SearchReport<<Self::G as Game>::A> {
        SearchReport::unavailable(SearchReportReason::StrategyUnsupported)
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
