//! Opening-book construction via Quasi-Best-First self-play (Chaslot,
//! Winands & Van Den Herik, "Parallel Monte-Carlo Tree Search", 2008 --
//! Algorithm 1, `REFERENCES.md`). Shared between `examples/build_book.rs`
//! (human-facing report + file output) and `main.rs`'s `book_build`
//! `GameAdapter` method (the subprocess-protocol path), so the two never
//! drift apart on how the strategy is wired up.
//!
//! See `mcts::algorithms::mcts::select::quasi`'s doc comment for the
//! algorithm itself and `mcts::algorithms::mcts::book` for the resulting
//! structure. This module supplies the game-specific plumbing for both
//! directions: building a book (`build`, below) and consulting one during
//! live play (`BookIndex`/`BookAugmented`, at the bottom of this file) --
//! a `TreeSearch<Gonnect, _>` over an epsilon-greedy-wrapped Quasi-Best-First
//! selection policy (see `BookBuild` below), configured the way
//! `TreeSearch::make_book_entry` requires (`expand_threshold: 0`,
//! `max_iterations: 1`), driven in a loop that folds each finished game
//! back into the book before the next one starts.

use crate::{Gonnect, Move, State};
use mcts::algorithms::mcts::book::OpeningBook;
use mcts::algorithms::mcts::index;
use mcts::algorithms::mcts::{
    backprop, node::QInit, profile::Mcts, select, simulate, SearchConfig, TreeSearch,
};
use mcts::algorithms::{ActionReport, RootReport, Search};
use mcts::game::Game;
use mcts::game::PlayerIndex;
use std::collections::HashMap;

/// The lower-level rollout search QBF falls back to: plain UCB1 selection with
/// epsilon-greedy MAST simulations (classic backprop, robust-child final move).
type BookInner = Mcts<select::Ucb1, simulate::EpsilonGreedy<Gonnect, simulate::Mast>>;

/// The book-building search: Quasi-Best-First selection wrapped in a top-level
/// epsilon-greedy exploration layer, uniform simulations, classic backprop, and
/// a highest-average-score final-move rule.
type BookBuild = Mcts<
    select::EpsilonGreedy<Gonnect, select::QuasiBestFirst<Gonnect, BookInner>>,
    simulate::Uniform,
    backprop::Classic,
    select::MaxAvgScore,
>;

#[derive(Clone, Debug)]
pub struct BookBuildConfig {
    /// Number of self-play games to fold into the book.
    pub rounds: u32,
    /// Iteration budget for the lower-level UCB1 + MAST search (`BookInner`)
    /// QBF falls
    /// back to whenever the book doesn't yet have a confident opinion at
    /// the current position (see `select::quasi`'s "MoGoChoice").
    pub inner_iterations: usize,
    /// Top-level uniform-random exploration rate, wrapping QBF itself
    /// (the `select::EpsilonGreedy` layer around `select::QuasiBestFirst`
    /// in `BookBuild`) -- distinct
    /// from `select::QuasiBestFirst`'s own `epsilon` field, which instead
    /// governs how eagerly the *lower-level* search's choice overrides an
    /// under-confident book score.
    pub top_epsilon: f64,
    pub seed: u64,
    /// Number of self-play workers `build` runs concurrently, splitting
    /// `rounds` as evenly as possible across them (see `build`'s doc
    /// comment). `1` (the default) runs everything on the calling thread,
    /// identical to `build`'s behavior before this field existed.
    pub num_workers: usize,
}

impl Default for BookBuildConfig {
    fn default() -> Self {
        Self {
            rounds: 60,
            inner_iterations: 400,
            top_epsilon: 0.1,
            seed: 0,
            num_workers: 1,
        }
    }
}

/// Runs `config.rounds` self-play games on a `size x size` board, split as
/// evenly as possible across `config.num_workers` worker threads, and
/// returns the combined book.
///
/// `seed`, if given, is folded in as each worker's starting point: every
/// worker plays its own share of games against a private clone of `seed`
/// (so QBF's action selection benefits from `seed`'s existing stats exactly
/// as it would benefit from earlier rounds within a single-worker run), but
/// only records its *own* new games into a private delta book, so combining
/// every worker's delta with one copy of `seed` (via `OpeningBook::merge`)
/// never double-counts `seed`'s own history. Passing `None` starts from an
/// empty book, as every build did before this parameter existed. This is
/// what lets a caller amend a previously built book -- load it, pass it as
/// `seed`, and the returned book is `seed` plus `rounds` additional games,
/// rather than `rounds` games starting over from scratch.
///
/// Calls `on_game(round, plies, utilities)` once per finished game as it
/// completes; `round` is a 0-based slot in `0..config.rounds` unique to
/// that game, but with `num_workers > 1` calls arrive in completion order
/// across workers, not strictly increasing. `plies` is the finished game's
/// length; `utilities` is that game's per-player outcome.
pub fn build(
    size: usize,
    config: &BookBuildConfig,
    seed: Option<&OpeningBook<<Gonnect as Game>::A>>,
    mut on_game: impl FnMut(u32, usize, &[f64]),
) -> OpeningBook<<Gonnect as Game>::A> {
    let num_players = Gonnect::num_players();
    let num_workers = config.num_workers.max(1);
    let base_rounds = config.rounds / num_workers as u32;
    let extra_rounds = config.rounds % num_workers as u32;
    let initial = State::new(size);

    let (tx, rx) = std::sync::mpsc::channel::<(u32, usize, Vec<f64>)>();
    let deltas: Vec<OpeningBook<<Gonnect as Game>::A>> = std::thread::scope(|scope| {
        let mut handles = Vec::with_capacity(num_workers);
        let mut round_start = 0u32;
        for worker in 0..num_workers {
            let worker_rounds = base_rounds + u32::from((worker as u32) < extra_rounds);
            let round_base = round_start;
            round_start += worker_rounds;
            let tx = tx.clone();
            let initial = initial.clone();
            // Distinct, well-separated per-worker RNG seeds so parallel
            // workers don't just replay the same games -- see `select::
            // quasi`'s epsilon-greedy fallback for where randomness enters.
            let worker_seed = config
                .seed
                .wrapping_add((worker as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15));

            handles.push(scope.spawn(move || {
                let inner_search = TreeSearch::<Gonnect, BookInner>::new().config(
                    SearchConfig::new()
                        .name("gonnect/book-inner")
                        .expand_threshold(1)
                        .max_iterations(config.inner_iterations)
                        .q_init(QInit::Infinity),
                );

                let qbf = select::QuasiBestFirst::<Gonnect, BookInner>::new().search(inner_search);

                let mut top_select = select::EpsilonGreedy::<Gonnect, _>::new()
                    .epsilon(config.top_epsilon)
                    .inner(qbf);
                if let Some(seed_book) = seed {
                    top_select.inner.book = seed_book.clone();
                }

                let mut search = TreeSearch::<Gonnect, BookBuild>::new().config(
                    SearchConfig::new()
                        .name("gonnect/book-build")
                        .select(top_select)
                        .expand_threshold(0)
                        .max_iterations(1)
                        .seed(worker_seed),
                );

                let mut delta = OpeningBook::new(num_players);
                for local_round in 0..worker_rounds {
                    let (actions, utilities) = search.make_book_entry(&initial);
                    let plies = actions.len();
                    if !actions.is_empty() {
                        search.config.select.inner.book.add(&actions, &utilities);
                        delta.add(&actions, &utilities);
                    }
                    let _ = tx.send((round_base + local_round, plies, utilities));
                }
                delta
            }));
        }
        drop(tx);
        for (round, plies, utilities) in rx.iter() {
            on_game(round, plies, &utilities);
        }
        handles.into_iter().map(|h| h.join().unwrap()).collect()
    });

    let mut book = seed
        .cloned()
        .unwrap_or_else(|| OpeningBook::new(num_players));
    for delta in deltas {
        book.merge(&delta);
    }
    book
}

/// A loaded opening book plus a state -> book-node index (see
/// `OpeningBook::build_state_index`'s doc comment), so a live position can
/// be looked up directly instead of needing the move sequence that reached
/// it -- `GameAdapter::ai_move`/`analyze` only ever see the current state.
pub struct BookIndex {
    book: OpeningBook<Move>,
    state_index: HashMap<State, index::Id>,
}

impl BookIndex {
    /// Reads and indexes the book file at `path`, for a `size x size` board.
    /// `None` on any failure (missing file, unparseable JSON) -- a book
    /// that hasn't been built yet for this size is the normal case, not an
    /// error, so callers fall back to unaugmented search rather than
    /// propagating one.
    pub fn load(path: &std::path::Path, size: usize) -> Option<Self> {
        let json = std::fs::read_to_string(path).ok()?;
        let book: OpeningBook<Move> = serde_json::from_str(&json).ok()?;
        let state_index = book.build_state_index::<Gonnect>(State::new(size));
        Some(Self { book, state_index })
    }
}

/// Minimum book visits a position needs before its top-visited reply is
/// trusted enough to play directly, instead of falling through to search --
/// guards against a single self-play game's outcome dictating a live move.
pub const MIN_BOOK_VISITS: u64 = 5;

/// Wraps a `Search` with an opening-book lookup consulted only at the outer
/// `choose_action` boundary, never fed into the wrapped search's own
/// selection bias -- so this works identically regardless of which
/// algorithm `inner` uses (UCB1, RAVE, MAST, PN-MCTS, ...). The book is
/// just a per-position visit/win-rate table; it has no algorithm-specific
/// state to integrate with, and a PN-MCTS-style exactness guarantee isn't
/// undermined since the book never touches search internals, only replaces
/// `choose_action`'s result outright when confident.
pub struct BookAugmented<'a> {
    inner: Box<dyn Search<G = Gonnect>>,
    book: &'a BookIndex,
    /// Set by the most recent `choose_action` call iff it was answered
    /// from the book, so `root_report` can report the book's own stats
    /// instead of `inner`'s (which never ran).
    last_book_id: Option<index::Id>,
}

impl<'a> BookAugmented<'a> {
    pub fn new(inner: Box<dyn Search<G = Gonnect>>, book: &'a BookIndex) -> Self {
        Self {
            inner,
            book,
            last_book_id: None,
        }
    }
}

impl Search for BookAugmented<'_> {
    type G = Gonnect;

    fn friendly_name(&self) -> String {
        format!("book+{}", self.inner.friendly_name())
    }

    fn set_friendly_name(&mut self, name: &str) {
        self.inner.set_friendly_name(name);
    }

    fn choose_action(&mut self, state: &State) -> Move {
        self.last_book_id = None;
        if let Some(&id) = self.book.state_index.get(state) {
            if self.book.book.num_visits_at(id) >= MIN_BOOK_VISITS {
                let player = Gonnect::player_to_move(state).to_index();
                if let Some((action, _, _)) =
                    self.book.book.children_at(id, player).into_iter().next()
                {
                    self.last_book_id = Some(id);
                    return action;
                }
            }
        }
        self.inner.choose_action(state)
    }

    fn principle_variation(&self) -> Vec<Move> {
        self.inner.principle_variation()
    }

    fn root_report(&self, state: &State) -> RootReport<Move> {
        let Some(id) = self.last_book_id else {
            return self.inner.root_report(state);
        };
        let player = Gonnect::player_to_move(state).to_index();
        let actions: Vec<_> = self
            .book
            .book
            .children_at(id, player)
            .into_iter()
            .map(|(action, visits, score)| ActionReport {
                action,
                visits: visits.min(u32::MAX as u64) as u32,
                // The book stores `score` on a 0..1 scale (`Entry::score`);
                // `ActionReport::mean_value` is documented as -1..1, same
                // convention every other search's `root_report` uses.
                mean_value: score.map_or(0.0, |s| s * 2.0 - 1.0),
                is_proven: false,
            })
            .collect();
        let principal_variation = actions.first().map(|a| a.action).into_iter().collect();
        RootReport {
            total_visits: self.book.book.num_visits_at(id).min(u32::MAX as u64) as u32,
            principal_variation,
            actions,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Arc;

    /// A `Search` stand-in that always returns a fixed action and counts
    /// how many times it was actually asked -- lets tests assert
    /// `BookAugmented` skips the wrapped search entirely on a confident
    /// book hit, rather than merely happening to agree with it.
    struct StubSearch {
        action: Move,
        calls: Arc<AtomicUsize>,
    }

    impl Search for StubSearch {
        type G = Gonnect;

        fn friendly_name(&self) -> String {
            "stub".into()
        }

        fn set_friendly_name(&mut self, _name: &str) {}

        fn choose_action(&mut self, _state: &State) -> Move {
            self.calls.fetch_add(1, Ordering::SeqCst);
            self.action
        }
    }

    /// A 3x3 board (matching `lib.rs`'s own small-board tests) with two
    /// root replies added to the book: `actions[0]` visited
    /// `MIN_BOOK_VISITS` times with a clean win (confident and good),
    /// `actions[1]` visited once with a loss (present, but not confident).
    fn tiny_book_and_moves() -> (OpeningBook<Move>, Vec<Move>) {
        let mut actions = Vec::new();
        Gonnect::generate_actions(&State::new(3), &mut actions);
        assert!(
            actions.len() >= 2,
            "3x3 empty board should have several legal moves"
        );

        let mut book = OpeningBook::new(Gonnect::num_players());
        for _ in 0..MIN_BOOK_VISITS {
            book.add(&[actions[0]], &[1.0, -1.0]);
        }
        book.add(&[actions[1]], &[-1.0, 1.0]);
        (book, actions)
    }

    #[test]
    fn book_index_load_returns_none_for_a_missing_file() {
        let path = std::path::Path::new("/nonexistent/gonnect-book-does-not-exist.json");
        assert!(BookIndex::load(path, 3).is_none());
    }

    #[test]
    fn book_index_load_round_trips_and_maps_replayed_states() {
        let (book, actions) = tiny_book_and_moves();
        let root_id = book.root_id;

        let path = std::env::temp_dir().join(format!(
            "gonnect-book-test-{}-{:?}.json",
            std::process::id(),
            std::thread::current().id()
        ));
        std::fs::write(&path, serde_json::to_string(&book).unwrap()).unwrap();
        let index = BookIndex::load(&path, 3);
        std::fs::remove_file(&path).ok();
        let index = index.expect("just-written book file should load");

        let root = State::new(3);
        assert_eq!(*index.state_index.get(&root).unwrap(), root_id);

        let after_first = Gonnect::apply(root, &actions[0]);
        assert!(index.state_index.contains_key(&after_first));
    }

    #[test]
    fn book_augmented_plays_the_confident_book_move_without_consulting_inner() {
        let (book, actions) = tiny_book_and_moves();
        let state_index = book.build_state_index::<Gonnect>(State::new(3));
        let index = BookIndex { book, state_index };

        let calls = Arc::new(AtomicUsize::new(0));
        let stub = StubSearch {
            action: actions[1],
            calls: calls.clone(),
        };
        let mut augmented = BookAugmented::new(Box::new(stub), &index);

        let chosen = augmented.choose_action(&State::new(3));
        assert_eq!(
            chosen, actions[0],
            "should play the book's confident top move"
        );
        assert_eq!(calls.load(Ordering::SeqCst), 0);
    }

    #[test]
    fn book_augmented_falls_through_below_the_visit_threshold() {
        let mut actions = Vec::new();
        Gonnect::generate_actions(&State::new(3), &mut actions);
        let mut book = OpeningBook::<Move>::new(Gonnect::num_players());
        book.add(&[actions[0]], &[1.0, -1.0]); // one visit -- below MIN_BOOK_VISITS
        let state_index = book.build_state_index::<Gonnect>(State::new(3));
        let index = BookIndex { book, state_index };

        let calls = Arc::new(AtomicUsize::new(0));
        let stub = StubSearch {
            action: actions[1],
            calls: calls.clone(),
        };
        let mut augmented = BookAugmented::new(Box::new(stub), &index);

        let chosen = augmented.choose_action(&State::new(3));
        assert_eq!(
            chosen, actions[1],
            "an under-visited entry shouldn't override the inner search"
        );
        assert_eq!(calls.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn book_augmented_falls_through_off_book() {
        let book = OpeningBook::<Move>::new(Gonnect::num_players());
        let state_index = book.build_state_index::<Gonnect>(State::new(3));
        let index = BookIndex { book, state_index };

        let mut actions = Vec::new();
        Gonnect::generate_actions(&State::new(3), &mut actions);
        let calls = Arc::new(AtomicUsize::new(0));
        let stub = StubSearch {
            action: actions[0],
            calls: calls.clone(),
        };
        let mut augmented = BookAugmented::new(Box::new(stub), &index);

        let chosen = augmented.choose_action(&State::new(3));
        assert_eq!(chosen, actions[0]);
        assert_eq!(calls.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn root_report_reflects_book_stats_after_a_book_hit() {
        let (book, actions) = tiny_book_and_moves();
        let state_index = book.build_state_index::<Gonnect>(State::new(3));
        let index = BookIndex { book, state_index };
        let stub = StubSearch {
            action: actions[1],
            calls: Arc::new(AtomicUsize::new(0)),
        };
        let mut augmented = BookAugmented::new(Box::new(stub), &index);

        let root = State::new(3);
        let chosen = augmented.choose_action(&root);
        let report = augmented.root_report(&root);

        assert_eq!(report.principal_variation, vec![chosen]);
        assert_eq!(report.total_visits, MIN_BOOK_VISITS as u32 + 1);
        let top = report
            .actions
            .iter()
            .find(|a| a.action == chosen)
            .expect("chosen action should be reported");
        assert_eq!(top.visits, MIN_BOOK_VISITS as u32);
        assert!(top.mean_value > 0.9, "a clean win should score close to +1");
    }

    #[test]
    fn root_report_falls_through_to_inner_when_the_book_was_not_consulted() {
        let book = OpeningBook::<Move>::new(Gonnect::num_players());
        let state_index = book.build_state_index::<Gonnect>(State::new(3));
        let index = BookIndex { book, state_index };

        let mut actions = Vec::new();
        Gonnect::generate_actions(&State::new(3), &mut actions);
        let stub = StubSearch {
            action: actions[0],
            calls: Arc::new(AtomicUsize::new(0)),
        };
        let mut augmented = BookAugmented::new(Box::new(stub), &index);

        let root = State::new(3);
        let _ = augmented.choose_action(&root);
        // `StubSearch` doesn't override `root_report`, so `Search`'s
        // default (empty) report proves the call really passed through.
        let report = augmented.root_report(&root);
        assert!(report.actions.is_empty());
        assert_eq!(report.total_visits, 0);
    }
}
