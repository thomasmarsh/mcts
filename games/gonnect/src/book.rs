//! Opening-book construction via Quasi-Best-First self-play (Chaslot,
//! Winands & Van Den Herik, "Parallel Monte-Carlo Tree Search", 2008 --
//! Algorithm 1, `REFERENCES.md`). Shared between `examples/build_book.rs`
//! (human-facing report + file output) and `main.rs`'s `book_build`
//! `GameAdapter` method (the subprocess-protocol path), so the two never
//! drift apart on how the strategy is wired up.
//!
//! See `mcts::strategies::mcts::select::quasi`'s doc comment for the
//! algorithm itself and `mcts::strategies::mcts::book` for the resulting
//! structure. This module supplies the game-specific plumbing for both
//! directions: building a book (`build`, below) and consulting one during
//! live play (`BookIndex`/`BookAugmented`, at the bottom of this file) --
//! a `TreeSearch<Gonnect<N, WORDS>, strategy::QuasiBestFirst>` configured
//! the way `TreeSearch::make_book_entry` requires (`expand_threshold: 0`,
//! `max_iterations: 1`), driven in a loop that folds each finished game
//! back into the book before the next one starts.

use crate::{Gonnect, Move, State};
use mcts::game::Game;
use mcts::game::PlayerIndex;
use mcts::strategies::mcts::book::OpeningBook;
use mcts::strategies::mcts::index;
use mcts::strategies::mcts::{node::QInit, select, strategy, SearchConfig, TreeSearch};
use mcts::strategies::{ActionReport, RootReport, Search};
use std::collections::HashMap;

#[derive(Clone, Debug)]
pub struct BookBuildConfig {
    /// Number of self-play games to fold into the book.
    pub rounds: u32,
    /// Iteration budget for the lower-level `Ucb1Mast` search QBF falls
    /// back to whenever the book doesn't yet have a confident opinion at
    /// the current position (see `select::quasi`'s "MoGoChoice").
    pub inner_iterations: usize,
    /// Top-level uniform-random exploration rate, wrapping QBF itself
    /// (`strategy::QuasiBestFirst`'s `EpsilonGreedy` layer) -- distinct
    /// from `select::QuasiBestFirst`'s own `epsilon` field, which instead
    /// governs how eagerly the *lower-level* search's choice overrides an
    /// under-confident book score.
    pub top_epsilon: f64,
    pub seed: u64,
}

impl Default for BookBuildConfig {
    fn default() -> Self {
        Self {
            rounds: 60,
            inner_iterations: 400,
            top_epsilon: 0.1,
            seed: 0,
        }
    }
}

/// Runs `config.rounds` self-play games for board size `(N, WORDS)`,
/// calling `on_game(round, plies, utilities)` after each one, and returns
/// the finished book. `round` is 0-based; `plies` is the finished game's
/// length; `utilities` is that game's per-player outcome.
pub fn build<const N: usize, const WORDS: usize>(
    config: &BookBuildConfig,
    mut on_game: impl FnMut(u32, usize, &[f64]),
) -> OpeningBook<<Gonnect<N, WORDS> as Game>::A> {
    let inner_search = TreeSearch::<Gonnect<N, WORDS>, strategy::Ucb1Mast>::new().config(
        SearchConfig::new()
            .name("gonnect/book-inner")
            .expand_threshold(1)
            .max_iterations(config.inner_iterations)
            .q_init(QInit::Infinity),
    );

    let qbf =
        select::QuasiBestFirst::<Gonnect<N, WORDS>, strategy::Ucb1Mast>::new().search(inner_search);

    let top_select = select::EpsilonGreedy::<Gonnect<N, WORDS>, _>::new()
        .epsilon(config.top_epsilon)
        .inner(qbf);

    let mut search = TreeSearch::<Gonnect<N, WORDS>, strategy::QuasiBestFirst>::new().config(
        SearchConfig::new()
            .name("gonnect/book-build")
            .select(top_select)
            .expand_threshold(0)
            .max_iterations(1)
            .seed(config.seed),
    );

    let initial = State::<N, WORDS>::default();
    for round in 0..config.rounds {
        let (actions, utilities) = search.make_book_entry(&initial);
        let plies = actions.len();
        if !actions.is_empty() {
            search.config.select.inner.book.add(&actions, &utilities);
        }
        on_game(round, plies, &utilities);
    }

    search.config.select.inner.book.clone()
}

/// A loaded opening book plus a state -> book-node index (see
/// `OpeningBook::build_state_index`'s doc comment), so a live position can
/// be looked up directly instead of needing the move sequence that reached
/// it -- `GameAdapter::ai_move`/`analyze` only ever see the current state.
pub struct BookIndex<const N: usize, const WORDS: usize> {
    book: OpeningBook<Move<N, WORDS>>,
    state_index: HashMap<State<N, WORDS>, index::Id>,
}

impl<const N: usize, const WORDS: usize> BookIndex<N, WORDS> {
    /// Reads and indexes the book file at `path`. `None` on any failure
    /// (missing file, unparseable JSON) -- a book that hasn't been built
    /// yet for this size is the normal case, not an error, so callers fall
    /// back to unaugmented search rather than propagating one.
    pub fn load(path: &std::path::Path) -> Option<Self> {
        let json = std::fs::read_to_string(path).ok()?;
        let book: OpeningBook<Move<N, WORDS>> = serde_json::from_str(&json).ok()?;
        let state_index = book.build_state_index::<Gonnect<N, WORDS>>(State::default());
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
pub struct BookAugmented<'a, const N: usize, const WORDS: usize> {
    inner: Box<dyn Search<G = Gonnect<N, WORDS>>>,
    book: &'a BookIndex<N, WORDS>,
    /// Set by the most recent `choose_action` call iff it was answered
    /// from the book, so `root_report` can report the book's own stats
    /// instead of `inner`'s (which never ran).
    last_book_id: Option<index::Id>,
}

impl<'a, const N: usize, const WORDS: usize> BookAugmented<'a, N, WORDS> {
    pub fn new(
        inner: Box<dyn Search<G = Gonnect<N, WORDS>>>,
        book: &'a BookIndex<N, WORDS>,
    ) -> Self {
        Self {
            inner,
            book,
            last_book_id: None,
        }
    }
}

impl<const N: usize, const WORDS: usize> Search for BookAugmented<'_, N, WORDS> {
    type G = Gonnect<N, WORDS>;

    fn friendly_name(&self) -> String {
        format!("book+{}", self.inner.friendly_name())
    }

    fn set_friendly_name(&mut self, name: &str) {
        self.inner.set_friendly_name(name);
    }

    fn choose_action(&mut self, state: &State<N, WORDS>) -> Move<N, WORDS> {
        self.last_book_id = None;
        if let Some(&id) = self.book.state_index.get(state) {
            if self.book.book.num_visits_at(id) >= MIN_BOOK_VISITS {
                let player = Gonnect::<N, WORDS>::player_to_move(state).to_index();
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

    fn principle_variation(&self) -> Vec<Move<N, WORDS>> {
        self.inner.principle_variation()
    }

    fn root_report(&self, state: &State<N, WORDS>) -> RootReport<Move<N, WORDS>> {
        let Some(id) = self.last_book_id else {
            return self.inner.root_report(state);
        };
        let player = Gonnect::<N, WORDS>::player_to_move(state).to_index();
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
    struct StubSearch<const N: usize, const WORDS: usize> {
        action: Move<N, WORDS>,
        calls: Arc<AtomicUsize>,
    }

    impl<const N: usize, const WORDS: usize> Search for StubSearch<N, WORDS> {
        type G = Gonnect<N, WORDS>;

        fn friendly_name(&self) -> String {
            "stub".into()
        }

        fn set_friendly_name(&mut self, _name: &str) {}

        fn choose_action(&mut self, _state: &State<N, WORDS>) -> Move<N, WORDS> {
            self.calls.fetch_add(1, Ordering::SeqCst);
            self.action
        }
    }

    /// A 3x3 board (matching `lib.rs`'s own small-board tests) with two
    /// root replies added to the book: `actions[0]` visited
    /// `MIN_BOOK_VISITS` times with a clean win (confident and good),
    /// `actions[1]` visited once with a loss (present, but not confident).
    fn tiny_book_and_moves() -> (OpeningBook<Move<3, 1>>, Vec<Move<3, 1>>) {
        let mut actions = Vec::new();
        Gonnect::<3, 1>::generate_actions(&State::default(), &mut actions);
        assert!(
            actions.len() >= 2,
            "3x3 empty board should have several legal moves"
        );

        let mut book = OpeningBook::new(Gonnect::<3, 1>::num_players());
        for _ in 0..MIN_BOOK_VISITS {
            book.add(&[actions[0]], &[1.0, -1.0]);
        }
        book.add(&[actions[1]], &[-1.0, 1.0]);
        (book, actions)
    }

    #[test]
    fn book_index_load_returns_none_for_a_missing_file() {
        let path = std::path::Path::new("/nonexistent/gonnect-book-does-not-exist.json");
        assert!(BookIndex::<3, 1>::load(path).is_none());
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
        let index = BookIndex::<3, 1>::load(&path);
        std::fs::remove_file(&path).ok();
        let index = index.expect("just-written book file should load");

        let root = State::<3, 1>::default();
        assert_eq!(*index.state_index.get(&root).unwrap(), root_id);

        let after_first = Gonnect::<3, 1>::apply(root, &actions[0]);
        assert!(index.state_index.contains_key(&after_first));
    }

    #[test]
    fn book_augmented_plays_the_confident_book_move_without_consulting_inner() {
        let (book, actions) = tiny_book_and_moves();
        let state_index = book.build_state_index::<Gonnect<3, 1>>(State::default());
        let index = BookIndex { book, state_index };

        let calls = Arc::new(AtomicUsize::new(0));
        let stub = StubSearch {
            action: actions[1],
            calls: calls.clone(),
        };
        let mut augmented = BookAugmented::new(Box::new(stub), &index);

        let chosen = augmented.choose_action(&State::default());
        assert_eq!(
            chosen, actions[0],
            "should play the book's confident top move"
        );
        assert_eq!(calls.load(Ordering::SeqCst), 0);
    }

    #[test]
    fn book_augmented_falls_through_below_the_visit_threshold() {
        let mut actions = Vec::new();
        Gonnect::<3, 1>::generate_actions(&State::default(), &mut actions);
        let mut book = OpeningBook::<Move<3, 1>>::new(Gonnect::<3, 1>::num_players());
        book.add(&[actions[0]], &[1.0, -1.0]); // one visit -- below MIN_BOOK_VISITS
        let state_index = book.build_state_index::<Gonnect<3, 1>>(State::default());
        let index = BookIndex { book, state_index };

        let calls = Arc::new(AtomicUsize::new(0));
        let stub = StubSearch {
            action: actions[1],
            calls: calls.clone(),
        };
        let mut augmented = BookAugmented::new(Box::new(stub), &index);

        let chosen = augmented.choose_action(&State::default());
        assert_eq!(
            chosen, actions[1],
            "an under-visited entry shouldn't override the inner search"
        );
        assert_eq!(calls.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn book_augmented_falls_through_off_book() {
        let book = OpeningBook::<Move<3, 1>>::new(Gonnect::<3, 1>::num_players());
        let state_index = book.build_state_index::<Gonnect<3, 1>>(State::default());
        let index = BookIndex { book, state_index };

        let mut actions = Vec::new();
        Gonnect::<3, 1>::generate_actions(&State::default(), &mut actions);
        let calls = Arc::new(AtomicUsize::new(0));
        let stub = StubSearch {
            action: actions[0],
            calls: calls.clone(),
        };
        let mut augmented = BookAugmented::new(Box::new(stub), &index);

        let chosen = augmented.choose_action(&State::default());
        assert_eq!(chosen, actions[0]);
        assert_eq!(calls.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn root_report_reflects_book_stats_after_a_book_hit() {
        let (book, actions) = tiny_book_and_moves();
        let state_index = book.build_state_index::<Gonnect<3, 1>>(State::default());
        let index = BookIndex { book, state_index };
        let stub = StubSearch {
            action: actions[1],
            calls: Arc::new(AtomicUsize::new(0)),
        };
        let mut augmented = BookAugmented::new(Box::new(stub), &index);

        let root = State::<3, 1>::default();
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
        let book = OpeningBook::<Move<3, 1>>::new(Gonnect::<3, 1>::num_players());
        let state_index = book.build_state_index::<Gonnect<3, 1>>(State::default());
        let index = BookIndex { book, state_index };

        let mut actions = Vec::new();
        Gonnect::<3, 1>::generate_actions(&State::default(), &mut actions);
        let stub = StubSearch {
            action: actions[0],
            calls: Arc::new(AtomicUsize::new(0)),
        };
        let mut augmented = BookAugmented::new(Box::new(stub), &index);

        let root = State::<3, 1>::default();
        let _ = augmented.choose_action(&root);
        // `StubSearch` doesn't override `root_report`, so `Search`'s
        // default (empty) report proves the call really passed through.
        let report = augmented.root_report(&root);
        assert!(report.actions.is_empty());
        assert_eq!(report.total_visits, 0);
    }
}
