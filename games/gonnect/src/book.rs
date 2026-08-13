//! Opening-book construction via Quasi-Best-First self-play (Chaslot,
//! Winands & Van Den Herik, "Parallel Monte-Carlo Tree Search", 2008 --
//! Algorithm 1, `REFERENCES.md`). Shared between `examples/build_book.rs`
//! (human-facing report + file output) and `main.rs`'s `book_build`
//! `GameAdapter` method (the subprocess-protocol path), so the two never
//! drift apart on how the strategy is wired up.
//!
//! See `mcts::strategies::mcts::select::quasi`'s doc comment for the
//! algorithm itself and `mcts::strategies::mcts::book` for the resulting
//! structure. This module just supplies the game-specific plumbing: a
//! `TreeSearch<Gonnect<N, WORDS>, strategy::QuasiBestFirst>` configured the
//! way `TreeSearch::make_book_entry` requires (`expand_threshold: 0`,
//! `max_iterations: 1`), driven in a loop that folds each finished game
//! back into the book before the next one starts.

use crate::{Gonnect, State};
use mcts::game::Game;
use mcts::strategies::mcts::book::OpeningBook;
use mcts::strategies::mcts::{node::QInit, select, strategy, SearchConfig, TreeSearch};
use mcts::strategies::Search;

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
