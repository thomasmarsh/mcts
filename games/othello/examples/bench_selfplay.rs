// Measures how fast Othello self-play games can be generated, which is the
// load-bearing cost number for any offline-learning pipeline built on this
// game: a learned evaluator is only cheaper than AlphaZero if the *labelled*
// position supply is cheap, and the two ends of that cost range are uniform
// random play (no search at all) and search-guided play (one MCTS per ply).
//
// Reports positions/sec and games/sec for uniform-random self-play, plus the
// per-ply cost of a fixed-iteration MCTS move, so a data budget ("how many
// positions can I label in an hour?") can be read off directly instead of
// guessed.
//
// Usage: cargo run --release --example bench_selfplay [games] [mcts_iters]
use std::time::Instant;

use game_othello::{Move, Othello, State};
use mcts::algorithms::mcts::{node::QInit, profile, select, simulate, SearchConfig, TreeSearch};
use mcts::algorithms::Search;
use mcts::game::Game;
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};

/// Plays one uniformly-random game, returning the number of plies played.
fn random_game(rng: &mut SmallRng, actions: &mut Vec<Move>) -> usize {
    let mut state = State::default();
    let mut plies = 0;
    while !Othello::is_terminal(&state) {
        actions.clear();
        Othello::generate_actions(&state, actions);
        if actions.is_empty() {
            break;
        }
        let action = actions[rng.gen_range(0..actions.len())];
        state = Othello::apply(state, &action);
        plies += 1;
    }
    plies
}

type Ucb1 = profile::Mcts<select::Ucb1, simulate::Uniform>;

fn mcts_search(iterations: usize) -> TreeSearch<Othello, Ucb1> {
    TreeSearch::new().config(
        SearchConfig::new()
            .name("bench/ucb1")
            .expand_threshold(1)
            .q_init(QInit::Infinity)
            .max_iterations(iterations),
    )
}

fn main() {
    let mut args = std::env::args().skip(1);
    let games: usize = args.next().and_then(|s| s.parse().ok()).unwrap_or(20_000);
    let mcts_iters: usize = args.next().and_then(|s| s.parse().ok()).unwrap_or(800);

    let mut rng = SmallRng::seed_from_u64(0x0741_1E10);
    let mut actions = Vec::new();

    let start = Instant::now();
    let mut plies = 0usize;
    for _ in 0..games {
        plies += random_game(&mut rng, &mut actions);
    }
    let elapsed = start.elapsed().as_secs_f64();
    println!("uniform-random self-play");
    println!("  {games} games, {plies} positions in {elapsed:.3}s");
    println!(
        "  {:.0} games/sec, {:.0} positions/sec",
        games as f64 / elapsed,
        plies as f64 / elapsed
    );

    // One search-labelled ply, from the opening position, so the two costs
    // sit side by side: this is what a *guided* (and thus on-distribution)
    // position costs, versus the random number above.
    let mut search = mcts_search(mcts_iters);
    let state = State::default();
    let probe = 50;
    let start = Instant::now();
    for _ in 0..probe {
        let _ = search.choose_action(&state);
    }
    let elapsed = start.elapsed().as_secs_f64();
    println!("MCTS-labelled plies ({mcts_iters} iterations/move, single-threaded)");
    println!(
        "  {:.2} ms/ply, {:.0} labelled positions/sec",
        elapsed * 1000. / probe as f64,
        probe as f64 / elapsed
    );
}
