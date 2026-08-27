// Background strength comparison: single-tree ISMCTS with and without
// per-node re-determinization, on 2-player Oh Hell.
//
// `OhHell::State` always stores every seat's hand in full -- that's just how
// the game's `apply`/`generate_actions` are implemented, and it never
// changes. The three strategies compared here differ only in what a search
// is *allowed to look at* before it moves:
//
// - "cheating" runs one ordinary tree search directly against the literal
//   state, so it can see the opponent's true hand the whole game.
// - "ismcts" runs `SearchConfig::use_ismcts` alone: every iteration walks
//   its own `OhHell::determinize`d sample from the root, but that one sample
//   is only ever advanced by `G::apply` for the rest of the iteration's
//   descent -- an opponent's node several plies deep still sees the exact
//   hand guessed at the root, not a fresh guess of its own.
// - "ismcts_redet" additionally sets `SearchConfig::ismcts_redeterminize`:
//   every node visited during descent draws its own fresh
//   `OhHell::determinize`d sample before its legal actions are read, so an
//   opponent's node several plies deep sees a hand resampled from its own
//   point of view, consistent with everything actually played by then, not
//   the one guess made at the root.
//
// Oh Hell is the sharper testbed for this than Ingenious: a full hand of
// trick play gives many decision points per hidden holder (one bid plus one
// card per trick, repeated for every seat, every trick), unlike Ingenious's
// racks (mostly hidden-until-played, one reveal per tile pair, monotonic
// board) -- more opportunities for the root-only guess to still be in force
// many plies later.
//
// Total node budget is matched across all three (`ITERATIONS` each, all
// single-threaded, single-tree). Sequential execution (no rayon fan-out
// across games) so nothing competes with anything else for the machine.
// This is a long-running job -- run as a background process, not
// synchronously.
//
// Usage: cargo run --release --example strength_oh_hell_redeterminize
use game_oh_hell::OhHell;
use mcts::strategies::mcts::{strategy, SearchConfig, TreeSearch};
use mcts::strategies::Search;
use mcts::util::{AnySearch, Verbosity};
use mcts_bench::tournament::{round_robin_multiple, Result as GameResult};

type OhHell2 = OhHell<2, 7>;

const ITERATIONS: usize = 8000;
const ROUNDS: usize = 15;

fn cheating_config(name: &str) -> TreeSearch<OhHell2, strategy::Ucb1> {
    TreeSearch::new().config(
        SearchConfig::new()
            .name(name)
            .max_iterations(ITERATIONS)
            .seed(1),
    )
}

fn ismcts_config(name: &str) -> TreeSearch<OhHell2, strategy::Ucb1> {
    TreeSearch::new().config(
        SearchConfig::new()
            .name(name)
            .max_iterations(ITERATIONS)
            .use_ismcts(true)
            .seed(2),
    )
}

fn ismcts_redeterminize_config(name: &str) -> TreeSearch<OhHell2, strategy::Ucb1> {
    TreeSearch::new().config(
        SearchConfig::new()
            .name(name)
            .max_iterations(ITERATIONS)
            .use_ismcts(true)
            .ismcts_redeterminize(true)
            .seed(3),
    )
}

fn fmt_result(r: &GameResult) -> String {
    let (point, (lo, hi)) = r.win_rate_ci(1.96);
    format!(
        "W={} L={} D={} total={} win_rate={:.1}% [{:.1}%, {:.1}%] (95% Wilson)",
        r.wins,
        r.losses,
        r.draws,
        r.total(),
        point * 100.0,
        lo * 100.0,
        hi * 100.0
    )
}

fn main() {
    println!("=== Oh Hell ISMCTS re-determinization strength comparison (background job) ===");
    println!(
        "Matched node budget: {ITERATIONS} simulations/move per tree. 3 strategies, {ROUNDS} \
         rounds, {} games (6 ordered pairs/round).",
        ROUNDS * 6
    );
    println!();

    let mut strategies: Vec<AnySearch<OhHell2>> = vec![
        AnySearch::new(cheating_config("cheating")),
        AnySearch::new(ismcts_config("ismcts")),
        AnySearch::new(ismcts_redeterminize_config("ismcts_redet")),
    ];
    let results = round_robin_multiple::<OhHell2, _>(
        &mut strategies,
        ROUNDS,
        &mut std::io::stdout(),
        Verbosity::Verbose,
    );

    println!();
    println!("=== Summary ===");
    for (i, r) in results.iter().enumerate() {
        println!("  {} : {}", strategies[i].friendly_name(), fmt_result(r));
    }
    println!();
    println!(
        "Interpretation: \"ismcts\" and \"ismcts_redet\" both search only realistic \
         determinizations against \"cheating\"'s full view of the opponent's hand, at matched \
         total node budget, isolating what hiding the hand costs a search that can no longer \
         see it -- not a claim that either should beat cheating, since cheating has strictly \
         more information available to it. The \"ismcts\" vs \"ismcts_redet\" head-to-head \
         result is the one that matters: if re-determinizing every node during descent doesn't \
         clearly beat leaving the root's one guess in force for the whole iteration, that gap \
         may not be worth the extra determinize calls it costs."
    );
}
