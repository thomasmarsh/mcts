// Background strength comparison: the shipped move-split (Piece/Orientation/
// Cell sub-action) Druid representation vs. the flat (whole-turn PlacedPiece)
// representation it replaced, both playing the shipped Strong preset
// (Ucb1DmNst) at its shipped 3s budget. The two engines are the same
// `DruidGame<M>` type, differing only in the `M` parameter (`Split` vs
// `Flat`), selected via the `--flat` flag.
//
// Question this answers: does linearization, as actually shipped -- which
// lets Strong spend up to ~3x its nominal per-move budget on a real turn,
// since every sub-decision gets the *full* configured `max_time` -- play
// stronger Druid than the flat representation did at the same nominal budget?
// This is a real head-to-head (both engines play the same game, alternating
// whole turns, state translated after every ply), not an indirect
// anchor-opponent comparison.
//
// Each real turn move-split side moves costs up to 3x a flat turn's
// wall-clock (one full budget per sub-decision) -- run this in the
// background via `nohup`, matching this repo's other strength_*.rs scripts,
// never synchronously in-session.
//
// Usage: cargo run --release --example strength_move_splitting [--flat]
use std::time::Duration;

use mcts::game::{Game, TerminalStatus};
use mcts::algorithms::mcts::{node::QInit, select, simulate, strategy, SearchConfig, TreeSearch};
use mcts::algorithms::Search;

use game_druid::{DruidFlat, DruidSplit, HashedState, Pending, Player, Size};

/// Self-contained match tally (draws count as half a win for the rate).
/// Mirrors `mcts_bench::tournament::Result` so this script doesn't need the
/// bench harness crate as a dependency.
#[derive(Default)]
struct GameResult {
    wins: usize,
    losses: usize,
    draws: usize,
}

impl GameResult {
    fn total(&self) -> usize {
        self.wins + self.losses + self.draws
    }

    fn score(&self) -> f64 {
        self.wins as f64 + 0.5 * self.draws as f64
    }

    /// Point estimate + a Wilson score interval at confidence `z` (1.96 ≈
    /// 95%), the same interval `mcts_bench` reports.
    fn win_rate_ci(&self, z: f64) -> (f64, (f64, f64)) {
        let total = self.total();
        if total == 0 {
            return (0.5, (0.0, 1.0));
        }
        let n = total as f64;
        let successes = self.score();
        let p_hat = successes / n;
        let z2 = z * z;
        let denom = 1.0 + z2 / n;
        let center = p_hat + z2 / (2.0 * n);
        let margin = z * ((p_hat * (1.0 - p_hat) / n) + z2 / (4.0 * n * n)).sqrt();
        let lower = ((center - margin) / denom).max(0.0);
        let upper = ((center + margin) / denom).min(1.0);
        (p_hat, (lower, upper))
    }
}

/// Shipped Strong preset shape (`Ucb1DmNst`), generic over `G` so the
/// same alias drives both engines.
type Ucb1DmNst<G> = strategy::Compose<
    select::Ucb1,
    simulate::DecisiveMove<G, simulate::EpsilonGreedy<G, simulate::Nst>>,
>;

fn ai_thread_count() -> usize {
    std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1)
}

/// Byte-for-byte the shipped Strong/Master config shape from
/// `server/adapters/druid.rs::build_ai`, generic over `G` so it searches
/// either `DruidGame<Split>` or `DruidGame<Flat>`.
fn strong_config<G: Game>(budget: Duration) -> TreeSearch<G, Ucb1DmNst<G>> {
    TreeSearch::new().config(
        SearchConfig::new()
            .name("strength/strong")
            .expand_threshold(1)
            .use_transpositions(true)
            .use_mcts_solver(true)
            .reuse_tree(true)
            .q_init(QInit::Infinity)
            .max_time(budget)
            .num_tree_threads(ai_thread_count())
            .select(select::Ucb1::with_c(1.414))
            .simulate(
                simulate::DecisiveMove::new().inner(
                    simulate::EpsilonGreedy::default()
                        .epsilon(0.3)
                        .inner(simulate::Nst::new().backoff_threshold(5)),
                ),
            ),
    )
}

const BOARD: Size = Size { w: 5, h: 5 };

/// Play one game to completion. Both engines persist across the whole game.
/// Both `DruidGame<Split>` and `DruidGame<Flat>` act on the *same*
/// `HashedState`, so one shared state object is passed from side to side
/// (no translation needed -- that is the point of the unified core).
/// `split_moves_first=true` means the split side plays Black.
fn play_one_game(
    split_engine: &mut TreeSearch<DruidSplit, Ucb1DmNst<DruidSplit>>,
    flat_engine: &mut TreeSearch<DruidFlat, Ucb1DmNst<DruidFlat>>,
    split_moves_first: bool,
) -> TerminalStatus<Player> {
    let mut state = HashedState::new(BOARD);
    let mut split_to_move = split_moves_first;

    loop {
        let status = DruidSplit::terminal_status(&state);
        if status != TerminalStatus::NotTerminal {
            return status;
        }
        if split_to_move {
            // Loop through split sub-actions (Piece -> Orientation? -> Cell).
            loop {
                let mv = split_engine.choose_action(&state);
                state = DruidSplit::apply(state, &mv);
                if state.state().pending == Pending::None {
                    break;
                }
            }
        } else {
            let mv = flat_engine.choose_action(&state);
            state = DruidFlat::apply(state, &mv);
        }
        split_to_move = !split_to_move;
    }
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
    let mut args = std::env::args().skip(1);
    let rounds: usize = args.next().and_then(|s| s.parse().ok()).unwrap_or(15);
    let budget_ms: u64 = args.next().and_then(|s| s.parse().ok()).unwrap_or(3000);
    let budget = Duration::from_millis(budget_ms);

    println!("=== move-splitting strength comparison (background job) ===");
    println!("Board: 5x5. Shipped Strong preset (Ucb1DmNst), 3s/sub-decision, tree-parallel across {} cores.", ai_thread_count());
    println!(
        "new = move-split (currently shipped, `DruidGame<Split>`), old = flat `DruidGame<Flat>`."
    );
    println!(
        "Sequential games, {rounds} rounds x 2 (alternating first mover) = {} games.",
        rounds * 2
    );
    println!();

    let mut split_engine = strong_config::<DruidSplit>(budget);
    let mut flat_engine = strong_config::<DruidFlat>(budget);

    let mut result = GameResult::default();
    for round in 0..rounds {
        for split_moves_first in [true, false] {
            let game_start = std::time::Instant::now();
            let status = play_one_game(&mut split_engine, &mut flat_engine, split_moves_first);
            let split_color = if split_moves_first {
                Player::Black
            } else {
                Player::White
            };
            match status {
                TerminalStatus::Winner(w) if w == split_color => result.wins += 1,
                TerminalStatus::Winner(_) => result.losses += 1,
                TerminalStatus::Draw => result.draws += 1,
                TerminalStatus::NotTerminal => {
                    unreachable!("play_one_game only returns on terminal")
                }
            }
            println!(
                "round {round} (split {}): {status:?} in {:.1}s -- running: {}",
                if split_moves_first { "first" } else { "second" },
                game_start.elapsed().as_secs_f64(),
                fmt_result(&result)
            );
        }
    }

    println!();
    println!("=== Summary ===");
    println!(
        "new (move-split, currently shipped): {}",
        fmt_result(&result)
    );
    println!();
    println!("Interpretation: `new`'s win rate is from `new`'s perspective (wins/losses/draws");
    println!("against `old`, not a total game count for either side alone). A 95% Wilson CI");
    println!("entirely above 50% means linearization, as actually shipped (including its up to");
    println!("~3x wall-clock cost per real turn), plays measurably stronger Druid than the flat");
    println!("representation did at the same nominal per-sub-decision budget; a CI straddling");
    println!("50% means no significant difference was found at this sample size.");
}
