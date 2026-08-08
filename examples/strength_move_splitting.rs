// Background strength comparison: the currently-shipped linearized
// (Piece/Orientation/Cell sub-action) Druid representation vs. the flat
// Move(Piece, u8) representation it replaced, both playing the shipped
// Strong preset (server/adapters/druid.rs's `Ucb1DmNst`) at its shipped 3s
// budget. `src/games/druid_flat.rs` is a snapshot of the flat representation
// (its Zobrist hash separately extended to cover player-to-move and hand
// counts, matching the fix the linearized representation already has --
// otherwise `use_transpositions(true)` corrupts its tree the same way it did
// before that fix existed), carried forward solely so this example can run
// both engines in one binary and pit them head-to-head -- the two
// `Druid`/`HashedState` types are otherwise unrelated to each other and
// shouldn't be used for anything else. This cross-engine state translation
// works by round-tripping each side's `State` through JSON
// (`new_to_old`/`old_to_new` below), exploiting the fact that both `State`
// shapes are serde-compatible -- the same wire-boundary property that lets
// the server and `ui/` stay untouched by which internal action
// representation Druid uses.
//
// Question this answers: does linearization, as actually shipped -- which
// lets Strong spend up to ~3x its nominal per-move budget on a real turn,
// since `server/adapters/druid.rs::ai_move` gives every sub-decision the
// *full* configured `max_time` -- play stronger Druid than the flat
// representation did at the same nominal budget? This is a real
// head-to-head (both engines play the same game, alternating whole turns,
// state translated after every ply), not an indirect anchor-opponent
// comparison.
//
// Each real turn the linearized side moves costs up to 3x a flat turn's
// wall-clock (one full budget per sub-decision) -- run this in the
// background via `nohup`, matching this repo's other strength_*.rs scripts,
// never synchronously in-session.
//
// Usage: cargo run --release --example strength_move_splitting
use std::time::Duration;

use mcts::game::{Game, TerminalStatus};
use mcts::games::druid as new_druid;
use mcts::games::druid_flat as old_druid;
use mcts::strategies::mcts::{node::QInit, select, simulate, strategy, SearchConfig, TreeSearch};
use mcts::strategies::Search;
use mcts::bench::tournament::Result as GameResult;

/// Shipped Strong preset shape (`server/adapters/druid.rs`'s `Ucb1DmNst`),
/// generic over `Game` so the same alias drives both engines under test.
type Ucb1DmNst<G> = strategy::Compose<select::Ucb1, simulate::DecisiveMove<G, simulate::EpsilonGreedy<G, simulate::Nst>>>;

fn ai_thread_count() -> usize {
    std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1)
}

/// Byte-for-byte the shipped Strong/Master config shape from
/// `server/adapters/druid.rs::build_ai`, generic over which `Druid` (old
/// flat or new linearized) it searches.
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
            .simulate(simulate::DecisiveMove::new().inner(
                simulate::EpsilonGreedy::default()
                    .epsilon(0.3)
                    .inner(simulate::Nst::new().backoff_threshold(5)),
            )),
    )
}

/// Translate the new (linearized) side's `State` to the old (flat) side's,
/// via their shared JSON shape. `pending` (new-only) is dropped; safe here
/// because translation only ever happens between whole turns, when a real
/// state's `pending` is always `Pending::None` anyway.
fn new_to_old(state: &new_druid::HashedState) -> old_druid::HashedState {
    let value = serde_json::to_value(state.state()).expect("State always serializes");
    let state: old_druid::State = serde_json::from_value(value).expect("State shapes agree");
    old_druid::HashedState::from_state(state)
}

/// The reverse of `new_to_old`. `pending` is absent on the source and
/// defaults to `Pending::None` (`#[serde(default)]`) on the target -- always
/// correct for the same reason.
fn old_to_new(state: &old_druid::HashedState) -> new_druid::HashedState {
    let value = serde_json::to_value(state.state()).expect("State always serializes");
    let state: new_druid::State = serde_json::from_value(value).expect("State shapes agree");
    new_druid::HashedState::from_state(state)
}

const BOARD: new_druid::Size = new_druid::Size { w: 5, h: 5 };
const BOARD_OLD: old_druid::Size = old_druid::Size { w: 5, h: 5 };

/// Play one game to completion. Both `TreeSearch`es persist across the
/// whole game (not recreated per move), matching `strength_reuse_tree.rs`'s
/// convention for these offline comparisons, so `reuse_tree` does real work
/// turn over turn. Returns the terminal status from the new engine's side
/// (kept as the single source of truth after every ply, including plies the
/// old engine played, via `old_to_new`).
fn play_one_game(
    new_engine: &mut TreeSearch<new_druid::Druid, Ucb1DmNst<new_druid::Druid>>,
    old_engine: &mut TreeSearch<old_druid::Druid, Ucb1DmNst<old_druid::Druid>>,
    new_moves_first: bool,
) -> TerminalStatus<new_druid::Player> {
    let mut new_state = new_druid::HashedState::new(BOARD);
    let mut old_state = old_druid::HashedState::new(BOARD_OLD);
    let mut new_to_move = new_moves_first;

    loop {
        let status = new_druid::Druid::terminal_status(&new_state);
        if status != TerminalStatus::NotTerminal {
            return status;
        }
        if new_to_move {
            // Loop through linearized sub-actions (Piece -> Orientation? ->
            // Cell) until the placement completes, same shape as
            // `server/adapters/druid.rs::ai_move`'s loop.
            loop {
                let mv = new_engine.choose_action(&new_state);
                new_state = new_druid::Druid::apply(new_state, &mv);
                if new_state.state().pending == new_druid::Pending::None {
                    break;
                }
            }
            old_state = new_to_old(&new_state);
        } else {
            let mv = old_engine.choose_action(&old_state);
            old_state = old_druid::Druid::apply(old_state, &mv);
            new_state = old_to_new(&old_state);
        }
        new_to_move = !new_to_move;
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
    // Args: [rounds] [budget_ms]. Defaults match the shipped Strong preset's
    // 3s budget, with enough rounds (15 x 2 = 30 games) for a meaningful
    // Wilson CI. A smaller budget_ms is useful for a quick end-to-end smoke
    // test before committing to the real (long) background run.
    let mut args = std::env::args().skip(1);
    let rounds: usize = args.next().and_then(|s| s.parse().ok()).unwrap_or(15);
    let budget_ms: u64 = args.next().and_then(|s| s.parse().ok()).unwrap_or(3000);
    let budget = Duration::from_millis(budget_ms);

    println!("=== move-splitting strength comparison (background job) ===");
    println!("Board: 5x5. Shipped Strong preset (Ucb1DmNst), 3s/sub-decision, tree-parallel across {} cores.", ai_thread_count());
    println!("new = linearized (currently shipped), old = flat Move(Piece,u8) (pre-09eca53, snapshot in druid_flat.rs).");
    println!("Sequential games, {rounds} rounds x 2 (alternating first mover) = {} games.", rounds * 2);
    println!();

    let mut new_engine = strong_config::<new_druid::Druid>(budget);
    let mut old_engine = strong_config::<old_druid::Druid>(budget);

    let mut result = GameResult::default();
    for round in 0..rounds {
        for new_moves_first in [true, false] {
            let game_start = std::time::Instant::now();
            let status = play_one_game(&mut new_engine, &mut old_engine, new_moves_first);
            let new_color = if new_moves_first {
                new_druid::Player::Black
            } else {
                new_druid::Player::White
            };
            match status {
                TerminalStatus::Winner(w) if w == new_color => result.wins += 1,
                TerminalStatus::Winner(_) => result.losses += 1,
                TerminalStatus::Draw => result.draws += 1,
                TerminalStatus::NotTerminal => unreachable!("play_one_game only returns on terminal"),
            }
            println!(
                "round {round} (new {}): {status:?} in {:.1}s -- running: {}",
                if new_moves_first { "first" } else { "second" },
                game_start.elapsed().as_secs_f64(),
                fmt_result(&result)
            );
        }
    }

    println!();
    println!("=== Summary ===");
    println!("new (linearized, currently shipped): {}", fmt_result(&result));
    println!();
    println!("Interpretation: `new`'s win rate is from `new`'s perspective (wins/losses/draws");
    println!("against `old`, not a total game count for either side alone). A 95% Wilson CI");
    println!("entirely above 50% means linearization, as actually shipped (including its up to");
    println!("~3x wall-clock cost per real turn), plays measurably stronger Druid than the flat");
    println!("representation did at the same nominal per-sub-decision budget; a CI straddling");
    println!("50% means no significant difference was found at this sample size.");
}
