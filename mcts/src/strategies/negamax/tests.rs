// A tiny hand-solvable subtraction game (take 1 or 2 stones, whoever takes
// the last stone wins) used only by these tests, so correctness can be
// checked against a known-by-hand optimal strategy (mod-3 losing
// positions) rather than only "didn't panic, picked a legal move" -- the
// way tic-tac-toe-based tests elsewhere in this crate work, but without
// pulling a `games/*` crate into `mcts` itself (they depend on `mcts`, not
// the other way around; see `strategies::tests::converge_game_tests` for
// the same reason another hand-built `Game` lives in-crate).

use crate::game::{Game, PlayerIndex};
use crate::strategies::negamax::{
    self, Evaluator, MaterialBlind, Negamax, NegamaxOptions, Replacement, DRAW_SCORE, WIN_SCORE,
};
use crate::strategies::Search;

#[derive(Clone, Copy, PartialEq, Eq, Debug, serde::Serialize)]
struct Player(usize);

impl PlayerIndex for Player {
    fn to_index(&self) -> usize {
        self.0
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Default, Debug)]
struct State {
    stones: u8,
    turn: u8,
}

impl std::fmt::Display for State {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "stones={} turn={}", self.stones, self.turn)
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, serde::Serialize)]
struct Take(u8);

#[derive(Clone)]
struct Subtraction;

impl Game for Subtraction {
    type S = State;
    type A = Take;
    type P = Player;

    fn apply(state: Self::S, action: &Self::A) -> Self::S {
        State {
            stones: state.stones - action.0,
            turn: 1 - state.turn,
        }
    }

    fn generate_actions(state: &Self::S, actions: &mut Vec<Self::A>) {
        for k in 1..=2u8.min(state.stones) {
            actions.push(Take(k));
        }
    }

    fn winner(state: &Self::S) -> Option<Self::P> {
        (state.stones == 0).then(|| Player(1 - state.turn as usize))
    }

    fn player_to_move(state: &Self::S) -> Self::P {
        Player(state.turn as usize)
    }

    fn zobrist_hash(state: &Self::S) -> u64 {
        ((state.stones as u64) << 1) | state.turn as u64
    }
}

fn solver() -> Negamax<Subtraction, MaterialBlind> {
    Negamax::new_with_options(MaterialBlind, NegamaxOptions::default().with_max_depth(20))
}

#[test]
fn supports_reports_true_for_a_plain_alternating_two_player_game() {
    assert!(negamax::supports::<Subtraction>());
}

#[test]
fn finds_the_winning_move_from_a_non_multiple_of_three() {
    // 5 stones: the only move that leaves a losing position (a multiple of
    // 3) for the opponent is taking 2 (leaving 3).
    let state = State { stones: 5, turn: 0 };
    let mut search = solver();
    let action = search.choose_action(&state);
    assert_eq!(action, Take(2));
    assert!(
        search.root_score() >= WIN_SCORE - search.depth_reached() as i32,
        "a fully solved forced win should score at (or within mate distance of) WIN_SCORE, got {}",
        search.root_score()
    );
}

#[test]
fn reports_a_loss_from_a_multiple_of_three() {
    // 3 stones is a losing position for whoever is to move -- every legal
    // reply (take 1 leaving 2, take 2 leaving 1) hands the opponent a
    // forced win.
    let state = State { stones: 3, turn: 0 };
    let mut search = solver();
    let _ = search.choose_action(&state);
    assert!(
        search.root_score() <= -WIN_SCORE + search.depth_reached() as i32,
        "every move from a multiple-of-three pile should be a proven loss, got {}",
        search.root_score()
    );
}

#[test]
fn principal_variation_starts_with_the_chosen_move() {
    let state = State { stones: 5, turn: 0 };
    let mut search = solver();
    let action = search.choose_action(&state);
    let pv = search.principle_variation();
    assert_eq!(pv.first(), Some(&action));
    assert!(!pv.is_empty());
}

#[test]
fn root_report_covers_every_legal_move() {
    let state = State { stones: 5, turn: 0 };
    let mut search = solver();
    search.choose_action(&state);
    let report = search.root_report(&state);
    assert_eq!(
        report.actions.len(),
        2,
        "Take(1) and Take(2) are both legal from 5 stones"
    );
    let winning = report
        .actions
        .iter()
        .find(|a| a.action == Take(2))
        .expect("Take(2) should be a reported root action");
    assert!(
        winning.is_proven,
        "the winning move should be reported as proven"
    );
}

#[test]
fn transposition_table_toggle_does_not_change_the_chosen_move() {
    let state = State { stones: 5, turn: 0 };
    let mut with_table = Negamax::<Subtraction, MaterialBlind>::new_with_options(
        MaterialBlind,
        NegamaxOptions::default()
            .with_max_depth(20)
            .with_table_bits(10),
    );
    let mut without_table = Negamax::<Subtraction, MaterialBlind>::new_with_options(
        MaterialBlind,
        NegamaxOptions::default()
            .with_max_depth(20)
            .with_table_bits(0),
    );
    assert_eq!(
        with_table.choose_action(&state),
        without_table.choose_action(&state)
    );
}

#[test]
fn replacement_policy_does_not_change_the_answer() {
    // A tiny table (few bits) is deliberately used so entries actually
    // collide within `choose_action`'s depth-20 search, exercising each
    // policy's eviction path rather than just its never-hit fallback.
    let state = State { stones: 5, turn: 0 };
    let expected = solver().choose_action(&state);
    for replacement in [
        Replacement::Always,
        Replacement::DepthPreferred,
        Replacement::TwoTier,
    ] {
        let mut search = Negamax::<Subtraction, MaterialBlind>::new_with_options(
            MaterialBlind,
            NegamaxOptions::default()
                .with_max_depth(20)
                .with_table_bits(2)
                .with_replacement(replacement),
        );
        assert_eq!(
            search.choose_action(&state),
            expected,
            "replacement policy {replacement:?} changed the chosen move"
        );
    }
}

#[test]
fn max_depth_below_the_game_length_falls_back_to_the_evaluator() {
    // A depth-1 search can't see the game out to a terminal state from 5
    // stones (it needs up to 5 plies), so it must fall back to the static
    // evaluator at the cutoff instead of ever touching `terminal_status`'s
    // WIN/LOSS sentinels.
    struct AlwaysDraw;
    impl Evaluator<Subtraction> for AlwaysDraw {
        fn evaluate(&self, _state: &State) -> i32 {
            DRAW_SCORE
        }
    }
    let state = State { stones: 5, turn: 0 };
    let mut search =
        Negamax::new_with_options(AlwaysDraw, NegamaxOptions::default().with_max_depth(1));
    let action = search.choose_action(&state);
    assert!(action == Take(1) || action == Take(2));
    assert_eq!(
        search.root_score(),
        DRAW_SCORE,
        "an always-draw evaluator at depth 1 should score every move as a draw"
    );
}

#[test]
fn principal_variation_search_toggle_agrees_with_plain_alpha_beta() {
    let state = State { stones: 5, turn: 0 };
    let mut pvs = Negamax::<Subtraction, MaterialBlind>::new_with_options(
        MaterialBlind,
        NegamaxOptions::default()
            .with_max_depth(20)
            .with_principal_variation_search(true),
    );
    let mut plain = Negamax::<Subtraction, MaterialBlind>::new_with_options(
        MaterialBlind,
        NegamaxOptions::default()
            .with_max_depth(20)
            .with_principal_variation_search(false),
    );
    assert_eq!(pvs.choose_action(&state), plain.choose_action(&state));
    assert_eq!(pvs.root_score(), plain.root_score());
}

#[test]
fn history_heuristic_toggle_does_not_change_the_answer() {
    let state = State { stones: 5, turn: 0 };
    let mut with_history = Negamax::<Subtraction, MaterialBlind>::new_with_options(
        MaterialBlind,
        NegamaxOptions::default()
            .with_max_depth(20)
            .with_history_heuristic(true),
    );
    let mut without_history = Negamax::<Subtraction, MaterialBlind>::new_with_options(
        MaterialBlind,
        NegamaxOptions::default()
            .with_max_depth(20)
            .with_history_heuristic(false),
    );
    assert_eq!(
        with_history.choose_action(&state),
        without_history.choose_action(&state)
    );
    assert_eq!(with_history.root_score(), without_history.root_score());
}

#[test]
fn countermove_heuristic_toggle_does_not_change_the_answer() {
    let state = State { stones: 5, turn: 0 };
    let mut with_countermove = Negamax::<Subtraction, MaterialBlind>::new_with_options(
        MaterialBlind,
        NegamaxOptions::default()
            .with_max_depth(20)
            .with_countermove_heuristic(true),
    );
    let mut without_countermove = Negamax::<Subtraction, MaterialBlind>::new_with_options(
        MaterialBlind,
        NegamaxOptions::default()
            .with_max_depth(20)
            .with_countermove_heuristic(false),
    );
    assert_eq!(
        with_countermove.choose_action(&state),
        without_countermove.choose_action(&state)
    );
    assert_eq!(
        with_countermove.root_score(),
        without_countermove.root_score()
    );
}

#[test]
fn singular_extension_toggle_does_not_change_the_answer() {
    let state = State { stones: 5, turn: 0 };
    let mut with_extension = Negamax::<Subtraction, MaterialBlind>::new_with_options(
        MaterialBlind,
        NegamaxOptions::default()
            .with_max_depth(20)
            .with_singular_extension(true),
    );
    let mut without_extension = Negamax::<Subtraction, MaterialBlind>::new_with_options(
        MaterialBlind,
        NegamaxOptions::default()
            .with_max_depth(20)
            .with_singular_extension(false),
    );
    assert_eq!(
        with_extension.choose_action(&state),
        without_extension.choose_action(&state)
    );
    assert_eq!(with_extension.root_score(), without_extension.root_score());
}

#[test]
fn root_split_parallel_search_agrees_with_the_serial_answer() {
    // See AGENTS.md's "Rust tests" section: any test spawning its own
    // thread pool must take this guard so cargo's default per-binary test
    // concurrency can't overlap two thread-spawning tests' worker bursts.
    let _guard = crate::strategies::parallel_test_guard();

    let state = State { stones: 5, turn: 0 };
    let expected = solver().choose_action(&state);
    let mut parallel = Negamax::<Subtraction, MaterialBlind>::new_with_options(
        MaterialBlind,
        NegamaxOptions::default()
            .with_max_depth(20)
            .with_num_threads(4),
    );
    assert_eq!(
        parallel.choose_action(&state),
        expected,
        "root-split parallel search should find the same optimal move as serial search"
    );
    assert!(
        parallel.root_score() >= WIN_SCORE - parallel.depth_reached() as i32,
        "the winning move should still be reported as (mate-distance) proven under \
         root splitting, got {}",
        parallel.root_score()
    );
}

#[test]
fn aspiration_window_does_not_change_the_answer() {
    let state = State { stones: 5, turn: 0 };
    let mut baseline = solver();
    let mut with_aspiration = Negamax::<Subtraction, MaterialBlind>::new_with_options(
        MaterialBlind,
        NegamaxOptions::default()
            .with_max_depth(20)
            .with_aspiration_window(50_000),
    );
    assert_eq!(
        baseline.choose_action(&state),
        with_aspiration.choose_action(&state)
    );
}
