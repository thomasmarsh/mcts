#![allow(unused)]
use std::time::Duration;

use mcts::game::Game;
use mcts::games::druid::{Druid, Player, State};
use mcts::strategies::flat_mc::FlatMonteCarloStrategy;
use mcts::strategies::mcts::util;
use mcts::strategies::Search;

pub fn play() -> Option<Player> {
    const PLAYOUT_DEPTH: usize = 200;
    const C_LOW: f64 = 0.1;
    const C_STD: f64 = 1.414;
    const MAX_ITER: usize = usize::MAX; // 10000;
    const BIAS: f64 = 700.0;
    const EXPAND_THRESHOLD: u32 = 5;
    const VERBOSE: bool = true;

    let rave_new = {
        use mcts::strategies::mcts;

        // let mut x: mcts2::mcts::TreeSearch<Druid, mcts2::util::Ucb1Grave> = Default::default();
        let mut x: mcts::TreeSearch<Druid, mcts::util::ScalarAmaf> = Default::default();
        x.strategy.max_iterations = MAX_ITER;
        x.strategy.select.bias = BIAS;
        x.strategy.select.exploration_constant = C_LOW;
        x.strategy.max_playout_depth = PLAYOUT_DEPTH;
        x.strategy.playouts_before_expanding = EXPAND_THRESHOLD;
        x.strategy.max_time = Duration::from_secs(3);
        x.verbose = VERBOSE;
        x
    };

    let uct_new = {
        use mcts::strategies::mcts;

        let mut x: mcts::TreeSearch<Druid, mcts::util::Ucb1> = Default::default();

        x.strategy.max_iterations = MAX_ITER;
        x.strategy.select.exploration_constant = C_STD;
        x.strategy.max_playout_depth = PLAYOUT_DEPTH;
        x.strategy.playouts_before_expanding = EXPAND_THRESHOLD;
        x.verbose = VERBOSE;
        x
    };

    let tuned = {
        use mcts::strategies::mcts;

        let mut x: mcts::TreeSearch<Druid, mcts::util::Ucb1Tuned> = Default::default();

        x.strategy.max_iterations = MAX_ITER;
        x.strategy.select.exploration_constant = C_LOW;
        x.strategy.max_playout_depth = PLAYOUT_DEPTH;
        x.strategy.playouts_before_expanding = EXPAND_THRESHOLD;
        x.strategy.max_time = Duration::from_secs(3);
        x.verbose = VERBOSE;
        x
    };

    // let mut flat = AgentFlat::new().set_samples_per_move(1000);

    // let mut w = tuned; // uct_new;
    // let mut b = rave_new;

    let mut w = rave_new;
    let mut b = tuned;

    let mut state = State::new();
    while !Druid::is_terminal(&state) {
        if VERBOSE {
            println!("{}", state);
        }
        let m = match state.player {
            Player::Black => b.choose_action(&state),
            Player::White => w.choose_action(&state),
        };
        if VERBOSE {
            println!(
                "move: {:?} {}",
                Druid::player_to_move(&state),
                Druid::notation(&state, &m)
            );
        }
        state.apply(m);
    }

    if VERBOSE {
        println!("{}", state);
        println!("winner: {:?}", state.connection());
    }
    state.connection()
}

fn main() {
    color_backtrace::install();

    let mut w = 0;
    let mut b = 0;
    let mut x = 0;
    for _ in 0..100 {
        match play() {
            None => x += 1,
            Some(Player::Black) => b += 1,
            Some(Player::White) => w += 1,
        }

        let total = w + b + x;
        let pct_w = w as f32 / total as f32 * 100.;
        let pct_b = b as f32 / total as f32 * 100.;

        println!("==========================================");
        println!(
            "b[old]={} ({:.2}%) w[new]={} ({:.2}%) draw={}",
            b, pct_b, w, pct_w, x
        );
    }
}
