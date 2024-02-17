#![allow(unused)]
use std::time::Duration;

use ::mcts::strategies::mcts::simulate::EpsilonGreedy;
use ::mcts::strategies::mcts::TreeSearch;
use ::mcts::util::Verbosity;
use mcts::game::Game;
use mcts::games::druid::{Druid, Player, State};
use mcts::strategies::flat_mc::FlatMonteCarloStrategy;
use mcts::strategies::mcts::util;
use mcts::strategies::{self, Search};
use mcts::util::{round_robin_multiple, AnySearch};

pub fn play() {
    // }-> Option<Player> {
    const PLAYOUT_DEPTH: usize = 200;
    const C_LOW: f64 = 0.1;
    const C_STD: f64 = 1.414;
    const MAX_ITER: usize = 10000; //usize::MAX; // 10000;
    const BIAS: f64 = 700.0;
    const EXPAND_THRESHOLD: u32 = 5;
    const VERBOSE: bool = false;
    const RAVE_THRESHOLD: u32 = 400;
    const MAX_TIME_SECS: u64 = 0; // 0 = infinite

    assert_eq!(Duration::default(), Duration::from_secs(0));

    let grave = {
        use mcts::strategies::mcts;

        type HP = mcts::util::McGrave;
        type TS = mcts::TreeSearch<Druid, HP>;
        /*
        let mut params = Vec::new();
        for t in [20, 30, 40] {
            for bias in [5., 10., 20.] {
                let config = {
                    let mut x: mcts::MctsStrategy<Druid, HP> = mcts::MctsStrategy::default()
                        .max_iterations(MAX_ITER)
                        .max_playout_depth(PLAYOUT_DEPTH)
                        .max_time(Duration::from_secs(MAX_TIME_SECS))
                        .playouts_before_expanding(EXPAND_THRESHOLD);
                    x.select.threshold = t;
                    x.select.bias = bias;
                    x
                };
                params.push(config);
            }
        }

        let mut proto: TS = Default::default();
        let mut strategies: Vec<AnySearch<'_, Druid>> = Vec::new();
        for param in params {
            let mut ts = TS::default();
            let name = format!("grave({},{})", param.select.threshold, param.select.bias);
            ts.strategy = param;
            ts.set_friendly_name(name.as_str());
            strategies.push(AnySearch::new(ts));
        }

        round_robin_multiple::<Druid, TS>(&mut strategies, 10, &State::new());
        */

        let strategy = {
            let mut x: mcts::MctsStrategy<Druid, HP> = mcts::MctsStrategy::default()
                .max_iterations(MAX_ITER)
                .max_playout_depth(PLAYOUT_DEPTH)
                .max_time(Duration::from_secs(MAX_TIME_SECS))
                .playouts_before_expanding(EXPAND_THRESHOLD);
            x.select.threshold = 40;
            x.select.bias = 5.;
            x
        };

        TS {
            strategy,
            verbose: VERBOSE,
            ..Default::default()
        }
    };

    let mut brave = {
        use mcts::strategies::mcts;

        let mut x: mcts::TreeSearch<Druid, mcts::util::McBrave> = Default::default();
        x.strategy.max_iterations = MAX_ITER;
        x.strategy.max_playout_depth = PLAYOUT_DEPTH;
        x.strategy.max_time = Duration::from_secs(MAX_TIME_SECS);
        x.strategy.playouts_before_expanding = EXPAND_THRESHOLD;
        x.strategy.select.bias = BIAS;
        x.verbose = VERBOSE;
        x
    };

    let mut ucb1_grave = {
        use mcts::strategies::mcts;

        let mut x: mcts::TreeSearch<Druid, mcts::util::Ucb1Grave> = Default::default();
        x.strategy.max_iterations = MAX_ITER;
        x.strategy.max_playout_depth = PLAYOUT_DEPTH;
        x.strategy.max_time = Duration::from_secs(MAX_TIME_SECS);
        x.strategy.playouts_before_expanding = EXPAND_THRESHOLD;
        x.strategy.select.bias = BIAS;
        x.strategy.select.exploration_constant = C_LOW;
        x.verbose = VERBOSE;
        x
    };

    let mut ucb1_grave_mast = {
        use mcts::strategies::mcts;

        // Configuration(values={
        //   'bias': 266.8785210698843,
        //   'c': 1.86169408634305,
        //   'epsilon': 0.10750788170844316,
        //   'threshold': 211,
        // })

        let mut x: mcts::TreeSearch<Druid, mcts::util::Ucb1GraveMast> = Default::default();
        x.strategy.max_iterations = MAX_ITER;
        x.strategy.max_playout_depth = PLAYOUT_DEPTH;
        x.strategy.max_time = Duration::from_secs(MAX_TIME_SECS);
        x.strategy.playouts_before_expanding = EXPAND_THRESHOLD;
        x.strategy.select.bias = 266.8785210698843;
        x.strategy.select.exploration_constant = 1.86169408634305;
        x.strategy.simulate.epsilon = 0.10750788170844316;
        x.strategy.select.threshold = 211;
        x.verbose = VERBOSE;
        x
    };

    let mut amaf = {
        use mcts::strategies::mcts;

        // let mut x: mcts2::mcts::TreeSearch<Druid, mcts2::util::Ucb1Grave> = Default::default();
        let mut x: mcts::TreeSearch<Druid, mcts::util::ScalarAmaf> = Default::default();
        x.strategy.max_iterations = MAX_ITER;
        x.strategy.max_playout_depth = PLAYOUT_DEPTH;
        x.strategy.max_time = Duration::from_secs(MAX_TIME_SECS);
        x.strategy.playouts_before_expanding = EXPAND_THRESHOLD;
        x.strategy.select.bias = BIAS;
        x.strategy.select.exploration_constant = C_LOW;
        x.verbose = VERBOSE;
        x
    };

    let mut amaf_mast = {
        use mcts::strategies::mcts;

        // let mut x: mcts2::mcts::TreeSearch<Druid, mcts2::util::Ucb1Grave> = Default::default();
        let mut x: mcts::TreeSearch<Druid, mcts::util::ScalarAmafMast> = Default::default();
        x.strategy.max_iterations = MAX_ITER;
        x.strategy.max_playout_depth = PLAYOUT_DEPTH;
        x.strategy.max_time = Duration::from_secs(MAX_TIME_SECS);
        x.strategy.playouts_before_expanding = EXPAND_THRESHOLD;
        x.strategy.select.bias = BIAS;
        x.strategy.select.exploration_constant = C_LOW;
        x.strategy.simulate.epsilon = 0.1;
        x.verbose = VERBOSE;
        x
    };

    let mut amaf_mast_high = {
        use mcts::strategies::mcts;

        // let mut x: mcts2::mcts::TreeSearch<Druid, mcts2::util::Ucb1Grave> = Default::default();
        let mut x: mcts::TreeSearch<Druid, mcts::util::ScalarAmafMast> = Default::default();

        x.strategy.max_iterations = MAX_ITER;
        x.strategy.max_playout_depth = PLAYOUT_DEPTH;
        x.strategy.max_time = Duration::from_secs(MAX_TIME_SECS);
        x.strategy.playouts_before_expanding = EXPAND_THRESHOLD;
        x.strategy.select.bias = BIAS;
        x.strategy.select.exploration_constant = C_LOW;
        x.strategy.simulate.epsilon = 0.9;
        x.verbose = VERBOSE;
        x
    };

    let mut uct = {
        use mcts::strategies::mcts;

        if false {
            type HP = mcts::util::Ucb1;
            type TS = mcts::TreeSearch<Druid, HP>;
            let mut params = Vec::new();
            for c in [1.5, 1.625, 1.75, 1.885, 2., 2.125, 2.25] {
                let config = {
                    let mut x: mcts::MctsStrategy<Druid, HP> = mcts::MctsStrategy::default()
                        .max_iterations(MAX_ITER)
                        .max_playout_depth(PLAYOUT_DEPTH)
                        .max_time(Duration::from_secs(MAX_TIME_SECS))
                        .playouts_before_expanding(EXPAND_THRESHOLD);
                    x.select.exploration_constant = c;
                    x
                };
                params.push(config);
            }

            let mut proto: TS = Default::default();
            let mut strategies: Vec<AnySearch<'_, Druid>> = Vec::new();
            for param in params {
                let mut ts = TS::default();
                let name = format!("ucb[{}]", param.select.exploration_constant);
                ts.strategy = param;
                ts.set_friendly_name(name.as_str());
                strategies.push(AnySearch::new(ts));
            }

            round_robin_multiple::<Druid, TS>(
                &mut strategies,
                10,
                &State::new(),
                Verbosity::Verbose,
            );
        }
        let mut x: mcts::TreeSearch<Druid, mcts::util::Ucb1> = Default::default();
        x.strategy.max_iterations = MAX_ITER;
        x.strategy.max_playout_depth = PLAYOUT_DEPTH;
        x.strategy.max_time = Duration::from_secs(MAX_TIME_SECS);
        x.strategy.playouts_before_expanding = EXPAND_THRESHOLD;
        x.strategy.select.exploration_constant = 1.625;
        x.verbose = VERBOSE;
        x
    };

    let mut uct_mast_low = {
        use mcts::strategies::mcts;

        let mut x: mcts::TreeSearch<Druid, mcts::util::Ucb1Mast> = Default::default();

        x.strategy.max_iterations = MAX_ITER;
        x.strategy.max_playout_depth = PLAYOUT_DEPTH;
        x.strategy.max_time = Duration::from_secs(MAX_TIME_SECS);
        x.strategy.playouts_before_expanding = EXPAND_THRESHOLD;
        x.strategy.select.exploration_constant = 1.625;
        x.strategy.simulate.epsilon = 0.1;
        x.verbose = VERBOSE;
        x
    };

    let mut uct_mast_high = {
        use mcts::strategies::mcts;

        let mut x: mcts::TreeSearch<Druid, mcts::util::Ucb1Mast> = Default::default();

        x.strategy.max_iterations = MAX_ITER;
        x.strategy.max_playout_depth = PLAYOUT_DEPTH;
        x.strategy.max_time = Duration::from_secs(MAX_TIME_SECS);
        x.strategy.playouts_before_expanding = EXPAND_THRESHOLD;
        x.strategy.select.exploration_constant = C_STD;
        x.strategy.simulate.epsilon = 0.9;
        x.verbose = VERBOSE;
        x
    };
    let mut tuned = {
        use mcts::strategies::mcts;

        let mut x: mcts::TreeSearch<Druid, mcts::util::Ucb1Tuned> = Default::default();

        x.strategy.max_iterations = MAX_ITER;
        x.strategy.max_playout_depth = PLAYOUT_DEPTH;
        x.strategy.max_time = Duration::from_secs(MAX_TIME_SECS);
        x.strategy.playouts_before_expanding = EXPAND_THRESHOLD;
        x.strategy.select.exploration_constant = C_LOW;
        x.verbose = VERBOSE;
        x
    };

    // let mut flat = AgentFlat::new().set_samples_per_move(1000);

    // let mut w = tuned; // uct_new;
    // let mut b = amaf;

    let mut strategies: Vec<AnySearch<'_, Druid>> = vec![
        AnySearch::new(amaf),
        AnySearch::new(amaf_mast),
        AnySearch::new(tuned),
        AnySearch::new(uct),
        AnySearch::new(uct_mast_high),
        AnySearch::new(grave),
        AnySearch::new(ucb1_grave_mast),
    ];

    // Convert the vector of trait objects into a vector of mutable references

    round_robin_multiple::<Druid, AnySearch<'_, Druid>>(
        &mut strategies,
        10,
        &State::new(),
        Verbosity::Verbose,
    );
}

/*
    let mut b = tuned;
    let mut w = uct_mast_high;

    let mut state = State::new();
    while !Druid::is_terminal(&state) {
        if VERBOSE {
            println!("{}", state);
        }
        let mut agent: &mut dyn Search<G = Druid> = match state.player {
            Player::Black => &mut b,
            Player::White => &mut w,
        };

        let m = agent.choose_action(&state);
        if VERBOSE {
            println!(
                "move: {} {:?} {}",
                agent.friendly_name(),
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
    */

fn main() {
    play();
}
