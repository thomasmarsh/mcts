//! How damaging are epsilon-greedy exploratory moves in a capture game? For a
//! trained 5x5 model, play self-play games at fixed epsilons and count, per move
//! kind (random vs greedy), how often the move hands the opponent an immediate
//! winning reply (a one-move blunder). No training happens here.
//!
//! ```text
//! LIBRARY_PATH=/opt/homebrew/lib cargo run --release --example gonnect_epsilon -p game-gonnect -- \
//!     --model-dir local/output/gonnect/td5-a-s1 [--games 2000] [--epsilons 0.0,0.1,0.2] [--out FILE.jsonl]
//! ```

use std::io::Write;
use std::path::Path;

use game_gonnect::sized::{SizedGonnect, SizedState};
use game_gonnect::td_cells::GonnectCells;
use mcts::game::{Game, PlayerIndex};
use ntuple::{terminal_value, CellFeatures, Model};
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};

type G = SizedGonnect<5>;

fn opponent_has_immediate_win(state: &SizedState<5>) -> bool {
    if G::is_terminal(state) {
        return false;
    }
    let mover = G::player_to_move(state).to_index();
    let mut actions = Vec::new();
    G::generate_actions(state, &mut actions);
    actions.iter().any(|a| {
        let next = G::apply(state.clone(), a);
        G::is_terminal(&next) && G::winner(&next).map(|p| p.to_index()) == Some(mover)
    })
}

fn main() {
    let (mut dir, mut games, mut eps_list, mut out) =
        (String::new(), 2000usize, vec![0.0f32, 0.1, 0.2], None::<String>);
    let mut it = std::env::args().skip(1);
    while let Some(arg) = it.next() {
        let mut val = || it.next().unwrap_or_else(|| panic!("{arg} needs a value"));
        match arg.as_str() {
            "--model-dir" => dir = val(),
            "--games" => games = val().parse().unwrap(),
            "--epsilons" => eps_list = val().split(',').map(|x| x.parse().unwrap()).collect(),
            "--out" => out = Some(val()),
            other => panic!("unknown argument {other}"),
        }
    }
    assert!(!dir.is_empty(), "--model-dir is required");
    let feats = GonnectCells::<5>;
    let model = Model::load(Path::new(&dir), &feats.orientations());
    let spc = model.geometry().states_per_cell();
    let mut codes = vec![0u8; 25];

    for &eps in &eps_list {
        let mut rng = SmallRng::seed_from_u64(0xe95 + (eps * 1000.0) as u64);
        // [random, greedy] x (moves, blunders)
        let (mut moves, mut blunders) = ([0u64; 2], [0u64; 2]);
        let mut plies = 0u64;
        for _ in 0..games {
            let mut s = SizedState::<5>::default();
            while !G::is_terminal(&s) {
                let mut actions = Vec::new();
                G::generate_actions(&s, &mut actions);
                let actor = G::player_to_move(&s).to_index();
                let random = actions.len() > 1 && rng.gen::<f32>() < eps;
                let k = if random {
                    rng.gen_range(0..actions.len())
                } else {
                    let vals: Vec<f32> = actions
                        .iter()
                        .map(|a| {
                            let next = G::apply(s.clone(), a);
                            if G::is_terminal(&next) {
                                terminal_value::<G>(&next, actor)
                            } else {
                                feats.cell_codes(&next, spc, &mut codes);
                                let v = model.value(&codes);
                                if G::player_to_move(&next).to_index() == actor { v } else { -v }
                            }
                        })
                        .collect();
                    let max = vals.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                    let ties: Vec<usize> = (0..vals.len()).filter(|&i| vals[i] == max).collect();
                    ties[rng.gen_range(0..ties.len())]
                };
                s = G::apply(s, &actions[k]);
                plies += 1;
                if actions.len() > 1 {
                    let kind = usize::from(!random);
                    moves[kind] += 1;
                    blunders[kind] += u64::from(opponent_has_immediate_win(&s));
                }
            }
        }
        let rate = |i: usize| blunders[i] as f64 / moves[i].max(1) as f64;
        println!(
            "eps {eps:.2}: {games} games, {:.1} plies/game; random moves {} blunder rate {:.3}; \
             greedy moves {} blunder rate {:.3}",
            plies as f64 / games as f64,
            moves[0],
            rate(0),
            moves[1],
            rate(1)
        );
        if let Some(path) = &out {
            let mut f = std::fs::OpenOptions::new().create(true).append(true).open(path).unwrap();
            writeln!(
                f,
                "{}",
                serde_json::json!({
                    "type": "epsilon_damage", "model_dir": dir, "epsilon": eps, "games": games,
                    "plies_per_game": plies as f64 / games as f64,
                    "random_moves": moves[0], "random_blunders": blunders[0],
                    "greedy_moves": moves[1], "greedy_blunders": blunders[1],
                })
            )
            .unwrap();
        }
    }
}
