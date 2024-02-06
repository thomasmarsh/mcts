use mcts::game::Game;
use mcts::games::druid::{Druid, Player, State};
use mcts::strategies::flat_mc::FlatMonteCarloStrategy;
use mcts::strategies::mcts::config::SelectionStrategy;
use mcts::strategies::mcts::TreeSearch;

type AgentMCTS = TreeSearch<Druid>;
type AgentFlat = FlatMonteCarloStrategy<Druid>;

pub fn play() -> Option<Player> {
    let mut high_rave = {
        let mut x = AgentMCTS::new();
        x.config.verbose = false;
        x.set_max_rollouts(80000);
        x.config.tree_selection_strategy = SelectionStrategy::UCT(0.1, 8000.);
        x.config.use_mast = false;
        x
    };

    let mut low_c = {
        let mut x = AgentMCTS::new();
        x.config.verbose = false;
        x.set_max_rollouts(20000);
        x.config.tree_selection_strategy = SelectionStrategy::UCT(0.1, 6000.);
        x.config.use_mast = false;
        x
    };

    let mut flat = AgentFlat::new().set_samples_per_move(1000);

    let mut w = high_rave;
    let mut b = low_c;

    println!("=========================");
    let mut state = State::new();
    while !Druid::is_terminal(&state) {
        println!("{}", state);
        let m = match state.player {
            Player::Black => b.choose_move(&state),
            Player::White => w.choose_move(&state),
        }
        .unwrap();
        println!(
            "move: {:?} {}",
            Druid::player_to_move(&state),
            Druid::notation(&state, &m)
        );
        state.apply(m);
    }

    println!("{}", state);
    println!("winner: {:?}", state.connection());
    state.connection()
}

fn main() {
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

        println!("b={} ({:.2}%) w={} ({:.2}%) draw={}", b, pct_b, w, pct_w, x);
    }
}
