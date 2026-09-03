// Scratch diagnostic (not meant to be kept): dumps RootReport for a single
// fixed bandit tree at a large iteration budget, comparing classic UCT vs
// Bayes-UCT2's visit distribution and empirical mean_value against the true
// per-arm minimax values -- to see *how* Bayes-UCT2 is going wrong, not
// just that its regret is high.
use std::cell::RefCell;

use mcts::game::{Game, PlayerIndex};
use mcts::algorithms::mcts::{node::QInit, select, strategy::Compose, SearchConfig, TreeSearch};
use mcts::algorithms::Search;

const WIDTH: usize = 5;

thread_local! {
    static PROBS: RefCell<[[f64; WIDTH]; WIDTH]> = const { RefCell::new([[0.0; WIDTH]; WIDTH]) };
}

fn set_tree(probs: [[f64; WIDTH]; WIDTH]) {
    PROBS.with(|t| *t.borrow_mut() = probs);
}

fn true_value(probs: &[[f64; WIDTH]; WIDTH], j: usize) -> f64 {
    probs[j].iter().cloned().fold(f64::INFINITY, f64::min)
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
struct BanditState {
    depth: u8,
    j: u8,
    k: u8,
}

impl std::fmt::Display for BanditState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "depth={} j={} k={}", self.depth, self.j, self.k)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum P {
    Max,
    Min,
}

impl PlayerIndex for P {
    fn to_index(&self) -> usize {
        match self {
            P::Max => 0,
            P::Min => 1,
        }
    }
}

#[derive(Clone)]
struct BanditTree;

impl Game for BanditTree {
    type S = BanditState;
    type A = u8;
    type P = P;

    fn apply(state: BanditState, action: &u8) -> BanditState {
        match state.depth {
            0 => BanditState {
                depth: 1,
                j: *action,
                k: 0,
            },
            1 => BanditState {
                depth: 2,
                j: state.j,
                k: *action,
            },
            _ => unreachable!(),
        }
    }

    fn generate_actions(state: &BanditState, actions: &mut Vec<u8>) {
        if state.depth < 2 {
            actions.extend(0..WIDTH as u8);
        }
    }

    fn is_terminal(state: &BanditState) -> bool {
        state.depth == 2
    }

    fn player_to_move(state: &BanditState) -> P {
        if state.depth == 0 {
            P::Max
        } else {
            P::Min
        }
    }

    fn num_players() -> usize {
        2
    }

    fn winner(state: &BanditState) -> Option<P> {
        if state.depth != 2 {
            return None;
        }
        let p = PROBS.with(|t| t.borrow()[state.j as usize][state.k as usize]);
        if rand::random::<f64>() < p {
            Some(P::Max)
        } else {
            Some(P::Min)
        }
    }
}

fn main() {
    let mut probs = [[0.0; WIDTH]; WIDTH];
    for row in probs.iter_mut() {
        for cell in row.iter_mut() {
            *cell = rand::random::<f64>();
        }
    }
    set_tree(probs);

    println!("True p[j][k]:");
    for (j, row) in probs.iter().enumerate() {
        println!(
            "  j={} : {:?}  (true_value={:.4})",
            j,
            row,
            true_value(&probs, j)
        );
    }
    println!();

    let n = 400;

    let mut ucb1 = TreeSearch::<
        BanditTree,
        Compose<select::Ucb1, mcts::algorithms::mcts::simulate::Uniform>,
    >::new()
    .config(
        SearchConfig::new()
            .max_iterations(n)
            .q_init(QInit::Infinity)
            .select(select::Ucb1::with_c(1.0)),
    );
    let chosen = ucb1.choose_action(&BanditState::default());
    let report = ucb1.root_report(&BanditState::default());
    println!(
        "UCB1 chose j={} (true best j={})",
        chosen,
        (0..WIDTH)
            .max_by(|&a, &b| true_value(&probs, a)
                .partial_cmp(&true_value(&probs, b))
                .unwrap())
            .unwrap()
    );
    for a in &report.actions {
        println!(
            "  action={} visits={} mean_value={:.4} true_value={:.4}",
            a.action,
            a.visits,
            a.mean_value,
            true_value(&probs, a.action as usize)
        );
    }
    println!();

    let mut bayes2 = TreeSearch::<
        BanditTree,
        Compose<
            select::BayesUct2,
            mcts::algorithms::mcts::simulate::Uniform,
            mcts::algorithms::mcts::backprop::BayesGaussian,
        >,
    >::new()
    .config(
        SearchConfig::new()
            .max_iterations(n)
            .q_init(QInit::Infinity)
            .select(select::BayesUct2::with_c(1.0))
            .backprop(mcts::algorithms::mcts::backprop::BayesGaussian::default()),
    );
    let chosen2 = bayes2.choose_action(&BanditState::default());
    let report2 = bayes2.root_report(&BanditState::default());
    println!("Bayes-UCT2 chose j={}", chosen2);
    for a in &report2.actions {
        println!(
            "  action={} visits={} mean_value={:.4} true_value={:.4}",
            a.action,
            a.visits,
            a.mean_value,
            true_value(&probs, a.action as usize)
        );
    }
}
