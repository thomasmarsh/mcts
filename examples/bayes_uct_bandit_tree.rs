// Reproduces Tesauro/Rajan/Segal 2010's own toy testbed (section 4, figure
// 1(a)): a 2-ply MAX-root/MIN-children minimax tree whose leaves are
// independent bandit arms with i.i.d. uniform random win-probabilities
// `p_{j,k}` -- exactly the domain their empirical Bayes-UCT-over-UCT
// advantage was measured on, and exactly the domain their own paper says
// Bayes-UCT needs (independent, uncorrelated leaves; a prior that matches
// the true generating distribution) to show an advantage.
//
// `examples/strength_bayes_uct.rs` instead plays real 7x7 Gonnect games and
// found Bayes-UCT2 losing every game to classic UCT -- expected per the
// paper's own admission that correlated-sibling domains like Go don't show
// the same advantage without a correlation model this implementation
// doesn't have (see that file's doc comment). This script exists to answer
// a narrower question: is that a real limitation of Bayes-UCT on
// correlated domains, or a bug in this crate's `select::BayesUct1`/
// `BayesUct2` + `backprop::BayesGaussian` implementation? Reproducing the
// paper's own controlled domain and its "greedy decision error" metric
// (figure 4(a): does the algorithm's top-level choice match the true-best
// root move, as a function of simulation budget) isolates the algorithm
// from every real-game confound (sibling correlation, unknown reward
// scale, mismatched exploration constants).
//
// Usage: cargo run --release --example bayes_uct_bandit_tree
use std::cell::RefCell;

use mcts::algorithms::mcts::{node::QInit, profile::Mcts, select, SearchConfig, TreeSearch};
use mcts::algorithms::Search;
use mcts::game::{Game, PlayerIndex};

const WIDTH: usize = 5;

thread_local! {
    static PROBS: RefCell<[[f64; WIDTH]; WIDTH]> = const { RefCell::new([[0.0; WIDTH]; WIDTH]) };
}

fn set_tree(probs: [[f64; WIDTH]; WIDTH]) {
    PROBS.with(|t| *t.borrow_mut() = probs);
}

/// True minimax value of root action `j`: MIN picks `k` to minimize MAX's
/// win probability, in the infinite-sample limit (i.e. reads `p_{j,k}`
/// directly, not an estimate).
fn true_value(probs: &[[f64; WIDTH]; WIDTH], j: usize) -> f64 {
    probs[j].iter().cloned().fold(f64::INFINITY, f64::min)
}

fn true_best(probs: &[[f64; WIDTH]; WIDTH]) -> (usize, f64) {
    (0..WIDTH)
        .map(|j| (j, true_value(probs, j)))
        .fold(
            (0, f64::NEG_INFINITY),
            |acc, x| if x.1 > acc.1 { x } else { acc },
        )
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
            _ => unreachable!("apply called on terminal bandit-tree state"),
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

    /// The one intentionally-stochastic part of this synthetic game: each
    /// visit to leaf `(j, k)` draws a fresh Bernoulli(`p_{j,k}`) outcome,
    /// exactly matching the paper's "leaf nodes are ordinary bandit arms"
    /// setup (a real arm keeps paying out fresh independent draws, not one
    /// fixed value it converges to memorizing).
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

fn ucb1(
    iterations: usize,
) -> TreeSearch<BanditTree, Mcts<select::Ucb1, mcts::algorithms::mcts::simulate::Uniform>> {
    TreeSearch::new().config(
        SearchConfig::new()
            .name("ucb1")
            .max_iterations(iterations)
            .q_init(QInit::Infinity)
            .select(select::Ucb1::with_c(1.0)),
    )
}

fn bayes_uct1(
    iterations: usize,
) -> TreeSearch<
    BanditTree,
    Mcts<
        select::BayesUct1,
        mcts::algorithms::mcts::simulate::Uniform,
        mcts::algorithms::mcts::backprop::BayesGaussian,
    >,
> {
    TreeSearch::new().config(
        SearchConfig::new()
            .name("bayes_uct1")
            .max_iterations(iterations)
            .q_init(QInit::Infinity)
            .select(select::BayesUct1::with_c(1.0))
            .backprop(mcts::algorithms::mcts::backprop::BayesGaussian::default()),
    )
}

fn bayes_uct2(
    iterations: usize,
) -> TreeSearch<
    BanditTree,
    Mcts<
        select::BayesUct2,
        mcts::algorithms::mcts::simulate::Uniform,
        mcts::algorithms::mcts::backprop::BayesGaussian,
    >,
> {
    TreeSearch::new().config(
        SearchConfig::new()
            .name("bayes_uct2")
            .max_iterations(iterations)
            .q_init(QInit::Infinity)
            .select(select::BayesUct2::with_c(1.0))
            .backprop(mcts::algorithms::mcts::backprop::BayesGaussian::default()),
    )
}

/// Average root-decision regret over `trials` freshly-drawn random trees,
/// for one `choose_action` call per tree at the given search's fixed
/// iteration budget -- matches the paper's figure 4(a) methodology
/// ("recompute ... note the absolute difference between mean and true
/// value") but applied to the realized top-level decision rather than the
/// node's internal value estimate, since that's what a player experiences.
fn mean_regret<S: Search<G = BanditTree>>(
    mut make_search: impl FnMut() -> S,
    trials: usize,
) -> (f64, f64) {
    let mut regrets = Vec::with_capacity(trials);
    for _ in 0..trials {
        let mut probs = [[0.0; WIDTH]; WIDTH];
        for row in probs.iter_mut() {
            for cell in row.iter_mut() {
                *cell = rand::random::<f64>();
            }
        }
        set_tree(probs);

        let (_best_j, best_val) = true_best(&probs);
        let mut search = make_search();
        let chosen = search.choose_action(&BanditState::default());
        let chosen_val = true_value(&probs, chosen as usize);
        regrets.push(best_val - chosen_val);
    }
    let mean = regrets.iter().sum::<f64>() / trials as f64;
    let var = regrets.iter().map(|r| (r - mean).powi(2)).sum::<f64>() / trials as f64;
    let stderr = (var / trials as f64).sqrt();
    (mean, stderr)
}

fn main() {
    println!("=== Bayes-UCT vs UCT: paper's own bandit-tree domain (5x5, 2-ply) ===");
    println!("Metric: mean root-decision regret (0 = always picks the true-best root move)");
    println!("Lower is better. Both algorithms use c=1.0 (UCB1's plain form == Bayes-UCT1's form)");
    println!();

    let trials = 400;
    let budgets = [20usize, 50, 100, 200, 500, 1000, 2000];

    println!(
        "{:>10} | {:>18} | {:>18} | {:>18}",
        "iters", "ucb1", "bayes_uct1", "bayes_uct2"
    );
    println!("{:-<10}-+-{:-<18}-+-{:-<18}-+-{:-<18}", "", "", "", "");
    for &n in &budgets {
        let (u_mean, u_se) = mean_regret(|| ucb1(n), trials);
        let (b1_mean, b1_se) = mean_regret(|| bayes_uct1(n), trials);
        let (b2_mean, b2_se) = mean_regret(|| bayes_uct2(n), trials);
        println!(
            "{:>10} | {:>7.4} +/- {:<6.4} | {:>7.4} +/- {:<6.4} | {:>7.4} +/- {:<6.4}",
            n, u_mean, u_se, b1_mean, b1_se, b2_mean, b2_se
        );
    }

    println!();
    println!("Interpretation: the paper reports Bayes-UCT1 initially worse than UCT but");
    println!("crossing over to a clear win by ~50 trials, with Bayes-UCT2 beating both from");
    println!("the start. If this table shows the same qualitative pattern, the implementation");
    println!("is doing what the paper says on the domain the paper says it should work on --");
    println!("the real-game (Gonnect) loss is then a domain-correlation limitation, not a bug.");
}
