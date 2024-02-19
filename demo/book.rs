use std::sync::Arc;
use std::sync::Mutex;
use std::time::Duration;

use mcts::game::Game;
use mcts::strategies::mcts::meta::OpeningBook;
use mcts::strategies::mcts::meta::QuasiBestFirst;
use mcts::strategies::mcts::select;
use mcts::strategies::mcts::simulate;
use mcts::strategies::mcts::util;
use mcts::strategies::mcts::MctsStrategy;
use mcts::strategies::mcts::TreeSearch;

use mcts::games::druid::Druid;
use mcts::games::druid::Move;
use mcts::games::druid::State;

use rand::rngs::SmallRng;
use rand_core::SeedableRng;

// QBF Config
const NUM_THREADS: usize = 8;
const NUM_GAMES: usize = 18000;

// MCTS Config
const PLAYOUT_DEPTH: usize = 200;
const C_TUNED: f64 = 1.625;
const MAX_ITER: usize = usize::MAX;
const EXPAND_THRESHOLD: u32 = 1;
const VERBOSE: bool = false;
const MAX_TIME_SECS: u64 = 5; // 0 = infinite

pub fn debug(book: &OpeningBook<Move>) {
    println!("book.len() = {}", book.index.len());
    let root = book.index.get(book.root_id);
    let actions = root.children.keys();

    println!("root: {}", root.num_visits);
    actions.enumerate().for_each(|(i, action)| {
        let child_id_opt = root.children.get(action);
        let child = child_id_opt.map(|child_id| book.index.get(*child_id));
        let score = book.score(&[*action], 0);
        println!(
            "- {i}: {:?}, {score:?} {action:?}",
            child.map_or(0, |c| c.num_visits),
        );
    });
}

fn make_mcts() -> TreeSearch<Druid, util::Ucb1Mast> {
    TreeSearch::default()
        .strategy(
            MctsStrategy::default()
                .max_iterations(MAX_ITER)
                .max_playout_depth(PLAYOUT_DEPTH)
                .max_time(Duration::from_secs(MAX_TIME_SECS))
                .playouts_before_expanding(EXPAND_THRESHOLD)
                .select(select::Ucb1 {
                    exploration_constant: C_TUNED,
                })
                .simulate(simulate::EpsilonGreedy::with_epsilon(0.1)),
        )
        .verbose(VERBOSE)
}

fn main() {
    color_backtrace::install();

    let book: Arc<Mutex<OpeningBook<Move>>> =
        Arc::new(Mutex::new(OpeningBook::new(Druid::num_players())));

    std::thread::scope(|scope| {
        for _ in 0..NUM_THREADS {
            scope.spawn(|| {
                let book = Arc::clone(&book);
                for _ in 0..(NUM_GAMES / NUM_THREADS) {
                    let search = make_mcts();
                    let mut qbf: QuasiBestFirst<Druid, util::Ucb1Mast> = QuasiBestFirst::new(
                        book.lock().unwrap().clone(),
                        search,
                        SmallRng::from_entropy(),
                    );

                    let (stack, utilities) = qbf.search(&State::new());

                    let mut book_mut = book.lock().unwrap();
                    book_mut.add(stack.as_slice(), utilities.as_slice());
                    debug(&book_mut);
                }
            });
        }
    });
}
