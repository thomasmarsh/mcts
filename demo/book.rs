use std::sync::Arc;
use std::sync::Mutex;
use std::time::Duration;

use mcts::game::Game;
use mcts::strategies::mcts::book::OpeningBook;
use mcts::strategies::mcts::select;
use mcts::strategies::mcts::simulate;
use mcts::strategies::mcts::strategy;
use mcts::strategies::mcts::SearchConfig;
use mcts::strategies::mcts::TreeSearch;
use mcts::strategies::Search;

use mcts::games::druid::Druid;
use mcts::games::druid::Move;

// QBF Config
const NUM_THREADS: usize = 8;
const NUM_GAMES: usize = 18000;
const EPSILON: f64 = 0.5;

// MCTS Config
const PLAYOUT_DEPTH: usize = 200;
const C_TUNED: f64 = 1.625;
const MAX_ITER: usize = 1000; // usize::MAX;
const EXPAND_THRESHOLD: u32 = 1;
const VERBOSE: bool = false;
const MAX_TIME_SECS: u64 = 0; // infinite

pub fn debug(book: &OpeningBook<Move>) {
    println!("book.len() = {}", book.index.len());
    let root = book.index.get(book.root_id);
    let actions = root.children.keys();

    println!("root: {}", root.num_visits);
    actions.enumerate().for_each(|(i, action)| {
        let child_id_opt = root.children.get(action);
        let child = child_id_opt.map(|child_id| book.index.get(*child_id));
        let score = book.score(&[*action], 0.into());
        println!(
            "- {i}: {:?}, {score:?} {action:?}",
            child.map_or(0, |c| c.num_visits),
        );
    });
}

fn make_mcts() -> TreeSearch<Druid, strategy::Ucb1Mast> {
    TreeSearch::new().config(
        SearchConfig::new()
            .max_iterations(MAX_ITER)
            .max_playout_depth(PLAYOUT_DEPTH)
            .max_time(Duration::from_secs(MAX_TIME_SECS))
            .expand_threshold(EXPAND_THRESHOLD)
            .select(select::Ucb1::with_c(C_TUNED))
            .simulate(simulate::EpsilonGreedy::with_epsilon(0.1))
            .verbose(VERBOSE),
    )
}

fn make_qbf(book: OpeningBook<Move>) -> TreeSearch<Druid, strategy::QuasiBestFirst> {
    TreeSearch::new().config(
        SearchConfig::new().select(
            select::EpsilonGreedy::new()
                .epsilon(EPSILON)
                .inner(select::QuasiBestFirst::new().book(book).search(make_mcts())),
        ),
    )
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
                    let mut ts = make_qbf(book.lock().unwrap().clone());
                    let (key, utilities) = ts.make_book_entry(&<Druid as Game>::S::default());
                    let mut book_mut = book.lock().unwrap();
                    book_mut.add(key.as_slice(), utilities.as_slice());
                    debug(&book_mut);
                }
            });
        }
    });
}
