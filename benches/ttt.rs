use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};

use mcts::games::ttt;
use mcts::strategies::mcts::util;
use mcts::strategies::mcts::SearchConfig;
use mcts::strategies::mcts::TreeSearch;
use mcts::strategies::Search;

type TicTacToeTS = TreeSearch<ttt::TicTacToe, util::Ucb1>;

fn ponder(c: &mut Criterion) {
    let mut group = c.benchmark_group("ttt");
    for n in [250, 500, 750, 1000] {
        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, &n| {
            b.iter(|| {
                let mut ts =
                    TicTacToeTS::default().config(SearchConfig::default().max_iterations(n));

                ts.choose_action(&ttt::HashedPosition::new());
            });
        });
    }

    group.finish();
}

criterion_group!(benches, ponder);
criterion_main!(benches);
