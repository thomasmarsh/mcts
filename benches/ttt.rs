use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};

use mcts::games::ttt;
use mcts::strategies::mcts::TreeSearch;

type TicTacToeTS = TreeSearch<ttt::TicTacToe>;

fn ponder(c: &mut Criterion) {
    let mut group = c.benchmark_group("ttt");
    for n in [250, 500, 750, 1000] {
        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, &n| {
            b.iter(|| {
                let mut ts = TicTacToeTS::new();
                ts.set_max_rollouts(n);
                ts.choose_move(&ttt::HashedPosition::new());
            });
        });
    }

    group.finish();
}

criterion_group!(benches, ponder);
criterion_main!(benches);
