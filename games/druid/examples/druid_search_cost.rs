//! Per-move cost of the "strong" preset at a fixed iteration count, on mid-game positions
//! (`cargo run --release --example druid_search_cost -p game-druid [iterations] [sizes..]`). Sizes the
//! deferred strong-10000 gate: one move at 10,000 iterations, single-threaded and on all cores.

use std::path::PathBuf;
use std::time::Instant;

use game_druid::{Druid, HashedState, Size};
use mcts::game::Game;
use mcts_tune::presets::PresetTable;
use mcts_tune::SearchBudget;
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};

fn main() {
    let mut args = std::env::args().skip(1);
    let iterations: usize = args.next().and_then(|a| a.parse().ok()).unwrap_or(10_000);
    let sizes: Vec<u8> = args.filter_map(|a| a.parse().ok()).collect();
    let sizes = if sizes.is_empty() { vec![5, 7, 9] } else { sizes };
    let path = PathBuf::from(concat!(env!("CARGO_MANIFEST_DIR"), "/presets.json"));
    let table = PresetTable::load_from_path(&path).expect("presets.json must parse");

    for n in sizes {
        let size = Size { w: n, h: n };
        // A mid-game position: a fixed number of random turn-starts in.
        let mut rng = SmallRng::seed_from_u64(3);
        let mut s = HashedState::new(size);
        for _ in 0..(3 * usize::from(n)) {
            let mut acts = Vec::new();
            Druid::generate_actions(&s, &mut acts);
            s = Druid::apply(s, &acts[rng.gen_range(0..acts.len())]);
        }
        for threads in [1usize, 0] {
            let mut times = Vec::new();
            for seed in 0..3u64 {
                let mut search = table
                    .build_with::<Druid>("strong", seed, |b: &mut SearchBudget| {
                        b.threads = threads;
                        b.max_iterations = Some(iterations);
                        b.max_time = None;
                    })
                    .expect("strong preset builds");
                let t = Instant::now();
                let _ = search.choose_action(&s);
                times.push(t.elapsed().as_secs_f64());
            }
            let mean = times.iter().sum::<f64>() / times.len() as f64;
            println!(
                "{n}x{n} strong-{iterations} threads={} : {:.2}s per search step ({:.0} it/s)",
                if threads == 0 { "all".into() } else { threads.to_string() },
                mean,
                iterations as f64 / mean
            );
        }
    }
}
