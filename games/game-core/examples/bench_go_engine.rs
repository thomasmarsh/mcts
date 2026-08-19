// `go_engine.rs`'s `GoEngine` trades a bigger, `Copy`-heavier state
// (union-find group ids + per-group liberty counts) for an O(neighbors)
// legality check instead of `bigbitboard::check_go_move`'s per-candidate
// flood fill. This measures both sides of that trade in isolation --
// `size_of`/clone cost, and legality-check throughput -- at Gonnect/
// AtariGo's three supported board sizes, without wiring the engine into
// either game.
//
// Usage: cargo run --release --example bench_go_engine
use std::time::Instant;

use game_core::bigbitboard::{self, BigBitBoard};
use game_core::go_engine::GoEngine;

/// Minimal xorshift64 PRNG -- avoids adding a `rand` dependency to this
/// crate just for one benchmark example.
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }

    fn range(&mut self, n: usize) -> usize {
        (self.next() % n as u64) as usize
    }
}

/// Builds a random-but-legally-reachable board by playing `moves` random
/// legal placements (skipping illegal candidates), alternating colors.
/// Returns both the resulting `GoEngine` and a plain `(black, white)` pair
/// -- the two representations under comparison.
fn random_legal_board<const N: usize, const WORDS: usize, const CELLS: usize>(
    rng: &mut Rng,
    moves: usize,
) -> (
    GoEngine<N, WORDS, CELLS>,
    BigBitBoard<N, N, WORDS>,
    BigBitBoard<N, N, WORDS>,
) {
    let mut engine = GoEngine::<N, WORDS, CELLS>::new();
    let mut black_to_move = true;
    let mut played = 0;
    let mut attempts = 0;
    while played < moves && attempts < moves * 20 {
        attempts += 1;
        let index = rng.range(N * N);
        if engine.black().get(index) || engine.white().get(index) {
            continue;
        }
        if engine.play(black_to_move, index).is_some() {
            played += 1;
            black_to_move = !black_to_move;
        }
    }
    (engine, engine.black(), engine.white())
}

fn bench_size<const N: usize, const WORDS: usize, const CELLS: usize>(
    label: &str,
    fill_moves: usize,
    probes: usize,
) {
    let mut rng = Rng(0x9E3779B97F4A7C15 ^ (N as u64));
    let (engine, black, white) = random_legal_board::<N, WORDS, CELLS>(&mut rng, fill_moves);

    let empties: Vec<usize> = (0..N * N)
        .filter(|&i| !black.get(i) && !white.get(i))
        .collect();
    if empties.is_empty() {
        println!("=== {label}: board filled, skipping (raise board size or lower fill_moves) ===");
        return;
    }

    // Legality-check throughput: `check_go_move` (flood-based, stateless)
    // vs. `GoEngine::check` (cached liberty counts, O(neighbors)).
    let t0 = Instant::now();
    let mut flood_legal_count = 0usize;
    for _ in 0..probes {
        let index = empties[rng.range(empties.len())];
        let (legal, _) = bigbitboard::check_go_move::<N, WORDS>(black, white, index);
        flood_legal_count += legal as usize;
    }
    let flood_elapsed = t0.elapsed();

    let mut rng2 = Rng(0x9E3779B97F4A7C15 ^ (N as u64));
    let t0 = Instant::now();
    let mut engine_legal_count = 0usize;
    for _ in 0..probes {
        let index = empties[rng2.range(empties.len())];
        let (legal, _) = engine.check(true, index);
        engine_legal_count += legal as usize;
    }
    let engine_elapsed = t0.elapsed();

    // Clone/copy cost: how much bigger `GoEngine` is to move around a
    // search tree (MCTS clones `Game::S` constantly) than a plain
    // black/white `BigBitBoard` pair.
    let clones = 2_000_000usize;
    let t0 = Instant::now();
    let mut engine_sink = engine;
    for _ in 0..clones {
        engine_sink = std::hint::black_box(engine_sink);
    }
    let engine_clone_elapsed = t0.elapsed();

    let t0 = Instant::now();
    let mut pair_sink = (black, white);
    for _ in 0..clones {
        pair_sink = std::hint::black_box(pair_sink);
    }
    let pair_clone_elapsed = t0.elapsed();

    println!("=== {label} ===");
    println!(
        "  size_of: GoEngine = {} bytes, plain black+white pair = {} bytes ({:.1}x)",
        std::mem::size_of::<GoEngine<N, WORDS, CELLS>>(),
        std::mem::size_of::<(BigBitBoard<N, N, WORDS>, BigBitBoard<N, N, WORDS>)>(),
        std::mem::size_of::<GoEngine<N, WORDS, CELLS>>() as f64
            / std::mem::size_of::<(BigBitBoard<N, N, WORDS>, BigBitBoard<N, N, WORDS>)>() as f64,
    );
    println!(
        "  legality check ({probes} probes on a {}-stone board, {} empty cells):",
        fill_moves,
        empties.len(),
    );
    println!(
        "    check_go_move (flood): {:.3}s -> {:.1} checks/sec ({flood_legal_count} legal)",
        flood_elapsed.as_secs_f64(),
        probes as f64 / flood_elapsed.as_secs_f64(),
    );
    println!(
        "    GoEngine::check:       {:.3}s -> {:.1} checks/sec ({engine_legal_count} legal), {:.2}x",
        engine_elapsed.as_secs_f64(),
        probes as f64 / engine_elapsed.as_secs_f64(),
        flood_elapsed.as_secs_f64() / engine_elapsed.as_secs_f64(),
    );
    println!(
        "  copy cost ({clones} copies): GoEngine {:.3}s vs. plain pair {:.3}s ({:.2}x)",
        engine_clone_elapsed.as_secs_f64(),
        pair_clone_elapsed.as_secs_f64(),
        engine_clone_elapsed.as_secs_f64() / pair_clone_elapsed.as_secs_f64(),
    );
    println!();
}

fn main() {
    println!("=== bench_go_engine: GoEngine vs. check_go_move state-size/perf trade ===");
    println!();

    // Fill moves chosen well below N*N so `random_legal_board` reliably
    // reaches the target without exhausting empty cells; capture attrition
    // means far fewer than `fill_moves` stones typically remain.
    bench_size::<9, 2, 81>("9x9", 40, 2_000_000);
    bench_size::<13, 3, 169>("13x13", 90, 2_000_000);
    bench_size::<19, 6, 361>("19x19", 180, 2_000_000);
}
