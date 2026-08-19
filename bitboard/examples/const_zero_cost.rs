// `Board<S, Const<N>, Const<M>>` is meant to compile down to the same code
// as a raw const-generic bit-twiddling struct, so games migrating off a
// hand-written bitboard type see no regression. This times an identical
// set/get/count_ones workload through `Board` and through a hand-written
// struct with `N`/`M` hardcoded as consts (no `Dim`/`Storage` indirection)
// and compares. If `Board` is meaningfully slower here, `Const<N>` isn't
// optimizing away and callers of `Const<N>` dims need a specialized
// inherent impl overriding the hot methods instead.
//
// Usage: cargo run --release --example const_zero_cost -p bitboard
use std::hint::black_box;
use std::time::Instant;

use bitboard::{Board, Const};

/// Same 8x8 row-major single-`u64` layout as `Board<u64, Const<8>, Const<8>>`,
/// written by hand with `N`/`M` as literal consts -- the baseline `Board`
/// is being compared against.
#[derive(Clone, Copy, Debug, Default)]
struct RawBoard8(u64);

impl RawBoard8 {
    const M: usize = 8;

    #[inline(always)]
    fn get(&self, row: usize, col: usize) -> bool {
        let index = row * Self::M + col;
        (self.0 >> index) & 1 != 0
    }

    #[inline(always)]
    fn set(&mut self, row: usize, col: usize) {
        let index = row * Self::M + col;
        self.0 |= 1u64 << index;
    }

    #[inline(always)]
    fn count_ones(&self) -> u32 {
        self.0.count_ones()
    }
}

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

fn bench_board(iters: usize, coords: &[(usize, usize)]) -> (std::time::Duration, u64) {
    let mut board: Board<u64, Const<8>, Const<8>> = Board::new_const();
    let mut sink = 0u64;
    let t0 = Instant::now();
    for i in 0..iters {
        let (row, col) = coords[i % coords.len()];
        board.set(row, col);
        sink ^= board.get(row, col) as u64;
        sink ^= board.count_ones() as u64;
        board = black_box(board);
    }
    (t0.elapsed(), sink)
}

fn bench_raw(iters: usize, coords: &[(usize, usize)]) -> (std::time::Duration, u64) {
    let mut board = RawBoard8::default();
    let mut sink = 0u64;
    let t0 = Instant::now();
    for i in 0..iters {
        let (row, col) = coords[i % coords.len()];
        board.set(row, col);
        sink ^= board.get(row, col) as u64;
        sink ^= board.count_ones() as u64;
        board = black_box(board);
    }
    (t0.elapsed(), sink)
}

fn main() {
    println!(
        "=== const_zero_cost: Board<u64, Const<8>, Const<8>> vs. a hand-written 8x8 bitboard ==="
    );
    println!();

    let mut rng = Rng(0x9E3779B97F4A7C15);
    let coords: Vec<(usize, usize)> = (0..4096).map(|_| (rng.range(8), rng.range(8))).collect();
    let iters = 20_000_000usize;

    // Warm up both paths once before the timed trials, so the first
    // measured trial isn't paying for icache/branch-predictor warmup that
    // the other path already amortized.
    bench_board(iters / 10, &coords);
    bench_raw(iters / 10, &coords);

    let trials = 5;
    let mut board_total = std::time::Duration::ZERO;
    let mut raw_total = std::time::Duration::ZERO;
    let mut sink = 0u64;
    for _ in 0..trials {
        let (d, s) = bench_board(iters, &coords);
        board_total += d;
        sink ^= s;
        let (d, s) = bench_raw(iters, &coords);
        raw_total += d;
        sink ^= s;
    }
    black_box(sink);

    let board_avg = board_total.as_secs_f64() / trials as f64;
    let raw_avg = raw_total.as_secs_f64() / trials as f64;
    println!(
        "  Board:     {:.4}s avg ({:.2} ns/iter)",
        board_avg,
        board_avg * 1e9 / iters as f64,
    );
    println!(
        "  raw u64:   {:.4}s avg ({:.2} ns/iter)",
        raw_avg,
        raw_avg * 1e9 / iters as f64,
    );
    println!("  ratio (Board / raw): {:.3}x", board_avg / raw_avg);
    println!();
    println!(
        "  {}",
        if board_avg <= raw_avg * 1.05 {
            "PASS: Board is within 5% of the raw baseline -- Const<N> appears zero-cost."
        } else {
            "FAIL: Board is more than 5% slower than the raw baseline -- Const<N> is not \
             optimizing away; Const<N> dims need a specialized inherent impl overriding \
             the hot methods before other code builds on Board."
        }
    );
}
