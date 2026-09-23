//! Architecture-audit prototype: can `crates/mcts`'s ordinary per-node
//! `TreeSearch`, given a batching `Evaluator`/`PolicyLogits` wrapper that
//! coalesces concurrent leaf requests from many independently-running self-play
//! games into one `convnet::mlx::evaluate_batch` GPU call, reach comparable
//! self-play throughput to `crates/mcts-batch`'s flat-array lockstep design?
//!
//! `mcts-batch`'s own self-play benchmark compares its batched engine only
//! against a single-threaded, no-batching per-node baseline
//! (`mlx_selfplay_bench.rs`, one game at a time, one evaluator call per
//! leaf, no concurrency) -- it never tries batching `crates/mcts`'s own
//! evaluator calls. This binary is that missing comparison: unlike
//! `mlx_selfplay_bench.rs`, many games run concurrently
//! (one OS thread per live game, up to `workers`), and every leaf's
//! `Evaluator::evaluate`/`PolicyLogits::logits` call blocks on a shared queue
//! that flushes into one `evaluate_batch` MLX call once enough requests have
//! piled up (or a background poller flushes whatever's pending, bounding
//! tail latency when fewer than `batch_size` games are still live). No
//! `num_tree_threads` involved -- each individual game's own tree search
//! stays single-threaded, exactly like `mcts-batch`'s own per-tree
//! selection/backprop; only the leaf-evaluation seam is batched, and it is
//! batched *across independently-running games*, which is the actual shape
//! `mcts-batch`'s self-play benchmark uses (`games` independent trees), not
//! `num_tree_threads`'s single-tree virtual-loss parallelism (which would
//! only ever produce a useful batch for one game at a time and gains nothing
//! extra here).
//!
//! Run:
//!   cargo run --release -p game-othello --features mlx --example mlx_tree_batch_bench \
//!     [games] [sims] [workers] [batch_size] [poll_us]
//!
//! Defaults are games=128, sims=16, matching `mcts-batch`'s own self-play
//! gate scale. All-zero weights on both sides, same as
//! `mlx_selfplay_bench.rs` -- this measures throughput, not strength.

use std::sync::atomic::{AtomicUsize, Ordering::Relaxed};
use std::sync::mpsc::sync_channel;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use game_othello::convnet::mlx::evaluate_batch;
use game_othello::convnet::CnnValueNet;
use game_othello::{Move, Othello, State};
use mcts::algorithms::mcts::gumbel::{gumbel_search_with_root_value, GumbelConfig};
use mcts::algorithms::mcts::node::QInit;
use mcts::algorithms::mcts::policy::PolicyLogits;
use mcts::algorithms::mcts::profile::Mcts;
use mcts::algorithms::mcts::select::GumbelCompletedQ;
use mcts::algorithms::mcts::simulate::EvaluatedCutoff;
use mcts::algorithms::mcts::{SearchConfig, TreeSearch};
use mcts::evaluator::{Evaluator, Score, EVAL_MAGNITUDE_LIMIT};
use mcts::game::Game;

struct Pending {
    state: State,
    tx: std::sync::mpsc::SyncSender<(f32, Vec<f64>)>,
}

/// Coalesces concurrent leaf-evaluation requests (from many games' own
/// single-threaded tree searches, each running on its own OS thread) into
/// batched `evaluate_batch` GPU calls. Flushes early once `batch_size`
/// requests are queued (the common case while `workers` games are all
/// still live); a background thread also flushes whatever's pending every
/// `poll` interval, so the tail (fewer live games than `batch_size` late
/// in the run) never stalls waiting for a batch that will never fill.
struct Batcher {
    net: CnnValueNet,
    chunk_size: usize,
    batch_size: usize,
    pending: Mutex<Vec<Pending>>,
    flush_lock: Mutex<()>,
    pub batches_run: AtomicUsize,
    pub requests_served: AtomicUsize,
}

impl Batcher {
    fn new(net: CnnValueNet, chunk_size: usize, batch_size: usize) -> Self {
        Batcher {
            net,
            chunk_size,
            batch_size,
            pending: Mutex::new(Vec::new()),
            flush_lock: Mutex::new(()),
            batches_run: AtomicUsize::new(0),
            requests_served: AtomicUsize::new(0),
        }
    }

    fn flush(&self, batch: Vec<Pending>) {
        if batch.is_empty() {
            return;
        }
        let _guard = self.flush_lock.lock().unwrap();
        let states: Vec<State> = batch.iter().map(|p| p.state).collect();
        let (values, policies) = evaluate_batch(&self.net, &states, self.chunk_size);
        self.batches_run.fetch_add(1, Relaxed);
        self.requests_served.fetch_add(batch.len(), Relaxed);
        for (p, (v, pol)) in batch.into_iter().zip(values.into_iter().zip(policies)) {
            let _ = p.tx.send((v, pol.to_vec()));
        }
    }

    fn request(&self, state: State) -> (f32, Vec<f64>) {
        let (tx, rx) = sync_channel(1);
        let mut guard = self.pending.lock().unwrap();
        guard.push(Pending { state, tx });
        let batch = if guard.len() >= self.batch_size {
            Some(std::mem::take(&mut *guard))
        } else {
            None
        };
        drop(guard);
        if let Some(batch) = batch {
            self.flush(batch);
        }
        rx.recv().unwrap()
    }

    fn poll_once(&self) {
        let mut guard = self.pending.lock().unwrap();
        if guard.is_empty() {
            return;
        }
        let batch = std::mem::take(&mut *guard);
        drop(guard);
        self.flush(batch);
    }
}

/// `Option` only to satisfy `EvaluatedCutoff`/`TreeSearch`'s `Default` bound
/// (needed for their builder-style construction); every real instance this
/// binary constructs carries `Some` and the `None` arm is unreachable.
#[derive(Clone, Default)]
struct BatchedNet(Option<Arc<Batcher>>);

impl BatchedNet {
    fn new(batcher: Arc<Batcher>) -> Self {
        BatchedNet(Some(batcher))
    }

    fn batcher(&self) -> &Batcher {
        self.0.as_deref().expect("BatchedNet always constructed with BatchedNet::new")
    }
}

// A leaf expansion (`search/shared.rs::expand`) always calls
// `PolicyLogits::logits` immediately followed by `Evaluator::evaluate` on
// the very same state, from the same OS thread (one game per thread here).
// `evaluate_batch` computes both heads together in one shared-trunk MLX
// call, so without this cache the two calls would each independently
// enqueue a full request (as the two per-state baseline functions
// `value`/`all_policy_logits` also each independently recompute the trunk)
// -- but unlike the baseline, a batcher request always pays for *both*
// heads regardless of which one the caller wanted, so an uncached pair of
// requests here would do roughly 2x the baseline's per-leaf FLOPs for no
// batching benefit. This single-slot, single-thread cache makes the second
// call of the pair free, matching (not exceeding) the baseline's per-leaf
// compute.
thread_local! {
    static LAST: std::cell::RefCell<Option<(State, f32, Vec<f64>)>> = const { std::cell::RefCell::new(None) };
}

fn cached_request(batcher: &Batcher, state: State) -> (f32, Vec<f64>) {
    if let Some((v, p)) = LAST.with(|cell| {
        let mut cell = cell.borrow_mut();
        match cell.take() {
            Some((s, v, p)) if s == state => Some((v, p)),
            other => {
                *cell = other;
                None
            }
        }
    }) {
        return (v, p);
    }
    let (v, p) = batcher.request(state);
    LAST.with(|cell| *cell.borrow_mut() = Some((state, v, p.clone())));
    (v, p)
}

impl Evaluator<Othello> for BatchedNet {
    fn evaluate(&self, state: &State) -> Score {
        let (v, _policy) = cached_request(self.batcher(), *state);
        (v * EVAL_MAGNITUDE_LIMIT as f32).round() as Score
    }
}

impl PolicyLogits<Othello> for BatchedNet {
    fn logits(&mut self, state: &State, actions: &[Move]) -> Vec<f64> {
        let (_v, all) = cached_request(self.batcher(), *state);
        let sum: f64 = all.iter().sum();
        actions
            .iter()
            .map(|a| if *a == Move::PASS { sum / all.len() as f64 } else { all[a.0 as usize] })
            .collect()
    }
}

type Profile = Mcts<GumbelCompletedQ, EvaluatedCutoff<Othello, BatchedNet>>;

fn play_one_game(net: BatchedNet, gcfg: GumbelConfig, seed: u64) -> usize {
    let mut search: TreeSearch<Othello, Profile> = TreeSearch::default().config(
        SearchConfig::default()
            .expand_threshold(1)
            .max_playout_depth(0)
            .q_init(QInit::Loss)
            .select(GumbelCompletedQ::with_config(gcfg))
            .simulate(EvaluatedCutoff::new().evaluator(net.clone()))
            .with_policy_logits(net.clone())
            .seed(seed),
    );
    let mut state = State::default();
    let mut plies = 0;
    while !Othello::is_terminal(&state) {
        let root_value = net.evaluate(&state) as f64 / EVAL_MAGNITUDE_LIMIT as f64;
        let outcome = gumbel_search_with_root_value(&mut search, &state, &gcfg, root_value);
        state = Othello::apply(state, &outcome.action);
        plies += 1;
    }
    plies
}

/// Runs `games` self-play games with at most `workers` concurrently live,
/// every leaf evaluation funneled through one shared [`Batcher`]. Returns
/// (total plies, wall-clock seconds, batches run, requests served).
fn play_games_batched(
    games: usize,
    sims: u32,
    workers: usize,
    batch_size: usize,
    poll: Duration,
) -> (usize, f64, usize, usize) {
    let gcfg = GumbelConfig { sims, max_considered: 8, ..GumbelConfig::default() };
    let batcher = Arc::new(Batcher::new(CnnValueNet::default(), 64, batch_size));

    let stop = Arc::new(std::sync::atomic::AtomicBool::new(false));
    let poller_batcher = batcher.clone();
    let poller_stop = stop.clone();
    let poller = std::thread::spawn(move || {
        while !poller_stop.load(Relaxed) {
            std::thread::sleep(poll);
            poller_batcher.poll_once();
        }
        poller_batcher.poll_once();
    });

    let next_game = Arc::new(AtomicUsize::new(0));
    let total_plies = Arc::new(AtomicUsize::new(0));
    let start = Instant::now();

    std::thread::scope(|scope| {
        for _ in 0..workers {
            let next_game = next_game.clone();
            let total_plies = total_plies.clone();
            let batcher = batcher.clone();
            scope.spawn(move || loop {
                let g = next_game.fetch_add(1, Relaxed);
                if g >= games {
                    break;
                }
                let net = BatchedNet::new(batcher.clone());
                let plies = play_one_game(net, gcfg, 1000 + g as u64 * 100_000);
                total_plies.fetch_add(plies, Relaxed);
            });
        }
    });

    let wall = start.elapsed().as_secs_f64();
    stop.store(true, Relaxed);
    poller.join().unwrap();

    (
        total_plies.load(Relaxed),
        wall,
        batcher.batches_run.load(Relaxed),
        batcher.requests_served.load(Relaxed),
    )
}

fn main() {
    let mut args = std::env::args().skip(1);
    let games: usize = args.next().and_then(|s| s.parse().ok()).unwrap_or(128);
    let sims: u32 = args.next().and_then(|s| s.parse().ok()).unwrap_or(16);
    let workers: usize = args.next().and_then(|s| s.parse().ok()).unwrap_or(games.min(64));
    let batch_size: usize = args.next().and_then(|s| s.parse().ok()).unwrap_or(workers);
    let poll_us: u64 = args.next().and_then(|s| s.parse().ok()).unwrap_or(200);

    println!(
        "batched crates/mcts: {games} games, {sims} sims/move, {workers} concurrent workers, \
         batch_size={batch_size}, poll={poll_us}us (all-zero weights)..."
    );

    let (plies, wall, batches, requests) =
        play_games_batched(games, sims, workers, batch_size, Duration::from_micros(poll_us));

    println!(
        "batched: {plies} plies in {wall:.3}s -- {:.3} ms/ply, {:.2} games/sec",
        wall * 1000.0 / plies as f64,
        games as f64 / wall
    );
    println!(
        "batching stats: {batches} GPU calls, {requests} requests served, {:.2} requests/call avg",
        requests as f64 / batches.max(1) as f64
    );
}
