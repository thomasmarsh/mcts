// Measures how often independent random continuations from the same root
// reconverge on the same information set, for each hidden-information game
// in this workspace (Phantom, Ingenious, Oh Hell). This is a structural
// property of a game's legal-move graph, not of any particular search
// algorithm: it estimates how much smaller a search tree could be if nodes
// were merged whenever they represent the same information set, instead of
// getting a fresh tree node for every distinct history that reaches them.
//
// Each game exposes two hashes on its state: `Game::zobrist_hash`, which
// folds in hidden information (a rack, a hand, which ambiguous cells hold
// an opponent's marks), and an information-set hash (`public_hash` on
// Ingenious/Oh Hell, `info_set_hash` on Phantom) that omits it, so that two
// states differing only in hidden content that the mover doesn't know
// about hash equal. Bucketing many sampled continuations by each hash and
// counting how many distinct histories land in the same bucket gives two
// numbers per depth: how much a tree keyed on literal state would shrink
// (essentially never, since hidden information independently resampled
// per rollout almost never collides), and how much a tree keyed on the
// information set would shrink (which folds together both real move-order
// commuting and hidden-information collapsing).
//
// Two sampling modes separate those two effects:
// - "single determinization": every rollout from a root starts from the
//   exact same hidden-information sample and only the subsequent action
//   order varies, isolating move-order commuting in an otherwise-fixed
//   state.
// - "resampled": every rollout draws its own fresh `Game::determinize`
//   sample of the root before playing on, so both move-order commuting and
//   independently-resampled hidden information contribute -- this is the
//   quantity that matters for deciding whether merging same-information-set
//   nodes reached from different determinizations is worth doing.
//
// A rollout's "history" is identified by hashing the sequence of actions it
// took (not the state itself), so that two rollouts landing in the same
// hash bucket only count as a real reconvergence when they actually took
// different actions to get there -- two rollouts that happen to resample
// the same short prefix by chance share one history, not two, and a plain
// tree already merges that case for free without needing any special
// support.
//
// Usage: cargo run --release --example transposition_density
use game_ingenious::Ingenious2;
use game_oh_hell::OhHellStandard;
use game_phantom::{Phantom, Position};
use mcts::game::Game;
use rand::rngs::SmallRng;
use rand::SeedableRng;
use rustc_hash::{FxHashMap, FxHashSet, FxHasher};
use std::hash::{Hash, Hasher};

const NUM_ROLLOUTS: usize = 20_000;
const MAX_DEPTH: usize = 8;

struct DepthStats {
    depth: usize,
    samples: usize,
    distinct_histories: usize,
    distinct_info_states: usize,
    distinct_full_states: usize,
}

impl DepthStats {
    fn info_multiplicity(&self) -> f64 {
        self.distinct_histories as f64 / self.distinct_info_states as f64
    }

    fn full_multiplicity(&self) -> f64 {
        self.distinct_histories as f64 / self.distinct_full_states as f64
    }
}

fn mix_action(history: u64, action: &impl Hash) -> u64 {
    let mut hasher = FxHasher::default();
    action.hash(&mut hasher);
    (history ^ hasher.finish()).wrapping_mul(0x0100_0000_01B3)
}

/// Plays `plies` random actions forward from `state`, stopping early at a
/// terminal state or when no action is available.
fn advance<G: Game>(mut state: G::S, plies: usize, seed: u64) -> G::S {
    let mut rng = SmallRng::seed_from_u64(seed);
    for _ in 0..plies {
        if G::is_terminal(&state) {
            break;
        }
        let Some(action) = G::random_action(&state, &mut rng) else {
            break;
        };
        state = G::apply(state, &action);
    }
    state
}

/// Samples `num_rollouts` independent random continuations from `root` out
/// to `max_depth` plies, bucketing the state reached at each depth by both
/// `info_hash` and `full_hash`. When `redeterminize` is set, every rollout
/// starts from its own fresh `Game::determinize` sample of `root`; when
/// unset, every rollout starts from the literal `root` state and only the
/// action order varies.
fn sample<G: Game>(
    root: &G::S,
    max_depth: usize,
    num_rollouts: usize,
    seed: u64,
    redeterminize: bool,
    info_hash: impl Fn(&G::S) -> u64,
    full_hash: impl Fn(&G::S) -> u64,
) -> Vec<DepthStats> {
    let mut rng = SmallRng::seed_from_u64(seed);

    let mut info_buckets: Vec<FxHashMap<u64, FxHashSet<u64>>> =
        (0..=max_depth).map(|_| FxHashMap::default()).collect();
    let mut full_buckets: Vec<FxHashMap<u64, FxHashSet<u64>>> =
        (0..=max_depth).map(|_| FxHashMap::default()).collect();
    let mut all_histories: Vec<FxHashSet<u64>> =
        (0..=max_depth).map(|_| FxHashSet::default()).collect();
    let mut samples = vec![0usize; max_depth + 1];

    for _ in 0..num_rollouts {
        let mut state = if redeterminize {
            G::determinize(root.clone(), &mut rng)
        } else {
            root.clone()
        };
        // Seeded from the starting state's own full hash (not a constant) so
        // that two rollouts sharing an action sequence but starting from
        // different resampled hidden information are correctly treated as
        // different histories -- otherwise a state that's a deterministic
        // function of (starting state, actions taken) could land in more
        // distinct full-state buckets than there are distinct histories.
        let mut history: u64 = full_hash(&state);

        for depth in 0..=max_depth {
            samples[depth] += 1;
            info_buckets[depth]
                .entry(info_hash(&state))
                .or_default()
                .insert(history);
            full_buckets[depth]
                .entry(full_hash(&state))
                .or_default()
                .insert(history);
            all_histories[depth].insert(history);

            if G::is_terminal(&state) {
                break;
            }
            let Some(action) = G::random_action(&state, &mut rng) else {
                break;
            };
            history = mix_action(history, &action);
            state = G::apply(state, &action);
        }
    }

    (0..=max_depth)
        .map(|depth| DepthStats {
            depth,
            samples: samples[depth],
            distinct_histories: all_histories[depth].len(),
            distinct_info_states: info_buckets[depth].len(),
            distinct_full_states: full_buckets[depth].len(),
        })
        .collect()
}

fn print_table(rows: &[DepthStats]) {
    println!(
        "      {:>5}  {:>8}  {:>9}  {:>9}  {:>10}  {:>10}",
        "depth", "samples", "distinct", "distinct", "mult", "mult"
    );
    println!(
        "      {:>5}  {:>8}  {:>9}  {:>9}  {:>10}  {:>10}",
        "", "", "(info)", "(full)", "(info)", "(full)"
    );
    for row in rows.iter().skip(1) {
        if row.samples == 0 {
            println!(
                "      {:>5}  {:>8}  (root reached a terminal state before this depth)",
                row.depth, row.samples
            );
            continue;
        }
        println!(
            "      {:>5}  {:>8}  {:>9}  {:>9}  {:>9.2}x  {:>9.2}x",
            row.depth,
            row.samples,
            row.distinct_info_states,
            row.distinct_full_states,
            row.info_multiplicity(),
            row.full_multiplicity(),
        );
    }
}

/// Average full-state multiplicity over depths with at least one sample --
/// a root close to the end of the game reaches terminal states before
/// `MAX_DEPTH`, and depths past that carry no rollouts to average in.
fn avg_full_multiplicity(rows: &[DepthStats]) -> f64 {
    let tail: Vec<&DepthStats> = rows.iter().skip(1).filter(|r| r.samples > 0).collect();
    tail.iter().map(|r| r.full_multiplicity()).sum::<f64>() / tail.len() as f64
}

/// The deepest row with at least one sample -- the last depth this root's
/// rollouts actually reached before running out of game or hitting
/// `MAX_DEPTH`.
fn deepest_sampled(rows: &[DepthStats]) -> &DepthStats {
    rows.iter()
        .rev()
        .find(|r| r.samples > 0)
        .expect("depth 0 always has samples")
}

struct SummaryRow {
    game: &'static str,
    phase: &'static str,
    order_commuting_baseline: f64,
    depth_used: usize,
    info_mult_at_max_depth: f64,
    full_mult_at_max_depth: f64,
}

fn measure_phase<G: Game>(
    game: &'static str,
    phase: &'static str,
    root: &G::S,
    info_hash: impl Fn(&G::S) -> u64 + Copy,
    full_hash: impl Fn(&G::S) -> u64 + Copy,
    summary: &mut Vec<SummaryRow>,
) {
    println!("  -- {phase} root --");

    let baseline = sample::<G>(
        root,
        MAX_DEPTH,
        NUM_ROLLOUTS,
        1,
        false,
        info_hash,
        full_hash,
    );
    let order_commuting_baseline = avg_full_multiplicity(&baseline);
    println!(
        "  order-commuting baseline (single determinization, action order only): \
         avg full-state multiplicity across depths 1-{MAX_DEPTH} = {order_commuting_baseline:.3}x"
    );

    let resampled = sample::<G>(root, MAX_DEPTH, NUM_ROLLOUTS, 2, true, info_hash, full_hash);
    println!("  resampled-root (fresh determinization + action order per rollout):");
    print_table(&resampled);
    println!();

    let deepest = deepest_sampled(&resampled);
    summary.push(SummaryRow {
        game,
        phase,
        order_commuting_baseline,
        depth_used: deepest.depth,
        info_mult_at_max_depth: deepest.info_multiplicity(),
        full_mult_at_max_depth: deepest.full_multiplicity(),
    });
}

fn main() {
    println!(
        "=== Transposition density: information-set vs literal-state reconvergence \
         ({NUM_ROLLOUTS} rollouts/root, depth 1-{MAX_DEPTH}) ==="
    );
    println!();

    let mut summary = Vec::new();

    println!("=== Phantom (4,4,4) ===");
    measure_phase::<Phantom>(
        "phantom",
        "opening",
        &Position::new(),
        Position::info_set_hash,
        Position::ground_truth_hash,
        &mut summary,
    );
    measure_phase::<Phantom>(
        "phantom",
        "midgame",
        &advance::<Phantom>(Position::new(), 10, 42),
        Position::info_set_hash,
        Position::ground_truth_hash,
        &mut summary,
    );

    println!("=== Ingenious (2 players) ===");
    measure_phase::<Ingenious2>(
        "ingenious",
        "opening",
        &game_ingenious::State::<2>::new(1),
        game_ingenious::State::<2>::public_hash,
        Ingenious2::zobrist_hash,
        &mut summary,
    );
    measure_phase::<Ingenious2>(
        "ingenious",
        "midgame",
        &advance::<Ingenious2>(game_ingenious::State::<2>::new(1), 30, 42),
        game_ingenious::State::<2>::public_hash,
        Ingenious2::zobrist_hash,
        &mut summary,
    );

    println!("=== Oh Hell (4 players x 7 cards) ===");
    measure_phase::<OhHellStandard>(
        "oh_hell",
        "opening",
        &game_oh_hell::State::<4, 7>::new(1),
        game_oh_hell::State::<4, 7>::public_hash,
        OhHellStandard::zobrist_hash,
        &mut summary,
    );
    measure_phase::<OhHellStandard>(
        "oh_hell",
        "midgame",
        &advance::<OhHellStandard>(game_oh_hell::State::<4, 7>::new(1), 16, 42),
        game_oh_hell::State::<4, 7>::public_hash,
        OhHellStandard::zobrist_hash,
        &mut summary,
    );

    println!("=== Summary ===");
    println!(
        "  {:<10} {:<8} {:>5} {:>18} {:>13} {:>13}",
        "game", "phase", "depth", "order-commuting", "info mult", "full mult"
    );
    for row in &summary {
        println!(
            "  {:<10} {:<8} {:>5} {:>17.2}x {:>12.2}x {:>12.2}x",
            row.game,
            row.phase,
            row.depth_used,
            row.order_commuting_baseline,
            row.info_mult_at_max_depth,
            row.full_mult_at_max_depth,
        );
    }
    println!();
    println!(
        "Interpretation: \"order-commuting\" isolates how much a tree over a single fixed \
         determinization would already shrink from real move-order commuting alone (an ordinary \
         tree over the literal state, ignoring hidden information entirely). \"info mult\"/\"full \
         mult\" are read at the deepest depth each root's rollouts actually reached (\"depth\" \
         column) -- the number that matters for merging nodes across determinizations is how \
         many distinct histories, on average, land on the same information set once both \
         move-order commuting and independently-resampled hidden information are in play. \
         \"full mult\" close to 1.0x confirms the literal state essentially never reconverges on \
         its own once hidden information is resampled per rollout, so any info-mult gap above \
         that reflects real transposition density available to be merged, not measurement noise."
    );
}
