//! Phase-1 generation gate for the Gumbel AlphaZero loop on Connect Four.
//!
//!     cargo run --release -p mcts-tests --example gumbel_connect4_gate -- \
//!         <baseline_weights.bin> <candidate_weights.bin> [games] [sims]
//!
//! Connect Four has no free perfect-play oracle (unlike tic-tac-toe), so the
//! tic-tac-toe gate's move-agreement check is replaced by an absolute
//! strength anchor. Two checks, both must pass (non-zero exit otherwise):
//!
//!  1. **Head to head.** `games` alternating-colour games, candidate vs
//!     baseline, both playing the Gumbel self-play search. The candidate's
//!     score share (win 1, draw 0.5) must have a Wilson 95% lower bound
//!     strictly above 0.5 -- this generation genuinely beat the last one.
//!  2. **Rollout anchor.** The same head-to-head, candidate vs a fixed
//!     net-free reference player (the all-zero value head, playouts rolled
//!     out to a natural terminal). The reference never changes, so its
//!     score share is directly comparable generation over generation; the
//!     candidate's Wilson lower bound against it must exceed 0.5.

use std::process::ExitCode;

use mcts::algorithms::mcts::gumbel::GumbelConfig;
use mcts::util::battle_royale;

use game_connect4::policynet::NTuplePolicyNet;
use game_connect4::selfplay::GumbelPlayer;
use game_connect4::valuenet::NTupleValueNet;
use game_connect4::Standard;

/// Deepest a 6x7 game can run -- every cell filled.
const MAX_DEPTH: usize = 42;

fn wilson_lower_bound(successes: f64, n: usize, z: f64) -> f64 {
    if n == 0 {
        return 0.0;
    }
    let n = n as f64;
    let phat = successes / n;
    let z2 = z * z;
    let denom = 1.0 + z2 / n;
    let center = phat + z2 / (2.0 * n);
    let margin = z * ((phat * (1.0 - phat) + z2 / (4.0 * n)) / n).sqrt();
    (center - margin) / denom
}

fn load(path: &str) -> NTupleValueNet {
    NTupleValueNet::load(path).unwrap_or_else(|e| panic!("cannot load {path}: {e}"))
}
fn load_policy(path: Option<&String>) -> NTuplePolicyNet {
    path.map_or_else(NTuplePolicyNet::default, |p| {
        NTuplePolicyNet::load(p).unwrap_or_else(|e| panic!("cannot load {p}: {e}"))
    })
}

/// Candidate's score share (win 1, draw 0.5) over `games` alternating-colour
/// games against `opponent`, with the Wilson 95% lower bound. `opponent` is
/// built fresh each call from `make_opponent` so a persistent search tree
/// never leaks between the two checks.
fn score_share(
    candidate: &NTupleValueNet,
    candidate_policy: &NTuplePolicyNet,
    cfg: GumbelConfig,
    games: usize,
    make_opponent: impl Fn(u64) -> GumbelPlayer,
) -> (usize, usize, usize, f64, f64) {
    let mut cand = GumbelPlayer::with_policy(candidate.clone(), candidate_policy.clone(), cfg, 7);
    let mut opp = make_opponent(11);
    let (mut wins, mut draws, mut losses) = (0usize, 0usize, 0usize);
    let mut score = 0.0f64;
    for i in 0..games {
        let (result, candidate_first) = if i % 2 == 0 {
            (battle_royale::<Standard, _, _>(&mut cand, &mut opp), true)
        } else {
            (battle_royale::<Standard, _, _>(&mut opp, &mut cand), false)
        };
        match result {
            None => {
                score += 0.5;
                draws += 1;
            }
            Some(0) if candidate_first => {
                score += 1.0;
                wins += 1;
            }
            Some(1) if !candidate_first => {
                score += 1.0;
                wins += 1;
            }
            _ => losses += 1,
        }
    }
    let lb = wilson_lower_bound(score, games, 1.96);
    (wins, draws, losses, score / games as f64, lb)
}

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 {
        eprintln!(
            "usage: gumbel_connect4_gate <baseline_weights.bin> <candidate_weights.bin> \
             [games] [sims] [--baseline-policy file] [--candidate-policy file]"
        );
        return ExitCode::FAILURE;
    }
    let baseline = load(&args[1]);
    let candidate = load(&args[2]);
    let games: usize = args.get(3).map_or(200, |s| s.parse().expect("games"));
    let sims: u32 = args.get(4).map_or(32, |s| s.parse().expect("sims"));
    let cfg = GumbelConfig {
        sims,
        ..GumbelConfig::default()
    };
    let baseline_policy = load_policy(
        args.windows(2)
            .find(|x| x[0] == "--baseline-policy")
            .map(|x| &x[1]),
    );
    let candidate_policy = load_policy(
        args.windows(2)
            .find(|x| x[0] == "--candidate-policy")
            .map(|x| &x[1]),
    );

    let (w, d, l, share, lb) = score_share(&candidate, &candidate_policy, cfg, games, |seed| {
        GumbelPlayer::with_policy(baseline.clone(), baseline_policy.clone(), cfg, seed)
    });
    let h2h_pass = lb > 0.5;
    println!(
        "head to head vs baseline: candidate {w}-{d}-{l} (W-D-L) over {games}, \
         score share {share:.3}, Wilson LB {lb:.3}  -> {}",
        if h2h_pass { "PASS" } else { "FAIL" }
    );

    let (w, d, l, share, lb) = score_share(&candidate, &candidate_policy, cfg, games, |seed| {
        GumbelPlayer::with_playout_depth(NTupleValueNet::default(), cfg, seed, MAX_DEPTH)
    });
    let anchor_pass = lb > 0.5;
    println!(
        "rollout anchor: candidate {w}-{d}-{l} (W-D-L) over {games}, \
         score share {share:.3}, Wilson LB {lb:.3}  -> {}",
        if anchor_pass { "PASS" } else { "FAIL" }
    );

    if h2h_pass && anchor_pass {
        println!("GATE: PASS");
        ExitCode::SUCCESS
    } else {
        println!("GATE: FAIL");
        ExitCode::FAILURE
    }
}
