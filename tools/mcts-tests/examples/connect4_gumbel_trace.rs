//! Per-hop sign/perspective trace of one real Gumbel search on hand-built
//! Connect Four positions.
//!
//! For each position it prints, with perspective labels: the board and the
//! side to move; the CNN root value; the value the search actually cached on
//! the root node (`EvaluatedCutoff::raw_evaluator_value`); every legal child's
//! CNN value in the child mover's perspective and negated to the root mover's;
//! the realized visit counts and edge Q values (root mover perspective); the
//! `completed_q` vector fed into the Sequential-Halving ranking key; the
//! improved policy; and the final chosen action.
//!
//! Usage: `cargo run --release -p mcts-tests --example connect4_gumbel_trace -- <gen0.c4cnn>`

use std::process::ExitCode;

use game_connect4::{convnet::CnnValuePolicyNet, Move, Standard, State};
use mcts::algorithms::mcts::gumbel::{
    completed_q, gumbel_search_with_root_value, improved_policy, transform_completed_q, GumbelConfig,
};
use mcts::algorithms::mcts::node::QInit;
use mcts::algorithms::mcts::profile::Mcts;
use mcts::algorithms::mcts::select::GumbelCompletedQ;
use mcts::algorithms::mcts::simulate::EvaluatedCutoff;
use mcts::algorithms::mcts::{SearchConfig, TreeSearch};
use mcts::game::{Game, PlayerIndex};

type Profile = Mcts<GumbelCompletedQ, EvaluatedCutoff<Standard, CnnValuePolicyNet>>;

fn apply_cols(cols: &[u8]) -> State<6, 7> {
    let mut state = State::<6, 7>::default();
    for &c in cols {
        assert!(!Standard::is_terminal(&state), "script hit a terminal early");
        state = Standard::apply(state, &Move(c));
    }
    state
}

struct Position {
    name: &'static str,
    what: &'static str,
    state: State<6, 7>,
    /// The column that hand analysis says search must choose, if unambiguous.
    expected: Option<u8>,
}

fn positions() -> Vec<Position> {
    vec![
        Position {
            name: "immediate-win",
            what: "Black to move has a vertical four in column 0 available now",
            // B0 W1 B0 W1 B0 W1 -> Black 3 in col 0, White 3 in col 1, 3-3
            // discs, Black to move, Black plays col 0 and wins.
            state: apply_cols(&[0, 1, 0, 1, 0, 1]),
            expected: Some(0),
        },
        Position {
            name: "must-block",
            what: "White threatens a vertical four in column 0; Black to move must block col 0",
            // B1 W0 B2 W0 B5 W0 -> White 3 in col 0, Black scattered in
            // 1/2/5 (no Black threat), 3-3 discs, Black to move. Any move but
            // col 0 loses to White's vertical four next ply.
            state: apply_cols(&[1, 0, 2, 0, 5, 0]),
            expected: Some(0),
        },
        Position {
            name: "neutral-midgame",
            what: "quiet, roughly balanced middlegame, Black to move",
            state: apply_cols(&[3, 3, 2, 4, 4, 2]),
            expected: None,
        },
    ]
}

fn run(net: &CnnValuePolicyNet, cfg: &GumbelConfig, pos: &Position) -> bool {
    let state = &pos.state;
    let player = Standard::player_to_move(state).to_index();
    let mover = Standard::player_to_move(state);

    println!("\n================ {} ================", pos.name);
    println!("{}", pos.what);
    println!("{state}");
    println!("side to move: {mover:?} (player index {player})");
    println!("terminal? {}", Standard::is_terminal(state));

    let root_value = net.value(state) as f64;
    println!(
        "\nCNN root value (in {mover:?}'s perspective, + = good for {mover:?}): {root_value:+.4}"
    );

    let mut legal = Vec::new();
    Standard::generate_actions(state, &mut legal);
    println!("\nlegal children (child value is in the CHILD mover's perspective):");
    for mv in &legal {
        let child = Standard::apply(*state, mv);
        let terminal = Standard::is_terminal(&child);
        let child_mover = Standard::player_to_move(&child);
        let cv = net.value(&child) as f64;
        // Negate to the root mover's perspective. For a winning move the
        // child is terminal and `apply` leaves `turn` on the winner, so the
        // "child mover" label is the winner, not the real next mover -- the
        // search never evaluates that node with the CNN (expand skips
        // terminals), so this row is informational only.
        println!(
            "  col {}: child_mover={:?} value={:+.4}  -> root-perspective {:+.4}{}",
            mv.0,
            child_mover,
            cv,
            -cv,
            if terminal { "   [TERMINAL - not CNN-evaluated in search]" } else { "" }
        );
    }

    let mut search: TreeSearch<Standard, Profile> = TreeSearch::default().config(
        SearchConfig::default()
            .expand_threshold(1)
            .max_playout_depth(0)
            .q_init(QInit::Loss)
            .select(GumbelCompletedQ::with_config(*cfg))
            .simulate(EvaluatedCutoff::new().evaluator(net.clone()))
            .with_policy_logits(net.clone())
            .seed(20260909),
    );

    let outcome = gumbel_search_with_root_value(&mut search, state, cfg, root_value);

    let children = search.index.get(search.root_id).children();
    let k = children.len();
    let cached_root = children.raw_evaluator_value();
    println!(
        "\ncached root raw_evaluator_value (node mover perspective): {:?}  (== CNN root value up to eval-magnitude rounding)",
        cached_root.map(|v| format!("{v:+.4}"))
    );

    let logits: Vec<f64> = children.policy_logits().to_vec();
    let visits: Vec<u32> = (0..k).map(|i| children.num_visits(i)).collect();
    let qs: Vec<f64> = (0..k).map(|i| children.expected_score(i, player)).collect();
    let completed = completed_q(root_value, &logits, &visits, &qs);
    let transformed = transform_completed_q(&completed, cfg.rescale_q);
    let improved = improved_policy(&logits, &visits, &completed, cfg);

    println!(
        "\nper-action (Q and completed_q are in {mover:?}'s / root mover's perspective):"
    );
    println!("  col   logit   visits     Q(root)   completed_q   transformed   improved_policy");
    for i in 0..k {
        println!(
            "  {:>3}  {:+.3}   {:>6}   {:+.4}     {:+.4}      {:+.4}       {:.4}",
            children.action(i).0,
            logits[i],
            visits[i],
            qs[i],
            completed[i],
            transformed[i],
            improved[i],
        );
    }

    let completed_argmax = (0..k)
        .max_by(|&a, &b| completed[a].partial_cmp(&completed[b]).unwrap())
        .unwrap();
    let improved_argmax = (0..k)
        .max_by(|&a, &b| improved[a].partial_cmp(&improved[b]).unwrap())
        .unwrap();
    println!(
        "\nargmax completed_q  -> col {}", children.action(completed_argmax).0
    );
    println!("argmax improved_policy -> col {}", children.action(improved_argmax).0);
    println!("Gumbel final chosen action -> col {}", outcome.action.0);

    let mut ok = true;
    if let Some(exp) = pos.expected {
        println!("\nhand-verified expectation: col {exp}");
        let pick_ok = outcome.action.0 == exp;
        let cq_ok = children.action(completed_argmax).0 == exp;
        println!(
            "  final pick matches: {}   completed_q argmax matches: {}",
            pick_ok, cq_ok
        );
        if pos.name == "immediate-win" {
            // The only bulletproof invariant: a forced descent into the
            // winning column reaches a terminal win, so its edge Q is exactly
            // +1 in the root mover's perspective and nothing can exceed it.
            let win_q = qs[exp as usize];
            println!("  winning-column edge Q (expect +1.0000): {win_q:+.4}");
            ok &= (win_q - 1.0).abs() < 1e-9;
            ok &= cq_ok && pick_ok;
        }
    }
    ok
}

fn main() -> ExitCode {
    let args: Vec<_> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("usage: connect4_gumbel_trace <gen0.c4cnn>");
        return ExitCode::FAILURE;
    }
    let net = match CnnValuePolicyNet::load(&args[1]) {
        Ok(net) => net,
        Err(error) => {
            eprintln!("{}: {error}", args[1]);
            return ExitCode::FAILURE;
        }
    };

    let configs = [
        (
            "mctx-verbatim (c_visit=50, c_scale=0.1, rescale=true)",
            GumbelConfig::default(),
        ),
        (
            "small-offset (c_visit=0, c_scale=0.05, rescale=false)",
            GumbelConfig {
                c_visit: 0.0,
                c_scale: 0.05,
                rescale_q: false,
                ..GumbelConfig::default()
            },
        ),
    ];

    let mut all_ok = true;
    for (label, cfg) in &configs {
        println!("\n########################################################");
        println!("# CONFIG: {label}");
        println!("########################################################");
        for pos in positions() {
            all_ok &= run(&net, cfg, &pos);
        }
    }

    if all_ok {
        println!("\nALL HAND-VERIFIED INVARIANTS HOLD (winning move: edge Q +1, top completed_q, chosen)");
        ExitCode::SUCCESS
    } else {
        println!("\nAN INVARIANT FAILED - see above");
        ExitCode::FAILURE
    }
}
