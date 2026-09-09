//! The visit-evidence-scaled `sigma` modes must not break tactical play: on a
//! hand-built forced-win Connect Four position the zero-net Gumbel search still
//! has to drop the winning disc under every sigma mode, not just the
//! Mctx-verbatim default.

use game_connect4::convnet::CnnValuePolicyNet;
use game_connect4::{Move, Standard, State};
use mcts::algorithms::mcts::gumbel::{
    gumbel_search_with_root_value, GumbelConfig, SigmaMode,
};
use mcts::algorithms::mcts::node::QInit;
use mcts::algorithms::mcts::profile::Mcts;
use mcts::algorithms::mcts::select::GumbelCompletedQ;
use mcts::algorithms::mcts::simulate::EvaluatedCutoff;
use mcts::algorithms::mcts::{SearchConfig, TreeSearch};
use mcts::game::Game;

type Profile = Mcts<GumbelCompletedQ, EvaluatedCutoff<Standard, CnnValuePolicyNet>>;

/// Black to move has a vertical four in column 0 available immediately:
/// B0 W1 B0 W1 B0 W1 leaves Black owning three discs in column 0.
fn forced_win() -> State<6, 7> {
    let mut state = State::<6, 7>::default();
    for c in [0u8, 1, 0, 1, 0, 1] {
        assert!(!Standard::is_terminal(&state));
        state = Standard::apply(state, &Move(c));
    }
    state
}

fn chooses_winning_drop(cfg: GumbelConfig) -> Move {
    let net = CnnValuePolicyNet::default();
    let state = forced_win();
    let mut search: TreeSearch<Standard, Profile> = TreeSearch::default().config(
        SearchConfig::default()
            .expand_threshold(1)
            .max_playout_depth(0)
            .q_init(QInit::Loss)
            .select(GumbelCompletedQ::with_config(cfg))
            .simulate(EvaluatedCutoff::new().evaluator(net.clone()))
            .with_policy_logits(net.clone())
            .seed(20260909),
    );
    gumbel_search_with_root_value(&mut search, &state, &cfg, 0.0).action
}

#[test]
fn every_sigma_mode_still_plays_the_forced_win() {
    for mode in [
        SigmaMode::NodeFloor,
        SigmaMode::Smooth,
        SigmaMode::HardGate(2),
        SigmaMode::HardGate(3),
        SigmaMode::HardGate(4),
        SigmaMode::RealizedOnly,
    ] {
        for &(c_visit, c_scale, rescale_q) in &[(50.0, 0.1, true), (0.0, 0.05, false)] {
            let cfg = GumbelConfig {
                sims: 32,
                c_visit,
                c_scale,
                rescale_q,
                sigma_mode: mode,
                ..GumbelConfig::default()
            };
            assert_eq!(
                chooses_winning_drop(cfg),
                Move(0),
                "sigma_mode={mode:?} c_visit={c_visit} c_scale={c_scale} rescale={rescale_q}"
            );
        }
    }
}
