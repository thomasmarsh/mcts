use super::*;
use game_nim::Nim;
use mcts::backprop::{self as mcts_backprop, BackpropStrategy};
use mcts::game::Game;
use mcts::node::QInit;
use mcts::select::{self as mcts_select, SelectStrategy};
use mcts::simulate::{self as mcts_simulate, SimulateStrategy};
use mcts::strategies::mcts::strategy::Compose;
use mcts::strategies::Search;
use mcts::{GraphSearch, Requirements, SearchConfig, TreeSearch};

/// Builds and runs a `TreeSearch<Nim, Compose<S, mcts_simulate::Uniform>>`
/// for whatever concrete `S` `with_select` resolves -- the end-to-end
/// proof that a `SelectSpec` parsed from JSON reaches an optimized,
/// monomorphized search, not just a type-erased stand-in.
struct RunCont<'a, G: Game> {
    state: &'a G::S,
}

impl<'a, G: Game> SelectCont<G> for RunCont<'a, G> {
    type Output = G::A;

    fn call<S: SelectStrategy<G>>(self, select: S) -> G::A {
        let mut ts = TreeSearch::<G, Compose<S, mcts_simulate::Uniform>>::default().config(
            SearchConfig::default()
                .select(select)
                .max_iterations(200)
                .seed(1),
        );
        ts.choose_action(self.state)
    }
}

#[test]
fn select_spec_round_trips_through_json() {
    let json = r#"{"kind":"rave","threshold":700,"schedule":{"kind":"threshold","rave":700},"ucb":{"kind":"ucb1","exploration_constant":1.5}}"#;
    let spec: SelectSpec = serde_json::from_str(json).unwrap();
    assert_eq!(
        spec,
        SelectSpec::Rave {
            threshold: 700,
            schedule: mcts_select::RaveSchedule::Threshold { rave: 700 },
            ucb: mcts_select::RaveUcb::Ucb1 {
                exploration_constant: 1.5
            },
        }
    );
    assert_eq!(serde_json::to_string(&spec).unwrap(), json.replace(' ', ""));
}

#[test]
fn progressive_history_spec_round_trips_through_json() {
    let json = r#"{"kind":"progressive_history","c":1.4,"ph_weight":2.5}"#;
    let spec: SelectSpec = serde_json::from_str(json).unwrap();
    assert_eq!(
        spec,
        SelectSpec::ProgressiveHistory {
            c: 1.4,
            ph_weight: 2.5
        }
    );
    assert_eq!(serde_json::to_string(&spec).unwrap(), json.replace(' ', ""));
}

#[test]
fn bayes_uct_spec_round_trips_through_json() {
    let json = r#"{"kind":"bayes_uct1","c":1.0}"#;
    let spec: SelectSpec = serde_json::from_str(json).unwrap();
    assert_eq!(spec, SelectSpec::BayesUct1 { c: 1.0 });
    assert_eq!(serde_json::to_string(&spec).unwrap(), json);

    let json = r#"{"kind":"bayes_uct2","c":1.0}"#;
    let spec: SelectSpec = serde_json::from_str(json).unwrap();
    assert_eq!(spec, SelectSpec::BayesUct2 { c: 1.0 });
    assert_eq!(serde_json::to_string(&spec).unwrap(), json);
}

#[test]
fn bayes_backprop_spec_round_trips_through_json() {
    let json = r#"{"kind":"bayes_gaussian","prior_variance":1.0,"obs_variance":1.0}"#;
    let spec: BackpropSpec = serde_json::from_str(json).unwrap();
    assert_eq!(
        spec,
        BackpropSpec::BayesGaussian {
            prior_variance: 1.0,
            obs_variance: 1.0,
        }
    );
    assert_eq!(serde_json::to_string(&spec).unwrap(), json);

    let json = r#"{"kind":"bayes_numeric","prior_variance":1.0,"obs_variance":1.0,"value_lo":-1.0,"value_hi":1.0}"#;
    let spec: BackpropSpec = serde_json::from_str(json).unwrap();
    assert_eq!(
        spec,
        BackpropSpec::BayesNumeric {
            prior_variance: 1.0,
            obs_variance: 1.0,
            value_lo: -1.0,
            value_hi: 1.0,
        }
    );
    assert_eq!(serde_json::to_string(&spec).unwrap(), json);
}

/// The select<->backprop coupling this whole feature exists to exercise:
/// `BayesUct1`/`BayesUct2` set `Requirements::needs_posterior`, which
/// only `BayesGaussian`/`BayesNumeric` satisfy -- `Classic` (or any other
/// backprop) must be rejected, not silently read zeroed posterior
/// fields.
#[test]
fn validate_search_spec_rejects_bayes_select_paired_with_classic_backprop() {
    let spec = SearchSpec {
        select: SelectSpec::BayesUct1 { c: 1.0 },
        simulate: SimulateSpec::Uniform {},
        backprop: BackpropSpec::Classic {},
        final_action: FinalActionSpec::RobustChild {},
    };
    let err = validate_search_spec::<Nim>(&spec).unwrap_err();
    assert!(err.contains("Bayesian backprop"), "{err}");
}

#[test]
fn validate_search_spec_accepts_bayes_select_paired_with_bayes_backprop() {
    let spec = SearchSpec {
        select: SelectSpec::BayesUct2 { c: 1.0 },
        simulate: SimulateSpec::Uniform {},
        backprop: BackpropSpec::BayesGaussian {
            prior_variance: 1.0,
            obs_variance: 1.0,
        },
        final_action: FinalActionSpec::RobustChild {},
    };
    assert!(validate_search_spec::<Nim>(&spec).is_ok());

    let spec = SearchSpec {
        backprop: BackpropSpec::BayesNumeric {
            prior_variance: 1.0,
            obs_variance: 1.0,
            value_lo: -1.0,
            value_hi: 1.0,
        },
        ..spec
    };
    assert!(validate_search_spec::<Nim>(&spec).is_ok());
}

/// A non-Bayes select paired with a Bayes backprop is fine -- the
/// backprop just does extra work nothing reads, no different from any
/// other over-provisioned `Requirements`.
#[test]
fn validate_search_spec_accepts_classic_select_paired_with_bayes_backprop() {
    let spec = SearchSpec {
        select: SelectSpec::Ucb1 { c: 1.4 },
        simulate: SimulateSpec::Uniform {},
        backprop: BackpropSpec::BayesGaussian {
            prior_variance: 1.0,
            obs_variance: 1.0,
        },
        final_action: FinalActionSpec::RobustChild {},
    };
    assert!(validate_search_spec::<Nim>(&spec).is_ok());
}

/// End-to-end proof that a `BayesUct1` select paired with a
/// `BayesGaussian` backprop, both parsed from JSON via `build_search`,
/// runs a real search rather than tripping the `needs_posterior`
/// rejection or panicking on the posterior fields it reads.
#[test]
fn build_search_runs_bayes_uct_paired_with_bayes_backprop() {
    let spec = SearchSpec {
        select: SelectSpec::BayesUct1 { c: 1.0 },
        simulate: SimulateSpec::Uniform {},
        backprop: BackpropSpec::BayesGaussian {
            prior_variance: 1.0,
            obs_variance: 1.0,
        },
        final_action: FinalActionSpec::RobustChild {},
    };
    validate_search_spec::<Nim>(&spec).unwrap();
    let mut search = build_search::<Nim>(&spec, &nim_search_settings());
    let state = <Nim as Game>::S::default();
    let action = search.choose_action(&state);
    let mut legal = Vec::new();
    Nim::generate_actions(&state, &mut legal);
    assert!(legal.contains(&action));
}

#[test]
fn ments_select_spec_round_trips_through_json() {
    let json = r#"{"kind":"ments","tau":1.0,"epsilon":0.1}"#;
    let spec: SelectSpec = serde_json::from_str(json).unwrap();
    assert_eq!(
        spec,
        SelectSpec::Ments {
            tau: 1.0,
            epsilon: 0.1
        }
    );
    assert_eq!(serde_json::to_string(&spec).unwrap(), json);
}

#[test]
fn softmax_backprop_spec_round_trips_through_json() {
    let json = r#"{"kind":"softmax","tau":1.0}"#;
    let spec: BackpropSpec = serde_json::from_str(json).unwrap();
    assert_eq!(spec, BackpropSpec::Softmax { tau: 1.0 });
    assert_eq!(serde_json::to_string(&spec).unwrap(), json);
}

/// `select::Ments` sets `Requirements::needs_softmax_value`, which only
/// `backprop::SoftmaxBackprop` satisfies -- `Classic` must be rejected.
#[test]
fn validate_search_spec_rejects_ments_select_paired_with_classic_backprop() {
    let spec = SearchSpec {
        select: SelectSpec::Ments {
            tau: 1.0,
            epsilon: 0.1,
        },
        simulate: SimulateSpec::Uniform {},
        backprop: BackpropSpec::Classic {},
        final_action: FinalActionSpec::RobustChild {},
    };
    let err = validate_search_spec::<Nim>(&spec).unwrap_err();
    assert!(err.contains("softmax"), "{err}");
}

#[test]
fn validate_search_spec_accepts_ments_select_paired_with_softmax_backprop() {
    let spec = SearchSpec {
        select: SelectSpec::Ments {
            tau: 1.0,
            epsilon: 0.1,
        },
        simulate: SimulateSpec::Uniform {},
        backprop: BackpropSpec::Softmax { tau: 1.0 },
        final_action: FinalActionSpec::RobustChild {},
    };
    assert!(validate_search_spec::<Nim>(&spec).is_ok());
}

#[test]
fn build_search_runs_ments_paired_with_softmax_backprop() {
    let spec = SearchSpec {
        select: SelectSpec::Ments {
            tau: 0.5,
            epsilon: 0.1,
        },
        simulate: SimulateSpec::Uniform {},
        backprop: BackpropSpec::Softmax { tau: 0.5 },
        final_action: FinalActionSpec::RobustChild {},
    };
    validate_search_spec::<Nim>(&spec).unwrap();
    let mut search = build_search::<Nim>(&spec, &nim_search_settings());
    let state = <Nim as Game>::S::default();
    let action = search.choose_action(&state);
    let mut legal = Vec::new();
    Nim::generate_actions(&state, &mut legal);
    assert!(legal.contains(&action));
}

#[test]
fn epsilon_greedy_wraps_an_arbitrary_inner_spec() {
    let json =
        r#"{"kind":"epsilon_greedy","epsilon":0.2,"inner":{"kind":"uct_pn","c":1.4,"c_pn":1.0}}"#;
    let spec: SelectSpec = serde_json::from_str(json).unwrap();
    let SelectSpec::EpsilonGreedy { epsilon, inner } = &spec else {
        panic!("expected EpsilonGreedy");
    };
    assert_eq!(*epsilon, 0.2);
    assert_eq!(*inner, BaseSelectSpec::UctPn { c: 1.4, c_pn: 1.0 });
}

#[test]
fn requirements_of_matches_the_real_components_own_answer() {
    // `UctPn` is the one hand-picked case in mcts/src that overrides
    // `requirements()` beyond `backprop_flags()` -- proving this table's
    // `requirements_of` reports the same thing the concrete component
    // does, with no second copy of "UctPn needs the solver" written here.
    let spec = SelectSpec::UctPn { c: 1.4, c_pn: 1.0 };
    let reqs = requirements_of::<Nim>(&spec);
    assert!(reqs.solver);
    assert_eq!(
        reqs.max_players, None,
        "UctPn's rank bonus is sound at any player count -- see its requirements() doc comment"
    );

    // `mcts_select::Rave` reads its own ancestor-keyed GRAVE table
    // (`SelectContext::grave`), not the per-child AMAF field
    // `mcts_select::Amaf` uses -- so its real requirement is `grave`, not
    // `amaf` (see `Rave::backprop_flags`). Asserting the wrong one here
    // would have passed silently if this table just repeated a
    // hand-guessed answer instead of calling the real component.
    let rave = SelectSpec::Rave {
        threshold: 700,
        schedule: mcts_select::RaveSchedule::Threshold { rave: 700 },
        ucb: mcts_select::RaveUcb::Ucb1 {
            exploration_constant: 1.4,
        },
    };
    assert!(requirements_of::<Nim>(&rave).grave);

    let amaf = SelectSpec::Amaf { alpha: 1.0, c: 1.4 };
    assert!(requirements_of::<Nim>(&amaf).amaf);

    // Wrapping in EpsilonGreedy must not lose UctPn's requirements --
    // the same property `mcts-tests`' select-side test checks against
    // the real `mcts_select::EpsilonGreedy` type, exercised here through the
    // spec/dispatch layer instead.
    let wrapped = SelectSpec::EpsilonGreedy {
        epsilon: 0.1,
        inner: BaseSelectSpec::UctPn { c: 1.4, c_pn: 1.0 },
    };
    assert_eq!(requirements_of::<Nim>(&wrapped), reqs);
}

#[test]
fn with_select_builds_a_working_tree_search_from_a_json_spec() {
    let spec: SelectSpec = serde_json::from_str(r#"{"kind":"ucb1","c":1.5}"#).unwrap();
    let state = <Nim as Game>::S::default();
    let action = with_select::<Nim, _>(&spec, RunCont { state: &state });
    let mut legal = Vec::new();
    Nim::generate_actions(&state, &mut legal);
    assert!(
        legal.contains(&action),
        "the action chosen by a JSON-configured search must be legal"
    );
}

#[test]
fn with_select_builds_a_working_tree_search_for_progressive_history() {
    let spec: SelectSpec =
        serde_json::from_str(r#"{"kind":"progressive_history","c":1.4,"ph_weight":2.5}"#).unwrap();
    let state = <Nim as Game>::S::default();
    let action = with_select::<Nim, _>(&spec, RunCont { state: &state });
    let mut legal = Vec::new();
    Nim::generate_actions(&state, &mut legal);
    assert!(
        legal.contains(&action),
        "the action chosen by a JSON-configured search must be legal"
    );
}

/// Builds and runs a `TreeSearch<Nim, Compose<mcts_select::Ucb1, S>>` for
/// whatever concrete `S` `with_simulate` resolves -- the `simulate`-axis
/// counterpart of `RunCont` above.
struct RunSimulateCont<'a, G: Game> {
    state: &'a G::S,
}

impl<'a, G: Game> SimulateCont<G> for RunSimulateCont<'a, G> {
    type Output = G::A;

    fn call<S: SimulateStrategy<G>>(self, simulate: S) -> G::A {
        let mut ts = TreeSearch::<G, Compose<mcts_select::Ucb1, S>>::default().config(
            SearchConfig::default()
                .simulate(simulate)
                .max_iterations(200)
                .seed(1),
        );
        ts.choose_action(self.state)
    }
}

#[test]
fn simulate_spec_round_trips_through_json() {
    let json = r#"{"kind":"nst","backoff_threshold":10}"#;
    let spec: SimulateSpec = serde_json::from_str(json).unwrap();
    assert_eq!(
        spec,
        SimulateSpec::Nst {
            backoff_threshold: 10
        }
    );
    assert_eq!(serde_json::to_string(&spec).unwrap(), json.replace(' ', ""));
}

#[test]
fn lgr_simulate_spec_round_trips_through_json() {
    let json = r#"{"kind":"lgr"}"#;
    let spec: SimulateSpec = serde_json::from_str(json).unwrap();
    assert_eq!(spec, SimulateSpec::Lgr {});
    assert_eq!(serde_json::to_string(&spec).unwrap(), json);
}

#[test]
fn lgr2_simulate_spec_round_trips_through_json() {
    let json = r#"{"kind":"lgr2"}"#;
    let spec: SimulateSpec = serde_json::from_str(json).unwrap();
    assert_eq!(spec, SimulateSpec::Lgr2 {});
    assert_eq!(serde_json::to_string(&spec).unwrap(), json);
}

#[test]
fn lgr2_mast_simulate_spec_round_trips_through_json() {
    let json = r#"{"kind":"lgr2_mast"}"#;
    let spec: SimulateSpec = serde_json::from_str(json).unwrap();
    assert_eq!(spec, SimulateSpec::Lgr2Mast {});
    assert_eq!(serde_json::to_string(&spec).unwrap(), json);
}

#[test]
fn simulate_epsilon_greedy_and_decisive_move_wrap_an_arbitrary_inner_spec() {
    let json = r#"{"kind":"epsilon_greedy","epsilon":0.2,"inner":{"kind":"mast"}}"#;
    let spec: SimulateSpec = serde_json::from_str(json).unwrap();
    let SimulateSpec::EpsilonGreedy { epsilon, inner } = &spec else {
        panic!("expected EpsilonGreedy");
    };
    assert_eq!(*epsilon, 0.2);
    assert_eq!(*inner, BaseSimulateSpec::Mast {});

    let json = r#"{"kind":"decisive_move","mode":"win_loss","inner":{"kind":"nst","backoff_threshold":5}}"#;
    let spec: SimulateSpec = serde_json::from_str(json).unwrap();
    let SimulateSpec::DecisiveMove { mode, inner } = &spec else {
        panic!("expected DecisiveMove");
    };
    assert_eq!(*mode, mcts_simulate::DecisiveMoveMode::WinLoss);
    assert_eq!(
        *inner,
        BaseSimulateSpec::Nst {
            backoff_threshold: 5
        }
    );
}

#[test]
fn requirements_of_simulate_matches_the_real_components_own_answer() {
    // `Nst` sets both `global` and `nst` (see `mcts_simulate::Nst`'s doc
    // comment on why it needs the unigram table on top of its own
    // bigram one) -- asserting only `nst` here would have passed
    // silently if this table dropped `global` from the real
    // `backprop_flags()` answer.
    let nst = SimulateSpec::Nst {
        backoff_threshold: 5,
    };
    let reqs = requirements_of_simulate::<Nim>(&nst);
    assert!(reqs.global);
    assert!(reqs.nst);

    let mast = SimulateSpec::Mast {};
    assert!(requirements_of_simulate::<Nim>(&mast).global);
    assert!(!requirements_of_simulate::<Nim>(&mast).nst);

    let uniform = SimulateSpec::Uniform {};
    assert_eq!(
        requirements_of_simulate::<Nim>(&uniform),
        Requirements::default()
    );

    // Wrapping in EpsilonGreedy/DecisiveMove must not lose Nst's
    // requirements -- both wrappers delegate `requirements()` straight
    // to `inner` (see `mcts_simulate::EpsilonGreedy`/`DecisiveMove`'s own
    // doc comments), so this checks that survives the spec/dispatch
    // layer too.
    let wrapped_eg = SimulateSpec::EpsilonGreedy {
        epsilon: 0.1,
        inner: BaseSimulateSpec::Nst {
            backoff_threshold: 5,
        },
    };
    assert_eq!(requirements_of_simulate::<Nim>(&wrapped_eg), reqs);

    let wrapped_dm = SimulateSpec::DecisiveMove {
        mode: mcts_simulate::DecisiveMoveMode::Win,
        inner: BaseSimulateSpec::Nst {
            backoff_threshold: 5,
        },
    };
    assert_eq!(requirements_of_simulate::<Nim>(&wrapped_dm), reqs);

    // `Lgr` sets its own `lgr` bit only -- unlike `Nst`, it doesn't
    // touch the `global`/unigram table at all (see `mcts_simulate::Lgr`'s
    // doc comment).
    let lgr = SimulateSpec::Lgr {};
    let lgr_reqs = requirements_of_simulate::<Nim>(&lgr);
    assert!(lgr_reqs.lgr);
    assert!(!lgr_reqs.global);
    assert!(!lgr_reqs.nst);

    let wrapped_lgr_eg = SimulateSpec::EpsilonGreedy {
        epsilon: 0.1,
        inner: BaseSimulateSpec::Lgr {},
    };
    assert_eq!(requirements_of_simulate::<Nim>(&wrapped_lgr_eg), lgr_reqs);

    // `Lgr2` sets both its own `lgr2` bit *and* `lgr` -- its default
    // inner is `Lgr` (LGR-1), the fallback the resolved
    // `mcts_simulate::Lgr2::<G>::new()` actually nests, so its requirements
    // must union in whatever that nested `Lgr` needs too.
    let lgr2 = SimulateSpec::Lgr2 {};
    let lgr2_reqs = requirements_of_simulate::<Nim>(&lgr2);
    assert!(lgr2_reqs.lgr2);
    assert!(lgr2_reqs.lgr);
    assert!(!lgr2_reqs.global);
    assert!(!lgr2_reqs.nst);

    let wrapped_lgr2_eg = SimulateSpec::EpsilonGreedy {
        epsilon: 0.1,
        inner: BaseSimulateSpec::Lgr2 {},
    };
    assert_eq!(requirements_of_simulate::<Nim>(&wrapped_lgr2_eg), lgr2_reqs);

    // `Lgr2Mast` nests `Mast` as its ultimate fallback, so it must pull
    // in `global` (the unigram table `Mast` reads) on top of `lgr2`/
    // `lgr`.
    let lgr2_mast = SimulateSpec::Lgr2Mast {};
    let lgr2_mast_reqs = requirements_of_simulate::<Nim>(&lgr2_mast);
    assert!(lgr2_mast_reqs.lgr2);
    assert!(lgr2_mast_reqs.lgr);
    assert!(lgr2_mast_reqs.global);
    assert!(!lgr2_mast_reqs.nst);
}

#[test]
fn with_simulate_builds_a_working_tree_search_from_a_json_spec() {
    let spec: SimulateSpec = serde_json::from_str(r#"{"kind":"uniform"}"#).unwrap();
    let state = <Nim as Game>::S::default();
    let action = with_simulate::<Nim, _>(&spec, RunSimulateCont { state: &state });
    let mut legal = Vec::new();
    Nim::generate_actions(&state, &mut legal);
    assert!(
        legal.contains(&action),
        "the action chosen by a JSON-configured search must be legal"
    );
}

#[test]
fn meta_mcts_spec_round_trips_through_json() {
    let json = r#"{"kind":"meta_mcts","iterations":50}"#;
    let spec: SimulateSpec = serde_json::from_str(json).unwrap();
    assert_eq!(spec, SimulateSpec::MetaMcts { iterations: 50 });
    assert_eq!(serde_json::to_string(&spec).unwrap(), json.replace(' ', ""));
}

#[test]
fn with_simulate_builds_a_working_nested_search_for_meta_mcts() {
    // The inner search is always `Compose<Ucb1, Uniform>` -- see
    // `register_simulate!`'s doc comment on why `MetaMcts`'s inner
    // strategy isn't independently configurable.
    let spec = SimulateSpec::MetaMcts { iterations: 25 };
    let state = <Nim as Game>::S::default();
    let action = with_simulate::<Nim, _>(&spec, RunSimulateCont { state: &state });
    let mut legal = Vec::new();
    Nim::generate_actions(&state, &mut legal);
    assert!(
        legal.contains(&action),
        "the action chosen by a JSON-configured MetaMcts search must be legal"
    );
}

#[test]
fn decisive_move_mast_spec_round_trips_through_json() {
    let json = r#"{"kind":"decisive_move_mast","mode":"win_loss","epsilon":0.2}"#;
    let spec: SimulateSpec = serde_json::from_str(json).unwrap();
    assert_eq!(
        spec,
        SimulateSpec::DecisiveMoveMast {
            mode: mcts_simulate::DecisiveMoveMode::WinLoss,
            epsilon: 0.2,
        }
    );
    assert_eq!(serde_json::to_string(&spec).unwrap(), json.replace(' ', ""));
}

#[test]
fn with_simulate_builds_a_working_search_for_decisive_move_mast() {
    let spec = SimulateSpec::DecisiveMoveMast {
        mode: mcts_simulate::DecisiveMoveMode::WinLoss,
        epsilon: 0.2,
    };
    let state = <Nim as Game>::S::default();
    let action = with_simulate::<Nim, _>(&spec, RunSimulateCont { state: &state });
    let mut legal = Vec::new();
    Nim::generate_actions(&state, &mut legal);
    assert!(
        legal.contains(&action),
        "the action chosen by a JSON-configured DecisiveMoveMast search must be legal"
    );
}

#[test]
fn anti_decisive_mode_round_trips_through_json_and_builds_a_working_search() {
    let json = r#"{"kind":"decisive_move_mast","mode":"anti_decisive","epsilon":0.2}"#;
    let spec: SimulateSpec = serde_json::from_str(json).unwrap();
    assert_eq!(
        spec,
        SimulateSpec::DecisiveMoveMast {
            mode: mcts_simulate::DecisiveMoveMode::AntiDecisive,
            epsilon: 0.2,
        }
    );
    assert_eq!(serde_json::to_string(&spec).unwrap(), json.replace(' ', ""));

    let state = <Nim as Game>::S::default();
    let action = with_simulate::<Nim, _>(&spec, RunSimulateCont { state: &state });
    let mut legal = Vec::new();
    Nim::generate_actions(&state, &mut legal);
    assert!(
        legal.contains(&action),
        "the action chosen by a JSON-configured AntiDecisive search must be legal"
    );
}

#[test]
fn backprop_spec_round_trips_through_json() {
    let json = r#"{"kind":"classic"}"#;
    let spec: BackpropSpec = serde_json::from_str(json).unwrap();
    assert_eq!(spec, BackpropSpec::Classic {});
    assert_eq!(serde_json::to_string(&spec).unwrap(), json);
}

/// `BackpropSpec`'s `Deserialize` is hand-implemented (see
/// `register_backprop!`'s doc comment), not `#[derive]`d -- these pin the
/// same error behavior a derive would have given for free, since nothing
/// enforces that automatically anymore.
#[test]
fn backprop_spec_deserialize_rejects_unknown_kind_and_missing_fields() {
    let err = serde_json::from_str::<BackpropSpec>(r#"{"kind":"not_a_real_kind"}"#).unwrap_err();
    assert!(err.to_string().contains("not_a_real_kind"), "{err}");

    let err = serde_json::from_str::<BackpropSpec>(r#"{"kind":"bayes_gaussian"}"#).unwrap_err();
    assert!(err.to_string().contains("prior_variance"), "{err}");

    let err = serde_json::from_str::<BackpropSpec>(r#"{"prior_variance":1.0}"#).unwrap_err();
    assert!(err.to_string().contains("kind"), "{err}");
}

/// A `BackpropCont` whose `Output` is just a marker proving `with_backprop`
/// actually resolved to a real `BackpropStrategy` -- there's no
/// `requirements()` to check (see `register_backprop!`'s doc comment on
/// why), so this is the `backprop`-axis analogue of the `select`/
/// `simulate` "build a working search" tests, minus the search: any
/// `BackpropStrategy` is usable in a `Compose<..>` without further
/// per-type configuration.
struct ResolvedCont;

impl BackpropCont for ResolvedCont {
    type Output = &'static str;

    fn call<B: BackpropStrategy>(self, _backprop: B) -> &'static str {
        "resolved"
    }
}

#[test]
fn with_backprop_resolves_a_real_backprop_strategy() {
    let spec = BackpropSpec::Classic {};
    assert_eq!(with_backprop(&spec, ResolvedCont), "resolved");
}

#[test]
fn final_action_spec_round_trips_through_json() {
    let json = r#"{"kind":"secure_child","a":2.5}"#;
    let spec: FinalActionSpec = serde_json::from_str(json).unwrap();
    assert_eq!(spec, FinalActionSpec::SecureChild { a: 2.5 });
    assert_eq!(serde_json::to_string(&spec).unwrap(), json);
}

#[test]
fn requirements_of_final_action_matches_the_real_components_own_answer() {
    // None of the four `final_action` families override `requirements()`
    // beyond the `SelectStrategy` default, unlike `UctPn` on the `select`
    // axis -- so this just pins that they all resolve to
    // `Requirements::none()` rather than silently picking up some future
    // override without a test noticing.
    for spec in [
        FinalActionSpec::RobustChild {},
        FinalActionSpec::MaxAvg {},
        FinalActionSpec::MaxRobustChild {},
        FinalActionSpec::SecureChild { a: 4.0 },
    ] {
        assert_eq!(
            requirements_of_final_action::<Nim>(&spec),
            Requirements::default()
        );
    }
}

/// Builds and runs a `TreeSearch<Nim, Compose<mcts_select::Ucb1, mcts_simulate::Uniform,
/// mcts_backprop::Classic, S>>` for whatever concrete `S` `with_final_action`
/// resolves -- the `final_action`-axis counterpart of `RunCont`/
/// `RunSimulateCont` above.
struct RunFinalActionCont<'a, G: Game> {
    state: &'a G::S,
}

impl<'a, G: Game> SelectCont<G> for RunFinalActionCont<'a, G> {
    type Output = G::A;

    fn call<S: SelectStrategy<G>>(self, final_action: S) -> G::A {
        let mut ts = TreeSearch::<
            G,
            Compose<mcts_select::Ucb1, mcts_simulate::Uniform, mcts_backprop::Classic, S>,
        >::default()
        .config(
            SearchConfig::default()
                .final_action(final_action)
                .max_iterations(200)
                .seed(1),
        );
        ts.choose_action(self.state)
    }
}

#[test]
fn with_final_action_builds_a_working_tree_search_from_a_json_spec() {
    let spec: FinalActionSpec = serde_json::from_str(r#"{"kind":"robust_child"}"#).unwrap();
    let state = <Nim as Game>::S::default();
    let action = with_final_action::<Nim, _>(&spec, RunFinalActionCont { state: &state });
    let mut legal = Vec::new();
    Nim::generate_actions(&state, &mut legal);
    assert!(
        legal.contains(&action),
        "the action chosen by a JSON-configured search must be legal"
    );
}

fn nim_search_settings() -> SearchSettings {
    SearchSettings {
        max_iterations: 200,
        max_playout_depth: 200,
        expand_threshold: 1,
        q_init: QInit::Parent,
        use_transpositions: false,
        use_mcts_solver: false,
        reuse_tree: false,
        num_tree_threads: 1,
        num_threads: 1,
        determinize_root: false,
        seed: 1,
        max_time: None,
        graph_search: None,
        transposition_keying: mcts::TranspositionKeying::PerPly,
        solver_loss_threshold: None,
        contempt_factor: None,
    }
}

#[test]
fn search_spec_round_trips_through_json() {
    let json = r#"{
            "select": {"kind": "ucb1", "c": 1.4},
            "simulate": {"kind": "uniform"},
            "backprop": {"kind": "classic"},
            "final_action": {"kind": "robust_child"}
        }"#;
    let spec: SearchSpec = serde_json::from_str(json).unwrap();
    assert_eq!(
        spec,
        SearchSpec {
            select: SelectSpec::Ucb1 { c: 1.4 },
            simulate: SimulateSpec::Uniform {},
            backprop: BackpropSpec::Classic {},
            final_action: FinalActionSpec::RobustChild {},
        }
    );
}

#[test]
fn build_search_builds_a_working_tree_search_from_a_full_json_spec() {
    // Unlike every other test in this file, this drives all four axes'
    // specs through one call, proving they compose into a real, runnable
    // `Box<dyn Search<G>>` -- not just that each axis resolves on its
    // own.
    let spec: SearchSpec = serde_json::from_str(
            r#"{
                "select": {"kind": "epsilon_greedy", "epsilon": 0.1, "inner": {"kind": "ucb1", "c": 1.4}},
                "simulate": {"kind": "mast"},
                "backprop": {"kind": "classic"},
                "final_action": {"kind": "secure_child", "a": 2.0}
            }"#,
        )
        .unwrap();
    let mut search = build_search::<Nim>(&spec, &nim_search_settings());
    let state = <Nim as Game>::S::default();
    let action = search.choose_action(&state);
    let mut legal = Vec::new();
    Nim::generate_actions(&state, &mut legal);
    assert!(
        legal.contains(&action),
        "the action chosen by a JSON-configured search must be legal"
    );
}

#[test]
fn build_search_wires_meta_mcts_through_the_full_spec() {
    let spec = SearchSpec {
        select: SelectSpec::Ucb1 { c: 1.4 },
        simulate: SimulateSpec::MetaMcts { iterations: 25 },
        backprop: BackpropSpec::Classic {},
        final_action: FinalActionSpec::RobustChild {},
    };
    let mut search = build_search::<Nim>(&spec, &nim_search_settings());
    let state = <Nim as Game>::S::default();
    let action = search.choose_action(&state);
    let mut legal = Vec::new();
    Nim::generate_actions(&state, &mut legal);
    assert!(
        legal.contains(&action),
        "the action chosen by a JSON-configured MetaMcts search must be legal"
    );
}

#[test]
fn build_search_applies_graph_search_setting() {
    // `Nim` has no real `zobrist_hash` (defaults to a constant `0`),
    // which collapses every position into one graph node -- fine for a
    // single-iteration root expansion (only one node is ever visited),
    // but running deeper would corrupt move legality across positions
    // that fold into the same hash. `mcts-tune::lib.rs`'s
    // `mcgs_trial_selects_combined_graph_statistics` test uses the same
    // one-iteration trick for the same reason. Real `mcgs`-enabled
    // callers are guarded by the "mcgs requires a game with a zobrist
    // hash" check (step 4c), which this config-IR layer intentionally
    // doesn't duplicate (see this file's `SearchSettings` doc comment).
    let spec = SearchSpec {
        select: SelectSpec::Ucb1 { c: 1.4 },
        simulate: SimulateSpec::Uniform {},
        backprop: BackpropSpec::Classic {},
        final_action: FinalActionSpec::RobustChild {},
    };
    let mut settings = nim_search_settings();
    settings.max_iterations = 1;
    settings.expand_threshold = 0;
    settings.graph_search = Some(GraphSearch::Dag(mcts::GraphStats::Both));
    let mut search = build_search::<Nim>(&spec, &settings);
    let state = <Nim as Game>::S::default();
    let action = search.choose_action(&state);
    let mut legal = Vec::new();
    Nim::generate_actions(&state, &mut legal);
    assert!(
        legal.contains(&action),
        "the action chosen by a graph-search-configured search must be legal"
    );
}

#[test]
fn build_search_applies_solver_settings() {
    let spec = SearchSpec {
        select: SelectSpec::UctPn { c: 1.4, c_pn: 1.4 },
        simulate: SimulateSpec::Uniform {},
        backprop: BackpropSpec::Classic {},
        final_action: FinalActionSpec::RobustChild {},
    };
    let mut settings = nim_search_settings();
    settings.solver_loss_threshold = Some(1);
    settings.contempt_factor = Some(0.1);
    let mut search = build_search::<Nim>(&spec, &settings);
    let state = <Nim as Game>::S::default();
    let action = search.choose_action(&state);
    let mut legal = Vec::new();
    Nim::generate_actions(&state, &mut legal);
    assert!(
        legal.contains(&action),
        "the action chosen by a solver-configured search must be legal"
    );
}
