use game_host::{
    ConfiguredCandidateSide, ConfiguredMatchResult, ConfiguredOutcome, ConfiguredStrategyMetrics,
    HostError, SearchReport,
};
use mcts::game::{Game, PlayerIndex};
use mcts::strategies::Search;
use serde_json::{json, Value};

use crate::{
    build_search, family_catalog::TrialParams, presets, search::make_candidate, trace, SearchBudget,
};

type StateEncoder<'a, G> = dyn Fn(&<G as Game>::S) -> Value + 'a;
type MoveEncoder<'a, G> = dyn Fn(&<G as Game>::S, &<G as Game>::A) -> Option<Value> + 'a;

/// Result of one `strategy_tune_eval` call: `cost` is what tuner minimizes
/// (the candidate's loss rate against the baseline); `wins`/`losses`/`draws`
/// are from the candidate's perspective, for display.
#[derive(Debug)]
pub struct TuneEvalOutcome {
    pub cost: f64,
    pub wins: u32,
    pub losses: u32,
    pub draws: u32,
}

/// Play `rounds` rounds (each round: one game candidate-first, one game
/// baseline-first) of `params`-built candidate vs `baseline_build()`, and
/// return the aggregate outcome. `baseline_build` is a thunk rather than an
/// already-built search so a fresh, un-treed instance is used for every
/// single game, matching how each preset's `build_*` function is used
/// elsewhere in a game's own crate.
///
/// `use_transpositions` should only be `true` for a game with a real
/// `Game::zobrist_hash` override -- the default hash is a constant `0`, and
/// enabling transpositions against it merges every node in the tree into
/// one, silently corrupting the search rather than erroring. Pass `true`
/// only for games that have actually implemented `zobrist_hash`.
///
/// `initial_state` is the board every game in this call starts from --
/// almost every caller passes `G::S::default()` (`Game::S` requires
/// `Default`), matching this function's behavior before `initial_state`
/// existed. A game with a real runtime-configurable setup (e.g. Druid's
/// board size) builds one from its own `tune_eval`'s `game_config` argument
/// instead; every other game ignores that argument entirely, since its
/// board is fixed at compile time regardless.
/// `candidate_budget` is the compute the *candidate* side gets -- see
/// `SearchBudget`'s doc comment. Pass `SearchBudget::default()` when
/// `baseline_build` builds an opponent already on the same iteration-based
/// footing as the candidate (a `baseline_config`-backed self-play opponent,
/// built via `build_search`, or any of this crate's own floor families);
/// pass a budget mirroring the opponent's own compute when `baseline_build`
/// wraps a named, wall-clock/thread-budgeted preset instead (see
/// `games/druid/src/main.rs`'s `tune_eval`).
#[allow(clippy::too_many_arguments)]
pub fn strategy_tune_eval<G: Game + 'static>(
    params: &Value,
    rounds: u32,
    seed: Option<u64>,
    use_transpositions: bool,
    candidate_budget: SearchBudget,
    baseline_build: impl Fn() -> Box<dyn Search<G = G>>,
    initial_state: G::S,
    state_to_value: impl Fn(&G::S) -> Value,
    move_to_value: impl Fn(&G::S, &G::A) -> Option<Value>,
    trace_path: Option<&std::path::Path>,
    trace_game_sequence_start: Option<u64>,
    on_game: &mut dyn FnMut(ConfiguredMatchResult) -> Result<(), HostError>,
) -> Result<TuneEvalOutcome, HostError> {
    let trial: TrialParams = serde_json::from_value(params.clone())
        .map_err(|e| HostError::bad_request(format!("invalid tuning params: {e}")))?;
    let seed = seed.unwrap_or(0);

    if rounds == 0 {
        let _ = make_candidate::<G>(&trial, seed, use_transpositions, &candidate_budget)?;
        return Ok(TuneEvalOutcome {
            cost: 0.0,
            wins: 0,
            losses: 0,
            draws: 0,
        });
    }

    let mut tracer = match (trace_path, trace_game_sequence_start) {
        (Some(path), Some(sequence)) => Some(
            trace::MoveTracer::open_with_sequence(path, sequence)
                .map_err(|e| HostError::bad_request(format!("failed to open --trace-path: {e}")))?,
        ),
        (Some(path), None) => Some(
            trace::MoveTracer::open(path)
                .map_err(|e| HostError::bad_request(format!("failed to open --trace-path: {e}")))?,
        ),
        (None, None) => None,
        (None, Some(_)) => {
            return Err(HostError::bad_request(
                "--trace-game-sequence-start requires --trace-path",
            ))
        }
    };

    let (mut wins, mut losses, mut draws) = (0u32, 0u32, 0u32);
    let mut seq = 0u64;
    for round in 1..=rounds {
        let mut candidate = make_candidate(&trial, seed, use_transpositions, &candidate_budget)?;
        let mut baseline = baseline_build();

        seq += 1;
        let result = play_one(
            candidate.as_mut(),
            baseline.as_mut(),
            initial_state.clone(),
            &state_to_value,
            &move_to_value,
            tracer.as_mut(),
            round,
            seq,
            seed,
            ConfiguredCandidateSide::First,
        )?;
        match result.outcome {
            ConfiguredOutcome::CandidateWin => wins += 1,
            ConfiguredOutcome::BaselineWin => losses += 1,
            ConfiguredOutcome::Draw => draws += 1,
        }
        on_game(result)?;

        // Swap move order so the candidate plays second half the time.
        let mut candidate = make_candidate(&trial, seed, use_transpositions, &candidate_budget)?;
        let mut baseline = baseline_build();
        seq += 1;
        let result = play_one(
            baseline.as_mut(),
            candidate.as_mut(),
            initial_state.clone(),
            &state_to_value,
            &move_to_value,
            tracer.as_mut(),
            round,
            seq,
            seed,
            ConfiguredCandidateSide::Second,
        )?;
        match result.outcome {
            ConfiguredOutcome::CandidateWin => wins += 1,
            ConfiguredOutcome::BaselineWin => losses += 1,
            ConfiguredOutcome::Draw => draws += 1,
        }
        on_game(result)?;
    }

    Ok(TuneEvalOutcome {
        cost: cost_from_losses(losses, rounds),
        wins,
        losses,
        draws,
    })
}

/// Implements the common shape of `GameAdapter::tune_eval` directly, for a
/// game whose board is fixed at compile time -- it ignores `game_config` and
/// always starts from `G::S::default()`. `baseline` is a named entry of
/// `presets`' own id list (whatever `tuner()` reports via `ai_preset_ids`),
/// `None` meaning "the table's first/default entry" -- covers every current
/// game except the handful with a runtime-configurable board (e.g.
/// `games/druid`, whose `tune_eval` builds `initial_state` from its own
/// `game_config` argument instead); those keep writing out
/// `strategy_tune_eval`/`build_search` directly, matching what this function
/// does internally.
///
/// `presets_source` names the preset file in the panic message if the
/// resolved baseline preset fails to build (e.g. `"games/ttt/presets.json"`)
/// -- this only fires if that file's own preset is broken, which should
/// never happen since it ships with the crate.
#[allow(clippy::too_many_arguments)]
pub fn generic_tune_eval<G: Game + 'static>(
    presets: &presets::PresetTable,
    presets_source: &str,
    use_transpositions: bool,
    preset_seed: u64,
    baseline: Option<String>,
    params: Value,
    rounds: u32,
    seed: Option<u64>,
    baseline_config: Option<Value>,
    max_iterations: Option<usize>,
    max_time_ms: Option<u64>,
    state_to_value: impl Fn(&G::S) -> Value,
    move_to_value: impl Fn(&G::S, &G::A) -> Option<Value>,
    trace_path: Option<std::path::PathBuf>,
    trace_game_sequence_start: Option<u64>,
    on_game: &mut dyn FnMut(ConfiguredMatchResult) -> Result<(), HostError>,
) -> Result<Value, HostError> {
    // `use_transpositions: true` requires a real `Game::zobrist_hash`
    // override -- the caller is responsible for only passing `true` when `G`
    // has one, so merging transposed nodes during the candidate's search is
    // safe here (see `strategy_tune_eval`'s doc comment).
    let outcome = if let Some(cfg) = baseline_config {
        let baseline_seed = seed.unwrap_or(0);
        // This opponent is itself a `build_search`-built config, on the
        // same iteration-based footing as the candidate -- both sides get
        // the *same* budget (an operator's `max_iterations` override
        // included) so there's nothing to match asymmetrically (see
        // `SearchBudget`'s and `build_search`'s doc comments).
        let budget = SearchBudget {
            max_iterations,
            max_time: max_time_ms.map(std::time::Duration::from_millis),
            ..Default::default()
        };
        // Fail fast on an invalid baseline config, before any games are
        // played -- mirrors how a bad candidate `params` is already
        // rejected during `TrialParams` deserialization inside
        // `strategy_tune_eval` itself.
        build_search::<G>(&cfg, baseline_seed, use_transpositions, &budget)?;
        strategy_tune_eval::<G>(
            &params,
            rounds,
            seed,
            use_transpositions,
            budget,
            move || {
                build_search::<G>(&cfg, baseline_seed, use_transpositions, &budget)
                    .expect("baseline_config already validated above")
            },
            Default::default(),
            &state_to_value,
            &move_to_value,
            trace_path.as_deref(),
            trace_game_sequence_start,
            on_game,
        )?
    } else {
        let baseline_id = baseline
            .as_deref()
            .or_else(|| presets.ai_preset_ids().first().copied())
            .expect("presets table must declare at least one preset");
        strategy_tune_eval::<G>(
            &params,
            rounds,
            seed,
            use_transpositions,
            SearchBudget {
                max_iterations,
                max_time: max_time_ms.map(std::time::Duration::from_millis),
                ..Default::default()
            },
            move || {
                presets
                    .build::<G>(baseline_id, preset_seed)
                    .unwrap_or_else(|e| {
                        panic!("{presets_source}'s {baseline_id:?} preset must build: {e}")
                    })
            },
            Default::default(),
            &state_to_value,
            &move_to_value,
            trace_path.as_deref(),
            trace_game_sequence_start,
            on_game,
        )?
    };
    Ok(json!({
        "cost": outcome.cost,
        "wins": outcome.wins,
        "losses": outcome.losses,
        "draws": outcome.draws,
    }))
}

/// `cost = losses / (2*rounds)`: the candidate's loss rate across the
/// `2*rounds` games it plays (moving first and second each round), the
/// quantity tuner minimizes. Draws and wins both count as "not a loss" --
/// this doesn't reward wins over draws, only penalizes outright losses.
pub(crate) fn cost_from_losses(losses: u32, rounds: u32) -> f64 {
    let total = 2.0 * rounds as f64;
    if total > 0.0 {
        losses as f64 / total
    } else {
        0.0
    }
}

/// Play one game between `first` (moves when `player_to_move` is index 0)
/// and `second` (index 1). Returns `(first_result, second_result, draws)`
/// win/loss counts -- each pair is `(1,0)`/`(0,1)`/`(0,0)` with the third
/// element set on a draw.
///
/// Assumes a 2-player, win/loss/draw game (`G::winner() -> Option<P>` with
/// indices 0/1) -- every current caller is, but a 3+-player game would need
/// this generalized first.
#[allow(clippy::too_many_arguments)]
fn play_one<G: Game>(
    first: &mut dyn Search<G = G>,
    second: &mut dyn Search<G = G>,
    initial_state: G::S,
    state_to_value: &StateEncoder<'_, G>,
    move_to_value: &MoveEncoder<'_, G>,
    mut tracer: Option<&mut trace::MoveTracer>,
    round: u32,
    seq: u64,
    seed: u64,
    candidate_side: ConfiguredCandidateSide,
) -> Result<ConfiguredMatchResult, HostError> {
    let started = std::time::Instant::now();
    let game_seq = tracer.as_mut().map(|t| t.start_game());
    let mut state = initial_state;
    let mut ply = 0u32;
    let mut trace_ply = 0u32;
    let mut measurements = Vec::new();

    if let (Some(t), Some(seq)) = (tracer.as_mut(), game_seq) {
        t.write_ply(seq, ply, state_to_value(&state), None, None, None);
    }

    while !G::is_terminal(&state) {
        let mover = G::player_to_move(&state).to_index();
        let pre_move_state = state.clone();
        let move_started = std::time::Instant::now();
        let (action, search): (G::A, SearchReport) = if mover == 0 {
            crate::choose_action_with_report(first, &state, |action| {
                move_to_value(&pre_move_state, action).unwrap_or(Value::Null)
            })
        } else {
            crate::choose_action_with_report(second, &state, |action| {
                move_to_value(&pre_move_state, action).unwrap_or(Value::Null)
            })
        };
        let visits = search.root_visits;
        state = G::apply(state, &action);
        ply += 1;
        measurements.push((mover, visits as u64, move_started.elapsed()));

        if let (Some(t), Some(seq), Some(mv)) = (
            tracer.as_mut(),
            game_seq,
            move_to_value(&pre_move_state, &action),
        ) {
            let player = if mover == 0 {
                if candidate_side == ConfiguredCandidateSide::First {
                    "candidate"
                } else {
                    "baseline"
                }
            } else {
                if candidate_side == ConfiguredCandidateSide::Second {
                    "candidate"
                } else {
                    "baseline"
                }
            };
            t.write_ply(
                seq,
                trace_ply + 1,
                state_to_value(&state),
                Some(mv),
                Some(player),
                Some(search),
            );
            trace_ply += 1;
        }
    }
    let outcome = match G::winner(&state) {
        None => ConfiguredOutcome::Draw,
        Some(p) if (p.to_index() == 0) == (candidate_side == ConfiguredCandidateSide::First) => {
            ConfiguredOutcome::CandidateWin
        }
        Some(_) => ConfiguredOutcome::BaselineWin,
    };
    let first_half = ply / 2;
    let mut candidate = ConfiguredStrategyMetrics::default();
    let mut baseline = ConfiguredStrategyMetrics::default();
    for (index, (mover, visits, elapsed)) in measurements.into_iter().enumerate() {
        let metrics = if (mover == 0) == (candidate_side == ConfiguredCandidateSide::First) {
            &mut candidate
        } else {
            &mut baseline
        };
        metrics.iterations_total = metrics.iterations_total.saturating_add(visits);
        metrics.move_time_ms = metrics
            .move_time_ms
            .saturating_add(elapsed.as_millis().min(u64::MAX as u128) as u64);
        if (index as u32 + 1) <= first_half {
            metrics.iterations_first_half = metrics.iterations_first_half.saturating_add(visits);
        }
    }
    Ok(ConfiguredMatchResult {
        record_type: "configured_match_result".into(),
        seq,
        round,
        seed,
        candidate_side,
        outcome,
        trace_game_seq: game_seq,
        plies: ply,
        elapsed_ms: started.elapsed().as_millis().min(u64::MAX as u128) as u64,
        candidate,
        baseline,
    })
}
