use game_host::{
    run_cli, AiMoveResult, AiPresetInfo, Analysis, AnalysisAction, GameAdapter, HostError,
    TunerInfo,
};
use serde::{Deserialize, Serialize};
use serde_json::Value;

use game_congo::{Congo, Move, Piece, Player, State, MAX_CAPTURES, NUM_SQUARES};
use mcts::game::Game;
use mcts::strategies::mcts::{node::QInit, strategy, SearchConfig, TreeSearch};
use mcts::strategies::Search;

/// Number of self-play games one `tune_eval` call runs when the caller
/// doesn't override it -- also reported as `eval_rounds` in `tuner()`.
const TUNE_EVAL_ROUNDS: u32 = 20;

fn piece_code(piece: Piece) -> &'static str {
    match piece {
        Piece::Giraffe => "giraffe",
        Piece::Monkey => "monkey",
        Piece::Elephant => "elephant",
        Piece::Lion => "lion",
        Piece::Crocodile => "crocodile",
        Piece::Zebra => "zebra",
        Piece::Pawn => "pawn",
        Piece::Superpawn => "superpawn",
    }
}

fn parse_piece(code: &str) -> Result<Piece, HostError> {
    Ok(match code {
        "giraffe" => Piece::Giraffe,
        "monkey" => Piece::Monkey,
        "elephant" => Piece::Elephant,
        "lion" => Piece::Lion,
        "crocodile" => Piece::Crocodile,
        "zebra" => Piece::Zebra,
        "pawn" => Piece::Pawn,
        "superpawn" => Piece::Superpawn,
        other => return Err(HostError::bad_request(format!("invalid piece: {other}"))),
    })
}

fn player_name(p: Player) -> &'static str {
    match p {
        Player::Black => "Black",
        Player::White => "White",
    }
}

fn parse_player(name: &str) -> Result<Player, HostError> {
    match name {
        "Black" => Ok(Player::Black),
        "White" => Ok(Player::White),
        _ => Err(HostError::bad_request(format!("invalid player: {name:?}"))),
    }
}

#[derive(Serialize, Deserialize)]
struct WireCell {
    player: String,
    piece: String,
}

#[derive(Serialize, Deserialize)]
struct WireState {
    squares: Vec<Option<WireCell>>,
    river_since: Vec<u8>,
    turn: String,
}

#[derive(Serialize, Deserialize)]
struct WireMove {
    from: u8,
    to: u8,
    captures: Vec<u8>,
    /// Ordered landing squares (see `Move::hops`'s doc comment): lets a
    /// client disambiguate which specific jump order it's choosing when
    /// several land on the same square via different capture sets.
    hops: Vec<u8>,
}

fn move_to_value(m: &Move) -> Value {
    serde_json::to_value(WireMove {
        from: m.from,
        to: m.to,
        captures: m.captures().to_vec(),
        hops: m.hops().to_vec(),
    })
    .expect("WireMove serializes")
}

fn value_to_move(v: &Value) -> Result<Move, HostError> {
    let w: WireMove = serde_json::from_value(v.clone())
        .map_err(|e| HostError::bad_request(format!("invalid move: {e}")))?;
    if w.captures.len() > MAX_CAPTURES || w.hops.len() > MAX_CAPTURES {
        return Err(HostError::bad_request("too many captures in move"));
    }
    let mut captures = [0u8; MAX_CAPTURES];
    captures[..w.captures.len()].copy_from_slice(&w.captures);
    let mut hops = [0u8; MAX_CAPTURES];
    hops[..w.hops.len()].copy_from_slice(&w.hops);
    Ok(Move {
        from: w.from,
        to: w.to,
        num_captures: w.captures.len() as u8,
        captures,
        num_hops: w.hops.len() as u8,
        hops,
    })
}

fn state_to_value(s: &State) -> Value {
    let squares = (0..NUM_SQUARES)
        .map(|i| {
            s.get(i).map(|(player, piece)| WireCell {
                player: player_name(player).to_string(),
                piece: piece_code(piece).to_string(),
            })
        })
        .collect();
    serde_json::to_value(WireState {
        squares,
        river_since: (0..NUM_SQUARES).map(|i| s.river_since(i)).collect(),
        turn: player_name(s.turn()).into(),
    })
    .expect("WireState serializes")
}

fn value_to_state(v: &Value) -> Result<State, HostError> {
    let w: WireState = serde_json::from_value(v.clone())
        .map_err(|e| HostError::bad_request(format!("invalid state: {e}")))?;
    if w.squares.len() != NUM_SQUARES || w.river_since.len() != NUM_SQUARES {
        return Err(HostError::bad_request("invalid state: wrong square count"));
    }
    let mut cells = [None; NUM_SQUARES];
    for (i, cell) in w.squares.iter().enumerate() {
        cells[i] = match cell {
            Some(c) => Some((parse_player(&c.player)?, parse_piece(&c.piece)?)),
            None => None,
        };
    }
    let mut river_since = [0u8; NUM_SQUARES];
    river_since.copy_from_slice(&w.river_since);
    Ok(State::from_parts(
        cells,
        river_since,
        parse_player(&w.turn)?,
    ))
}

fn build_easy() -> Box<dyn Search<G = Congo>> {
    Box::new(
        TreeSearch::<Congo, strategy::Ucb1>::new().config(
            SearchConfig::new()
                .name("congo/easy")
                .expand_threshold(1)
                .max_iterations(200)
                .q_init(QInit::Infinity),
        ),
    )
}

fn build_strong() -> Box<dyn Search<G = Congo>> {
    Box::new(
        TreeSearch::<Congo, strategy::Ucb1>::new().config(
            SearchConfig::new()
                .name("congo/strong")
                .expand_threshold(0)
                .max_iterations(5000)
                .use_mcts_solver(true)
                .q_init(QInit::Loss),
        ),
    )
}

const PRESETS: &[PresetEntry] = &[
    PresetEntry {
        id: "easy",
        label: "Easy",
        description: "Plain UCB1 with a modest budget.",
        build: build_easy,
    },
    PresetEntry {
        id: "strong",
        label: "Strong",
        description: "UCB1 with MCTS-Solver, deep iterations.",
        build: build_strong,
    },
];

struct PresetEntry {
    id: &'static str,
    label: &'static str,
    description: &'static str,
    build: fn() -> Box<dyn Search<G = Congo>>,
}

struct CongoAdapter;

impl GameAdapter for CongoAdapter {
    fn kind(&self) -> &'static str {
        "congo"
    }
    fn label(&self) -> &'static str {
        "Congo"
    }
    fn description(&self) -> &'static str {
        "A 7x7 chess variant by Demian Freeling: giraffes, monkeys, elephants, a lion, a \
         crocodile, a zebra, and pawns cross a river. Capture the enemy lion to win."
    }
    fn default_config(&self) -> Value {
        serde_json::json!({})
    }

    fn new_state(&self, _config: Value) -> Result<Value, HostError> {
        Ok(state_to_value(&State::initial()))
    }

    fn legal_moves(&self, state: &Value) -> Result<Vec<Value>, HostError> {
        let s = value_to_state(state)?;
        let mut moves = Vec::new();
        if !Congo::is_terminal(&s) {
            Congo::generate_actions(&s, &mut moves);
        }
        Ok(moves.iter().map(move_to_value).collect())
    }

    fn apply(&self, state: &Value, mv: &Value) -> Result<Value, HostError> {
        let s = value_to_state(state)?;
        let action = value_to_move(mv)?;
        if Congo::is_terminal(&s) {
            return Err(HostError::bad_request("game is over"));
        }
        let mut legal = Vec::new();
        Congo::generate_actions(&s, &mut legal);
        if !legal.contains(&action) {
            return Err(HostError::bad_request("illegal move"));
        }
        Ok(state_to_value(&Congo::apply(s, &action)))
    }

    fn view(&self, state: &Value) -> Result<Value, HostError> {
        let s = value_to_state(state)?;
        let winner = Congo::winner(&s);
        let mut view = state_to_value(&s);
        view["winner"] = match winner {
            Some(p) => Value::String(player_name(p).into()),
            None => Value::Null,
        };
        view["terminal"] = Value::Bool(Congo::is_terminal(&s));
        Ok(view)
    }

    fn ai_presets(&self) -> Vec<AiPresetInfo> {
        PRESETS
            .iter()
            .map(|p| AiPresetInfo {
                id: p.id.into(),
                label: p.label.into(),
                description: p.description.into(),
            })
            .collect()
    }

    fn ai_move(&self, state: &Value, preset: &str) -> Result<AiMoveResult, HostError> {
        let s = value_to_state(state)?;
        let spec = PRESETS
            .iter()
            .find(|p| p.id == preset)
            .ok_or_else(|| HostError::not_found(format!("unknown preset: {preset}")))?;
        if Congo::is_terminal(&s) {
            return Err(HostError::bad_request("game is over"));
        }
        let mut ai = (spec.build)();
        let action = ai.choose_action(&s);
        let next = Congo::apply(s, &action);
        Ok(AiMoveResult {
            mv: move_to_value(&action),
            state: state_to_value(&next),
        })
    }

    fn analyze(
        &self,
        state: &Value,
        preset: &str,
        _budget_ms: Option<u64>,
    ) -> Result<Analysis, HostError> {
        let s = value_to_state(state)?;
        let spec = PRESETS
            .iter()
            .find(|p| p.id == preset)
            .ok_or_else(|| HostError::not_found(format!("unknown preset: {preset}")))?;
        if Congo::is_terminal(&s) {
            return Err(HostError::bad_request("game is over"));
        }
        let mut ai = (spec.build)();
        let _ = ai.choose_action(&s);
        let report = ai.root_report(&s);
        let suggested = report.principal_variation.first().map(move_to_value);
        Ok(Analysis {
            actions: report
                .actions
                .into_iter()
                .map(|a| AnalysisAction {
                    action: move_to_value(&a.action),
                    visits: a.visits,
                    mean_value: a.mean_value,
                    is_proven: a.is_proven,
                })
                .collect(),
            principal_variation: report
                .principal_variation
                .iter()
                .map(move_to_value)
                .collect(),
            total_visits: report.total_visits,
            suggested_move: suggested,
        })
    }

    fn tuner(&self) -> Option<TunerInfo> {
        Some(TunerInfo {
            game_config: self.default_config(),
            ..mcts_tune::strategy_tuner_info(&["strong"], TUNE_EVAL_ROUNDS)
        })
    }

    fn tune_eval(
        &self,
        params: Value,
        rounds: u32,
        seed: Option<u64>,
        _baseline: Option<String>,
        baseline_config: Option<Value>,
        _game_config: Option<Value>,
    ) -> Result<Value, HostError> {
        // Congo's `Game::zobrist_hash` is the default constant `0`, so
        // transpositions must stay off -- see `mcts-tune`'s
        // `strategy_tune_eval` doc comment.
        let outcome = if let Some(cfg) = baseline_config {
            let baseline_seed = seed.unwrap_or(0);
            mcts_tune::build_search::<Congo>(&cfg, baseline_seed, false)?;
            mcts_tune::strategy_tune_eval(
                &params,
                rounds,
                seed,
                false,
                mcts_tune::SearchBudget::default(),
                move || {
                    mcts_tune::build_search::<Congo>(&cfg, baseline_seed, false)
                        .expect("baseline_config already validated above")
                },
                Default::default(),
            )?
        } else {
            mcts_tune::strategy_tune_eval(
                &params,
                rounds,
                seed,
                false,
                mcts_tune::SearchBudget::default(),
                build_strong,
                Default::default(),
            )?
        };
        Ok(serde_json::json!({
            "cost": outcome.cost,
            "wins": outcome.wins,
            "losses": outcome.losses,
            "draws": outcome.draws,
        }))
    }
}

fn main() {
    run_cli(CongoAdapter);
}

#[cfg(test)]
mod tests {
    use super::*;
    use game_congo::{Piece, Player};

    /// `Move::hops` is what lets the UI disambiguate two different Monkey
    /// jump-chains that happen to converge on the same square (see
    /// `Move`'s doc comment in `lib.rs`) -- this only helps if the wire
    /// format actually carries it through `move_to_value`/`value_to_move`
    /// intact, since a client only ever sees a move via this JSON shape.
    #[test]
    fn wire_move_round_trips_hops() {
        let mut cells = [None; game_congo::NUM_SQUARES];
        let idx = |r: i32, c: i32| (r * 7 + c) as usize;
        let m = idx(0, 3);
        cells[m] = Some((Player::Black, Piece::Monkey));
        cells[idx(0, 4)] = Some((Player::White, Piece::Pawn));
        cells[idx(1, 5)] = Some((Player::White, Piece::Pawn));
        let s = game_congo::State::from_parts(cells, [0; game_congo::NUM_SQUARES], Player::Black);

        let mut actions = Vec::new();
        s.generate_moves(&mut actions);
        let chain = actions
            .iter()
            .find(|a| a.from as usize == m && a.num_captures == 2)
            .expect("two-capture chain exists");
        assert!(
            chain.hops().len() > 1,
            "test move should exercise a real chain"
        );

        let round_tripped = value_to_move(&move_to_value(chain)).expect("round-trips");
        assert_eq!(round_tripped, *chain);
        assert_eq!(round_tripped.hops(), chain.hops());
    }
}
