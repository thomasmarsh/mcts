//! Heuristic-guided playouts: the `DruidHeuristic` simulate strategy built on
//! `heuristic_scores`'s three Browne-motivated detectors, plus the
//! `RaveDecisiveHeuristic` preset wrapper. The score core is shared between
//! the flat and move-split encodings (it reasons purely in `PlacedPiece`s);
//! `DruidHeuristic<M>` just asks the encoding to turn each of its actions
//! into a score.
//
// Cameron Browne's guidance (quoted at the bottom of this crate) is specific:
// bias playouts toward (1) blocking a threat to your own piece, (2)
// defending a fork/virtual connection, and (3) threatening the opponent's
// best connection, each "with high probability" rather than deterministically
// -- a fixed heuristic is exploitable, so the randomness matters as much as
// the bias.

use std::marker::PhantomData;

use rand::rngs::SmallRng;
use rustc_hash::FxHashMap as HashMap;
use rustc_hash::FxHashSet as HashSet;

use mcts::strategies::mcts::{
    backprop, select,
    simulate::{self, SimulateStrategy},
    Strategy, TreeStats,
};
use mcts::util::random_best;

use crate::connectivity::Connectivity;
use crate::game::{DruidGame, HashedState};
use crate::moves::{MoveEncoding, Split};
use crate::state::State;
use crate::types::{Piece, PlacedPiece, Player, Pos};
use mcts::game::PlayerIndex;

/// Per-heuristic weights, combined as a weighted sum (a move can satisfy more
/// than one heuristic at once, and should score higher for it) rather than a
/// priority/first-applicable scheme -- simpler to reason about and to tune.
/// Defaults weight blocking/forking (tactically forced, per Browne's own
/// ordering) above the more strategic "threaten their connection".
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DruidHeuristicWeights {
    pub block_threat: f64,
    pub defend_fork: f64,
    pub threaten_connection: f64,
}

impl Default for DruidHeuristicWeights {
    fn default() -> Self {
        Self {
            block_threat: 3.0,
            defend_fork: 3.0,
            threaten_connection: 1.0,
        }
    }
}

/// The largest same-`color` component (by cell count) under `conn`, and how
/// far it currently reaches along `color`'s goal axis (row for Black, column
/// for White). Used to approximate Browne's "threaten the opponent's best
/// connection" / "extend your own" heuristic without a full path-probability
/// model.
fn largest_component(s: &State, conn: &Connectivity, color: Player) -> (HashSet<usize>, u8) {
    let mut groups: HashMap<usize, Vec<usize>> = HashMap::default();
    for i in 0..s.size.area() as usize {
        if s.board[i].matches(color) {
            let r = conn.set(color).find(i);
            groups.entry(r).or_default().push(i);
        }
    }
    match groups.into_values().max_by_key(|v| v.len()) {
        Some(members) => {
            let axis = |i: usize| -> u8 {
                let Pos(x, y) = Pos::from(i, s.size);
                match color {
                    Player::Black => y,
                    Player::White => x,
                }
            };
            let max_axis = members.iter().map(|&i| axis(i)).max().unwrap_or(0);
            (members.into_iter().collect(), max_axis)
        }
        None => (HashSet::default(), 0),
    }
}

/// Per-candidate-move heuristic score for `mover` to move, higher is better.
/// Three detectors, matched to Browne's three heuristics:
///
/// 1. **Block a threat.** `threatened_cells` is every cell of `mover`'s that
///    a currently-legal opponent lintel placement would repaint -- a lintel
///    only needs 2 of its 3 touched cells to already carry the placer's
///    color (`State::moves`), so the third can be a cell `mover` already
///    built on. A candidate move that touches one of those cells breaks the
///    pattern (raises the cell's height past the opponent's `h[0] == h[2]`
///    window, or repaints it first).
/// 2. **Defend a fork.** Group currently-available lintel moves by the pair
///    of `mover`-color component roots they'd merge; if two or more distinct
///    moves would complete the *same* connection, each is a "save" for that
///    fork (either one alone secures it).
/// 3. **Threaten the opponent's connection.** Union-find proxy for Browne's
///    path-probability fitness: prefer touching a cell that extends `mover`'s
///    largest component past its current reach toward the far border, or
///    that's part of `opponent`'s largest component (repainting it via a
///    lintel deletes a node from their biggest group -- the same mechanic
///    `Connectivity::update`'s rebuild-on-repaint already relies on).
pub(crate) fn heuristic_scores(
    state: &HashedState,
    mover: Player,
    available: &[PlacedPiece],
    weights: &DruidHeuristicWeights,
) -> Vec<f64> {
    let s = &state.0;
    let opponent = match mover {
        Player::Black => Player::White,
        Player::White => Player::Black,
    };

    let mut threatened_cells: HashSet<usize> = HashSet::default();
    if s.hand(opponent).lintels > 0 {
        for (_, cells) in s.lintel_candidates_for(opponent) {
            for c in cells {
                if s.at(c) == Some(mover) {
                    threatened_cells.insert(c);
                }
            }
        }
    }

    let mut fork_targets: HashMap<(usize, usize), Vec<usize>> = HashMap::default();
    for (idx, m) in available.iter().enumerate() {
        if !matches!(m.0, Piece::Lintel(_)) {
            continue;
        }
        let (cells, n) = s.move_cells(*m);
        let mut roots = Vec::new();
        for &c in &cells[..n] {
            if s.at(c) == Some(mover) {
                let r = state.2.set(mover).find(c);
                if !roots.contains(&r) {
                    roots.push(r);
                }
            }
        }
        if roots.len() == 2 {
            let key = (roots[0].min(roots[1]), roots[0].max(roots[1]));
            fork_targets.entry(key).or_default().push(idx);
        }
    }
    let mut is_fork_move = vec![false; available.len()];
    for idxs in fork_targets.values() {
        if idxs.len() >= 2 {
            for &i in idxs {
                is_fork_move[i] = true;
            }
        }
    }

    let (my_members, my_max_axis) = largest_component(s, &state.2, mover);
    let mut advance_cells: HashSet<usize> = HashSet::default();
    for &i in &my_members {
        for adj in Pos::from(i, s.size).adjacent(s.size) {
            let j = adj.index(s.size.w);
            let axis_j = match mover {
                Player::Black => Pos::from(j, s.size).1,
                Player::White => Pos::from(j, s.size).0,
            };
            if axis_j > my_max_axis {
                advance_cells.insert(j);
            }
        }
    }
    let (opp_members, _) = largest_component(s, &state.2, opponent);

    available
        .iter()
        .enumerate()
        .map(|(idx, m)| {
            let (cells, n) = s.move_cells(*m);
            let touched = &cells[..n];

            let mut score = 0.0;
            if touched.iter().any(|c| threatened_cells.contains(c)) {
                score += weights.block_threat;
            }
            if is_fork_move[idx] {
                score += weights.defend_fork;
            }
            if touched
                .iter()
                .any(|c| advance_cells.contains(c) || opp_members.contains(c))
            {
                score += weights.threaten_connection;
            }
            score
        })
        .collect()
}

/// Max `heuristic_scores` value over the placements of `piece` on `cells`.
pub(crate) fn max_heuristic_for_cells(
    state: &HashedState,
    mover: Player,
    piece: Piece,
    cells: &[usize],
    weights: &DruidHeuristicWeights,
) -> f64 {
    if cells.is_empty() {
        return 0.0;
    }
    let placed: Vec<PlacedPiece> = cells.iter().map(|&c| PlacedPiece(piece, c as u8)).collect();
    let scores = heuristic_scores(state, mover, &placed, weights);
    scores.into_iter().fold(f64::NEG_INFINITY, f64::max)
}

/// `SimulateStrategy<DruidGame<M>>` driven by `heuristic_scores`: picks among the
/// max-scoring candidate moves (ties broken randomly by `random_best`, same
/// as `simulate::Mast`). On its own this only ever *narrows* the choice --
/// when no heuristic condition fires every move scores 0 and it degrades to
/// uniform-random, but when one does fire it always takes it. Browne's "high
/// probability, not always" warning (a deterministic heuristic playout is
/// exploitable) is the caller's job to satisfy, by wrapping this
/// in `simulate::EpsilonGreedy` rather than using it bare.
#[derive(Clone, Copy, Debug, Default)]
pub struct DruidHeuristic<M: MoveEncoding = Split> {
    pub weights: DruidHeuristicWeights,
    _marker: PhantomData<M>,
}

impl<M: MoveEncoding> DruidHeuristic<M> {
    pub fn new(weights: DruidHeuristicWeights) -> Self {
        Self {
            weights,
            _marker: PhantomData,
        }
    }
}

impl<M: MoveEncoding> SimulateStrategy<DruidGame<M>> for DruidHeuristic<M> {
    #[allow(clippy::too_many_arguments)]
    fn select_move<'a>(
        &mut self,
        state: &<DruidGame<M> as mcts::game::Game>::S,
        available: &'a [<DruidGame<M> as mcts::game::Game>::A],
        _stats: &TreeStats<DruidGame<M>>,
        player: usize,
        _prev_action: Option<&<DruidGame<M> as mcts::game::Game>::A>,
        _own_prev_action: Option<&<DruidGame<M> as mcts::game::Game>::A>,
        rng: &mut SmallRng,
    ) -> &'a <DruidGame<M> as mcts::game::Game>::A {
        let mover = if player == Player::Black.to_index() {
            Player::Black
        } else {
            Player::White
        };
        let scores: Vec<f64> = available
            .iter()
            .map(|m| M::score_action(state, mover, m, &self.weights))
            .collect();
        let scored: Vec<(f64, &<DruidGame<M> as mcts::game::Game>::A)> =
            scores.into_iter().zip(available.iter()).collect();
        random_best(&scored, rng, |(score, _)| *score).unwrap().1
    }
}

/// Pairs `DruidHeuristic`-guided playouts (wrapped in `DecisiveMove` +
/// `EpsilonGreedy`, same nesting Strong/Master's `RaveMastDm` uses for
/// `Mast`) with `RaveMastDm`'s exact select/backprop/final-action
/// configuration (`select::Rave`, `backprop::Classic`, `select::RobustChild`)
/// -- so a search built from this type differs from the already-tuner tuned
/// `RaveMastDm` config in server/main.rs's Strong/Master presets *only* in
/// playout policy (`DruidHeuristic` in place of `Mast`), keeping the
/// validation isolated to exactly that one change.
#[derive(Clone, Copy, Default)]
pub struct RaveDecisiveHeuristic<M: MoveEncoding = Split>(PhantomData<M>);

impl<M: MoveEncoding> Strategy<DruidGame<M>> for RaveDecisiveHeuristic<M> {
    type Select = select::Rave;
    type Simulate = simulate::DecisiveMove<
        DruidGame<M>,
        simulate::EpsilonGreedy<DruidGame<M>, DruidHeuristic<M>>,
    >;
    type Backprop = backprop::Classic;
    type FinalAction = select::RobustChild;

    fn friendly_name() -> String {
        "rave+decisive+druid_heuristic".into()
    }
}
