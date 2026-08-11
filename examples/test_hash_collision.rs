// Standalone reproduction of a Druid transposition-table corruption, with
// no MCTS tree involved at all: just random legal play, hashing every
// state visited.
//
// `Druid::zobrist_hash` covers only board cells + `pending`, not
// `player` or hand counts. That's normally fine for games where the board
// alone determines total-turns-played parity. But a lintel placement sets
// all 3 touched cells to `height(cells[0]) + 1` in one turn, regardless of
// the other two cells' prior heights (only `h[1] <= h[0]` is required) --
// so a single lintel turn can raise a cell's height by more than 1. That
// means the number of turns needed to reach a given board is *not*
// recoverable from the board alone: the same visual board can be built via
// fewer turns (leaning on lintels to "level up" unevenly-prepped cells) or
// more turns (building the same heights via individual sarsens). Since
// turn-count parity determines whose turn it is and hand counts stay
// unhashed too, two genuinely different, both-legally-reachable states can
// collide on hash while disagreeing on player-to-move and/or available
// actions -- exactly the corruption the MCTS run observed.
//
// This script plays many random games, hashing every (pending-boundary or
// not) state reached, and reports the first pair of visited states that
// share a hash but disagree on player-to-move or action count.

use mcts::game::Game;
use mcts::game::PlayerIndex;
use game_druid::{Druid, HashedState, Size};
use rand::rngs::SmallRng;
use rand::seq::SliceRandom;
use rand::SeedableRng;
use rustc_hash::FxHashMap;

fn main() {
    let size = Size { w: 5, h: 5 };
    let mut rng = SmallRng::seed_from_u64(1);

    // hash -> (player index, sorted action debug strings, full state debug, visited count)
    let mut seen: FxHashMap<u64, (usize, Vec<String>, String)> = FxHashMap::default();

    let mut games = 0usize;
    let mut states_visited = 0usize;

    for game in 0..200_000u64 {
        games = game as usize + 1;
        let mut state = HashedState::new(size);
        for _ply in 0..400 {
            if Druid::is_terminal(&state) {
                break;
            }
            let mut actions = Vec::new();
            Druid::generate_actions(&state, &mut actions);
            if actions.is_empty() {
                break;
            }

            let hash = Druid::zobrist_hash(&state);
            let player = Druid::player_to_move(&state).to_index();
            let mut action_strs: Vec<String> = actions.iter().map(|a| format!("{a:?}")).collect();
            action_strs.sort();
            let state_dbg = format!("{state:?}");
            states_visited += 1;

            match seen.get(&hash) {
                None => {
                    seen.insert(hash, (player, action_strs, state_dbg));
                }
                Some((prev_player, prev_actions, prev_state_dbg)) => {
                    if *prev_player != player || *prev_actions != action_strs {
                        println!("COLLISION FOUND after {games} games, {states_visited} states visited");
                        println!("hash = {hash:016x}");
                        println!("\n--- first-seen state (player={prev_player}) ---\n{prev_state_dbg}");
                        println!("actions ({}): {:?}", prev_actions.len(), prev_actions);
                        println!("\n--- new state (player={player}) ---\n{state_dbg}");
                        println!("actions ({}): {:?}", action_strs.len(), action_strs);
                        return;
                    }
                }
            }

            let action = *actions.choose(&mut rng).unwrap();
            state = Druid::apply(state, &action);
        }
    }

    println!(
        "No collision found after {games} games, {states_visited} states visited, {} distinct hashes",
        seen.len()
    );
}
