//! Exact solve of small Gonnect boards by retrograde analysis over the explicit
//! reachable game graph.
//!
//! ```text
//! LIBRARY_PATH=/opt/homebrew/lib cargo run --release --example solve_small \
//!     -p game-gonnect -- <size> [--max-states N] [--out FILE.jsonl] [--table FILE.bin]
//! ```
//!
//! Plain alpha-beta with a transposition table is unsound here: the ko rule is
//! positional against only the previous position, so longer repetition cycles
//! exist and a search that meets a repeated position has no value to return.
//! Retrograde analysis needs no cycle handling: terminal states are known, a
//! state is won for its mover if any successor is, lost if every successor is
//! won for the opponent, and whatever never resolves is a draw by endless play.
//!
//! State identity is `(black, white, turn, can_swap, winner, legal-move mask)`.
//! The ko snapshot only ever affects which placements are legal right now (it
//! is overwritten by the next placement, and the swap window opens only while
//! the snapshot is the empty board), so two states with the same key play
//! identically. Nodes are discovered breadth first and their successors are
//! stored as compact ids; only the current frontier holds full `State` values.

use std::collections::{HashMap, VecDeque};
use std::io::Write;
use std::time::Instant;

use game_gonnect::{Gonnect, Move, Player, State};
use mcts::game::Game;

type Key = u128;

const UNRESOLVED: u8 = 0;
const BLACK: u8 = 1;
const WHITE: u8 = 2;

fn word(b: bitboard::Board<[u64; 6], bitboard::Dyn, bitboard::Dyn>) -> u64 {
    b.words().next().unwrap_or(0)
}

fn key(state: &State, legal: &[Move], n: usize) -> Key {
    let cells = n * n;
    let mut mask = 0u64;
    for m in legal {
        let i = m.index();
        let bit = if *m == Move::SWAP {
            cells
        } else if *m == Move::NO_MOVE {
            cells + 1
        } else {
            i as usize
        };
        mask |= 1u64 << bit;
    }
    let flags = (state.turn() == Player::White) as u128
        | (state.can_swap() as u128) << 1
        | (state.has_winner() as u128) << 2;
    word(state.black()) as u128
        | (word(state.white()) as u128) << 25
        | (mask as u128) << 50
        | flags << 80
}

fn player_id(p: Player) -> u8 {
    if p == Player::Black {
        BLACK
    } else {
        WHITE
    }
}

struct Args {
    size: usize,
    max_states: usize,
    out: Option<String>,
    table: Option<String>,
}

fn parse_args() -> Args {
    let mut a = Args {
        size: 0,
        max_states: usize::MAX,
        out: None,
        table: None,
    };
    let mut it = std::env::args().skip(1);
    while let Some(arg) = it.next() {
        match arg.as_str() {
            "--max-states" => a.max_states = it.next().unwrap().parse().unwrap(),
            "--out" => a.out = it.next(),
            "--table" => a.table = it.next(),
            s => a.size = s.parse().expect("first argument is the board size"),
        }
    }
    assert!((2..=5).contains(&a.size), "size must be 2..=5 (25-bit board words)");
    a
}

fn main() {
    let args = parse_args();
    let n = args.size;
    let t0 = Instant::now();

    let mut ids: HashMap<Key, u32> = HashMap::new();
    let mut keys: Vec<Key> = Vec::new();
    let mut player: Vec<u8> = Vec::new();
    let mut terminal: Vec<bool> = Vec::new();
    let mut edge_start: Vec<u64> = vec![0];
    let mut edges: Vec<u32> = Vec::new();
    let mut pending: VecDeque<State> = VecDeque::new();
    let mut peak_pending = 0usize;

    let root = State::new(n);
    let mut root_legal = Vec::new();
    Gonnect::generate_actions(&root, &mut root_legal);
    ids.insert(key(&root, &root_legal, n), 0);
    keys.push(key(&root, &root_legal, n));
    pending.push_back(root);

    let mut truncated = false;
    let mut actions: Vec<Move> = Vec::new();
    let mut successors = 0u64;
    let mut done_terminal_edges = 0u64;
    while let Some(state) = pending.pop_front() {
        let is_term = Gonnect::is_terminal(&state);
        player.push(player_id(state.turn()));
        terminal.push(is_term);
        if !is_term {
            actions.clear();
            Gonnect::generate_actions(&state, &mut actions);
            for a in &actions {
                let next = Gonnect::apply(state.clone(), a);
                let nk = if Gonnect::is_terminal(&next) {
                    key(&next, &[], n)
                } else {
                    let mut legal = Vec::new();
                    Gonnect::generate_actions(&next, &mut legal);
                    key(&next, &legal, n)
                };
                let id = match ids.get(&nk) {
                    Some(&id) => id,
                    None => {
                        let id = keys.len() as u32;
                        ids.insert(nk, id);
                        keys.push(nk);
                        pending.push_back(next);
                        id
                    }
                };
                edges.push(id);
                successors += 1;
            }
        } else {
            done_terminal_edges += 1;
        }
        edge_start.push(edges.len() as u64);
        peak_pending = peak_pending.max(pending.len());
        let processed = edge_start.len() - 1;
        if processed.is_multiple_of(1_000_000) {
            eprintln!(
                "  {processed} expanded, {} discovered, {:.0}s, {:.0} states/s",
                keys.len(),
                t0.elapsed().as_secs_f64(),
                processed as f64 / t0.elapsed().as_secs_f64()
            );
        }
        if keys.len() > args.max_states {
            truncated = true;
            break;
        }
    }
    let build_secs = t0.elapsed().as_secs_f64();
    let expanded = edge_start.len() - 1;
    println!(
        "size {n}: {} states discovered, {expanded} expanded, {} terminal, {successors} edges, \
         peak frontier {peak_pending}, {:.1}s ({:.0} expanded/s){}",
        keys.len(),
        done_terminal_edges,
        build_secs,
        expanded as f64 / build_secs,
        if truncated { "  TRUNCATED at --max-states, no solve" } else { "" }
    );
    if truncated {
        if let Some(path) = &args.out {
            log(path, serde_json::json!({
                "type": "solve", "size": n, "truncated": true, "discovered": keys.len(),
                "expanded": expanded, "secs": build_secs,
            }));
        }
        return;
    }

    // Retrograde sweeps to a fixed point. `res[s]` is the winner's colour once known.
    let total = keys.len();
    let mut res = vec![UNRESOLVED; total];
    for s in 0..total {
        if terminal[s] {
            res[s] = player[s];
        }
    }
    let mut sweeps = 0;
    loop {
        sweeps += 1;
        let mut changed = 0usize;
        for s in (0..total).rev() {
            if res[s] != UNRESOLVED {
                continue;
            }
            let (lo, hi) = (edge_start[s] as usize, edge_start[s + 1] as usize);
            let me = player[s];
            let other = 3 - me;
            let mut all_other = true;
            let mut any_mine = false;
            for &c in &edges[lo..hi] {
                let r = res[c as usize];
                if r == me {
                    any_mine = true;
                    break;
                }
                if r != other {
                    all_other = false;
                }
            }
            if any_mine {
                res[s] = me;
                changed += 1;
            } else if all_other && hi > lo {
                res[s] = other;
                changed += 1;
            }
        }
        if changed == 0 {
            break;
        }
    }
    let (mut b, mut w, mut u) = (0usize, 0usize, 0usize);
    for &r in &res {
        match r {
            BLACK => b += 1,
            WHITE => w += 1,
            _ => u += 1,
        }
    }
    let root_res = res[0];
    println!(
        "solved in {:.1}s ({sweeps} sweeps): black wins {b}, white wins {w}, unresolved {u}; \
         initial position: {}",
        t0.elapsed().as_secs_f64(),
        match root_res {
            BLACK => "Black (first player) wins with best play",
            WHITE => "White (second player, with the swap option) wins with best play",
            _ => "unresolved (endless play)",
        }
    );

    // Value of each opening placement for White (the swap decision).
    let start = State::new(n);
    let mut acts = Vec::new();
    Gonnect::generate_actions(&start, &mut acts);
    let mut opening = Vec::new();
    for a in &acts {
        let next = Gonnect::apply(start.clone(), a);
        let mut legal = Vec::new();
        Gonnect::generate_actions(&next, &mut legal);
        let k = key(&next, &legal, n);
        let r = res[ids[&k] as usize];
        opening.push((Gonnect::notation(&start, a), r));
    }
    let black_wins_openings: Vec<&str> = opening
        .iter()
        .filter(|(_, r)| *r == BLACK)
        .map(|(m, _)| m.as_str())
        .collect();
    println!(
        "openings where Black wins even after White's best reply (incl. swap): {} of {}: {:?}",
        black_wins_openings.len(),
        opening.len(),
        black_wins_openings
    );

    if let Some(path) = &args.out {
        log(
            path,
            serde_json::json!({
                "type": "solve", "size": n, "truncated": false, "states": total,
                "expanded": expanded, "terminal": done_terminal_edges, "edges": successors,
                "black_wins": b, "white_wins": w, "unresolved": u, "sweeps": sweeps,
                "root_result": match root_res { BLACK => "black", WHITE => "white", _ => "unresolved" },
                "black_winning_openings": black_wins_openings, "n_openings": opening.len(),
                "build_secs": build_secs, "total_secs": t0.elapsed().as_secs_f64(),
                "peak_frontier": peak_pending,
            }),
        );
    }
    if let Some(path) = &args.table {
        // Sorted (key, result) records: 16 key bytes + 1 result byte, for oracle lookups.
        let mut order: Vec<usize> = (0..total).collect();
        order.sort_unstable_by_key(|&i| keys[i]);
        let mut f = std::io::BufWriter::new(std::fs::File::create(path).unwrap());
        for i in order {
            f.write_all(&keys[i].to_le_bytes()).unwrap();
            f.write_all(&[res[i]]).unwrap();
        }
        println!("table written to {path}");
    }
}

fn log(path: &str, row: serde_json::Value) {
    let mut f = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)
        .unwrap_or_else(|e| panic!("cannot open {path}: {e}"));
    writeln!(f, "{row}").unwrap();
}
