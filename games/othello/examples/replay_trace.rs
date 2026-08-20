use std::env;
use std::fs;

use game_othello::{Move, Othello, Player, State, BB as BitBoard};
use mcts::game::Game;
use serde::Deserialize;
use serde_json::Value;

#[derive(Deserialize)]
struct NodeState {
    black: u64,
    white: u64,
    turn: String,
    last_pass: bool,
}

#[derive(Deserialize)]
struct Node {
    state: NodeState,
    #[serde(rename = "move")]
    mv: Option<u64>,
    #[serde(rename = "childIds")]
    child_ids: Vec<String>,
}

fn parse_player(s: &str) -> Player {
    match s {
        "Black" => Player::Black,
        "White" => Player::White,
        _ => panic!("bad player"),
    }
}

fn to_state(n: &NodeState) -> State {
    State {
        black: BitBoard::from_bits(n.black),
        white: BitBoard::from_bits(n.white),
        turn: parse_player(&n.turn),
        last_pass: n.last_pass,
        hashes: [0u64; 8],
    }
}

fn main() {
    let path = env::args().nth(1).expect("usage: replay_trace <path.json>");
    let raw = fs::read_to_string(path).unwrap();
    let v: Value = serde_json::from_str(&raw).unwrap();
    let tree = &v["tree"];
    let nodes_v = &tree["nodes"];
    let nodes: std::collections::HashMap<String, Node> =
        serde_json::from_value(nodes_v.clone()).unwrap();
    let root_id = tree["rootId"].as_str().unwrap().to_string();

    let mut order = vec![];
    let mut nid = Some(root_id);
    while let Some(id) = nid {
        let node = &nodes[&id];
        order.push(id.clone());
        nid = node.child_ids.first().cloned();
    }

    let mut prev_state: Option<State> = None;
    let mut mismatches = 0;
    let mut unreachable = 0;
    let mut first_mismatch_printed = false;
    for (i, id) in order.iter().enumerate() {
        let node = &nodes[id];
        let state = to_state(&node.state);
        if let Some(ref prev) = prev_state {
            if let Some(mv) = node.mv {
                let action = if mv == 64 { None } else { Some(Move(mv as u8)) };
                let mut legal = Vec::new();
                Othello::generate_actions(prev, &mut legal);
                let legal_idxs: Vec<u8> = legal.iter().map(|m| m.0).collect();
                let ok = match action {
                    Some(a) => legal.contains(&a),
                    None => legal_idxs.contains(&64),
                };
                if !ok {
                    println!(
                        "ILLEGAL at step {i} (node {id}): move={mv} not in legal={legal_idxs:?}"
                    );
                    println!(
                        "  prev: black={:#018x} white={:#018x} turn={:?}",
                        prev.black.bits(),
                        prev.white.bits(),
                        prev.turn
                    );
                    println!(
                        "  new : black={:#018x} white={:#018x} turn={:?}",
                        state.black.bits(),
                        state.white.bits(),
                        state.turn
                    );
                } else if let Some(a) = action {
                    let computed = Othello::apply(*prev, &a);
                    let matches_declared_move = computed.black.bits() == state.black.bits()
                        && computed.white.bits() == state.white.bits()
                        && computed.turn == state.turn;
                    if !matches_declared_move {
                        mismatches += 1;
                        // Is the recorded next-state reachable via *any*
                        // legal move from prev, not just the declared one?
                        let reachable_via_some_move = legal.iter().any(|m| {
                            let c = Othello::apply(*prev, m);
                            c.black.bits() == state.black.bits()
                                && c.white.bits() == state.white.bits()
                                && c.turn == state.turn
                        });
                        if !reachable_via_some_move {
                            unreachable += 1;
                        }
                        if !first_mismatch_printed {
                            first_mismatch_printed = true;
                            println!(
                                "FIRST STATE MISMATCH at step {i} (node {id}): move={mv} legal, but recorded next-state != Othello::apply(prev, move); reachable via some other legal move: {reachable_via_some_move}"
                            );
                            println!(
                                "  prev    : black={:#018x} white={:#018x} turn={:?}",
                                prev.black.bits(),
                                prev.white.bits(),
                                prev.turn
                            );
                            println!(
                                "  recorded: black={:#018x} white={:#018x} turn={:?}",
                                state.black.bits(),
                                state.white.bits(),
                                state.turn
                            );
                            println!(
                                "  computed: black={:#018x} white={:#018x} turn={:?}",
                                computed.black.bits(),
                                computed.white.bits(),
                                computed.turn
                            );
                        }
                    }
                }
            }
        }
        prev_state = Some(state);
    }
    println!(
        "done, {} nodes, {mismatches} state mismatches ({unreachable} not reachable via any legal move from prev)",
        order.len()
    );
}
