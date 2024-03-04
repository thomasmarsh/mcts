use crate::{game::Game, zobrist::ZobristHash};

use super::{index, table::TranspositionTable, Strategy, TreeIndex, TreeSearch};

pub fn render<G: Game, S: Strategy<G>>(search: &TreeSearch<G, S>)
where
    G::S: NodeRender,
{
    print::<G>(&search.index, search.root_id);
}

pub fn render_trans<G: Game, S: Strategy<G>>(search: &TreeSearch<G, S>, state: &G::S)
where
    G::S: NodeRender,
{
    print_trans::<G>(&search.index, &search.table, search.root_id, state.clone());
}

pub trait NodeRender {
    fn preamble() -> String {
        "  node [shape=point];".into()
    }

    fn render(&self) -> String {
        "".into()
    }
}

//////////////////////////////////////////////////////////////////////////////////////////////////

fn canonical_id(k: u64, table: &TranspositionTable) -> Option<index::Id> {
    table
        .table
        .0
        .get(&ZobristHash(k))
        .map(|ts| *ts.iter().min_by_key(|x| x.get_raw()).unwrap())
}

fn print_trans<G>(
    index: &TreeIndex<G::A>,
    table: &TranspositionTable,
    root_id: index::Id,
    init_state: G::S,
) where
    G: Game,
    G::S: NodeRender,
{
    println!("graph {{");
    println!("  graph [layout=twopi, ranksep=3, ratio=auto, concentrate=true, bgcolor=black];");
    println!("  edge [color=white];");
    println!("{}", G::S::preamble());
    let mut stack = vec![(root_id, root_id, root_id, init_state)];
    while let Some((parent_id, parent_print_id, node_id, state)) = stack.pop() {
        let hash = G::zobrist_hash(&state);
        let print_id = canonical_id(hash, table).unwrap_or(root_id);
        println!("  \"{}\" {};", print_id.get_raw(), state.render());
        if parent_id != node_id {
            println!(
                "  \"{}\" -- \"{}\";",
                parent_print_id.get_raw(),
                print_id.get_raw()
            );
        }
        let node = index.get(node_id);
        if node.is_expanded() {
            for edge in node.edges().iter().filter(|edge| edge.is_explored()) {
                stack.push((
                    node_id,
                    print_id,
                    edge.node_id.unwrap(),
                    G::apply(state.clone(), &edge.action),
                ));
            }
        }
    }
    println!("}}");
}

fn print<G>(index: &TreeIndex<G::A>, root_id: index::Id)
where
    G: Game,
    G::S: NodeRender,
{
    println!("graph {{");
    println!("  graph [layout=twopi, ranksep=3, ratio=auto, bgcolor=black];");
    println!("  edge [color=white];");
    println!("{}", G::S::preamble());
    let mut stack = vec![(root_id, root_id, G::S::default())];
    while let Some((parent_id, node_id, state)) = stack.pop() {
        println!("  \"{}\" {};", node_id.get_raw(), state.render());
        if parent_id != node_id {
            println!(
                "  \"{}\" -- \"{}\";",
                parent_id.get_raw(),
                node_id.get_raw()
            );
        }
        let node = index.get(node_id);
        if node.is_expanded() {
            for edge in node.edges().iter().filter(|x| x.is_explored()) {
                stack.push((
                    node_id,
                    edge.node_id.unwrap(),
                    G::apply(state.clone(), &edge.action),
                ));
            }
        }
    }
    println!("}}");
}
