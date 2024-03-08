use crate::game::{Game, ZobristHash};

use super::{index, table::TranspositionTable, Strategy, TreeIndex, TreeSearch};

pub fn render<G: Game, S: Strategy<G>>(search: &TreeSearch<G, S>)
where
    G::S: NodeRender,
{
    print::<G>(&search.index, search.root_id);
}

pub fn render_trans<G, S>(search: &TreeSearch<G, S>, state: &G::S)
where
    G: Game,
    S: Strategy<G>,
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

fn canonical_id<G: Game>(
    k: ZobristHash<G::K>,
    table: &TranspositionTable<G>,
    state: G::S,
) -> Option<index::Id> {
    table.get_const(&k.hash, state).map(|ts| ts.node_id)
}

fn print_trans<G>(
    index: &TreeIndex<G::A, G::K>,
    table: &TranspositionTable<G>,
    root_id: index::Id,
    init_state: G::S,
) where
    G: Game,
    G::S: NodeRender,
{
    println!("graph {{");
    println!("  graph [ranksep=3, ratio=auto, concentrate=true, bgcolor=black];");
    println!("  edge [color=white];");
    println!("{}", G::S::preamble());
    let mut stack = vec![(root_id, root_id, root_id, init_state.clone())];
    while let Some((parent_id, parent_print_id, node_id, state)) = stack.pop() {
        let hash = G::zobrist_hash(&state);
        let print_id = canonical_id(hash, table, state.clone()).unwrap_or(root_id);
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

fn print<G>(index: &TreeIndex<G::A, G::K>, root_id: index::Id)
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
