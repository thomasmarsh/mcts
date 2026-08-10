use crate::game::Game;

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
    println!("  graph [ranksep=3, ratio=auto, concentrate=true, bgcolor=black];");
    println!("  edge [color=white];");
    println!("{}", G::S::preamble());
    let mut stack = vec![(root_id, root_id, root_id, init_state.clone())];
    while let Some((parent_id, parent_print_id, node_id, state)) = stack.pop() {
        let hash = G::zobrist_hash(&state);
        let print_id = table.get_const(hash).unwrap_or(root_id);
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
            let children = node.children();
            for i in (0..children.len()).filter(|&i| children.is_explored(i)) {
                stack.push((
                    node_id,
                    print_id,
                    children.node_id(i).unwrap(),
                    G::apply(state.clone(), children.action(i)),
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
            let children = node.children();
            for i in (0..children.len()).filter(|&i| children.is_explored(i)) {
                stack.push((
                    node_id,
                    children.node_id(i).unwrap(),
                    G::apply(state.clone(), children.action(i)),
                ));
            }
        }
    }
    println!("}}");
}
