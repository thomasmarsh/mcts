use crate::game::Game;

use super::{index, Strategy, TreeIndex, TreeSearch};

pub fn render<G: Game, S: Strategy<G>>(search: &TreeSearch<G, S>)
where
    G::S: NodeRender,
{
    print::<G>(&search.index, search.root_id);
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

fn print<G>(index: &TreeIndex<G::A>, root_id: index::Id)
where
    G: Game,
    G::S: NodeRender,
{
    println!("graph {{");
    println!("  graph [layout=twopi, ranksep=3, ratio=auto];");
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
            for (child_id, action) in node
                .children()
                .iter()
                .zip(node.actions())
                .filter(|x| x.0.is_some())
            {
                stack.push((node_id, child_id.unwrap(), G::apply(state.clone(), action)));
            }
        }
    }
    println!("}}");
}
