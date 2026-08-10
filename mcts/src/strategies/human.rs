use std::io;
use std::io::Write;
use std::marker::PhantomData;

use crate::{game::Game, strategies::Search};

pub struct HumanAgent<G: Game> {
    name: String,
    marker: PhantomData<G>,
}

impl<G: Game> Default for HumanAgent<G> {
    fn default() -> Self {
        Self::new()
    }
}

impl<G: Game> HumanAgent<G> {
    pub fn new() -> Self {
        Self {
            name: "human".into(),
            marker: PhantomData,
        }
    }
}

impl<G: Game> Search for HumanAgent<G>
where
    G::S: std::fmt::Display,
{
    type G = G;

    fn choose_action(&mut self, state: &<Self::G as Game>::S) -> <Self::G as Game>::A {
        print!("State is:\n{}", state);
        let mut input = String::new();
        loop {
            input.clear();
            print!("> ");
            io::stdout().flush().expect("Failed to flush stdout");
            match io::stdin().read_line(&mut input) {
                Ok(_) => match G::parse_action(state, input.as_str()) {
                    None => eprintln!("Error parsing input: >{}<", input.trim()),
                    Some(action) => return action,
                },
                Err(error) => {
                    eprintln!("Error reading input: {}", error);
                }
            }
        }
    }

    fn friendly_name(&self) -> String {
        self.name.clone()
    }

    fn set_friendly_name(&mut self, name: &str) {
        self.name = name.to_string();
    }
}
