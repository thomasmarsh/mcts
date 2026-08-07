pub mod flat_mc;
pub mod human;
pub mod mcts;
pub mod random;

use crate::game::Game;

pub trait Search: Sync + Send {
    type G: Game;

    fn friendly_name(&self) -> String;

    fn choose_action(&mut self, state: &<Self::G as Game>::S) -> <Self::G as Game>::A;

    fn principle_variation(&self) -> Vec<<Self::G as Game>::A> {
        vec![]
    }

    fn estimated_depth(&self) -> usize {
        0
    }

    /// Number of nodes held in this search's arena, for callers that only
    /// have a type-erased `Box<dyn Search>` and want to observe tree reuse
    /// (`mcts::SearchConfig::reuse_tree`) without downcasting. `0` for every
    /// strategy that doesn't keep a persistent arena.
    fn arena_len(&self) -> usize {
        0
    }

    fn set_friendly_name(&mut self, name: &str);

    #[allow(unused_variables)]
    fn make_book_entry(
        &mut self,
        state: &<Self::G as Game>::S,
    ) -> (Vec<<Self::G as Game>::A>, Vec<f64>) {
        unimplemented!();
    }
}

#[cfg(test)]
static PARALLEL_TEST_MUTEX: std::sync::OnceLock<std::sync::Mutex<()>> =
    std::sync::OnceLock::new();

#[cfg(test)]
fn parallel_test_guard() -> std::sync::MutexGuard<'static, ()> {
    PARALLEL_TEST_MUTEX
        .get_or_init(|| std::sync::Mutex::new(()))
        .lock()
        .unwrap()
}

#[cfg(test)]
mod tests;
