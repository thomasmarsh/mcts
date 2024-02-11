use serde::Serialize;

#[derive(Clone, Copy, Debug, PartialEq, Serialize)]
pub struct Id(usize);

impl Id {
    pub fn invalid_id() -> Id {
        Id(usize::MAX)
    }
}

// TODO: benchmark keeping child/sibling relationships here vs. on Node (space vs. time)
#[derive(Debug, Serialize)]
struct Entry<T: Serialize> {
    value: T,
}

#[derive(Default, Debug, Serialize)]
pub struct Arena<T: Serialize>(Vec<Entry<T>>);

impl<T: Serialize> Arena<T> {
    pub fn new() -> Self {
        Self(Vec::new())
    }

    pub fn clear(&mut self) {
        self.0.clear();
    }
    pub fn insert(&mut self, value: T) -> Id {
        let id = self.0.len();
        self.0.push(Entry { value });
        Id(id)
    }

    pub fn get(&self, id: Id) -> &T {
        &self.0.get(id.0).unwrap().value
    }

    pub fn get_mut(&mut self, id: Id) -> &mut T {
        &mut self.0[id.0].value
    }

    pub fn len(&self) -> usize {
        self.0.len()
    }
}
