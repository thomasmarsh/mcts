use super::*;

#[derive(Default, Clone)]
pub struct Classic;

impl BackpropPolicy for Classic {
    fn label(&self) -> String {
        "classic".into()
    }
}
