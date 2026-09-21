//! Searches driven by a trained policy/value network (the `net_gumbel` algorithm).
//!
//! This crate knows the algorithm's parameters (which network, how many simulations, how many root
//! actions the Gumbel search considers, ...) but not how to evaluate a position: that needs the
//! game's encoding and an inference backend. A game crate implements [`NetSearchFactory`] for its
//! `Game` type and [`register`]s it once at startup; from then on `algorithm: "net_gumbel"` builds
//! through the same [`crate::build_search`] as every other algorithm, appears in the game's
//! strategy catalog ([`add_net_algorithm`]), and can be named by a preset or a custom strategy.

use std::any::{Any, TypeId};
use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock};

use game_host::{HostError, TunerInfo};
use mcts::algorithms::Search;
use mcts::game::Game;
use serde_json::{json, Value};

use crate::config_ir::codec::field_opt;
use crate::fields::{condition, param};

/// The `algorithm` value this module adds.
pub const ALGORITHM: &str = "net_gumbel";

const PARAMS: &[&str] = &[
    "net_model",
    "net_simulations",
    "net_considered_actions",
    "net_value_scale",
    "net_max_visit_init",
    "net_selection",
];

/// How the search picks its move once the simulations are spent.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NetSelection {
    /// Gumbel noise plus Sequential Halving over the considered root actions (Danihelka et al.
    /// 2022); the noise is seeded from the position, so a position always gets the same move.
    Gumbel,
    /// No noise: the most visited root action of a completed-Q search.
    MostVisited,
}

impl NetSelection {
    fn from_name(name: &str) -> Result<Self, HostError> {
        match name {
            "gumbel" => Ok(Self::Gumbel),
            "most_visited" => Ok(Self::MostVisited),
            other => Err(HostError::bad_request(format!(
                "unknown net_selection: {other}"
            ))),
        }
    }
}

/// A resolved `net_gumbel` configuration.
#[derive(Debug, Clone, PartialEq)]
pub struct NetSearchSpec {
    /// A model id the game's factory lists in [`NetSearchFactory::models`].
    pub model: String,
    /// Network evaluations spent per move (the search's iteration count).
    pub simulations: usize,
    /// Root actions Sequential Halving starts from (Gumbel selection only).
    pub considered_actions: usize,
    /// The completed-Q scale `c_scale` of the Gumbel search's sigma transform.
    pub value_scale: f64,
    /// The visit offset `c_visit` of the sigma transform.
    pub max_visit_init: u32,
    pub selection: NetSelection,
}

pub trait NetSearchFactory<G: Game>: Send + Sync {
    /// Ids of the models this game can currently load.
    fn models(&self) -> Vec<String>;

    /// A search playing `spec.model` for `spec`'s settings. Errors (unknown model, ...) are
    /// reported to the caller as a bad request. A returned search reports positions it cannot
    /// play through [`Search::unsupported_reason`].
    fn build(&self, spec: &NetSearchSpec, seed: u64) -> Result<Box<dyn Search<G = G>>, HostError>;
}

type Registry = Mutex<HashMap<TypeId, Arc<dyn Any + Send + Sync>>>;

fn registry() -> &'static Registry {
    static REGISTRY: OnceLock<Registry> = OnceLock::new();
    REGISTRY.get_or_init(Default::default)
}

/// Makes `net_gumbel` buildable for game `G`, replacing any earlier factory for it.
pub fn register<G: Game + 'static>(factory: Arc<dyn NetSearchFactory<G>>) {
    registry()
        .lock()
        .unwrap()
        .insert(TypeId::of::<G>(), Arc::new(factory));
}

fn factory<G: Game + 'static>() -> Option<Arc<dyn NetSearchFactory<G>>> {
    let entry = registry().lock().unwrap().get(&TypeId::of::<G>())?.clone();
    entry
        .downcast_ref::<Arc<dyn NetSearchFactory<G>>>()
        .cloned()
}

/// The model ids game `G` can load; empty when no factory is registered.
pub fn available_models<G: Game + 'static>() -> Vec<String> {
    factory::<G>().map(|f| f.models()).unwrap_or_default()
}

/// The model a `params` object names when it selects `net_gumbel`, `None` for any other algorithm.
pub fn model_of_params(params: &Value) -> Option<&str> {
    if params.get("algorithm").and_then(Value::as_str) != Some(ALGORITHM) {
        return None;
    }
    params.get("net_model").and_then(Value::as_str)
}

pub(crate) fn build<G: Game + 'static>(
    spec: &NetSearchSpec,
    seed: u64,
) -> Result<Box<dyn Search<G = G>>, HostError> {
    let factory = factory::<G>().ok_or_else(|| {
        HostError::bad_request(format!(
            "algorithm {ALGORITHM} is not available for this game (no network is loaded)"
        ))
    })?;
    if !factory.models().contains(&spec.model) {
        return Err(HostError::bad_request(format!(
            "unknown or unavailable net_model {:?}",
            spec.model
        )));
    }
    factory.build(spec, seed)
}

pub(crate) fn spec_from_params(cfg: &Value) -> Result<NetSearchSpec, HostError> {
    let get = |name: &str| -> Result<Value, HostError> {
        cfg.get(name)
            .cloned()
            .ok_or_else(|| HostError::bad_request(format!("missing param: {name}")))
    };
    let num = |name: &str| -> Result<f64, HostError> {
        field_opt::<f64>(cfg, name)
            .map_err(HostError::bad_request)?
            .ok_or_else(|| HostError::bad_request(format!("missing param: {name}")))
    };
    let count = |name: &str| -> Result<usize, HostError> {
        let v = num(name)?;
        if v.fract() != 0.0 || v < 1.0 {
            return Err(HostError::bad_request(format!(
                "{name} must be a positive integer"
            )));
        }
        Ok(v as usize)
    };
    let model = get("net_model")?
        .as_str()
        .ok_or_else(|| HostError::bad_request("net_model must be a string"))?
        .to_string();
    let value_scale = num("net_value_scale")?;
    if !(value_scale.is_finite() && value_scale > 0.0) {
        return Err(HostError::bad_request("net_value_scale must be positive"));
    }
    let selection = NetSelection::from_name(
        get("net_selection")?
            .as_str()
            .ok_or_else(|| HostError::bad_request("net_selection must be a string"))?,
    )?;
    Ok(NetSearchSpec {
        model,
        simulations: count("net_simulations")?,
        considered_actions: count("net_considered_actions")?,
        value_scale,
        max_visit_init: count("net_max_visit_init")? as u32,
        selection,
    })
}

/// Adds `net_gumbel` to a strategy catalog: the `algorithm` choice, its parameters, and the
/// condition that activates them. `models` are the ids [`available_models`] returned; nothing is
/// added when it is empty, so a game without a loaded network keeps its plain catalog.
pub fn add_net_algorithm(info: &mut TunerInfo, models: &[String]) {
    let Some(first) = models.first() else {
        return;
    };
    for p in &mut info.parameters {
        if p.name == "algorithm" {
            if let Some(choices) = p.spec.get_mut("choices").and_then(Value::as_array_mut) {
                choices.push(json!(ALGORITHM));
            }
        }
    }
    info.parameters.extend([
        param(
            "net_model",
            json!({"type": "categorical", "choices": models, "default": first}),
        ),
        param(
            "net_simulations",
            json!({"type": "int", "bounds": [1, 4000], "default": 100}),
        ),
        param(
            "net_considered_actions",
            json!({"type": "int", "bounds": [1, 64], "default": 16}),
        ),
        param(
            "net_value_scale",
            json!({"type": "float", "bounds": [0.01, 2.0], "default": 0.1}),
        ),
        param(
            "net_max_visit_init",
            json!({"type": "int", "bounds": [1, 200], "default": 50}),
        ),
        param(
            "net_selection",
            json!({"type": "categorical", "choices": ["gumbel", "most_visited"], "default": "gumbel"}),
        ),
    ]);
    info.conditions
        .push(condition(json!({"algorithm": ALGORITHM}), PARAMS));
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::strategy_tuner_info;
    use game_nim::Nim;
    use mcts::algorithms::random::Random;

    struct Fake;

    impl NetSearchFactory<Nim> for Fake {
        fn models(&self) -> Vec<String> {
            vec!["fake-model".into()]
        }
        fn build(
            &self,
            spec: &NetSearchSpec,
            seed: u64,
        ) -> Result<Box<dyn Search<G = Nim>>, HostError> {
            assert_eq!(spec.simulations, 7);
            Ok(Box::new(Random::<Nim>::new().with_seed(seed)))
        }
    }

    fn params() -> Value {
        json!({
            "algorithm": "net_gumbel",
            "net_model": "fake-model",
            "net_simulations": 7,
            "net_considered_actions": 4,
            "net_value_scale": 0.25,
            "net_max_visit_init": 50,
            "net_selection": "most_visited",
        })
    }

    #[test]
    fn params_resolve_to_a_spec() {
        let spec = spec_from_params(&params()).unwrap();
        assert_eq!(
            spec,
            NetSearchSpec {
                model: "fake-model".into(),
                simulations: 7,
                considered_actions: 4,
                value_scale: 0.25,
                max_visit_init: 50,
                selection: NetSelection::MostVisited,
            }
        );
        let mut bad = params();
        bad["net_simulations"] = json!(0);
        assert!(spec_from_params(&bad).is_err());
        let mut bad = params();
        bad["net_selection"] = json!("nope");
        assert!(spec_from_params(&bad).is_err());
        let mut bad = params();
        bad.as_object_mut().unwrap().remove("net_model");
        assert!(spec_from_params(&bad).is_err());
    }

    #[test]
    fn build_search_goes_through_the_registered_factory() {
        // The registry is per game type and process-wide; Nim is only registered by this test.
        let e = crate::build_search::<Nim>(&params(), 0, false, &Default::default())
            .err()
            .expect("no factory is registered yet");
        assert_eq!(e.code, 400);
        assert!(e.message.contains("not available"), "{}", e.message);
        register::<Nim>(Arc::new(Fake));
        assert_eq!(available_models::<Nim>(), vec!["fake-model".to_string()]);
        assert!(crate::build_search::<Nim>(&params(), 0, false, &Default::default()).is_ok());
        let mut other = params();
        other["net_model"] = json!("not-a-model");
        assert!(crate::build_search::<Nim>(&other, 0, false, &Default::default()).is_err());
    }

    #[test]
    fn the_catalog_gains_the_algorithm_only_when_a_model_is_listed() {
        let plain = strategy_tuner_info(&["easy"], 1);
        let mut none = plain.clone();
        add_net_algorithm(&mut none, &[]);
        assert_eq!(
            serde_json::to_value(&none).unwrap(),
            serde_json::to_value(&plain).unwrap()
        );

        let mut with = plain.clone();
        add_net_algorithm(&mut with, &["m1".to_string(), "m2".to_string()]);
        let algorithm = with
            .parameters
            .iter()
            .find(|p| p.name == "algorithm")
            .unwrap();
        assert!(algorithm.spec["choices"]
            .as_array()
            .unwrap()
            .contains(&json!(ALGORITHM)));
        for name in PARAMS {
            assert!(with.parameters.iter().any(|p| p.name == *name), "{name}");
        }
        let model = with
            .parameters
            .iter()
            .find(|p| p.name == "net_model")
            .unwrap();
        assert_eq!(model.spec["choices"], json!(["m1", "m2"]));
        assert_eq!(model.spec["default"], json!("m1"));
        let activated = with
            .conditions
            .iter()
            .find(|c| c.if_ == json!({"algorithm": ALGORITHM}))
            .unwrap();
        assert_eq!(activated.then.len(), PARAMS.len());
    }

    #[test]
    fn model_of_params_only_reads_net_algorithms() {
        assert_eq!(model_of_params(&params()), Some("fake-model"));
        assert_eq!(model_of_params(&json!({"algorithm": "mcts"})), None);
    }
}
