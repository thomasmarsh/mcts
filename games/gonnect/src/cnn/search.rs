//! The trained networks as `net_gumbel` searches for the play host and the strategy catalog.
//!
//! Models are listed in `games/gonnect/cnn/models.json` (or the file named by
//! [`MODELS_ENV`]); a model whose weights file is missing or does not fit is skipped with a line
//! on stderr, and `net_gumbel` is only offered while at least one model loaded. Each move builds
//! a [`CnnAgent`] over the shared, parsed weights (about 5 ms), so nothing GPU-side outlives a
//! request.

use std::collections::BTreeMap;
use std::env;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, Instant};

use game_host::HostError;
use grid_cnn::Weights;
use mcts::algorithms::{
    ActionReport, RootReport, Search, SearchActionReport, SearchGraphMode, SearchReport,
    SearchReportReason, SearchReportStatus, SearchTermination,
};
use mcts_tune::net_search::{NetSearchFactory, NetSearchSpec, NetSelection};
use serde::Deserialize;

use super::agent::{CnnAgent, Kind, RootSummary};
use super::encode::{num_actions, IN_PLANES};
use crate::sized::SizedState;
use crate::{Gonnect, Move, State};

/// Overrides the models file.
pub const MODELS_ENV: &str = "GONNECT_CNN_MODELS";

/// States per GPU forward call. The play search evaluates one leaf at a time, so this only bounds
/// memory.
const CHUNK_SIZE: usize = 64;

/// Board sizes a loaded network can play (one monomorphic agent per size).
const SUPPORTED_SIZES: &[usize] = &[7];

#[derive(Deserialize)]
struct ModelEntry {
    id: String,
    weights: PathBuf,
}

fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn default_models_path() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("cnn/models.json")
}

fn load_model(entry: &ModelEntry) -> Result<Arc<Weights>, String> {
    let path = if entry.weights.is_absolute() {
        entry.weights.clone()
    } else {
        repo_root().join(&entry.weights)
    };
    let weights =
        Weights::load(&path).map_err(|e| format!("cannot load {}: {e}", path.display()))?;
    let g = weights.geometry;
    if !SUPPORTED_SIZES.contains(&g.size)
        || (g.in_planes, g.policy_out) != (IN_PLANES, num_actions(g.size))
    {
        return Err(format!(
            "{} is not a supported Gonnect net: {g:?}",
            path.display()
        ));
    }
    Ok(Arc::new(weights))
}

pub struct GonnectNets {
    models: BTreeMap<String, Arc<Weights>>,
}

impl GonnectNets {
    pub fn new(models: BTreeMap<String, Arc<Weights>>) -> Self {
        GonnectNets { models }
    }

    /// Loads every model of the models file that can be loaded, logging the ones that cannot.
    pub fn from_env() -> Self {
        let path = env::var_os(MODELS_ENV)
            .map(PathBuf::from)
            .unwrap_or_else(default_models_path);
        let entries: Vec<ModelEntry> = match std::fs::read_to_string(&path)
            .map_err(|e| e.to_string())
            .and_then(|text| serde_json::from_str(&text).map_err(|e| e.to_string()))
        {
            Ok(entries) => entries,
            Err(why) => {
                eprintln!("game-gonnect: no CNN models ({}: {why})", path.display());
                return GonnectNets::new(BTreeMap::new());
            }
        };
        let mut models = BTreeMap::new();
        for entry in &entries {
            match load_model(entry) {
                Ok(weights) => {
                    eprintln!("game-gonnect: CNN model {:?} loaded", entry.id);
                    models.insert(entry.id.clone(), weights);
                }
                Err(why) => eprintln!(
                    "game-gonnect: CNN model {:?} unavailable: {why} (edit {} or set {MODELS_ENV})",
                    entry.id,
                    path.display()
                ),
            }
        }
        GonnectNets::new(models)
    }
}

impl NetSearchFactory<Gonnect> for GonnectNets {
    fn models(&self) -> Vec<String> {
        self.models.keys().cloned().collect()
    }

    fn build(
        &self,
        spec: &NetSearchSpec,
        seed: u64,
    ) -> Result<Box<dyn Search<G = Gonnect>>, HostError> {
        let weights = self
            .models
            .get(&spec.model)
            .ok_or_else(|| HostError::bad_request(format!("unknown net_model {:?}", spec.model)))?;
        let cfg = mcts_batch::Config {
            num_simulations: spec.simulations,
            num_considered_actions: spec.considered_actions,
            value_scale: spec.value_scale as f32,
            max_visit_init: spec.max_visit_init as i32,
        };
        let kind = match spec.selection {
            NetSelection::Gumbel => Kind::Gumbel,
            NetSelection::MostVisited => Kind::Deterministic,
        };
        match weights.geometry.size {
            7 => Ok(Box::new(GonnectCnnSearch::<7>::new(
                &spec.model,
                weights,
                cfg,
                kind,
                seed,
            ))),
            n => Err(HostError::internal(format!("no {n}x{n} network support"))),
        }
    }
}

/// A [`CnnAgent`] as a `Search` over plain [`Gonnect`], remembering the last search for the
/// report accessors.
pub struct GonnectCnnSearch<const N: usize> {
    agent: CnnAgent<N>,
    simulation_limit: usize,
    last: Option<(RootSummary, Duration)>,
}

impl<const N: usize> GonnectCnnSearch<N> {
    pub fn new(
        name: &str,
        weights: &Arc<Weights>,
        cfg: mcts_batch::Config,
        kind: Kind,
        seed: u64,
    ) -> Self {
        GonnectCnnSearch {
            simulation_limit: cfg.num_simulations,
            agent: CnnAgent::<N>::new(name, weights, cfg, kind, CHUNK_SIZE, seed),
            last: None,
        }
    }
}

impl<const N: usize> Search for GonnectCnnSearch<N> {
    type G = Gonnect;

    fn friendly_name(&self) -> String {
        self.agent.friendly_name()
    }

    fn set_friendly_name(&mut self, name: &str) {
        self.agent.set_friendly_name(name);
    }

    fn unsupported_reason(&self, state: &State) -> Option<String> {
        let size = state.black().rows();
        (size != N).then(|| format!("this network only plays {N}x{N} boards, not {size}x{size}"))
    }

    /// Callers check [`Self::unsupported_reason`] first; a state of another size is a caller bug.
    fn choose_action(&mut self, state: &State) -> Move {
        if let Some(why) = self.unsupported_reason(state) {
            panic!("{why}");
        }
        let started = Instant::now();
        let (mv, summary) = self.agent.choose_with_summary(&SizedState(state.clone()));
        self.last = Some((summary, started.elapsed()));
        mv
    }

    fn principle_variation(&self) -> Vec<Move> {
        self.last
            .as_ref()
            .map(|(summary, _)| summary.principal_variation.clone())
            .unwrap_or_default()
    }

    fn root_report(&self, _state: &State) -> RootReport<Move> {
        let Some((summary, _)) = &self.last else {
            return RootReport {
                actions: Vec::new(),
                principal_variation: Vec::new(),
                total_visits: 0,
            };
        };
        RootReport {
            actions: summary
                .actions
                .iter()
                .map(|a| ActionReport {
                    action: a.mv,
                    visits: a.visits,
                    mean_value: f64::from(a.q),
                    is_proven: false,
                })
                .collect(),
            principal_variation: summary.principal_variation.clone(),
            total_visits: summary.simulations as u32,
        }
    }

    fn search_report(&self, _state: &State, selected: &Move) -> SearchReport<Move> {
        let Some((summary, elapsed)) = &self.last else {
            return SearchReport::unavailable(SearchReportReason::SearchNotRun);
        };
        let seconds = elapsed.as_secs_f64();
        if summary.simulations == 0 {
            // A position with a single legal move is answered without a search.
            let mut report = SearchReport::unavailable(SearchReportReason::SearchNotRun);
            report.elapsed_seconds = Some(seconds);
            report.iteration_limit = Some(self.simulation_limit);
            report.selected_action = Some(*selected);
            report.principal_variation = vec![*selected];
            return report;
        }
        let total: u32 = summary.actions.iter().map(|a| a.visits).sum();
        SearchReport {
            schema_version: 1,
            status: SearchReportStatus::Available,
            reason: None,
            elapsed_seconds: Some(seconds),
            iteration_limit: Some(self.simulation_limit),
            time_limit_seconds: None,
            completed_iterations: summary.simulations,
            termination: Some(SearchTermination::Iterations),
            selected_action: Some(*selected),
            actions: summary
                .actions
                .iter()
                .map(|a| SearchActionReport {
                    action: a.mv,
                    visits: a.visits,
                    share: f64::from(a.visits) / f64::from(total.max(1)),
                    mean_value: f64::from(a.q),
                    is_proven: false,
                })
                .collect(),
            principal_variation: summary.principal_variation.clone(),
            root_visits: summary.simulations as u32,
            tree_nodes: summary.simulations,
            mean_depth: None,
            max_depth: None,
            graph_mode: Some(SearchGraphMode::Tree),
            tt_reads: 0,
            tt_writes: 0,
            tt_hits: 0,
            tt_hit_ratio: None,
            iterations_per_second: (seconds > 0.0).then(|| summary.simulations as f64 / seconds),
            warnings: Vec::new(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use grid_cnn::Geometry;
    use mcts::game::Game;

    fn zero_weights() -> Arc<Weights> {
        Arc::new(Weights::zeros(Geometry {
            size: 7,
            in_planes: IN_PLANES,
            channels: 8,
            blocks: 1,
            policy_planes: 2,
            policy_out: num_actions(7),
            value_planes: 1,
            value_hidden: 8,
        }))
    }

    pub(crate) fn nets() -> GonnectNets {
        GonnectNets::new(BTreeMap::from([("zero-7x7".to_string(), zero_weights())]))
    }

    fn spec(simulations: usize) -> NetSearchSpec {
        NetSearchSpec {
            model: "zero-7x7".into(),
            simulations,
            considered_actions: 4,
            value_scale: 0.1,
            max_visit_init: 50,
            selection: NetSelection::Gumbel,
        }
    }

    #[test]
    fn a_7x7_search_plays_a_legal_move_and_reports_its_root() {
        let mut search = nets().build(&spec(8), 0).unwrap();
        let state = State::new(7);
        assert_eq!(search.unsupported_reason(&state), None);
        let mv = search.choose_action(&state);
        let mut legal = Vec::new();
        Gonnect::generate_actions(&state, &mut legal);
        assert!(legal.contains(&mv));

        let report = search.search_report(&state, &mv);
        assert_eq!(report.status, SearchReportStatus::Available);
        assert_eq!(report.completed_iterations, 8);
        assert_eq!(report.iteration_limit, Some(8));
        assert_eq!(report.selected_action, Some(mv));
        assert!(!report.actions.is_empty() && report.actions.len() <= 4);
        let share: f64 = report.actions.iter().map(|a| a.share).sum();
        assert!((share - 1.0).abs() < 1e-9);
        assert!(report
            .actions
            .iter()
            .all(|a| (-1.0..=1.0).contains(&a.mean_value)));
        let root = search.root_report(&state);
        assert_eq!(root.total_visits, 8);
        assert_eq!(root.actions.len(), report.actions.len());
    }

    #[test]
    fn other_board_sizes_are_unsupported_not_a_panic_at_the_check() {
        let search = nets().build(&spec(8), 0).unwrap();
        for size in [3, 5, 9, 13] {
            let why = search.unsupported_reason(&State::new(size)).unwrap();
            assert!(why.contains("7x7"), "{why}");
        }
    }

    #[test]
    fn the_noise_free_selection_and_the_gumbel_selection_both_build() {
        let mut det = spec(8);
        det.selection = NetSelection::MostVisited;
        let mut search = nets().build(&det, 0).unwrap();
        let state = State::new(7);
        let mv = search.choose_action(&state);
        assert_eq!(
            search.search_report(&state, &mv).status,
            SearchReportStatus::Available
        );
    }

    #[test]
    fn an_unknown_model_is_a_bad_request() {
        let mut bad = spec(8);
        bad.model = "nope".into();
        assert_eq!(nets().build(&bad, 0).err().unwrap().code, 400);
    }

    #[test]
    fn a_net_of_an_unsupported_size_is_refused_at_load() {
        let mut g = zero_weights().geometry;
        g.size = 5;
        g.policy_out = num_actions(5);
        let dir = std::env::temp_dir().join(format!("gonnect-cnn-test-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("five.bin");
        Weights::zeros(g).save(&path).unwrap();
        let entry = ModelEntry {
            id: "five".into(),
            weights: path,
        };
        assert!(load_model(&entry).is_err());
        std::fs::remove_dir_all(&dir).unwrap();
    }
}
