//! The trained networks as `net_gumbel` searches for the play host and the strategy catalog.
//!
//! Models are listed in `games/druid/cnn/models.json` (or the file named by
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
use super::encode::{num_actions, supported_planes};
use super::SUPPORTED_SIZES;
use crate::{Druid, HashedState, Move};

/// Overrides the models file.
pub const MODELS_ENV: &str = "DRUID_CNN_MODELS";

/// States per GPU forward call. The play search evaluates one leaf at a time, so this only bounds
/// memory.
const CHUNK_SIZE: usize = 64;

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
        || !supported_planes(g.in_planes)
        || g.policy_out != num_actions(g.size)
    {
        return Err(format!(
            "{} is not a supported Druid net: {g:?}",
            path.display()
        ));
    }
    Ok(Arc::new(weights))
}

pub struct DruidNets {
    models: BTreeMap<String, Arc<Weights>>,
}

impl DruidNets {
    pub fn new(models: BTreeMap<String, Arc<Weights>>) -> Self {
        DruidNets { models }
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
                eprintln!("game-druid: no CNN models ({}: {why})", path.display());
                return DruidNets::new(BTreeMap::new());
            }
        };
        let mut models = BTreeMap::new();
        for entry in &entries {
            match load_model(entry) {
                Ok(weights) => {
                    eprintln!("game-druid: CNN model {:?} loaded", entry.id);
                    models.insert(entry.id.clone(), weights);
                }
                Err(why) => eprintln!(
                    "game-druid: CNN model {:?} unavailable: {why} (edit {} or set {MODELS_ENV})",
                    entry.id,
                    path.display()
                ),
            }
        }
        DruidNets::new(models)
    }
}

impl NetSearchFactory<Druid> for DruidNets {
    fn models(&self) -> Vec<String> {
        self.models.keys().cloned().collect()
    }

    fn build(
        &self,
        spec: &NetSearchSpec,
        seed: u64,
    ) -> Result<Box<dyn Search<G = Druid>>, HostError> {
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
        let max_time = spec.max_time.unwrap_or_default();
        let max_iterations = spec.max_iterations;
        crate::with_board_size!(
            weights.geometry.size,
            N => Ok(Box::new(
                DruidCnnSearch::<N>::new(&spec.model, weights, cfg, kind, max_time, max_iterations, seed),
            )),
            n => Err(HostError::internal(format!("no {n}x{n} network support"))),
        )
    }
}

/// A [`CnnAgent`] as a `Search` over plain [`Druid`], remembering the last search for the
/// report accessors.
pub struct DruidCnnSearch<const N: usize> {
    agent: CnnAgent<N>,
    simulation_limit: usize,
    max_time: Duration,
    /// The caller's real iteration cap, or `None` when the budget was time-only and
    /// `simulation_limit` is just `mcts_tune::search::MAX_ITER` sizing the Sequential Halving
    /// schedule -- see `mcts_tune::net_search::NetSearchSpec::max_iterations`'s doc comment.
    /// `search_report` must report this, not `simulation_limit`, as `iteration_limit`.
    max_iterations: Option<usize>,
    last: Option<(RootSummary, Duration)>,
}

impl<const N: usize> DruidCnnSearch<N> {
    pub fn new(
        name: &str,
        weights: &Arc<Weights>,
        cfg: mcts_batch::Config,
        kind: Kind,
        max_time: Duration,
        max_iterations: Option<usize>,
        seed: u64,
    ) -> Self {
        DruidCnnSearch {
            simulation_limit: cfg.num_simulations,
            agent: CnnAgent::<N>::new(name, weights, cfg, kind, CHUNK_SIZE, max_time, seed),
            max_time,
            max_iterations,
            last: None,
        }
    }

    fn time_limit_seconds(&self) -> Option<f64> {
        (self.max_time != Duration::default()).then_some(self.max_time.as_secs_f64())
    }

    /// The only two ways `gumbel_search_with_root_value`'s Sequential-Halving loop stops: the
    /// deadline fires mid-schedule (`simulations < simulation_limit`), or the schedule runs to
    /// completion (`simulations == simulation_limit`, regardless of whether that total came from
    /// a real `max_iterations` or just `MAX_ITER` sizing a time-only budget's schedule).
    fn termination(&self, simulations: usize) -> SearchTermination {
        if simulations < self.simulation_limit {
            SearchTermination::Time
        } else {
            SearchTermination::Iterations
        }
    }
}

impl<const N: usize> Search for DruidCnnSearch<N> {
    type G = Druid;

    fn friendly_name(&self) -> String {
        self.agent.friendly_name()
    }

    fn set_friendly_name(&mut self, name: &str) {
        self.agent.set_friendly_name(name);
    }

    fn unsupported_reason(&self, state: &HashedState) -> Option<String> {
        let size = usize::from(state.state().size.w);
        (size != N).then(|| format!("this network only plays {N}x{N} boards, not {size}x{size}"))
    }

    /// Callers check [`Self::unsupported_reason`] first; a state of another size is a caller bug.
    fn choose_action(&mut self, state: &HashedState) -> Move {
        if let Some(why) = self.unsupported_reason(state) {
            panic!("{why}");
        }
        let started = Instant::now();
        let (mv, summary) = self.agent.choose_with_summary(state);
        self.last = Some((summary, started.elapsed()));
        mv
    }

    fn principle_variation(&self) -> Vec<Move> {
        self.last
            .as_ref()
            .map(|(summary, _)| summary.principal_variation.clone())
            .unwrap_or_default()
    }

    fn root_report(&self, _state: &HashedState) -> RootReport<Move> {
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

    fn search_report(&self, _state: &HashedState, selected: &Move) -> SearchReport<Move> {
        let Some((summary, elapsed)) = &self.last else {
            return SearchReport::unavailable(SearchReportReason::SearchNotRun);
        };
        let seconds = elapsed.as_secs_f64();
        if summary.simulations == 0 {
            // A position with a single legal move is answered without a search.
            let mut report = SearchReport::unavailable(SearchReportReason::SearchNotRun);
            report.elapsed_seconds = Some(seconds);
            report.iteration_limit = self.max_iterations;
            report.time_limit_seconds = self.time_limit_seconds();
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
            iteration_limit: self.max_iterations,
            time_limit_seconds: self.time_limit_seconds(),
            completed_iterations: summary.simulations,
            termination: Some(self.termination(summary.simulations)),
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
    use crate::cnn::encode::IN_PLANES;
    use crate::Size;
    use grid_cnn::Geometry;
    use mcts::game::Game;

    fn zero_weights_of(size: usize) -> Arc<Weights> {
        Arc::new(Weights::zeros(Geometry {
            size,
            in_planes: IN_PLANES,
            channels: 8,
            blocks: 1,
            policy_planes: 2,
            policy_out: num_actions(size),
            value_planes: 1,
            value_hidden: 8,
        }))
    }

    fn nets() -> DruidNets {
        DruidNets::new(BTreeMap::from([
            ("zero-5x5".to_string(), zero_weights_of(5)),
            ("zero-7x7".to_string(), zero_weights_of(7)),
        ]))
    }

    fn spec(model: &str, simulations: usize) -> NetSearchSpec {
        NetSearchSpec {
            model: model.into(),
            simulations,
            max_iterations: Some(simulations),
            max_time: None,
            considered_actions: 4,
            value_scale: 0.1,
            max_visit_init: 50,
            selection: NetSelection::Gumbel,
        }
    }

    fn state(n: u8) -> HashedState {
        HashedState::new(Size { w: n, h: n })
    }

    /// A position with a real choice: the opening's sarsen, then the cell decision.
    fn cell_decision(n: u8) -> HashedState {
        Druid::apply(state(n), &Move::Piece(crate::PieceKind::Sarsen))
    }

    #[test]
    fn a_5x5_search_plays_a_legal_move_and_reports_its_root() {
        let mut search = nets().build(&spec("zero-5x5", 8), 0).unwrap();
        let s = cell_decision(5);
        assert_eq!(search.unsupported_reason(&s), None);
        let mv = search.choose_action(&s);
        let mut legal = Vec::new();
        Druid::generate_actions(&s, &mut legal);
        assert!(legal.contains(&mv));

        let report = search.search_report(&s, &mv);
        assert_eq!(report.status, SearchReportStatus::Available);
        assert_eq!(report.completed_iterations, 8);
        assert_eq!(report.iteration_limit, Some(8));
        assert_eq!(report.selected_action, Some(mv));
        assert!(!report.actions.is_empty() && report.actions.len() <= 4);
        let share: f64 = report.actions.iter().map(|a| a.share).sum();
        assert!((share - 1.0).abs() < 1e-9);
        let root = search.root_report(&s);
        assert_eq!(root.total_visits, 8);
        assert_eq!(root.actions.len(), report.actions.len());
    }

    #[test]
    fn a_time_only_budget_reports_the_time_limit_not_the_schedule_sizing_iteration_count() {
        // Mirrors `crate::search::make_candidate`'s `AlgorithmSpec::Net` arm for a budget with
        // only `max_time` set: `simulations` is just `MAX_ITER` sizing Sequential Halving's
        // schedule, and `max_iterations` (the real, caller-set cap) is `None`. Before this fix,
        // `search_report` reported `simulation_limit` as `iteration_limit` unconditionally and
        // always claimed `SearchTermination::Iterations`, so a pure time budget read back as an
        // iteration count that was never actually set.
        let mut budget_only_spec = spec("zero-5x5", 5_000);
        budget_only_spec.max_iterations = None;
        budget_only_spec.max_time = Some(Duration::from_millis(5));
        let mut search = nets().build(&budget_only_spec, 0).unwrap();
        let s = cell_decision(5);
        let mv = search.choose_action(&s);

        let report = search.search_report(&s, &mv);
        assert_eq!(report.status, SearchReportStatus::Available);
        assert_eq!(report.iteration_limit, None);
        assert_eq!(report.time_limit_seconds, Some(0.005));
        assert_eq!(report.termination, Some(SearchTermination::Time));
        assert!(
            report.completed_iterations < 5_000,
            "a 5ms deadline should cut the 5,000-sim schedule short, got {}",
            report.completed_iterations
        );
    }

    #[test]
    fn a_net_refuses_other_board_sizes_without_panicking_at_the_check() {
        let mut nine = nets().build(&spec("zero-7x7", 8), 0).unwrap();
        assert_eq!(nine.unsupported_reason(&state(7)), None);
        let why = nine.unsupported_reason(&state(5)).unwrap();
        assert!(why.contains("7x7"), "{why}");
        assert!(nine.choose_action(&cell_decision(7)) != Move::Piece(crate::PieceKind::Sarsen));
    }

    #[test]
    fn the_noise_free_selection_builds_too() {
        let mut det = spec("zero-5x5", 8);
        det.selection = NetSelection::MostVisited;
        let mut search = nets().build(&det, 0).unwrap();
        let s = cell_decision(5);
        let mv = search.choose_action(&s);
        assert_eq!(search.search_report(&s, &mv).status, SearchReportStatus::Available);
    }

    #[test]
    fn an_unknown_model_is_a_bad_request() {
        assert_eq!(nets().build(&spec("nope", 8), 0).err().unwrap().code, 400);
    }

    #[test]
    fn a_net_of_an_unsupported_size_is_refused_at_load() {
        let mut g = zero_weights_of(5).geometry;
        g.size = 8;
        g.policy_out = num_actions(8);
        let dir = std::env::temp_dir().join(format!("druid-cnn-test-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("eight.bin");
        Weights::zeros(g).save(&path).unwrap();
        let entry = ModelEntry { id: "eight".into(), weights: path };
        assert!(load_model(&entry).is_err());
        std::fs::remove_dir_all(&dir).unwrap();
    }
}
