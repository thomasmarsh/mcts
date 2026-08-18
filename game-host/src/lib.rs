//! Protocol helper for game subprocess binaries.
//!
//! Each game kind (druid, ttt, othello, …) builds a standalone binary that
//! speaks the JSON-line subprocess protocol over stdin/stdout using the
//! types and `run_host` function in this crate.
//!
//! The server/bench crates also depend on this crate for the `GameAdapter`
//! trait and the request/response types used by the `SubprocessAdapter`
//! (Step 3 of the workspace migration).

pub mod subprocess;

use serde_json::Value;
use std::fmt;
use std::io::{self, BufRead, BufReader, BufWriter, Read, Write};

// ---------------------------------------------------------------------------
// Wire protocol types
// ---------------------------------------------------------------------------

/// One request read from stdin: a single JSON line.
#[derive(Debug, serde::Serialize, serde::Deserialize)]
pub struct Request {
    /// Unique request identifier, echoed back in the response.
    pub id: u64,
    /// Method name — maps to a `GameAdapter` method.
    pub method: String,
    /// Method-specific parameters.
    pub params: Value,
}

/// One response written to stdout: a single JSON line.
#[derive(Debug, serde::Serialize, serde::Deserialize)]
#[serde(untagged)]
pub enum Response {
    /// Successful method call.
    Success { id: u64, result: Value },
    /// Failed method call.
    Error { id: u64, error: ErrorBody },
}

/// Structured error body within an error response.
#[derive(Debug, serde::Serialize, serde::Deserialize)]
pub struct ErrorBody {
    /// HTTP-style status code (400, 404, 500, …).
    pub code: u16,
    /// Human-readable error description.
    pub message: String,
}

// ---------------------------------------------------------------------------
// HostError
// ---------------------------------------------------------------------------

/// A simple, HTTP-style error type used by the `GameAdapter` trait methods.
///
/// Carries an integer code (matching HTTP status conventions) and a
/// human-readable message.  The `run_host` function converts these into
/// `Response::Error` when a method fails.  No external HTTP framework
/// dependency — the server crate wraps this in its own `AdapterError` if
/// axum integration is needed.
#[derive(Debug)]
pub struct HostError {
    pub code: u16,
    pub message: String,
}

impl HostError {
    pub fn bad_request(message: impl Into<String>) -> Self {
        Self {
            code: 400,
            message: message.into(),
        }
    }
    pub fn not_found(message: impl Into<String>) -> Self {
        Self {
            code: 404,
            message: message.into(),
        }
    }
    pub fn internal(message: impl Into<String>) -> Self {
        Self {
            code: 500,
            message: message.into(),
        }
    }
}

impl fmt::Display for HostError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} ({})", self.message, self.code)
    }
}

impl std::error::Error for HostError {}

// ---------------------------------------------------------------------------
// Response types (mirror `server/adapters/` shapes)
// ---------------------------------------------------------------------------

/// Information about one AI preset exposed by a game.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct AiPresetInfo {
    pub id: String,
    pub label: String,
    pub description: String,
}

/// The result of a completed `ai_move`: the chosen move and the resulting
/// state, so the caller can apply both without a second round-trip.
#[derive(Debug, serde::Serialize, serde::Deserialize)]
pub struct AiMoveResult {
    pub mv: Value,
    pub state: Value,
}

/// One candidate root action returned from `analyze`.
#[derive(Debug, serde::Serialize, serde::Deserialize)]
pub struct AnalysisAction {
    pub action: Value,
    pub visits: u32,
    pub mean_value: f64,
    pub is_proven: bool,
}

/// Full analysis returned from `analyze`.
#[derive(Debug, serde::Serialize, serde::Deserialize)]
pub struct Analysis {
    pub actions: Vec<AnalysisAction>,
    pub principal_variation: Vec<Value>,
    pub total_visits: u32,
    pub suggested_move: Option<Value>,
}

/// Which side the candidate configuration played in one configured match.
#[derive(Debug, Clone, Copy, serde::Serialize, serde::Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ConfiguredCandidateSide {
    First,
    Second,
}

/// The result of one configured candidate-versus-baseline game.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, PartialEq, Eq)]
pub struct ConfiguredMatchResult {
    #[serde(rename = "type")]
    pub record_type: String,
    pub seq: u64,
    pub round: u32,
    pub seed: u64,
    pub candidate_side: ConfiguredCandidateSide,
    pub outcome: ConfiguredOutcome,
    pub trace_game_seq: Option<u64>,
    pub plies: u32,
    pub elapsed_ms: u64,
    pub candidate: ConfiguredStrategyMetrics,
    pub baseline: ConfiguredStrategyMetrics,
}

/// One configured strategy's aggregate work in a completed game.
#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize, PartialEq, Eq)]
pub struct ConfiguredStrategyMetrics {
    pub iterations_total: u64,
    pub iterations_first_half: u64,
    pub move_time_ms: u64,
}

/// A configured match outcome from the candidate's perspective.
#[derive(Debug, Clone, Copy, serde::Serialize, serde::Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ConfiguredOutcome {
    CandidateWin,
    BaselineWin,
    Draw,
}

/// Aggregate result for a completed configured comparison.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, PartialEq, Eq)]
pub struct ConfiguredComparisonSummary {
    #[serde(rename = "type")]
    pub record_type: String,
    pub games: u32,
    pub wins: u32,
    pub losses: u32,
    pub draws: u32,
}

// ---------------------------------------------------------------------------
// Tuner metadata (SMAC3-style hyperparameter search)
// ---------------------------------------------------------------------------

/// One parameter in a tuner's search space (mirrors the shape of the SMAC3
/// harness's YAML search space), reported by `tuner()` so a launch form or
/// CLI consumer can render/validate fields without a per-game hardcoded
/// schema. `spec` carries the type-specific keys verbatim (`type`/`bounds`/
/// `default` for `float`/`int`, `type`/`choices`/`default` for
/// `categorical`, `type`/`default` for `bool`, or `type`/`value` for
/// `constant`).
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct TunerParameter {
    pub name: String,
    #[serde(flatten)]
    pub spec: Value,
}

/// A conditional activation rule: when `if` matches the trial's active
/// config, every name in `then` also becomes active. `if` is a single-entry
/// object mapping a parent parameter name to either one value or a list of
/// values (mirrors the YAML `if:`/`then:` shape).
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct TunerCondition {
    #[serde(rename = "if")]
    pub if_: Value,
    pub then: Vec<String>,
}

/// Metadata describing a game's tunable strategy search space, as reported
/// by the `tune describe` subcommand -- the parameter space and baseline
/// instances a SMAC3-style harness needs to run trials, without embedding
/// the actual search/eval logic (that stays behind `tune_eval`).
///
/// `baselines` is a list rather than a single id so a harness can evaluate
/// each trial config against multiple opponent strengths (SMAC3's
/// `Scenario(instances=...)` mechanism) instead of one fixed baseline --
/// once a config saturates 100% win rate against an easy baseline, cost
/// floors at `0.0` and a harder second instance is the only way to keep
/// ranking top candidates against each other. Most games report exactly one
/// entry here (a single preset stands in for "the" baseline); a game with a
/// genuine second preset can list it as a second instance.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct TunerInfo {
    pub id: String,
    pub baselines: Vec<String>,
    pub eval_rounds: u32,
    pub parameters: Vec<TunerParameter>,
    pub conditions: Vec<TunerCondition>,
    /// The game's own `default_config()` -- a game-setup axis (e.g. Druid's
    /// board size) that's separate from `parameters` (the strategy search
    /// space) entirely: SMAC3 never searches over it, `tune_eval`'s
    /// `game_config` argument just pins every trial in a run to it. `{}` for
    /// every game whose board is fixed at compile time (everything but
    /// Druid today) -- a caller should treat that as "nothing to configure",
    /// same as `default_config()` itself already means for `new_state`.
    pub game_config: Value,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, PartialEq, Eq)]
pub struct CompareValidationField {
    pub field: String,
    pub message: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub candidate_index: Option<usize>,
}

/// Stable 53-bit SplitMix64-derived seed used by configured comparisons.
/// Inputs are a seed and a zero-based ordinal; the result is safe to carry
/// through JSON and JavaScript without losing integer precision.
pub fn derive_seed(seed: u64, ordinal: u64) -> u64 {
    let mut value = seed.wrapping_add(ordinal.wrapping_mul(0x9e37_79b9_7f4a_7c15));
    value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    (value ^ (value >> 31)) & 9_007_199_254_740_991
}

// ---------------------------------------------------------------------------
// Opening-book metadata (Quasi-Best-First self-play)
// ---------------------------------------------------------------------------

/// Metadata describing a game's opening-book support, as reported by the
/// `book describe` subcommand -- mirrors `TunerInfo`'s shape and reasoning:
/// enough for a generic caller (a launch form, a CLI wrapper script) to
/// know book generation exists and what its default knob values are,
/// without embedding the self-play loop itself (that stays behind
/// `book_build`).
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct BookInfo {
    pub id: String,
    /// Default number of self-play games `book_build` runs when the caller
    /// doesn't override `rounds`.
    pub default_rounds: u32,
    /// The game's own `default_config()` -- same purpose as
    /// `TunerInfo::game_config`: a game-setup axis (e.g. board size)
    /// `book_build`'s `game_config` argument pins the run to, separate from
    /// `rounds`/`seed`. `{}` for a game whose board is fixed at compile
    /// time.
    pub game_config: Value,
}

// ---------------------------------------------------------------------------
// GameAdapter trait
// ---------------------------------------------------------------------------

/// Type-erased, per-game-kind adapter over `mcts::game::Game` +
/// `mcts::strategies::Search`.
///
/// Every method is stateless: state flows in as a JSON `Value` and back out
/// as another.  Concrete adapters deserialize `Value` arguments into real
/// game types, call through to `Game`/`Search`, and re-serialize the result.
/// This is the same shape as `server/adapters/`'s `GameAdapter` trait but
/// with a simpler error type (no axum dependency).
pub trait GameAdapter: Send + Sync {
    fn kind(&self) -> &'static str;
    fn label(&self) -> &'static str;
    fn description(&self) -> &'static str;

    /// A default/example config value for `new_state`.  Also serves as a
    /// config schema hint for generic new-game forms.
    fn default_config(&self) -> Value;

    fn new_state(&self, config: Value) -> Result<Value, HostError>;
    fn legal_moves(&self, state: &Value) -> Result<Vec<Value>, HostError>;
    fn apply(&self, state: &Value, mv: &Value) -> Result<Value, HostError>;
    fn view(&self, state: &Value) -> Result<Value, HostError>;

    fn ai_presets(&self) -> Vec<AiPresetInfo>;
    fn ai_move(
        &self,
        state: &Value,
        preset: &str,
        custom: Option<&Value>,
    ) -> Result<AiMoveResult, HostError>;
    fn analyze(
        &self,
        state: &Value,
        preset: &str,
        custom: Option<&Value>,
        budget_ms: Option<u64>,
    ) -> Result<Analysis, HostError>;

    /// Tunable strategy search-space metadata, for games that support
    /// SMAC3-style hyperparameter tuning via `tune_eval`. `None` (the
    /// default) for every game that doesn't -- tuning support is opt-in per
    /// game, not a universal requirement.
    fn tuner(&self) -> Option<TunerInfo> {
        None
    }

    /// Validate one configured comparison without playing a game. Concrete
    /// adapters retain ownership of game setup and strategy construction;
    /// this default implementation exercises those same adapter paths with
    /// zero evaluation rounds.
    fn validate_compare(
        &self,
        candidate_config: Value,
        baseline_config: Value,
        game_config: Option<Value>,
    ) -> Vec<CompareValidationField> {
        let mut errors =
            self.validate_compare_many(vec![candidate_config], baseline_config, game_config);
        for error in &mut errors {
            error.candidate_index = None;
        }
        errors
    }

    /// Validate several candidate configurations against one baseline without
    /// playing a game. Game setup and the baseline are built once; each
    /// candidate is built independently so every configuration is checked.
    fn validate_compare_many(
        &self,
        candidate_configs: Vec<Value>,
        baseline_config: Value,
        game_config: Option<Value>,
    ) -> Vec<CompareValidationField> {
        let mut errors = Vec::new();
        if self.tuner().is_none() {
            errors.push(CompareValidationField {
                field: "candidate_config".into(),
                message: "game does not support configured strategy validation".into(),
                candidate_index: None,
            });
            return errors;
        }
        let setup = game_config.clone().unwrap_or_else(|| self.default_config());
        if let Err(error) = self.new_state(setup) {
            errors.push(CompareValidationField {
                field: "game_config".into(),
                message: error.message,
                candidate_index: None,
            });
        }
        let baseline_name = self
            .tuner()
            .and_then(|info| info.baselines.first().cloned());
        if !errors.iter().any(|error| error.field == "game_config") {
            for (candidate_index, candidate_config) in candidate_configs.into_iter().enumerate() {
                let mut sink = |_result: ConfiguredMatchResult| Ok(());
                if let Err(error) = self.tune_eval(
                    candidate_config,
                    0,
                    Some(0),
                    baseline_name.clone(),
                    None,
                    game_config.clone(),
                    Some(1),
                    None,
                    None,
                    &mut sink,
                ) {
                    errors.push(CompareValidationField {
                        field: "candidate_config".into(),
                        message: error.message,
                        candidate_index: Some(candidate_index),
                    });
                }
            }
            let mut sink = |_result: ConfiguredMatchResult| Ok(());
            if let Err(error) = self.tune_eval(
                baseline_config,
                0,
                Some(0),
                None,
                None,
                game_config,
                Some(1),
                None,
                None,
                &mut sink,
            ) {
                errors.push(CompareValidationField {
                    field: "baseline_config".into(),
                    message: error.message,
                    candidate_index: None,
                });
            }
        }
        errors
    }

    /// Play `rounds` games of a `params`-built candidate strategy against an
    /// opponent and return a cost (lower is better) plus win/loss/draw
    /// counts, as a JSON object. The opponent comes from exactly one of:
    /// `baseline`, a named entry of `tuner()`'s `baselines` list (`None`
    /// means the first/default entry), or `baseline_config`, a raw params
    /// JSON object (same schema as `params`) built the same way the
    /// candidate is -- e.g. a previously discovered config used as the next
    /// run's opponent instead of a hand-authored preset. The only real
    /// caller (`run_tune_eval`) guarantees at most one of the two is
    /// `Some` before invoking this method. CLI-only (`tune eval`) -- never
    /// dispatched over the JSONL loop, since a full multi-game match is a
    /// batch job, not a per-move request.
    ///
    /// `game_config` pins every game in this call to a non-default game
    /// setup (same schema as `new_state`'s `config` argument, e.g. Druid's
    /// `{"size": {...}}`), falling back to `default_config()`'s value when
    /// `None`. A game whose board is fixed at compile time (`default_config`
    /// returns `{}`) has nothing to vary here and ignores this argument.
    ///
    /// `max_iterations` is an operator-set, per-*run* compute budget (not a
    /// per-trial hyperparameter SMAC3 searches over -- see
    /// `mcts_tune::SearchBudget`'s doc comment for why), forwarded verbatim
    /// from `--max-iterations`. `None` means "use `mcts-tune`'s own
    /// historical default." An implementation threading this into
    /// `mcts_tune::SearchBudget` must apply the *same* value to both the
    /// candidate and, for a `baseline_config`-backed opponent,
    /// `mcts_tune::build_search`'s own budget -- see that function's doc
    /// comment for why an asymmetric override there is a real bug, not a
    /// simplification.
    /// `trace_path`, forwarded verbatim from `--trace-path`, is a plain
    /// file path a `mcts_tune::trace::MoveTracer` appends move-trace JSON
    /// lines to as self-play games are played -- for live monitoring/
    /// sanity-checking a SMAC3 run in progress. `None` disables tracing
    /// entirely (no file opened, no per-ply overhead).
    #[allow(unused_variables)]
    #[allow(clippy::too_many_arguments)]
    fn tune_eval(
        &self,
        params: Value,
        rounds: u32,
        seed: Option<u64>,
        baseline: Option<String>,
        baseline_config: Option<Value>,
        game_config: Option<Value>,
        max_iterations: Option<usize>,
        max_time_ms: Option<u64>,
        trace_path: Option<std::path::PathBuf>,
        on_game: &mut dyn FnMut(ConfiguredMatchResult) -> Result<(), HostError>,
    ) -> Result<Value, HostError> {
        Err(HostError::not_found("tuning not supported"))
    }

    /// Opening-book metadata, for games that support Quasi-Best-First
    /// self-play book generation via `book_build`. `None` (the default) for
    /// every game that hasn't wired this up -- opt-in, same shape as
    /// `tuner()`.
    fn book(&self) -> Option<BookInfo> {
        None
    }

    /// Runs `rounds` self-play games (Chaslot et al.'s Quasi-Best-First
    /// algorithm) and returns the resulting opening book, serialized as
    /// JSON. `seed` seeds the run for reproducibility; `None` means
    /// whatever the game's own default is. `game_config` pins the run to a
    /// non-default game setup, same convention as `tune_eval`'s argument of
    /// the same name. CLI-only (`book build`), never dispatched over the
    /// JSONL loop -- same rationale as `tune_eval`: a many-game self-play
    /// run is a batch job, not a per-move request.
    #[allow(unused_variables)]
    fn book_build(
        &self,
        rounds: u32,
        seed: Option<u64>,
        game_config: Option<Value>,
    ) -> Result<Value, HostError> {
        Err(HostError::not_found(
            "opening book generation not supported",
        ))
    }
}

// ---------------------------------------------------------------------------
// run_host
// ---------------------------------------------------------------------------

/// Read JSON-line requests from `reader`, dispatch each to `adapter`, write
/// JSON-line responses to `writer`.
///
/// Terminates when `reader` reaches EOF (stdin closed or pipe broken).
/// Errors on individual lines (malformed JSON, missing params, adapter
/// failures) produce error responses and continue — a single bad request
/// never kills the host.
pub fn run_host<R: Read, W: Write, A: GameAdapter>(reader: R, writer: W, adapter: A) {
    let reader = BufReader::new(reader);
    let mut writer = BufWriter::new(writer);

    for line in reader.lines() {
        let line = match line {
            Ok(l) => l,
            Err(e) => {
                // stdin error (broken pipe etc.) — stop
                eprintln!("game-host: stdin error: {e}");
                break;
            }
        };

        let trimmed = line.trim().to_owned();
        if trimmed.is_empty() {
            continue;
        }

        let req: Request = match serde_json::from_str(&trimmed) {
            Ok(r) => r,
            Err(e) => {
                let resp = Response::Error {
                    id: 0,
                    error: ErrorBody {
                        code: 400,
                        message: format!("invalid request: {e}"),
                    },
                };
                let _ = writeln!(writer, "{}", serde_json::to_string(&resp).unwrap());
                let _ = writer.flush();
                continue;
            }
        };

        let result = dispatch(&adapter, &req);
        let resp = match result {
            Ok(v) => Response::Success {
                id: req.id,
                result: v,
            },
            Err(e) => Response::Error {
                id: req.id,
                error: ErrorBody {
                    code: e.code,
                    message: e.message,
                },
            },
        };
        let json = serde_json::to_string(&resp).expect("Response always serializes");
        let _ = writeln!(writer, "{json}");
        let _ = writer.flush();
    }
}

/// Convenience wrapper that reads from stdin and writes to stdout.
pub fn run_stdin_stdout<A: GameAdapter>(adapter: A) {
    run_host(io::stdin(), io::stdout(), adapter);
}

// ---------------------------------------------------------------------------
// CLI entry point
// ---------------------------------------------------------------------------

/// Static description of a game, as reported by the `describe` subcommand.
#[derive(Debug, serde::Serialize, serde::Deserialize)]
pub struct GameDescription {
    pub kind: String,
    pub label: String,
    pub description: String,
    pub default_config: Value,
    pub ai_presets: Vec<AiPresetInfo>,
    pub tuning: Option<TunerInfo>,
}

impl GameDescription {
    fn of<A: GameAdapter>(adapter: &A) -> Self {
        Self {
            kind: adapter.kind().to_owned(),
            label: adapter.label().to_owned(),
            description: adapter.description().to_owned(),
            default_config: adapter.default_config(),
            ai_presets: adapter.ai_presets(),
            tuning: adapter.tuner(),
        }
    }
}

/// Entry point every game binary's `main()` should call instead of
/// [`run_stdin_stdout`] directly.
///
/// Inspects `std::env::args()` for a `describe` subcommand, which prints one
/// JSON line describing the game (kind/label/description/default_config/
/// ai_presets) and exits — a one-shot query that doesn't require opening a
/// JSONL session. Any other argument (including none) falls back to the
/// existing stdin/stdout protocol loop unchanged, so `SubprocessAdapter`
/// (which never passes args) is unaffected.
pub fn run_cli<A: GameAdapter>(adapter: A) {
    let code = run_cli_with(std::env::args().skip(1), io::stdin(), io::stdout(), adapter);
    std::process::exit(code);
}

/// Testable core of [`run_cli`]: takes the args iterator and reader/writer
/// as parameters instead of reaching for the real process environment.
/// Returns the process exit code the caller should use.
fn run_cli_with<I, R, W, A>(mut args: I, reader: R, mut writer: W, adapter: A) -> i32
where
    I: Iterator<Item = String>,
    R: Read,
    W: Write,
    A: GameAdapter,
{
    match args.next().as_deref() {
        Some("describe") => {
            let description = GameDescription::of(&adapter);
            let json =
                serde_json::to_string(&description).expect("GameDescription always serializes");
            let _ = writeln!(writer, "{json}");
            0
        }
        Some("tune") => match args.next().as_deref() {
            Some("describe") => match adapter.tuner() {
                Some(info) => {
                    let json = serde_json::to_string(&info).expect("TunerInfo always serializes");
                    let _ = writeln!(writer, "{json}");
                    0
                }
                None => {
                    eprintln!("tuning not supported");
                    1
                }
            },
            Some("eval") => run_tune_eval(args, &mut writer, &adapter),
            // Any other (or missing) `tune` argument falls back to the
            // stdin/stdout loop rather than erroring, same as an unknown
            // top-level subcommand -- a future flag addition can't
            // accidentally break `SubprocessAdapter`.
            _ => {
                run_host(reader, writer, adapter);
                0
            }
        },
        Some("compare") => match args.next().as_deref() {
            Some("describe") => match adapter.tuner() {
                Some(info) => {
                    let json = serde_json::to_string(&info).expect("TunerInfo always serializes");
                    let _ = writeln!(writer, "{json}");
                    0
                }
                None => {
                    eprintln!("tuning not supported");
                    1
                }
            },
            Some("validate") => run_compare_validate(args, &mut writer, &adapter),
            Some("eval") => run_compare_eval(args, &mut writer, &adapter),
            _ => {
                run_host(reader, writer, adapter);
                0
            }
        },
        Some("book") => match args.next().as_deref() {
            Some("describe") => match adapter.book() {
                Some(info) => {
                    let json = serde_json::to_string(&info).expect("BookInfo always serializes");
                    let _ = writeln!(writer, "{json}");
                    0
                }
                None => {
                    eprintln!("opening book generation not supported");
                    1
                }
            },
            Some("build") => run_book_build(args, &mut writer, &adapter),
            // Same fallback rationale as the `tune` arm above.
            _ => {
                run_host(reader, writer, adapter);
                0
            }
        },
        _ => {
            run_host(reader, writer, adapter);
            0
        }
    }
}

/// Parses `--config <json> --rounds <n> [--seed <n>] [--baseline <id> |
/// --baseline-config <json>] [--game-config <json>] [--max-iterations <n>]`
/// from the remaining CLI args, calls [`GameAdapter::tune_eval`], and prints
/// its JSON result verbatim to `writer`. Returns the process exit code.
/// `--baseline` and `--baseline-config` are mutually exclusive -- supplying
/// both is rejected before the adapter is ever called. `--max-iterations`
/// is the per-run compute-budget override -- see `tune_eval`'s own doc
/// comment.
fn run_tune_eval<I, W, A>(args: I, writer: &mut W, adapter: &A) -> i32
where
    I: Iterator<Item = String>,
    W: Write,
    A: GameAdapter,
{
    let mut args = args;
    let mut config: Option<String> = None;
    let mut rounds: Option<u32> = None;
    let mut seed: Option<u64> = None;
    let mut baseline: Option<String> = None;
    let mut baseline_config: Option<String> = None;
    let mut game_config: Option<String> = None;
    let mut max_iterations: Option<usize> = None;
    let mut trace_path: Option<String> = None;
    while let Some(flag) = args.next() {
        match flag.as_str() {
            "--config" => config = args.next(),
            "--rounds" => rounds = args.next().and_then(|s| s.parse().ok()),
            "--seed" => seed = args.next().and_then(|s| s.parse().ok()),
            "--baseline" => baseline = args.next(),
            "--baseline-config" => baseline_config = args.next(),
            "--game-config" => game_config = args.next(),
            "--max-iterations" => max_iterations = args.next().and_then(|s| s.parse().ok()),
            "--trace-path" => trace_path = args.next(),
            _ => {}
        }
    }

    let result = (|| -> Result<Value, HostError> {
        let config = config.ok_or_else(|| HostError::bad_request("missing --config"))?;
        let rounds = rounds.ok_or_else(|| HostError::bad_request("missing --rounds"))?;
        if rounds == 0 {
            return Err(HostError::bad_request("--rounds must be positive"));
        }
        let params: Value = serde_json::from_str(&config)
            .map_err(|e| HostError::bad_request(format!("invalid --config JSON: {e}")))?;
        if baseline.is_some() && baseline_config.is_some() {
            return Err(HostError::bad_request(
                "--baseline and --baseline-config are mutually exclusive",
            ));
        }
        let baseline_config = baseline_config
            .map(|s| {
                serde_json::from_str(&s).map_err(|e| {
                    HostError::bad_request(format!("invalid --baseline-config JSON: {e}"))
                })
            })
            .transpose()?;
        let game_config = game_config
            .map(|s| {
                serde_json::from_str(&s)
                    .map_err(|e| HostError::bad_request(format!("invalid --game-config JSON: {e}")))
            })
            .transpose()?;
        let mut on_game = |_result: ConfiguredMatchResult| Ok(());
        adapter.tune_eval(
            params,
            rounds,
            seed,
            baseline,
            baseline_config,
            game_config,
            max_iterations,
            None,
            trace_path.map(std::path::PathBuf::from),
            &mut on_game,
        )
    })();

    match result {
        Ok(v) => {
            let json = serde_json::to_string(&v).expect("tune_eval result always serializes");
            let _ = writeln!(writer, "{json}");
            0
        }
        Err(e) => {
            eprintln!("{e}");
            1
        }
    }
}

#[derive(Debug, serde::Serialize)]
struct CompareValidationResponse {
    valid: bool,
    errors: Vec<CompareValidationField>,
}

fn run_compare_validate<I, W, A>(args: I, writer: &mut W, adapter: &A) -> i32
where
    I: Iterator<Item = String>,
    W: Write,
    A: GameAdapter,
{
    let mut candidate_configs: Vec<String> = Vec::new();
    let mut baseline_config: Option<String> = None;
    let mut game_config: Option<String> = None;
    let result = (|| -> Result<CompareValidationResponse, HostError> {
        let mut args = args;
        while let Some(flag) = args.next() {
            let value = |name: &str, args: &mut I| {
                args.next()
                    .ok_or_else(|| HostError::bad_request(format!("missing value for {name}")))
            };
            match flag.as_str() {
                "--candidate-config" => candidate_configs.push(value(&flag, &mut args)?),
                "--baseline-config" => baseline_config = Some(value(&flag, &mut args)?),
                "--game-config" => game_config = Some(value(&flag, &mut args)?),
                _ => return Err(HostError::bad_request(format!("unknown flag: {flag}"))),
            }
        }
        if candidate_configs.is_empty() {
            return Err(HostError::bad_request("missing --candidate-config"));
        }
        let candidate_configs = candidate_configs
            .into_iter()
            .map(|raw| {
                serde_json::from_str(&raw).map_err(|e| {
                    HostError::bad_request(format!("invalid --candidate-config JSON: {e}"))
                })
            })
            .collect::<Result<Vec<Value>, HostError>>()?;
        let baseline_config = serde_json::from_str(
            &baseline_config.ok_or_else(|| HostError::bad_request("missing --baseline-config"))?,
        )
        .map_err(|e| HostError::bad_request(format!("invalid --baseline-config JSON: {e}")))?;
        let game_config = game_config
            .map(|raw| {
                serde_json::from_str(&raw)
                    .map_err(|e| HostError::bad_request(format!("invalid --game-config JSON: {e}")))
            })
            .transpose()?;
        let errors = if candidate_configs.len() == 1 {
            adapter.validate_compare(
                candidate_configs.into_iter().next().expect("one candidate"),
                baseline_config,
                game_config,
            )
        } else {
            adapter.validate_compare_many(candidate_configs, baseline_config, game_config)
        };
        Ok(CompareValidationResponse {
            valid: errors.is_empty(),
            errors,
        })
    })();
    match result {
        Ok(response) => {
            let json = serde_json::to_string(&response).expect("validation response serializes");
            let _ = writeln!(writer, "{json}");
            if response.valid {
                0
            } else {
                1
            }
        }
        Err(error) => {
            eprintln!("{error}");
            1
        }
    }
}

/// Parses and runs the foreground configured comparison command. Validation
/// is completed before invoking the adapter so malformed invocations cannot
/// start a game or write partial output.
fn run_compare_eval<I, W, A>(args: I, writer: &mut W, adapter: &A) -> i32
where
    I: Iterator<Item = String>,
    W: Write,
    A: GameAdapter,
{
    let mut args = args;
    let mut candidate_config: Option<String> = None;
    let mut baseline_config: Option<String> = None;
    let mut rounds: Option<u32> = None;
    let mut seed: Option<u64> = None;
    let mut max_iterations: Option<usize> = None;
    let mut max_time_ms: Option<u64> = None;
    let mut game_config: Option<String> = None;
    let mut trace_path: Option<String> = None;

    let result = (|| -> Result<(), HostError> {
        while let Some(flag) = args.next() {
            let value = |name: &str, args: &mut I| {
                args.next()
                    .ok_or_else(|| HostError::bad_request(format!("missing value for {name}")))
            };
            match flag.as_str() {
                "--candidate-config" => candidate_config = Some(value(&flag, &mut args)?),
                "--baseline-config" => baseline_config = Some(value(&flag, &mut args)?),
                "--rounds" => {
                    let raw = value(&flag, &mut args)?;
                    rounds = Some(
                        raw.parse()
                            .map_err(|_| HostError::bad_request("invalid --rounds"))?,
                    );
                }
                "--seed" => {
                    let raw = value(&flag, &mut args)?;
                    seed = Some(
                        raw.parse()
                            .map_err(|_| HostError::bad_request("invalid --seed"))?,
                    );
                }
                "--max-iterations" => {
                    let raw = value(&flag, &mut args)?;
                    max_iterations = Some(
                        raw.parse()
                            .map_err(|_| HostError::bad_request("invalid --max-iterations"))?,
                    );
                }
                "--max-time-ms" => {
                    let raw = value(&flag, &mut args)?;
                    max_time_ms = Some(
                        raw.parse()
                            .map_err(|_| HostError::bad_request("invalid --max-time-ms"))?,
                    );
                }
                "--game-config" => game_config = Some(value(&flag, &mut args)?),
                "--trace-path" => trace_path = Some(value(&flag, &mut args)?),
                _ => return Err(HostError::bad_request(format!("unknown flag: {flag}"))),
            }
        }

        let candidate_config =
            candidate_config.ok_or_else(|| HostError::bad_request("missing --candidate-config"))?;
        let baseline_config =
            baseline_config.ok_or_else(|| HostError::bad_request("missing --baseline-config"))?;
        let rounds = rounds.ok_or_else(|| HostError::bad_request("missing --rounds"))?;
        if rounds == 0 {
            return Err(HostError::bad_request("--rounds must be positive"));
        }
        let seed = seed.ok_or_else(|| HostError::bad_request("missing --seed"))?;
        if max_iterations.is_some() == max_time_ms.is_some() {
            return Err(HostError::bad_request(
                "exactly one of --max-iterations and --max-time-ms is required",
            ));
        }
        if max_iterations == Some(0) {
            return Err(HostError::bad_request("--max-iterations must be positive"));
        }
        if max_time_ms == Some(0) {
            return Err(HostError::bad_request("--max-time-ms must be positive"));
        }
        let candidate_config: Value = serde_json::from_str(&candidate_config)
            .map_err(|e| HostError::bad_request(format!("invalid --candidate-config JSON: {e}")))?;
        let baseline_config: Value = serde_json::from_str(&baseline_config)
            .map_err(|e| HostError::bad_request(format!("invalid --baseline-config JSON: {e}")))?;
        let game_config = game_config
            .map(|raw| {
                serde_json::from_str(&raw)
                    .map_err(|e| HostError::bad_request(format!("invalid --game-config JSON: {e}")))
            })
            .transpose()?;

        let mut sequence = 0_u64;
        let (mut wins, mut losses, mut draws) = (0_u32, 0_u32, 0_u32);
        for round_index in 0..rounds {
            let round = round_index + 1;
            let round_seed = derive_seed(seed, u64::from(round_index));
            let candidate_config = candidate_config.clone();
            let baseline_config = baseline_config.clone();
            let game_config = game_config.clone();
            let trace_path = trace_path.clone();
            let mut on_game = |mut record: ConfiguredMatchResult| -> Result<(), HostError> {
                sequence = sequence
                    .checked_add(1)
                    .ok_or_else(|| HostError::internal("comparison sequence overflow"))?;
                record.seq = sequence;
                record.round = round;
                record.seed = round_seed;
                let json = serde_json::to_string(&record).map_err(|e| {
                    HostError::internal(format!("failed to serialize match result: {e}"))
                })?;
                writeln!(writer, "{json}")
                    .and_then(|_| writer.flush())
                    .map_err(|e| HostError::internal(format!("failed to write match result: {e}")))
            };
            let value = adapter.tune_eval(
                candidate_config,
                1,
                Some(round_seed),
                None,
                Some(baseline_config),
                game_config,
                max_iterations,
                max_time_ms,
                trace_path.map(std::path::PathBuf::from),
                &mut on_game,
            )?;
            wins =
                wins.saturating_add(value["wins"].as_u64().ok_or_else(|| {
                    HostError::internal("configured comparison returned invalid wins")
                })? as u32);
            losses = losses.saturating_add(value["losses"].as_u64().ok_or_else(|| {
                HostError::internal("configured comparison returned invalid losses")
            })? as u32);
            draws = draws.saturating_add(value["draws"].as_u64().ok_or_else(|| {
                HostError::internal("configured comparison returned invalid draws")
            })? as u32);
        }
        let summary = ConfiguredComparisonSummary {
            record_type: "configured_comparison_summary".into(),
            games: wins.saturating_add(losses).saturating_add(draws),
            wins,
            losses,
            draws,
        };
        let json = serde_json::to_string(&summary)
            .map_err(|e| HostError::internal(format!("failed to serialize summary: {e}")))?;
        writeln!(writer, "{json}")
            .and_then(|_| writer.flush())
            .map_err(|e| HostError::internal(format!("failed to write summary: {e}")))?;
        Ok(())
    })();

    match result {
        Ok(()) => 0,
        Err(e) => {
            eprintln!("{e}");
            1
        }
    }
}

/// Parses `--rounds <n> [--seed <n>] [--game-config <json>]` from the
/// remaining CLI args, calls [`GameAdapter::book_build`], and prints its
/// JSON result (the serialized opening book) verbatim to `writer` -- same
/// shape as [`run_tune_eval`], including leaving the actual work (and the
/// question of where the resulting book gets saved) to the caller, which
/// redirects stdout to a file same as it would for `tune eval`.
fn run_book_build<I, W, A>(args: I, writer: &mut W, adapter: &A) -> i32
where
    I: Iterator<Item = String>,
    W: Write,
    A: GameAdapter,
{
    let mut args = args;
    let mut rounds: Option<u32> = None;
    let mut seed: Option<u64> = None;
    let mut game_config: Option<String> = None;
    while let Some(flag) = args.next() {
        match flag.as_str() {
            "--rounds" => rounds = args.next().and_then(|s| s.parse().ok()),
            "--seed" => seed = args.next().and_then(|s| s.parse().ok()),
            "--game-config" => game_config = args.next(),
            _ => {}
        }
    }

    let result = (|| -> Result<Value, HostError> {
        let rounds = rounds.ok_or_else(|| HostError::bad_request("missing --rounds"))?;
        let game_config = game_config
            .map(|s| {
                serde_json::from_str(&s)
                    .map_err(|e| HostError::bad_request(format!("invalid --game-config JSON: {e}")))
            })
            .transpose()?;
        adapter.book_build(rounds, seed, game_config)
    })();

    match result {
        Ok(v) => {
            let json = serde_json::to_string(&v).expect("book_build result always serializes");
            let _ = writeln!(writer, "{json}");
            0
        }
        Err(e) => {
            eprintln!("{e}");
            1
        }
    }
}

// ---------------------------------------------------------------------------
// Dispatch
// ---------------------------------------------------------------------------

fn dispatch<A: GameAdapter>(adapter: &A, req: &Request) -> Result<Value, HostError> {
    match req.method.as_str() {
        // --- Metadata (no params needed) ---
        "kind" => ok_value(adapter.kind()),
        "label" => ok_value(adapter.label()),
        "description" => ok_value(adapter.description()),
        "default_config" => Ok(adapter.default_config()),

        // --- State methods ---
        "new" => adapter.new_state(req.params["config"].clone()),

        "legal_moves" => {
            let state = param(&req.params, "state")?;
            adapter.legal_moves(state).and_then(ok_value)
        }

        "apply" => {
            let state = param(&req.params, "state")?;
            let mv = param(&req.params, "move")?;
            adapter.apply(state, mv)
        }

        "view" => {
            let state = param(&req.params, "state")?;
            adapter.view(state)
        }

        "terminal" => {
            let state = param(&req.params, "state")?;
            view_terminal(adapter, state)
        }

        // --- AI methods ---
        "ai_presets" => ok_value(adapter.ai_presets()),
        "tuner" => ok_value(adapter.tuner()),

        "ai_move" => {
            let state = param(&req.params, "state")?;
            let preset = param_str(&req.params, "preset")?;
            let custom = req.params.get("custom");
            adapter.ai_move(state, preset, custom).and_then(ok_value)
        }

        "analyze" => {
            let state = param(&req.params, "state")?;
            let preset = param_str(&req.params, "preset")?;
            let custom = req.params.get("custom");
            let budget_ms = req.params.get("budget_ms").and_then(|v| v.as_u64());
            adapter
                .analyze(state, preset, custom, budget_ms)
                .and_then(ok_value)
        }

        other => Err(HostError::not_found(format!("unknown method: {other}"))),
    }
}

// ---------------------------------------------------------------------------
// Dispatch helpers
// ---------------------------------------------------------------------------

/// Extract a named field from a JSON object.
fn param<'a>(params: &'a Value, name: &str) -> Result<&'a Value, HostError> {
    params
        .get(name)
        .ok_or_else(|| HostError::bad_request(format!("missing parameter: {name}")))
}

/// Extract a named string field from a JSON object.
fn param_str<'a>(params: &'a Value, name: &str) -> Result<&'a str, HostError> {
    let v = param(params, name)?;
    v.as_str()
        .ok_or_else(|| HostError::bad_request(format!("parameter {name} must be a string")))
}

/// Serialize a value to JSON Value.
fn ok_value<T: serde::Serialize>(t: T) -> Result<Value, HostError> {
    serde_json::to_value(t).map_err(|e| HostError::internal(format!("serialization: {e}")))
}

/// Extract terminal/winner info from a state via the `view` method.
fn view_terminal<A: GameAdapter>(adapter: &A, state: &Value) -> Result<Value, HostError> {
    let view = adapter.view(state)?;
    let terminal = view
        .get("terminal")
        .and_then(|t| t.as_bool())
        .unwrap_or(false);
    Ok(serde_json::json!({
        "terminal": terminal,
        "winner": view.get("winner"),
    }))
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::RefCell;
    use std::io::Cursor;

    #[derive(Default)]
    struct ValidationCounts {
        new_state: usize,
        builds: usize,
        plays: usize,
    }

    thread_local! {
        static VALIDATION_COUNTS: RefCell<ValidationCounts> = RefCell::new(ValidationCounts::default());
    }

    /// A minimal fake adapter for testing the protocol dispatch loop.
    /// Responds with just enough data to verify round-trip correctness.
    struct FakeAdapter;

    impl GameAdapter for FakeAdapter {
        fn kind(&self) -> &'static str {
            "fake"
        }
        fn label(&self) -> &'static str {
            "Fake Game"
        }
        fn description(&self) -> &'static str {
            "A minimal fake adapter for testing"
        }

        fn default_config(&self) -> Value {
            serde_json::json!({})
        }

        fn new_state(&self, _config: Value) -> Result<Value, HostError> {
            Ok(serde_json::json!({"board": [], "turn": "X"}))
        }

        fn legal_moves(&self, _state: &Value) -> Result<Vec<Value>, HostError> {
            Ok(vec![serde_json::json!(0), serde_json::json!(1)])
        }

        fn apply(&self, state: &Value, mv: &Value) -> Result<Value, HostError> {
            let turn = state.get("turn").and_then(|t| t.as_str()).unwrap_or("X");
            let next_turn = if turn == "X" { "O" } else { "X" };
            Ok(serde_json::json!({
                "board": [mv],
                "turn": next_turn,
            }))
        }

        fn view(&self, state: &Value) -> Result<Value, HostError> {
            Ok(serde_json::json!({
                "terminal": false,
                "turn": state.get("turn"),
            }))
        }

        fn ai_presets(&self) -> Vec<AiPresetInfo> {
            vec![AiPresetInfo {
                id: "random".into(),
                label: "Random".into(),
                description: "Picks a random legal move".into(),
            }]
        }

        fn ai_move(
            &self,
            state: &Value,
            preset: &str,
            _custom: Option<&Value>,
        ) -> Result<AiMoveResult, HostError> {
            if preset == "random" {
                let next = self.apply(state, &serde_json::json!(0))?;
                Ok(AiMoveResult {
                    mv: serde_json::json!(0),
                    state: next,
                })
            } else {
                Err(HostError::not_found(format!("unknown preset: {preset}")))
            }
        }

        fn analyze(
            &self,
            _state: &Value,
            preset: &str,
            _custom: Option<&Value>,
            _budget_ms: Option<u64>,
        ) -> Result<Analysis, HostError> {
            if preset != "random" {
                return Err(HostError::not_found(format!("unknown preset: {preset}")));
            }
            let mv = serde_json::json!(0);
            Ok(Analysis {
                actions: vec![AnalysisAction {
                    action: mv.clone(),
                    visits: 10,
                    mean_value: 0.5,
                    is_proven: false,
                }],
                principal_variation: vec![mv],
                total_visits: 10,
                suggested_move: Some(serde_json::json!(0)),
            })
        }
    }

    // -----------------------------------------------------------------------
    // Helper: send JSON lines into run_host and collect responses
    // -----------------------------------------------------------------------

    fn send_requests(lines: &[&str]) -> Vec<String> {
        let input = Cursor::new(lines.join("\n"));
        let mut output = Cursor::new(Vec::new());
        run_host(input, &mut output, FakeAdapter);
        let raw = String::from_utf8(output.into_inner()).unwrap();
        raw.lines().map(|l| l.to_owned()).collect()
    }

    fn parse_response(line: &str) -> Response {
        serde_json::from_str(line).unwrap()
    }

    // -----------------------------------------------------------------------
    // Tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_kind() {
        let lines = send_requests(&[r#"{"id":1,"method":"kind","params":{}}"#]);
        assert_eq!(lines.len(), 1);
        let resp = parse_response(&lines[0]);
        match resp {
            Response::Success { id, result } => {
                assert_eq!(id, 1);
                assert_eq!(result, "fake");
            }
            _ => panic!("expected success response"),
        }
    }

    #[test]
    fn test_label() {
        let lines = send_requests(&[r#"{"id":2,"method":"label","params":{}}"#]);
        assert_eq!(lines.len(), 1);
        let resp = parse_response(&lines[0]);
        match resp {
            Response::Success { id, result } => {
                assert_eq!(id, 2);
                assert_eq!(result, "Fake Game");
            }
            _ => panic!("expected success response"),
        }
    }

    #[test]
    fn test_description() {
        let lines = send_requests(&[r#"{"id":3,"method":"description","params":{}}"#]);
        assert_eq!(lines.len(), 1);
        let resp = parse_response(&lines[0]);
        match resp {
            Response::Success { id, result } => {
                assert_eq!(id, 3);
                assert!(result.as_str().unwrap().contains("fake"));
            }
            _ => panic!("expected success response"),
        }
    }

    #[test]
    fn test_default_config() {
        let lines = send_requests(&[r#"{"id":4,"method":"default_config","params":{}}"#]);
        assert_eq!(lines.len(), 1);
        let resp = parse_response(&lines[0]);
        match resp {
            Response::Success { id, result } => {
                assert_eq!(id, 4);
                assert_eq!(result, serde_json::json!({}));
            }
            _ => panic!("expected success response"),
        }
    }

    #[test]
    fn test_new_state() {
        let lines = send_requests(&[r#"{"id":5,"method":"new","params":{"config":{}}}"#]);
        assert_eq!(lines.len(), 1);
        let resp = parse_response(&lines[0]);
        match resp {
            Response::Success { id, result } => {
                assert_eq!(id, 5);
                assert_eq!(result.get("turn").and_then(|t| t.as_str()), Some("X"));
            }
            _ => panic!("expected success response"),
        }
    }

    #[test]
    fn test_legal_moves() {
        let lines = send_requests(&[
            r#"{"id":6,"method":"new","params":{"config":{}}}"#,
            r#"{"id":7,"method":"legal_moves","params":{"state":{"board":[],"turn":"X"}}}"#,
        ]);
        assert_eq!(lines.len(), 2);
        let resp = parse_response(&lines[1]);
        match resp {
            Response::Success { id, result } => {
                assert_eq!(id, 7);
                let moves = result.as_array().unwrap();
                assert_eq!(moves.len(), 2);
                assert_eq!(moves[0], 0);
                assert_eq!(moves[1], 1);
            }
            _ => panic!("expected success response"),
        }
    }

    #[test]
    fn test_apply() {
        let lines = send_requests(&[
            r#"{"id":8,"method":"new","params":{"config":{}}}"#,
            r#"{"id":9,"method":"apply","params":{"state":{"board":[],"turn":"X"},"move":0}}"#,
        ]);
        assert_eq!(lines.len(), 2);
        let resp = parse_response(&lines[1]);
        match resp {
            Response::Success { id, result } => {
                assert_eq!(id, 9);
                assert_eq!(result.get("turn").and_then(|t| t.as_str()), Some("O"));
            }
            _ => panic!("expected success response"),
        }
    }

    #[test]
    fn test_apply_missing_params() {
        let lines = send_requests(&[r#"{"id":10,"method":"apply","params":{}}"#]);
        assert_eq!(lines.len(), 1);
        let resp = parse_response(&lines[0]);
        match resp {
            Response::Error { id, error } => {
                assert_eq!(id, 10);
                assert_eq!(error.code, 400);
                assert!(error.message.contains("missing parameter"));
            }
            _ => panic!("expected error response"),
        }
    }

    #[test]
    fn test_view() {
        let lines = send_requests(&[
            r#"{"id":11,"method":"new","params":{"config":{}}}"#,
            r#"{"id":12,"method":"view","params":{"state":{"board":[],"turn":"X"}}}"#,
        ]);
        assert_eq!(lines.len(), 2);
        let resp = parse_response(&lines[1]);
        match resp {
            Response::Success { id, result } => {
                assert_eq!(id, 12);
                assert_eq!(
                    result.get("terminal").and_then(|t| t.as_bool()),
                    Some(false)
                );
                assert_eq!(result.get("turn").and_then(|t| t.as_str()), Some("X"));
            }
            _ => panic!("expected success response"),
        }
    }

    #[test]
    fn test_terminal() {
        let lines = send_requests(&[
            r#"{"id":13,"method":"terminal","params":{"state":{"board":[],"turn":"X"}}}"#,
        ]);
        assert_eq!(lines.len(), 1);
        let resp = parse_response(&lines[0]);
        match resp {
            Response::Success { id, result } => {
                assert_eq!(id, 13);
                assert_eq!(
                    result.get("terminal").and_then(|t| t.as_bool()),
                    Some(false)
                );
            }
            _ => panic!("expected success response"),
        }
    }

    #[test]
    fn test_ai_presets() {
        let lines = send_requests(&[r#"{"id":14,"method":"ai_presets","params":{}}"#]);
        assert_eq!(lines.len(), 1);
        let resp = parse_response(&lines[0]);
        match resp {
            Response::Success { id, result } => {
                assert_eq!(id, 14);
                let presets = result.as_array().unwrap();
                assert_eq!(presets.len(), 1);
                assert_eq!(
                    presets[0].get("id").and_then(|v| v.as_str()),
                    Some("random")
                );
            }
            _ => panic!("expected success response"),
        }
    }

    #[test]
    fn test_ai_move() {
        let lines = send_requests(&[
            r#"{"id":15,"method":"new","params":{"config":{}}}"#,
            r#"{"id":16,"method":"ai_move","params":{"state":{"board":[],"turn":"X"},"preset":"random"}}"#,
        ]);
        assert_eq!(lines.len(), 2);
        let resp = parse_response(&lines[1]);
        match resp {
            Response::Success { id, result } => {
                assert_eq!(id, 16);
                assert_eq!(result.get("mv").and_then(|v| v.as_u64()), Some(0));
                assert!(result.get("state").is_some());
            }
            _ => panic!("expected success response"),
        }
    }

    #[test]
    fn test_ai_move_unknown_preset() {
        let lines = send_requests(&[
            r#"{"id":17,"method":"new","params":{"config":{}}}"#,
            r#"{"id":18,"method":"ai_move","params":{"state":{"board":[],"turn":"X"},"preset":"nope"}}"#,
        ]);
        assert_eq!(lines.len(), 2);
        let resp = parse_response(&lines[1]);
        match resp {
            Response::Error { id, error } => {
                assert_eq!(id, 18);
                assert_eq!(error.code, 404);
            }
            _ => panic!("expected error response"),
        }
    }

    #[test]
    fn test_analyze() {
        let lines = send_requests(&[
            r#"{"id":19,"method":"new","params":{"config":{}}}"#,
            r#"{"id":20,"method":"analyze","params":{"state":{"board":[],"turn":"X"},"preset":"random"}}"#,
        ]);
        assert_eq!(lines.len(), 2);
        let resp = parse_response(&lines[1]);
        match resp {
            Response::Success { id, result } => {
                assert_eq!(id, 20);
                let actions = result.get("actions").and_then(|a| a.as_array()).unwrap();
                assert_eq!(actions.len(), 1);
                assert_eq!(actions[0].get("visits").and_then(|v| v.as_u64()), Some(10));
            }
            _ => panic!("expected success response"),
        }
    }

    #[test]
    fn test_unknown_method() {
        let lines = send_requests(&[r#"{"id":99,"method":"nonexistent","params":{}}"#]);
        assert_eq!(lines.len(), 1);
        let resp = parse_response(&lines[0]);
        match resp {
            Response::Error { id, error } => {
                assert_eq!(id, 99);
                assert_eq!(error.code, 404);
            }
            _ => panic!("expected error response"),
        }
    }

    #[test]
    fn test_malformed_json() {
        let lines = send_requests(&["this is not json"]);
        assert_eq!(lines.len(), 1);
        let resp = parse_response(&lines[0]);
        match resp {
            Response::Error { id, error } => {
                assert_eq!(id, 0);
                assert_eq!(error.code, 400);
                assert!(error.message.contains("invalid request"));
            }
            _ => panic!("expected error response"),
        }
    }

    #[test]
    fn test_multiple_requests() {
        let lines = send_requests(&[
            r#"{"id":1,"method":"kind","params":{}}"#,
            r#"{"id":2,"method":"label","params":{}}"#,
            r#"{"id":3,"method":"ai_presets","params":{}}"#,
        ]);
        assert_eq!(lines.len(), 3);
        for (i, line) in lines.iter().enumerate() {
            let resp = parse_response(line);
            match resp {
                Response::Success { id, .. } => assert_eq!(id, (i + 1) as u64),
                _ => panic!("expected success for request {}", i + 1),
            }
        }
    }

    #[test]
    fn test_blank_lines_are_skipped() {
        let lines = send_requests(&[
            r#"{"id":1,"method":"kind","params":{}}"#,
            "", // blank line
            r#"{"id":2,"method":"label","params":{}}"#,
        ]);
        // blank line produces no output, so we get 2 responses
        assert_eq!(lines.len(), 2);
    }

    // -----------------------------------------------------------------------
    // run_cli tests
    // -----------------------------------------------------------------------

    /// A second fake adapter that also implements `tuner`/`tune_eval`, for
    /// testing the `tune` subcommand family without polluting `FakeAdapter`
    /// (used bare across dozens of protocol tests above) with tuning state.
    struct TunableFakeAdapter;

    impl GameAdapter for TunableFakeAdapter {
        fn kind(&self) -> &'static str {
            "tunable-fake"
        }
        fn label(&self) -> &'static str {
            "Tunable Fake Game"
        }
        fn description(&self) -> &'static str {
            "A fake adapter that supports tuning, for testing `tune` subcommands"
        }
        fn default_config(&self) -> Value {
            serde_json::json!({})
        }
        fn new_state(&self, config: Value) -> Result<Value, HostError> {
            VALIDATION_COUNTS.with(|counts| counts.borrow_mut().new_state += 1);
            if config.get("invalid").and_then(Value::as_str) == Some("game") {
                return Err(HostError::bad_request("game rejected"));
            }
            Ok(serde_json::json!({}))
        }
        fn legal_moves(&self, _state: &Value) -> Result<Vec<Value>, HostError> {
            Ok(vec![])
        }
        fn apply(&self, state: &Value, _mv: &Value) -> Result<Value, HostError> {
            Ok(state.clone())
        }
        fn view(&self, _state: &Value) -> Result<Value, HostError> {
            Ok(serde_json::json!({"terminal": true}))
        }
        fn ai_presets(&self) -> Vec<AiPresetInfo> {
            vec![]
        }
        fn ai_move(
            &self,
            _state: &Value,
            _preset: &str,
            _custom: Option<&Value>,
        ) -> Result<AiMoveResult, HostError> {
            VALIDATION_COUNTS.with(|counts| counts.borrow_mut().plays += 1);
            Err(HostError::not_found("not implemented in test fake"))
        }
        fn analyze(
            &self,
            _state: &Value,
            _preset: &str,
            _custom: Option<&Value>,
            _budget_ms: Option<u64>,
        ) -> Result<Analysis, HostError> {
            Err(HostError::not_found("not implemented in test fake"))
        }

        fn tuner(&self) -> Option<TunerInfo> {
            Some(TunerInfo {
                id: "test".into(),
                baselines: vec!["baseline".into()],
                eval_rounds: 5,
                parameters: vec![TunerParameter {
                    name: "c".into(),
                    spec: serde_json::json!({"type": "float", "bounds": [0, 3], "default": 1.4}),
                }],
                conditions: vec![],
                game_config: serde_json::json!({}),
            })
        }

        fn tune_eval(
            &self,
            params: Value,
            rounds: u32,
            seed: Option<u64>,
            baseline: Option<String>,
            baseline_config: Option<Value>,
            game_config: Option<Value>,
            max_iterations: Option<usize>,
            max_time_ms: Option<u64>,
            trace_path: Option<std::path::PathBuf>,
            on_game: &mut dyn FnMut(ConfiguredMatchResult) -> Result<(), HostError>,
        ) -> Result<Value, HostError> {
            VALIDATION_COUNTS.with(|counts| counts.borrow_mut().builds += 1);
            match params.get("invalid").and_then(Value::as_str) {
                Some("candidate") => return Err(HostError::bad_request("candidate rejected")),
                Some("baseline") => return Err(HostError::bad_request("baseline rejected")),
                _ => {}
            }
            let _ = (max_iterations, max_time_ms, trace_path);
            for round in 1..=rounds {
                on_game(ConfiguredMatchResult {
                    record_type: "configured_match_result".into(),
                    seq: (round * 2 - 1) as u64,
                    round,
                    seed: seed.unwrap_or(0),
                    candidate_side: ConfiguredCandidateSide::First,
                    outcome: ConfiguredOutcome::CandidateWin,
                    trace_game_seq: None,
                    plies: 0,
                    elapsed_ms: 0,
                    candidate: ConfiguredStrategyMetrics::default(),
                    baseline: ConfiguredStrategyMetrics::default(),
                })?;
                on_game(ConfiguredMatchResult {
                    record_type: "configured_match_result".into(),
                    seq: (round * 2) as u64,
                    round,
                    seed: seed.unwrap_or(0),
                    candidate_side: ConfiguredCandidateSide::Second,
                    outcome: ConfiguredOutcome::BaselineWin,
                    trace_game_seq: None,
                    plies: 0,
                    elapsed_ms: 0,
                    candidate: ConfiguredStrategyMetrics::default(),
                    baseline: ConfiguredStrategyMetrics::default(),
                })?;
            }
            Ok(serde_json::json!({
                "cost": 0.25,
                "params": params,
                "rounds": rounds,
                "seed": seed,
                "baseline": baseline,
                "baseline_config": baseline_config,
                "game_config": game_config,
                "wins": rounds,
                "losses": rounds,
                "draws": 0,
            }))
        }
    }

    fn run_cli_capture_with<A: GameAdapter>(
        adapter: A,
        args: &[&str],
        stdin: &str,
    ) -> (String, i32) {
        let args = args.iter().map(|s| s.to_string());
        let input = Cursor::new(stdin.to_owned());
        let mut output = Cursor::new(Vec::new());
        let code = run_cli_with(args, input, &mut output, adapter);
        (String::from_utf8(output.into_inner()).unwrap(), code)
    }

    fn run_cli_capture(args: &[&str], stdin: &str) -> (String, i32) {
        run_cli_capture_with(FakeAdapter, args, stdin)
    }

    #[test]
    fn test_run_cli_no_args_drives_stdin_stdout_loop_unchanged() {
        let (out, code) = run_cli_capture(&[], "{\"id\":1,\"method\":\"kind\",\"params\":{}}\n");
        assert_eq!(code, 0);
        let resp = parse_response(out.lines().next().unwrap());
        match resp {
            Response::Success { id, result } => {
                assert_eq!(id, 1);
                assert_eq!(result, "fake");
            }
            _ => panic!("expected success response"),
        }
    }

    #[test]
    fn test_run_cli_describe_matches_adapter_fields() {
        let (out, code) = run_cli_capture(&["describe"], "");
        assert_eq!(code, 0);
        let lines: Vec<&str> = out.lines().collect();
        assert_eq!(lines.len(), 1);
        let description: GameDescription = serde_json::from_str(lines[0]).unwrap();

        let adapter = FakeAdapter;
        assert_eq!(description.kind, adapter.kind());
        assert_eq!(description.label, adapter.label());
        assert_eq!(description.description, adapter.description());
        assert_eq!(description.default_config, adapter.default_config());
        assert_eq!(description.ai_presets.len(), adapter.ai_presets().len());
        assert_eq!(description.ai_presets[0].id, adapter.ai_presets()[0].id);
        assert!(description.tuning.is_none());
    }

    #[test]
    fn test_run_cli_describe_folds_in_tuning_when_present() {
        let (out, code) = run_cli_capture_with(TunableFakeAdapter, &["describe"], "");
        assert_eq!(code, 0);
        let description: GameDescription =
            serde_json::from_str(out.lines().next().unwrap()).unwrap();
        let tuning = description.tuning.expect("expected tuning metadata");
        assert_eq!(tuning.id, "test");
        assert_eq!(tuning.eval_rounds, 5);
    }

    #[test]
    fn test_run_cli_unknown_subcommand_falls_back_to_stdin_stdout_loop() {
        let (out, code) = run_cli_capture(
            &["some-unknown-flag"],
            "{\"id\":7,\"method\":\"kind\",\"params\":{}}\n",
        );
        assert_eq!(code, 0);
        let resp = parse_response(out.lines().next().unwrap());
        match resp {
            Response::Success { id, result } => {
                assert_eq!(id, 7);
                assert_eq!(result, "fake");
            }
            _ => panic!("expected success response, describe-only args must not error"),
        }
    }

    #[test]
    fn test_run_cli_tune_describe_unsupported_when_tuner_none() {
        let (out, code) = run_cli_capture(&["tune", "describe"], "");
        assert_eq!(code, 1);
        assert!(out.is_empty());
    }

    #[test]
    fn test_run_cli_tune_describe_prints_tuner_info() {
        let (out, code) = run_cli_capture_with(TunableFakeAdapter, &["tune", "describe"], "");
        assert_eq!(code, 0);
        let info: TunerInfo = serde_json::from_str(out.lines().next().unwrap()).unwrap();
        assert_eq!(info.id, "test");
        assert_eq!(info.baselines, vec!["baseline".to_string()]);
        assert_eq!(info.eval_rounds, 5);
        assert_eq!(info.parameters.len(), 1);
        assert_eq!(info.parameters[0].name, "c");
    }

    #[test]
    fn test_run_cli_compare_describe_matches_tune_describe() {
        let (tune, tune_code) = run_cli_capture_with(TunableFakeAdapter, &["tune", "describe"], "");
        let (compare, compare_code) =
            run_cli_capture_with(TunableFakeAdapter, &["compare", "describe"], "");
        assert_eq!(tune_code, 0);
        assert_eq!(compare_code, 0);
        assert_eq!(compare, tune);
    }

    #[test]
    fn test_run_cli_compare_eval_streams_games_then_summary() {
        let (out, code) = run_cli_capture_with(
            TunableFakeAdapter,
            &[
                "compare",
                "eval",
                "--candidate-config",
                "{}",
                "--baseline-config",
                "{}",
                "--rounds",
                "1",
                "--seed",
                "42",
                "--max-iterations",
                "1",
            ],
            "",
        );
        assert_eq!(code, 0);
        let lines: Vec<Value> = out
            .lines()
            .map(|line| serde_json::from_str(line).unwrap())
            .collect();
        assert_eq!(lines.len(), 3);
        assert_eq!(lines[0]["type"], "configured_match_result");
        assert_eq!(lines[0]["seq"], 1);
        assert_eq!(lines[0]["candidate_side"], "first");
        assert_eq!(lines[1]["seq"], 2);
        assert_eq!(lines[1]["candidate_side"], "second");
        assert_eq!(lines[2]["type"], "configured_comparison_summary");
        assert_eq!(lines[2]["games"], 2);
        assert_eq!(lines[2]["wins"], 1);
        assert_eq!(lines[2]["losses"], 1);
    }

    #[test]
    fn compare_eval_uses_stable_round_seeds_and_run_sequences() {
        let (out, code) = run_cli_capture_with(
            TunableFakeAdapter,
            &[
                "compare",
                "eval",
                "--candidate-config",
                "{}",
                "--baseline-config",
                "{}",
                "--rounds",
                "2",
                "--seed",
                "42",
                "--max-iterations",
                "1",
            ],
            "",
        );
        assert_eq!(code, 0);
        let lines: Vec<Value> = out
            .lines()
            .map(|line| serde_json::from_str(line).unwrap())
            .collect();
        assert_eq!(lines.len(), 5);
        for (index, line) in lines[..4].iter().enumerate() {
            assert_eq!(line["seq"], (index + 1) as u64);
            assert_eq!(line["round"], (index / 2 + 1) as u64);
            assert_eq!(line["seed"], derive_seed(42, (index / 2) as u64));
            assert_eq!(line["seed"], lines[index ^ 1]["seed"]);
        }
        assert_eq!(lines[0]["seed"], derive_seed(42, 0));
        assert_ne!(lines[0]["seed"], lines[2]["seed"]);
        assert_eq!(lines[4]["games"], 4);
    }

    #[test]
    fn compare_validate_returns_structured_success_without_matches() {
        let (out, code) = run_cli_capture_with(
            TunableFakeAdapter,
            &[
                "compare",
                "validate",
                "--candidate-config",
                "{}",
                "--baseline-config",
                "{}",
            ],
            "",
        );
        assert_eq!(code, 0);
        let response: Value = serde_json::from_str(out.trim()).unwrap();
        assert_eq!(response["valid"], true);
        assert_eq!(response["errors"], serde_json::json!([]));
    }

    fn validation_counts() -> ValidationCounts {
        VALIDATION_COUNTS.with(|counts| std::mem::take(&mut *counts.borrow_mut()))
    }

    #[test]
    fn compare_validate_checks_game_and_strategies_without_playing() {
        let _ = validation_counts();
        let (out, code) = run_cli_capture_with(
            TunableFakeAdapter,
            &[
                "compare",
                "validate",
                "--candidate-config",
                "{}",
                "--candidate-config",
                "{}",
                "--candidate-config",
                "{}",
                "--baseline-config",
                "{}",
            ],
            "",
        );
        assert_eq!(code, 0);
        let response: Value = serde_json::from_str(out.trim()).unwrap();
        assert_eq!(response["valid"], true);
        let counts = validation_counts();
        assert_eq!(counts.new_state, 1);
        assert_eq!(counts.builds, 4);
        assert_eq!(counts.plays, 0);
    }

    #[test]
    fn compare_validate_attributes_game_candidate_and_baseline_errors() {
        let _ = validation_counts();
        let cases = [
            (
                vec![
                    "compare",
                    "validate",
                    "--candidate-config",
                    "{}",
                    "--baseline-config",
                    "{}",
                    "--game-config",
                    r#"{"invalid":"game"}"#,
                ],
                "game_config",
                "game rejected",
            ),
            (
                vec![
                    "compare",
                    "validate",
                    "--candidate-config",
                    r#"{"invalid":"candidate"}"#,
                    "--baseline-config",
                    "{}",
                ],
                "candidate_config",
                "candidate rejected",
            ),
            (
                vec![
                    "compare",
                    "validate",
                    "--candidate-config",
                    "{}",
                    "--baseline-config",
                    r#"{"invalid":"baseline"}"#,
                ],
                "baseline_config",
                "baseline rejected",
            ),
        ];

        for (args, field, message) in cases {
            let (out, code) = run_cli_capture_with(TunableFakeAdapter, &args, "");
            assert_eq!(code, 1);
            let response: Value = serde_json::from_str(out.trim()).unwrap();
            assert_eq!(response["valid"], false);
            assert_eq!(response["errors"][0]["field"], field);
            assert_eq!(response["errors"][0]["message"], message);
            assert!(!out.contains("configured_match_result"));
            assert!(!out.contains("configured_comparison_summary"));
            assert_eq!(validation_counts().plays, 0);
        }

        let (out, code) = run_cli_capture_with(
            TunableFakeAdapter,
            &[
                "compare",
                "validate",
                "--candidate-config",
                "{}",
                "--candidate-config",
                r#"{"invalid":"candidate"}"#,
                "--candidate-config",
                "{}",
                "--baseline-config",
                "{}",
            ],
            "",
        );
        assert_eq!(code, 1);
        let response: Value = serde_json::from_str(out.trim()).unwrap();
        assert_eq!(response["errors"][0]["field"], "candidate_config");
        assert_eq!(response["errors"][0]["candidate_index"], 1);
        assert_eq!(validation_counts().plays, 0);
    }

    #[test]
    fn tune_eval_rejects_zero_rounds_before_calling_adapter() {
        let (out, code) = run_cli_capture_with(
            TunableFakeAdapter,
            &["tune", "eval", "--config", "{}", "--rounds", "0"],
            "",
        );
        assert_eq!(code, 1);
        assert!(out.is_empty());
    }

    #[test]
    fn test_run_cli_compare_eval_rejects_invalid_invocations_before_play() {
        let base = [
            "compare",
            "eval",
            "--candidate-config",
            "{}",
            "--baseline-config",
            "{}",
            "--rounds",
            "1",
            "--seed",
            "42",
        ];
        let mut invalid = vec![base.to_vec()];
        let mut zero_rounds = base.to_vec();
        zero_rounds[7] = "0";
        zero_rounds.extend(["--max-iterations", "1"]);
        invalid.push(zero_rounds);
        let mut malformed_candidate = base.to_vec();
        malformed_candidate[3] = "not json";
        malformed_candidate.extend(["--max-iterations", "1"]);
        invalid.push(malformed_candidate);
        let mut missing_value = base.to_vec();
        missing_value.push("--max-iterations");
        invalid.push(missing_value);
        for extra in [
            vec!["--max-iterations", "0"],
            vec!["--max-time-ms", "0"],
            vec!["--max-iterations", "1", "--max-time-ms", "1"],
            vec!["--max-iterations", "1", "--unknown"],
        ] {
            let mut args = base.to_vec();
            args.extend(extra);
            invalid.push(args);
        }
        for extra in invalid {
            let (out, code) = run_cli_capture_with(TunableFakeAdapter, &extra, "");
            assert_eq!(code, 1);
            assert!(out.is_empty());
        }
    }

    struct FailingWriter;

    impl Write for FailingWriter {
        fn write(&mut self, _buf: &[u8]) -> io::Result<usize> {
            Err(io::Error::other("write failed"))
        }

        fn flush(&mut self) -> io::Result<()> {
            Err(io::Error::other("flush failed"))
        }
    }

    #[test]
    fn test_run_cli_compare_eval_sink_failure_stops_without_summary() {
        let args = [
            "compare",
            "eval",
            "--candidate-config",
            "{}",
            "--baseline-config",
            "{}",
            "--rounds",
            "2",
            "--seed",
            "42",
            "--max-iterations",
            "1",
        ]
        .into_iter()
        .map(str::to_owned);
        let code = run_cli_with(args, Cursor::new(""), FailingWriter, TunableFakeAdapter);
        assert_eq!(code, 1);
    }

    #[test]
    fn test_run_cli_tune_eval_prints_result_verbatim() {
        let (out, code) = run_cli_capture_with(
            TunableFakeAdapter,
            &[
                "tune",
                "eval",
                "--config",
                r#"{"rave":700,"c":0.3}"#,
                "--rounds",
                "3",
                "--seed",
                "42",
            ],
            "",
        );
        assert_eq!(code, 0);
        let result: Value = serde_json::from_str(out.lines().next().unwrap()).unwrap();
        assert_eq!(result["cost"], 0.25);
        assert_eq!(result["params"]["rave"], 700);
        assert_eq!(result["rounds"], 3);
        assert_eq!(result["seed"], 42);
    }

    #[test]
    fn test_run_cli_tune_eval_missing_rounds_errors() {
        let (out, code) =
            run_cli_capture_with(TunableFakeAdapter, &["tune", "eval", "--config", "{}"], "");
        assert_eq!(code, 1);
        assert!(out.is_empty());
    }

    #[test]
    fn test_run_cli_tune_eval_baseline_config_threads_through() {
        let (out, code) = run_cli_capture_with(
            TunableFakeAdapter,
            &[
                "tune",
                "eval",
                "--config",
                "{}",
                "--rounds",
                "1",
                "--baseline-config",
                r#"{"family":"ucb1","c":1.4}"#,
            ],
            "",
        );
        assert_eq!(code, 0);
        let result: Value = serde_json::from_str(out.lines().next().unwrap()).unwrap();
        assert!(result["baseline"].is_null());
        assert_eq!(result["baseline_config"]["family"], "ucb1");
        assert_eq!(result["baseline_config"]["c"], 1.4);
    }

    #[test]
    fn test_run_cli_tune_eval_rejects_both_baseline_and_baseline_config() {
        let (out, code) = run_cli_capture_with(
            TunableFakeAdapter,
            &[
                "tune",
                "eval",
                "--config",
                "{}",
                "--rounds",
                "1",
                "--baseline",
                "strong",
                "--baseline-config",
                "{}",
            ],
            "",
        );
        assert_eq!(code, 1);
        assert!(out.is_empty());
    }

    #[test]
    fn test_run_cli_tune_eval_rejects_invalid_baseline_config_json() {
        let (out, code) = run_cli_capture_with(
            TunableFakeAdapter,
            &[
                "tune",
                "eval",
                "--config",
                "{}",
                "--rounds",
                "1",
                "--baseline-config",
                "not json",
            ],
            "",
        );
        assert_eq!(code, 1);
        assert!(out.is_empty());
    }

    #[test]
    fn test_run_cli_tune_with_no_further_args_falls_back_to_stdin_stdout_loop() {
        let (out, code) =
            run_cli_capture(&["tune"], "{\"id\":9,\"method\":\"kind\",\"params\":{}}\n");
        assert_eq!(code, 0);
        let resp = parse_response(out.lines().next().unwrap());
        match resp {
            Response::Success { id, result } => {
                assert_eq!(id, 9);
                assert_eq!(result, "fake");
            }
            _ => panic!("expected success response"),
        }
    }

    // -----------------------------------------------------------------------
    // `book` subcommand tests
    // -----------------------------------------------------------------------

    /// A fake adapter that implements `book`/`book_build`, mirroring
    /// `TunableFakeAdapter` above but for the `book` subcommand family --
    /// kept separate for the same reason: `FakeAdapter` stays free of
    /// opt-in state so the dozens of protocol tests above don't have to
    /// think about it.
    struct BookableFakeAdapter;

    impl GameAdapter for BookableFakeAdapter {
        fn kind(&self) -> &'static str {
            "bookable-fake"
        }
        fn label(&self) -> &'static str {
            "Bookable Fake Game"
        }
        fn description(&self) -> &'static str {
            "A fake adapter that supports book generation, for testing `book` subcommands"
        }
        fn default_config(&self) -> Value {
            serde_json::json!({})
        }
        fn new_state(&self, _config: Value) -> Result<Value, HostError> {
            Ok(serde_json::json!({}))
        }
        fn legal_moves(&self, _state: &Value) -> Result<Vec<Value>, HostError> {
            Ok(vec![])
        }
        fn apply(&self, state: &Value, _mv: &Value) -> Result<Value, HostError> {
            Ok(state.clone())
        }
        fn view(&self, _state: &Value) -> Result<Value, HostError> {
            Ok(serde_json::json!({"terminal": true}))
        }
        fn ai_presets(&self) -> Vec<AiPresetInfo> {
            vec![]
        }
        fn ai_move(
            &self,
            _state: &Value,
            _preset: &str,
            _custom: Option<&Value>,
        ) -> Result<AiMoveResult, HostError> {
            Err(HostError::not_found("not implemented in test fake"))
        }
        fn analyze(
            &self,
            _state: &Value,
            _preset: &str,
            _custom: Option<&Value>,
            _budget_ms: Option<u64>,
        ) -> Result<Analysis, HostError> {
            Err(HostError::not_found("not implemented in test fake"))
        }

        fn book(&self) -> Option<BookInfo> {
            Some(BookInfo {
                id: "test".into(),
                default_rounds: 20,
                game_config: serde_json::json!({}),
            })
        }

        fn book_build(
            &self,
            rounds: u32,
            seed: Option<u64>,
            game_config: Option<Value>,
        ) -> Result<Value, HostError> {
            Ok(serde_json::json!({
                "rounds": rounds,
                "seed": seed,
                "game_config": game_config,
            }))
        }
    }

    #[test]
    fn test_run_cli_book_describe_unsupported_when_book_none() {
        let (out, code) = run_cli_capture(&["book", "describe"], "");
        assert_eq!(code, 1);
        assert!(out.is_empty());
    }

    #[test]
    fn test_run_cli_book_describe_prints_book_info() {
        let (out, code) = run_cli_capture_with(BookableFakeAdapter, &["book", "describe"], "");
        assert_eq!(code, 0);
        let info: BookInfo = serde_json::from_str(out.lines().next().unwrap()).unwrap();
        assert_eq!(info.id, "test");
        assert_eq!(info.default_rounds, 20);
    }

    #[test]
    fn test_run_cli_book_build_prints_result_verbatim() {
        let (out, code) = run_cli_capture_with(
            BookableFakeAdapter,
            &["book", "build", "--rounds", "5", "--seed", "7"],
            "",
        );
        assert_eq!(code, 0);
        let result: Value = serde_json::from_str(out.lines().next().unwrap()).unwrap();
        assert_eq!(result["rounds"], 5);
        assert_eq!(result["seed"], 7);
    }

    #[test]
    fn test_run_cli_book_build_missing_rounds_errors() {
        let (out, code) = run_cli_capture_with(BookableFakeAdapter, &["book", "build"], "");
        assert_eq!(code, 1);
        assert!(out.is_empty());
    }

    #[test]
    fn test_run_cli_book_build_unsupported_by_default() {
        let (out, code) = run_cli_capture(&["book", "build", "--rounds", "5"], "");
        assert_eq!(code, 1);
        assert!(out.is_empty());
    }

    #[test]
    fn test_run_cli_book_with_no_further_args_falls_back_to_stdin_stdout_loop() {
        let (out, code) =
            run_cli_capture(&["book"], "{\"id\":11,\"method\":\"kind\",\"params\":{}}\n");
        assert_eq!(code, 0);
        let resp = parse_response(out.lines().next().unwrap());
        match resp {
            Response::Success { id, result } => {
                assert_eq!(id, 11);
                assert_eq!(result, "fake");
            }
            _ => panic!("expected success response"),
        }
    }
}
