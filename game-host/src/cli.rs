use crate::{
    derive_seed, run_host, AiPresetInfo, CompareValidationField, ConfiguredComparisonSummary,
    ConfiguredMatchResult, GameAdapter, HostError, TunerInfo,
};
use serde_json::Value;
use std::io::{self, Read, Write};

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
pub(crate) fn run_cli_with<I, R, W, A>(mut args: I, reader: R, mut writer: W, adapter: A) -> i32
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
/// --baseline-config <json>] [--game-config <json>] [--max-iterations <n> |
/// --max-time-ms <n>]` from the remaining CLI args, calls
/// [`GameAdapter::tune_eval`], and prints its JSON result verbatim to
/// `writer`. Returns the process exit code. `--baseline` and
/// `--baseline-config` are mutually exclusive -- supplying both is rejected
/// before the adapter is ever called. `--max-iterations` and
/// `--max-time-ms` are the per-run compute-budget overrides -- see
/// `tune_eval`'s own doc comment; supplying both is likewise rejected
/// (unlike `SearchBudget`, which tolerates both being set, a caller
/// supplying both here almost certainly meant only one and the other is
/// leftover from a prior override).
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
    let mut max_time_ms: Option<u64> = None;
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
            "--max-time-ms" => max_time_ms = args.next().and_then(|s| s.parse().ok()),
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
        if max_iterations.is_some() && max_time_ms.is_some() {
            return Err(HostError::bad_request(
                "--max-iterations and --max-time-ms are mutually exclusive",
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
            max_time_ms,
            trace_path.map(std::path::PathBuf::from),
            None,
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
    let mut trace_game_sequence_start: Option<u64> = None;

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
                "--trace-game-sequence-start" => {
                    let raw = value(&flag, &mut args)?;
                    trace_game_sequence_start = Some(raw.parse().map_err(|_| {
                        HostError::bad_request("invalid --trace-game-sequence-start")
                    })?);
                }
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
        if trace_game_sequence_start.is_some() && trace_path.is_none() {
            return Err(HostError::bad_request(
                "--trace-game-sequence-start requires --trace-path",
            ));
        }
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
            let trace_game_sequence_start = match trace_game_sequence_start {
                Some(start) => Some(
                    start
                        .checked_add(sequence)
                        .ok_or_else(|| HostError::bad_request("trace game sequence overflow"))?,
                ),
                None => None,
            };
            let mut on_game = |mut record: ConfiguredMatchResult| -> Result<(), HostError> {
                sequence = sequence
                    .checked_add(1)
                    .ok_or_else(|| HostError::internal("comparison sequence overflow"))?;
                record.seq = sequence;
                record.round = round;
                record.seed = round_seed;
                if let Some(start) = trace_game_sequence_start {
                    let expected = start
                        .checked_add(sequence - 1)
                        .ok_or_else(|| HostError::internal("trace game sequence overflow"))?;
                    if record.trace_game_seq != Some(expected) {
                        return Err(HostError::internal(
                            "configured comparison trace sequence does not match candidate side",
                        ));
                    }
                }
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
                trace_game_sequence_start,
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
