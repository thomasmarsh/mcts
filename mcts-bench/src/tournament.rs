//! Tournament engine: round-robin match execution with structured JSONL
//! output.  Contains the typed (`AnySearch`-based) convenience API that
//! existing callers (examples, demos) use, plus the trait-based
//! (`BenchGame`) API that the new benchmark harness uses.  Both emit one
//! `LogRecord::MatchResult` per completed game to a provided `Write`, with
//! human-readable progress bars going to stderr (indicatif).

use indicatif::{MultiProgress, ProgressBar, ProgressDrawTarget, ProgressStyle};
use rayon::prelude::*;
use std::io::Write;
use std::sync::atomic::AtomicU32;
use std::ops::{Add, AddAssign};

use mcts::game::{Game, PlayerIndex};
use mcts::strategies::Search;
use mcts::util::{AnySearch, Verbosity};

use crate::log::LogRecord;
use crate::BenchGame;

// ---------------------------------------------------------------------------
// Result type (moved from util.rs)
// ---------------------------------------------------------------------------

/// Aggregated win/loss/draw counts for one strategy across one or more
/// matches.
#[derive(Copy, Clone, Debug, Default)]
pub struct Result {
    pub wins: usize,
    pub losses: usize,
    pub draws: usize,
}

impl Result {
    pub fn total(&self) -> usize {
        self.wins + self.losses + self.draws
    }

    /// Score counting a draw as half a win -- the standard way to fold draws
    /// into a single win-rate proportion for a confidence interval.
    pub fn score(&self) -> f64 {
        self.wins as f64 + 0.5 * self.draws as f64
    }

    /// Win-rate proportion (draws counted as half a win) with its Wilson
    /// score interval at confidence level `z` (e.g. `1.96` for ~95%).
    /// Returns `(point_estimate, (lower, upper))`.
    pub fn win_rate_ci(&self, z: f64) -> (f64, (f64, f64)) {
        let total = self.total();
        let point = if total == 0 {
            0.5
        } else {
            self.score() / total as f64
        };
        (point, wilson_interval(self.score(), total, z))
    }
}

impl Add for Result {
    type Output = Self;
    fn add(self, rhs: Self) -> Self::Output {
        Result {
            wins: self.wins + rhs.wins,
            losses: self.losses + rhs.losses,
            draws: self.draws + rhs.draws,
        }
    }
}

impl AddAssign for Result {
    fn add_assign(&mut self, rhs: Self) {
        self.wins += rhs.wins;
        self.losses += rhs.losses;
        self.draws += rhs.draws;
    }
}

// ---------------------------------------------------------------------------
// Wilson score interval (moved from util.rs)
// ---------------------------------------------------------------------------

/// Wilson score confidence interval for a binomial proportion --
/// `successes` out of `total` trials, at confidence level `z` (e.g. `1.96`
/// for ~95%, `2.576` for ~99%). Unlike the naive `p_hat +/- z*sqrt(p_hat*(1
/// -p_hat)/n)` normal-approximation interval, this stays inside `[0, 1]` and
/// is accurate at the small-`n`/extreme-`p_hat` sizes self-play tournaments
/// actually produce (a handful of dozens of games, sometimes a lopsided
/// score), where the naive interval can be badly wrong or even leave `[0,
/// 1]` entirely.
///
/// `successes` is a plain `f64` rather than an integer count so callers can
/// pass a half-credit-for-draws score (see `Result::score`) directly --
/// the derivation only uses `successes / total` as the sample proportion,
/// it never needs `successes` to itself be a count of discrete Bernoulli
/// trials.
///
/// Returns `(0.0, 1.0)` for `total == 0` (no information).
pub fn wilson_interval(successes: f64, total: usize, z: f64) -> (f64, f64) {
    if total == 0 {
        return (0.0, 1.0);
    }
    let n = total as f64;
    let p_hat = successes / n;
    let z2 = z * z;
    let denom = 1.0 + z2 / n;
    let center = p_hat + z2 / (2.0 * n);
    let margin = z * ((p_hat * (1.0 - p_hat) / n) + z2 / (4.0 * n * n)).sqrt();
    let lower = ((center - margin) / denom).max(0.0);
    let upper = ((center + margin) / denom).min(1.0);
    (lower, upper)
}

// ---------------------------------------------------------------------------
// Helpers for emitting result records
// ---------------------------------------------------------------------------

fn write_match_result<W: Write>(
    writer: &mut W,
    seq: u64,
    strategy_a: &str,
    strategy_b: &str,
    outcome: &str,
    winner: Option<String>,
    extra: Option<serde_json::Value>,
) {
    let rec = LogRecord::MatchResult {
        seq,
        strategy_a: strategy_a.to_owned(),
        strategy_b: strategy_b.to_owned(),
        outcome: outcome.to_owned(),
        winner,
        extra,
    };
    let mut line = rec.to_json_line();
    line.push('\n');
    let _ = writer.write_all(line.as_bytes());
    let _ = writer.flush();
}

// ---------------------------------------------------------------------------
// Typed round_robin (takes concrete AnySearch<G> strategies)
// ---------------------------------------------------------------------------

/// Play a round-robin tournament with the provided strategies.  Progress
/// bars go to stderr (or are hidden when `verbose` is `Verbosity::Silent`).
///
/// Returns the aggregate `Result` for each strategy across all pairwise
/// matches.
fn round_robin<G>(
    strategies: &mut [AnySearch<'_, G>],
    verbose: Verbosity,
) -> Vec<Result>
where
    G: Game + Clone,
    G::S: Sync,
{
    let init = G::S::default();

    let mut pairs = Vec::new();
    for i in 0..strategies.len() {
        for j in 0..strategies.len() {
            if i != j {
                pairs.push((i, j));
            }
        }
    }

    let mp = if verbose.verbose() {
        MultiProgress::new()
    } else {
        MultiProgress::with_draw_target(ProgressDrawTarget::hidden())
    };

    let pb_overall = mp.add(ProgressBar::new(pairs.len() as u64));
    pb_overall.set_style(
        ProgressStyle::with_template(
            "[{elapsed_precise}] {bar:40.white/blue} {pos:>7}/{len:7} {msg:.bold}",
        )
        .unwrap(),
    );
    pb_overall.set_message("Tournament:");

    let sty = ProgressStyle::with_template(
        "[{elapsed_precise}] {bar:40.cyan/blue} {pos:>7}/{len:7} {msg}",
    )
    .unwrap();

    let counter: AtomicU32 = AtomicU32::new(0);

    let results = pairs
        .into_par_iter()
        .map(|(i, j)| {
            counter.fetch_add(1, std::sync::atomic::Ordering::SeqCst);

            let mut results = vec![Result::default(); strategies.len()];
            let si = strategies[i].clone();
            let sj = strategies[j].clone();

            let pb = mp.add(ProgressBar::new(1));
            pb.set_style(sty.clone());
            let vs_str = format!("{:>25} | {:<25}", si.friendly_name(), sj.friendly_name());
            pb.set_message(format!("{:^53}", vs_str));

            let mut strat = [si, sj];
            let players = [i, j];
            let mut current;
            let mut depth = 0;
            let mut state = init.clone();
            loop {
                current = G::player_to_move(&state).to_index();
                if G::is_terminal(&state) {
                    break;
                }

                let action = strat[current].choose_action(&state);
                pb.set_length(depth + strat[current].estimated_depth() as u64);
                state = G::apply(state, &action);
                pb.inc(1);
                depth += 1;
            }

            match G::winner(&state) {
                None => {
                    results[i].draws += 1;
                    results[j].draws += 1;
                }
                Some(p) => {
                    let winner = players[p.to_index()];
                    let loser = players[1 - p.to_index()];
                    results[winner].wins += 1;
                    results[loser].losses += 1;
                }
            }
            pb.finish();
            mp.remove(&pb);
            pb_overall.inc(1);
            counter.fetch_sub(1, std::sync::atomic::Ordering::SeqCst);
            results
        })
        .reduce_with(|acc, x| {
            acc.into_iter()
                .zip(x.iter())
                .map(|(r1, r2)| r1 + *r2)
                .collect()
        })
        .unwrap_or_else(|| panic!());

    assert_eq!(counter.load(std::sync::atomic::Ordering::SeqCst), 0);
    results
}

/// Play a round-robin tournament multiple times with the provided strategies.
/// Emits one `LogRecord::MatchResult` per completed game to `writer` (though
/// for the parallel typed version, individual match records are deferred to
/// a future session -- currently only round-complete heartbeats are written
/// here).  Progress bars and human-readable tables go to stderr.
pub fn round_robin_multiple<G, W: Write>(
    strategies: &mut [AnySearch<'_, G>],
    rounds: usize,
    writer: &mut W,
    verbose: Verbosity,
) -> Vec<Result>
where
    G: Game + Clone,
    G::S: Sync,
{
    let mut results = vec![Result::default(); strategies.len()];

    for _ in 0..rounds {
        let new_results = round_robin::<G>(strategies, verbose);
        for (index, result) in new_results.iter().enumerate() {
            results[index] += *result;
        }

        if verbose.verbose() {
            // Human-readable table to stderr.
            eprintln!("{:=<63}", "");
            eprintln!(
                "{0:^25} | {1:^10} | {2:^10} | {3:^4}",
                "match", "won", "lost", "draw"
            );
            eprintln!("{:-<59}", "");

            let mut copy = results.iter().enumerate().collect::<Vec<_>>();
            copy.sort_unstable_by_key(|x| (-(x.1.wins as i64), x.1.losses, x.1.draws));

            for (index, _) in &copy {
                let total = results[*index].wins + results[*index].losses + results[*index].draws;
                let win_pct = 100. * results[*index].wins as f64 / total as f64;
                let loss_pct = 100. * results[*index].losses as f64 / total as f64;
                eprintln!(
                    "{0:<25} | {1:>4} ({win_pct:2.0}%) | {2:>4} ({loss_pct:2.0}%) | {3:<4}",
                    strategies[*index].friendly_name(),
                    results[*index].wins,
                    results[*index].losses,
                    results[*index].draws,
                );
            }
        }

        // Write a round-complete heartbeat record.
        let heartbeat = LogRecord::Heartbeat {
            games_played: results.iter().map(|r| r.total()).sum::<usize>() as u64,
        };
        let mut line = heartbeat.to_json_line();
        line.push('\n');
        let _ = writer.write_all(line.as_bytes());
        let _ = writer.flush();
    }

    results
}

// ---------------------------------------------------------------------------
// Trait-based tournament functions (for BenchGame)
// ---------------------------------------------------------------------------

/// Play a single match between two strategies identified by their
/// `BenchGame` strategy IDs, write the result to `writer`, and return the
/// per-strategy `Result` for the match.
fn play_one_match<W: Write>(
    game: &dyn BenchGame,
    a_id: &str,
    b_id: &str,
    seq: u64,
    writer: &mut W,
) -> (usize, usize) /* (a_win, b_win) or draw */ {
    let outcome = game.play_match(a_id, b_id);
    let (outcome_str, winner_str) = match outcome.winner {
        None => ("draw", None),
        Some(0) => ("win_a", Some(a_id.to_owned())),
        Some(1) => ("win_b", Some(b_id.to_owned())),
        _ => ("draw", None),
    };
    write_match_result(writer, seq, a_id, b_id, outcome_str, winner_str, outcome.extra);
    match outcome.winner {
        None => (0, 0),
        Some(0) => (1, 0),
        Some(1) => (0, 1),
        _ => (0, 0),
    }
}

/// Trait-based round-robin: play every ordered pair of strategies from
/// `strategy_ids` using the provided `BenchGame`.  Emits one
/// `LogRecord::MatchResult` per game to `writer`.  Returns aggregate
/// `Result` per strategy (same index order as `strategy_ids`).
///
/// `seq` is a mutable counter that increments monotonically across all
/// games played, so callers like `round_robin_bench_multiple` can chain
/// calls without sequence-number collisions.
pub fn round_robin_bench<W: Write>(
    game: &dyn BenchGame,
    strategy_ids: &[String],
    writer: &mut W,
    verbose: Verbosity,
    seq: &mut u64,
) -> Vec<Result> {
    let mut results = vec![Result::default(); strategy_ids.len()];
    let n = strategy_ids.len();
    let total_games = n * (n - 1);

    let mp = if verbose.verbose() {
        MultiProgress::new()
    } else {
        MultiProgress::with_draw_target(ProgressDrawTarget::hidden())
    };

    let pb = mp.add(ProgressBar::new(total_games as u64));
    pb.set_style(
        ProgressStyle::with_template(
            "[{elapsed_precise}] {bar:40.white/blue} {pos:>7}/{len:7} {msg:.bold}",
        )
        .unwrap(),
    );
    pb.set_message("Tournament:");

    for i in 0..n {
        for j in 0..n {
            if i == j {
                continue;
            }
            let (a_wins, b_wins) = play_one_match(game, &strategy_ids[i], &strategy_ids[j], *seq, writer);
            *seq += 1;
            results[i].wins += a_wins;
            results[i].losses += b_wins;
            results[j].wins += b_wins;
            results[j].losses += a_wins;
            if a_wins == 0 && b_wins == 0 {
                results[i].draws += 1;
                results[j].draws += 1;
            }
            pb.inc(1);
        }
    }

    pb.finish();
    results
}

/// Trait-based round-robin multiple: run `rounds` full round-robins,
/// aggregating results across all rounds.
pub fn round_robin_bench_multiple<W: Write>(
    game: &dyn BenchGame,
    strategy_ids: &[String],
    rounds: usize,
    writer: &mut W,
    verbose: Verbosity,
) -> Vec<Result> {
    let mut results = vec![Result::default(); strategy_ids.len()];
    let mut seq: u64 = 1;
    for _ in 0..rounds {
        let new_results = round_robin_bench(game, strategy_ids, writer, verbose, &mut seq);
        for (index, result) in new_results.iter().enumerate() {
            results[index] += *result;
        }
    }
    results
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{BenchGame, MatchOutcome, StrategyInfo};

    // -----------------------------------------------------------------------
    // Tests moved from util.rs (Result + wilson_interval)
    // -----------------------------------------------------------------------

    #[test]
    fn test_wilson_interval_matches_known_reference_values() {
        let (lo, hi) = wilson_interval(8.0, 10, 1.96);
        assert!((lo - 0.4902).abs() < 1e-3, "lo={lo}");
        assert!((hi - 0.9433).abs() < 1e-3, "hi={hi}");

        let (lo_small, hi_small) = wilson_interval(5.0, 10, 1.96);
        let (lo_big, hi_big) = wilson_interval(500.0, 1000, 1.96);
        assert!(lo_small < 0.5 && hi_small > 0.5);
        assert!(lo_big < 0.5 && hi_big > 0.5);
        assert!(hi_big - lo_big < hi_small - lo_small);
    }

    #[test]
    fn test_wilson_interval_stays_within_unit_range() {
        for &successes in &[0.0, 1.0, 3.0] {
            let (lo, hi) = wilson_interval(successes, 3, 1.96);
            assert!((0.0..=1.0).contains(&lo));
            assert!((0.0..=1.0).contains(&hi));
            assert!(lo <= hi);
        }
    }

    #[test]
    fn test_wilson_interval_empty_sample_is_maximally_uncertain() {
        assert_eq!(wilson_interval(0.0, 0, 1.96), (0.0, 1.0));
    }

    #[test]
    fn test_result_win_rate_ci_counts_draws_as_half_wins() {
        let r = Result {
            wins: 6,
            losses: 2,
            draws: 4,
        };
        let (point, (lo, hi)) = r.win_rate_ci(1.96);
        assert!((point - 8.0 / 12.0).abs() < 1e-9);
        assert!(lo < point && point < hi);
    }

    // -----------------------------------------------------------------------
    // Fake BenchGame for tournament engine tests
    // -----------------------------------------------------------------------

    /// A fake game that produces deterministic outcomes: strategy "a" always
    /// beats "b", "b" always beats "c", "c" always beats "a" (rock-paper-
    /// scissors), and any self-match is a draw.  This lets us verify the
    /// tournament engine emits the correct sequence and counts of match
    /// results without running any real MCTS search.
    struct RockPaperScissors;

    impl BenchGame for RockPaperScissors {
        fn kind(&self) -> &'static str {
            "rock_paper_scissors"
        }

        fn strategies(&self) -> Vec<StrategyInfo> {
            vec![
                StrategyInfo {
                    id: "a".into(),
                    label: "Strategy A".into(),
                    description: "Beats B, loses to C".into(),
                },
                StrategyInfo {
                    id: "b".into(),
                    label: "Strategy B".into(),
                    description: "Beats C, loses to A".into(),
                },
                StrategyInfo {
                    id: "c".into(),
                    label: "Strategy C".into(),
                    description: "Beats A, loses to B".into(),
                },
            ]
        }

        fn play_match(&self, strategy_a: &str, strategy_b: &str) -> MatchOutcome {
            let winner = match (strategy_a, strategy_b) {
                ("a", "b") | ("b", "a") => {
                    if strategy_a == "a" { Some(0) } else { Some(1) }
                }
                ("b", "c") | ("c", "b") => {
                    if strategy_a == "b" { Some(0) } else { Some(1) }
                }
                ("c", "a") | ("a", "c") => {
                    if strategy_a == "c" { Some(0) } else { Some(1) }
                }
                _ => None,
            };
            MatchOutcome {
                winner,
                extra: None,
            }
        }
    }

    #[test]
    fn test_round_robin_bench_emits_correct_number_of_match_records() {
        let game = RockPaperScissors;
        let ids: Vec<String> = game.strategies().into_iter().map(|s| s.id).collect();

        let mut buf: Vec<u8> = Vec::new();
        let mut seq: u64 = 1;
        let results = round_robin_bench(&game, &ids, &mut buf, Verbosity::Silent, &mut seq);

        let records: Vec<LogRecord> = buf
            .split(|&b| b == b'\n')
            .filter(|line| !line.is_empty())
            .map(|line| serde_json::from_slice(line).unwrap())
            .collect();
        assert_eq!(records.len(), 6, "expected 6 match records for 3 strategies");
        for rec in &records {
            assert!(matches!(rec, LogRecord::MatchResult { .. }));
        }

        assert_eq!(results.len(), 3);
        assert_eq!(results[0].total(), 4, "strategy a should have 4 games");
        assert_eq!(results[1].total(), 4, "strategy b should have 4 games");
        assert_eq!(results[2].total(), 4, "strategy c should have 4 games");

        assert_eq!(results[0].wins, 2);
        assert_eq!(results[0].losses, 2);
        assert_eq!(results[0].draws, 0);

        assert_eq!(results[1].wins, 2);
        assert_eq!(results[1].losses, 2);
        assert_eq!(results[1].draws, 0);

        assert_eq!(results[2].wins, 2);
        assert_eq!(results[2].losses, 2);
        assert_eq!(results[2].draws, 0);
    }

    #[test]
    fn test_round_robin_bench_self_match_is_draw() {
        let game = RockPaperScissors;
        let ids: Vec<String> = vec!["a".into()];

        let mut buf: Vec<u8> = Vec::new();
        let mut seq: u64 = 1;
        let results = round_robin_bench(&game, &ids, &mut buf, Verbosity::Silent, &mut seq);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].total(), 0);
        assert!(buf.is_empty());
    }

    #[test]
    fn test_round_robin_bench_multiple_rounds_aggregates_correctly() {
        let game = RockPaperScissors;
        let ids: Vec<String> = game.strategies().into_iter().map(|s| s.id).collect();

        let mut buf: Vec<u8> = Vec::new();
        let results = round_robin_bench_multiple(&game, &ids, 3, &mut buf, Verbosity::Silent);

        assert_eq!(results[0].total(), 12);
        assert_eq!(results[0].wins, 6);
        assert_eq!(results[0].losses, 6);
    }
}