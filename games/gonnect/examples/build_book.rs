//! Builds a Gonnect opening book via Quasi-Best-First self-play and writes
//! it to disk. The self-play/book-accumulation logic itself lives in
//! `game_gonnect::book` (shared with `main.rs`'s `book_build` `GameAdapter`
//! method, the subprocess-protocol path); this example is the
//! human-facing entry point: progress output, a ranked top-moves report,
//! and a round-trip serialization check.
//!
//! Usage:
//!   cargo run --release --example build_book -p game-gonnect -- \
//!     [--size 9|13|19] [--rounds N] [--inner-iterations N] \
//!     [--top-epsilon F] [--seed N] [--top N] [--out PATH]
//!
//! `--out` defaults to `books/gonnect-{size}.json` -- the path
//! `game_gonnect::book::BookIndex::load` (consulted by `main.rs`'s
//! `ai_move`/`analyze`) looks for at that size, so a run without `--out`
//! is immediately picked up by live play. Run from the repo root: nothing
//! sets a `current_dir` for `game-host`/`server`'s subprocess, so this
//! relative path and live play's need to agree on where they're invoked
//! from.
//!
//! Each run starts from an empty book and overwrites `--out` outright --
//! `book::build` never loads the existing file first, so back-to-back runs
//! are not additive (two 60-round runs do not add up to 120 rounds of
//! data, the second just replaces the first). `books/` is gitignored, so
//! there's no git history to fall back on if an accidental low-round rerun
//! clobbers a book you cared about.
//!
//! `--rounds` (default 60, mainly a smoke-test value) has no fixed "right"
//! number: QBF's greedy reinforcement converges the *root* reply fast (150
//! rounds was enough for a 40%-of-games opening cell on a 9x9 board), but
//! a book node only gets consulted live once it independently clears
//! `book::MIN_BOOK_VISITS`, and deeper plies see a shrinking share of
//! total games as the tree branches -- so depth costs more rounds than the
//! opening move does. Watch the printed top-moves report's visit counts
//! and raise `--rounds` until what you care about clears that bar, rather
//! than picking a number up front.
use std::time::Instant;

use game_gonnect::book::{self, BookBuildConfig};
use game_gonnect::{Gonnect, State};
use mcts::game::Game;
use mcts::game::PlayerIndex;
use mcts::strategies::mcts::book::OpeningBook;

struct Args {
    size: usize,
    book: BookBuildConfig,
    top: usize,
    /// `None` means "derive from `size`" -- resolved after parsing, so
    /// `--size` can come after `--out` on the command line, and so a run
    /// without `--out` lands at the path `game_gonnect::book::BookIndex`
    /// (the live-play consultation side) actually looks for.
    out: Option<String>,
}

impl Default for Args {
    fn default() -> Self {
        Self {
            size: 9,
            book: BookBuildConfig::default(),
            top: 8,
            out: None,
        }
    }
}

fn parse_args() -> Args {
    let mut args = Args::default();
    let mut it = std::env::args().skip(1);
    while let Some(flag) = it.next() {
        macro_rules! next {
            () => {
                it.next()
                    .expect("missing value")
                    .parse()
                    .expect("invalid value")
            };
        }
        match flag.as_str() {
            "--size" => args.size = next!(),
            "--rounds" => args.book.rounds = next!(),
            "--inner-iterations" => args.book.inner_iterations = next!(),
            "--top-epsilon" => args.book.top_epsilon = next!(),
            "--seed" => args.book.seed = next!(),
            "--top" => args.top = next!(),
            "--out" => args.out = Some(it.next().expect("missing value")),
            other => panic!("unknown flag: {other}"),
        }
    }
    args
}

/// Builds the book for one `(N, WORDS)` monomorphization, prints a
/// progress line per game, then reports and serializes the result. `N`/
/// `WORDS` are a single macro-dispatched pair (mirrors `game-gonnect`'s own
/// `dispatch_size!` in `main.rs`) since Gonnect's board size is a const
/// generic, not a runtime value.
fn run<const N: usize, const WORDS: usize>(args: &Args) {
    let out = args
        .out
        .clone()
        .unwrap_or_else(|| format!("books/gonnect-{}.json", args.size));
    let start = Instant::now();
    let book: OpeningBook<<Gonnect<N, WORDS> as Game>::A> =
        book::build::<N, WORDS>(&args.book, |round, plies, utilities| {
            println!(
                "game {:>4}/{}: {:>3} plies, utilities {:?}",
                round + 1,
                args.book.rounds,
                plies,
                utilities,
            );
        });
    println!(
        "\nbuilt book from {} games in {:.2?}",
        args.book.rounds,
        start.elapsed()
    );

    let initial = State::<N, WORDS>::default();
    report_top_moves(&book, &initial, args.top);

    if let Some(parent) = std::path::Path::new(&out).parent() {
        std::fs::create_dir_all(parent).expect("failed to create output directory");
    }
    let json = serde_json::to_string_pretty(&book).expect("book always serializes");
    std::fs::write(&out, &json).expect("failed to write book file");
    println!("\nwrote {out} ({} bytes) for size {N}x{N}", json.len());

    // Round-trip check: reload what was just written and confirm the root's
    // top reply for `player` still scores identically, so a corrupted or
    // lossy serialization (e.g. the map-key issue `Entry`'s custom
    // `Serialize`/`Deserialize` wire shape exists to avoid) fails loudly
    // here instead of silently shipping a bad book file.
    let reloaded: OpeningBook<<Gonnect<N, WORDS> as Game>::A> =
        serde_json::from_str(&json).expect("just-written book always deserializes");
    let player = Gonnect::<N, WORDS>::player_to_move(&initial).to_index();
    assert_eq!(
        reloaded.score(&[], player),
        book.score(&[], player),
        "round-tripped book disagrees with the in-memory book"
    );
    println!("round-trip check passed");
}

/// Prints the book's ranked replies from `state`, using `Game::notation`
/// for human-readable cells (e.g. `D4`) instead of raw indices.
fn report_top_moves<const N: usize, const WORDS: usize>(
    book: &OpeningBook<<Gonnect<N, WORDS> as Game>::A>,
    state: &State<N, WORDS>,
    top: usize,
) {
    let player = Gonnect::<N, WORDS>::player_to_move(state).to_index();
    match book.children(&[], player) {
        None => println!("\n(root has no book entries -- run more rounds)"),
        Some(mut candidates) => {
            candidates.truncate(top);
            println!(
                "\ntop {} opening replies for player {player} at the empty board:",
                candidates.len()
            );
            for (action, visits, score) in candidates {
                let notation = Gonnect::<N, WORDS>::notation(state, &action);
                let score_str = score.map_or("--".to_string(), |s| format!("{s:.3}"));
                println!("  {notation:>4}  visits={visits:<5} score={score_str}");
            }
        }
    }
}

fn main() {
    let args = parse_args();
    match args.size {
        9 => run::<9, 2>(&args),
        13 => run::<13, 3>(&args),
        19 => run::<19, 6>(&args),
        other => panic!("unsupported board size {other} (supported: 9, 13, 19)"),
    }
}
