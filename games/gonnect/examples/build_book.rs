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
//!     [--top-epsilon F] [--seed N] [--workers N] [--top N] \
//!     [--out PATH] [--fresh]
//!
//! `--out` defaults to `books/gonnect-{size}.json` -- the path
//! `game_gonnect::book::BookIndex::load` (consulted by `main.rs`'s
//! `ai_move`/`analyze`) looks for at that size, so a run without `--out`
//! is immediately picked up by live play. Run from the repo root: nothing
//! sets a `current_dir` for `game-host`/`server`'s subprocess, so this
//! relative path and live play's need to agree on where they're invoked
//! from.
//!
//! If `--out` already exists, it's loaded first and passed to `book::build`
//! as a seed, so back-to-back runs are additive: two 60-round runs add up
//! to 120 rounds of data in the same book, rather than the second replacing
//! the first. Pass `--fresh` to discard the existing file instead and start
//! over from an empty book.
//!
//! `--workers` (default 1) splits `--rounds` across that many self-play
//! threads (see `book::build`'s doc comment) -- each worker's games still
//! fold into one combined book, so raising it only affects wall-clock time,
//! not the result's shape.
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
    /// Skip loading `--out` as a seed even if it already exists, so `--out`
    /// is fully overwritten instead of amended.
    fresh: bool,
}

impl Default for Args {
    fn default() -> Self {
        Self {
            size: 9,
            book: BookBuildConfig::default(),
            top: 8,
            out: None,
            fresh: false,
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
            "--workers" => args.book.num_workers = next!(),
            "--top" => args.top = next!(),
            "--out" => args.out = Some(it.next().expect("missing value")),
            "--fresh" => args.fresh = true,
            other => panic!("unknown flag: {other}"),
        }
    }
    args
}

/// Builds the book for `args.size`, prints a progress line per game, then
/// reports and serializes the result.
fn run(args: &Args) {
    let out = args
        .out
        .clone()
        .unwrap_or_else(|| format!("books/gonnect-{}.json", args.size));

    // Load `out` as a seed unless `--fresh` was passed -- a missing or
    // unparseable file (the common case for a first run at this size) just
    // means "start from an empty book", same as `BookIndex::load`'s own
    // fallback.
    let seed: Option<OpeningBook<<Gonnect as Game>::A>> = if args.fresh {
        None
    } else {
        std::fs::read_to_string(&out)
            .ok()
            .and_then(|json| serde_json::from_str(&json).ok())
    };
    match &seed {
        Some(book) => println!(
            "amending {out} ({} existing root visits) with {} more games",
            book.num_visits_at(book.root_id),
            args.book.rounds,
        ),
        None => println!(
            "building {out} from scratch with {} games",
            args.book.rounds
        ),
    }

    let start = Instant::now();
    let book: OpeningBook<<Gonnect as Game>::A> = book::build(
        args.size,
        &args.book,
        seed.as_ref(),
        |round, plies, utilities| {
            println!(
                "game {:>4}/{}: {:>3} plies, utilities {:?}",
                round + 1,
                args.book.rounds,
                plies,
                utilities,
            );
        },
    );
    println!(
        "\nbuilt book from {} games in {:.2?}",
        args.book.rounds,
        start.elapsed()
    );

    let initial = State::new(args.size);
    report_top_moves(&book, &initial, args.top);

    if let Some(parent) = std::path::Path::new(&out).parent() {
        std::fs::create_dir_all(parent).expect("failed to create output directory");
    }
    let json = serde_json::to_string_pretty(&book).expect("book always serializes");
    std::fs::write(&out, &json).expect("failed to write book file");
    let size = args.size;
    println!(
        "\nwrote {out} ({} bytes) for size {size}x{size}",
        json.len()
    );

    // Round-trip check: reload what was just written and confirm the root's
    // top reply for `player` still scores identically, so a corrupted or
    // lossy serialization (e.g. the map-key issue `Entry`'s custom
    // `Serialize`/`Deserialize` wire shape exists to avoid) fails loudly
    // here instead of silently shipping a bad book file.
    let reloaded: OpeningBook<<Gonnect as Game>::A> =
        serde_json::from_str(&json).expect("just-written book always deserializes");
    let player = Gonnect::player_to_move(&initial).to_index();
    assert_eq!(
        reloaded.score(&[], player),
        book.score(&[], player),
        "round-tripped book disagrees with the in-memory book"
    );
    println!("round-trip check passed");
}

/// Prints the book's ranked replies from `state`, using `Game::notation`
/// for human-readable cells (e.g. `D4`) instead of raw indices.
fn report_top_moves(book: &OpeningBook<<Gonnect as Game>::A>, state: &State, top: usize) {
    let player = Gonnect::player_to_move(state).to_index();
    match book.children(&[], player) {
        None => println!("\n(root has no book entries -- run more rounds)"),
        Some(mut candidates) => {
            candidates.truncate(top);
            println!(
                "\ntop {} opening replies for player {player} at the empty board:",
                candidates.len()
            );
            for (action, visits, score) in candidates {
                let notation = Gonnect::notation(state, &action);
                let score_str = score.map_or("--".to_string(), |s| format!("{s:.3}"));
                println!("  {notation:>4}  visits={visits:<5} score={score_str}");
            }
        }
    }
}

fn main() {
    let args = parse_args();
    if !(3..=19).contains(&args.size) {
        panic!("unsupported board size {} (supported: 3..=19)", args.size);
    }
    run(&args);
}
