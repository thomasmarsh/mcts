// Bake git SHA and dirty-worktree flag into every binary at compile time.
// Search & round-robin results carry this attribution so regression hunting
// can filter/join on `git_sha` rather than relying on imprecise timestamps.

use std::path::Path;
use std::process::Command;

fn main() {
    let repo_root = Path::new(env!("CARGO_MANIFEST_DIR"));

    // --- git SHA ---
    let sha = Command::new("git")
        .args(["rev-parse", "HEAD"])
        .current_dir(repo_root)
        .output()
        .ok()
        .and_then(|out| {
            if out.status.success() {
                String::from_utf8(out.stdout).ok().map(|s| s.trim().to_owned())
            } else {
                None
            }
        })
        .unwrap_or_else(|| "unknown".to_owned());

    println!("cargo:rustc-env=MCTS_GIT_SHA={sha}");

    // --- dirty-worktree flag ---
    let dirty = Command::new("git")
        .args(["status", "--porcelain"])
        .current_dir(repo_root)
        .output()
        .ok()
        .map(|out| !out.stdout.is_empty())
        .unwrap_or(false);

    println!(
        "cargo:rustc-env=MCTS_GIT_DIRTY={}",
        if dirty { "true" } else { "false" }
    );

    // Re-run if HEAD moves or the ref it points at changes.
    let git_dir = repo_root.join(".git");
    println!("cargo:rerun-if-changed={}", git_dir.join("HEAD").display());

    // Also watch the current branch ref (e.g. .git/refs/heads/main) so a
    // `git commit` that only changes the ref file triggers a rebuild without
    // needing to touch HEAD itself.
    //
    // We parse HEAD's contents to find which ref file to watch. If HEAD is
    // detached (just a raw SHA), there's no ref file to watch beyond HEAD
    // itself, which we already handle above.
    if let Ok(head_contents) = std::fs::read_to_string(git_dir.join("HEAD")) {
        if let Some(ref_path) = head_contents
            .strip_prefix("ref: ")
            .map(|s| s.trim())
            .filter(|s| !s.is_empty())
        {
            let full = repo_root.join(".git").join(ref_path);
            println!("cargo:rerun-if-changed={}", full.display());
        }
    }
}