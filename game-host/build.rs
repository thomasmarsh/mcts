// Bake git SHA and dirty-worktree flag into every binary at compile time.
// Search & round-robin results carry this attribution so regression hunting
// can filter/join on `git_sha` rather than relying on imprecise timestamps.
//
// The git dir is resolved with `git rev-parse --absolute-git-dir` rather than
// guessed from the manifest dir, so this works no matter where the workspace
// lives relative to the checkout (and with worktrees, `GIT_DIR`, etc.). The
// emitted `rerun-if-changed` paths must exist: cargo treats a referenced but
// missing file as always-dirty, which would rerun this script (and, through
// it, every crate that consumes game-host) on every build.

use std::path::{Path, PathBuf};
use std::process::Command;

fn main() {
    let manifest_dir = Path::new(env!("CARGO_MANIFEST_DIR"));

    // --- git SHA ---
    let sha = git(&["rev-parse", "HEAD"], manifest_dir).unwrap_or_else(|| "unknown".to_owned());
    println!("cargo:rustc-env=GIT_SHA={sha}");

    // --- dirty-worktree flag ---
    let dirty = Command::new("git")
        .args(["status", "--porcelain"])
        .current_dir(manifest_dir)
        .output()
        .map(|out| !out.stdout.is_empty())
        .unwrap_or(false);
    println!(
        "cargo:rustc-env=GIT_DIRTY={}",
        if dirty { "true" } else { "false" }
    );

    // --- re-run on commit / ref move ---
    // Resolve the actual git dir so the rerun-if-changed paths below exist;
    // a missing path turns a build script permanently dirty and forces the
    // whole workspace to rebuild every time. HEAD itself is per-worktree, so
    // it's watched under `--absolute-git-dir`. But the ref HEAD points at
    // (e.g. `refs/heads/gdl`) is *not* per-worktree -- it lives under the
    // common git dir shared by the main checkout and all worktrees. Joining
    // it against the worktree-private dir instead (as this used to do)
    // produces a path that never exists, e.g.
    // `.git/worktrees/<name>/refs/heads/gdl`, which cargo treats as
    // permanently stale -- rerunning this script, and everything downstream
    // of game-host, on every single build.
    let Some(git_dir) = git(&["rev-parse", "--absolute-git-dir"], manifest_dir).map(PathBuf::from)
    else {
        return; // not a git checkout: keep the "unknown"/clean defaults
    };
    println!("cargo:rerun-if-changed={}", git_dir.join("HEAD").display());

    let Some(common_dir) =
        git(&["rev-parse", "--path-format=absolute", "--git-common-dir"], manifest_dir)
            .map(PathBuf::from)
    else {
        return;
    };
    if let Some(ref_path) = read_ref(&git_dir.join("HEAD"), &common_dir) {
        println!("cargo:rerun-if-changed={}", ref_path.display());
    }
}

/// Support a `ref: refs/heads/<branch>` HEAD by returning the ref file path,
/// resolved against the common git dir (shared across worktrees) rather than
/// `head_path`'s own directory, since refs aren't per-worktree.
fn read_ref(head_path: &Path, common_dir: &Path) -> Option<PathBuf> {
    let contents = std::fs::read_to_string(head_path).ok()?;
    contents
        .strip_prefix("ref: ")
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(|ref_path| common_dir.join(ref_path))
}

/// Run a git command, returning trimmed stdout as a String (or None on error).
fn git(args: &[&str], cwd: &Path) -> Option<String> {
    Command::new("git")
        .args(args)
        .current_dir(cwd)
        .output()
        .ok()
        .filter(|o| o.status.success())
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .map(|s| s.trim().to_owned())
}
