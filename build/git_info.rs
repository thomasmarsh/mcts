use std::path::{Path, PathBuf};
use std::process::Command;

pub fn emit() {
    let manifest_dir = Path::new(env!("CARGO_MANIFEST_DIR"));
    let sha = git(&["rev-parse", "HEAD"], manifest_dir).unwrap_or_else(|| "unknown".to_owned());
    println!("cargo:rustc-env=GIT_SHA={sha}");

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

    let Some(git_dir) = git(&["rev-parse", "--absolute-git-dir"], manifest_dir).map(PathBuf::from)
    else {
        return;
    };
    println!("cargo:rerun-if-changed={}", git_dir.join("HEAD").display());

    let Some(common_dir) = git(
        &["rev-parse", "--path-format=absolute", "--git-common-dir"],
        manifest_dir,
    )
    .map(PathBuf::from) else {
        return;
    };
    if let Some(ref_path) = read_ref(&git_dir.join("HEAD"), &common_dir) {
        println!("cargo:rerun-if-changed={}", ref_path.display());
    }
}

fn read_ref(head_path: &Path, common_dir: &Path) -> Option<PathBuf> {
    let contents = std::fs::read_to_string(head_path).ok()?;
    contents
        .strip_prefix("ref: ")
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(|ref_path| common_dir.join(ref_path))
}

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
