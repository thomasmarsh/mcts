//! Raw `bindgen`-generated FFI over `mlx-c`. See `build.rs` for why this
//! links Homebrew's prebuilt dylibs instead of vendoring/building MLX from
//! source. This is a `-sys` crate by convention: no safe wrappers, just the
//! C API as bindgen sees it -- callers (`game_othello::convnet::mlx`) own
//! all the `unsafe`.
#![allow(non_camel_case_types, non_snake_case, non_upper_case_globals, dead_code)]

include!(concat!(env!("OUT_DIR"), "/bindings.rs"));
