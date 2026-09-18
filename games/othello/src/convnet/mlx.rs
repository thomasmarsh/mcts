//! GPU-backed (MLX) reimplementation of [`super::CnnValueNet`]'s forward
//! pass, gated behind the `mlx` Cargo feature. This depends on the
//! hand-rolled `mlx-sys` crate rather than the published `mlx-rs`/`mlxrs`
//! crates because those vendor and build MLX from source (Metal shader
//! compilation, which most machines don't have the toolchain for); see
//! `crates/mlx-sys/build.rs` for how it links Homebrew's prebuilt
//! `mlx`/`mlx-c` instead, without compiling any Metal shaders.
//!
//! This is a from-scratch reimplementation of [`super::trunk`],
//! [`super::raw_value`] and [`super::raw_policy`]'s architecture against
//! MLX's tensor ops rather than a wrapper around the CPU path -- it reads
//! [`super::CnnValueNet`]'s weights directly (this module is a descendant
//! of `convnet`, so it can see that module's private weight-offset
//! constants and `input` helper) and must reproduce the CPU path's numbers
//! bit-for-bit within float tolerance; `tests` below pins that against the
//! same fixtures `convnet.rs`'s own `value_matches_python_reference_fixture`/
//! `policy_matches_python_reference_fixture` use.
//!
//! ## Layout conversions
//!
//! MLX's `conv2d` is channels-last: input `(N, H, W, C)`, weight
//! `(C_out, KH, KW, C_in)` (confirmed against the upstream MLX docs, not
//! guessed -- neither is documented in the `mlx-c` headers themselves).
//! [`super::CnnValueNet`]'s CPU path is channels-first per plane
//! (`input`'s `[plane][row][col]`) with conv weights laid out
//! `(C_out, C_in, KH, KW)` (see `super::conv3`). [`chw_to_nhwc`] and
//! [`permute_conv_weight`] do those two conversions; the dense-layer
//! (value/policy head) weights need no permutation, since their CPU layout
//! (`weight[cell * out_units + unit]`, `(in_features, out_features)`
//! row-major) already matches what `mlx_matmul` expects directly.
//!
//! ## Batching
//!
//! One [`value`]/[`all_policy_logits`] call still evaluates exactly one
//! board position, same as the CPU path -- no leaf-batching across tree
//! nodes. What *is* batched is the 8 D4-orientation forward passes a
//! single call already has to do (the CPU path does the same 8 passes,
//! just in a loop) -- stacking them into one `(8, 8, 8, 2)` MLX call is
//! not cross-leaf batching, it's using the fixed ensemble every call
//! already needs. [`value`] and [`all_policy_logits`] each still call
//! [`trunk`] independently, exactly like the CPU path's
//! `raw_value`/`raw_policy` -- both recompute the shared trunk rather than
//! caching it across the two heads, so the same forward pass runs twice
//! per position whenever both a value and a policy are needed.

use mlx_sys::*;

use mcts::algorithms::mcts::policy::PolicyLogits;
use mcts::evaluator::{Evaluator, Score, EVAL_MAGNITUDE_LIMIT};

use crate::policy::INV;
use crate::{Move, Othello, State};

use super::{
    CnnValueNet, BLOCKS, BOARD, CHANNELS, CNN_WEIGHTS, POLICY_OUTPUTS, VALUE_HIDDEN,
};

const ORIENTATIONS: i32 = 8;

// MLX device/stream setup is a one-time cost (device selection, Metal
// queue creation), but MLX's default device/stream registration is
// thread-local ("There is no Stream(gpu, 0) in current thread" if a stream
// created on one thread is passed to an op running on another) -- a
// process-wide cache broke immediately under `cargo test`'s multi-threaded
// runner. `thread_local!` matches MLX's own model and is still a one-time
// cost per thread, which is all self-play needs (single-threaded end to
// end, per this plan's own premise).
thread_local! {
    static STREAM: mlx_stream = unsafe {
        let gpu = mlx_device_new_type(mlx_device_type__MLX_GPU, 0);
        mlx_set_default_device(gpu);
        mlx_default_gpu_stream_new()
    };
}

fn stream() -> mlx_stream {
    STREAM.with(|s| *s)
}

unsafe fn check(rc: i32, what: &str) {
    assert_eq!(rc, 0, "mlx-sys call failed: {what}");
}

/// RAII handle over a raw `mlx_array`. `mlx-c` arrays are refcounted, but
/// nothing in the C API frees one automatically -- every `mlx_array` a
/// caller receives needs exactly one `mlx_array_free`. Doing that by
/// convention (every op function remembers to free every input it
/// consumes) is exactly the bug class that made the first version of this
/// module leak past Metal's resource limit mid-benchmark: one missed
/// `mlx_array_free` in `conv3`/`dense`/`channel_reduce`/`relu`, silent
/// until a long-running loop finally exhausted the limit. Wrapping every
/// handle in this type moves that responsibility onto the compiler instead
/// of onto memory: every op function below takes `MlxArray` by value and
/// simply lets Rust's normal drop order free its inputs at scope exit --
/// there is no longer a manual `mlx_array_free` call anywhere outside this
/// wrapper's own `Drop` impl.
struct MlxArray(mlx_array);

impl Drop for MlxArray {
    fn drop(&mut self) {
        unsafe {
            mlx_array_free(self.0);
        }
    }
}

impl Clone for MlxArray {
    /// A second independent handle to the *same* underlying array (via
    /// `mlx_array_set`'s retain, not a data copy) -- needed wherever a
    /// value must survive being consumed by an op that takes ownership of
    /// its input, e.g. a residual block's skip connection.
    fn clone(&self) -> Self {
        let mut new = unsafe { mlx_array_new() };
        unsafe {
            check(mlx_array_set(&mut new, self.0), "array_set (clone)");
        }
        MlxArray(new)
    }
}

fn from_data(data: &[f32], shape: &[i32]) -> MlxArray {
    MlxArray(unsafe {
        mlx_array_new_data(
            data.as_ptr() as *const _,
            shape.as_ptr(),
            shape.len() as i32,
            mlx_dtype__MLX_FLOAT32,
        )
    })
}

fn scalar(v: f32) -> MlxArray {
    MlxArray(unsafe { mlx_array_new_float(v) })
}

/// `x` is consumed (dropped, and so freed) when this returns.
fn relu(x: MlxArray, s: mlx_stream) -> MlxArray {
    let zero = scalar(0.0);
    let mut out = unsafe { mlx_array_new() };
    unsafe {
        check(mlx_maximum(&mut out, x.0, zero.0, s), "maximum");
    }
    MlxArray(out)
}

/// `x` is consumed.
fn add(a: MlxArray, b: MlxArray, s: mlx_stream) -> MlxArray {
    let mut out = unsafe { mlx_array_new() };
    unsafe {
        check(mlx_add(&mut out, a.0, b.0, s), "add");
    }
    MlxArray(out)
}

/// `x` is consumed.
fn add_bias(x: MlxArray, bias: &[f32], s: mlx_stream) -> MlxArray {
    add(x, from_data(bias, &[bias.len() as i32]), s)
}

/// One padded 3x3 conv (stride 1, padding 1) -- no activation. `x` is
/// consumed.
fn conv3(x: MlxArray, weight_chw: &[f32], bias: &[f32], out_ch: usize, in_ch: usize, s: mlx_stream) -> MlxArray {
    let w = permute_conv_weight(weight_chw, out_ch, in_ch);
    let w_arr = from_data(&w, &[out_ch as i32, 3, 3, in_ch as i32]);
    let mut out = unsafe { mlx_array_new() };
    unsafe {
        check(mlx_conv2d(&mut out, x.0, w_arr.0, 1, 1, 1, 1, 1, 1, 1, s), "conv2d");
    }
    add_bias(MlxArray(out), bias, s)
}

/// `(C_out, C_in, KH, KW)` (this crate's CPU conv layout, see
/// `super::conv3`) -> `(C_out, KH, KW, C_in)` (MLX's `conv2d` weight
/// layout).
fn permute_conv_weight(flat: &[f32], out_ch: usize, in_ch: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; out_ch * 3 * 3 * in_ch];
    for o in 0..out_ch {
        for ci in 0..in_ch {
            for kr in 0..3 {
                for kc in 0..3 {
                    let src = ((o * in_ch + ci) * 3 + kr) * 3 + kc;
                    let dst = ((o * 3 + kr) * 3 + kc) * in_ch + ci;
                    out[dst] = flat[src];
                }
            }
        }
    }
    out
}

/// One orientation's `[plane][row][col]` input (`super::CnnValueNet::input`)
/// -> `[row][col][plane]` (MLX's NHWC).
fn chw_to_nhwc(chw: &[f32; 2 * BOARD * BOARD]) -> [f32; 2 * BOARD * BOARD] {
    std::array::from_fn(|dst| {
        let plane = dst % 2;
        let cell = dst / 2;
        chw[plane * BOARD * BOARD + cell]
    })
}

/// Runs the shared stem+residual trunk over all 8 D4 orientations of
/// `state`, batched as one `(8, 8, 8, CHANNELS)` MLX array -- the GPU
/// counterpart of `super::CnnValueNet::trunk`, called once per orientation
/// in the CPU path but batched here (see the module docs' "Batching"
/// section for why this doesn't change the result).
fn trunk(net: &CnnValueNet, state: &State, s: mlx_stream) -> MlxArray {
    let mut input = vec![0.0f32; (ORIENTATIONS as usize) * 2 * BOARD * BOARD];
    for sym in 0..8usize {
        let chw = CnnValueNet::input(state, sym);
        let nhwc = chw_to_nhwc(&chw);
        input[sym * 2 * BOARD * BOARD..(sym + 1) * 2 * BOARD * BOARD].copy_from_slice(&nhwc);
    }
    let x = from_data(&input, &[ORIENTATIONS, BOARD as i32, BOARD as i32, 2]);
    trunk_rows(net, x, s)
}

/// Same trunk as [`trunk`], stacking every state's 8 D4 orientations into
/// one `(states.len() * 8, 8, 8, CHANNELS)` MLX array -- the batched
/// counterpart [`evaluate_batch`] needs so a whole live self-play batch
/// shares a single GPU forward pass instead of one call per state.
fn trunk_batch(net: &CnnValueNet, states: &[State], s: mlx_stream) -> MlxArray {
    let n = states.len();
    let mut input = vec![0.0f32; n * (ORIENTATIONS as usize) * 2 * BOARD * BOARD];
    for (i, state) in states.iter().enumerate() {
        for sym in 0..8usize {
            let chw = CnnValueNet::input(state, sym);
            let nhwc = chw_to_nhwc(&chw);
            let row = i * 8 + sym;
            input[row * 2 * BOARD * BOARD..(row + 1) * 2 * BOARD * BOARD].copy_from_slice(&nhwc);
        }
    }
    let x = from_data(&input, &[(n as i32) * ORIENTATIONS, BOARD as i32, BOARD as i32, 2]);
    trunk_rows(net, x, s)
}

/// The stem+residual-block compute shared by [`trunk`] and [`trunk_batch`],
/// taking the already-stacked NHWC input so it is agnostic to how many
/// orientation rows the caller stacked into it (`conv2d`/`relu`/`add` all
/// operate on the leading batch dimension unchanged regardless of its
/// size).
fn trunk_rows(net: &CnnValueNet, x: MlxArray, s: mlx_stream) -> MlxArray {
    let w = net.weights();
    let mut at = 0usize;
    let stem_w = &w[at..at + CHANNELS * 2 * 9];
    let stem_b = &w[at + CHANNELS * 2 * 9..at + CHANNELS * 2 * 9 + CHANNELS];
    at += CHANNELS * 2 * 9 + CHANNELS;
    let mut h = relu(conv3(x, stem_w, stem_b, CHANNELS, 2, s), s);

    for _ in 0..BLOCKS {
        let branch = h.clone();

        let w1 = &w[at..at + CHANNELS * CHANNELS * 9];
        let b1 = &w[at + CHANNELS * CHANNELS * 9..at + CHANNELS * CHANNELS * 9 + CHANNELS];
        at += CHANNELS * CHANNELS * 9 + CHANNELS;
        let y = relu(conv3(h, w1, b1, CHANNELS, CHANNELS, s), s);

        let w2 = &w[at..at + CHANNELS * CHANNELS * 9];
        let b2 = &w[at + CHANNELS * CHANNELS * 9..at + CHANNELS * CHANNELS * 9 + CHANNELS];
        at += CHANNELS * CHANNELS * 9 + CHANNELS;
        let z = conv3(y, w2, b2, CHANNELS, CHANNELS, s);

        h = relu(add(z, branch, s), s);
    }
    debug_assert_eq!(at, CNN_WEIGHTS - super::VALUE_HEAD_WEIGHTS - super::POLICY_HEAD_WEIGHTS);
    h
}

/// `(8, BOARD, BOARD, CHANNELS)` trunk output -> `(8, BOARD*BOARD)`
/// per-cell scalar reduction (a 1x1 conv down to one channel) + ReLU, the
/// shared first step of both heads (`super::raw_value`/`super::raw_policy`'s
/// `*_features` computation).
fn channel_reduce(h: MlxArray, conv_w: &[f32], bias: f32, s: mlx_stream) -> MlxArray {
    channel_reduce_rows(h, conv_w, bias, ORIENTATIONS, s)
}

/// [`channel_reduce`] generalized to `rows` orientation rows (`states.len()
/// * 8` in the batched path, rather than always exactly 8) -- see
/// [`trunk_rows`] for why this is safe to parameterize.
fn channel_reduce_rows(h: MlxArray, conv_w: &[f32], bias: f32, rows: i32, s: mlx_stream) -> MlxArray {
    let mut flat_h = unsafe { mlx_array_new() };
    let shape = [rows * (BOARD * BOARD) as i32, CHANNELS as i32];
    unsafe {
        check(mlx_reshape(&mut flat_h, h.0, shape.as_ptr(), 2, s), "reshape (flatten trunk)");
    }
    drop(h);

    let reduced = dense(MlxArray(flat_h), conv_w, &[bias], CHANNELS, 1, s);
    let reduced = relu(reduced, s);

    let mut out = unsafe { mlx_array_new() };
    let shape2 = [rows, (BOARD * BOARD) as i32];
    unsafe {
        check(mlx_reshape(&mut out, reduced.0, shape2.as_ptr(), 2, s), "reshape (per-orientation features)");
    }
    drop(reduced);
    MlxArray(out)
}

/// `x` is consumed.
fn dense(x: MlxArray, weight: &[f32], bias: &[f32], in_features: usize, out_features: usize, s: mlx_stream) -> MlxArray {
    let w_arr = from_data(weight, &[in_features as i32, out_features as i32]);
    let mut out = unsafe { mlx_array_new() };
    unsafe {
        check(mlx_matmul(&mut out, x.0, w_arr.0, s), "matmul (dense)");
    }
    add_bias(MlxArray(out), bias, s)
}

fn eval_to_vec(x: MlxArray, len: usize) -> Vec<f32> {
    unsafe {
        check(mlx_array_eval(x.0), "eval");
        let ptr = mlx_array_data_float32(x.0);
        if ptr.is_null() {
            vec![0.0; len]
        } else {
            std::slice::from_raw_parts(ptr, len).to_vec()
        }
    }
    // `x` drops (and frees) here.
}

/// GPU-backed equivalent of [`super::CnnValueNet::value`]: the D4-averaged
/// value, computed as one batched (8-orientation) MLX call.
pub fn value(net: &CnnValueNet, state: &State) -> f32 {
    let s = stream();
    let w = net.weights();
    let h = trunk(net, state, s);

    let mut at = CNN_WEIGHTS - super::VALUE_HEAD_WEIGHTS - super::POLICY_HEAD_WEIGHTS;
    let value_conv = &w[at..at + CHANNELS];
    let value_bias = w[at + CHANNELS];
    at += CHANNELS + 1;
    let features = channel_reduce(h, value_conv, value_bias, s);

    let w1 = &w[at..at + BOARD * BOARD * VALUE_HIDDEN];
    let b1 = &w[at + BOARD * BOARD * VALUE_HIDDEN..at + BOARD * BOARD * VALUE_HIDDEN + VALUE_HIDDEN];
    at += BOARD * BOARD * VALUE_HIDDEN + VALUE_HIDDEN;
    let hidden = relu(dense(features, w1, b1, BOARD * BOARD, VALUE_HIDDEN, s), s);

    let w2 = &w[at..at + VALUE_HIDDEN];
    let b2 = &w[at + VALUE_HIDDEN..at + VALUE_HIDDEN + 1];
    at += VALUE_HIDDEN + 1;
    let out = dense(hidden, w2, b2, VALUE_HIDDEN, 1, s);
    let mut tanh_out = unsafe { mlx_array_new() };
    unsafe {
        check(mlx_tanh(&mut tanh_out, out.0, s), "tanh");
    }
    drop(out);

    debug_assert_eq!(at, CNN_WEIGHTS - super::POLICY_HEAD_WEIGHTS);
    let per_orientation = eval_to_vec(MlxArray(tanh_out), ORIENTATIONS as usize);
    per_orientation.iter().sum::<f32>() / ORIENTATIONS as f32
}

/// GPU-backed equivalent of [`super::CnnValueNet::all_policy_logits`]: the
/// D4-averaged, real-board-frame policy logits, computed as one batched
/// (8-orientation) MLX call.
pub fn all_policy_logits(net: &CnnValueNet, state: &State) -> [f64; POLICY_OUTPUTS] {
    let s = stream();
    let w = net.weights();
    let h = trunk(net, state, s);

    let mut at = CNN_WEIGHTS - super::POLICY_HEAD_WEIGHTS;
    let policy_conv = &w[at..at + CHANNELS];
    let policy_bias = w[at + CHANNELS];
    at += CHANNELS + 1;
    let features = channel_reduce(h, policy_conv, policy_bias, s);

    let policy_w = &w[at..at + BOARD * BOARD * POLICY_OUTPUTS];
    let policy_b = &w[at + BOARD * BOARD * POLICY_OUTPUTS..at + BOARD * BOARD * POLICY_OUTPUTS + POLICY_OUTPUTS];
    at += BOARD * BOARD * POLICY_OUTPUTS + POLICY_OUTPUTS;
    let logits = dense(features, policy_w, policy_b, BOARD * BOARD, POLICY_OUTPUTS, s);

    debug_assert_eq!(at, CNN_WEIGHTS);
    let raw = eval_to_vec(logits, (ORIENTATIONS as usize) * POLICY_OUTPUTS);

    let mut out = [0.0f64; POLICY_OUTPUTS];
    for sym in 0..8usize {
        for real_sq in 0..POLICY_OUTPUTS {
            out[real_sq] += raw[sym * POLICY_OUTPUTS + INV[sym][real_sq] as usize] as f64;
        }
    }
    for v in out.iter_mut() {
        *v /= 8.0;
    }
    out
}

/// Batched GPU forward pass: value and policy for a whole slice of states
/// in one MLX call each, sharing a single stacked trunk between both heads
/// *and* across every state -- unlike [`value`]/[`all_policy_logits`],
/// which each recompute the trunk independently and only ever see one
/// state. This is the entry point `mcts_batch::othello::MlxOthelloOracle`
/// uses: it already hands a whole simulation round's live batch to one
/// oracle call, so routing that straight into one GPU call (instead of
/// spreading per-state GPU calls across CPU threads, which would just
/// serialize on the same physical GPU) is what actually cashes in
/// `mcts-gpu.md`'s batching premise for the GPU evaluator.
pub fn evaluate_batch(net: &CnnValueNet, states: &[State]) -> (Vec<f32>, Vec<[f64; POLICY_OUTPUTS]>) {
    // A self-play caller's live-batch size shrinks by a different amount
    // every call as games finish, so this function rarely sees the same
    // `states.len()` twice in a row -- and `mlx-c`'s allocator keeps a
    // permanent, never-shrinking cache entry per distinct shape it has ever
    // been asked to allocate (confirmed via `mlx_get_cache_memory`/
    // `mlx_clear_cache`, not just inferred from process-level memory), sized
    // proportional to `states.len() * CHANNELS`. At a wide enough net a
    // single shape's cache entry can be gigabytes, so accumulating even a
    // handful of distinct shapes over a run exhausts memory outright. This
    // guard clears that cache on every return path (measured: no wall-clock
    // cost on a workload that keeps reusing the same shape, since there's
    // nothing to clear when the shape hasn't changed).
    let _clear_cache_on_return = ClearCacheOnDrop;
    if states.is_empty() {
        return (Vec::new(), Vec::new());
    }
    let s = stream();
    let w = net.weights();
    let n = states.len();
    let rows = (n as i32) * ORIENTATIONS;

    let h = trunk_batch(net, states, s);
    let h_for_policy = h.clone();

    let mut at = CNN_WEIGHTS - super::VALUE_HEAD_WEIGHTS - super::POLICY_HEAD_WEIGHTS;
    let value_conv = &w[at..at + CHANNELS];
    let value_bias = w[at + CHANNELS];
    at += CHANNELS + 1;
    let features = channel_reduce_rows(h, value_conv, value_bias, rows, s);

    let w1 = &w[at..at + BOARD * BOARD * VALUE_HIDDEN];
    let b1 = &w[at + BOARD * BOARD * VALUE_HIDDEN..at + BOARD * BOARD * VALUE_HIDDEN + VALUE_HIDDEN];
    at += BOARD * BOARD * VALUE_HIDDEN + VALUE_HIDDEN;
    let hidden = relu(dense(features, w1, b1, BOARD * BOARD, VALUE_HIDDEN, s), s);

    let w2 = &w[at..at + VALUE_HIDDEN];
    let b2 = &w[at + VALUE_HIDDEN..at + VALUE_HIDDEN + 1];
    at += VALUE_HIDDEN + 1;
    let out = dense(hidden, w2, b2, VALUE_HIDDEN, 1, s);
    let mut tanh_out = unsafe { mlx_array_new() };
    unsafe {
        check(mlx_tanh(&mut tanh_out, out.0, s), "tanh");
    }
    drop(out);
    debug_assert_eq!(at, CNN_WEIGHTS - super::POLICY_HEAD_WEIGHTS);

    let per_orientation = eval_to_vec(MlxArray(tanh_out), rows as usize);
    let values: Vec<f32> =
        (0..n).map(|i| per_orientation[i * 8..(i + 1) * 8].iter().sum::<f32>() / ORIENTATIONS as f32).collect();

    let mut at2 = CNN_WEIGHTS - super::POLICY_HEAD_WEIGHTS;
    let policy_conv = &w[at2..at2 + CHANNELS];
    let policy_bias = w[at2 + CHANNELS];
    at2 += CHANNELS + 1;
    let pfeatures = channel_reduce_rows(h_for_policy, policy_conv, policy_bias, rows, s);

    let policy_w = &w[at2..at2 + BOARD * BOARD * POLICY_OUTPUTS];
    let policy_b = &w[at2 + BOARD * BOARD * POLICY_OUTPUTS..at2 + BOARD * BOARD * POLICY_OUTPUTS + POLICY_OUTPUTS];
    at2 += BOARD * BOARD * POLICY_OUTPUTS + POLICY_OUTPUTS;
    let logits = dense(pfeatures, policy_w, policy_b, BOARD * BOARD, POLICY_OUTPUTS, s);
    debug_assert_eq!(at2, CNN_WEIGHTS);

    let raw = eval_to_vec(logits, rows as usize * POLICY_OUTPUTS);
    let policies: Vec<[f64; POLICY_OUTPUTS]> = (0..n)
        .map(|i| {
            let mut out = [0.0f64; POLICY_OUTPUTS];
            for sym in 0..8usize {
                let base = (i * 8 + sym) * POLICY_OUTPUTS;
                for real_sq in 0..POLICY_OUTPUTS {
                    out[real_sq] += raw[base + INV[sym][real_sq] as usize] as f64;
                }
            }
            for v in out.iter_mut() {
                *v /= 8.0;
            }
            out
        })
        .collect();

    (values, policies)
}

/// Snapshot of `mlx-c`'s own memory accounting (`active`: bytes backing
/// currently-live arrays; `cache`: bytes the allocator is holding onto but
/// isn't backing anything live right now -- freed buffers kept around for
/// reuse rather than returned to the OS; `peak`: the high-water mark of
/// `active` since the last [`reset_peak_memory`]). Diagnostic only, not used
/// by any production call path -- exists to distinguish "GPU memory is
/// genuinely in use" from "the allocator's cache is holding stale buffers"
/// when investigating memory growth from the Rust side, since neither is
/// visible in a process's own RSS.
pub fn memory_stats() -> (usize, usize, usize) {
    unsafe {
        let mut active = 0usize;
        let mut cache = 0usize;
        let mut peak = 0usize;
        check(mlx_get_active_memory(&mut active), "get_active_memory");
        check(mlx_get_cache_memory(&mut cache), "get_cache_memory");
        check(mlx_get_peak_memory(&mut peak), "get_peak_memory");
        (active, cache, peak)
    }
}

/// Drops every buffer `mlx-c`'s allocator is holding in its reuse cache
/// (see [`memory_stats`]'s `cache` field) without touching any array that's
/// still live. Called automatically by [`ClearCacheOnDrop`]; exposed on its
/// own too for diagnostics (e.g. `mcts-batch`'s `mlx_shape_churn_probe`
/// example) that want to clear on a different cadence than one call.
pub fn clear_cache() {
    unsafe {
        check(mlx_clear_cache(), "clear_cache");
    }
}

/// RAII guard that calls [`clear_cache`] when dropped, so a function that
/// builds a shape-varying computation graph (like [`evaluate_batch`]) clears
/// the allocator's resulting cache entry on every return path -- including
/// an early return -- without every call site needing to remember to do it
/// itself. See [`evaluate_batch`]'s own docs for why this specific function
/// needs it (varying batch shapes) while [`value`]/[`all_policy_logits`]
/// don't (always the same fixed 8-orientation shape).
struct ClearCacheOnDrop;

impl Drop for ClearCacheOnDrop {
    fn drop(&mut self) {
        clear_cache();
    }
}

/// GPU-backed drop-in for `CnnValueNet` as a self-play evaluator: same
/// `Evaluator<Othello>`/`PolicyLogits<Othello>` contract as
/// `super::CnnValueNet` (see `crate::selfplay::CnnGumbelPlayer`, which
/// plugs the CPU version into `TreeSearch` the same way), routing both
/// through [`value`]/[`all_policy_logits`] instead of the CPU forward pass.
/// This is what the throughput comparison in
/// `examples/mlx_selfplay_bench.rs` swaps in for `CnnValueNet` to compare
/// like for like -- same search, same call pattern (one evaluation per
/// tree leaf, no batching across leaves), only the per-call forward pass
/// differs.
#[derive(Clone, Default)]
pub struct MlxCnnValueNet(CnnValueNet);

impl MlxCnnValueNet {
    pub fn from_weights(weights: Vec<f32>) -> Self {
        Self(CnnValueNet::from_weights(weights))
    }

    pub fn load(path: impl AsRef<std::path::Path>) -> std::io::Result<Self> {
        Ok(Self(CnnValueNet::load(path)?))
    }

    /// The wrapped weights, for callers (e.g. `mcts_batch::othello::
    /// MlxOthelloOracle`) that need to hand this net's weights to a
    /// free function like [`evaluate_batch`] rather than a single-state
    /// trait method.
    pub fn inner(&self) -> &CnnValueNet {
        &self.0
    }
}

impl Evaluator<Othello> for MlxCnnValueNet {
    fn evaluate(&self, state: &State) -> Score {
        (value(&self.0, state) * EVAL_MAGNITUDE_LIMIT as f32).round() as Score
    }
}

impl PolicyLogits<Othello> for MlxCnnValueNet {
    fn logits(&mut self, state: &State, actions: &[Move]) -> Vec<f64> {
        let all = all_policy_logits(&self.0, state);
        actions
            .iter()
            .map(|a| {
                if *a == Move::PASS {
                    all.iter().sum::<f64>() / POLICY_OUTPUTS as f64
                } else {
                    all[a.0 as usize]
                }
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Player, BB};

    fn state(black: u64, white: u64, turn: Player) -> State {
        State { black: BB::from_bits(black), white: BB::from_bits(white), turn, last_pass: false, hashes: [0u64; 8] }
    }

    /// Same weight formula, geometry and state as `convnet.rs`'s
    /// `value_matches_python_reference_fixture` -- the MLX and CPU forward
    /// passes must agree, not just each independently pass their own tests.
    #[test]
    fn value_matches_cpu_reference_fixture() {
        let weights: Vec<f32> = (0..CNN_WEIGHTS).map(|i| ((i as f64 - CNN_WEIGHTS as f64 / 2.0) * 1e-6) as f32).collect();
        let net = CnnValueNet::from_weights(weights);
        let s = state((1 << 0) | (1 << 2) | (1 << 8), 1 << 1 | (1 << 7), Player::Black);
        let got = value(&net, &s);
        assert!((got - 0.011_578_533).abs() < 1e-5, "{got}");
    }

    /// Same weight formula, geometry and state as `convnet.rs`'s
    /// `policy_matches_python_reference_fixture`.
    #[test]
    fn policy_matches_cpu_reference_fixture() {
        let weights: Vec<f32> = (0..CNN_WEIGHTS).map(|i| ((i as f64 - CNN_WEIGHTS as f64 / 2.0) * 1e-6) as f32).collect();
        let net = CnnValueNet::from_weights(weights);
        let s = state((1 << 0) | (1 << 2) | (1 << 8), 1 << 1 | (1 << 7), Player::Black);
        let got = all_policy_logits(&net, &s);
        let expected = [
            0.019_165_495_410_561_56,
            0.019_165_497_273_206_71,
            0.019_165_497_273_206_71,
            0.019_165_497_273_206_71,
            0.019_165_497_273_206_71,
            0.019_165_497_273_206_71,
            0.019_165_497_273_206_71,
            0.019_165_495_410_561_56,
        ];
        for (actual, expected) in got.iter().take(8).zip(expected) {
            assert!((actual - expected).abs() < 1e-4, "{actual} vs {expected}");
        }
    }

    /// Cross-check against the live CPU path (not just the pinned fixture
    /// constants above) on non-trivial, non-symmetric random weights, so a
    /// bug that happens to cancel out on the fixture's specific weights
    /// can't hide.
    #[test]
    fn matches_cpu_on_random_weights_and_states() {
        let weights: Vec<f32> = (0..CNN_WEIGHTS).map(|i| (i as f32 * 0.0013).sin() * 0.1).collect();
        let net = CnnValueNet::from_weights(weights);
        let boards = [
            (1u64 << 27 | 1 << 28 | 1 << 35, 1u64 << 26 | 1 << 34 | 1 << 36),
            ((1u64 << 0) | (1 << 9) | (1 << 20), (1u64 << 27) | (1 << 36) | (1 << 45)),
        ];
        for (black, white) in boards {
            for turn in [Player::Black, Player::White] {
                let s = state(black, white, turn);
                let cpu_value = net.value(&s);
                let mlx_value = value(&net, &s);
                assert!((cpu_value - mlx_value).abs() < 1e-4, "value mismatch: cpu {cpu_value} mlx {mlx_value}");

                let cpu_policy = net.all_policy_logits(&s);
                let mlx_policy = all_policy_logits(&net, &s);
                for sq in 0..POLICY_OUTPUTS {
                    assert!(
                        (cpu_policy[sq] - mlx_policy[sq]).abs() < 1e-3,
                        "policy mismatch at square {sq}: cpu {} mlx {}",
                        cpu_policy[sq],
                        mlx_policy[sq]
                    );
                }
            }
        }
    }

    /// [`evaluate_batch`] must agree with the per-state [`value`]/
    /// [`all_policy_logits`] path it's meant to replace at real batch
    /// sizes -- a batch of one has to match exactly (same trunk math, just
    /// stacked differently), and a mixed batch of several distinct states
    /// must not let one state's rows leak into another's D4 average.
    #[test]
    fn batched_matches_per_state_calls() {
        let weights: Vec<f32> = (0..CNN_WEIGHTS).map(|i| (i as f32 * 0.0013).sin() * 0.1).collect();
        let net = CnnValueNet::from_weights(weights);
        let states = [
            state(1u64 << 27 | 1 << 28 | 1 << 35, 1u64 << 26 | 1 << 34 | 1 << 36, Player::Black),
            state((1u64 << 0) | (1 << 9) | (1 << 20), (1u64 << 27) | (1 << 36) | (1 << 45), Player::White),
            state((1 << 0) | (1 << 2) | (1 << 8), 1 << 1 | (1 << 7), Player::Black),
        ];

        let (batch_values, batch_policies) = evaluate_batch(&net, &states);
        assert_eq!(batch_values.len(), states.len());
        assert_eq!(batch_policies.len(), states.len());

        for (i, s) in states.iter().enumerate() {
            let want_value = value(&net, s);
            assert!((batch_values[i] - want_value).abs() < 1e-4, "value[{i}]: batch {} vs per-state {want_value}", batch_values[i]);

            let want_policy = all_policy_logits(&net, s);
            for sq in 0..POLICY_OUTPUTS {
                assert!(
                    (batch_policies[i][sq] - want_policy[sq]).abs() < 1e-4,
                    "policy[{i}][{sq}]: batch {} vs per-state {}",
                    batch_policies[i][sq],
                    want_policy[sq]
                );
            }
        }

        // A batch of one is the degenerate case `trunk_batch`/
        // `channel_reduce_rows` must also handle correctly.
        let (one_value, one_policy) = evaluate_batch(&net, &states[..1]);
        assert!((one_value[0] - batch_values[0]).abs() < 1e-6);
        assert_eq!(one_policy[0], batch_policies[0]);

        // An empty batch must not panic (`mcts_batch::search::eval_batch`
        // calls the oracle with zero non-terminal entries whenever every
        // live game's frontier is already terminal this round).
        let (empty_values, empty_policies) = evaluate_batch(&net, &[]);
        assert!(empty_values.is_empty());
        assert!(empty_policies.is_empty());
    }
}
