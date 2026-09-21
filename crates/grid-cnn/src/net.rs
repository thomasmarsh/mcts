//! The MLX forward pass. MLX convolutions are channels-last: activations `(n, row, col, channel)`
//! and conv weights `(out, kh, kw, in)`; the weight file is channels-first, so weights are
//! permuted once when a [`Net`] is built.

use mlx_sys::*;

use crate::{Geometry, Weights};

// MLX's default device and stream registration is thread-local, so each thread that evaluates a
// net gets its own GPU stream. The arrays a `Net` owns are immutable host-copied buffers and are
// safe to read from any thread.
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

fn check(rc: i32, what: &str) {
    assert_eq!(rc, 0, "mlx call failed: {what}");
}

/// Owns one reference to an `mlx_array`; freed on drop.
struct Array(mlx_array);

impl Drop for Array {
    fn drop(&mut self) {
        unsafe { mlx_array_free(self.0) };
    }
}

impl Clone for Array {
    fn clone(&self) -> Self {
        let mut new = unsafe { mlx_array_new() };
        check(unsafe { mlx_array_set(&mut new, self.0) }, "array_set");
        Array(new)
    }
}

fn from_data(data: &[f32], shape: &[usize]) -> Array {
    assert_eq!(data.len(), shape.iter().product::<usize>());
    let shape: Vec<i32> = shape.iter().map(|&d| d as i32).collect();
    Array(unsafe {
        mlx_array_new_data(
            data.as_ptr().cast(),
            shape.as_ptr(),
            shape.len() as i32,
            mlx_dtype__MLX_FLOAT32,
        )
    })
}

fn binary(
    f: unsafe extern "C" fn(*mut mlx_array, mlx_array, mlx_array, mlx_stream) -> i32,
    a: &Array,
    b: &Array,
    what: &str,
) -> Array {
    let mut out = unsafe { mlx_array_new() };
    check(unsafe { f(&mut out, a.0, b.0, stream()) }, what);
    Array(out)
}

fn reshape(x: &Array, shape: &[usize]) -> Array {
    let shape: Vec<i32> = shape.iter().map(|&d| d as i32).collect();
    let mut out = unsafe { mlx_array_new() };
    check(
        unsafe { mlx_reshape(&mut out, x.0, shape.as_ptr(), shape.len(), stream()) },
        "reshape",
    );
    Array(out)
}

fn conv3(x: &Array, weight: &Array) -> Array {
    let mut out = unsafe { mlx_array_new() };
    check(
        unsafe { mlx_conv2d(&mut out, x.0, weight.0, 1, 1, 1, 1, 1, 1, 1, stream()) },
        "conv2d",
    );
    Array(out)
}

fn tanh(x: &Array) -> Array {
    let mut out = unsafe { mlx_array_new() };
    check(unsafe { mlx_tanh(&mut out, x.0, stream()) }, "tanh");
    Array(out)
}

struct Conv3 {
    weight: Array,
    bias: Array,
}

/// A dense layer, also used for a 1x1 conv over flattened cells.
struct Dense {
    weight: Array,
    bias: Array,
}

struct Head {
    conv: Dense,
    dense: Dense,
}

/// Reads consecutive tensors out of the flat weight vector.
struct Reader<'a> {
    data: &'a [f32],
    at: usize,
}

impl Reader<'_> {
    fn take(&mut self, n: usize) -> &[f32] {
        let out = &self.data[self.at..self.at + n];
        self.at += n;
        out
    }

    fn conv3(&mut self, in_ch: usize, out_ch: usize) -> Conv3 {
        let src = self.take(out_ch * in_ch * 9).to_vec();
        let mut permuted = vec![0.0f32; src.len()];
        for o in 0..out_ch {
            for i in 0..in_ch {
                for k in 0..9 {
                    permuted[(o * 9 + k) * in_ch + i] = src[(o * in_ch + i) * 9 + k];
                }
            }
        }
        let weight = from_data(&permuted, &[out_ch, 3, 3, in_ch]);
        let bias = from_data(self.take(out_ch), &[out_ch]);
        Conv3 { weight, bias }
    }

    /// A 1x1 conv stored `(out, in)`, kept as an `(in, out)` matrix.
    fn conv1(&mut self, in_ch: usize, out_ch: usize) -> Dense {
        let src = self.take(out_ch * in_ch).to_vec();
        let mut transposed = vec![0.0f32; src.len()];
        for o in 0..out_ch {
            for i in 0..in_ch {
                transposed[i * out_ch + o] = src[o * in_ch + i];
            }
        }
        let weight = from_data(&transposed, &[in_ch, out_ch]);
        let bias = from_data(self.take(out_ch), &[out_ch]);
        Dense { weight, bias }
    }

    fn dense(&mut self, in_features: usize, out_features: usize) -> Dense {
        let weight = from_data(
            self.take(in_features * out_features),
            &[in_features, out_features],
        );
        let bias = from_data(self.take(out_features), &[out_features]);
        Dense { weight, bias }
    }

    /// A dense layer fed by a head's feature map, whose rows are stored `(plane, row, col)`;
    /// activations here are `(row, col, plane)`, so the rows are reordered to match.
    fn dense_from_map(&mut self, planes: usize, cells: usize, out_features: usize) -> Dense {
        let src = self.take(planes * cells * out_features).to_vec();
        let mut permuted = vec![0.0f32; src.len()];
        for p in 0..planes {
            for cell in 0..cells {
                let from = (p * cells + cell) * out_features;
                let to = (cell * planes + p) * out_features;
                permuted[to..to + out_features].copy_from_slice(&src[from..from + out_features]);
            }
        }
        let weight = from_data(&permuted, &[planes * cells, out_features]);
        let bias = from_data(self.take(out_features), &[out_features]);
        Dense { weight, bias }
    }
}

pub struct Net {
    geometry: Geometry,
    zero: Array,
    stem: Conv3,
    blocks: Vec<(Conv3, Conv3)>,
    policy: Head,
    value: Head,
    value_out: Dense,
}

// SAFETY: a `Net` never mutates its arrays after construction (see the note on `STREAM`), and
// every operation on them runs on the calling thread's own stream.
unsafe impl Send for Net {}
unsafe impl Sync for Net {}

pub struct Forward {
    /// `(n)` tanh values from each position's side to move.
    pub values: Vec<f32>,
    /// `(n, policy_out)` raw logits, row-major.
    pub logits: Vec<f32>,
}

impl Net {
    pub fn new(weights: &Weights) -> Self {
        let g = weights.geometry;
        assert_eq!(
            weights.data.len(),
            g.n_weights(),
            "weight vector does not match its geometry"
        );
        let mut r = Reader {
            data: &weights.data,
            at: 0,
        };
        let cells = g.cells();
        let stem = r.conv3(g.in_planes, g.channels);
        let blocks = (0..g.blocks)
            .map(|_| {
                (
                    r.conv3(g.channels, g.channels),
                    r.conv3(g.channels, g.channels),
                )
            })
            .collect();
        let policy = Head {
            conv: r.conv1(g.channels, g.policy_planes),
            dense: r.dense_from_map(g.policy_planes, cells, g.policy_out),
        };
        let value = Head {
            conv: r.conv1(g.channels, g.value_planes),
            dense: r.dense_from_map(g.value_planes, cells, g.value_hidden),
        };
        let value_out = r.dense(g.value_hidden, 1);
        assert_eq!(r.at, weights.data.len());
        Net {
            geometry: g,
            zero: from_data(&[0.0], &[1]),
            stem,
            blocks,
            policy,
            value,
            value_out,
        }
    }

    pub fn geometry(&self) -> Geometry {
        self.geometry
    }

    fn relu(&self, x: &Array) -> Array {
        binary(mlx_maximum, x, &self.zero, "maximum")
    }

    fn conv_bias(&self, x: &Array, conv: &Conv3) -> Array {
        binary(mlx_add, &conv3(x, &conv.weight), &conv.bias, "add bias")
    }

    fn dense(&self, x: &Array, layer: &Dense) -> Array {
        binary(
            mlx_add,
            &binary(mlx_matmul, x, &layer.weight, "matmul"),
            &layer.bias,
            "add bias",
        )
    }

    /// Evaluate `n` positions, each `size * size * in_planes` floats laid out `(row, col, plane)`,
    /// one MLX call. Callers bound `n` to what fits in memory.
    pub fn forward(&self, planes: &[f32], n: usize) -> Forward {
        let g = &self.geometry;
        let cells = g.cells();
        assert_eq!(
            planes.len(),
            n * cells * g.in_planes,
            "input is not n positions of the net's geometry"
        );
        if n == 0 {
            return Forward {
                values: Vec::new(),
                logits: Vec::new(),
            };
        }
        let x = from_data(planes, &[n, g.size, g.size, g.in_planes]);
        let mut h = self.relu(&self.conv_bias(&x, &self.stem));
        for (c1, c2) in &self.blocks {
            let y = self.relu(&self.conv_bias(&h, c1));
            let z = self.conv_bias(&y, c2);
            h = self.relu(&binary(mlx_add, &z, &h, "residual add"));
        }
        let flat = reshape(&h, &[n * cells, g.channels]);
        let head_features = |head: &Head, planes: usize| {
            let f = self.relu(&self.dense(&flat, &head.conv));
            reshape(&f, &[n, cells * planes])
        };
        let logits = self.dense(
            &head_features(&self.policy, g.policy_planes),
            &self.policy.dense,
        );
        let hidden = self.relu(&self.dense(
            &head_features(&self.value, g.value_planes),
            &self.value.dense,
        ));
        let value = tanh(&self.dense(&hidden, &self.value_out));

        let outputs = [value.0, logits.0];
        let vec = unsafe { mlx_vector_array_new_data(outputs.as_ptr(), outputs.len()) };
        let rc = unsafe { mlx_eval(vec) };
        unsafe { mlx_vector_array_free(vec) };
        check(rc, "eval");
        let read = |a: &Array, len: usize| -> Vec<f32> {
            let ptr = unsafe { mlx_array_data_float32(a.0) };
            assert!(!ptr.is_null(), "evaluated array has no data");
            unsafe { std::slice::from_raw_parts(ptr, len) }.to_vec()
        };
        Forward {
            values: read(&value, n),
            logits: read(&logits, n * g.policy_out),
        }
    }
}

/// Release the buffers MLX's allocator keeps for reuse. Its cache holds one entry per distinct
/// array shape it has served, so a caller whose batch size changes every call should clear it.
pub fn clear_cache() {
    check(unsafe { mlx_clear_cache() }, "clear_cache");
}
