//! A residual CNN over a square board grid, evaluated on the GPU through MLX.
//!
//! The network is described entirely by [`Geometry`] (board size, input planes, channels, residual
//! blocks, policy/value head widths), so one forward pass serves any square-board game. A trained
//! net is one file ([`Weights`]): a small header carrying the geometry, then every weight as
//! little-endian `f32`. Batch-norm is folded into the convolutions before export, so the network
//! here is only convolutions, ReLUs, residual additions and dense layers.
//!
//! Layout of the flat weight vector (see [`Geometry::n_weights`]); convolution weights are
//! `(out, in, kh, kw)`, dense weights `(in, out)` row-major, and dense inputs that come from a
//! head's feature map are ordered `(plane, row, col)`:
//!
//! 1. stem conv (`in_planes -> channels`, 3x3) weight, bias
//! 2. per residual block: conv1 weight, bias, conv2 weight, bias (all `channels -> channels`, 3x3)
//! 3. policy head: 1x1 conv (`channels -> policy_planes`) weight, bias; dense
//!    (`policy_planes * size^2 -> policy_out`) weight, bias
//! 4. value head: 1x1 conv (`channels -> value_planes`) weight, bias; dense
//!    (`value_planes * size^2 -> value_hidden`) weight, bias; dense (`value_hidden -> 1`) weight,
//!    bias

mod d4;
mod net;

pub use d4::{cell_map, transform_planes, SYMMETRIES};
pub use net::{clear_cache, Forward, Net};

use std::io;
use std::path::Path;

const MAGIC: &[u8; 8] = b"GRIDCNN1";
const VERSION: u32 = 1;
const HEADER_FIELDS: usize = 8;
const HEADER_BYTES: usize = 8 + 4 + 4 * HEADER_FIELDS + 8;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Geometry {
    pub size: usize,
    pub in_planes: usize,
    pub channels: usize,
    pub blocks: usize,
    pub policy_planes: usize,
    pub policy_out: usize,
    pub value_planes: usize,
    pub value_hidden: usize,
}

impl Geometry {
    pub fn cells(&self) -> usize {
        self.size * self.size
    }

    pub fn n_weights(&self) -> usize {
        let (c, cells) = (self.channels, self.cells());
        let conv3 = |i: usize, o: usize| o * i * 9 + o;
        let conv1 = |i: usize, o: usize| o * i + o;
        let dense = |i: usize, o: usize| i * o + o;
        conv3(self.in_planes, c)
            + self.blocks * 2 * conv3(c, c)
            + conv1(c, self.policy_planes)
            + dense(self.policy_planes * cells, self.policy_out)
            + conv1(c, self.value_planes)
            + dense(self.value_planes * cells, self.value_hidden)
            + dense(self.value_hidden, 1)
    }

    fn header_fields(&self) -> [u32; HEADER_FIELDS] {
        [
            self.size,
            self.in_planes,
            self.channels,
            self.blocks,
            self.policy_planes,
            self.policy_out,
            self.value_planes,
            self.value_hidden,
        ]
        .map(|v| v as u32)
    }
}

#[derive(Clone, Debug)]
pub struct Weights {
    pub geometry: Geometry,
    pub data: Vec<f32>,
}

impl Weights {
    pub fn zeros(geometry: Geometry) -> Self {
        Weights {
            geometry,
            data: vec![0.0; geometry.n_weights()],
        }
    }

    pub fn to_bytes(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(HEADER_BYTES + 4 * self.data.len());
        out.extend_from_slice(MAGIC);
        out.extend_from_slice(&VERSION.to_le_bytes());
        for f in self.geometry.header_fields() {
            out.extend_from_slice(&f.to_le_bytes());
        }
        out.extend_from_slice(&(self.data.len() as u64).to_le_bytes());
        for w in &self.data {
            out.extend_from_slice(&w.to_le_bytes());
        }
        out
    }

    pub fn from_bytes(bytes: &[u8]) -> io::Result<Self> {
        let bad = |msg: String| io::Error::new(io::ErrorKind::InvalidData, msg);
        if bytes.len() < HEADER_BYTES || &bytes[..8] != MAGIC {
            return Err(bad("not a GRIDCNN1 weights file".into()));
        }
        let u32_at = |at: usize| u32::from_le_bytes(bytes[at..at + 4].try_into().unwrap());
        if u32_at(8) != VERSION {
            return Err(bad(format!("unsupported weights version {}", u32_at(8))));
        }
        let f = |i: usize| u32_at(12 + 4 * i) as usize;
        let geometry = Geometry {
            size: f(0),
            in_planes: f(1),
            channels: f(2),
            blocks: f(3),
            policy_planes: f(4),
            policy_out: f(5),
            value_planes: f(6),
            value_hidden: f(7),
        };
        let count_at = 12 + 4 * HEADER_FIELDS;
        let count = u64::from_le_bytes(bytes[count_at..count_at + 8].try_into().unwrap()) as usize;
        if count != geometry.n_weights() {
            return Err(bad(format!(
                "header says {count} weights, geometry needs {}",
                geometry.n_weights()
            )));
        }
        let body = &bytes[HEADER_BYTES..];
        if body.len() != 4 * count {
            return Err(bad(format!(
                "expected {} weight bytes, found {}",
                4 * count,
                body.len()
            )));
        }
        let data = body
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
            .collect();
        Ok(Weights { geometry, data })
    }

    pub fn load(path: impl AsRef<Path>) -> io::Result<Self> {
        Self::from_bytes(&std::fs::read(path)?)
    }

    pub fn save(&self, path: impl AsRef<Path>) -> io::Result<()> {
        std::fs::write(path, self.to_bytes())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn small() -> Geometry {
        Geometry {
            size: 5,
            in_planes: 3,
            channels: 4,
            blocks: 2,
            policy_planes: 2,
            policy_out: 27,
            value_planes: 1,
            value_hidden: 6,
        }
    }

    #[test]
    fn weights_file_round_trips() {
        let g = small();
        let data: Vec<f32> = (0..g.n_weights()).map(|i| (i as f32).sin()).collect();
        let w = Weights { geometry: g, data };
        let back = Weights::from_bytes(&w.to_bytes()).unwrap();
        assert_eq!(back.geometry, g);
        assert_eq!(back.data, w.data);

        let path =
            std::env::temp_dir().join(format!("grid-cnn-roundtrip-{}.bin", std::process::id()));
        w.save(&path).unwrap();
        let loaded = Weights::load(&path).unwrap();
        std::fs::remove_file(&path).unwrap();
        assert_eq!(loaded.data, w.data);
    }

    #[test]
    fn malformed_files_are_rejected() {
        let w = Weights::zeros(small());
        let mut bytes = w.to_bytes();
        assert!(Weights::from_bytes(&bytes[..bytes.len() - 4]).is_err());
        bytes[0] = b'X';
        assert!(Weights::from_bytes(&bytes).is_err());
        assert!(Weights::from_bytes(b"short").is_err());
    }

    #[test]
    fn n_weights_matches_a_hand_count() {
        // stem 4*3*9+4, 2 blocks x 2 convs x (4*4*9+4), policy conv 2*4+2, policy dense 50*27+27,
        // value conv 1*4+1, value dense 25*6+6, value out 6+1.
        let expect = (4 * 3 * 9 + 4)
            + 4 * (4 * 4 * 9 + 4)
            + (2 * 4 + 2)
            + (50 * 27 + 27)
            + (4 + 1)
            + (25 * 6 + 6)
            + 7;
        assert_eq!(small().n_weights(), expect);
    }
}
