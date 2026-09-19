//! The value model: a [`Geometry`] plus one flat `f32` weight vector, and its
//! on-disk form (`model.toml`, `weights.bin`, `weights.meta.json`).
//!
//! `weights.bin` is `n_weights` little-endian `f32`s. `weights.meta.json`
//! records the SHA-256 of `model.toml` and `n_weights`; loading refuses a
//! weights file whose geometry differs from the one it was trained against.
//! Extra keys in the meta file are allowed and ignored by readers.

use std::path::Path;

use serde::Deserialize;

use crate::geometry::Geometry;

#[derive(Clone, Debug)]
pub struct Model {
    geom: Geometry,
    w: Vec<f32>,
}

#[derive(Deserialize)]
struct Meta {
    model_toml_sha256: String,
    n_weights: usize,
}

impl Model {
    pub fn zeros(geom: Geometry) -> Model {
        let n = geom.n_weights();
        Model {
            geom,
            w: vec![0.0; n],
        }
    }

    pub fn from_weights(geom: Geometry, w: Vec<f32>) -> Model {
        assert_eq!(w.len(), geom.n_weights(), "weights.len() must equal n_weights");
        Model { geom, w }
    }

    pub fn geometry(&self) -> &Geometry {
        &self.geom
    }

    pub fn weights(&self) -> &[f32] {
        &self.w
    }

    pub fn weights_mut(&mut self) -> &mut [f32] {
        &mut self.w
    }

    /// Raw linear score: the sum of every selected weight, from the side to
    /// move's point of view.
    #[inline]
    pub fn logit(&self, codes: &[u8]) -> f32 {
        let mut acc = 0.0f32;
        for i in 0..self.geom.n_images() {
            acc += self.w[self.geom.image_index(i, codes)];
        }
        acc
    }

    /// `tanh` of the logit, in (-1, 1): the expected result for the side to
    /// move (+1 win, 0 draw, -1 loss).
    #[inline]
    pub fn value(&self, codes: &[u8]) -> f32 {
        self.logit(codes).tanh()
    }

    /// Write `model.toml`, `weights.bin` and `weights.meta.json` into `dir`
    /// (created if needed). `extra` is merged into the meta JSON.
    pub fn save(&self, dir: &Path, extra: serde_json::Value) {
        std::fs::create_dir_all(dir)
            .unwrap_or_else(|e| panic!("cannot create {}: {e}", dir.display()));
        write(&dir.join("model.toml"), self.geom.toml_text().as_bytes());
        let mut bytes = Vec::with_capacity(self.w.len() * 4);
        for v in &self.w {
            bytes.extend_from_slice(&v.to_le_bytes());
        }
        write(&dir.join("weights.bin"), &bytes);
        let mut meta = serde_json::json!({
            "model_toml_sha256": self.geom.sha256_hex(),
            "n_weights": self.geom.n_weights(),
        });
        if let (Some(m), serde_json::Value::Object(x)) = (meta.as_object_mut(), extra) {
            m.extend(x);
        }
        write(
            &dir.join("weights.meta.json"),
            serde_json::to_string_pretty(&meta).unwrap().as_bytes(),
        );
    }

    /// Load `model.toml` + `weights.bin` + `weights.meta.json` from `dir`,
    /// panicking with an actionable message on any mismatch. `orientations`
    /// is the adapter's [`CellFeatures::orientations`].
    ///
    /// [`CellFeatures::orientations`]: crate::geometry::CellFeatures::orientations
    pub fn load(dir: &Path, orientations: &[Vec<u8>]) -> Model {
        let toml_path = dir.join("model.toml");
        let toml_bytes = std::fs::read(&toml_path)
            .unwrap_or_else(|e| panic!("cannot read {}: {e}", toml_path.display()));
        let geom = Geometry::from_toml(&toml_bytes, orientations);

        let meta_path = dir.join("weights.meta.json");
        let meta_text = std::fs::read_to_string(&meta_path)
            .unwrap_or_else(|e| panic!("cannot read {}: {e}", meta_path.display()));
        let meta: Meta = serde_json::from_str(&meta_text).expect("weights.meta.json must parse");
        assert_eq!(
            meta.model_toml_sha256,
            geom.sha256_hex(),
            "weights.meta.json SHA-256 does not match {} - the weights were trained against a \
             different geometry",
            toml_path.display()
        );
        assert_eq!(meta.n_weights, geom.n_weights(), "meta n_weights disagrees with the geometry");

        let bin_path = dir.join("weights.bin");
        let raw = std::fs::read(&bin_path)
            .unwrap_or_else(|e| panic!("cannot read {}: {e}", bin_path.display()));
        assert_eq!(
            raw.len(),
            geom.n_weights() * 4,
            "{} holds {} bytes, expected {}",
            bin_path.display(),
            raw.len(),
            geom.n_weights() * 4
        );
        let w = raw
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        Model { geom, w }
    }
}

fn write(path: &Path, bytes: &[u8]) {
    std::fs::write(path, bytes).unwrap_or_else(|e| panic!("cannot write {}: {e}", path.display()));
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geometry::Tuple;

    fn perms() -> Vec<Vec<u8>> {
        vec![(0..4).collect(), vec![3, 2, 1, 0]]
    }

    fn model() -> Model {
        let g = Geometry::from_tuples(
            3,
            vec![
                Tuple {
                    name: "a".into(),
                    squares: vec![0, 1],
                },
                Tuple {
                    name: "b".into(),
                    squares: vec![2],
                },
            ],
            &perms(),
        );
        let w = (0..g.n_weights()).map(|i| (i as f32) * 0.25 - 1.0).collect();
        Model::from_weights(g, w)
    }

    fn scratch_dir(tag: &str) -> std::path::PathBuf {
        let dir = std::env::temp_dir().join(format!("ntuple-test-{tag}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        dir
    }

    #[test]
    fn logit_is_the_sum_of_selected_weights() {
        let m = model();
        // codes for cells 0..4 = [1, 2, 0, 1]. Tuple a: cells (0,1) -> 1 + 2*3 = 7
        // and mirrored cells (3,2) -> 1 + 0*3 = 1. Tuple b at offset 9: cell 2 -> 0
        // and mirrored cell 1 -> 2.
        let codes = [1u8, 2, 0, 1];
        let w = |i: usize| i as f32 * 0.25 - 1.0;
        let want = w(7) + w(1) + w(9) + w(9 + 2);
        assert!((m.logit(&codes) - want).abs() < 1e-6);
        assert!((m.value(&codes) - want.tanh()).abs() < 1e-6);
    }

    #[test]
    fn round_trips_through_disk_and_checks_the_sha() {
        let m = model();
        let dir = scratch_dir("roundtrip");
        m.save(&dir, serde_json::json!({"episodes": 12}));
        let back = Model::load(&dir, &perms());
        assert_eq!(back.weights(), m.weights());
        assert_eq!(back.geometry().sha256_hex(), m.geometry().sha256_hex());
        let meta: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(dir.join("weights.meta.json")).unwrap())
                .unwrap();
        assert_eq!(meta["episodes"], 12);
        assert_eq!(meta["model_toml_sha256"], m.geometry().sha256_hex());

        // Editing the geometry after the fact must be refused.
        let mut text = std::fs::read_to_string(dir.join("model.toml")).unwrap();
        text.push_str("\n# edited\n");
        std::fs::write(dir.join("model.toml"), text).unwrap();
        let err = std::panic::catch_unwind(|| Model::load(&dir, &perms()));
        assert!(err.is_err(), "a geometry that no longer matches the meta sha must not load");
        let _ = std::fs::remove_dir_all(&dir);
    }
}
